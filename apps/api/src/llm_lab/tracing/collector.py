"""Bounded owned CPU snapshots, semantic coverage and finite population statistics."""

import math

import torch

from llm_lab.schemas.components import TensorDescriptor, TensorStatistics


def storage_shape(spec):
    return (
        tuple(len(indices) for indices in spec.coordinates)
        if spec.coordinates is not None
        else spec.logical_shape
    )


def should_retain(spec, plan):
    if spec.component_id in {"output.logits", "output.probabilities"}:
        return True
    return plan.effective_raw and (
        spec.layer_index is None
        or plan.profile == "overview"
        or spec.layer_index in plan.detailed_layers
    )


def tensor_statistics(tensor, *, exclude=None):
    finite = torch.isfinite(tensor)
    nonfinite_count = tensor.numel() - int(finite.sum())
    masked_count = 0 if exclude is None else int(exclude.sum())
    eligible = finite if exclude is None else finite & ~exclude
    values = tensor[eligible].float()
    if values.numel():
        # Float32 is the declared reduction dtype, including population std.
        low, high, mean, std, norm = (
            float(values.min()),
            float(values.max()),
            float(values.mean()),
            float(values.std(correction=0)),
            float(torch.linalg.vector_norm(values)),
        )
        # Never encode an overflow as an ordinary scientific scalar.
        low, high, mean, std, norm = (
            number if math.isfinite(number) else None for number in (low, high, mean, std, norm)
        )
    else:
        low = high = mean = std = norm = None
    return TensorStatistics(
        finite_count=values.numel(),
        nonfinite_count=nonfinite_count,
        min=low,
        max=high,
        mean=mean,
        std=std,
        l2_norm=norm,
        masked_count=masked_count,
        total_count=tensor.numel(),
        reduction_scope=(
            "retained coordinates; finite unmasked entries only; std correction=0"
            if exclude is not None
            else "retained coordinates; finite entries only; std correction=0"
        ),
    )


class TraceCollector:
    def __init__(self, plan, budget_bytes, *, on_layer=None):
        self.plan = plan
        self.budget_bytes = budget_bytes
        self.on_layer = on_layer
        self.descriptors = {}
        self.tensors = {}
        self.workspace = {}
        self._storage = {}
        self._counter = 0
        self.warnings = []

    @property
    def actual_retained_bytes(self):
        return sum(self._storage.values())

    def observe(self, spec, tensor, *, selected=False, original_device=None):
        expected = storage_shape(spec) if selected else spec.logical_shape
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{spec.component_id}: observed {tuple(tensor.shape)}, expected {expected}"
            )
        original_device = original_device or str(tensor.device)
        original_dtype = str(tensor.dtype).removeprefix("torch.")
        view = tensor.detach()
        if spec.coordinates is not None and not selected:
            for axis, indices in enumerate(spec.coordinates):
                if indices != tuple(range(spec.logical_shape[axis])):
                    view = view.index_select(axis, torch.tensor(indices, device=view.device))
        # copy=True is essential even when execution itself is on CPU.
        snapshot = view.to(device="cpu", copy=True, memory_format=torch.contiguous_format)
        component = spec.component_id
        tensor_id = None
        retained = should_retain(spec, self.plan)
        if retained:
            for source in spec.sources:
                previous = self.tensors.get(source)
                descriptor = self.descriptors.get(source)
                candidate = previous
                coverage_matches = (
                    descriptor is not None and descriptor.spec.coordinates == spec.coordinates
                )
                # Full head concatenation is a verified view of the same owned
                # CPU storage. Partial head selections retain independent data.
                if (
                    previous is not None
                    and descriptor is not None
                    and descriptor.axes == ("batch", "token", "head_feature")
                    and spec.axes == ("batch", "query_head", "query_token", "head_dimension")
                    and snapshot.shape[1] == spec.logical_shape[1]
                    and snapshot.shape[3] == spec.logical_shape[3]
                    and previous.shape[2] == snapshot.shape[1] * snapshot.shape[3]
                ):
                    source_tokens = (
                        descriptor.coordinates[1]
                        if descriptor.coordinates
                        else tuple(range(descriptor.logical_shape[1]))
                    )
                    query_tokens = (
                        spec.coordinates[2]
                        if spec.coordinates
                        else tuple(range(spec.logical_shape[2]))
                    )
                    coverage_matches = source_tokens == query_tokens
                    candidate = previous.reshape(
                        previous.shape[0], previous.shape[1], snapshot.shape[1], snapshot.shape[3]
                    ).transpose(1, 2)
                if (
                    candidate is not None
                    and descriptor is not None
                    and coverage_matches
                    and candidate.dtype == snapshot.dtype
                    and tuple(candidate.shape) == tuple(snapshot.shape)
                    and torch.equal(candidate, snapshot)
                    and "alias" in spec.method
                ):
                    snapshot = candidate
                    tensor_id = descriptor.tensor_id
                    break
            if tensor_id is None:
                tensor_id = f"tensor-{self._counter}"
                self._counter += 1
                size = snapshot.numel() * snapshot.element_size()
                if self.actual_retained_bytes + size > self.budget_bytes:
                    raise ValueError("Capture retained bytes exceed the preflight budget")
                self._storage[tensor_id] = size
            self.tensors[component] = snapshot
        self.workspace[component] = snapshot
        excluded = None
        if component == "position.attention_mask":
            excluded = snapshot != 0
        elif component.endswith("attention.masked_scores"):
            mask = self.workspace.get("position.attention_mask")
            if mask is not None:
                if spec.coordinates is not None:
                    for axis in (2, 3):
                        indices = spec.coordinates[axis]
                        if (
                            mask.shape[axis] == spec.logical_shape[axis]
                            and len(indices) != mask.shape[axis]
                        ):
                            mask = mask.index_select(axis, torch.tensor(indices))
                excluded = (mask != 0).expand_as(snapshot)
        statistics = tensor_statistics(snapshot, exclude=excluded) if self.plan.statistics else None
        if statistics is not None and statistics.nonfinite_count:
            self.warnings.append(f"{component}: {statistics.nonfinite_count} non-finite entries")
        self.descriptors[component] = TensorDescriptor(
            spec=spec,
            storage_shape=tuple(snapshot.shape),
            dtype=original_dtype,
            storage_dtype=str(snapshot.dtype).removeprefix("torch."),
            original_device=original_device,
            numel=snapshot.numel(),
            byte_size=snapshot.numel() * snapshot.element_size(),
            tensor_id=tensor_id,
            statistics=statistics,
            availability="captured" if retained else "summary_only",
            unavailable_reason=None if retained else "Raw storage omitted by CapturePlan",
        )
        if spec.layer_index is not None and component == f"layer[{spec.layer_index}].output":
            if self.on_layer is not None:
                self.on_layer(spec.layer_index, self.workspace)
            if spec.layer_index not in self.plan.detailed_layers:
                prefix = f"layer[{spec.layer_index}]."
                for key in tuple(self.workspace):
                    if key.startswith(prefix):
                        del self.workspace[key]

    def missing(self, specs, dtype, device):
        for spec in specs:
            if spec.component_id not in self.descriptors:
                shape = storage_shape(spec)
                numel = math.prod(shape)
                self.descriptors[spec.component_id] = TensorDescriptor(
                    spec=spec,
                    storage_shape=shape,
                    dtype=dtype,
                    storage_dtype=dtype,
                    original_device=device,
                    numel=numel,
                    byte_size=0,
                    availability="unavailable",
                    unavailable_reason="Boundary was not observed",
                )

    def clear(self):
        self.workspace.clear()
        self.tensors.clear()
        self._storage.clear()
        self.descriptors.clear()
        self.on_layer = None
