"""CPU-only presentation of validated Stage 2 observations; no model execution."""

import base64
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import torch
from fastapi import Response
from pydantic import BaseModel

from llm_lab.config import Settings
from llm_lab.errors import ServiceError
from llm_lab.schemas.components import TensorDescriptor
from llm_lab.schemas.trace import TraceResult
from llm_lab.schemas.trace_api import IndexSelector, SliceRequest

_LAYER_SUFFIXES = {
    "residual.before",
    "attention.output",
    "residual.after_attention",
    "mlp.output",
    "residual.after_mlp",
    "output",
    "attention.input_normalized",
    "attention.query.projected",
    "attention.key.projected",
    "attention.value",
    "attention.query.normalized",
    "attention.key.normalized",
    "attention.query.rotary",
    "attention.key.rotary",
    "attention.similarity",
    "attention.scaled_scores",
    "attention.masked_scores",
    "attention.weights",
    "attention.weights_reconstructed",
    "attention.head_output",
    "attention.head_output_reconstructed",
    "attention.concatenated",
    "mlp.input_normalized",
    "mlp.gate",
    "mlp.up",
    "mlp.activated_gate",
    "mlp.gated",
}
_GLOBAL_IDS = {
    "embedding.output",
    "position.ids",
    "position.rotary_cos",
    "position.rotary_sin",
    "position.attention_mask",
    "output.residual_final",
    "output.normalized",
    "output.lm_head_input",
    "output.logits",
    "output.probabilities",
}


def plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return plain(value.model_dump())
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def coordinates(indices) -> dict:
    size = len(indices)
    step = indices[1] - indices[0] if size > 1 else 1
    if step > 0 and all(b - a == step for a, b in zip(indices, indices[1:], strict=False)):
        return {"count": size, "start": indices[0], "stop": indices[-1] + 1, "step": step}
    return {"count": size, "indices": list(indices)}


def page(items: list[dict], limit: int, cursor: str | None) -> dict:
    offset = 0
    if cursor:
        try:
            decoded = base64.urlsafe_b64decode(cursor.encode()).decode()
            if not re.fullmatch(r"offset:[0-9]+", decoded):
                raise ValueError("Malformed cursor")
            offset = int(decoded.split(":")[1])
            if offset > len(items):
                raise ValueError("Cursor is beyond collection")
        except (ValueError, UnicodeError) as exc:
            raise ServiceError("INVALID_CURSOR", "Invalid collection cursor.", 422) from exc
    end = min(offset + limit, len(items))
    next_cursor = (
        base64.urlsafe_b64encode(f"offset:{end}".encode()).decode() if end < len(items) else None
    )
    return {"items": items[offset:end], "total": len(items), "next_cursor": next_cursor}


class TraceSerializer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def response(self, body: Any, *, status: int = 200, headers: dict | None = None) -> Response:
        data = json.dumps(
            plain(body), allow_nan=False, ensure_ascii=False, separators=(",", ":")
        ).encode()
        if len(data) > self.settings.trace_api_max_response_bytes:
            self.too_large(bytes_count=len(data))
        return Response(data, status_code=status, media_type="application/json", headers=headers)

    def too_large(self, *, elements: int | None = None, bytes_count: int | None = None):
        raise ServiceError(
            "RESPONSE_TOO_LARGE",
            "Request a smaller slice or collection page.",
            413,
            details={
                "elements": elements,
                "bytes": bytes_count,
                "max_elements": self.settings.trace_api_max_elements,
                "max_response_bytes": self.settings.trace_api_max_response_bytes,
            },
        )

    @staticmethod
    def unavailable(trace: TraceResult, component_id: str):
        raise ServiceError(
            "COMPONENT_NOT_CAPTURED",
            "This component was not retained by the capture plan.",
            409,
            details={
                "trace_id": trace.metadata.trace_id,
                "requested_component": component_id,
                "available_components": [k for k, d in trace.descriptors.items() if d.tensor_id],
                "hint": "Create a new explicit trace with the required capture selection.",
            },
        )

    def component(self, trace: TraceResult, component_id: str) -> TensorDescriptor:
        descriptor = trace.descriptors.get(component_id)
        if descriptor:
            return descriptor
        match = re.fullmatch(r"layer\[([0-9]+)\]\.(.+)", component_id)
        known = component_id in _GLOBAL_IDS
        if match:
            layer = int(match[1])
            self.layer(trace, layer)
            known = match[2] in _LAYER_SUFFIXES
        if known:
            self.unavailable(trace, component_id)
        raise ServiceError(
            "COMPONENT_NOT_FOUND",
            "Unknown semantic component.",
            404,
            details={"component_id": component_id},
        )

    @staticmethod
    def descriptor(descriptor: TensorDescriptor) -> dict:
        stats = plain(descriptor.statistics)
        if stats:
            stats["finite"] = descriptor.statistics.nonfinite_count == 0
            stats["undefined_fields"] = [
                k for k in ("min", "max", "mean", "std", "l2_norm") if stats[k] is None
            ]
        coverage = descriptor.coordinates or tuple(range(s) for s in descriptor.logical_shape)
        return {
            "component_id": descriptor.component_id,
            "tensor_id": descriptor.tensor_id,
            "axes": list(descriptor.axes),
            "logical_shape": list(descriptor.logical_shape),
            "storage_shape": list(descriptor.storage_shape),
            "coverage": {
                axis: coordinates(c) for axis, c in zip(descriptor.axes, coverage, strict=True)
            },
            "dtype": descriptor.dtype,
            "storage_dtype": descriptor.storage_dtype,
            "original_device": descriptor.original_device,
            "storage_device": descriptor.storage_device,
            "numel": descriptor.numel,
            "byte_size": descriptor.byte_size,
            "provenance": descriptor.spec.value_kind.value,
            "sources": list(descriptor.spec.sources),
            "method": descriptor.spec.method,
            "availability": descriptor.availability,
            "raw_available": descriptor.tensor_id is not None,
            "unavailable_reason": descriptor.unavailable_reason,
            "statistics": stats,
        }

    @staticmethod
    def layer(trace: TraceResult, index: int):
        if not 0 <= index < trace.metadata.architecture.layers:
            raise ServiceError(
                "INVALID_LAYER",
                "Layer index exceeds the captured model bounds.",
                422,
                details={"layer_index": index, "layer_count": trace.metadata.architecture.layers},
            )
        return trace.layers[index]

    @staticmethod
    def validation(trace: TraceResult, index: int | None = None) -> dict:
        prefix = f"layer[{index}]." if index is not None else ""
        return {k: plain(v) for k, v in trace.metadata.validation.items() if k.startswith(prefix)}

    def layer_summary(self, trace: TraceResult, index: int) -> dict:
        layer = self.layer(trace, index)
        ds = [d for d in trace.descriptors.values() if d.spec.layer_index == index]
        stats = {
            name: plain(trace.descriptors[ref].statistics)
            for name, ref in plain(layer.residual).items()
        }
        checks = self.validation(trace, index)
        return {
            "layer_index": index,
            "attention_available": layer.attention is not None,
            "mlp_available": layer.mlp is not None,
            "components": [d.component_id for d in ds],
            "raw_components": [d.component_id for d in ds if d.tensor_id],
            "referenced_bytes": sum(d.byte_size for d in ds if d.tensor_id),
            "byte_scope": "component references; aliases may overlap other layers",
            "residual_statistics": stats,
            "validation": checks,
        }

    def logits(self, trace: TraceResult, top_k: int = 10, token_index: int | None = None) -> dict:
        position = trace.output.prediction_position
        if token_index is not None:
            if not 0 <= token_index < trace.metadata.sequence_length:
                raise ServiceError("INVALID_TOKEN", "Token index exceeds input bounds.", 422)
            if token_index != position:
                self.unavailable(trace, "output.logits at requested token")
        candidates = trace.output.candidates[:top_k]
        return {
            "trace_id": trace.metadata.trace_id,
            "prediction_position": position,
            "next_position": trace.output.next_position,
            "vocabulary_size": trace.metadata.architecture.vocabulary_size,
            "normalization": (
                "full vocabulary; Stage 1 CPU float64 softmax, stored float32 probabilities"
            ),
            "top_k": plain(candidates),
            "omitted_probability_mass": max(0.0, 1.0 - sum(c.probability for c in candidates)),
            "logits": self.descriptor(trace.descriptors[trace.output.logits]),
            "probabilities": self.descriptor(trace.descriptors[trace.output.probabilities]),
        }

    def overview(self, trace: TraceResult) -> dict:
        m = trace.metadata
        available = {
            "layers": [layer.layer_index for layer in trace.layers],
            "attention_layers": [layer.layer_index for layer in trace.layers if layer.attention],
            "mlp_layers": [layer.layer_index for layer in trace.layers if layer.mlp],
            "component_count": len(trace.descriptors),
            "raw_component_count": sum(d.tensor_id is not None for d in trace.descriptors.values()),
            "components_url": f"/api/v1/traces/{m.trace_id}/components",
        }
        counts = Counter(
            v.get(
                "status",
                "passed"
                if v.get("passed") is True
                else "failed"
                if v.get("passed") is False
                else "unknown",
            )
            for v in m.validation.values()
        )
        return {
            "model": {
                "model_id": m.model_id,
                "hf_model_id": m.hf_model_id,
                "revision": m.revision,
                "tokenizer_revision": m.tokenizer_revision,
                "device": m.device,
                "dtype": m.dtype,
                "architecture": plain(m.architecture),
            },
            "input": {
                "text": m.prompt,
                "mode": "raw",
                "add_special_tokens": False,
                "sequence_length": m.sequence_length,
                "tokens": page(plain(trace.tokens), 50, None),
            },
            "capture": plain(m.capture_plan),
            "summary": {
                "layer_count": len(trace.layers),
                "retained_bytes": m.actual_retained_bytes,
                "estimated_bytes": m.estimated_bytes,
                "transient_estimated_bytes": m.transient_estimated_bytes,
                "duration_ms": m.duration_ms,
                "forward_duration_ms": m.forward_duration_ms,
                "retention_policy": "ttl_or_capacity",
            },
            "prediction": {
                k: v
                for k, v in self.logits(trace).items()
                if k not in {"trace_id", "logits", "probabilities"}
            },
            "available": available,
            "provenance": dict(
                Counter(d.spec.value_kind.value for d in trace.descriptors.values())
            ),
            "validation": {
                "counts": dict(counts),
                "failed": [
                    k
                    for k, v in m.validation.items()
                    if v.get("status") == "failed" or v.get("passed") is False
                ],
                "scope": "existing Stage 2 checks; HTTP does not recompute scientific gates",
            },
            "warnings": list(m.warnings),
            "runtime": plain(m.runtime),
            "links": {
                name: f"/api/v1/traces/{m.trace_id}/{name}"
                for name in ("tokens", "layers", "components", "logits")
            },
        }

    def envelope(self, snapshot: dict, trace: TraceResult | None, display_top_k: int = 10) -> dict:
        overview = self.overview(trace) if trace else None
        if overview:
            overview["prediction"] = {
                k: v
                for k, v in self.logits(trace, display_top_k).items()
                if k not in {"trace_id", "logits", "probabilities"}
            }
        return {**snapshot, "schema_version": "1.0.0", "overview": overview}

    def group(self, trace: TraceResult, index: int, kind: str) -> dict:
        layer = self.layer(trace, index)
        structure = layer if kind == "layer" else getattr(layer, kind)
        if structure is None:
            self.unavailable(trace, f"layer[{index}].{kind}")
        ds = [
            d
            for d in trace.descriptors.values()
            if d.spec.layer_index == index
            and (
                kind == "layer"
                or kind == "residual"
                and d.component_id in plain(layer.residual).values()
                or kind in {"attention", "mlp"}
                and f"].{kind}." in d.component_id
            )
        ]
        data = plain(structure)
        if kind == "attention":
            arch = trace.metadata.architecture
            data.update(
                query_heads=arch.query_heads,
                kv_heads=arch.kv_heads,
                head_dimension=arch.head_dim,
                sequence_length=trace.metadata.sequence_length,
                captured_query_heads=list(trace.metadata.capture_plan.query_heads or ()),
            )
        return {
            "trace_id": trace.metadata.trace_id,
            "layer_index": index,
            "kind": kind,
            "structure": data,
            "components": [self.descriptor(d) for d in ds],
            "validation": self.validation(trace, index),
        }

    def select_tensor(self, trace: TraceResult, component_id: str, request: SliceRequest):
        d = self.component(trace, component_id)
        if not d.tensor_id:
            self.unavailable(trace, component_id)
        if request.component_id is not None and request.component_id != component_id:
            raise ServiceError(
                "INVALID_SLICE", "Component reference does not match the request.", 422
            )
        unknown = set(request.selectors) - set(d.axes)
        if unknown:
            raise ServiceError(
                "INVALID_SLICE",
                "Unknown semantic axis.",
                422,
                details={"unknown_axes": sorted(unknown), "axes": list(d.axes)},
            )
        retained = d.coordinates or tuple(range(s) for s in d.logical_shape)
        requested = []
        count = 1
        for axis, size, coords in zip(d.axes, d.logical_shape, retained, strict=True):
            selector = request.selectors.get(axis)
            if selector is None:
                chosen = coords
            elif isinstance(selector, IndexSelector):
                chosen = (selector.index,)
            else:
                if selector.start >= selector.stop or selector.stop > size:
                    raise ServiceError(
                        "INVALID_SLICE", "Range must be nonempty and inside the logical axis.", 422
                    )
                chosen = range(selector.start, selector.stop, selector.step)
            if any(i >= size for i in chosen):
                code = (
                    "INVALID_HEAD"
                    if "head" in axis
                    else "INVALID_TOKEN"
                    if "token" in axis
                    else "INVALID_SLICE"
                )
                raise ServiceError(code, "Index exceeds the logical axis bounds.", 422)
            count *= len(chosen)
            requested.append(chosen)
        if count > self.settings.trace_api_max_elements:
            self.too_large(elements=count)
        indices = []
        for coords, chosen in zip(retained, requested, strict=True):
            lookup = {v: i for i, v in enumerate(coords)}
            if any(v not in lookup for v in chosen):
                self.unavailable(trace, component_id + " at requested coordinates")
            indices.append([lookup[v] for v in chosen])
        tensor = trace.tensor(component_id, copy=False)
        # Select dimensions with the greatest reduction first to bound intermediates.
        for axis in sorted(range(len(indices)), key=lambda a: len(indices[a]) / tensor.shape[a]):
            if len(indices[axis]) != tensor.shape[axis] or indices[axis] != list(
                range(tensor.shape[axis])
            ):
                tensor = tensor.index_select(axis, torch.tensor(indices[axis], dtype=torch.long))
        return d, requested, tensor

    def slice_response(self, trace: TraceResult, component_id: str, request: SliceRequest):
        if request.format == "json":
            return self.response(self.slice(trace, component_id, request))
        d, selected, tensor = self.select_tensor(trace, component_id, request)
        # Explicit float32 transport widens fp16/bfloat16 exactly. Integer
        # observations are rejected, rather than silently rounding their IDs.
        if not tensor.is_floating_point():
            raise ServiceError(
                "INVALID_SLICE",
                "Binary float32 transport requires a floating tensor; use JSON for integer IDs.",
                422,
            )
        wire = tensor.detach().to(torch.float32).contiguous().numpy().astype("<f4", copy=False)
        data = wire.tobytes()
        if len(data) > self.settings.trace_api_max_response_bytes:
            self.too_large(bytes_count=len(data))
        metadata = {
            "trace_id": trace.metadata.trace_id,
            "tensor_id": d.tensor_id,
            "component_id": component_id,
            "provenance": d.spec.value_kind.value,
            "sources": list(d.spec.sources),
            "method": d.spec.method,
            "logical_shape": list(d.logical_shape),
            "shape": list(tensor.shape),
            "axes": list(d.axes),
            "selectors": plain(request.selectors),
            "coordinates": {axis: coordinates(c) for axis, c in zip(d.axes, selected, strict=True)},
            "dtype": d.dtype,
            "storage_dtype": d.storage_dtype,
            "wire_dtype": "little_endian_float32",
            "nonfinite_encoding": "IEEE-754; native mask is a separate component",
        }
        header = json.dumps(metadata, allow_nan=False, ensure_ascii=True, separators=(",", ":"))
        if len(header.encode()) > min(8192, self.settings.trace_api_max_response_bytes):
            self.too_large(bytes_count=len(header.encode()))
        return Response(
            data, media_type="application/octet-stream", headers={"X-Tensor-Metadata": header}
        )

    def slice(self, trace: TraceResult, component_id: str, request: SliceRequest) -> dict:
        d, requested, tensor = self.select_tensor(trace, component_id, request)
        values = tensor.reshape(-1).tolist()
        nonfinite = []
        for i, value in enumerate(values):
            if isinstance(value, float) and not math.isfinite(value):
                kind = (
                    "nan"
                    if math.isnan(value)
                    else "positive_infinity"
                    if value > 0
                    else "negative_infinity"
                )
                nonfinite.append({"flat_index": i, "kind": kind})
                values[i] = None
        masked = []
        if (
            component_id.endswith("attention.masked_scores")
            and "position.attention_mask" in trace._tensors
        ):
            mask_request = SliceRequest(
                selectors={k: v for k, v in request.selectors.items() if k != "query_head"}
            )
            mask = self.slice(trace, "position.attention_mask", mask_request)
            # The retained native mask has one broadcast head; report it separately
            # through the mask component, never infer a new causal mask here.
            mask_values = mask["values"]
            masked = [
                head * len(mask_values) + i
                for head in range(len(values) // len(mask_values))
                for i, v in enumerate(mask_values)
                if v is None or v < 0
            ]
        return {
            "trace_id": trace.metadata.trace_id,
            "tensor_id": d.tensor_id,
            "component_id": component_id,
            "provenance": d.spec.value_kind.value,
            "sources": list(d.spec.sources),
            "method": d.spec.method,
            "axes": list(d.axes),
            "logical_shape": list(d.logical_shape),
            "shape": list(tensor.shape),
            "coordinates": {
                axis: coordinates(c) for axis, c in zip(d.axes, requested, strict=True)
            },
            "selectors": plain(request.selectors),
            "dtype": d.dtype,
            "storage_dtype": d.storage_dtype,
            "wire_dtype": "json_integer" if not tensor.is_floating_point() else "json_number",
            "values": values,
            "nonfinite": nonfinite,
            "masked_flat_indices": masked,
        }

    def head(
        self,
        trace: TraceResult,
        index: int,
        head: int,
        query_token: int | None,
        include_weights: bool,
        include_query: bool,
    ) -> dict:
        layer = self.layer(trace, index)
        if not 0 <= head < trace.metadata.architecture.query_heads:
            raise ServiceError("INVALID_HEAD", "Query head exceeds model bounds.", 422)
        if query_token is not None and not 0 <= query_token < trace.metadata.sequence_length:
            raise ServiceError("INVALID_TOKEN", "Query token exceeds input bounds.", 422)
        if layer.attention is None:
            self.unavailable(trace, f"layer[{index}].attention")
        heads = trace.metadata.capture_plan.query_heads
        if head not in heads:
            self.unavailable(trace, f"layer[{index}].attention query head {head}")
        kv_head = layer.attention.query_to_kv_head[heads.index(head)]
        selectors = {"batch": {"index": 0}, "query_head": {"index": head}}
        if query_token is not None:
            selectors["query_token"] = {"index": query_token}
        weights = (
            self.slice(trace, layer.attention.weights, SliceRequest(selectors=selectors))
            if include_weights
            else None
        )
        query = (
            self.slice(trace, layer.attention.query_rotary, SliceRequest(selectors=selectors))
            if include_query
            else None
        )
        elements = sum(len(v["values"]) for v in (weights, query) if v)
        if elements > self.settings.trace_api_max_elements:
            self.too_large(elements=elements)
        refs = [
            layer.attention.weights,
            layer.attention.query_rotary,
            layer.attention.key_rotary,
            layer.attention.value,
        ]
        return {
            "trace_id": trace.metadata.trace_id,
            "layer_index": index,
            "head_index": head,
            "kv_head_index": kv_head,
            "components": [self.descriptor(trace.descriptors[r]) for r in refs if r],
            "weights": weights,
            "query": query,
            "validation": self.validation(trace, index),
        }
