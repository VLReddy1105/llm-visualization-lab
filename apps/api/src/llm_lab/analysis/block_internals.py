"""Bounded Qwen3 block diagnostics from one immutable capture, without model execution."""

import torch
from torch.nn import functional as F

from llm_lab.analysis.token_journey import cosine, source_metadata
from llm_lab.config import HF_MODEL_ID, MODEL_REVISION
from llm_lab.errors import ServiceError
from llm_lab.schemas.analysis import MLPInternalsResult, ResidualStreamResult
from llm_lab.tracing.validation import compare_tensors

# Symbols are presentation aliases only; semantic component IDs remain authoritative.
COMPONENTS = {
    "r0": "residual.before",
    "n_attn": "attention.input_normalized",
    "a": "attention.output",
    "r1": "residual.after_attention",
    "n_mlp": "mlp.input_normalized",
    "g": "mlp.gate",
    "u": "mlp.up",
    "s": "mlp.activated_gate",
    "p": "mlp.gated",
    "m": "mlp.output",
    "r2": "residual.after_mlp",
}
INTERMEDIATE = frozenset({"g", "u", "s", "p"})
RANKING_POLICY = (
    "absolute: descending |value|; positive: positive values descending; "
    "negative: negative values ascending; exact ties use ascending original coordinate. "
    "At most top_k rows; sign-filtered tables can be empty."
)
METHOD = (
    "One saved original-forward source; actual captured module-boundary vectors. "
    "CPU float64 summaries, norms, differences and cosines; native float32 saved-operand "
    "reconciliation at atol=1e-5, rtol=1e-4. Signed delta is after-minus-before, a derived "
    "difference, while attention and MLP updates are directly captured. Relative L2 "
    "audit uses the existing declared 1e-12 reference-norm floor. No model calls. "
    "Magnitude, geometry and coordinate ranks are descriptive, not causal importance "
    "or evidence of named concepts."
)
OPERATIONS = {
    "r0": ("Residual input", "r0 = incoming residual", ()),
    "n_attn": ("Attention RMSNorm", "n_attn = input_layernorm(r0)", ("r0",)),
    "a": (
        "Attention update",
        "a = self_attention(n_attn), including output projection",
        ("n_attn",),
    ),
    "r1": ("After attention", "r1 = r0 + a", ("r0", "a")),
    "n_mlp": ("MLP RMSNorm", "n_mlp = post_attention_layernorm(r1)", ("r1",)),
    "g": ("Gate projection", "g = W_gate n_mlp", ("n_mlp",)),
    "u": ("Up projection", "u = W_up n_mlp", ("n_mlp",)),
    "s": ("SiLU activation", "s = SiLU(g) = g * sigmoid(g)", ("g",)),
    "p": ("Elementwise product", "p[i] = s[i] * u[i]", ("s", "u")),
    "m": ("MLP update / down projection", "m = W_down p", ("p",)),
    "r2": ("Residual output", "r2 = r1 + m", ("r1", "m")),
}


def _component(layer, symbol):
    return f"layer[{layer}].{COMPONENTS[symbol]}"


def _same_device(observed, execution):
    # PyTorch reports actual MPS tensor placement as mps:0 while the manager
    # records the requested effective device as mps. Preserve both descriptions.
    left, right = torch.device(observed), torch.device(execution)
    return left.type == right.type and (left.index or 0) == (right.index or 0)


def _missing(component, parameters):
    return ServiceError(
        "COMPONENT_NOT_CAPTURED",
        "The selected layer/token requires retained residual and MLP detail.",
        409,
        details={
            "component_id": component,
            "layer_index": parameters.layer_index,
            "token_index": parameters.token_index,
            "hint": "Explicitly capture profile mlp for this layer and absolute token position.",
            "capture": {
                "profile": "mlp",
                "layers": [parameters.layer_index],
                "tokens": [parameters.token_index],
            },
        },
    )


def require_block_capture(trace, parameters, kind=None):
    """Validate provenance and absolute axes before any ranking or derived scalar work."""
    m, arch = trace.metadata, trace.metadata.architecture
    if not 0 <= parameters.token_index < m.sequence_length:
        raise ServiceError("INVALID_TOKEN", "Input token position is out of range.", 422)
    if not 0 <= parameters.layer_index < arch.layers:
        raise ServiceError(
            "INVALID_LAYER", "Layer index is outside the captured architecture.", 422
        )
    if (
        m.hf_model_id != HF_MODEL_ID
        or m.revision != MODEL_REVISION
        or m.tokenizer_revision != m.revision
        or m.dtype != "float32"
        or arch.adapter_id != "qwen3"
        or arch.attention_backend != "eager"
        or m.device not in {"cpu", "mps"}
    ):
        raise ServiceError(
            "ANALYSIS_UNSUPPORTED",
            "Block analysis requires the pinned Qwen3 eager float32 capture configuration.",
            409,
        )
    if (
        len(trace.tokens) != m.sequence_length
        or len(m.token_ids) != m.sequence_length
        or trace.tokens[parameters.token_index].token_index != parameters.token_index
        or trace.tokens[parameters.token_index].token_id != m.token_ids[parameters.token_index]
    ):
        raise ServiceError("SOURCE_IDENTITY_MISMATCH", "Source token coordinates disagree.", 409)
    for field, width in (
        ("hidden_coordinate_index", arch.hidden_size),
        ("intermediate_coordinate_index", arch.intermediate_size),
    ):
        if not 0 <= getattr(parameters, field) < width:
            raise ServiceError(
                "INVALID_COORDINATE",
                "Coordinate is outside its model vector width.",
                422,
                details={"field": field, "upper_bound_exclusive": width},
            )
    vectors = {}
    for symbol in COMPONENTS:
        component = _component(parameters.layer_index, symbol)
        descriptor = trace.descriptors.get(component)
        if descriptor is None or descriptor.tensor_id is None:
            raise _missing(component, parameters)
        width = arch.intermediate_size if symbol in INTERMEDIATE else arch.hidden_size
        axes = ("batch", "token", "intermediate" if symbol in INTERMEDIATE else "hidden")
        if (
            descriptor.component_id != component
            or descriptor.spec.layer_index != parameters.layer_index
            or descriptor.axes != axes
            or descriptor.logical_shape != (1, m.sequence_length, width)
            or descriptor.dtype != m.dtype
            or not _same_device(descriptor.original_device, m.device)
            or descriptor.spec.value_kind != "raw"
        ):
            raise ServiceError(
                "SOURCE_IDENTITY_MISMATCH",
                "Block component provenance or axes disagree.",
                409,
                details={"component_id": component},
            )
        coverage = descriptor.coordinates or (
            tuple(range(1)),
            tuple(range(m.sequence_length)),
            tuple(range(width)),
        )
        if coverage[0] != (0,) or coverage[2] != tuple(range(width)):
            raise _missing(component, parameters)
        if parameters.token_index not in coverage[1]:
            raise _missing(component, parameters)
        try:
            tensor = trace.tensor(component, copy=False)
        except KeyError as exc:
            raise _missing(component, parameters) from exc
        if (
            tuple(tensor.shape) != tuple(len(c) for c in coverage)
            or tuple(tensor.shape) != descriptor.storage_shape
            or tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
        ):
            raise ServiceError(
                "SOURCE_IDENTITY_MISMATCH", "Stored block tensor metadata disagrees.", 409
            )
        values = tensor[0, coverage[1].index(parameters.token_index)]
        if not bool(torch.isfinite(values).all()):
            raise ServiceError(
                "NONFINITE_ANALYSIS",
                "Captured block vector contains nonfinite values.",
                409,
                details={"component_id": component},
            )
        vectors[symbol] = values
    return vectors


def rank_coordinates(values, ranking, top_k):
    """Return original indices, with deterministic stable tie resolution and sign filtering."""
    values = values.double()
    if values.ndim != 1 or not bool(torch.isfinite(values).all()):
        raise ServiceError("NONFINITE_ANALYSIS", "Coordinate ranking needs a finite vector.", 409)
    indices = torch.arange(values.numel())
    if ranking == "positive":
        indices = indices[values > 0]
        order = torch.argsort(values[indices], descending=True, stable=True)
    elif ranking == "negative":
        indices = indices[values < 0]
        order = torch.argsort(values[indices], stable=True)
    elif ranking == "absolute":
        order = torch.argsort(values.abs(), descending=True, stable=True)
    else:
        raise ServiceError("INVALID_COORDINATE", "Unknown coordinate ranking.", 422)
    return indices[order[:top_k]].tolist()


def _operation(trace, parameters, vectors, symbol):
    component = _component(parameters.layer_index, symbol)
    descriptor = trace.descriptors[component]
    values = vectors[symbol].double()
    name, formula, inputs = OPERATIONS[symbol]
    selected = (
        parameters.intermediate_coordinate_index
        if symbol in INTERMEDIATE
        else parameters.hidden_coordinate_index
    )
    return {
        "symbol": symbol,
        "component_id": component,
        "name": name,
        "formula": formula,
        "input_components": [_component(parameters.layer_index, item) for item in inputs],
        "input_shapes": [
            list(trace.descriptors[_component(parameters.layer_index, item)].logical_shape)
            for item in inputs
        ],
        "output_shape": list(descriptor.logical_shape),
        "selected_shape": list(values.shape),
        "dtype": descriptor.dtype,
        "original_device": descriptor.original_device,
        "provenance": descriptor.spec.value_kind,
        "method": descriptor.spec.method,
        "statistics": {
            "count": values.numel(),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std(correction=0)),
            "l2_norm": float(torch.linalg.vector_norm(values)),
        },
        "coordinates": [
            {"index": i, "value": float(values[i])}
            for i in rank_coordinates(values, parameters.ranking, parameters.top_k)
        ],
        "selected_coordinate": {"index": selected, "value": float(values[selected])},
    }


def _check(name, formula, actual, reference, sources):
    metrics = compare_tensors(actual, reference)
    if not metrics["passed"]:
        raise ServiceError(
            "RECONCILIATION_FAILED",
            "Saved block activations failed numerical reconciliation.",
            409,
            details={"name": name, "formula": formula, "metrics": metrics},
        )
    return {
        "name": name,
        "formula": formula,
        "sources": sources,
        "scope": "selected token; all vector coordinates",
        "metrics": metrics,
    }


def _reconcile(trace, parameters, vectors):
    r0, a, r1, m, r2 = (vectors[s] for s in ("r0", "a", "r1", "m", "r2"))
    checks = []
    for name, formula, actual, reference, symbols in (
        ("attention_addition", "r1 ≈ r0 + a", r1, r0 + a, ("r0", "a", "r1")),
        ("mlp_addition", "r2 ≈ r1 + m", r2, r1 + m, ("r1", "m", "r2")),
        ("complete_block", "r2 ≈ (r0 + a) + m", r2, (r0 + a) + m, ("r0", "a", "m", "r2")),
        ("silu", "s ≈ SiLU(g)", vectors["s"], F.silu(vectors["g"]), ("g", "s")),
        ("product", "p ≈ s * u", vectors["p"], vectors["s"] * vectors["u"], ("s", "u", "p")),
    ):
        checks.append(
            _check(
                name,
                formula,
                actual,
                reference,
                [_component(parameters.layer_index, s) for s in symbols],
            )
        )
    # Original-capture references were computed with the actual installed learned modules
    # after hooks were removed. Reading them never reruns normalization or a projection.
    for suffix, formula, symbols in (
        ("attention.rmsnorm", "n_attn ≈ input_layernorm(r0)", ("r0", "n_attn")),
        ("mlp.rmsnorm", "n_mlp ≈ post_attention_layernorm(r1)", ("r1", "n_mlp")),
        ("mlp.gate_projection", "g ≈ W_gate n_mlp", ("n_mlp", "g")),
        ("mlp.up_projection", "u ≈ W_up n_mlp", ("n_mlp", "u")),
        ("mlp.down_projection", "m ≈ W_down p", ("p", "m")),
    ):
        name = f"layer[{parameters.layer_index}].{suffix}"
        metrics = trace.metadata.validation.get(name)
        if metrics is None or "passed" not in metrics:
            raise ServiceError(
                "COMPONENT_NOT_CAPTURED",
                "Capture lacks a required original module-reference check; "
                "explicitly recapture detail.",
                409,
                details={"validation_check": name},
            )
        if not metrics["passed"]:
            raise ServiceError(
                "RECONCILIATION_FAILED",
                "Original capture failed a learned-module reference check.",
                409,
                details={"name": name, "metrics": dict(metrics)},
            )
        checks.append(
            {
                "name": name,
                "formula": formula,
                "provenance": "independent_reference",
                "sources": [_component(parameters.layer_index, s) for s in symbols],
                "scope": "all retained tokens at this layer during original capture validation",
                "metrics": dict(metrics),
            }
        )
    return checks


def _comparison(parameters, vectors, name, before_symbol, update_symbol, after_symbol):
    before, after = vectors[before_symbol].double(), vectors[after_symbol].double()
    update = (
        (vectors["a"].double() + vectors["m"].double())
        if update_symbol == "a+m"
        else vectors[update_symbol].double()
    )
    before_norm, update_norm, after_norm = [
        float(torch.linalg.vector_norm(v)) for v in (before, update, after)
    ]
    similarity = cosine(before, after)
    ratio = update_norm / before_norm if before_norm else None
    undefined = {}
    if similarity is None:
        undefined["cosine_before_after"] = "zero vector norm"
    if ratio is None:
        undefined["update_to_residual_ratio"] = "zero incoming residual norm"

    def row(i):
        return {
            "index": i,
            "before": float(before[i]),
            "update": float(update[i]),
            "after": float(after[i]),
            "signed_delta": float(after[i] - before[i]),
            "addition_error": float(after[i] - (before[i] + update[i])),
        }

    return {
        "name": name,
        "before_symbol": before_symbol,
        "update_symbol": update_symbol,
        "after_symbol": after_symbol,
        "update_provenance": "derived" if update_symbol == "a+m" else "raw",
        "before_norm": before_norm,
        "update_norm": update_norm,
        "after_norm": after_norm,
        "delta_norm": float(torch.linalg.vector_norm(after - before)),
        "cosine_before_after": similarity,
        "update_to_residual_ratio": ratio,
        "undefined": undefined,
        "coordinates": [
            row(i) for i in rank_coordinates(update, parameters.ranking, parameters.top_k)
        ],
        "selected_coordinate": row(parameters.hidden_coordinate_index),
    }


def residual_stream(trace, parameters, progress=lambda done, total: None):
    progress(0, 1)
    vectors = require_block_capture(trace, parameters)
    checks = _reconcile(trace, parameters, vectors)
    result = ResidualStreamResult(
        source=source_metadata(trace, parameters.token_index),
        layer_index=parameters.layer_index,
        operations=[
            _operation(trace, parameters, vectors, s)
            for s in ("r0", "n_attn", "a", "r1", "n_mlp", "m", "r2")
        ],
        comparisons=[
            _comparison(parameters, vectors, "attention", "r0", "a", "r1"),
            _comparison(parameters, vectors, "mlp", "r1", "m", "r2"),
            _comparison(parameters, vectors, "block", "r0", "a+m", "r2"),
        ],
        reconciliation=checks,
        method=METHOD,
        ranking_policy=RANKING_POLICY,
    )
    progress(1, 1)
    return result


def mlp_internals(trace, parameters, progress=lambda done, total: None):
    progress(0, 1)
    vectors = require_block_capture(trace, parameters)
    checks = _reconcile(trace, parameters, vectors)

    def row(i):
        gate, activated, up, product = [float(vectors[s][i]) for s in ("g", "s", "u", "p")]
        return {
            "index": i,
            "gate": gate,
            "activated_gate": activated,
            "up": up,
            "product": product,
            "multiplication_error": product - activated * up,
        }

    result = MLPInternalsResult(
        source=source_metadata(trace, parameters.token_index),
        layer_index=parameters.layer_index,
        intermediate_size=trace.metadata.architecture.intermediate_size,
        operations=[
            _operation(trace, parameters, vectors, s)
            for s in ("r1", "n_mlp", "g", "u", "s", "p", "m", "r2")
        ],
        selected_gate_coordinate=row(parameters.intermediate_coordinate_index),
        gate_coordinates=[
            row(i) for i in rank_coordinates(vectors["p"], parameters.ranking, parameters.top_k)
        ],
        reconciliation=checks,
        method=METHOD,
        ranking_policy=RANKING_POLICY,
    )
    progress(1, 1)
    return result
