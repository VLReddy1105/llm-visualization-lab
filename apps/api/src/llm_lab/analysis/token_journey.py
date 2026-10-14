"""Selected-position metrics from immutable observations, without resident weights."""

import torch

from llm_lab.errors import ServiceError
from llm_lab.schemas.analysis import AnalysisSource, Checkpoint, EvolutionResult
from llm_lab.tracing.collector import tensor_statistics
from llm_lab.tracing.serialization import plain


def source_metadata(trace, position):
    m, a = trace.metadata, trace.metadata.architecture
    if not 0 <= position < m.sequence_length:
        raise ServiceError("INVALID_TOKEN", "Input token position is out of range.", 422)
    return AnalysisSource(
        trace_id=m.trace_id,
        token=plain(trace.tokens[position]),
        model_id=m.model_id,
        hf_model_id=m.hf_model_id,
        revision=m.revision,
        tokenizer_revision=m.tokenizer_revision,
        device=m.device,
        dtype=m.dtype,
        layer_count=a.layers,
        hidden_size=a.hidden_size,
        vocabulary_size=a.vocabulary_size,
        config_digest=a.config_digest,
        implementation_digest=a.implementation_digest,
    )


def checkpoints(trace):
    states = [(trace.embedding.output, "Embedding", "embedding", None)]
    states.extend(
        (
            layer.residual.after_mlp,
            f"Layer {layer.layer_index}",
            "post_block_residual",
            layer.layer_index,
        )
        for layer in trace.layers
    )
    states.append((trace.output.normalized, "Final norm", "final_normalized", None))
    if [layer.layer_index for layer in trace.layers] != list(
        range(trace.metadata.architecture.layers)
    ):
        raise ServiceError(
            "INVALID_TRACE", "Layer order does not match the captured architecture.", 409
        )
    return [
        Checkpoint(checkpoint_index=i, component_id=c, label=label, stage=s, layer_index=k)
        for i, (c, label, s, k) in enumerate(states)
    ]


def vector(trace, component, position):
    """Resolve absolute token coordinates; summary statistics are not vector substitutes."""
    source_metadata(trace, position)
    descriptor = trace.descriptors.get(component)
    if descriptor is None or descriptor.tensor_id is None:
        raise ServiceError(
            "COMPONENT_NOT_CAPTURED",
            "Complete depth vectors were not retained.",
            409,
            details={"component_id": component, "hint": "Explicitly capture overview."},
        )
    if descriptor.axes != ("batch", "token", "hidden"):
        raise ServiceError("INVALID_TRACE", "Unexpected residual axes.", 409)
    indices = (
        descriptor.coordinates[1]
        if descriptor.coordinates
        else range(trace.metadata.sequence_length)
    )
    if position not in indices:
        raise ServiceError(
            "COMPONENT_NOT_CAPTURED",
            "This token position was not retained.",
            409,
            details={"component_id": component, "token_index": position},
        )
    values = trace.tensor(component, copy=False)[0, list(indices).index(position)]
    if not bool(torch.isfinite(values).all()):
        raise ServiceError("NONFINITE_ANALYSIS", "Hidden state contains nonfinite values.", 409)
    return values


def cosine(a, b):
    a, b = a.double(), b.double()
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denominator) == 0:
        return None
    return float(torch.clamp(torch.dot(a, b) / denominator, -1, 1))


def evolution(trace, position, progress=lambda done, total: None):
    states = checkpoints(trace)
    vectors = [vector(trace, state.component_id, position).double() for state in states]
    norms = [float(torch.linalg.vector_norm(v)) for v in vectors]
    metrics = []
    for i, (state, values) in enumerate(zip(states, vectors, strict=True)):
        progress(i, len(states))
        previous = vectors[i - 1] if i else None
        cosines = {
            "cosine_previous": cosine(values, previous) if previous is not None else None,
            "cosine_embedding": cosine(values, vectors[0]),
            "cosine_final_residual": cosine(values, vectors[-2]),
        }
        stats = plain(tensor_statistics(values))
        stats.update(
            finite=True,
            undefined_fields=[
                name for name in ("min", "max", "mean", "std", "l2_norm") if stats[name] is None
            ],
            reduction_scope="selected token; all hidden coordinates; std correction=0",
        )
        metrics.append(
            {
                **state.model_dump(),
                "l2_norm": norms[i],
                "norm_change": norms[i] - norms[i - 1] if i else None,
                "consecutive_distance": float(torch.linalg.vector_norm(values - previous))
                if i
                else None,
                **cosines,
                "undefined": {
                    key: "no preceding checkpoint"
                    if key == "cosine_previous" and not i
                    else "zero vector norm"
                    for key, value in cosines.items()
                    if value is None
                },
                "statistics": stats,
            }
        )
    # Upper triangle supplies every pair without retaining another vector copy in the result.
    comparisons = []
    for i, a in enumerate(vectors):
        for j in range(i, len(vectors)):
            similarity = cosine(a, vectors[j])
            comparisons.append(
                {
                    "from_checkpoint": i,
                    "to_checkpoint": j,
                    "l2_distance": float(torch.linalg.vector_norm(vectors[j] - a)),
                    "norm_change": norms[j] - norms[i],
                    "cosine_similarity": similarity,
                    "undefined_reason": "zero vector norm" if similarity is None else None,
                }
            )
    progress(len(states), len(states))
    return EvolutionResult(
        source=source_metadata(trace, position),
        checkpoints=metrics,
        comparisons=comparisons,
        method=(
            "CPU float64 L2/differences/cosines on saved token vectors; final "
            "reference is the last raw post-block residual. Collector statistics use "
            "float32. Final norm is a distinct checkpoint, not a residual block."
        ),
    )
