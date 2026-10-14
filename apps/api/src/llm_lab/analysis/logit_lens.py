"""Streaming readout projection; called only on the resident model executor."""

import torch

from llm_lab.analysis.distributions import PROBABILITY_POLICY, RANK_POLICY, candidate, distribution
from llm_lab.analysis.token_journey import checkpoints, source_metadata, vector
from llm_lab.config import HF_MODEL_ID
from llm_lab.errors import ServiceError
from llm_lab.schemas.analysis import LensResult
from llm_lab.tracing.validation import compare_tensors


def require_matching_adapter(trace, adapter):
    m = trace.metadata
    if (
        m.hf_model_id != HF_MODEL_ID
        or m.revision != adapter.revision
        or m.tokenizer_revision != adapter.revision
        or m.architecture != adapter.architecture
        or m.device != adapter.device
        or m.dtype != str(adapter.dtype).removeprefix("torch.")
    ):
        raise ServiceError(
            "MODEL_TRACE_MISMATCH",
            (
                "Resident readout does not match this trace's model, tokenizer, "
                "architecture, device and precision."
            ),
            409,
        )
    if m.dtype != "float32":
        raise ServiceError(
            "ANALYSIS_UNSUPPORTED", "Stage 6 readout is validated for float32 only.", 409
        )
    adapter.validate_for_trace()


def logit_lens(trace, parameters, adapter, progress=lambda done, total: None):
    require_matching_adapter(trace, adapter)
    position = parameters.token_index
    source = source_metadata(trace, position)
    states = checkpoints(trace)
    # Validate complete availability before the first projection.
    vectors = [vector(trace, s.component_id, position) for s in states]
    if any(i >= source.vocabulary_size for i in parameters.candidate_ids):
        raise ServiceError("INVALID_CANDIDATE", "Candidate IDs exceed the vocabulary.", 422)

    def project(values, normalized=False):
        values = values.reshape(1, 1, -1)
        if not normalized:
            values = adapter.native_norm("final", values)
        logits = adapter.native_lm_head(values).reshape(-1).float()
        if not bool(torch.isfinite(logits).all()):
            raise ServiceError("NONFINITE_ANALYSIS", "Readout returned nonfinite logits.", 409)
        return logits

    progress(0, len(states))
    final_logits = project(vectors[-2])
    _, order, _ = distribution(final_logits)
    candidate_ids = list(parameters.candidate_ids)
    for token_id in order[:3].tolist():
        if token_id not in candidate_ids and len(candidate_ids) < 8:
            candidate_ids.append(token_id)
    records = []
    for i, (state, values) in enumerate(zip(states, vectors, strict=True)):
        progress(i, len(states))
        logits = (
            final_logits
            if i == len(states) - 2
            else project(values, state.stage == "final_normalized")
        )
        probabilities, order, ranks = distribution(logits)
        top_ids = order[: parameters.display_top_k].tolist()
        nonzero = probabilities.double()[probabilities > 0]
        records.append(
            {
                **state.model_dump(),
                "normalization_applied": state.stage != "final_normalized",
                "top_candidates": [
                    candidate(adapter.tokenizer, t, logits, probabilities, ranks) for t in top_ids
                ],
                "comparison_candidates": [
                    candidate(adapter.tokenizer, t, logits, probabilities, ranks)
                    for t in candidate_ids
                ],
                "probability_sum": float(probabilities.double().sum()),
                "entropy_nats": float(-(nonzero * nonzero.log()).sum()),
                "omitted_probability_mass": max(
                    0.0, 1.0 - float(probabilities[top_ids].double().sum())
                ),
            }
        )
    final_normalized = adapter.native_norm("final", vectors[-2].reshape(1, 1, -1)).reshape(-1)
    norm_check = compare_tensors(final_normalized, vectors[-1])
    logits_check = probability_check = None
    if position == trace.output.prediction_position:
        logits_check = compare_tensors(
            final_logits, trace.tensor(trace.output.logits, copy=False).reshape(-1)
        )
        probability_check = compare_tensors(
            distribution(final_logits)[0],
            trace.tensor(trace.output.probabilities, copy=False).reshape(-1),
            atol=1e-6,
        )
    progress(len(states), len(states))
    return LensResult(
        source=source,
        prediction_position=position,
        next_position=position + 1,
        candidate_ids=candidate_ids,
        checkpoints=records,
        final_logits_comparison=logits_check,
        final_probabilities_comparison=probability_check,
        final_normalization_comparison=norm_check,
        comparison_note="Compared against original retained final-position output."
        if logits_check
        else (
            "Original logits were retained only at the final prompt position; this "
            "position has no original-logit comparison."
        ),
        method=(
            "Each saved embedding/post-block residual → independently applied "
            "resident final RMSNorm → resident LM head over full vocabulary. "
            "Captured final-normalized checkpoint uses head only; no double "
            "normalization. Diagnostic Logit Lens, not intermediate predictions or "
            "causal evidence."
        ),
        rank_policy=RANK_POLICY,
        probability_policy=PROBABILITY_POLICY,
    )
