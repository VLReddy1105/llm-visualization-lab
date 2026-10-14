"""Saved-logit preview. No RNG, tokenizer, weights, generation or forward execution."""

import torch
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from llm_lab.analysis.distributions import PROBABILITY_POLICY, RANK_POLICY, distribution
from llm_lab.analysis.token_journey import source_metadata
from llm_lab.errors import ServiceError
from llm_lab.schemas.analysis import SamplingResult


def process_logits(logits, parameters):
    if parameters.top_k > logits.numel():
        raise ServiceError("INVALID_TOP_K", "Sampling top-k exceeds the vocabulary.", 422)
    # Promote before processing: the pinned top-p implementation itself performs
    # a softmax. Float64 avoids the measured CPU float32 normalization discrepancy.
    scores = logits.detach().cpu().double().reshape(1, -1).clone()
    if not bool(torch.isfinite(scores).all()):
        raise ServiceError("NONFINITE_ANALYSIS", "Saved logits must be finite.", 409)
    if parameters.mode == "greedy":
        keep = scores.argmax(dim=-1)
        filtered = torch.full_like(scores, -torch.inf)
        filtered[0, keep] = scores[0, keep]
        return filtered.reshape(-1)
    ids = torch.empty((1, 0), dtype=torch.long)
    scores = TemperatureLogitsWarper(parameters.temperature)(ids, scores)
    if not bool(torch.isfinite(scores).all()):
        raise ServiceError(
            "INVALID_TEMPERATURE", "Temperature overflows the processing precision.", 422
        )
    if parameters.top_k:
        scores = TopKLogitsWarper(parameters.top_k)(ids, scores)
    if parameters.top_p < 1:
        scores = TopPLogitsWarper(parameters.top_p)(ids, scores)
    return scores.reshape(-1)


def sampling_preview(trace, parameters):
    source = source_metadata(trace, parameters.token_index)
    if parameters.token_index != trace.output.prediction_position:
        raise ServiceError(
            "COMPONENT_NOT_CAPTURED",
            "Sampling preview requires the retained final prompt position.",
            409,
        )
    logits = trace.tensor(trace.output.logits, copy=False).reshape(-1)
    processed = process_logits(logits, parameters)
    probabilities, order, ranks = distribution(processed)
    base, _, _ = distribution(logits)
    eligible = torch.isfinite(processed)
    top = order[: min(parameters.display_top_k, int(eligible.sum()))].tolist()
    identities = {c.token_id: c for c in trace.output.candidates}
    if any(t not in identities for t in top):
        raise ServiceError(
            "CANDIDATE_METADATA_NOT_CAPTURED",
            (
                "Requested candidates lack retained tokenizer metadata; capture an API "
                "trace with its full candidate summary."
            ),
            409,
        )
    candidates = [
        {
            "token_id": t,
            "tokenizer_piece": identities[t].tokenizer_piece,
            "decoded_token": identities[t].decoded_token,
            "rank": int(ranks[t]),
            "raw_logit": float(logits[t]),
            "model_probability": float(base[t]),
            "processed_logit": float(processed[t]),
            "probability": float(probabilities[t]),
            "eligible": True,
        }
        for t in top
    ]
    return SamplingResult(
        source=source,
        parameters=parameters,
        candidates=candidates,
        eligible_count=int(eligible.sum()),
        probability_sum=float(probabilities.double().sum()),
        omitted_probability_mass=max(0.0, 1 - float(probabilities[top].double().sum())),
        processor_order=["argmax"]
        if parameters.mode == "greedy"
        else ["temperature", "top_k", "top_p", "full_remaining_vocabulary_softmax"],
        method=(
            "Pinned Transformers processors on CPU float64 saved logits. Top-k "
            "retains cutoff ties; top-p uses native ascending-sort boundary policy "
            "and keeps at least one. No inherited penalties, RNG draw or "
            "continuation."
        ),
        rank_policy=RANK_POLICY,
        probability_policy=PROBABILITY_POLICY,
    )
