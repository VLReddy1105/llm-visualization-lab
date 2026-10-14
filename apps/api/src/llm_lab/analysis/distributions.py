"""Full-vocabulary normalization and deterministic bounded candidate summaries."""

import torch

from llm_lab.errors import ServiceError

PROBABILITY_POLICY = (
    "Full-vocabulary CPU float64 softmax, returned probabilities cast to "
    "float32; never display-only normalization."
)
RANK_POLICY = (
    "Descending full-vocabulary logit; exact ties use ascending vocabulary ID (stable sort)."
)


def distribution(logits):
    values = logits.detach().cpu().reshape(-1)
    if (
        bool(torch.isnan(values).any())
        or bool(torch.isposinf(values).any())
        or not bool(torch.isfinite(values).any())
    ):
        raise ServiceError(
            "NONFINITE_ANALYSIS", "No finite vocabulary distribution is available.", 409
        )
    probabilities = torch.softmax(values, dim=-1, dtype=torch.float64).float()
    order = torch.argsort(values, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, len(order) + 1)
    return probabilities, order, ranks


def candidate(tokenizer, token_id, logits, probabilities, ranks):
    return {
        "token_id": token_id,
        "tokenizer_piece": tokenizer.convert_ids_to_tokens(token_id),
        "decoded_token": tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ),
        "rank": int(ranks[token_id]),
        "raw_logit": float(logits[token_id]),
        "probability": float(probabilities[token_id]),
    }
