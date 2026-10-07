"""Independent tracing-off reference, reusable for Stage 2 equivalence tests.

Uses a direct Hugging Face forward with all-position logits before selecting
the last position. It does not call the service inference implementation.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BaselineResult:
    token_ids: list[int]
    logits: torch.Tensor
    probabilities: torch.Tensor


def direct_hf_baseline(model, tokenizer, text: str, device: str) -> BaselineResult:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=False)
    inputs = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        output = model(
            **inputs,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        logits = output.logits[0, -1].float().cpu()
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float64).float()
    return BaselineResult(encoded["input_ids"][0].tolist(), logits, probabilities)
