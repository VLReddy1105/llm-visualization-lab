from llm_lab.models.adapters.base import BaseModelAdapter
from llm_lab.models.adapters.qwen3 import Qwen3Adapter


def create_adapter(model, tokenizer, revision: str, device: str) -> BaseModelAdapter:
    """Select by the resident model's actual family, never by a display label."""
    family = getattr(model.config, "model_type", None)
    if family != "qwen3":
        raise ValueError(f"No scientific trace adapter is available for model family {family!r}")
    return Qwen3Adapter(model, tokenizer, revision, device)
