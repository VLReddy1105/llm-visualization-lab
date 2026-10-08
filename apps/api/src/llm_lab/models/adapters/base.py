from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from llm_lab.schemas.components import ComponentSpec
from llm_lab.schemas.trace import ArchitectureMetadata
from llm_lab.tracing.capture_plan import CapturePlan


@dataclass(frozen=True)
class Observation:
    spec: ComponentSpec
    tensor: torch.Tensor


@dataclass(frozen=True)
class HookTarget:
    module: torch.nn.Module
    kind: Literal["pre", "forward"]
    specs: tuple[ComponentSpec, ...]
    observations: Callable[[tuple, dict, Any], Iterable[Observation]]


class BaseModelAdapter(ABC):
    """Translate an existing resident model's boundaries into semantic observations.

    The adapter never installs hooks or owns another model. Registration, bounded
    CPU snapshots and cleanup belong to the run-scoped trace session.
    """

    def __init__(self, model, tokenizer, revision: str, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.revision = revision
        self.device = str(device)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    @abstractmethod
    def architecture(self) -> ArchitectureMetadata: ...

    def architecture_metadata(self) -> ArchitectureMetadata:
        return self.architecture

    def tokenize(self, *args, **kwargs):
        """Pass through unchanged; the Stage 1 caller owns the exact raw policy."""
        return self.tokenizer(*args, **kwargs)

    def forward(self, *, logits_to_keep: int = 1, **inputs):
        """Use the same forward arguments as the trusted Stage 1 path."""
        return self.model(
            **inputs,
            logits_to_keep=logits_to_keep,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )

    @abstractmethod
    def validate_for_trace(self) -> None: ...

    @abstractmethod
    def hook_targets(self, plan: CapturePlan, sequence_length: int) -> tuple[HookTarget, ...]: ...

    @abstractmethod
    def component_specs(
        self, plan: CapturePlan, sequence_length: int
    ) -> tuple[ComponentSpec, ...]: ...

    @abstractmethod
    def native_mlp_projection(self, layer_index: int, gated: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def native_rope(
        self, normalized: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor: ...

    @abstractmethod
    def attention_scaling(self, layer_index: int) -> float: ...

    @abstractmethod
    def native_norm(
        self, stage: str, values: torch.Tensor, layer_index: int | None = None
    ) -> torch.Tensor: ...

    @abstractmethod
    def native_mlp_activation(self, layer_index: int, values: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def native_lm_head(self, normalized: torch.Tensor) -> torch.Tensor: ...
