"""Immutable in-process semantic trace types; these are not JSON/HTTP contracts."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

from llm_lab.schemas.components import TensorDescriptor

if TYPE_CHECKING:
    from llm_lab.tracing.capture_plan import CapturePlan


def _immutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_immutable(item) for item in value)
    if isinstance(value, set):
        return frozenset(_immutable(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ArchitectureMetadata:
    layers: int
    hidden_size: int
    intermediate_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    vocabulary_size: int
    rms_norm_epsilon: float
    rope_theta: float
    tied_embeddings: bool
    attention_dropout: float
    attention_backend: str
    config_digest: str
    implementation_digest: str
    adapter_id: str
    adapter_version: str


@dataclass(frozen=True, slots=True)
class TokenTrace:
    token_index: int
    token_id: int
    tokenizer_piece: str
    decoded_token: str
    position_id: int
    is_special: bool
    character_start: int | None = None
    character_end: int | None = None
    origin: str = "user"


@dataclass(frozen=True, slots=True)
class TraceMetadata:
    trace_id: str
    created_at: str
    completed_at: str
    model_id: str
    hf_model_id: str
    revision: str
    tokenizer_revision: str
    device: str
    dtype: str
    prompt: str
    token_ids: tuple[int, ...]
    pieces: tuple[str, ...]
    sequence_length: int
    architecture: ArchitectureMetadata
    capture_plan: "CapturePlan"
    estimated_bytes: int
    actual_retained_bytes: int
    duration_ms: float
    forward_duration_ms: float
    warnings: tuple[str, ...] = ()
    runtime: Mapping[str, Any] = field(default_factory=dict)
    transient_estimated_bytes: int = 0
    validation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime", _immutable(self.runtime))
        object.__setattr__(self, "validation", _immutable(self.validation))


@dataclass(frozen=True, slots=True)
class EmbeddingTrace:
    output: str = "embedding.output"
    position_encoding: str = "rotary attention encoding; no additive positional embedding"


@dataclass(frozen=True, slots=True)
class AttentionTrace:
    layer_index: int
    input_normalized: str | None = None
    query_projected: str | None = None
    key_projected: str | None = None
    value: str | None = None
    query_normalized: str | None = None
    key_normalized: str | None = None
    query_rotary: str | None = None
    key_rotary: str | None = None
    similarity: str | None = None
    scaled_scores: str | None = None
    masked_scores: str | None = None
    weights: str | None = None
    weights_reconstructed: str | None = None
    head_output: str | None = None
    head_output_reconstructed: str | None = None
    concatenated: str | None = None
    output: str | None = None
    mask: str | None = None
    query_to_kv_head: tuple[int, ...] = ()
    scaling: float | None = None
    mask_kind: str = "causal, unpadded, cache disabled"


@dataclass(frozen=True, slots=True)
class MLPTrace:
    layer_index: int
    input_normalized: str | None = None
    gate: str | None = None
    up: str | None = None
    activated_gate: str | None = None
    gated: str | None = None
    output: str | None = None
    activation: str = "silu"


@dataclass(frozen=True, slots=True)
class ResidualTrace:
    before: str
    attention_update: str
    after_attention: str
    mlp_update: str
    after_mlp: str


@dataclass(frozen=True, slots=True)
class LayerTrace:
    layer_index: int
    residual: ResidualTrace
    attention: AttentionTrace | None = None
    mlp: MLPTrace | None = None
    output: str | None = None


@dataclass(frozen=True, slots=True)
class OutputCandidate:
    rank: int
    token_id: int
    tokenizer_piece: str
    decoded_token: str
    raw_logit: float
    probability: float


@dataclass(frozen=True, slots=True)
class OutputTrace:
    prediction_position: int
    next_position: int
    residual_final: str = "output.residual_final"
    normalized: str = "output.normalized"
    lm_head_input: str = "output.lm_head_input"
    logits: str = "output.logits"
    probabilities: str = "output.probabilities"
    candidates: tuple[OutputCandidate, ...] = ()
    omitted_probability_mass: float = 0.0


@dataclass(frozen=True, slots=True)
class TraceResult:
    metadata: TraceMetadata
    descriptors: Mapping[str, TensorDescriptor]
    layers: tuple[LayerTrace, ...]
    embedding: EmbeddingTrace
    output: OutputTrace
    tokens: tuple[TokenTrace, ...]
    _tensors: Mapping[str, torch.Tensor] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", MappingProxyType(dict(self.descriptors)))
        object.__setattr__(self, "_tensors", MappingProxyType(dict(self._tensors)))

    def tensor(self, component_id: str, *, copy: bool = True) -> torch.Tensor:
        """Return an owned copy by default; internal borrowers must not mutate.

        Summaries are not scientific tensor substitutes. Omitted tensors raise
        ``KeyError`` rather than returning a fabricated or empty array.
        """
        tensor = self._tensors[component_id]
        return tensor.clone() if copy else tensor
