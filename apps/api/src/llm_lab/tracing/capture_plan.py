"""Strict, immutable capture selection; execution and estimation live in the engine."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from llm_lab.schemas.trace import ArchitectureMetadata

CaptureProfile = Literal["summary", "overview", "layer", "attention", "mlp", "deep"]
_PROFILES = frozenset({"summary", "overview", "layer", "attention", "mlp", "deep"})
_DETAILED = frozenset({"layer", "attention", "mlp", "deep"})
_ATTENTION = frozenset({"layer", "attention", "deep"})
_MLP = frozenset({"layer", "mlp", "deep"})


def _indices(name: str, value, *, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list of integer indices")
    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if any(type(index) is not int for index in value):
        raise TypeError(f"{name} requires integer indices; bool and coercion are not accepted")
    if any(index < 0 for index in value):
        raise ValueError(f"{name} cannot contain negative indices")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} cannot contain duplicate indices")
    return tuple(sorted(value))


@dataclass(frozen=True, slots=True)
class CapturePlan:
    """A bounded scientific request, separate from inference options.

    Summary retains statistics across all layers and only final output tensors.
    Overview retains base raw states across all layers. Detailed profiles capture
    base states and requested internals only for explicitly selected layers.
    Token selectors reduce query/residual positions, never the key context.
    """

    profile: CaptureProfile = "summary"
    layers: tuple[int, ...] = ()
    token_indices: tuple[int, ...] | None = None
    query_heads: tuple[int, ...] | None = None
    raw_tensors: bool | None = None
    statistics: bool = True
    retain_base_states: bool = False
    max_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.profile not in _PROFILES:
            raise ValueError(f"Unknown capture profile: {self.profile!r}")
        object.__setattr__(self, "layers", _indices("layers", self.layers, allow_empty=True))
        if self.token_indices is not None:
            object.__setattr__(self, "token_indices", _indices("token_indices", self.token_indices))
        if self.query_heads is not None:
            object.__setattr__(self, "query_heads", _indices("query_heads", self.query_heads))
            if not self.captures_attention:
                raise ValueError("query_heads requires an attention-detail capture profile")
        if self.profile in _DETAILED and not self.layers:
            raise ValueError("Detailed capture profiles require explicitly selected layers")
        if self.profile not in _DETAILED and self.layers:
            raise ValueError("summary and overview capture all base layers; omit layers")
        if self.raw_tensors is not None and type(self.raw_tensors) is not bool:
            raise TypeError("raw_tensors must be bool or None")
        if self.profile == "summary" and self.raw_tensors:
            raise ValueError(
                "summary is statistics-only for internals; select overview for raw states"
            )
        if type(self.retain_base_states) is not bool:
            raise TypeError("retain_base_states must be bool")
        if type(self.statistics) is not bool:
            raise TypeError("statistics must be bool")
        if type(self.max_bytes) is not int:
            raise TypeError("max_bytes must be an integer; bool is not accepted")
        if self.max_bytes < 1:
            raise ValueError("max_bytes must be positive")

    @property
    def effective_raw(self) -> bool:
        return self.profile != "summary" if self.raw_tensors is None else self.raw_tensors

    @property
    def captures_attention(self) -> bool:
        return self.profile in _ATTENTION

    @property
    def captures_mlp(self) -> bool:
        return self.profile in _MLP

    @property
    def detailed_layers(self) -> tuple[int, ...]:
        return self.layers if self.profile in _DETAILED else ()

    def resolve(self, architecture: "ArchitectureMetadata", sequence_length: int) -> "CapturePlan":
        """Resolve full coverage and reject out-of-bounds requests before hooks attach."""
        if type(sequence_length) is not int:
            raise TypeError("sequence_length must be an integer")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if any(index >= architecture.layers for index in self.layers):
            raise ValueError(f"Layer indices must be smaller than {architecture.layers}")
        tokens = self.token_indices or tuple(range(sequence_length))
        if any(index >= sequence_length for index in tokens):
            raise ValueError(f"Token indices must be smaller than {sequence_length}")
        heads = self.query_heads
        if self.captures_attention:
            heads = heads or tuple(range(architecture.query_heads))
            if any(index >= architecture.query_heads for index in heads):
                raise ValueError(
                    f"Query-head indices must be smaller than {architecture.query_heads}"
                )
        return replace(
            self, token_indices=tokens, query_heads=heads, raw_tensors=self.effective_raw
        )
