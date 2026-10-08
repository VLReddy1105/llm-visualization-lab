"""Public bounded contracts, independent of internal dataclasses and tensors."""

from typing import Annotated, Any, Literal

from pydantic import Field, StrictInt, field_validator

from llm_lab.schemas.inference import Candidate
from llm_lab.schemas.model import StrictModel

Index = Annotated[StrictInt, Field(ge=0)]


class CaptureRequest(StrictModel):
    profile: Literal["summary", "overview", "layer", "attention", "mlp", "deep"] = "summary"
    layers: list[Index] = Field(default_factory=list, max_length=28)
    tokens: list[Index] | None = Field(default=None, min_length=1, max_length=512)
    heads: list[Index] | None = Field(default=None, min_length=1, max_length=16)


class CreateTraceRequest(StrictModel):
    prompt: str = Field(min_length=1, max_length=32768)
    capture: CaptureRequest = Field(default_factory=CaptureRequest)
    display_top_k: Annotated[StrictInt, Field(ge=1, le=100)] = 10

    @field_validator("prompt")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must contain text")
        return value


class IndexSelector(StrictModel):
    index: Index


class RangeSelector(StrictModel):
    start: Index
    stop: Annotated[StrictInt, Field(ge=1)]
    step: Annotated[StrictInt, Field(ge=1)] = 1


class SliceRequest(StrictModel):
    component_id: str | None = Field(default=None, max_length=128)
    selectors: dict[str, IndexSelector | RangeSelector] = Field(default_factory=dict, max_length=6)
    format: Literal["json", "binary"] = "json"


class Coordinates(StrictModel):
    count: int
    start: int | None = None
    stop: int | None = None
    step: int | None = None
    indices: list[int] | None = None


class Statistics(StrictModel):
    finite_count: int
    nonfinite_count: int
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    l2_norm: float | None
    accumulation_dtype: str
    value_kind: Literal["summary"]
    reduction_scope: str
    masked_count: int
    total_count: int
    finite: bool
    undefined_fields: list[str]


class ComponentDescriptor(StrictModel):
    component_id: str
    tensor_id: str | None
    axes: list[str]
    logical_shape: list[int]
    storage_shape: list[int]
    coverage: dict[str, Coordinates]
    dtype: str
    storage_dtype: str
    original_device: str
    storage_device: str
    numel: int
    byte_size: int
    provenance: Literal["raw", "derived", "reconstructed", "summary"]
    sources: list[str]
    method: str
    availability: str
    raw_available: bool
    unavailable_reason: str | None
    statistics: Statistics | None


class Page(StrictModel):
    items: list[dict[str, Any]]
    total: int
    next_cursor: str | None = None


class TokenMetadata(StrictModel):
    token_index: int
    token_id: int
    tokenizer_piece: str
    decoded_token: str
    position_id: int
    is_special: bool
    character_start: int | None
    character_end: int | None
    origin: str


class TokenPage(StrictModel):
    items: list[TokenMetadata]
    total: int
    next_cursor: str | None = None


class ComponentPage(StrictModel):
    items: list[ComponentDescriptor]
    total: int
    next_cursor: str | None = None


class Architecture(StrictModel):
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


class TraceModel(StrictModel):
    model_id: str
    hf_model_id: str
    revision: str
    tokenizer_revision: str
    device: str
    dtype: str
    architecture: Architecture


class TraceInput(StrictModel):
    text: str
    mode: Literal["raw"]
    add_special_tokens: Literal[False]
    sequence_length: int
    tokens: TokenPage


class EffectiveCapture(StrictModel):
    profile: Literal["summary", "overview", "layer", "attention", "mlp", "deep"]
    layers: list[int]
    token_indices: list[int]
    query_heads: list[int] | None
    raw_tensors: bool
    statistics: bool
    max_bytes: int


class TraceSummary(StrictModel):
    layer_count: int
    retained_bytes: int
    estimated_bytes: int
    transient_estimated_bytes: int
    duration_ms: float
    forward_duration_ms: float
    retention_policy: Literal["ttl_or_capacity"]


class TracePrediction(StrictModel):
    prediction_position: int
    next_position: int
    vocabulary_size: int
    normalization: str
    top_k: list[Candidate]
    omitted_probability_mass: float


class AvailableComponents(StrictModel):
    layers: list[int]
    attention_layers: list[int]
    mlp_layers: list[int]
    component_count: int
    raw_component_count: int
    components_url: str


class TraceOverview(StrictModel):
    model: TraceModel
    input: TraceInput
    capture: EffectiveCapture
    summary: TraceSummary
    prediction: TracePrediction
    available: AvailableComponents
    provenance: dict[str, int]
    validation: dict[str, Any]
    warnings: list[str]
    runtime: dict[str, Any]
    links: dict[str, str]


class TraceEnvelope(StrictModel):
    trace_id: str
    schema_version: Literal["1.0.0"] = "1.0.0"
    status: Literal["queued", "running", "ready", "failed", "cancelled"]
    created_at: str
    last_access_at: str
    completed_at: str | None
    expires_at: str | None
    retained_bytes: int
    reserved_bytes: int
    metadata: dict[str, Any] | None
    overview: TraceOverview | None = None
    error: dict[str, Any] | None


class ComponentGroup(StrictModel):
    trace_id: str
    layer_index: int
    kind: str
    structure: dict[str, Any]
    components: list[ComponentDescriptor]
    validation: dict[str, Any]


class Nonfinite(StrictModel):
    flat_index: int
    kind: Literal["nan", "positive_infinity", "negative_infinity"]


class TensorSlice(StrictModel):
    trace_id: str
    tensor_id: str
    component_id: str
    provenance: Literal["raw", "derived", "reconstructed", "summary"]
    sources: list[str]
    method: str
    axes: list[str]
    logical_shape: list[int]
    shape: list[int]
    coordinates: dict[str, Coordinates]
    selectors: dict[str, Any]
    dtype: str
    storage_dtype: str
    wire_dtype: str
    values: list[float | int | None]
    nonfinite: list[Nonfinite]
    masked_flat_indices: list[int]


class AttentionHead(StrictModel):
    trace_id: str
    layer_index: int
    head_index: int
    kv_head_index: int
    components: list[ComponentDescriptor]
    weights: TensorSlice | None
    query: TensorSlice | None
    validation: dict[str, Any]


class LogitsResponse(StrictModel):
    trace_id: str
    prediction_position: int
    next_position: int
    vocabulary_size: int
    normalization: str
    top_k: list[Candidate]
    omitted_probability_mass: float
    logits: ComponentDescriptor
    probabilities: ComponentDescriptor


class StoreStatus(StrictModel):
    trace_count: int
    retained_bytes: int
    reserved_bytes: int
    accounted_bytes: int
    max_bytes: int
    ttl_seconds: float
    max_entries: int
    eviction_count: int
    expiration_count: int
    active_leases: int
    deferred_removals: int
    tombstone_count: int
    request_forward_passes: int
    counter_scope: str
    max_elements: int
    max_response_bytes: int


class CancellationResponse(StrictModel):
    trace_id: str
    status: Literal["cancelled"]
    cancellation_requested: Literal[True]


class AttentionRelation(StrictModel):
    trace_id: str
    layer_index: int
    head_index: int
    kv_head_index: int
    query_token_index: int
    key_token_index: int
    scaling: float
    masked: bool
    observations: dict[str, TensorSlice]
    normalization: str
