"""Versioned, bounded Stage 6 analyses. Pydantic is the wire-contract authority."""

from typing import Annotated, Literal

from pydantic import Field, StrictFloat, StrictInt, field_validator, model_validator

from llm_lab.schemas.inference import Candidate
from llm_lab.schemas.model import StrictModel
from llm_lab.schemas.trace_api import Index, Statistics, TokenMetadata

METHOD_VERSION = "stage6.v1"
BLOCK_METHOD_VERSION = "stage7.v1"


class ComparisonMetrics(StrictModel):
    passed: bool
    bitwise_equal: bool
    absolute_tolerance: float
    relative_tolerance: float
    max_absolute_error: float
    mean_absolute_error: float
    relative_l2_error: float
    relative_l2_epsilon: float
    comparison_dtype: str


class PositionParameters(StrictModel):
    token_index: Index


class EvolutionRequest(StrictModel):
    kind: Literal["token_evolution"]
    parameters: PositionParameters


class LensParameters(PositionParameters):
    display_top_k: Annotated[StrictInt, Field(ge=1, le=20)] = 5
    candidate_ids: list[Index] = Field(default_factory=list, max_length=8)
    recipe: Literal["final_norm_then_lm_head"] = "final_norm_then_lm_head"

    @field_validator("candidate_ids")
    @classmethod
    def canonical_candidates(cls, ids):
        if len(set(ids)) != len(ids):
            raise ValueError("Candidate IDs must be unique")
        return sorted(ids)


class LensRequest(StrictModel):
    kind: Literal["logit_lens"]
    parameters: LensParameters


class SamplingParameters(PositionParameters):
    temperature: Annotated[StrictFloat, Field(gt=0, le=100, allow_inf_nan=False)] = 1.0
    top_k: Annotated[StrictInt, Field(ge=0)] = 0
    top_p: Annotated[StrictFloat, Field(gt=0, le=1, allow_inf_nan=False)] = 1.0
    display_top_k: Annotated[StrictInt, Field(ge=1, le=100)] = 10
    mode: Literal["sample", "greedy"] = "sample"

    @model_validator(mode="after")
    def greedy_defaults(self):
        if self.mode == "greedy" and (self.temperature, self.top_k, self.top_p) != (1, 0, 1):
            raise ValueError("Greedy requires temperature=1, top_k=0, top_p=1")
        return self


class SamplingRequest(StrictModel):
    kind: Literal["sampling_preview"]
    parameters: SamplingParameters


class BlockParameters(PositionParameters):
    layer_index: Index
    top_k: Annotated[StrictInt, Field(ge=1, le=64)] = 12
    ranking: Literal["absolute", "positive", "negative"] = "absolute"
    hidden_coordinate_index: Index = 0
    intermediate_coordinate_index: Index = 0


class ResidualStreamRequest(StrictModel):
    kind: Literal["residual_stream"]
    parameters: BlockParameters


class MLPInternalsRequest(StrictModel):
    kind: Literal["mlp_internals"]
    parameters: BlockParameters


AnalysisRequest = Annotated[
    EvolutionRequest | LensRequest | SamplingRequest | ResidualStreamRequest | MLPInternalsRequest,
    Field(discriminator="kind"),
]


class Checkpoint(StrictModel):
    checkpoint_index: int
    component_id: str
    label: str
    stage: Literal["embedding", "post_block_residual", "final_normalized"]
    layer_index: int | None


class AnalysisSource(StrictModel):
    trace_id: str
    token: TokenMetadata
    model_id: str
    hf_model_id: str
    revision: str
    tokenizer_revision: str
    device: str
    dtype: str
    layer_count: int
    hidden_size: int
    vocabulary_size: int
    config_digest: str
    implementation_digest: str
    value_kind: Literal["derived"] = "derived"


class RepresentationMetric(Checkpoint):
    l2_norm: float
    norm_change: float | None
    consecutive_distance: float | None
    cosine_previous: float | None
    cosine_embedding: float | None
    cosine_final_residual: float | None
    undefined: dict[str, str]
    statistics: Statistics


class LayerComparison(StrictModel):
    from_checkpoint: int
    to_checkpoint: int
    l2_distance: float
    norm_change: float
    cosine_similarity: float | None
    undefined_reason: str | None


class EvolutionResult(StrictModel):
    kind: Literal["token_evolution"] = "token_evolution"
    source: AnalysisSource
    checkpoints: list[RepresentationMetric]
    comparisons: list[LayerComparison]
    accumulation_dtype: Literal["float64"] = "float64"
    method: str


class LensCheckpoint(Checkpoint):
    normalization_applied: bool
    top_candidates: list[Candidate]
    comparison_candidates: list[Candidate]
    probability_sum: float
    entropy_nats: float
    omitted_probability_mass: float


class LensResult(StrictModel):
    kind: Literal["logit_lens"] = "logit_lens"
    source: AnalysisSource
    prediction_position: int
    next_position: int
    candidate_ids: list[int]
    checkpoints: list[LensCheckpoint]
    final_logits_comparison: ComparisonMetrics | None
    final_probabilities_comparison: ComparisonMetrics | None
    final_normalization_comparison: ComparisonMetrics
    comparison_note: str
    method: str
    rank_policy: str
    probability_policy: str


class SamplingCandidate(Candidate):
    model_probability: float
    processed_logit: float | None
    eligible: bool


class SamplingResult(StrictModel):
    kind: Literal["sampling_preview"] = "sampling_preview"
    source: AnalysisSource
    parameters: SamplingParameters
    candidates: list[SamplingCandidate]
    eligible_count: int
    probability_sum: float
    omitted_probability_mass: float
    processor_order: list[str]
    method: str
    rank_policy: str
    probability_policy: str


class CoordinateValue(StrictModel):
    index: int
    value: float


class VectorStatistics(StrictModel):
    count: int
    min: float
    max: float
    mean: float
    std: float
    l2_norm: float
    accumulation_dtype: Literal["float64"] = "float64"


class BlockOperation(StrictModel):
    symbol: str
    component_id: str
    name: str
    formula: str
    input_components: list[str]
    input_shapes: list[list[int]]
    output_shape: list[int]
    selected_shape: list[int]
    dtype: str
    original_device: str
    provenance: Literal["raw", "derived", "reconstructed", "summary"]
    method: str
    statistics: VectorStatistics
    coordinates: list[CoordinateValue]
    selected_coordinate: CoordinateValue


class ResidualCoordinate(StrictModel):
    index: int
    before: float
    update: float
    after: float
    signed_delta: float
    addition_error: float


class ResidualComparison(StrictModel):
    name: Literal["attention", "mlp", "block"]
    before_symbol: str
    update_symbol: str
    after_symbol: str
    update_provenance: Literal["raw", "derived"]
    before_norm: float
    update_norm: float
    after_norm: float
    delta_norm: float
    cosine_before_after: float | None
    update_to_residual_ratio: float | None
    undefined: dict[str, str]
    coordinates: list[ResidualCoordinate]
    selected_coordinate: ResidualCoordinate


class BlockReconciliation(StrictModel):
    name: str
    formula: str
    provenance: Literal["derived", "independent_reference"] = "derived"
    sources: list[str]
    scope: str
    metrics: ComparisonMetrics


class GateCoordinate(StrictModel):
    index: int
    gate: float
    activated_gate: float
    up: float
    product: float
    multiplication_error: float


class ResidualStreamResult(StrictModel):
    kind: Literal["residual_stream"] = "residual_stream"
    source: AnalysisSource
    layer_index: int
    operations: list[BlockOperation]
    comparisons: list[ResidualComparison]
    reconciliation: list[BlockReconciliation]
    method: str
    ranking_policy: str


class MLPInternalsResult(StrictModel):
    kind: Literal["mlp_internals"] = "mlp_internals"
    source: AnalysisSource
    layer_index: int
    intermediate_size: int
    operations: list[BlockOperation]
    selected_gate_coordinate: GateCoordinate
    gate_coordinates: list[GateCoordinate]
    reconciliation: list[BlockReconciliation]
    method: str
    ranking_policy: str


class AnalysisError(StrictModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)
    retryable: bool = False


class AnalysisJob(StrictModel):
    analysis_id: str
    trace_id: str
    method_version: Literal["stage6.v1", "stage7.v1"] = METHOD_VERSION
    request: AnalysisRequest
    status: Literal["queued", "running", "ready", "failed", "cancelled"]
    completed_checkpoints: int
    total_checkpoints: int
    created_at: str
    expires_at: str
    retained_bytes: int
    cache_hit: bool = False
    error: AnalysisError | None = None
    result: (
        EvolutionResult
        | LensResult
        | SamplingResult
        | ResidualStreamResult
        | MLPInternalsResult
        | None
    ) = None
