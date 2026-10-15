"""Bounded Stage 8 sessions; immutable decision records have source-qualified identity."""

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from llm_lab.schemas.analysis import SamplingParameters
from llm_lab.schemas.inference import RawInput
from llm_lab.schemas.model import StrictModel


class GenerationSampling(SamplingParameters):
    token_index: Literal[0] = (
        0  # Decision position is resolved per forward, not supplied by clients.
    )
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1, strict=True)

    @model_validator(mode="after")
    def seed_policy(self):
        if self.mode == "sample" and self.seed is None:
            raise ValueError("Sampled generation requires an explicit seed")
        if self.mode == "greedy" and self.seed is not None:
            raise ValueError("Greedy generation does not consume a seed")
        return self


class GenerationRequest(StrictModel):
    model_id: Literal["qwen3-0.6b"] = "qwen3-0.6b"
    input: RawInput
    max_new_tokens: int = Field(default=8, ge=1, le=32, strict=True)
    sampling: GenerationSampling = Field(default_factory=lambda: GenerationSampling(mode="greedy"))
    capture_profile: Literal["none", "overview", "deep"] = "deep"
    capture_layer: int = Field(default=12, ge=0, le=27, strict=True)
    retain_traces: int = Field(default=4, ge=0, le=8, strict=True)
    use_cache: Literal[False] = False
    request_key: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")


class GenerationCandidate(StrictModel):
    token_id: int
    tokenizer_piece: str
    decoded_in_isolation: str
    raw_logit: float
    model_probability: float
    processed_logit: float | None
    adjusted_probability: float | None
    raw_rank: int
    adjusted_rank: int | None
    eligible: bool


class GenerationStep(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    generation_id: str
    step_id: str
    step_index: int
    prefix_token_ids: list[int]
    prefix_sha256: str
    decoded_prefix: str
    decision_position: int
    selected_position: int
    vocabulary_size: int
    selected_token_id: int
    tokenizer_piece: str
    selected_display_text: str | None
    display_note: str
    decoded_continuation: str
    decoded_full_sequence: str
    selected_raw_logit: float
    selected_model_probability: float
    selected_adjusted_probability: float | None
    raw_argmax_token_id: int
    candidates: list[GenerationCandidate]
    distribution_statistics: dict[str, float | int | None]
    sampling: GenerationSampling
    source_trace_id: str | None
    source_capture_coverage: dict
    capture_unavailable_reason: str | None
    forward_execution_id: str
    forward_completion_status: Literal["completed"] = "completed"
    forward_duration_ms: float
    elapsed_ms: float
    provenance: dict


class GenerationSession(StrictModel):
    generation_id: str
    request_key: str
    original_prompt: str
    original_input_ids: list[int]
    original_decoded_prompt: str
    model_id: str
    hf_model_id: str
    model_revision: str
    tokenizer_revision: str
    tokenizer_identity: str
    device: str
    dtype: str
    execution_configuration: dict
    sampling: GenerationSampling
    started_at: str
    status: Literal["queued", "running", "completed", "stopped", "cancelled", "failed"]
    completed_at: str | None = None
    stop_reason: str | None = None
    max_new_tokens: int
    budgets: dict
    completed_steps: int = 0
    attempted_forwards: int = 0
    active_forward_execution_id: str | None = None
    cancellation_requested: bool = False
    selected_token_ids: list[int] = Field(default_factory=list)
    decoded_continuation: str = ""
    decoded_full_sequence: str
    trace_retention_summary: dict = Field(default_factory=dict)
    result_bytes: int = 0
    accounted_state_bytes: int = 0
    session_duration_ms: float | None = None
    expires_at: str
    error: dict | None = None
    steps: list[GenerationStep] = Field(default_factory=list)


class GenerationStepPage(StrictModel):
    generation_id: str
    total: int
    offset: int
    items: list[GenerationStep]


class GenerationStoreStatus(StrictModel):
    session_count: int
    inflight: int
    reserved_state_bytes: int
    result_bytes: int
    eviction_count: int
    expiration_count: int
