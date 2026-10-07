from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_lab.config import HF_MODEL_ID, MODEL_ID


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorInfo(StrictModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)
    retryable: bool = False
    request_id: str


class ErrorResponse(StrictModel):
    error: ErrorInfo


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    api_version: Literal["v1"] = "v1"
    application_version: str = "0.1.0"


class ModelStatus(StrictModel):
    model_id: str = MODEL_ID
    hf_model_id: str = HF_MODEL_ID
    state: Literal["unloaded", "loading", "ready", "failed"]
    configured_revision: str
    resolved_revision: str | None = None
    detected_device: str | None = None
    available_devices: list[str] = Field(default_factory=list)
    device: str | None = None
    dtype: str | None = None
    dtype_reason: str | None = None
    loading_phase: str | None = None
    load_duration_ms: float | None = None
    error: str | None = None
    error_code: str | None = None
    max_input_tokens: int
    max_prompt_characters: int
    queue_capacity: int
    pending_operations: int = 0
    execution_busy: bool = False


class ServiceStatus(StrictModel):
    inference_ready: bool
    model: ModelStatus


class ModelLoadRequest(StrictModel):
    device: str = "auto"
    dtype_policy: Literal["scientific_default"] = "scientific_default"
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")


class ModelMetadata(StrictModel):
    model_id: str = MODEL_ID
    hf_model_id: str = HF_MODEL_ID
    revision: str
    tokenizer_revision: str
    device: str
    dtype: str
    attention_backend: Literal["eager"] = "eager"
    vocabulary_size: int
    layer_count: int
    hidden_size: int


class RuntimeMetadata(StrictModel):
    python_version: str
    torch_version: str
    transformers_version: str
    probability_dtype: Literal["float32"] = "float32"
    softmax_accumulation_dtype: Literal["float64"] = "float64"
    load_duration_ms: float
    forward_duration_ms: float
    inference_duration_ms: float
    cache_enabled: Literal[False] = False
    tracing_enabled: Literal[False] = False
