from typing import Literal

from pydantic import Field, field_validator

from llm_lab.config import MODEL_ID
from llm_lab.schemas.model import ModelMetadata, RuntimeMetadata, StrictModel


class RawInput(StrictModel):
    mode: Literal["raw"] = "raw"
    text: str = Field(min_length=1, max_length=32768)
    add_special_tokens: Literal[False] = False

    @field_validator("text")
    @classmethod
    def nonempty_without_modification(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must contain at least one non-whitespace character")
        return value


class NextTokenRequest(StrictModel):
    model_id: Literal["qwen3-0.6b"] = MODEL_ID
    input: RawInput
    top_k: int = Field(default=10, ge=1, le=100, strict=True)


class TokenRecord(StrictModel):
    token_index: int
    token_id: int
    tokenizer_piece: str
    decoded_token: str
    position_id: int
    is_special: bool
    character_start: int | None
    character_end: int | None


class InputResult(StrictModel):
    text: str
    effective_text: str
    mode: Literal["raw"] = "raw"
    add_special_tokens: Literal[False] = False
    token_count: int
    decoded_text: str
    tokens: list[TokenRecord]


class Candidate(StrictModel):
    rank: int
    token_id: int
    tokenizer_piece: str
    decoded_token: str
    raw_logit: float
    probability: float = Field(ge=0, le=1)


class Prediction(StrictModel):
    position: int
    next_position: int
    logits_shape: list[int]
    vocabulary_size: int
    top_k: list[Candidate]
    omitted_probability_mass: float = Field(ge=0, le=1)
    argmax_token_id: int


class NextTokenResponse(StrictModel):
    request_id: str
    model: ModelMetadata
    input: InputResult
    prediction: Prediction
    runtime: RuntimeMetadata
