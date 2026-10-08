from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_ID = "qwen3-0.6b"
HF_MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_LAB_", env_file=".env", extra="ignore")

    device: str = "auto"
    auto_load: bool = True
    model_revision: str = Field(default=MODEL_REVISION, pattern=r"^[0-9a-f]{40}$")
    cache_dir: str | None = None
    max_input_tokens: int = Field(default=128, ge=1, le=512)
    max_prompt_characters: int = Field(default=8192, ge=1, le=32768)
    queue_capacity: int = Field(default=3, ge=1, le=3)
    trace_max_bytes: int = Field(default=256 * 1024 * 1024, ge=1, le=1024 * 1024 * 1024)
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    @field_validator("cors_origins")
    @classmethod
    def local_origins_only(cls, value: list[str]) -> list[str]:
        from urllib.parse import urlparse

        for origin in value:
            parsed = urlparse(origin)
            if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"}:
                raise ValueError("Stage 1 CORS origins must use local HTTP hosts")
            if parsed.path or parsed.query or parsed.fragment:
                raise ValueError("CORS origins must not include a path, query, or fragment")
        return value
