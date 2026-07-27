from pathlib import Path
from typing import Self

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import (
    AnyHttpUrl,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when a runtime operation is missing required configuration."""


class AppSettings(BaseSettings):
    """Typed application settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
    )

    chat_model: str | None = Field(default=None, validation_alias="LITELLM_MODEL")
    chat_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="LITELLM_API_KEY",
    )
    chat_api_base: str | None = Field(
        default=None,
        validation_alias="LITELLM_API_BASE",
    )
    chat_streaming: bool = Field(
        default=True,
        validation_alias="LITELLM_STREAMING",
    )
    chat_timeout_seconds: PositiveFloat = Field(
        default=30.0,
        validation_alias="LITELLM_TIMEOUT_SECONDS",
    )

    embedding_model: str = Field(
        default="text-embedding-3-large",
        validation_alias="OPENAI_EMBEDDING_MODEL",
    )
    embedding_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    embedding_timeout_seconds: PositiveFloat = Field(
        default=30.0,
        validation_alias="OPENAI_EMBEDDING_TIMEOUT_SECONDS",
    )

    docs_dir: Path = Field(default=Path("docs"), validation_alias="RAG_DOCS_DIR")
    index_dir: Path = Field(
        default=Path(".rag-index"),
        validation_alias="RAG_INDEX_DIR",
    )
    chunk_size: PositiveInt = Field(
        default=800,
        validation_alias="RAG_CHUNK_SIZE",
    )
    chunk_overlap: NonNegativeInt = Field(
        default=120,
        validation_alias="RAG_CHUNK_OVERLAP",
    )
    top_k: PositiveInt = Field(default=4, validation_alias="RAG_TOP_K")
    fetch_k_multiplier: PositiveInt = Field(
        default=4,
        validation_alias="RAG_FETCH_K_MULTIPLIER",
    )
    use_reranker: bool = Field(
        default=False,
        validation_alias="RAG_USE_RERANKER",
    )
    reranker_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="RAG_RERANKER_EMBEDDING_MODEL",
    )

    phoenix_enabled: bool = Field(
        default=True,
        validation_alias="PHOENIX_ENABLED",
    )
    phoenix_project_name: str = Field(
        default="explainable-agentic-rag",
        validation_alias="PHOENIX_PROJECT_NAME",
    )
    phoenix_collector_endpoint: AnyHttpUrl = Field(
        default="http://localhost:6006/v1/traces",
        validation_alias="PHOENIX_COLLECTOR_ENDPOINT",
    )

    @field_validator(
        "chat_model",
        "chat_api_base",
        mode="before",
    )
    @classmethod
    def normalize_optional_strings(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "embedding_model",
        "reranker_embedding_model",
        "phoenix_project_name",
    )
    @classmethod
    def require_non_empty_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        return self

    def require_chat_credentials(self) -> None:
        missing = []
        if not self.chat_model:
            missing.append("LITELLM_MODEL")
        if self.chat_api_key is None or not self.chat_api_key.get_secret_value():
            missing.append("LITELLM_API_KEY")
        if missing:
            raise ConfigurationError(
                "Missing required chat settings: "
                f"{', '.join(missing)}. Set them in the environment or `.env`."
            )

    def require_embedding_credentials(self) -> None:
        if (
            self.embedding_api_key is None
            or not self.embedding_api_key.get_secret_value()
        ):
            raise ConfigurationError(
                "Missing required embedding setting: OPENAI_API_KEY. "
                "Set it in the environment or `.env`."
            )


def get_settings() -> AppSettings:
    """Load and validate the current process configuration."""

    return AppSettings()


def get_llm_client(
    *,
    settings: AppSettings | None = None,
    streaming: bool | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    resolved = settings or get_settings()
    resolved.require_chat_credentials()

    kwargs = {
        "model": resolved.chat_model,
        "api_key": resolved.chat_api_key,
        "base_url": resolved.chat_api_base,
        "streaming": (resolved.chat_streaming if streaming is None else streaming),
        "timeout": float(resolved.chat_timeout_seconds),
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    return ChatOpenAI(**kwargs)


def get_embedding_client(
    *,
    settings: AppSettings | None = None,
) -> OpenAIEmbeddings:
    resolved = settings or get_settings()
    resolved.require_embedding_credentials()

    return OpenAIEmbeddings(
        model=resolved.embedding_model,
        api_key=resolved.embedding_api_key,
        timeout=float(resolved.embedding_timeout_seconds),
    )
