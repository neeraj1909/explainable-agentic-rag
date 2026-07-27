from pathlib import Path

import pytest
from pydantic import ValidationError

from app import config
from app.config import (
    AppSettings,
    ConfigurationError,
    get_embedding_client,
    get_llm_client,
)
from app.observability import setup_phoenix_tracing


CONFIG_ENV_VARS = (
    "LITELLM_MODEL",
    "LITELLM_API_KEY",
    "LITELLM_API_BASE",
    "LITELLM_STREAMING",
    "LITELLM_TIMEOUT_SECONDS",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_TIMEOUT_SECONDS",
    "RAG_DOCS_DIR",
    "RAG_INDEX_DIR",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_TOP_K",
    "RAG_FETCH_K_MULTIPLIER",
    "RAG_USE_RERANKER",
    "RAG_RERANKER_EMBEDDING_MODEL",
    "PHOENIX_ENABLED",
    "PHOENIX_PROJECT_NAME",
    "PHOENIX_COLLECTOR_ENDPOINT",
)


@pytest.fixture(autouse=True)
def clear_configuration_environment(monkeypatch):
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_settings_have_portable_safe_defaults():
    settings = AppSettings(_env_file=None)

    assert settings.chat_streaming is True
    assert settings.chat_timeout_seconds == 30.0
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.embedding_timeout_seconds == 30.0
    assert settings.docs_dir == Path("docs")
    assert settings.index_dir == Path(".rag-index")
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.top_k == 4
    assert settings.fetch_k_multiplier == 4
    assert settings.use_reranker is False
    assert settings.reranker_embedding_model == "text-embedding-3-small"
    assert settings.phoenix_enabled is True
    assert str(settings.phoenix_collector_endpoint) == (
        "http://localhost:6006/v1/traces"
    )


def test_settings_parse_environment_values(monkeypatch):
    monkeypatch.setenv("LITELLM_STREAMING", "false")
    monkeypatch.setenv("LITELLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OPENAI_EMBEDDING_TIMEOUT_SECONDS", "18")
    monkeypatch.setenv("RAG_DOCS_DIR", "fixtures/docs")
    monkeypatch.setenv("RAG_INDEX_DIR", "var/index")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "512")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "64")
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_FETCH_K_MULTIPLIER", "3")
    monkeypatch.setenv("RAG_USE_RERANKER", "true")
    monkeypatch.setenv("PHOENIX_ENABLED", "false")

    settings = AppSettings(_env_file=None)

    assert settings.chat_streaming is False
    assert settings.chat_timeout_seconds == 12.5
    assert settings.embedding_timeout_seconds == 18.0
    assert settings.docs_dir == Path("fixtures/docs")
    assert settings.index_dir == Path("var/index")
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.top_k == 7
    assert settings.fetch_k_multiplier == 3
    assert settings.use_reranker is True
    assert settings.phoenix_enabled is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"top_k": 0}, "greater than 0"),
        ({"chat_timeout_seconds": 0}, "greater than 0"),
        (
            {"chunk_size": 100, "chunk_overlap": 100},
            "RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE",
        ),
        ({"phoenix_collector_endpoint": "not-a-url"}, "valid URL"),
    ],
)
def test_settings_reject_invalid_values(overrides, message):
    with pytest.raises(ValidationError, match=message):
        AppSettings(_env_file=None, **overrides)


def test_chat_client_fails_fast_with_actionable_missing_settings():
    settings = AppSettings(_env_file=None)

    with pytest.raises(
        ConfigurationError,
        match="LITELLM_MODEL.*LITELLM_API_KEY",
    ):
        get_llm_client(settings=settings)


def test_embedding_client_fails_fast_with_actionable_missing_settings():
    settings = AppSettings(_env_file=None)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        get_embedding_client(settings=settings)


def test_clients_receive_typed_settings(monkeypatch):
    chat_calls = []
    embedding_calls = []

    monkeypatch.setattr(
        config,
        "ChatOpenAI",
        lambda **kwargs: chat_calls.append(kwargs) or kwargs,
    )
    monkeypatch.setattr(
        config,
        "OpenAIEmbeddings",
        lambda **kwargs: embedding_calls.append(kwargs) or kwargs,
    )

    settings = AppSettings(
        _env_file=None,
        chat_model="test-chat",
        chat_api_key="test-chat-key",
        chat_api_base="https://models.example.test/v1",
        chat_streaming=True,
        chat_timeout_seconds=9,
        embedding_model="test-embedding",
        embedding_api_key="test-embedding-key",
        embedding_timeout_seconds=11,
    )

    chat_client = get_llm_client(
        settings=settings,
        streaming=False,
        temperature=0,
    )
    embedding_client = get_embedding_client(settings=settings)

    assert chat_client is chat_calls[0]
    assert chat_calls[0]["model"] == "test-chat"
    assert chat_calls[0]["api_key"].get_secret_value() == "test-chat-key"
    assert chat_calls[0]["base_url"] == "https://models.example.test/v1"
    assert chat_calls[0]["streaming"] is False
    assert chat_calls[0]["timeout"] == 9.0
    assert chat_calls[0]["temperature"] == 0

    assert embedding_client is embedding_calls[0]
    assert embedding_calls[0]["model"] == "test-embedding"
    assert embedding_calls[0]["api_key"].get_secret_value() == "test-embedding-key"
    assert embedding_calls[0]["timeout"] == 11.0


def test_tracing_can_be_disabled_without_registering_phoenix(monkeypatch):
    def fail_if_registered(**kwargs):
        raise AssertionError(f"Phoenix should be disabled, received {kwargs}")

    monkeypatch.setattr("app.observability.register", fail_if_registered)

    settings = AppSettings(_env_file=None, phoenix_enabled=False)

    assert setup_phoenix_tracing(settings=settings) is None
