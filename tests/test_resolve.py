"""Provider resolution from env defaults: embedding base_url/api_key isolation."""

from __future__ import annotations

import pytest

from leximeter.client import (
    DummyDecisionClient,
    DummyEmbeddingClient,
    SystemOneClient,
)
from leximeter.client.embedding_local import LocalBgeClient
from leximeter.client.embedding_openai import OpenAIEmbeddingClient
from leximeter.client.resolve import (
    resolve_decision_client,
    resolve_decision_model,
    resolve_embedding_client,
)

LLM_URL = "https://llm.example.com/v1"
EMB_URL = "https://embedding.example.com/v1"


def test_resolve_embedding_openai_uses_embedding_specific_env(monkeypatch):
    """LLM and embedding endpoints are independently configurable."""
    monkeypatch.setenv("LEXIMETER_BASE_URL", LLM_URL)
    monkeypatch.setenv("LEXIMETER_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("LEXIMETER_EMBEDDING_BASE_URL", EMB_URL)
    monkeypatch.setenv("LEXIMETER_EMBEDDING_API_KEY", "sk-embedding")
    client = resolve_embedding_client()
    assert isinstance(client, OpenAIEmbeddingClient)
    assert str(client._client.base_url).startswith(EMB_URL)


def test_resolve_embedding_openai_falls_back_to_llm_base_url(monkeypatch):
    """Without embedding-specific env, the LLM endpoint is used."""
    monkeypatch.setenv("LEXIMETER_BASE_URL", LLM_URL)
    monkeypatch.setenv("LEXIMETER_API_KEY", "sk-llm")
    monkeypatch.setenv("LEXIMETER_EMBEDDING_PROVIDER", "openai")
    # Project .env (loaded at import) may set embedding-specific vars; drop
    # them so this test genuinely exercises the LLM fallback.
    monkeypatch.delenv("LEXIMETER_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("LEXIMETER_EMBEDDING_API_KEY", raising=False)
    client = resolve_embedding_client()
    assert isinstance(client, OpenAIEmbeddingClient)
    assert str(client._client.base_url).startswith(LLM_URL)


def test_resolve_embedding_model_env_honored(monkeypatch):
    monkeypatch.setenv("LEXIMETER_API_KEY", "sk-test")
    monkeypatch.setenv("LEXIMETER_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("LEXIMETER_EMBEDDING_MODEL", "custom-embed-model")
    client = resolve_embedding_client()
    assert client._model_name == "custom-embed-model"


def test_resolve_embedding_explicit_base_url_wins(monkeypatch):
    monkeypatch.setenv("LEXIMETER_API_KEY", "sk-test")
    monkeypatch.setenv("LEXIMETER_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("LEXIMETER_EMBEDDING_BASE_URL", EMB_URL)
    client = resolve_embedding_client(base_url="https://override.example.com/v1")
    assert str(client._client.base_url).startswith("https://override.example.com/v1")


def test_resolve_embedding_none_returns_none(monkeypatch):
    """Provider "none" disables embeddings (callers skip, not fail)."""
    monkeypatch.setenv("LEXIMETER_EMBEDDING_PROVIDER", "none")
    assert resolve_embedding_client() is None


def test_resolve_embedding_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LEXIMETER_EMBEDDING_PROVIDER", "weird")
    with pytest.raises(ValueError):
        resolve_embedding_client()


def test_resolve_embedding_defaults_to_local(monkeypatch):
    """No env set: local BGE provider, model never loaded at construction."""
    monkeypatch.delenv("LEXIMETER_EMBEDDING_PROVIDER", raising=False)
    client = resolve_embedding_client()
    assert isinstance(client, LocalBgeClient)
    assert client._model is None  # lazy load


def test_resolve_embedding_injected_client_wins(monkeypatch):
    monkeypatch.setenv("LEXIMETER_EMBEDDING_PROVIDER", "openai")
    dummy = DummyEmbeddingClient()
    assert resolve_embedding_client(embedding_client=dummy) is dummy


# ---------------------------------------------------------------------------
# Decision Provider resolution (disabled by default)
# ---------------------------------------------------------------------------


def test_resolve_decision_defaults_to_none(monkeypatch):
    """Without LEXIMETER_DECISION_PROTOCOL, Decision Provider is disabled."""
    monkeypatch.delenv("LEXIMETER_DECISION_PROTOCOL", raising=False)
    assert resolve_decision_client() is None


def test_resolve_decision_none_returns_none(monkeypatch):
    monkeypatch.setenv("LEXIMETER_DECISION_PROTOCOL", "none")
    assert resolve_decision_client() is None


def test_resolve_decision_systemone_from_env(monkeypatch):
    monkeypatch.setenv("LEXIMETER_DECISION_PROTOCOL", "systemone")
    monkeypatch.setenv("LEXIMETER_DECISION_BASE_URL", "https://openrouter.ai/api")
    monkeypatch.setenv("LEXIMETER_DECISION_API_KEY", "or-test")
    client = resolve_decision_client()
    assert isinstance(client, SystemOneClient)
    assert client.url == "https://openrouter.ai/api/v1/systemone"


def test_resolve_decision_dummy_protocol(monkeypatch):
    monkeypatch.setenv("LEXIMETER_DECISION_PROTOCOL", "dummy")
    client = resolve_decision_client()
    assert isinstance(client, DummyDecisionClient)


def test_resolve_decision_injected_client_wins(monkeypatch):
    monkeypatch.setenv("LEXIMETER_DECISION_PROTOCOL", "systemone")
    dummy = DummyDecisionClient()
    assert resolve_decision_client(decision_client=dummy) is dummy


def test_resolve_decision_unknown_protocol_raises(monkeypatch):
    monkeypatch.setenv("LEXIMETER_DECISION_PROTOCOL", "unknown")
    with pytest.raises(ValueError, match="LEXIMETER_DECISION_PROTOCOL"):
        resolve_decision_client()


def test_resolve_decision_model_precedence(monkeypatch):
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)
    assert resolve_decision_model(fallback="gpt-5.4-mini") == "gpt-5.4-mini"
    monkeypatch.setenv("LEXIMETER_DECISION_MODEL", "cloudflare/clef-flash")
    assert resolve_decision_model(fallback="gpt-5.4-mini") == "cloudflare/clef-flash"
    assert (
        resolve_decision_model("typesafe/jev-1.14", fallback="gpt-5.4-mini")
        == "typesafe/jev-1.14"
    )
