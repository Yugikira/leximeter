"""Build providers from environment / config, with per-call override.

Environment variables (all optional)::

    LEXIMETER_PROTOCOL          "responses" (default) or "chat"
    LEXIMETER_BASE_URL          endpoint for the chosen provider
    LEXIMETER_API_KEY           API key (falls back to OPENAI_API_KEY)
    LEXIMETER_EMBEDDING_PROVIDER  "local" (default), "openai", or "none"
    LEXIMETER_EMBEDDING_MODEL   embedding model (bge-m3 default for local;
                                text-embedding-3-small default for openai)
    LEXIMETER_EMBEDDING_BASE_URL  embedding endpoint (openai only; defaults
                                to LEXIMETER_BASE_URL / OPENAI_BASE_URL)
    LEXIMETER_EMBEDDING_API_KEY   embedding key (openai only; defaults to
                                LEXIMETER_API_KEY / OPENAI_API_KEY)
    LEXIMETER_SEARCH_PROVIDER   "tavily" (default) or "none"
    LEXIMETER_DECISION_PROTOCOL "systemone", "dummy", or "none" (default: disabled)
    LEXIMETER_DECISION_BASE_URL endpoint base URL for the Decision Provider
    LEXIMETER_DECISION_API_KEY  API key for the Decision Provider
    LEXIMETER_DECISION_MODEL    default model ID for the Decision Provider
"""

from __future__ import annotations

import os
from typing import Any

from .base import LLMClient
from .decision import DecisionClient
from .embedding import EmbeddingClient
from .search import SearchTool


def resolve_client(
    *,
    protocol: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    client: LLMClient | None = None,
) -> LLMClient:
    """Return an LLM client from an explicit ``client`` or environment defaults."""
    if client is not None:
        return client

    protocol = (protocol or os.getenv("LEXIMETER_PROTOCOL") or "responses").lower()
    if protocol == "responses":
        from .responses import ResponsesClient

        return ResponsesClient(base_url=base_url, api_key=api_key)
    if protocol == "chat":
        from .chat import ChatCompletionsClient

        return ChatCompletionsClient(base_url=base_url, api_key=api_key)
    raise ValueError(
        f"Unknown LEXIMETER_PROTOCOL {protocol!r}; expected 'responses' or 'chat'."
    )


def resolve_embedding_client(
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> EmbeddingClient | None:
    """Return an embedding client from an explicit instance or env defaults.

    ``None`` (provider "none") means "embeddings are disabled": callers
    should skip the embedding step rather than fail — matching
    ``resolve_search_tool``.
    """
    if embedding_client is not None:
        return embedding_client

    provider = (
        provider or os.getenv("LEXIMETER_EMBEDDING_PROVIDER") or "local"
    ).lower()
    if provider in ("", "none"):
        return None
    if provider == "local":
        from .embedding_local import LocalBgeClient  # deferred to a later milestone

        return LocalBgeClient(model=model or os.getenv("LEXIMETER_EMBEDDING_MODEL"))
    if provider == "openai":
        from .embedding_openai import OpenAIEmbeddingClient  # deferred

        return OpenAIEmbeddingClient(
            model=model or os.getenv("LEXIMETER_EMBEDDING_MODEL"),
            base_url=base_url or os.getenv("LEXIMETER_EMBEDDING_BASE_URL"),
            api_key=api_key or os.getenv("LEXIMETER_EMBEDDING_API_KEY"),
        )
    raise ValueError(
        f"Unknown LEXIMETER_EMBEDDING_PROVIDER {provider!r}; "
        "expected 'local', 'openai', or 'none'."
    )


def resolve_search_tool(
    *,
    provider: str | None = None,
    search_tool: SearchTool | None = None,
    search_context_size: str = "medium",
    web_search_filters: dict[str, Any] | None = None,
) -> SearchTool | None:
    """Return a search tool, or ``None`` when none is configured.

    ``None`` means "search is unavailable": callers should warn and skip the
    search step rather than failing.  ``search_context_size`` /
    ``web_search_filters`` (GABRIEL's web-search knobs) are forwarded to the
    provider that supports them — matching ``resolve_embedding_client``'s
    ``model`` forwarding.
    """
    if search_tool is not None:
        return search_tool

    provider = (provider or os.getenv("LEXIMETER_SEARCH_PROVIDER") or "").lower()
    if provider in ("", "none"):
        return None
    if provider == "tavily":
        from .search_tavily import TavilySearchTool

        return TavilySearchTool(
            search_context_size=search_context_size,
            web_search_filters=web_search_filters,
        )
    raise ValueError(
        f"Unknown LEXIMETER_SEARCH_PROVIDER {provider!r}; expected 'tavily' or 'none'."
    )


def resolve_decision_client(
    *,
    protocol: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    endpoint_path: str | None = None,
    send_noul_criteria: bool | None = None,
    decision_client: DecisionClient | None = None,
) -> DecisionClient | None:
    """Return a Decision Provider client, or ``None`` when disabled (default).

    ``None`` means tasks fall back cleanly to their generative ``LLMClient``
    path (ADR-0002, ADR-0004).
    """
    if decision_client is not None:
        return decision_client

    proto = (protocol or os.getenv("LEXIMETER_DECISION_PROTOCOL") or "").strip().lower()
    if proto in ("", "none"):
        return None
    if proto == "dummy":
        from .decision import DummyDecisionClient

        return DummyDecisionClient()
    if proto == "systemone":
        from .systemone import SystemOneClient

        kwargs: dict[str, Any] = {"base_url": base_url, "api_key": api_key}
        if endpoint_path is not None:
            kwargs["endpoint_path"] = endpoint_path
        if send_noul_criteria is not None:
            kwargs["send_noul_criteria"] = send_noul_criteria
        return SystemOneClient(**kwargs)
    raise ValueError(
        f"Unknown LEXIMETER_DECISION_PROTOCOL {proto!r}; "
        "expected 'systemone', 'dummy', or 'none'."
    )


def _clean_model_id(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def resolve_decision_model(
    decision_model: str | None = None,
    *,
    fallback: str | None = None,
) -> str | None:
    """Resolve the Decision Provider model ID (explicit > env > fallback)."""
    return (
        _clean_model_id(decision_model)
        or _clean_model_id(os.getenv("LEXIMETER_DECISION_MODEL"))
        or _clean_model_id(fallback)
    )


__all__ = [
    "resolve_client",
    "resolve_decision_client",
    "resolve_decision_model",
    "resolve_embedding_client",
    "resolve_search_tool",
]
