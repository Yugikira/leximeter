"""Build providers from environment / config, with per-call override.

Environment variables (all optional)::

    LEXIMETER_PROTOCOL          "responses" (default) or "chat"
    LEXIMETER_BASE_URL          endpoint for the chosen provider
    LEXIMETER_API_KEY           API key (falls back to OPENAI_API_KEY)
    LEXIMETER_EMBEDDING_PROVIDER  "local" (default) or "openai"
    LEXIMETER_SEARCH_PROVIDER   "tavily" (default) or "none"
"""

from __future__ import annotations

import os
from typing import Optional

from .base import LLMClient
from .dummy import DummyClient
from .embedding import EmbeddingClient
from .search import SearchTool


def resolve_client(
    *,
    protocol: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Optional[LLMClient] = None,
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
    provider: Optional[str] = None,
    model: Optional[str] = None,
    embedding_client: Optional[EmbeddingClient] = None,
) -> EmbeddingClient:
    """Return an embedding client from an explicit instance or env defaults."""
    if embedding_client is not None:
        return embedding_client

    provider = (provider or os.getenv("LEXIMETER_EMBEDDING_PROVIDER") or "local").lower()
    if provider == "local":
        from .embedding_local import LocalBgeClient  # deferred to a later milestone

        return LocalBgeClient(model=model or os.getenv("LEXIMETER_EMBEDDING_MODEL"))
    if provider == "openai":
        from .embedding_openai import OpenAIEmbeddingClient  # deferred

        return OpenAIEmbeddingClient(model=model)
    raise ValueError(
        f"Unknown LEXIMETER_EMBEDDING_PROVIDER {provider!r}; expected 'local' or 'openai'."
    )


def resolve_search_tool(
    *,
    provider: Optional[str] = None,
    search_tool: Optional[SearchTool] = None,
) -> Optional[SearchTool]:
    """Return a search tool, or ``None`` when none is configured.

    ``None`` means "search is unavailable": callers should warn and skip the
    search step rather than failing.
    """
    if search_tool is not None:
        return search_tool

    provider = (provider or os.getenv("LEXIMETER_SEARCH_PROVIDER") or "").lower()
    if provider in ("", "none"):
        return None
    if provider == "tavily":
        from .search_tavily import TavilySearchTool  # deferred to a later milestone

        return TavilySearchTool()
    raise ValueError(
        f"Unknown LEXIMETER_SEARCH_PROVIDER {provider!r}; expected 'tavily' or 'none'."
    )


__all__ = ["resolve_client", "resolve_embedding_client", "resolve_search_tool"]
