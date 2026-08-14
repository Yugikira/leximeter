"""Tavily web-search provider (needs the ``search`` extra + ``TAVILY_API_KEY``).

The constructor mirrors GABRIEL's ``web_search`` call surface — a
``search_context_size`` (``low``/``medium``/``high``) and a ``web_search_filters``
dict (``allowed_domains`` plus optional location hints) — and maps them onto the
Tavily API.  Unlike the OpenAI Responses API native ``web_search`` tool, which
runs server-side and returns a model-grounded answer, Tavily is a standalone
search call: :meth:`search` returns prompt-ready context text (Tavily's AI
``answer`` summary first, then one bullet per result) that the caller injects
into the prompt before the model call.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .search import SearchTool

logger = logging.getLogger(__name__)

# GABRIEL search_context_size (low/medium/high) -> Tavily search_depth.
_CONTEXT_TO_DEPTH = {"low": "fast", "medium": "basic", "high": "advanced"}
_CONTEXT_ALIASES = {"small": "low", "large": "high"}
_VALID_CONTEXT = set(_CONTEXT_TO_DEPTH)


def _normalise_search_context_size(value: Optional[str]) -> str:
    """Clamp search-context size to GABRIEL's low/medium/high vocabulary."""
    if value is None:
        return "medium"
    normalised = str(value).strip().lower()
    resolved = _CONTEXT_ALIASES.get(normalised, normalised)
    if resolved not in _VALID_CONTEXT:
        raise ValueError(
            "search_context_size must be one of low/medium/high "
            f"(got {value!r})"
        )
    return resolved


def _clean_domains(value: Any) -> Optional[list]:
    """Normalise ``allowed_domains`` to a list of non-empty strings."""
    if not value:
        return None
    if isinstance(value, (str, bytes)):
        domains = [s.strip() for s in str(value).split(",") if s.strip()]
    else:
        try:
            domains = [str(d).strip() for d in value if str(d).strip()]
        except TypeError:
            return None
    return domains or None


def _clean_country(filters: Dict[str, Any]) -> Optional[str]:
    """Map a location hint onto Tavily's ``country`` (its only location param)."""
    for key in ("country", "city", "region"):
        if filters.get(key):
            return str(filters[key]).strip()
    return None


class TavilySearchTool(SearchTool):
    """Retrieve context text via the Tavily API.

    Missing API key or missing optional dependency degrades gracefully:
    ``search`` returns ``""`` and logs a warning — callers are expected to
    skip the retrieval step, not fail.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        max_results: int = 5,
        search_context_size: str = "medium",
        web_search_filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._max_results = max_results
        context_size = _normalise_search_context_size(search_context_size)
        self._search_depth = _CONTEXT_TO_DEPTH[context_size]
        filters = web_search_filters or {}
        self._include_domains = _clean_domains(filters.get("allowed_domains"))
        self._country = _clean_country(filters)
        self._client: Any = None
        if not self._api_key:
            logger.warning(
                "TavilySearchTool is configured but TAVILY_API_KEY is not set; "
                "search will return empty context."
            )

    def _ensure_client(self) -> Any:
        if self._client is None and self._api_key:
            try:
                from tavily import AsyncTavilyClient  # type: ignore[import-not-found]

                self._client = AsyncTavilyClient(api_key=self._api_key)
            except ImportError:
                logger.warning(
                    "tavily-python is not installed; run "
                    "`pip install 'leximeter[search]'` to enable web search."
                )
        return self._client

    async def search(self, query: str) -> str:
        client = self._ensure_client()
        if client is None:
            return ""
        try:
            result = await client.search(
                query=query,
                max_results=self._max_results,
                search_depth=self._search_depth,
                include_domains=self._include_domains,
                country=self._country,
                include_answer=True,
            )
        except Exception as exc:  # auth / quota / network failures
            logger.warning("Tavily search failed: %s", exc)
            return ""
        lines = []
        answer = (result.get("answer") or "").strip()
        if answer:
            lines.append(answer)
        for item in result.get("results", []):
            title = (item.get("title") or "").strip()
            content = (item.get("content") or "").strip()
            url = (item.get("url") or "").strip()
            parts = [p for p in (title, content) if p]
            line = " — ".join(parts)
            if url:
                line = f"{line} ({url})"
            if line:
                lines.append(f"- {line}")
        return "\n".join(lines)


__all__ = ["TavilySearchTool"]
