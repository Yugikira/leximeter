"""Web-search provider abstraction (separate from the LLM client)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SearchTool(ABC):
    """Minimal interface for retrieval tools (e.g. Tavily, Exa).

    A search tool augments a prompt with external context *before* the model
    call.  It is optional: if none is configured for the active protocol, the
    library warns and skips the search step rather than failing.
    """

    @abstractmethod
    async def search(self, query: str) -> str:
        """Return retrieved context text for ``query`` (empty if none found)."""
        raise NotImplementedError


class DummySearchTool(SearchTool):
    """No-op search tool for tests and dry runs."""

    async def search(self, query: str) -> str:
        return ""
