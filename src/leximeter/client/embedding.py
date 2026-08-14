"""Embedding provider abstraction (separate from the LLM client)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingClient(ABC):
    """Minimal interface for text-embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        raise NotImplementedError


class DummyEmbeddingClient(EmbeddingClient):
    """Deterministic zero-vector provider for tests."""

    _dim: int = 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dim for _ in texts]
