"""Local CPU embedding provider using sentence-transformers (e.g. BGE)."""

from __future__ import annotations

import asyncio

from .embedding import EmbeddingClient


class LocalBgeClient(EmbeddingClient):
    """Run a local sentence-transformers model (default ``BAAI/bge-m3``) on CPU.

    Requires the optional ``[embeddings]`` extra (``sentence-transformers``).
    Loading happens lazily on first use; encoding runs in a worker thread so it
    does not block the event loop.
    """

    def __init__(self, *, model: str | None = None, device: str = "cpu") -> None:
        self._model_name = model or "BAAI/bge-m3"
        self._device = device
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        vecs = await asyncio.to_thread(
            self._model.encode, texts, normalize_embeddings=True
        )
        return vecs.tolist()
