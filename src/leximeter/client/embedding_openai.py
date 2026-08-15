"""OpenAI embedding provider."""

from __future__ import annotations

import os
from typing import List, Optional

from .embedding import EmbeddingClient


class OpenAIEmbeddingClient(EmbeddingClient):
    """Embed texts via the OpenAI embeddings endpoint (or any compatible base_url)."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        from openai import AsyncOpenAI

        self._model_name = model or "text-embedding-3-small"
        self._client = AsyncOpenAI(
            api_key=api_key
            or os.getenv("LEXIMETER_EMBEDDING_API_KEY")
            or os.getenv("LEXIMETER_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
            or os.getenv("LEXIMETER_EMBEDDING_BASE_URL")
            or os.getenv("LEXIMETER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL"),
        )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        resp = await self._client.embeddings.create(model=self._model_name, input=texts)
        return [d.embedding for d in resp.data]
