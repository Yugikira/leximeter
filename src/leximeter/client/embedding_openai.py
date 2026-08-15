"""OpenAI embedding provider."""

from __future__ import annotations

import asyncio
import os

from .embedding import EmbeddingClient

# ``text-embedding-3-*`` models cap a single request at 8191 input tokens;
# stay under that with headroom for the cheap token estimate below.
DEFAULT_MAX_TOKENS_PER_BATCH = 4096
DEFAULT_MAX_ITEMS_PER_BATCH = 2048
DEFAULT_MAX_CONCURRENCY = 16


def _estimate_tokens(text: str) -> int:
    """Token estimate without a tokenizer dependency.

    ASCII is roughly 4 chars per token; every other character (CJK, accented
    Latin, etc.) is conservatively counted as a full token.
    """
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return (ascii_chars + 3) // 4 + (len(text) - ascii_chars)


def _split_batches(
    texts: list[str], *, max_tokens: int, max_items: int
) -> list[list[str]]:
    """Split ``texts`` into contiguous batches under the request limits.

    A single text that already exceeds ``max_tokens`` is emitted alone so
    oversized inputs still reach the API (and fail there with a clear error)
    instead of being dropped.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        tokens = _estimate_tokens(text)
        if current and (
            len(current) >= max_items or current_tokens + tokens > max_tokens
        ):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


class OpenAIEmbeddingClient(EmbeddingClient):
    """Embed texts via the OpenAI embeddings endpoint (or any compatible base_url).

    Large inputs are split into bounded batches (see ``_split_batches``) that
    run concurrently and are reassembled in input order, so a whole-list call
    can never exceed the single-request token/item caps.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH,
        max_items_per_batch: int = DEFAULT_MAX_ITEMS_PER_BATCH,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        from openai import AsyncOpenAI

        self._model_name = model or "text-embedding-3-small"
        self._max_tokens_per_batch = max_tokens_per_batch
        self._max_items_per_batch = max_items_per_batch
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = AsyncOpenAI(
            api_key=api_key
            or os.getenv("LEXIMETER_EMBEDDING_API_KEY")
            or os.getenv("LEXIMETER_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
            or os.getenv("LEXIMETER_EMBEDDING_BASE_URL")
            or os.getenv("LEXIMETER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL"),
            max_retries=3,  # matches GABRIEL's get_all_embeddings max_retries
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batches = _split_batches(
            texts,
            max_tokens=self._max_tokens_per_batch,
            max_items=self._max_items_per_batch,
        )

        async def _embed_batch(batch: list[str]) -> list[list[float]]:
            async with self._semaphore:
                resp = await self._client.embeddings.create(
                    model=self._model_name, input=batch
                )
            # ``index`` is the explicit positional contract; do not assume the
            # provider echoes the request order.
            return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

        results = await asyncio.gather(*(_embed_batch(b) for b in batches))
        return [vec for batch in results for vec in batch]
