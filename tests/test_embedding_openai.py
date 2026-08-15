"""OpenAI embedding client: batching behavior, with a fake SDK (no network)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from leximeter.client.embedding_openai import (
    DEFAULT_MAX_ITEMS_PER_BATCH,
    OpenAIEmbeddingClient,
    _estimate_tokens,
    _split_batches,
)


class FakeEmbeddings:
    """Fake ``embeddings.create``: records inputs, returns length-encoded vectors."""

    def __init__(self, shuffle_order: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.shuffle_order = shuffle_order

    async def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append(list(input))
        items = [
            SimpleNamespace(index=i, embedding=[float(len(t)), 0.0, 1.0])
            for i, t in enumerate(input)
        ]
        if self.shuffle_order:
            items.reverse()
        return SimpleNamespace(data=items)


@pytest.fixture
def client(monkeypatch) -> tuple[OpenAIEmbeddingClient, FakeEmbeddings]:
    cli = OpenAIEmbeddingClient(api_key="test", base_url="https://example.invalid")
    fake = FakeEmbeddings()
    monkeypatch.setattr(cli, "_client", SimpleNamespace(embeddings=fake))
    return cli, fake


def test_estimate_tokens() -> None:
    assert _estimate_tokens("hello") == 2  # ceil(5 / 4)
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("中") == 1  # non-ASCII counts as a full token
    assert _estimate_tokens("中国银行") == 4


def test_split_batches_honors_token_budget() -> None:
    long = "a" * 100  # ~25 tokens
    assert _split_batches([long] * 3, max_tokens=60, max_items=100) == [
        [long, long],
        [long],
    ]


def test_split_batches_honors_item_cap() -> None:
    assert _split_batches(["x"] * 5, max_tokens=10_000, max_items=2) == [
        ["x", "x"],
        ["x", "x"],
        ["x"],
    ]


def test_split_batches_oversized_text_goes_alone() -> None:
    huge = "中" * 5000  # ~5000 tokens, over any small budget
    assert _split_batches(["a", huge, "b"], max_tokens=100, max_items=100) == [
        ["a"],
        [huge],
        ["b"],
    ]


def test_embed_batches_and_preserves_order(client) -> None:
    cli, fake = client
    texts = [f"t{i}" for i in range(5000)]
    out = asyncio.run(cli.embed(texts))
    assert len(out) == 5000
    assert len(fake.calls) == 3  # 2048 + 2048 + 904
    assert all(len(call) <= DEFAULT_MAX_ITEMS_PER_BATCH for call in fake.calls)
    assert out == [[float(len(t)), 0.0, 1.0] for t in texts]


def test_embed_reorders_shuffled_response(client) -> None:
    cli, _ = client
    fake = FakeEmbeddings(shuffle_order=True)
    cli._client.embeddings = fake
    texts = [f"t{i}" for i in range(3000)]
    out = asyncio.run(cli.embed(texts))
    assert out == [[float(len(t)), 0.0, 1.0] for t in texts]


def test_embed_empty_list_makes_no_request(client) -> None:
    cli, fake = client
    assert asyncio.run(cli.embed([])) == []
    assert fake.calls == []


def test_embed_tiny_budget_splits_by_tokens(client) -> None:
    cli, fake = client
    cli._max_tokens_per_batch = 100
    cli._max_items_per_batch = 10_000
    long = "中" * 200  # 200 tokens
    out = asyncio.run(cli.embed([long] * 3))
    assert len(fake.calls) == 3  # each alone under the 100-token budget
    assert out == [[float(len(t)), 0.0, 1.0] for t in [long] * 3]
