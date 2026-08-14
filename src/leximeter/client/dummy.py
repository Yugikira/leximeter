"""Deterministic, no-network provider for tests and dry runs."""

from __future__ import annotations

from typing import Any

from .base import LLMClient


class DummyClient(LLMClient):
    """Return a fixed placeholder for every prompt.

    No network, no API key, no cost.  Used by the test suite and by users who
    want to dry-run a pipeline end-to-end before spending money.
    """

    async def complete(
        self,
        prompt: str,
        *,
        model: str = "dummy",
        json_mode: bool = False,
        **generation_kwargs: Any,
    ) -> str:
        if json_mode:
            return '{"dummy": true}'
        return f"DUMMY {prompt}"

    def estimate_tokens(self, prompt: str, *, model: str = "dummy") -> int:
        return 0

    def estimate_cost(self, prompt: str, *, model: str = "dummy") -> float:
        return 0.0
