"""Shared fake providers for the test suite."""

from __future__ import annotations

import json
from typing import Any, Dict

from leximeter.client.base import LLMClient


class DictClient(LLMClient):
    """Return a fixed JSON payload for every prompt."""

    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
        return json.dumps(self._payload)

    def estimate_tokens(self, prompt, *, model) -> int:
        return 0

    def estimate_cost(self, prompt, *, model) -> float:
        return 0.0


class CountingClient(DictClient):
    """DictClient that also counts how many completions were actually made."""

    def __init__(self, payload: Dict[str, Any]):
        super().__init__(payload)
        self.calls = 0

    async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
        self.calls += 1
        return await super().complete(prompt, model=model, json_mode=json_mode, **kwargs)
