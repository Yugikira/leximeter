"""OpenAI Responses API provider.

Also used for any backend that speaks the Responses protocol (e.g. DeepSeek's
recent Responses support) by pointing ``base_url`` at it.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import LLMClient


class ResponsesClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key
            or os.getenv("LEXIMETER_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
            or os.getenv("LEXIMETER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL"),
        )

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = False,
        **generation_kwargs: Any,
    ) -> str:
        params: dict[str, Any] = {"model": model, "input": prompt}
        if json_mode:
            params["text"] = {"format": {"type": "json_object"}}
        params.update(generation_kwargs)
        response = await self._client.responses.create(**params)
        return response.output_text

    def estimate_tokens(self, prompt: str, *, model: str) -> int:
        raise NotImplementedError("token estimation is reserved for a later version")

    def estimate_cost(self, prompt: str, *, model: str) -> float:
        raise NotImplementedError("cost estimation is reserved for a later version")
