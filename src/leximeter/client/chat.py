"""Traditional Chat Completions provider.

Speaks the OpenAI-compatible ``/v1/chat/completions`` protocol, which most
Chinese and open-source backends implement (DeepSeek, MiniMax, Ollama, vLLM,
gateways).  Point ``base_url`` at any such endpoint.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import LLMClient


class ChatCompletionsClient(LLMClient):
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
        params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        params.update(generation_kwargs)
        response = await self._client.chat.completions.create(**params)
        return response.choices[0].message.content or ""

    def estimate_tokens(self, prompt: str, *, model: str) -> int:
        raise NotImplementedError("token estimation is reserved for a later version")

    def estimate_cost(self, prompt: str, *, model: str) -> float:
        raise NotImplementedError("cost estimation is reserved for a later version")
