"""The single-completion protocol every LLM provider implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Minimal interface for language-model providers.

    The rest of the library depends only on :meth:`complete`; each provider is
    free to translate the call into whatever protocol it speaks (Responses API,
    Chat Completions, ...).
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = False,
        **generation_kwargs: Any,
    ) -> str:
        """Return the model's text completion for ``prompt``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Reserved: provider-specific usage / pricing.  Not implemented in v1.
    # ------------------------------------------------------------------
    @abstractmethod
    def estimate_tokens(self, prompt: str, *, model: str) -> int:
        """Estimate the number of tokens ``prompt`` would consume."""
        raise NotImplementedError

    @abstractmethod
    def estimate_cost(self, prompt: str, *, model: str) -> float:
        """Estimate the monetary cost of completing ``prompt``."""
        raise NotImplementedError
