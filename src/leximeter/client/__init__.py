"""Provider layer: LLM clients, decision providers, embeddings, and search tools.

All providers implement a small protocol (``complete`` / ``decide`` / ``embed``
/ ``search``); the rest of the library depends only on those protocols, never
on a specific vendor.
"""

from .base import LLMClient
from .chat import ChatCompletionsClient
from .decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionAPIError,
    DecisionClient,
    DecisionQuestion,
    DecisionResult,
    DummyDecisionClient,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from .dummy import DummyClient
from .embedding import DummyEmbeddingClient, EmbeddingClient
from .responses import ResponsesClient
from .search import DummySearchTool, SearchTool
from .systemone import SystemOneClient

__all__ = [
    "ChatCompletionsClient",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecisionAPIError",
    "DecisionAnswer",
    "DecisionClient",
    "DecisionQuestion",
    "DecisionResult",
    "DummyClient",
    "DummyDecisionClient",
    "DummyEmbeddingClient",
    "DummySearchTool",
    "EmbeddingClient",
    "LLMClient",
    "NoulAnswer",
    "NoulQuestion",
    "ResponsesClient",
    "ScoreAnswer",
    "ScoreQuestion",
    "SearchTool",
    "SystemOneClient",
]
