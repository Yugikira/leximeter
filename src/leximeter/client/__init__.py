"""Provider layer: LLM clients, embeddings, and search tools.

All providers implement a small protocol (``complete`` / ``embed`` /
``search``); the rest of the library depends only on those protocols, never on
a specific vendor.
"""

from .base import LLMClient
from .chat import ChatCompletionsClient
from .dummy import DummyClient
from .embedding import DummyEmbeddingClient, EmbeddingClient
from .responses import ResponsesClient
from .search import DummySearchTool, SearchTool

__all__ = [
    "ChatCompletionsClient",
    "DummyClient",
    "DummyEmbeddingClient",
    "DummySearchTool",
    "EmbeddingClient",
    "LLMClient",
    "ResponsesClient",
    "SearchTool",
]
