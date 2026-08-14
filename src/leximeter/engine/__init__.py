"""Provider-agnostic orchestration: concurrency, retries, checkpointing.

Depends only on ``LLMClient.complete`` — never on a specific vendor.
"""

from .batch import batch_execute

__all__ = ["batch_execute"]
