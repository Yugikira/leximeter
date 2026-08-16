"""Provider-agnostic orchestration: concurrency, retries, checkpointing.

Depends only on ``LLMClient.complete`` — never on a specific vendor.
"""

from .batch import batch_execute
from .retry import run_task_with_retry

__all__ = ["batch_execute", "run_task_with_retry"]
