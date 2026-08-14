"""Provider-agnostic orchestration: concurrency, retries, checkpointing, cost.

This layer depends only on ``LLMClient.complete`` — never on a specific vendor.
Implemented in milestone M2.
"""
