"""Leximeter: text-only, provider-neutral LLM measurement library.

Forked from openai/GABRIEL and refactored for maintainability. See PLAN.md.
"""

from .api import (
    bucket,
    classify,
    codify,
    compare,
    deduplicate,
    deidentify,
    extract,
    filter,
    merge,
    paraphrase,
    rank,
    rate,
    seed,
    whatever,
)
from .load import load

__version__ = "0.1.0"

__all__ = [
    "rate",
    "rank",
    "classify",
    "extract",
    "filter",
    "compare",
    "paraphrase",
    "deidentify",
    "seed",
    "merge",
    "deduplicate",
    "bucket",
    "codify",
    "whatever",
    "load",
]
