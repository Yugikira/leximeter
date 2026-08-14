"""Leximeter: text-only, provider-neutral LLM measurement library.

Forked from openai/GABRIEL and refactored for maintainability. See PLAN.md.
"""

from .api import (
    classify,
    compare,
    deidentify,
    extract,
    filter,
    paraphrase,
    rate,
    seed,
    whatever,
)

__version__ = "0.1.0"

__all__ = [
    "rate",
    "classify",
    "extract",
    "filter",
    "compare",
    "paraphrase",
    "deidentify",
    "seed",
    "whatever",
]
