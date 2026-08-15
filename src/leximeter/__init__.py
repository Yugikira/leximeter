"""Leximeter: text-only, provider-neutral LLM measurement library.

Forked from openai/GABRIEL and refactored for maintainability. See PLAN.md.
"""

from .config import load_env

# Read a local `.env` file (project root) into the environment before anything
# else runs, so provider defaults can be configured without shell exports.
load_env()

from .api import (
    bucket,
    classify,
    codify,
    compare,
    debias,
    deduplicate,
    deidentify,
    discover,
    extract,
    filter,
    ideate,
    merge,
    paraphrase,
    poll,
    rank,
    rate,
    seed,
    whatever,
)
from .load import load

__version__ = "0.1.0"

__all__ = [
    "bucket",
    "classify",
    "codify",
    "compare",
    "debias",
    "deduplicate",
    "deidentify",
    "discover",
    "extract",
    "filter",
    "ideate",
    "load",
    "merge",
    "paraphrase",
    "poll",
    "rank",
    "rate",
    "seed",
    "whatever",
]
