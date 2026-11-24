"""Jinja2 filters for randomized prompt rendering.

Ported and modified from openai/GABRIEL's ``utils/jinja.py`` (Apache-2.0).
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any


def shuffled(it: Iterable[Any], seed: int | None = None) -> list[Any]:
    """Return a new list with the same elements, shuffled."""
    seq = list(it)
    rnd = random.Random(seed) if seed is not None else random
    rnd.shuffle(seq)
    return seq


def shuffled_dict(d: dict[str, Any], seed: int | None = None) -> str:
    """Return a JSON-formatted dict string with items shuffled."""
    items = list(d.items())
    rnd = random.Random(seed) if seed is not None else random
    rnd.shuffle(items)
    return json.dumps(OrderedDict(items), ensure_ascii=False, indent=2)
