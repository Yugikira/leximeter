"""Jinja2 filters for randomized prompt rendering."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional


def shuffled(it: Iterable[Any], seed: Optional[int] = None) -> List[Any]:
    """Return a new list with the same elements, shuffled."""
    seq = list(it)
    rnd = random.Random(seed) if seed is not None else random
    rnd.shuffle(seq)
    return seq


def shuffled_dict(d: Dict[str, Any], seed: Optional[int] = None) -> str:
    """Return a JSON-formatted dict string with items shuffled."""
    items = list(d.items())
    rnd = random.Random(seed) if seed is not None else random
    rnd.shuffle(items)
    return json.dumps(OrderedDict(items), ensure_ascii=False, indent=2)
