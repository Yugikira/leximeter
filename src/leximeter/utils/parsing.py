"""JSON parsing helpers for model output."""

from __future__ import annotations

import json
from typing import Any, Optional


def safe_json(text: Any) -> Any:
    """Parse ``text`` as JSON, tolerating markdown fences and whitespace.

    Returns ``None`` on failure (or if ``text`` is empty / already non-string).
    """
    if text is None or not isinstance(text, str):
        return text

    cleaned = text.strip()
    if not cleaned:
        return None

    # Strip a surrounding ```json ... ``` fence if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1].strip()
        # Drop a trailing fence line if any.
        cleaned = cleaned.rsplit("\n```", 1)[0].strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract the first balanced {...} or [...] substring.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == opener:
                depth += 1
            elif cleaned[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
    return None
