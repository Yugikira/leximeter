"""Shared, provider-agnostic helpers used by task implementations.

Ported from GABRIEL's ``_run_utils`` / ``_attribute_utils``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")

ATTRIBUTE_WARNING_THRESHOLD = 12


def hash_identifier(value: Any) -> str:
    """Return a stable 16-char hex identifier for a piece of content."""
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]


def run_metadata_path(save_dir: str, base_name: str) -> str:
    return os.path.join(save_dir, f"{base_name}_run_metadata.json")


def load_run_metadata(save_dir: str, base_name: str, *, reset_files: bool) -> dict[str, Any]:
    path = run_metadata_path(save_dir, base_name)
    if reset_files:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def update_run_metadata(save_dir: str, base_name: str, **updates: Any) -> None:
    path = run_metadata_path(save_dir, base_name)
    payload: dict[str, Any] = {}
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                payload.update(loaded)
    except (OSError, ValueError):
        payload = {}
    payload.update(updates)
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except (OSError, TypeError):
        pass


def load_json_cache(path: str) -> dict[str, Any] | None:
    """Read a JSON object checkpoint; ``None`` when missing or malformed."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_json_cache(path: str, payload: dict[str, Any]) -> None:
    """Best-effort JSON checkpoint write (never raises)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except (OSError, TypeError):
        pass


def load_persisted_attributes(
    save_dir: str,
    incoming: dict[str, Any],
    *,
    reset_files: bool,
    task_name: str,
    item_name: str = "attributes",
    legacy_filename: str | None = None,
) -> dict[str, Any]:
    """Load attributes/labels from disk for reproducibility.

    Preference order:
    1) ``attributes.json`` in ``save_dir``
    2) ``legacy_filename`` (e.g., ``ratings_attrs.json``) in ``save_dir``
    When neither exists, ``incoming`` is written to both paths (when
    applicable) for future runs.
    """
    primary_path = os.path.join(save_dir, "attributes.json")
    legacy_path = os.path.join(save_dir, legacy_filename) if legacy_filename else None
    candidate_paths = [primary_path] + ([legacy_path] if legacy_path else [])

    if reset_files:
        for path in candidate_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    loaded: dict[str, Any] | None = None
    source_path: str | None = None
    for path in candidate_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            source_path = path
            break
        except (OSError, ValueError):
            continue

    if loaded is not None:
        message = (
            f"[{task_name}] Found saved {item_name} in {source_path}. "
            "Using them for consistency."
        )
        if loaded != incoming:
            message += (
                " The provided attributes differ; set reset_files=True or "
                "use a new save_dir to update them."
            )
        print(message)
        return loaded

    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError:
        pass
    for path in candidate_paths:
        if not path:
            continue
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(incoming, f, indent=2)
        except (OSError, TypeError):
            pass

    return incoming


def chunk_items(items: Sequence[T], n: int | None) -> list[list[T]]:
    if not items:
        return []
    if n is None:
        return [list(items)]
    return [list(items[i : i + n]) for i in range(0, len(items), n)]


def resolve_attribute_batches(
    items: Sequence[tuple[str, T]],
    *,
    requested_n: int | None,
    metadata: dict[str, Any],
    reset_files: bool,
) -> tuple[list[list[tuple[str, T]]], int | None]:
    """Split attribute items into prompt batches, reusing any saved plan.

    ``requested_n`` is the max attributes per prompt (``None`` = one batch).
    """
    if requested_n is not None and (not isinstance(requested_n, int) or requested_n < 1):
        raise ValueError("n_attributes_per_run must be None or an integer >= 1")
    if not items:
        return [], requested_n

    # Reuse a previously-persisted batch plan so resuming lines up with cache.
    if not reset_files and isinstance(metadata.get("attribute_batches"), list):
        lookup = {name: value for name, value in items}
        batches: list[list[tuple[str, T]]] = []
        valid = True
        for batch in metadata["attribute_batches"]:
            if not isinstance(batch, list):
                valid = False
                break
            if any(name not in lookup for name in batch):
                valid = False
                break
            batches.append([(name, lookup[name]) for name in batch])
        if valid and [name for name, _ in items] == [
            name for batch in batches for name, _ in batch
        ]:
            return batches, metadata.get("n_attributes_per_run")

    if requested_n is None and len(items) > ATTRIBUTE_WARNING_THRESHOLD:
        print(
            f"Processing {len(items)} attributes in one prompt. "
            "Set n_attributes_per_run=12 (or lower) for more consistent outputs."
        )

    return chunk_items(items, requested_n), requested_n


def _coerce_json_obj(raw: Any) -> Any:
    """Coerce ``raw`` to a JSON object, unwrapping single-element lists/strings.

    ``safe_json`` alone is not enough: a response may be a one-element list
    wrapping the real object, or a string that itself contains JSON.
    """

    from ..utils.parsing import safe_json

    obj = safe_json(raw)
    if isinstance(obj, list) and len(obj) == 1:
        obj = safe_json(obj[0])
    if isinstance(obj, str):
        obj = safe_json(obj)
    return obj


def _collect_attribute_keys(obj: Any, lookup: dict[str, str]) -> set[str]:
    """Recursively collect attribute keys (case-insensitive) present in ``obj``.

    ``lookup`` maps lower-cased attribute names to their canonical names.
    """

    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            norm = str(key).strip().lower()
            if norm in lookup:
                found.add(lookup[norm])
            if isinstance(value, (dict, list)):
                found.update(_collect_attribute_keys(value, lookup))
    elif isinstance(obj, list):
        for item in obj:
            found.update(_collect_attribute_keys(item, lookup))
    return found
