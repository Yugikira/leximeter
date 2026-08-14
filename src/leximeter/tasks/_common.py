"""Shared, provider-agnostic helpers used by task implementations.

Ported from GABRIEL's ``_run_utils`` / ``_attribute_utils`` and simplified for
a fresh fork (no legacy checkpoint compatibility to preserve).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")

ATTRIBUTE_WARNING_THRESHOLD = 12


def hash_identifier(value: Any) -> str:
    """Return a stable 16-char hex identifier for a piece of content."""
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]


def run_metadata_path(save_dir: str, base_name: str) -> str:
    return os.path.join(save_dir, f"{base_name}_run_metadata.json")


def load_run_metadata(save_dir: str, base_name: str, *, reset_files: bool) -> Dict[str, Any]:
    path = run_metadata_path(save_dir, base_name)
    if reset_files:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def update_run_metadata(save_dir: str, base_name: str, **updates: Any) -> None:
    path = run_metadata_path(save_dir, base_name)
    payload: Dict[str, Any] = {}
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                payload.update(loaded)
    except Exception:
        payload = {}
    payload.update(updates)
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def load_persisted_attributes(
    save_dir: str,
    incoming: Dict[str, Any],
    *,
    reset_files: bool,
    task_name: str,
    item_name: str = "attributes",
) -> Dict[str, Any]:
    """Load attributes from disk for reproducibility; write on first run."""
    path = os.path.join(save_dir, "attributes.json")
    if reset_files and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception:
            loaded = None
        if loaded is not None:
            if loaded != incoming:
                print(
                    f"[{task_name}] Using saved {item_name} for consistency. "
                    "Set reset_files=True or use a new save_dir to update them."
                )
            return loaded

    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(incoming, f, indent=2)
    except Exception:
        pass
    return incoming


def chunk_items(items: Sequence[T], n: Optional[int]) -> List[List[T]]:
    if not items:
        return []
    if n is None:
        return [list(items)]
    return [list(items[i : i + n]) for i in range(0, len(items), n)]


def resolve_attribute_batches(
    items: Sequence[Tuple[str, T]],
    *,
    requested_n: Optional[int],
    metadata: Dict[str, Any],
    reset_files: bool,
) -> Tuple[List[List[Tuple[str, T]]], Optional[int]]:
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
        batches: List[List[Tuple[str, T]]] = []
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
