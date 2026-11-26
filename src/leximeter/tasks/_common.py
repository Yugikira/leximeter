"""Shared, provider-agnostic helpers used by task implementations.

Ported from GABRIEL's ``_run_utils`` / ``_attribute_utils``.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeVar, cast

import pandas as pd

T = TypeVar("T")

ATTRIBUTE_WARNING_THRESHOLD = 12

OrderStrategy = Literal["per_run", "fixed", "per_item"]
VALID_ORDER_STRATEGIES: frozenset[str] = frozenset({"per_run", "fixed", "per_item"})


def validate_order_strategy(strategy: str) -> OrderStrategy:
    """Validate an ``order_strategy`` setting and return it typed."""
    if strategy not in VALID_ORDER_STRATEGIES:
        raise ValueError(
            "order_strategy must be one of "
            f"{sorted(VALID_ORDER_STRATEGIES)}, got {strategy!r}"
        )
    return cast(OrderStrategy, strategy)


_EXACT_POOL_MAX_LEN = 6


@functools.lru_cache(maxsize=256)
def _small_permutation_pool(batch_idx: int, n: int) -> tuple[tuple[int, ...], ...]:
    """Return all ``n!`` index permutations in a deterministic order.

    Index 0 is caller order ``(0, ..., n - 1)`` and index 1 is reversed order
    ``(n - 1, ..., 0)``; the remaining ``n! - 2`` permutations are shuffled
    deterministically from ``(batch_idx, n)``.
    """
    indices = tuple(range(n))
    rev_indices = tuple(reversed(indices))
    others = [
        perm
        for perm in itertools.permutations(indices)
        if perm != indices and perm != rev_indices
    ]
    seed_bytes = hashlib.sha256(f"{batch_idx}:{n}".encode()).digest()[:8]
    rng = random.Random(int.from_bytes(seed_bytes, "big"))
    rng.shuffle(others)
    return (indices, rev_indices, *others)


def order_mapping(
    mapping: dict[str, T],
    *,
    strategy: OrderStrategy = "per_run",
    run_idx: int = 1,
    batch_idx: int = 0,
) -> dict[str, T]:
    """Order a rubric/key dictionary according to ``strategy``.

    - ``"fixed"``: preserve caller insertion order across all items and runs.
    - ``"per_run"``: preserve caller order on ``run_idx == 1``, reverse on
      ``run_idx == 2``, and yield strictly distinct deterministic permutations
      across ``run_idx = 1..min(len(mapping)!, n_runs)`` seeded from
      ``(batch_idx, len(mapping))`` (cycling cleanly modulo ``len(mapping)!``).
    - ``"per_item"``: unseeded per-call shuffle (legacy GABRIEL compatibility).
    """
    validate_order_strategy(strategy)
    items = list(mapping.items())
    n = len(items)
    if n <= 1 or strategy == "fixed":
        return dict(items)
    if strategy == "per_item":
        random.shuffle(items)
        return dict(items)
    if run_idx <= 1:
        return dict(items)
    if n <= _EXACT_POOL_MAX_LEN:
        pool = _small_permutation_pool(batch_idx, n)
        perm = pool[(run_idx - 1) % len(pool)]
        return dict(items[i] for i in perm)

    effective_run = ((run_idx - 1) % math.factorial(n)) + 1 if n <= 12 else run_idx
    if effective_run == 1:
        return dict(items)
    indices = tuple(range(n))
    rev_indices = tuple(reversed(indices))
    if effective_run == 2:
        return dict(items[i] for i in rev_indices)

    seed_bytes = hashlib.sha256(f"{batch_idx}:{n}".encode()).digest()[:8]
    rng = random.Random(int.from_bytes(seed_bytes, "big"))
    seen: set[tuple[int, ...]] = {indices, rev_indices}
    perm = indices
    for _ in range(3, effective_run + 1):
        candidate = list(indices)
        rng.shuffle(candidate)
        perm = tuple(candidate)
        while perm in seen:
            rng.shuffle(candidate)
            perm = tuple(candidate)
        seen.add(perm)
    return dict(items[i] for i in perm)


def make_mapping_orderer(
    mapping: dict[str, T],
    *,
    strategy: OrderStrategy = "per_run",
    run_idx: int = 1,
    batch_idx: int = 0,
) -> Callable[[], dict[str, T]]:
    """Return a zero-arg callable producing the ordered mapping for each call.

    For ``"fixed"`` and ``"per_run"``, the ordered mapping is computed once and
    reused across every call in the run so prompt prefixes stay identical.
    For ``"per_item"``, each invocation returns a fresh random shuffle.
    """
    validate_order_strategy(strategy)
    if strategy == "per_item":
        return lambda: order_mapping(
            mapping,
            strategy="per_item",
            run_idx=run_idx,
            batch_idx=batch_idx,
        )
    ordered = order_mapping(
        mapping,
        strategy=strategy,
        run_idx=run_idx,
        batch_idx=batch_idx,
    )
    return lambda: ordered


def hash_identifier(value: Any) -> str:
    """Return a stable 16-char hex identifier for a piece of content."""
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]


def run_metadata_path(save_dir: str, base_name: str) -> str:
    return os.path.join(save_dir, f"{base_name}_run_metadata.json")


def load_run_metadata(
    save_dir: str, base_name: str, *, reset_files: bool
) -> dict[str, Any]:
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


def effective_decision_model(
    decision_client: Any | None,
    decision_model: str | None,
    *,
    fallback: str,
) -> str | None:
    """Return the resolved decision model when a Decision Provider is active, else ``None``."""
    if decision_client is None:
        return None
    from ..client.resolve import resolve_decision_model

    if getattr(decision_client, "requires_decision_model", False):
        resolved = resolve_decision_model(decision_model, fallback=None)
        if not resolved:
            raise ValueError(
                f"{type(decision_client).__name__} requires an explicit "
                "decision_model or LEXIMETER_DECISION_MODEL environment variable "
                f"(refusing to fall back to generative model {fallback!r})."
            )
        return resolved
    return resolve_decision_model(decision_model, fallback=fallback) or fallback


def should_reset_substage_checkpoint(
    save_dir: str,
    base_name: str,
    *,
    expected_model: str,
    decision_provider: bool,
    reset_files: bool,
    extra_metadata: Mapping[str, Any] | None = None,
    untracked_raw_path: str | None = None,
) -> bool:
    """Return ``True`` when a sub-stage checkpoint must reset due to model/provider/state change."""
    if reset_files:
        return True
    prev_meta = load_run_metadata(save_dir, base_name, reset_files=False)
    if not prev_meta:
        return bool(untracked_raw_path and os.path.exists(untracked_raw_path))
    if (
        prev_meta.get("model") != expected_model
        or bool(prev_meta.get("decision_provider", False)) != decision_provider
    ):
        return True
    if extra_metadata:
        for key, expected_val in extra_metadata.items():
            if key not in prev_meta or prev_meta.get(key) != expected_val:
                return True
    return False


def prepare_substage_checkpoint(
    save_dir: str,
    base_name: str,
    *,
    expected_model: str,
    decision_provider: bool,
    reset_files: bool,
    extra_metadata: Mapping[str, Any] | None = None,
    artifacts: Sequence[str] = (),
    untracked_raw_path: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Validate sub-stage metadata and purge stale checkpoint artifacts on mismatch."""
    effective_reset = should_reset_substage_checkpoint(
        save_dir,
        base_name,
        expected_model=expected_model,
        decision_provider=decision_provider,
        reset_files=reset_files,
        extra_metadata=extra_metadata,
        untracked_raw_path=untracked_raw_path,
    )
    if effective_reset:
        for artifact in artifacts:
            remove_file(artifact)
    run_metadata = load_run_metadata(save_dir, base_name, reset_files=effective_reset)
    return effective_reset, run_metadata


def list_numbered_files(
    directory: str,
    prefix: str,
    suffix: str = ".csv",
    *,
    min_index: int = 0,
) -> list[tuple[int, str]]:
    """Return sorted ``(index, full_path)`` pairs for ``{prefix}{index}{suffix}`` files."""
    if not directory or not os.path.isdir(directory):
        return []
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    found: list[tuple[int, str]] = []
    plen = len(prefix)
    slen = len(suffix)
    for fname in names:
        if not (fname.startswith(prefix) and fname.endswith(suffix)):
            continue
        middle = fname[plen : len(fname) - slen if slen else None]
        if not middle.isdigit():
            continue
        idx = int(middle)
        if idx >= min_index:
            found.append((idx, os.path.join(directory, fname)))
    found.sort(key=lambda item: item[0])
    return found


def purge_numbered_files(
    directory: str,
    prefix: str,
    suffix: str = ".csv",
    *,
    min_index: int = 0,
) -> None:
    """Remove ``{prefix}{index}{suffix}`` files with ``index >= min_index``."""
    for _, path in list_numbered_files(directory, prefix, suffix, min_index=min_index):
        remove_file(path)


def task_state_path(save_dir: str, task_name: str) -> str:
    return os.path.join(save_dir, f"{task_name}_state.json")


def load_task_state(save_dir: str, task_name: str) -> dict[str, Any] | None:
    """Read a task-level checkpoint manifest (``{task}_state.json``)."""
    return load_json_cache(task_state_path(save_dir, task_name))


def save_task_state(
    save_dir: str,
    task_name: str,
    *,
    input_signature: str,
    result_artifacts: Sequence[str],
    status: str = "complete",
) -> None:
    """Persist a task-level checkpoint manifest (best-effort, never raises)."""
    write_json_cache(
        task_state_path(save_dir, task_name),
        {
            "task": task_name,
            "status": status,
            "input_signature": input_signature,
            "result_artifacts": list(result_artifacts),
        },
    )


def clear_task_state(save_dir: str, task_name: str) -> None:
    """Remove a task-level checkpoint manifest (``reset_files=True`` path)."""
    remove_file(task_state_path(save_dir, task_name))


def load_completed_csv(
    save_dir: str,
    task_name: str,
    *,
    signature: str,
    file_name: str,
    reset_files: bool = False,
) -> pd.DataFrame | None:
    """Completed-run verification for CSV-backed task results.

    Clears the task manifest when ``reset_files`` is set, otherwise returns
    the saved frame when the manifest is complete and the input signature
    matches.  ``None`` means the run must (re)execute: state missing,
    inputs changed, or the result file missing/malformed/empty.
    """
    if reset_files:
        clear_task_state(save_dir, task_name)
        return None
    state = load_task_state(save_dir, task_name)
    if (
        not state
        or state.get("status") != "complete"
        or state.get("input_signature") != signature
    ):
        return None
    path = os.path.join(save_dir, file_name)
    if not os.path.exists(path):
        return None
    try:
        out = pd.read_csv(path)
    except (OSError, ValueError):
        return None
    if out.empty:
        return None
    return out


def remove_file(path: str | None) -> None:
    """Best-effort file removal (never raises)."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def compute_input_signature(*parts: Any) -> str:
    """Stable sha1 over task inputs; changes whenever any input changes.

    DataFrames are canonicalised to records (deterministic for identical
    data) so the same run with the same inputs always hashes the same.
    """

    def _canon(value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            return {"df": value.to_dict(orient="records")}
        if isinstance(value, pd.Series):
            return {"series": value.tolist()}
        if isinstance(value, dict):
            return {str(k): _canon(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_canon(v) for v in value]
        return value

    try:
        payload = json.dumps(
            [_canon(p) for p in parts],
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError):
        payload = repr(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


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
    max_batch_size: int | None = None,
) -> tuple[list[list[tuple[str, T]]], int | None]:
    """Split attribute items into prompt batches, reusing any saved plan.

    ``requested_n`` is the max attributes per prompt (``None`` = one batch).
    ``max_batch_size`` is an optional hard cap (e.g. a Decision Provider's
    ``max_questions_per_call``); any persisted plan containing a batch larger
    than ``max_batch_size`` is rejected and re-chunked.
    """
    if requested_n is not None and (
        not isinstance(requested_n, int) or requested_n < 1
    ):
        raise ValueError("n_attributes_per_run must be None or an integer >= 1")
    if max_batch_size is not None and (
        not isinstance(max_batch_size, int) or max_batch_size < 1
    ):
        raise ValueError("max_batch_size must be None or an integer >= 1")
    effective_n = (
        min(requested_n, max_batch_size)
        if requested_n is not None and max_batch_size is not None
        else (requested_n if requested_n is not None else max_batch_size)
    )
    if not items:
        return [], effective_n

    # Reuse a previously-persisted batch plan so resuming lines up with cache.
    if not reset_files and isinstance(metadata.get("attribute_batches"), list):
        lookup = {name: value for name, value in items}
        batches: list[list[tuple[str, T]]] = []
        valid = True
        for batch in metadata["attribute_batches"]:
            if not isinstance(batch, list):
                valid = False
                break
            if max_batch_size is not None and len(batch) > max_batch_size:
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

    if effective_n is None and len(items) > ATTRIBUTE_WARNING_THRESHOLD:
        print(
            f"Processing {len(items)} attributes in one prompt. "
            "Set n_attributes_per_run=12 (or lower) for more consistent outputs."
        )

    return chunk_items(items, effective_n), effective_n


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
