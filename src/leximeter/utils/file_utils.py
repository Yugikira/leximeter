"""Persist DataFrames with graceful degradation to split CSV files.

Ported from GABRIEL's ``utils/file_utils.py`` (the ``save_dataframe_with_fallback``
family); GABRIEL's folder-loading helpers already live in ``leximeter/load.py``.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _find_split_csv_parts(path: str) -> List[str]:
    """Return sorted split CSV parts matching ``<stem>_<n><suffix>``."""

    target = os.path.expandvars(os.path.expanduser(path))
    base_dir = os.path.dirname(target)
    stem = os.path.splitext(os.path.basename(target))[0]
    suffix = os.path.splitext(target)[1] or ".csv"
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$")
    matches: List[tuple[int, str]] = []
    if not base_dir or not os.path.isdir(base_dir):
        return []
    for name in os.listdir(base_dir):
        match = pattern.match(name)
        if not match:
            continue
        try:
            order = int(match.group(1))
        except Exception:
            continue
        matches.append((order, os.path.join(base_dir, name)))
    return [candidate for _, candidate in sorted(matches, key=lambda item: item[0])]


def _remove_paths(paths: Iterable[str]) -> None:
    """Best-effort path removal helper."""

    for candidate in paths:
        try:
            os.remove(candidate)
        except Exception:
            continue


def _resolve_chunk_sizes(
    primary_chunk_size: int,
    fallback_chunk_sizes: Optional[Iterable[int]],
) -> List[int]:
    """Build ordered, de-duplicated chunk-size attempts."""

    candidates: List[int] = []

    def _add(value: Any) -> None:
        try:
            parsed = int(value)
        except Exception:
            return
        if parsed <= 0:
            return
        if parsed not in candidates:
            candidates.append(parsed)

    _add(primary_chunk_size)
    if fallback_chunk_sizes is None:
        _add(10_000)
    else:
        for candidate in fallback_chunk_sizes:
            _add(candidate)
    if not candidates:
        candidates = [100_000, 10_000]
    return candidates


def save_dataframe_with_fallback(
    df: pd.DataFrame,
    path: str,
    *,
    index: bool = False,
    chunk_size: int = 100_000,
    fallback_chunk_sizes: Optional[Iterable[int]] = None,
    label: Optional[str] = None,
) -> List[str]:
    """Persist ``df`` to CSV, falling back to progressively smaller split files.

    The function first attempts to save to ``path``. If that fails (for example
    due to very large output files), it retries by splitting the DataFrame into
    numbered parts using dynamic row chunk sizes. By default it attempts 100k
    rows per file and then 10k rows per file.

    Returns
    -------
    list[str]
        Paths that were successfully written. An empty list indicates all save
        attempts failed.
    """

    resolved_path = os.path.expandvars(os.path.expanduser(path))
    prefix = f"[{label}] " if label else ""
    try:
        parent = os.path.dirname(resolved_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        logger.debug("Could not ensure parent directory for %s", resolved_path)

    try:
        df.to_csv(resolved_path, index=index)
        stale_parts = _find_split_csv_parts(resolved_path)
        if stale_parts:
            _remove_paths(stale_parts)
        return [resolved_path]
    except Exception as primary_exc:
        print(
            f"{prefix}Unable to save final CSV to {resolved_path}. "
            "Trying chunked backup files."
        )
        logger.warning(
            "Failed to save CSV to %s; attempting chunked backup files.",
            resolved_path,
            exc_info=primary_exc,
        )

    stem, ext = os.path.splitext(resolved_path)
    ext = ext or ".csv"
    total_rows = len(df)
    chunk_attempts = _resolve_chunk_sizes(chunk_size, fallback_chunk_sizes)

    for chunk in chunk_attempts:
        attempt_paths: List[str] = []
        part_count = max(1, int(math.ceil(total_rows / max(1, chunk))))
        print(
            f"{prefix}Trying chunked backup with {chunk:,} rows per file "
            f"({part_count} part(s))."
        )
        _remove_paths(_find_split_csv_parts(resolved_path))
        try:
            for idx in range(part_count):
                start = idx * chunk
                stop = start + chunk
                part_path = f"{stem}_{idx + 1}{ext}"
                df.iloc[start:stop].to_csv(part_path, index=index)
                attempt_paths.append(part_path)
        except Exception as split_exc:
            _remove_paths(attempt_paths)
            logger.warning(
                "Chunked CSV fallback failed for %s at chunk size %s.",
                resolved_path,
                chunk,
                exc_info=split_exc,
            )
            continue

        print(
            f"{prefix}Saved fallback split files ({part_count} part(s)) with base path "
            f"{stem}_*.{ext.lstrip('.')}"
        )
        logger.warning(
            "Saved chunked CSV fallback for %s into %d part(s) using chunk size %d.",
            resolved_path,
            part_count,
            chunk,
        )
        return attempt_paths

    print(
        f"{prefix}Final DataFrame could not be saved to disk. "
        "Returning the DataFrame in memory only."
    )
    logger.warning(
        "All chunked CSV fallback attempts failed for %s.",
        resolved_path,
    )
    return []


__all__ = [
    "_find_split_csv_parts",
    "_remove_paths",
    "_resolve_chunk_sizes",
    "save_dataframe_with_fallback",
]
