"""Provider-agnostic concurrent batch execution with retry and checkpointing.

The engine depends only on ``LLMClient.complete`` — never on a specific vendor.
It provides: bounded concurrency, per-request retry with backoff, and resumable
checkpointing (skip already-completed identifiers).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pandas as pd

from ..client.base import LLMClient
from . import checkpoint as _checkpoint

_SAVE_EVERY = 100


async def _complete_with_retry(
    client: LLMClient,
    prompt: str,
    *,
    model: str,
    json_mode: bool,
    max_retries: int,
    timeout: float | None,
    **generation_kwargs: Any,
) -> str:
    """Call ``client.complete``, retrying with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            if timeout:
                return await asyncio.wait_for(
                    client.complete(
                        prompt, model=model, json_mode=json_mode, **generation_kwargs
                    ),
                    timeout=timeout,
                )
            return await client.complete(
                prompt, model=model, json_mode=json_mode, **generation_kwargs
            )
        except Exception as exc:  # noqa: BLE001 - provider errors are not typed
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** attempt, 30))
    assert last_exc is not None
    raise last_exc


async def batch_execute(
    prompts: list[str],
    identifiers: list[str],
    *,
    client: LLMClient,
    model: str,
    json_mode: bool = False,
    n_parallels: int = 50,
    save_path: str | None = None,
    reset_files: bool = False,
    max_retries: int = 3,
    timeout: float | None = None,
    **generation_kwargs: Any,
) -> pd.DataFrame:
    """Run ``prompts`` through ``client`` concurrently, resuming from ``save_path``.

    Returns a DataFrame with ``Identifier``, ``Response``, ``Successful``,
    ``Error Log`` and ``Time Taken`` columns.
    """
    if len(prompts) != len(identifiers):
        raise ValueError("prompts and identifiers must have the same length")

    checkpoint = _checkpoint.load_checkpoint(save_path) if (save_path and not reset_files) else pd.DataFrame(columns=_checkpoint.COLUMNS)
    done = _checkpoint.completed_identifiers(checkpoint)

    pending = [
        (ident, prompt)
        for ident, prompt in zip(identifiers, prompts)
        if ident not in done
    ]

    sem = asyncio.Semaphore(max(1, n_parallels))
    lock = asyncio.Lock()
    new_rows: list[dict[str, Any]] = []

    async def worker(ident: str, prompt: str) -> None:
        async with sem:
            started = time.monotonic()
            try:
                text = await _complete_with_retry(
                    client,
                    prompt,
                    model=model,
                    json_mode=json_mode,
                    max_retries=max_retries,
                    timeout=timeout,
                    **generation_kwargs,
                )
                row = {
                    "Identifier": ident,
                    "Response": text,
                    "Successful": True,
                    "Error Log": "",
                    "Time Taken": round(time.monotonic() - started, 3),
                }
            except Exception as exc:  # noqa: BLE001
                row = {
                    "Identifier": ident,
                    "Response": "",
                    "Successful": False,
                    "Error Log": repr(exc),
                    "Time Taken": round(time.monotonic() - started, 3),
                }
            async with lock:
                new_rows.append(row)
                if save_path and len(new_rows) % _SAVE_EVERY == 0:
                    _checkpoint.append_rows(save_path, new_rows[-_SAVE_EVERY:])

    if pending:
        await asyncio.gather(*(worker(i, p) for i, p in pending))

    if save_path:
        _checkpoint.append_rows(save_path, new_rows)

    return _checkpoint.merge_with_checkpoint(checkpoint, new_rows)
