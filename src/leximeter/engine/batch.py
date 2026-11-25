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


async def complete_with_retry(
    client: LLMClient,
    prompt: str,
    *,
    model: str,
    json_mode: bool,
    max_retries: int,
    timeout: float | None,
    **generation_kwargs: Any,
) -> str:
    """Call ``client.complete``, retrying with exponential backoff.

    Also used directly by tasks that call the client outside a batch
    (e.g. :class:`~leximeter.tasks.seed.Seed`), so single-call failures get
    the same per-request retry coverage as batched jobs.
    """
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
                await asyncio.sleep(min(2**attempt, 30))
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
    warmup_first: bool | None = None,
    **generation_kwargs: Any,
) -> pd.DataFrame:
    """Run ``prompts`` through ``client`` concurrently, resuming from ``save_path``.

    When ``warmup_first`` is ``None`` (default) and
    ``len(pending) > max(8, n_parallels)``, the first pending prompt is awaited
    sequentially before launching the remaining items concurrently so provider
    prefix caches populate before wave 1 fans out.  Explicit ``True`` warms up
    whenever ``len(pending) > 1``; ``False`` disables warmup.

    Returns a DataFrame with ``Identifier``, ``Response``, ``Successful``,
    ``Error Log`` and ``Time Taken`` columns.
    """
    if len(prompts) != len(identifiers):
        raise ValueError("prompts and identifiers must have the same length")

    checkpoint = (
        _checkpoint.load_checkpoint(save_path)
        if (save_path and not reset_files)
        else pd.DataFrame(columns=_checkpoint.COLUMNS)
    )
    done = _checkpoint.completed_identifiers(checkpoint)

    pending = [
        (ident, prompt)
        for ident, prompt in zip(identifiers, prompts)
        if ident not in done
    ]

    sem = asyncio.Semaphore(max(1, n_parallels))
    lock = asyncio.Lock()
    new_rows: list[dict[str, Any]] = []
    flushed = 0

    async def worker(ident: str, prompt: str) -> None:
        nonlocal flushed
        async with sem:
            started = time.monotonic()
            try:
                text = await complete_with_retry(
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
                if save_path and len(new_rows) - flushed >= _SAVE_EVERY:
                    _checkpoint.append_rows(save_path, new_rows[flushed:])
                    flushed = len(new_rows)

    should_warmup = (
        len(pending) > 1
        if warmup_first is True
        else False
        if warmup_first is False
        else len(pending) > max(8, n_parallels)
    )

    if should_warmup:
        first_ident, first_prompt = pending[0]
        await worker(first_ident, first_prompt)
        rest = pending[1:]
        if rest:
            await asyncio.gather(*(worker(i, p) for i, p in rest))
    elif pending:
        await asyncio.gather(*(worker(i, p) for i, p in pending))

    if save_path and flushed < len(new_rows):
        _checkpoint.append_rows(save_path, new_rows[flushed:])

    return _checkpoint.merge_with_checkpoint(checkpoint, new_rows)
