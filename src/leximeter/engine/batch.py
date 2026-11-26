"""Provider-agnostic concurrent batch execution with retry and checkpointing.

The engine depends only on ``LLMClient.complete`` and ``DecisionClient.decide``
— never on a specific vendor.  It provides: bounded concurrency, per-request
retry with backoff (honoring ``Retry-After``), and resumable checkpointing
(skip already-completed identifiers).
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ..client.base import LLMClient
from ..client.decision import (
    DecisionAPIError,
    DecisionClient,
    DecisionQuestion,
    DecisionResult,
    parse_retry_after,
)
from . import checkpoint as _checkpoint

_SAVE_EVERY = 100
logger = logging.getLogger(__name__)


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


def _extract_retry_after(exc: Exception) -> float | None:
    """Extract a ``Retry-After`` delay (seconds) from a provider exception if present."""
    direct = getattr(exc, "retry_after", None)
    if direct is not None:
        try:
            return max(0.0, float(direct))
        except (TypeError, ValueError):
            pass
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        return parse_retry_after(headers.get("Retry-After"))
    return None


async def decide_with_retry(
    client: DecisionClient,
    state: str | dict[str, Any],
    questions: dict[str, DecisionQuestion],
    *,
    model: str,
    max_retries: int,
    timeout: float | None,
    **decision_kwargs: Any,
) -> DecisionResult:
    """Call ``client.decide``, retrying transient errors and honoring ``Retry-After``."""
    client_retries = getattr(client, "_max_retries", 0)
    effective_retries = (
        0
        if isinstance(client_retries, int)
        and not isinstance(client_retries, bool)
        and client_retries > 0
        else max(0, int(max_retries))
    )
    last_exc: Exception | None = None
    for attempt in range(effective_retries + 1):
        try:
            coro = client.decide(state, questions, model=model, **decision_kwargs)
            if timeout:
                return await asyncio.wait_for(coro, timeout=timeout)
            return await coro
        except (ValueError, TypeError):
            raise
        except DecisionAPIError as exc:
            if not exc.retryable:
                raise
            last_exc = exc
            if attempt < effective_retries:
                retry_after = _extract_retry_after(exc)
                delay = retry_after if retry_after is not None else min(2**attempt, 30)
                await asyncio.sleep(delay)
        except Exception as exc:  # noqa: BLE001 - provider errors are not typed
            last_exc = exc
            if attempt < effective_retries:
                retry_after = _extract_retry_after(exc)
                delay = retry_after if retry_after is not None else min(2**attempt, 30)
                await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def batch_decide(
    states: Sequence[str | dict[str, Any]],
    question_sets: Sequence[dict[str, DecisionQuestion]],
    identifiers: Sequence[str],
    *,
    client: DecisionClient,
    model: str,
    n_parallels: int = 50,
    save_path: str | None = None,
    reset_files: bool = False,
    max_retries: int = 3,
    timeout: float | None = None,
    **decision_kwargs: Any,
) -> pd.DataFrame:
    """Run decision requests concurrently through ``client``, resuming from ``save_path``.

    Each successful row serializes its normalized :class:`DecisionResult`
    (including calibrated raw probabilities) as JSON in the ``Response`` column.
    Returns a DataFrame with ``Identifier``, ``Response``, ``Successful``,
    ``Error Log``, and ``Time Taken`` columns.
    """
    if not (len(states) == len(question_sets) == len(identifiers)):
        raise ValueError(
            "states, question_sets, and identifiers must have the same length"
        )
    if getattr(client, "requires_decision_model", False) and (
        not isinstance(model, str) or not model.strip()
    ):
        raise ValueError(
            f"{type(client).__name__} requires a non-empty decision model ID "
            "(pass decision_model=... or set LEXIMETER_DECISION_MODEL)."
        )

    checkpoint = (
        _checkpoint.load_checkpoint(save_path)
        if (save_path and not reset_files)
        else pd.DataFrame(columns=_checkpoint.COLUMNS)
    )
    done = _checkpoint.completed_identifiers(checkpoint)

    pending = [
        (ident, state, questions)
        for ident, state, questions in zip(identifiers, states, question_sets)
        if ident not in done
    ]

    sem = asyncio.Semaphore(max(1, n_parallels))
    lock = asyncio.Lock()
    new_rows: list[dict[str, Any]] = []
    flushed = 0
    succeeded_count = 0
    failed_count = 0
    first_error: tuple[int, Exception] | None = None
    fatal_4xx_errors: dict[int, DecisionAPIError] = {}

    async def worker(
        idx: int,
        ident: str,
        state: str | dict[str, Any],
        questions: dict[str, DecisionQuestion],
    ) -> None:
        nonlocal flushed, succeeded_count, failed_count, first_error
        async with sem:
            started = time.monotonic()
            exc_caught: Exception | None = None
            try:
                result = await decide_with_retry(
                    client,
                    state,
                    questions,
                    model=model,
                    max_retries=max_retries,
                    timeout=timeout,
                    **decision_kwargs,
                )
                row = {
                    "Identifier": ident,
                    "Response": result.to_json(),
                    "Successful": True,
                    "Error Log": "",
                    "Time Taken": round(time.monotonic() - started, 3),
                }
            except Exception as exc:  # noqa: BLE001
                exc_caught = exc
                row = {
                    "Identifier": ident,
                    "Response": "",
                    "Successful": False,
                    "Error Log": repr(exc),
                    "Time Taken": round(time.monotonic() - started, 3),
                }
            async with lock:
                if exc_caught is None:
                    succeeded_count += 1
                else:
                    failed_count += 1
                    if first_error is None or idx < first_error[0]:
                        first_error = (idx, exc_caught)
                    if (
                        isinstance(exc_caught, DecisionAPIError)
                        and not exc_caught.retryable
                        and exc_caught.status_code is not None
                        and 400 <= exc_caught.status_code < 500
                    ):
                        fatal_4xx_errors[idx] = exc_caught
                new_rows.append(row)
                if save_path and len(new_rows) - flushed >= _SAVE_EVERY:
                    _checkpoint.append_rows(save_path, new_rows[flushed:])
                    flushed = len(new_rows)

    if pending:
        await asyncio.gather(
            *(worker(idx, i, s, q) for idx, (i, s, q) in enumerate(pending))
        )

    if save_path and len(new_rows) > flushed:
        _checkpoint.append_rows(save_path, new_rows[flushed:])

    if pending and failed_count > 0:
        summary_exc = (
            fatal_4xx_errors[min(fatal_4xx_errors)]
            if fatal_4xx_errors
            else first_error[1]
            if first_error is not None
            else None
        )
        msg = (
            f"batch_decide: {failed_count}/{len(pending)} pending item(s) failed "
            f"({succeeded_count} succeeded)"
            + (f"; first error: {summary_exc}" if summary_exc is not None else "")
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    if pending and succeeded_count == 0 and fatal_4xx_errors:
        raise fatal_4xx_errors[min(fatal_4xx_errors)]

    return _checkpoint.merge_with_checkpoint(checkpoint, new_rows)
