"""Task-level retry with checkpoint resume.

``batch_execute`` already retries individual requests; this module wraps an
entire task run (e.g. a complex pipeline like ``discover`` or ``debias``).
On retry, ``reset_files`` is forced to ``False`` so completed stages resume
from their on-disk checkpoints instead of re-running from scratch — that is
what makes whole-run retries cheap for long pipelines.

The wrapper is duck-typed on ``task.run(*args, **kwargs)`` and deliberately
does not import task classes: the engine layer depends only on the client
contract, never on the tasks layer (which imports the engine lazily).
"""

from __future__ import annotations

from typing import Any


async def run_task_with_retry(
    task: Any,
    max_retries: int,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run ``task.run(*args, **kwargs)``, retrying up to ``max_retries`` times.

    ``task`` needs only an async ``run`` method.  Each retry resumes from
    checkpoints: ``reset_files`` is forced to ``False`` so already-completed
    sub-stages are skipped rather than redone.  Deterministic errors
    (``ValueError`` / ``TypeError``, e.g. invalid arguments) are never
    retried.  Raises the last exception once retries are exhausted.
    """
    attempts = max(0, int(max_retries))
    last_exc: Exception | None = None
    for attempt in range(attempts + 1):
        try:
            return await task.run(*args, **kwargs)
        except (ValueError, TypeError):
            raise
        except Exception as exc:  # noqa: BLE001 - provider errors are not typed
            last_exc = exc
            if attempt < attempts:
                kwargs = {**kwargs, "reset_files": False}
                print(
                    f"[{type(task).__name__}] run attempt {attempt + 1}/{attempts + 1} "
                    f"failed ({exc!r}); retrying from checkpoints"
                )
    assert last_exc is not None
    raise last_exc
