---
status: accepted
---

# Checkpoint-backed task recovery with single-layer retry

Long multi-stage tasks (`discover`, `debias`, `poll`, `ideate`, `seed`) risk losing expensive LLM progress on transient failures, yet stacking inner task retries on top of SDK retries amplified latency and server-side billing during slow-response windows. We recover runs using completed-run manifests (`{task}_state.json` gated by an input signature) layered over existing per-request and sub-task checkpoints, paired with an optional task-level retry wrapper (`max_retries`, which forces `reset_files=False`) rather than an explicit stage state machine or nested inner retry loops.

## Considered Options

- **Explicit per-stage state machine**: Duplicates resumability already provided once sub-tasks (`codify`, `compare`, `bucket`, `batch_execute`) read their own checkpoints and outputs.
- **Nested per-request retries in tasks**: Multiplies billed requests when a slow server times out after accepting work.

## Consequences

- Re-running a completed task with unchanged inputs makes zero provider calls, while changing inputs invalidates stale manifests and positional-identifier checkpoints.
- Whole-run retries resume cheaply from disk without multiplying per-request calls, and deterministic errors (`ValueError`, `TypeError`) fail fast without retrying.
