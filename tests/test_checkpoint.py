"""Tests for task-level retry and checkpoint/resume (TICKET-001).

Covers: task-level ``max_retries`` wrapper, completed-run verification
(zero API calls on rerun), input-signature invalidation, ``reset_files``
state clearing, and sub-task idempotency (codify / compare / seed).
All tests use fake providers — no network, no keys.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import pytest
from conftest import CountingClient, DictClient, TextClient
from test_tasks import DebiasClient, StageClient

from leximeter import seed
from leximeter.engine.retry import run_task_with_retry


def run(coro):
    return asyncio.run(coro)


class FlakyClient(DictClient):
    """Fail the first ``fail_until`` completions, then behave normally."""

    def __init__(self, payload: dict, fail_until: int = 1):
        super().__init__(payload)
        self.fail_until = fail_until
        self.calls = 0

    async def complete(self, prompt, *, model, json_mode=False, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_until:
            raise RuntimeError("provider down")
        return await super().complete(
            prompt, model=model, json_mode=json_mode, **kwargs
        )


# ---------------------------------------------------------------------------
# Task-level retry wrapper
# ---------------------------------------------------------------------------


class FakeTask:
    """Minimal task stub recording each attempt's kwargs for wrapper tests."""

    def __init__(self, fail_attempts: int = 0, error: Exception | None = None):
        self.fail_attempts = fail_attempts
        self.error = error or RuntimeError("boom")
        self.attempts: list[dict[str, Any]] = []

    async def run(self, *args: Any, **kwargs: Any) -> str:
        self.attempts.append(kwargs)
        if len(self.attempts) <= self.fail_attempts:
            raise self.error
        return "ok"


def test_retry_wrapper_succeeds_on_first_attempt():
    task = FakeTask(fail_attempts=0)
    result = run(run_task_with_retry(task, max_retries=2, reset_files=True))
    assert result == "ok"
    assert len(task.attempts) == 1
    assert task.attempts[0]["reset_files"] is True


def test_retry_wrapper_resumes_from_checkpoints_on_retry():
    # First attempt fails with reset_files=True; retries must force
    # reset_files=False so completed stages resume instead of re-running.
    task = FakeTask(fail_attempts=2)
    result = run(run_task_with_retry(task, max_retries=5, reset_files=True))
    assert result == "ok"
    assert len(task.attempts) == 3
    assert task.attempts[0]["reset_files"] is True
    assert task.attempts[1]["reset_files"] is False
    assert task.attempts[2]["reset_files"] is False


def test_retry_wrapper_raises_after_exhausting_retries():
    task = FakeTask(fail_attempts=3)
    with pytest.raises(RuntimeError, match="boom"):
        run(run_task_with_retry(task, max_retries=2))
    assert len(task.attempts) == 3


def test_retry_wrapper_zero_retries_raises_immediately():
    task = FakeTask(fail_attempts=1)
    with pytest.raises(RuntimeError, match="boom"):
        run(run_task_with_retry(task, max_retries=0))
    assert len(task.attempts) == 1


def test_retry_wrapper_does_not_retry_validation_errors():
    # Deterministic errors (ValueError/TypeError, e.g. invalid arguments)
    # must propagate immediately instead of burning the retry budget.
    class BadArgsTask(FakeTask):
        async def run(self, *args, **kwargs):
            self.attempts.append(kwargs)
            raise ValueError("provide either column_name or both columns")

    task = BadArgsTask()
    with pytest.raises(ValueError, match="column_name"):
        run(run_task_with_retry(task, max_retries=5))
    assert len(task.attempts) == 1


# ---------------------------------------------------------------------------
# Shared task-state helpers (_common)
# ---------------------------------------------------------------------------


def test_task_state_roundtrip(tmp_path):
    from leximeter.tasks import _common

    d = str(tmp_path)
    assert _common.load_task_state(d, "ideate") is None
    _common.save_task_state(
        d, "ideate", input_signature="sig1", result_artifacts=["ideation.csv"]
    )
    state = _common.load_task_state(d, "ideate")
    assert state is not None
    assert state["status"] == "complete"
    assert state["input_signature"] == "sig1"
    assert state["result_artifacts"] == ["ideation.csv"]
    _common.clear_task_state(d, "ideate")
    assert _common.load_task_state(d, "ideate") is None


def test_input_signature_stable_and_sensitive():
    from leximeter.tasks import _common

    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pd.DataFrame({"a": [1, 2], "b": ["x", "z"]})
    # same input -> same signature; any input change -> different signature
    assert _common.compute_input_signature(df1, "k") == _common.compute_input_signature(
        df1, "k"
    )
    assert _common.compute_input_signature(df1, "k") != _common.compute_input_signature(
        df2, "k"
    )
    assert _common.compute_input_signature(df1, "k") != _common.compute_input_signature(
        df1, "k2"
    )


# ---------------------------------------------------------------------------
# Seed: per-batch retry + completed-run verification
# ---------------------------------------------------------------------------


def test_seed_task_level_retry_rescues_failed_run(tmp_path):
    # Whole-run retry (no per-batch leximeter retries — those were rolled
    # back per worksummary_0816): the first run fails on its only call, the
    # wrapper reruns the whole generation, and the second attempt succeeds.
    client = FlakyClient({"entity 1": "person A", "entity 2": "person B"}, fail_until=1)
    res = run(
        seed(
            "generate people",
            save_dir=str(tmp_path / "s"),
            client=client,
            num_entities=2,
            entities_per_generation=2,
            max_retries=2,
        )
    )
    assert len(res) == 2
    # attempt 1: 1 failed call; attempt 2: 1 successful call
    assert client.calls == 2


def test_seed_completed_verification_zero_calls(tmp_path):
    client = CountingClient({"entity 1": "person A", "entity 2": "person B"})
    save_dir = str(tmp_path / "s")
    first = run(
        seed(
            "generate people",
            save_dir=save_dir,
            client=client,
            num_entities=2,
            entities_per_generation=2,
        )
    )
    assert len(first) == 2
    assert client.calls == 1
    second = run(
        seed(
            "generate people",
            save_dir=save_dir,
            client=client,
            num_entities=2,
            entities_per_generation=2,
        )
    )
    assert client.calls == 1  # no new completions
    assert list(second.columns) == ["entity", "entity_id"]
    assert len(second) == 2


def test_seed_signature_invalidation_regenerates(tmp_path):
    client = CountingClient({"entity 1": "person A", "entity 2": "person B"})
    save_dir = str(tmp_path / "s")
    run(
        seed(
            "generate people",
            save_dir=save_dir,
            client=client,
            num_entities=2,
            entities_per_generation=2,
        )
    )
    calls_after_first = client.calls
    # Same instructions but a different target count -> must regenerate.
    run(
        seed(
            "generate people",
            save_dir=save_dir,
            client=client,
            num_entities=2,
            entities_per_generation=4,
        )
    )
    assert client.calls > calls_after_first


def test_seed_model_change_regenerates(tmp_path):
    client = CountingClient({"entity 1": "person A", "entity 2": "person B"})
    save_dir = str(tmp_path / "s")
    run(
        seed(
            "generate people",
            save_dir=save_dir,
            client=client,
            num_entities=2,
            entities_per_generation=2,
            model="gpt-5.5",
        )
    )
    calls_after_first = client.calls
    run(
        seed(
            "generate people",
            save_dir=save_dir,
            client=client,
            num_entities=2,
            entities_per_generation=2,
            model="gpt-5.4-mini",  # model change -> old artifacts not reused
        )
    )
    assert client.calls > calls_after_first


def test_seed_reset_files_reruns(tmp_path):
    client = CountingClient({"entity 1": "person A", "entity 2": "person B"})
    save_dir = str(tmp_path / "s")
    run(
        seed(
            "generate people",
            save_dir=save_dir,
            client=client,
            num_entities=2,
            entities_per_generation=2,
        )
    )
    calls_after_first = client.calls
    run(
        seed(
            "generate people",
            save_dir=save_dir,
            client=client,
            num_entities=2,
            entities_per_generation=2,
            reset_files=True,
        )
    )
    assert client.calls > calls_after_first


# ---------------------------------------------------------------------------
# Ideate: completed-run verification
# ---------------------------------------------------------------------------


def test_ideate_completed_verification_zero_calls(tmp_path):
    from leximeter import ideate

    client = TextClient("Title: Test\nIn a nutshell: A nutshell.")
    save_dir = str(tmp_path / "i")
    first = run(ideate("some topic", save_dir=save_dir, client=client, n_ideas=2))
    assert len(first) == 2
    second = run(ideate("some topic", save_dir=save_dir, client=client, n_ideas=2))
    assert len(second) == 2
    assert second["title"].iloc[0] == "Test"


def test_ideate_signature_invalidation_regenerates(tmp_path):
    from leximeter import ideate

    client = CountingClient({"x": 1})
    save_dir = str(tmp_path / "i")
    run(ideate("topic A", save_dir=save_dir, client=client, n_ideas=2))
    calls_after_first = client.calls
    run(ideate("topic B", save_dir=save_dir, client=client, n_ideas=2))
    assert client.calls > calls_after_first


def test_ideate_reset_files_reruns(tmp_path):
    from leximeter import ideate

    client = CountingClient({"x": 1})
    save_dir = str(tmp_path / "i")
    run(ideate("topic A", save_dir=save_dir, client=client, n_ideas=2))
    calls_after_first = client.calls
    run(
        ideate(
            "topic A",
            save_dir=save_dir,
            client=client,
            n_ideas=2,
            reset_files=True,
        )
    )
    assert client.calls > calls_after_first


# ---------------------------------------------------------------------------
# Poll: completed-run verification (seeds supplied as a frame)
# ---------------------------------------------------------------------------


def test_poll_completed_verification_zero_calls(tmp_path):
    from leximeter import poll

    df = pd.DataFrame({"person": ["alice", "bob"]})
    client = CountingClient({"age": [30, 40]})
    save_dir = str(tmp_path / "p")
    first = run(
        poll(
            df,
            "person",
            save_dir=save_dir,
            client=client,
            questions=["age"],
            n_questions_per_run=8,
        )
    )
    assert len(first) == 2
    calls_after_first = client.calls
    second = run(
        poll(
            df,
            "person",
            save_dir=save_dir,
            client=client,
            questions=["age"],
            n_questions_per_run=8,
        )
    )
    assert client.calls == calls_after_first  # zero new completions
    assert len(second) == 2
    assert list(second.columns) == ["seed", "persona", "age"]


def test_poll_signature_invalidation_regenerates(tmp_path):
    from leximeter import poll

    df1 = pd.DataFrame({"person": ["alice", "bob"]})
    df2 = pd.DataFrame({"person": ["alice", "carol"]})
    client = CountingClient({"age": [30, 40]})
    save_dir = str(tmp_path / "p")
    run(
        poll(
            df1,
            "person",
            save_dir=save_dir,
            client=client,
            questions=["age"],
            n_questions_per_run=8,
        )
    )
    calls_after_first = client.calls
    run(
        poll(
            df2,
            "person",
            save_dir=save_dir,
            client=client,
            questions=["age"],
            n_questions_per_run=8,
        )
    )
    assert client.calls > calls_after_first


def test_poll_reset_files_reruns(tmp_path):
    from leximeter import poll

    df = pd.DataFrame({"person": ["alice", "bob"]})
    client = CountingClient({"age": [30, 40]})
    save_dir = str(tmp_path / "p")
    run(
        poll(
            df,
            "person",
            save_dir=save_dir,
            client=client,
            questions=["age"],
            n_questions_per_run=8,
        )
    )
    calls_after_first = client.calls
    run(
        poll(
            df,
            "person",
            save_dir=save_dir,
            client=client,
            questions=["age"],
            n_questions_per_run=8,
            reset_files=True,
        )
    )
    assert client.calls > calls_after_first


# ---------------------------------------------------------------------------
# Debias: completed-run verification (rebuilds DebiasResult from disk)
# ---------------------------------------------------------------------------


class CountingDebiasClient(DebiasClient):
    """DebiasClient that also counts completions."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def complete(self, prompt, *, model, json_mode=False, **kwargs):
        self.calls += 1
        return await super().complete(
            prompt, model=model, json_mode=json_mode, **kwargs
        )


def _debias_df():
    return pd.DataFrame(
        {
            "text": [
                "Brand Alpha is great. CEO John leads the firm. Revenue is up.",
                "Brand Beta shines. CEO Mary runs the show. Costs are low.",
                "Brand Gamma excels. CEO Alex drives the plan. Growth is fast.",
                "Brand Delta grows. CEO Sam owns the market. Margins are high.",
            ]
        }
    )


def test_debias_completed_verification_zero_calls(tmp_path):
    from leximeter import debias

    df = _debias_df()
    client = CountingDebiasClient()
    save_dir = str(tmp_path / "db")
    first = run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            client=client,
            verbose=False,
        )
    )
    calls_after_first = client.calls
    assert calls_after_first > 0
    second = run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            client=client,
            verbose=False,
        )
    )
    assert client.calls == calls_after_first  # zero new completions
    assert list(second.results["brand"]) == list(first.results["brand"])
    assert (
        second.regression["stripped_100pct"].stage1_delta
        == first.regression["stripped_100pct"].stage1_delta
    )


def test_debias_signature_invalidation_regenerates(tmp_path):
    from leximeter import debias

    df = _debias_df()
    client = CountingDebiasClient()
    save_dir = str(tmp_path / "db")
    run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            client=client,
            verbose=False,
        )
    )
    calls_after_first = client.calls
    # Different measurement attribute -> different signature -> full rerun.
    run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            attributes={"quality": "overall quality"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            client=client,
            verbose=False,
        )
    )
    assert client.calls > calls_after_first


def test_debias_reset_files_reruns(tmp_path):
    from leximeter import debias

    df = _debias_df()
    client = CountingDebiasClient()
    save_dir = str(tmp_path / "db")
    run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            client=client,
            verbose=False,
        )
    )
    calls_after_first = client.calls
    run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            client=client,
            verbose=False,
            reset_files=True,
        )
    )
    assert client.calls > calls_after_first


def test_debias_signature_covers_non_measured_columns(tmp_path):
    # The result embeds every input column, so changing any of them must
    # invalidate the cached final result (AC3).  Sub-stage checkpoints are
    # content-addressed on the measured text, so the rerun legitimately
    # makes no new provider calls — but the returned frame must carry the
    # new column, which a stale cached load would not.
    from leximeter import debias

    df = _debias_df()
    client = CountingDebiasClient()
    save_dir = str(tmp_path / "db")
    run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            client=client,
            verbose=False,
        )
    )
    calls_after_first = client.calls
    changed = df.copy()
    changed["notes"] = ["a", "b", "c", "d"]  # column not involved in the pipeline
    second = run(
        debias(
            changed,
            "text",
            save_dir=save_dir,
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            client=client,
            verbose=False,
        )
    )
    assert "notes" in second.results.columns  # fresh result, not the cache
    assert client.calls == calls_after_first  # sub-checkpoints absorbed it


# ---------------------------------------------------------------------------
# Discover: completed-run verification + stage resume
# ---------------------------------------------------------------------------


class CountingStageClient(StageClient):
    """StageClient that also counts completions."""

    def __init__(self, payloads):
        super().__init__(payloads)
        self.calls = 0

    async def complete(self, prompt, *, model, json_mode=False, **kwargs):
        self.calls += 1
        return await super().complete(
            prompt, model=model, json_mode=json_mode, **kwargs
        )


def _discover_payloads():
    return {
        "compare": {"quality": "circle has more quality than square"},
        "bucket": {"quality": "a distinguishing feature"},
        "classify": {"quality": True, "quality (inverted)": False},
        "codify": {},
    }


def _discover_pair_df():
    return pd.DataFrame({"circle": ["sunny", "rainy"], "square": ["cloudy", "clear"]})


def test_discover_completed_verification_zero_calls(tmp_path):
    from leximeter import discover

    df = _discover_pair_df()
    client = CountingStageClient(_discover_payloads())
    save_dir = str(tmp_path / "d")
    first = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            client=client,
            bucket_count=1,
        )
    )
    calls_after_first = client.calls
    assert calls_after_first > 0
    second = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            client=client,
            bucket_count=1,
        )
    )
    assert client.calls == calls_after_first  # zero new completions
    assert second["buckets"] == first["buckets"]
    assert list(second["summary"]["net_pct"]) == list(first["summary"]["net_pct"])
    assert len(second["classification"]) == len(first["classification"])


def test_discover_resumes_stages_without_redoing_completed_work(tmp_path):
    # Simulate an interrupted run: the task manifest + final snapshot are
    # gone, but the sub-stage checkpoints (compare / bucket / classify)
    # survived.  Rerunning must reuse them: zero new completions.
    from leximeter import discover

    df = _discover_pair_df()
    client = CountingStageClient(_discover_payloads())
    save_dir = str(tmp_path / "d")
    first = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            client=client,
            bucket_count=1,
        )
    )
    calls_after_first = client.calls
    assert calls_after_first > 0
    # "interrupted before the final save"
    (tmp_path / "d" / "discover_state.json").unlink()
    (tmp_path / "d" / "discover_results_snapshot.json").unlink()
    second = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            client=client,
            bucket_count=1,
        )
    )
    assert client.calls == calls_after_first  # completed stages were reused
    # full result parity with the first run, not just the buckets
    assert second["buckets"] == first["buckets"]
    assert list(second["summary"]["net_pct"]) == list(first["summary"]["net_pct"])
    assert list(second["summary"]["label"]) == list(first["summary"]["label"])
    assert len(second["classification"]) == len(first["classification"])
    assert list(second["classification"].columns) == list(
        first["classification"].columns
    )
    pd.testing.assert_frame_equal(second["compare"], first["compare"])


def test_discover_signature_invalidation_regenerates(tmp_path):
    from leximeter import discover

    df = _discover_pair_df()
    client = CountingStageClient(_discover_payloads())
    save_dir = str(tmp_path / "d")
    run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            client=client,
            bucket_count=1,
        )
    )
    calls_after_first = client.calls
    changed = _discover_pair_df()
    changed["circle"] = ["sunny", "foggy"]  # one cell changed
    run(
        discover(
            changed,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            client=client,
            bucket_count=1,
        )
    )
    assert client.calls > calls_after_first


def test_discover_reset_files_reruns(tmp_path):
    from leximeter import discover

    df = _discover_pair_df()
    client = CountingStageClient(_discover_payloads())
    save_dir = str(tmp_path / "d")
    run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            client=client,
            bucket_count=1,
        )
    )
    calls_after_first = client.calls
    run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            client=client,
            bucket_count=1,
            reset_files=True,
        )
    )
    assert client.calls > calls_after_first


# ---------------------------------------------------------------------------
# Codify / Compare: sub-task idempotency
# ---------------------------------------------------------------------------


def test_codify_dynamic_short_circuit_zero_calls(tmp_path):
    from leximeter import codify

    df = pd.DataFrame({"text": ["hello world"]})
    client = CountingClient(
        {"code1": [{"beginning excerpt": "hello world", "ending excerpt": ""}]}
    )
    save_dir = str(tmp_path / "c")
    first = run(
        codify(
            df,
            "text",
            categories=None,
            additional_instructions="code passages",
            save_dir=save_dir,
            client=client,
        )
    )
    assert first["coded_passages"].iloc[0]["code1"] == ["hello world"]
    calls_after_first = client.calls
    assert calls_after_first > 0
    second = run(
        codify(
            df,
            "text",
            categories=None,
            additional_instructions="code passages",
            save_dir=save_dir,
            client=client,
        )
    )
    assert client.calls == calls_after_first  # zero new completions
    # list values survive the CSV round-trip
    assert second["coded_passages"].iloc[0]["code1"] == ["hello world"]


def test_codify_static_short_circuit_restores_lists(tmp_path):
    from leximeter import codify

    df = pd.DataFrame({"text": ["hello world"]})
    client = CountingClient(
        {"code1": [{"beginning excerpt": "hello world", "ending excerpt": ""}]}
    )
    save_dir = str(tmp_path / "c")
    first = run(
        codify(
            df,
            "text",
            categories={"code1": "def"},
            save_dir=save_dir,
            client=client,
        )
    )
    assert first["code1"].iloc[0] == ["hello world"]
    calls_after_first = client.calls
    second = run(
        codify(
            df,
            "text",
            categories={"code1": "def"},
            save_dir=save_dir,
            client=client,
        )
    )
    assert client.calls == calls_after_first
    assert second["code1"].iloc[0] == ["hello world"]


def test_codify_reset_files_reruns(tmp_path):
    from leximeter import codify

    df = pd.DataFrame({"text": ["hello world"]})
    client = CountingClient(
        {"code1": [{"beginning excerpt": "hello world", "ending excerpt": ""}]}
    )
    save_dir = str(tmp_path / "c")
    run(
        codify(
            df,
            "text",
            categories={"code1": "def"},
            save_dir=save_dir,
            client=client,
        )
    )
    calls_after_first = client.calls
    run(
        codify(
            df,
            "text",
            categories={"code1": "def"},
            save_dir=save_dir,
            client=client,
            reset_files=True,
        )
    )
    assert client.calls > calls_after_first


def test_codify_signature_invalidation_purges_checkpoints(tmp_path):
    from leximeter import codify

    df = pd.DataFrame({"text": ["hello world"]})
    client = CountingClient(
        {"code1": [{"beginning excerpt": "hello world", "ending excerpt": ""}]}
    )
    save_dir = str(tmp_path / "c")
    run(
        codify(
            df,
            "text",
            categories={"code1": "def"},
            save_dir=save_dir,
            client=client,
        )
    )
    calls_after_first = client.calls
    # Same row count but different content: positional identifiers would
    # otherwise reuse stale completions — the purge must prevent that.
    changed = pd.DataFrame({"text": ["goodbye moon"]})
    run(
        codify(
            changed,
            "text",
            categories={"code1": "def"},
            save_dir=save_dir,
            client=client,
        )
    )
    assert client.calls > calls_after_first


def test_compare_short_circuit_zero_calls(tmp_path):
    from leximeter import compare

    df = pd.DataFrame({"circle": ["sunny", "rainy"], "square": ["cloudy", "clear"]})
    client = CountingClient({"quality": "circle has more quality than square"})
    save_dir = str(tmp_path / "cmp")
    first = run(
        compare(
            df,
            "circle",
            "square",
            save_dir=save_dir,
            client=client,
        )
    )
    assert len(first) > 0
    calls_after_first = client.calls
    assert calls_after_first > 0
    second = run(
        compare(
            df,
            "circle",
            "square",
            save_dir=save_dir,
            client=client,
        )
    )
    assert client.calls == calls_after_first  # zero new completions
    assert len(second) == len(first)


# ---------------------------------------------------------------------------
# Task-level retry rescuing a complex run (TICKET-001 + max_retries wrap)
# ---------------------------------------------------------------------------


class SeedFailClient(CountingClient):
    """Fail seed-stage prompts ``failures`` times, then succeed everywhere."""

    def __init__(self, payload: dict, failures: int = 4):
        super().__init__(payload)
        self.failures = failures

    async def complete(self, prompt, *, model, json_mode=False, **kwargs):
        if "Generate representative" in prompt and self.failures > 0:
            self.failures -= 1
            raise RuntimeError("seed provider down")
        return await super().complete(
            prompt, model=model, json_mode=json_mode, **kwargs
        )


def test_poll_task_level_retry_rescues_seed_failure(tmp_path):
    # The seed stage's single call fails (no per-batch retries — rolled
    # back per worksummary_0816), raising through poll's run.  The
    # task-level max_retries wrapper reruns the pipeline; the seed
    # regenerates on the third attempt, personas/answers would resume.
    from leximeter import poll

    client = SeedFailClient({"age": [30, 40]}, failures=2)
    save_dir = str(tmp_path / "p")
    res = run(
        poll(
            save_dir=save_dir,
            client=client,
            population_description="urban professionals",
            questions=["age"],
            n_questions_per_run=8,
            num_personas=2,
            entities_per_generation=2,
            max_retries=2,
        )
    )
    assert len(res) == 2
    assert client.failures == 0  # all seed failures were absorbed
    # after the rescued run, the completed verification kicks in
    calls_after_rescue = client.calls
    run(
        poll(
            save_dir=save_dir,
            client=client,
            population_description="urban professionals",
            questions=["age"],
            n_questions_per_run=8,
            num_personas=2,
            entities_per_generation=2,
        )
    )
    assert client.calls == calls_after_rescue  # zero new completions
