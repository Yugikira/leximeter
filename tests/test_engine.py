"""Engine-level protections: bounded loops and adaptive KV-cache warmup."""

from __future__ import annotations

import asyncio
import json

import pandas as pd
from conftest import DictClient

from leximeter import poll, seed
from leximeter.client.base import LLMClient
from leximeter.engine import batch as batch_mod
from leximeter.engine import checkpoint as checkpoint_mod
from leximeter.engine.batch import batch_execute


def run(coro):
    return asyncio.run(coro)


def test_seed_generation_loop_has_budget(tmp_path):
    """A provider that keeps returning partial results must not loop forever."""
    client = DictClient({"entity 1": "only one"})  # always yields 1 entity
    res = run(
        seed(
            "generate people",
            save_dir=str(tmp_path / "seed"),
            client=client,
            num_entities=8,
            entities_per_generation=4,
        )
    )
    # budget = ceil(8/4) + 2 = 4 batches, then give up gracefully
    assert len(res) == 4
    assert "entity_id" in res.columns


# ---------------------------------------------------------------------------
# Issue #12: Adaptive 1-call KV-cache warmup in batch_execute
# ---------------------------------------------------------------------------


class ConcurrencyProbeClient(LLMClient):
    """Fake async LLMClient recording start/end ordering and peak concurrency."""

    def __init__(
        self,
        *,
        require_wave1_concurrent: bool = False,
        fail_prompts: set[str] | None = None,
        response_prefix: str = "ok",
    ) -> None:
        self.events: list[tuple[str, str]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.max_in_flight_after_first = 0
        self._first_done = False
        self._require_wave1_concurrent = require_wave1_concurrent
        self._wave1_barrier = asyncio.Event()
        self._fail_prompts = set(fail_prompts or ())
        self._response_prefix = response_prefix

    async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
        assert "warmup_first" not in kwargs
        self.events.append(("start", prompt))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self._first_done:
            self.max_in_flight_after_first = max(
                self.max_in_flight_after_first, self.in_flight
            )
        try:
            if self._require_wave1_concurrent:
                if self.in_flight >= 2:
                    self._wave1_barrier.set()
                await asyncio.wait_for(self._wave1_barrier.wait(), timeout=1.0)
            else:
                # Yield enough loop turns for any concurrently scheduled tasks to enter
                for _ in range(5):
                    await asyncio.sleep(0)
            if prompt in self._fail_prompts:
                raise RuntimeError(f"boom:{prompt}")
            return f"{self._response_prefix}:{prompt}"
        finally:
            self.in_flight -= 1
            if not self._first_done:
                self._first_done = True
            self.events.append(("end", prompt))

    def estimate_tokens(self, prompt, *, model) -> int:
        return 0

    def estimate_cost(self, prompt, *, model) -> float:
        return 0.0


def test_batch_execute_auto_warmup_activates_above_threshold():
    # len(pending)=9 > max(8, n_parallels=4) -> warmup first item, then fan out concurrently
    prompts = [f"p{i}" for i in range(9)]
    ids = [f"id{i}" for i in range(9)]
    client = ConcurrencyProbeClient()

    df = run(
        batch_execute(
            prompts,
            ids,
            client=client,
            model="test-model",
            n_parallels=4,
        )
    )

    assert len(df) == 9
    assert df["Successful"].all()
    assert client.events[0] == ("start", "p0")
    assert client.events[1] == ("end", "p0")
    assert client.max_in_flight_after_first >= 2

    # len(pending)=11 > max(8, n_parallels=10) -> warmup also activates when n_parallels > 8
    prompts_11 = [f"q{i}" for i in range(11)]
    ids_11 = [f"qid{i}" for i in range(11)]
    client_11 = ConcurrencyProbeClient()
    df_11 = run(
        batch_execute(
            prompts_11,
            ids_11,
            client=client_11,
            model="test-model",
            n_parallels=10,
        )
    )
    assert len(df_11) == 11
    assert client_11.events[0] == ("start", "q0")
    assert client_11.events[1] == ("end", "q0")
    assert client_11.max_in_flight_after_first >= 2


def test_batch_execute_auto_warmup_skipped_at_or_below_threshold():
    # len(pending)=8 <= max(8, n_parallels=4) -> wave 1 starts concurrently immediately
    prompts = [f"p{i}" for i in range(8)]
    ids = [f"id{i}" for i in range(8)]
    client = ConcurrencyProbeClient(require_wave1_concurrent=True)

    df = run(
        batch_execute(
            prompts,
            ids,
            client=client,
            model="test-model",
            n_parallels=4,
        )
    )

    assert len(df) == 8
    assert df["Successful"].all()
    assert client.max_in_flight >= 2

    # len(pending)=10 <= max(8, n_parallels=10) -> wave 1 starts concurrently immediately
    prompts_10 = [f"q{i}" for i in range(10)]
    ids_10 = [f"qid{i}" for i in range(10)]
    client_10 = ConcurrencyProbeClient(require_wave1_concurrent=True)
    df_10 = run(
        batch_execute(
            prompts_10,
            ids_10,
            client=client_10,
            model="test-model",
            n_parallels=10,
        )
    )
    assert len(df_10) == 10
    assert client_10.max_in_flight >= 2


def test_batch_execute_explicit_warmup_true_and_false():
    # warmup_first=True forces warmup on a small batch (len(pending)=3 > 1)
    client_true = ConcurrencyProbeClient()
    df_true = run(
        batch_execute(
            ["p0", "p1", "p2"],
            ["id0", "id1", "id2"],
            client=client_true,
            model="test-model",
            n_parallels=10,
            warmup_first=True,
        )
    )
    assert len(df_true) == 3
    assert client_true.events[0] == ("start", "p0")
    assert client_true.events[1] == ("end", "p0")
    assert client_true.max_in_flight_after_first == 2

    # warmup_first=True with single pending item runs cleanly once
    client_one = ConcurrencyProbeClient()
    df_one = run(
        batch_execute(
            ["only"],
            ["id_only"],
            client=client_one,
            model="test-model",
            n_parallels=10,
            warmup_first=True,
        )
    )
    assert list(df_one["Identifier"]) == ["id_only"]
    assert client_one.events == [("start", "only"), ("end", "only")]

    # warmup_first=False disables warmup even when len(pending)=12 > max(8, n_parallels=4)
    client_false = ConcurrencyProbeClient(require_wave1_concurrent=True)
    df_false = run(
        batch_execute(
            [f"p{i}" for i in range(12)],
            [f"id{i}" for i in range(12)],
            client=client_false,
            model="test-model",
            n_parallels=4,
            warmup_first=False,
        )
    )
    assert len(df_false) == 12
    assert df_false["Successful"].all()
    assert client_false.max_in_flight >= 2


def test_batch_execute_failed_warmup_records_failure_and_runs_rest():
    client = ConcurrencyProbeClient(fail_prompts={"p0"})
    df = run(
        batch_execute(
            ["p0", "p1", "p2", "p3"],
            ["id0", "id1", "id2", "id3"],
            client=client,
            model="test-model",
            n_parallels=4,
            max_retries=0,
            warmup_first=True,
        )
    )

    # p0 failed before p1..p3 started, and p1..p3 all still executed and succeeded
    assert client.events[0] == ("start", "p0")
    assert client.events[1] == ("end", "p0")
    by_id = df.set_index("Identifier")
    assert bool(by_id.loc["id0", "Successful"]) is False
    assert "boom:p0" in str(by_id.loc["id0", "Error Log"])
    for ident in ("id1", "id2", "id3"):
        assert bool(by_id.loc[ident, "Successful"]) is True
        assert by_id.loc[ident, "Response"] == f"ok:p{ident[-1]}"


def test_batch_execute_warmup_respects_checkpoint_and_reset_files(tmp_path):
    save_path = str(tmp_path / "ckpt.csv")

    # Seed checkpoint with id0 and id1 already completed (v1)
    run(
        batch_execute(
            ["p0", "p1"],
            ["id0", "id1"],
            client=ConcurrencyProbeClient(response_prefix="v1"),
            model="test-model",
            save_path=save_path,
            warmup_first=False,
        )
    )

    # Resume with 4 items and warmup_first=True: id0/id1 skipped, id2 warms up before id3
    resume_client = ConcurrencyProbeClient(response_prefix="v1")
    resumed = run(
        batch_execute(
            ["p0", "p1", "p2", "p3"],
            ["id0", "id1", "id2", "id3"],
            client=resume_client,
            model="test-model",
            save_path=save_path,
            reset_files=False,
            warmup_first=True,
        )
    )
    called_prompts = [p for ev, p in resume_client.events if ev == "start"]
    assert called_prompts == ["p2", "p3"]
    assert resume_client.events[0] == ("start", "p2")
    assert resume_client.events[1] == ("end", "p2")
    assert list(resumed["Identifier"]) == ["id0", "id1", "id2", "id3"]
    disk_df = pd.read_csv(save_path)
    assert list(disk_df["Identifier"]) == ["id0", "id1", "id2", "id3"]

    # reset_files=True reruns all 4 items with id0 warming up, appending fresh rows
    # (never deleting prior batches on disk) without duplicating the warmed-up row,
    # and load_checkpoint(..., keep="last") cleanly supersedes prior responses.
    reset_client = ConcurrencyProbeClient(response_prefix="v2")
    rerun = run(
        batch_execute(
            ["p0", "p1", "p2", "p3"],
            ["id0", "id1", "id2", "id3"],
            client=reset_client,
            model="test-model",
            save_path=save_path,
            reset_files=True,
            warmup_first=True,
        )
    )
    assert reset_client.events[0] == ("start", "p0")
    assert reset_client.events[1] == ("end", "p0")
    assert sorted(rerun["Identifier"]) == ["id0", "id1", "id2", "id3"]
    assert len(rerun) == 4
    assert all(str(r).startswith("v2:") for r in rerun["Response"])

    # Exactly 4 fresh rows appended (4 prior + 4 fresh = 8; no duplicate warmup row)
    disk_after_reset = pd.read_csv(save_path)
    assert len(disk_after_reset) == 8
    assert dict(disk_after_reset["Identifier"].value_counts()) == {
        "id0": 2,
        "id1": 2,
        "id2": 2,
        "id3": 2,
    }

    # load_checkpoint supersedes old v1 rows with the fresh v2 rows
    loaded = checkpoint_mod.load_checkpoint(save_path).set_index("Identifier")
    assert len(loaded) == 4
    for i in range(4):
        assert loaded.loc[f"id{i}", "Response"] == f"v2:p{i}"

    # Subsequent resume makes zero provider calls and returns the fresh v2 responses
    noop_client = ConcurrencyProbeClient(response_prefix="v3")
    after_resume = run(
        batch_execute(
            ["p0", "p1", "p2", "p3"],
            ["id0", "id1", "id2", "id3"],
            client=noop_client,
            model="test-model",
            save_path=save_path,
            reset_files=False,
            warmup_first=True,
        )
    )
    assert noop_client.events == []
    by_id = after_resume.set_index("Identifier")
    for i in range(4):
        assert by_id.loc[f"id{i}", "Response"] == f"v2:p{i}"


def test_batch_execute_warmup_incremental_and_final_flush_non_duplicating(
    tmp_path, monkeypatch
):
    """Warmup + incremental flush + final flush writes every row strictly once."""
    monkeypatch.setattr(batch_mod, "_SAVE_EVERY", 2)
    save_path = str(tmp_path / "flush.csv")
    prompts = [f"p{i}" for i in range(5)]
    ids = [f"id{i}" for i in range(5)]

    client = ConcurrencyProbeClient()
    df = run(
        batch_execute(
            prompts,
            ids,
            client=client,
            model="test-model",
            n_parallels=4,
            save_path=save_path,
            warmup_first=True,
        )
    )
    assert len(df) == 5
    disk_df = pd.read_csv(save_path)
    assert len(disk_df) == 5
    assert sorted(disk_df["Identifier"]) == ids


def test_poll_reset_files_preserves_all_question_batches_in_answers_raw(tmp_path):
    """Multi-batch Poll with reset_files=True must keep all batches in answers_raw.csv."""

    class MultiBatchPollClient(LLMClient):
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
            self.calls += 1
            if "BEGIN SURVEY QUESTIONS" in prompt:
                payload = {}
                for q in ("q1", "q2", "q3", "q4"):
                    if f"- {q}" in prompt:
                        payload[q] = [[f"ans_{q}", 1.0]]
                return json.dumps(payload)
            return "Synthetic persona bio."

        def estimate_tokens(self, prompt, *, model) -> int:
            return 0

        def estimate_cost(self, prompt, *, model) -> float:
            return 0.0

    save_dir = tmp_path / "poll_multi"
    df = pd.DataFrame({"seed": ["Alice", "Bob"]})
    questions = ["q1", "q2", "q3", "q4"]

    client = MultiBatchPollClient()
    res = run(
        poll(
            df,
            "seed",
            questions=questions,
            n_questions_per_run=2,
            save_dir=str(save_dir),
            client=client,
            reset_files=True,
        )
    )
    assert list(res.columns) == ["seed", "persona", "q1", "q2", "q3", "q4"]
    for q in questions:
        assert list(res[q]) == [f"ans_{q}", f"ans_{q}"]

    answers_path = save_dir / "answers_raw.csv"
    answers_df = pd.read_csv(answers_path)
    # 2 seeds x 2 question batches = 4 distinct answer identifiers preserved on disk
    assert len(answers_df) == 4
    assert answers_df["Identifier"].nunique() == 4

    # Simulate resuming after interruption before poll_state.json was written:
    # all 4 cached answers across both question batches must be reused (0 calls).
    (save_dir / "poll_state.json").unlink()
    resume_client = MultiBatchPollClient()
    resumed = run(
        poll(
            df,
            "seed",
            questions=questions,
            n_questions_per_run=2,
            save_dir=str(save_dir),
            client=resume_client,
            reset_files=False,
        )
    )
    assert resume_client.calls == 0
    for q in questions:
        assert list(resumed[q]) == [f"ans_{q}", f"ans_{q}"]
