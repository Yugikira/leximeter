"""Public API / task-seam tests for filter with DecisionClient.

Tests inject fake DecisionClient implementations into ``leximeter.filter``
and assert only external behavior: inverted fan-out (condition in ``state``,
unique candidate items as chunked parallel ``noul`` questions), calibrated
``score`` and ``keep`` columns, deduplication, refusal degradation,
checkpoint resume with zero duplicate calls, and fallback to LLMClient when
unconfigured.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import pytest
from conftest import CountingClient, CountingDecisionClient, DictDecisionClient

from leximeter import filter as leximeter_filter
from leximeter.client import (
    DecisionClient,
    DecisionResult,
    DummyDecisionClient,
    NoulAnswer,
    NoulQuestion,
    SystemOneClient,
)
from leximeter.tasks.filter import Filter, FilterConfig


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_filter_without_decision_client_uses_llm_path_unchanged(
    tmp_path, monkeypatch
) -> None:
    """When no Decision Provider is configured, filter executes the LLMClient path."""
    monkeypatch.delenv("LEXIMETER_DECISION_PROTOCOL", raising=False)
    df = pd.DataFrame({"tech": ["iPhone", "iPad", "banana", "F-16"]})
    llm = CountingClient({"entities meeting condition": ["iPhone", "iPad"]})

    res = run(
        leximeter_filter(
            df,
            "tech",
            condition="is a consumer electronic device",
            save_dir=str(tmp_path / "llm_fallback"),
            client=llm,
            entities_per_call=2,
            shuffle=False,
        )
    )

    assert llm.calls == 2
    assert list(res.columns) == ["tech", "keep", "score"]
    assert res["keep"].dtype == bool
    assert list(res["keep"]) == [True, True, False, False]
    assert list(res["score"]) == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_filter_inverted_fan_out_noul_chunking_and_state(tmp_path) -> None:
    """DecisionClient path puts condition + additional_instructions in state and chunks items as noul questions."""
    df = pd.DataFrame({"item": ["iPhone", "banana", "iPad", "cactus", "MacBook"]})
    probs = {
        "iPhone": 0.92,
        "banana": 0.05,
        "iPad": 0.85,
        "cactus": 0.12,
        "MacBook": 0.50,
    }

    def handler(state: Any, key: str, question: Any) -> float:
        assert isinstance(state, dict)
        assert state["condition"] == "is a consumer electronic device"
        assert state["additional_instructions"] == "Include laptops and phones."
        assert isinstance(question, NoulQuestion)
        assert key in str(question.instructions)
        return probs[key]

    decision_client = CountingDecisionClient(
        handler=handler,
        max_questions_per_call=2,
    )
    llm_unused = CountingClient({"entities meeting condition": []})

    res = run(
        leximeter_filter(
            df,
            "item",
            condition="is a consumer electronic device",
            additional_instructions="Include laptops and phones.",
            save_dir=str(tmp_path / "flt_decision"),
            entities_per_call=150,
            threshold=0.5,
            decision_model="cloudflare/clef-flash",
            client=llm_unused,
            decision_client=decision_client,
        )
    )

    # Generative LLMClient is never touched
    assert llm_unused.calls == 0
    # 5 unique items chunked by min(150, 2) = 2 -> 3 calls of sizes [2, 2, 1]
    assert decision_client.calls == 3
    assert sorted(len(qs) for _, qs, _ in decision_client.requests) == [1, 2, 2]
    assert all(m == "cloudflare/clef-flash" for _, _, m in decision_client.requests)

    assert list(res.columns) == ["item", "keep", "score"]
    assert res["keep"].dtype == bool
    assert list(res["score"]) == pytest.approx([0.92, 0.05, 0.85, 0.12, 0.50])
    assert list(res["keep"]) == [True, False, True, False, True]


def test_filter_multi_run_averages_calibrated_probabilities(tmp_path) -> None:
    """When n_runs > 1, filter averages calibrated noul probabilities across runs before thresholding."""
    df = pd.DataFrame({"item": ["borderline_pass", "borderline_fail"]})
    # Run 1 -> {pass: 0.80, fail: 0.60}; Run 2 -> {pass: 0.40, fail: 0.30}
    # Means: pass = 0.60 >= 0.6 -> True; fail = 0.45 < 0.6 -> False
    per_item_runs: dict[str, list[float]] = {
        "borderline_pass": [0.80, 0.40],
        "borderline_fail": [0.60, 0.30],
    }
    seen_counts: dict[str, int] = {"borderline_pass": 0, "borderline_fail": 0}

    def handler(state: Any, key: str, question: Any) -> float:
        assert state == {"condition": "meets quality bar"}
        assert "additional_instructions" not in state
        idx = seen_counts[key]
        seen_counts[key] += 1
        return per_item_runs[key][idx]

    decision_client = CountingDecisionClient(handler=handler)
    res = run(
        leximeter_filter(
            df,
            "item",
            condition="meets quality bar",
            additional_instructions="   ",
            save_dir=str(tmp_path / "flt_multirun"),
            n_runs=2,
            threshold=0.6,
            decision_client=decision_client,
        )
    )

    assert decision_client.calls == 2
    assert list(res["score"]) == pytest.approx([0.60, 0.45])
    assert list(res["keep"]) == [True, False]


def test_filter_respects_entities_per_call_below_provider_cap_and_env_model(
    tmp_path, monkeypatch
) -> None:
    """entities_per_call caps chunk size when smaller than max_questions_per_call; env model resolves cleanly."""
    monkeypatch.setenv("LEXIMETER_DECISION_MODEL", "typesafe/jev-1.14")
    df = pd.DataFrame({"item": ["a", "b", "c"]})
    decision_client = CountingDecisionClient(
        {"a": 0.9, "b": 0.2, "c": 0.7},
        max_questions_per_call=64,
    )

    res = run(
        leximeter_filter(
            df,
            "item",
            condition="passes check",
            save_dir=str(tmp_path / "flt_epc"),
            entities_per_call=1,
            threshold=0.5,
            decision_client=decision_client,
        )
    )

    assert decision_client.calls == 3
    assert all(len(qs) == 1 for _, qs, _ in decision_client.requests)
    assert all(m == "typesafe/jev-1.14" for _, _, m in decision_client.requests)
    assert list(res["keep"]) == [True, False, True]
    assert list(res["score"]) == pytest.approx([0.9, 0.2, 0.7])


def test_filter_deduplicates_items_and_preserves_input_rows_and_index(tmp_path) -> None:
    """Duplicate items are evaluated once and mapped back cleanly onto every input row."""
    df = pd.DataFrame(
        {"item": ["alpha", "beta", "alpha", "gamma", "beta", "alpha"]},
        index=[10, 20, 30, 40, 50, 60],
    )
    evaluated_keys: list[str] = []

    def handler(state: Any, key: str, question: Any) -> float:
        evaluated_keys.append(key)
        return {"alpha": 0.91, "beta": 0.15, "gamma": 0.72}[key]

    decision_client = CountingDecisionClient(handler=handler, max_questions_per_call=64)
    res = run(
        leximeter_filter(
            df,
            "item",
            condition="is target item",
            save_dir=str(tmp_path / "flt_dedup"),
            shuffle=True,
            random_seed=42,
            threshold=0.5,
            decision_client=decision_client,
        )
    )

    assert decision_client.calls == 1
    assert sorted(evaluated_keys) == ["alpha", "beta", "gamma"]
    assert list(res.index) == [10, 20, 30, 40, 50, 60]
    assert list(res["item"]) == ["alpha", "beta", "alpha", "gamma", "beta", "alpha"]
    assert list(res["score"]) == pytest.approx([0.91, 0.15, 0.91, 0.72, 0.15, 0.91])
    assert list(res["keep"]) == [True, False, True, True, False, True]


def test_filter_refused_questions_degrade_cleanly_without_crashing_batch(
    tmp_path,
) -> None:
    """Refused questions yield NaN score and keep=False while sibling items succeed."""
    df = pd.DataFrame({"item": ["allowed_yes", "refused_item", "allowed_no"]})
    decision_client = DictDecisionClient(
        {
            "allowed_yes": 0.88,
            "refused_item": NoulAnswer(probability=None, confidence=None, refused=True),
            "allowed_no": 0.12,
        }
    )

    res = run(
        leximeter_filter(
            df,
            "item",
            condition="matches policy",
            save_dir=str(tmp_path / "flt_refusal"),
            threshold=0.0,  # even at threshold=0.0 a refused item must NOT be kept
            decision_client=decision_client,
        )
    )

    assert res["keep"].dtype == bool
    assert list(res["keep"]) == [True, False, True]
    assert res["score"].iloc[0] == pytest.approx(0.88)
    assert pd.isna(res["score"].iloc[1])
    assert res["score"].iloc[2] == pytest.approx(0.12)
    # Boolean masking with res["keep"] works cleanly without NA errors
    assert list(res[res["keep"]]["item"]) == ["allowed_yes", "allowed_no"]


def test_filter_partial_refusal_across_runs_averages_valid_runs(tmp_path) -> None:
    """If one run refuses an item and another succeeds, score averages the valid run(s)."""
    df = pd.DataFrame({"item": ["flaky_item"]})
    responses: list[NoulAnswer] = [
        NoulAnswer(probability=None, confidence=None, refused=True),
        NoulAnswer(probability=0.80, confidence=0.60, refused=False),
    ]
    idx = 0

    def handler(state: Any, key: str, question: Any) -> NoulAnswer:
        nonlocal idx
        ans = responses[idx]
        idx += 1
        return ans

    decision_client = CountingDecisionClient(handler=handler)
    res = run(
        leximeter_filter(
            df,
            "item",
            condition="is relevant",
            save_dir=str(tmp_path / "flt_partial_refusal"),
            n_runs=2,
            threshold=0.5,
            decision_client=decision_client,
        )
    )

    assert decision_client.calls == 2
    assert res["score"].iloc[0] == pytest.approx(0.80)
    assert bool(res["keep"].iloc[0]) is True


def test_filter_empty_dataframe_makes_zero_calls(tmp_path) -> None:
    """Empty input DataFrame preserves keep/score columns and makes zero DecisionClient calls."""
    df = pd.DataFrame({"item": pd.Series([], dtype="object")})
    decision_client = CountingDecisionClient({"x": 0.9})

    res = run(
        leximeter_filter(
            df,
            "item",
            condition="any condition",
            save_dir=str(tmp_path / "flt_empty"),
            decision_client=decision_client,
        )
    )

    assert decision_client.calls == 0
    assert list(res.columns) == ["item", "keep", "score"]
    assert len(res) == 0
    assert res["keep"].dtype == bool
    assert res["score"].dtype == float


def test_filter_checkpoint_resume_completed_interrupted_and_reset(tmp_path) -> None:
    """Resuming interrupted or completed filter skips checkpointed batches with zero duplicate calls."""
    df = pd.DataFrame({"item": ["item_a", "item_b", "item_c", "item_d"]})
    save_dir = str(tmp_path / "flt_resume")

    class FailOnceOnSecondBatch(DecisionClient):
        max_questions_per_call = 2

        def __init__(self) -> None:
            self.seen_batches: list[list[str]] = []
            self.fail_on_c = True

        async def decide(self, state, questions, *, model, **kwargs):
            del state, kwargs
            batch_keys = list(questions.keys())
            self.seen_batches.append(batch_keys)
            if "item_c" in batch_keys and self.fail_on_c:
                raise RuntimeError("simulated interruption on batch 2")
            probs = {"item_a": 0.90, "item_b": 0.20, "item_c": 0.85, "item_d": 0.10}
            return DecisionResult(
                answers={k: NoulAnswer(probability=probs[k]) for k in batch_keys},
                model=model,
            )

    flaky = FailOnceOnSecondBatch()
    # 1. Interrupted run: batch ["item_a", "item_b"] succeeds; ["item_c", "item_d"] fails
    res_partial = run(
        leximeter_filter(
            df,
            "item",
            condition="passes filter",
            save_dir=save_dir,
            shuffle=False,
            max_retries=0,
            threshold=0.5,
            decision_client=flaky,
        )
    )
    assert flaky.seen_batches == [["item_a", "item_b"], ["item_c", "item_d"]]
    assert res_partial["score"].iloc[0] == pytest.approx(0.90)
    assert res_partial["score"].iloc[1] == pytest.approx(0.20)
    assert pd.isna(res_partial["score"].iloc[2])
    assert pd.isna(res_partial["score"].iloc[3])
    assert list(res_partial["keep"]) == [True, False, False, False]

    # 2. Resume after interruption: ONLY failed batch ["item_c", "item_d"] runs
    flaky.fail_on_c = False
    flaky.seen_batches.clear()
    res_resumed = run(
        leximeter_filter(
            df,
            "item",
            condition="passes filter",
            save_dir=save_dir,
            shuffle=False,
            reset_files=False,
            threshold=0.5,
            decision_client=flaky,
        )
    )
    assert flaky.seen_batches == [["item_c", "item_d"]]
    assert list(res_resumed["score"]) == pytest.approx([0.90, 0.20, 0.85, 0.10])
    assert list(res_resumed["keep"]) == [True, False, True, False]

    # 3. Re-run completed run (even with default higher max_questions_per_call=64): ZERO calls
    counting = CountingDecisionClient(default=0.05, max_questions_per_call=64)
    res_cached = run(
        leximeter_filter(
            df,
            "item",
            condition="passes filter",
            save_dir=save_dir,
            shuffle=False,
            reset_files=False,
            threshold=0.5,
            decision_client=counting,
        )
    )
    assert counting.calls == 0
    assert list(res_cached["score"]) == pytest.approx([0.90, 0.20, 0.85, 0.10])
    assert list(res_cached["keep"]) == [True, False, True, False]

    # 4. reset_files=True clears checkpoint and re-evaluates using current client cap
    res_reset = run(
        leximeter_filter(
            df,
            "item",
            condition="passes filter",
            save_dir=save_dir,
            shuffle=False,
            reset_files=True,
            threshold=0.5,
            decision_client=counting,
        )
    )
    assert counting.calls == 1
    assert list(res_reset["score"]) == pytest.approx([0.05, 0.05, 0.05, 0.05])
    assert list(res_reset["keep"]) == [False, False, False, False]


def test_filter_dummy_decision_client_and_task_class_seam(tmp_path) -> None:
    """DummyDecisionClient works offline through both api.filter and Filter(FilterConfig(...))."""
    df = pd.DataFrame({"entity": ["Solar panel", "Coal plant"]})

    res_api = run(
        leximeter_filter(
            df,
            "entity",
            condition="is renewable energy",
            save_dir=str(tmp_path / "flt_dummy_api"),
            threshold=0.6,
            decision_client=DummyDecisionClient(noul_probability=0.75),
        )
    )
    assert list(res_api["keep"]) == [True, True]
    assert list(res_api["score"]) == pytest.approx([0.75, 0.75])

    cfg = FilterConfig(
        condition="is renewable energy",
        save_dir=str(tmp_path / "flt_dummy_task"),
        threshold=0.6,
    )
    task = Filter(cfg, decision_client=DummyDecisionClient(noul_probability=0.40))
    res_task = run(task.run(df, "entity"))
    assert list(res_task["keep"]) == [False, False]
    assert list(res_task["score"]) == pytest.approx([0.40, 0.40])


def test_filter_end_to_end_systemone_mock_transport_non_ascii_and_refusal(
    tmp_path,
) -> None:
    """End-to-end filter + SystemOneClient sanitizes wire keys and handles per-question refusals."""
    import json

    import httpx

    item_1 = "半导体芯片 (> 5nm)"
    item_2 = "敏感内容: refusal / block"
    item_3 = "Fresh apple, organic"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "jev-latest"
        assert body["state"] == {"condition": "is a semiconductor technology"}
        assert list(body["questions"].keys()) == ["q_0", "q_1", "q_2"]
        assert all(q["type"] == "noul" for q in body["questions"].values())
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14.0",
                "answers": {
                    "q_0": {"type": "noul", "noul": 0.94},
                    "q_1": {"type": "refusal"},
                    "q_2": {"type": "noul", "noul": 0.04},
                },
                "usage": {"input_tokens": 55, "output_tokens": 0},
            },
        )

    client = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    df = pd.DataFrame({"item": [item_1, item_2, item_3, item_1]})
    save_dir = tmp_path / "flt_e2e_sys1"

    res = run(
        leximeter_filter(
            df,
            "item",
            condition="is a semiconductor technology",
            save_dir=str(save_dir),
            shuffle=False,
            threshold=0.5,
            decision_model="jev-latest",
            decision_client=client,
        )
    )

    assert list(res["keep"]) == [True, False, False, True]
    assert res["score"].iloc[0] == pytest.approx(0.94)
    assert pd.isna(res["score"].iloc[1])
    assert res["score"].iloc[2] == pytest.approx(0.04)
    assert res["score"].iloc[3] == pytest.approx(0.94)

    raw_df = pd.read_csv(save_dir / "filter_responses_raw_responses.csv")
    raw_resp = str(raw_df["Response"].iloc[0])
    assert "0.94" in raw_resp
    assert '"refused": true' in raw_resp.lower()


def test_filter_systemone_missing_decision_model_fails_fast_without_disk_mutation(
    tmp_path, monkeypatch
) -> None:
    """SystemOneClient without decision_model / LEXIMETER_DECISION_MODEL fails fast before HTTP or disk resets."""
    import httpx

    http_calls = 0

    def fail_transport(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        raise AssertionError(f"Unexpected HTTP call to {request.url}")

    save_dir = tmp_path / "flt_fail_fast"
    save_dir.mkdir(parents=True, exist_ok=True)
    meta_path = save_dir / "filter_responses_run_metadata.json"
    raw_path = save_dir / "filter_responses_raw_responses.csv"
    meta_sentinel = '{"model": "prev-model", "effective_batch_size": 2}'
    raw_sentinel = "Identifier,Response,Successful,Error Log,Time Taken\n"
    meta_path.write_text(meta_sentinel, encoding="utf-8")
    raw_path.write_text(raw_sentinel, encoding="utf-8")

    df = pd.DataFrame({"item": ["apple", "banana"]})
    sys1 = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=httpx.MockTransport(fail_transport),
    )

    for reset in (False, True):
        for bad_dm in (None, "   "):
            with pytest.raises(ValueError, match="decision_model"):
                run(
                    leximeter_filter(
                        df,
                        "item",
                        condition="is a fruit",
                        save_dir=str(save_dir),
                        model="gpt-5.4-nano",
                        decision_model=bad_dm,
                        decision_client=sys1,
                        reset_files=reset,
                    )
                )

    monkeypatch.setenv("LEXIMETER_DECISION_PROTOCOL", "systemone")
    monkeypatch.setenv("LEXIMETER_DECISION_BASE_URL", "https://api.typesafe.ai")
    monkeypatch.setenv("LEXIMETER_DECISION_MODEL", "   ")
    with pytest.raises(ValueError, match="decision_model"):
        run(
            leximeter_filter(
                df,
                "item",
                condition="is a fruit",
                save_dir=str(save_dir),
                model="gpt-5.4-nano",
                reset_files=True,
            )
        )

    assert http_calls == 0
    assert meta_path.read_text(encoding="utf-8") == meta_sentinel
    assert raw_path.read_text(encoding="utf-8") == raw_sentinel
