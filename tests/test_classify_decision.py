"""Public API / task-seam tests for classify with DecisionClient.

Tests inject fake DecisionClient implementations into ``leximeter.classify``
and assert only external behavior: DataFrame columns/values, calibrated
probabilities, differentiate mutual exclusivity, checkpoint resume with zero
duplicate calls, and fallback to LLMClient when unconfigured.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pandas as pd
import pytest
from conftest import CountingClient, CountingDecisionClient, DictDecisionClient

from leximeter import classify
from leximeter.client import (
    ChoiceQuestion,
    DecisionClient,
    DecisionResult,
    NoulAnswer,
    NoulQuestion,
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_classify_without_decision_client_uses_llm_path_unchanged(
    tmp_path, monkeypatch
) -> None:
    """When no Decision Provider is configured, classify executes the LLMClient path."""
    monkeypatch.delenv("LEXIMETER_DECISION_PROTOCOL", raising=False)
    df = pd.DataFrame({"text": ["global summit", "local match"]})
    llm = CountingClient({"world": True, "sports": False})

    res = run(
        classify(
            df,
            "text",
            labels={"world": "international affairs", "sports": "athletics"},
            save_dir=str(tmp_path / "llm_fallback"),
            client=llm,
            n_runs=1,
            min_frequency=0.6,
        )
    )

    assert llm.calls == 2
    assert list(res.columns) == ["text", "predicted_classes", "world", "sports"]
    assert res["world"].dtype == bool
    assert list(res["world"]) == [True, True]
    assert list(res["sports"]) == [False, False]
    assert list(res["predicted_classes"]) == [["world"], ["world"]]


def test_classify_single_column_noul_thresholding_and_chunking(tmp_path) -> None:
    """Single-column classify evaluates labels as chunked parallel noul questions."""
    df = pd.DataFrame(
        {"text": ["tech stock surges", "quiet market", "tech stock surges"]}
    )
    labels = {
        "finance": "discusses markets or stocks",
        "tech": "mentions technology",
        "urgent": "time-sensitive alert",
    }

    def decide_for_item(state: Any, key: str, question: Any) -> float:
        assert isinstance(question, NoulQuestion)
        text = state if isinstance(state, str) else state["text"]
        if "tech" in text:
            return {"finance": 0.85, "tech": 0.90, "urgent": 0.30}[key]
        return {"finance": 0.60, "tech": 0.10, "urgent": 0.59}[key]

    # Provider caps each call at 2 questions -> 3 labels split into [2, 1] batches
    decision_client = CountingDecisionClient(
        handler=decide_for_item,
        max_questions_per_call=2,
    )
    llm_unused = CountingClient({"finance": False, "tech": False, "urgent": False})

    res = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(tmp_path / "cls_noul"),
            min_frequency=0.6,
            decision_model="cloudflare/clef-flash",
            client=llm_unused,
            decision_client=decision_client,
        )
    )

    # Generative LLMClient is never touched
    assert llm_unused.calls == 0
    # 2 unique items * 2 chunks = 4 decision calls; every call respects max_questions_per_call=2
    assert decision_client.calls == 4
    assert all(len(qs) <= 2 for _, qs, _ in decision_client.requests)
    assert all(m == "cloudflare/clef-flash" for _, _, m in decision_client.requests)

    # Default include_probabilities=False omits _prob columns
    assert list(res.columns) == [
        "text",
        "predicted_classes",
        "finance",
        "tech",
        "urgent",
    ]
    assert list(res["finance"]) == [True, True, True]
    assert list(res["tech"]) == [True, False, True]
    assert list(res["urgent"]) == [False, False, False]
    assert list(res["predicted_classes"]) == [
        ["finance", "tech"],
        ["finance"],
        ["finance", "tech"],
    ]


def test_classify_respects_n_attributes_per_run_below_provider_cap(tmp_path) -> None:
    """When n_attributes_per_run < max_questions_per_call, n_attributes_per_run wins."""
    df = pd.DataFrame({"text": ["hello"]})
    labels = {"a": "desc a", "b": "desc b", "c": "desc c"}
    decision_client = CountingDecisionClient(
        {"a": 0.9, "b": 0.2, "c": 0.7},
        max_questions_per_call=64,
    )

    res = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(tmp_path / "cls_chunk_attr"),
            n_attributes_per_run=1,
            min_frequency=0.5,
            decision_client=decision_client,
        )
    )

    assert decision_client.calls == 3
    assert all(len(qs) == 1 for _, qs, _ in decision_client.requests)
    assert res["predicted_classes"].iloc[0] == ["a", "c"]


def test_classify_include_probabilities_and_refusal_and_checkpoint_persistence(
    tmp_path,
) -> None:
    """include_probabilities=True returns calibrated _prob cols; raw probs always checkpointed."""
    df = pd.DataFrame({"text": ["sample passage"]})
    labels = {"allowed": "safe label", "blocked": "sensitive label"}
    decision_client = DictDecisionClient(
        {
            "allowed": 0.78,
            "blocked": NoulAnswer(probability=None, confidence=None, refused=True),
        }
    )
    save_dir = str(tmp_path / "cls_probs")

    res = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=save_dir,
            min_frequency=0.6,
            include_probabilities=True,
            decision_client=decision_client,
        )
    )

    assert "allowed_prob" in res.columns
    assert "blocked_prob" in res.columns
    assert bool(res["allowed"].iloc[0]) is True
    assert res["allowed_prob"].iloc[0] == pytest.approx(0.78)
    assert pd.isna(res["blocked"].iloc[0])
    assert pd.isna(res["blocked_prob"].iloc[0])
    # Refused label is NOT included in predicted_classes
    assert res["predicted_classes"].iloc[0] == ["allowed"]

    # Raw probabilities are persisted in raw_responses checkpoint CSV
    raw_df = pd.read_csv(
        tmp_path / "cls_probs" / "classify_responses_raw_responses.csv"
    )
    assert "0.78" in str(raw_df["Response"].iloc[0])
    assert '"refused": true' in str(raw_df["Response"].iloc[0]).lower()


def test_classify_multi_run_averages_probabilities(tmp_path) -> None:
    """When n_runs > 1, classify averages per-run probabilities before thresholding."""
    df = pd.DataFrame({"text": ["ambiguous note"]})
    probs_by_call = [0.80, 0.50]  # mean = 0.65 >= 0.6 -> True
    call_idx = 0

    def next_prob(state: Any, key: str, question: Any) -> float:
        nonlocal call_idx
        val = probs_by_call[call_idx]
        call_idx += 1
        return val

    decision_client = CountingDecisionClient(handler=next_prob)
    save_dir = str(tmp_path / "cls_multirun")

    res = run(
        classify(
            df,
            "text",
            labels={"signal": "contains signal"},
            save_dir=save_dir,
            n_runs=2,
            min_frequency=0.6,
            include_probabilities=True,
            decision_client=decision_client,
        )
    )

    assert decision_client.calls == 2
    assert bool(res["signal"].iloc[0]) is True
    assert res["signal_prob"].iloc[0] == pytest.approx(0.65)
    assert (
        tmp_path / "cls_multirun" / "classify_responses_full_disaggregated.csv"
    ).exists()


def test_classify_differentiate_three_way_choice_and_mutual_exclusivity(
    tmp_path,
) -> None:
    """differentiate=True maps directional pairs to 3-way choice and forbids both-True."""
    df = pd.DataFrame(
        {
            "c": ["warm A", "cold B", "equal C", "close D", "tie E"],
            "s": ["cold A", "warm B", "equal C", "close D2", "tie E2"],
        }
    )
    labels = {
        "warmth_actual": "Circle is warmer than Square",
        "warmth_inverted": "Square is warmer than Circle",
    }

    probs_by_circle: dict[str, dict[str, float]] = {
        "warm A": {"circle": 0.85, "square": 0.05, "neither": 0.10},
        "cold B": {"circle": 0.10, "square": 0.80, "neither": 0.10},
        "equal C": {"circle": 0.10, "square": 0.10, "neither": 0.80},
        # Both circle and square exceed min_frequency=0.40, yet only strict winner is True
        "close D": {"circle": 0.48, "square": 0.45, "neither": 0.07},
        # Exact tie between circle and square -> neither wins (both False)
        "tie E": {"circle": 0.50, "square": 0.50, "neither": 0.00},
    }

    def handler(state: Any, key: str, question: Any) -> dict[str, float]:
        assert isinstance(state, dict)
        assert set(state.keys()) >= {"circle", "square"}
        assert isinstance(question, ChoiceQuestion)
        assert set(question.criteria.keys()) == {"circle", "square", "neither"}
        return probs_by_circle[state["circle"]]

    decision_client = CountingDecisionClient(handler=handler)
    res = run(
        classify(
            df,
            labels=labels,
            save_dir=str(tmp_path / "cls_diff"),
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            min_frequency=0.40,
            include_probabilities=True,
            decision_client=decision_client,
        )
    )

    # 5 unique pairs * 1 ChoiceQuestion each (not 2 noul questions!)
    assert decision_client.calls == 5
    assert all(len(qs) == 1 for _, qs, _ in decision_client.requests)

    # Mutual exclusivity invariant across every single row
    assert not ((res["warmth_actual"] == True) & (res["warmth_inverted"] == True)).any()

    assert list(res["warmth_actual"]) == [True, False, False, True, False]
    assert list(res["warmth_inverted"]) == [False, True, False, False, False]
    assert list(res["warmth_actual_prob"]) == pytest.approx(
        [0.85, 0.10, 0.10, 0.48, 0.50]
    )
    assert list(res["warmth_inverted_prob"]) == pytest.approx(
        [0.05, 0.80, 0.10, 0.45, 0.50]
    )
    assert list(res["predicted_classes"]) == [
        ["warmth_actual"],
        ["warmth_inverted"],
        [],
        ["warmth_actual"],
        [],
    ]


def test_classify_differentiate_preserves_circle_first_randomization(tmp_path) -> None:
    """circle_first=True/False/None controls Circle/Square key ordering in state."""
    labels = {
        "tone_actual": "Circle has warmer tone",
        "tone_inverted": "Square has warmer tone",
    }

    # Explicit circle_first=True
    df_small = pd.DataFrame({"c": ["c1", "c2"], "s": ["s1", "s2"]})
    client_true = CountingDecisionClient({"tone": "circle"})
    run(
        classify(
            df_small,
            labels=labels,
            save_dir=str(tmp_path / "cf_true"),
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            circle_first=True,
            decision_client=client_true,
        )
    )
    assert all(
        list(state.keys())[:2] == ["circle", "square"]
        for state, _, _ in client_true.requests
    )

    # Explicit circle_first=False
    client_false = CountingDecisionClient({"tone": "square"})
    run(
        classify(
            df_small,
            labels=labels,
            save_dir=str(tmp_path / "cf_false"),
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            circle_first=False,
            decision_client=client_false,
        )
    )
    assert all(
        list(state.keys())[:2] == ["square", "circle"]
        for state, _, _ in client_false.requests
    )

    # Randomized circle_first=None produces both orderings across 20 pairs
    df_many = pd.DataFrame(
        {
            "c": [f"circle_{i}" for i in range(20)],
            "s": [f"square_{i}" for i in range(20)],
        }
    )
    client_rand = CountingDecisionClient({"tone": "neither"})
    random.seed(0)
    run(
        classify(
            df_many,
            labels=labels,
            save_dir=str(tmp_path / "cf_rand"),
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            circle_first=None,
            decision_client=client_rand,
        )
    )
    orderings = {tuple(list(state.keys())[:2]) for state, _, _ in client_rand.requests}
    assert orderings == {("circle", "square"), ("square", "circle")}


def test_classify_differentiate_discover_pair_conventions(tmp_path) -> None:
    """Supports both '(inverted)' and circle/square-swapped directional pairs from Discover."""
    df = pd.DataFrame({"c": ["detailed text"], "s": ["brief text"]})
    labels = {
        "quality": "circle has higher quality",
        "quality (inverted)": "square has higher quality",
        "Circle is detailed": "circle gives more details",
        "Square is detailed": "square gives more details",
    }

    def handler(state: Any, key: str, question: Any) -> str:
        assert isinstance(question, ChoiceQuestion)
        if key == "quality":
            return "circle"
        return "square"

    decision_client = CountingDecisionClient(handler=handler)
    res = run(
        classify(
            df,
            labels=labels,
            save_dir=str(tmp_path / "cls_discover_pairs"),
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            decision_client=decision_client,
        )
    )

    # 4 directional labels collapsed into 2 ChoiceQuestions in 1 call
    assert decision_client.calls == 1
    assert len(decision_client.requests[0][1]) == 2
    assert bool(res["quality"].iloc[0]) is True
    assert bool(res["quality (inverted)"].iloc[0]) is False
    assert bool(res["Circle is detailed"].iloc[0]) is False
    assert bool(res["Square is detailed"].iloc[0]) is True


def test_classify_checkpoint_resume_completed_and_interrupted(tmp_path) -> None:
    """Re-running completed or interrupted classify skips checkpointed items with zero duplicate calls."""
    df = pd.DataFrame({"text": ["item_alpha", "item_beta"]})
    labels = {"pos": "positive sentiment"}
    save_dir = str(tmp_path / "cls_resume")

    # 1. Interrupted run: item_beta fails on first run while item_alpha checkpoints cleanly
    class FailOnceOnBeta(DecisionClient):
        def __init__(self) -> None:
            self.seen: list[str] = []
            self.fail_beta = True

        async def decide(self, state, questions, *, model, **kwargs):
            text = str(state)
            self.seen.append(text)
            if text == "item_beta" and self.fail_beta:
                raise RuntimeError("transient crash before beta finished")
            prob = 0.9 if text == "item_alpha" else 0.8
            return DecisionResult(
                answers={"pos": NoulAnswer(probability=prob)}, model=model
            )

    flaky = FailOnceOnBeta()
    res_partial = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=save_dir,
            max_retries=0,
            include_probabilities=False,
            decision_client=flaky,
        )
    )
    assert flaky.seen == ["item_alpha", "item_beta"]
    assert bool(res_partial["pos"].iloc[0]) is True
    assert pd.isna(res_partial["pos"].iloc[1])

    # 2. Resume after interruption: item_alpha is skipped, ONLY item_beta is evaluated
    flaky.fail_beta = False
    flaky.seen.clear()
    res_resumed = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=save_dir,
            reset_files=False,
            include_probabilities=True,
            decision_client=flaky,
        )
    )
    assert flaky.seen == ["item_beta"]  # zero duplicate call for item_alpha
    assert list(res_resumed["pos"]) == [True, True]
    assert list(res_resumed["pos_prob"]) == pytest.approx([0.9, 0.8])

    # 3. Re-run completed run: ZERO DecisionClient calls, probabilities still available
    counting = CountingDecisionClient({"pos": 0.1})
    res_cached = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=save_dir,
            reset_files=False,
            include_probabilities=True,
            decision_client=counting,
        )
    )
    assert counting.calls == 0
    assert list(res_cached["pos"]) == [True, True]
    assert list(res_cached["pos_prob"]) == pytest.approx([0.9, 0.8])

    # 4. reset_files=True forces full re-evaluation
    res_reset = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=save_dir,
            reset_files=True,
            include_probabilities=True,
            decision_client=counting,
        )
    )
    assert counting.calls == 2
    assert list(res_reset["pos"]) == [False, False]
    assert list(res_reset["pos_prob"]) == pytest.approx([0.1, 0.1])


def test_classify_differentiate_checkpoint_resume_zero_duplicate_calls(
    tmp_path,
) -> None:
    """Re-running completed differentiate classify makes zero duplicate DecisionClient calls."""
    df = pd.DataFrame({"c": ["alpha", "beta"], "s": ["gamma", "delta"]})
    labels = {
        "warmth_actual": "Circle is warmer",
        "warmth_inverted": "Square is warmer",
    }
    save_dir = str(tmp_path / "cls_diff_resume")
    client = CountingDecisionClient(
        {"warmth": {"circle": 0.85, "square": 0.05, "neither": 0.10}}
    )

    first = run(
        classify(
            df,
            labels=labels,
            save_dir=save_dir,
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            include_probabilities=False,
            decision_client=client,
        )
    )
    assert client.calls == 2
    assert list(first["warmth_actual"]) == [True, True]

    # Re-run with include_probabilities=True -> zero new calls, probabilities restored from checkpoint
    second = run(
        classify(
            df,
            labels=labels,
            save_dir=save_dir,
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            reset_files=False,
            include_probabilities=True,
            decision_client=client,
        )
    )
    assert client.calls == 2
    assert list(second["warmth_actual"]) == [True, True]
    assert list(second["warmth_inverted"]) == [False, False]
    assert list(second["warmth_actual_prob"]) == pytest.approx([0.85, 0.85])
    assert list(second["warmth_inverted_prob"]) == pytest.approx([0.05, 0.05])


def test_classify_end_to_end_with_systemone_mock_transport_and_non_ascii_labels(
    tmp_path,
) -> None:
    """End-to-end classify + SystemOneClient handles punctuation and non-ASCII labels."""
    import json

    import httpx

    from leximeter.client import SystemOneClient

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert list(body["questions"].keys()) == ["q_0", "q_1"]
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14.0",
                "answers": {
                    "q_0": {"type": "noul", "noul": 0.91},
                    "q_1": {"type": "noul", "noul": 0.12},
                },
                "usage": {"input_tokens": 42, "output_tokens": 0},
            },
        )

    client = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    df = pd.DataFrame({"text": ["财报超预期，利润增长 > 20%"]})
    k1 = "盈利增长 (> 10%)"
    k2 = "负面情绪: 担忧 / 风险"

    res = run(
        classify(
            df,
            "text",
            labels={k1: "提及盈利显著增长", k2: "表达负面情绪"},
            save_dir=str(tmp_path / "cls_e2e_sys1"),
            min_frequency=0.6,
            include_probabilities=True,
            decision_model="jev-latest",
            decision_client=client,
        )
    )
    assert bool(res[k1].iloc[0]) is True
    assert bool(res[k2].iloc[0]) is False
    assert res[f"{k1}_prob"].iloc[0] == pytest.approx(0.91)
    assert res[f"{k2}_prob"].iloc[0] == pytest.approx(0.12)
    assert res["predicted_classes"].iloc[0] == [k1]
