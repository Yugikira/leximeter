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
        "warm A": {"option_a": 0.85, "option_b": 0.05, "neither": 0.10},
        "cold B": {"option_a": 0.10, "option_b": 0.80, "neither": 0.10},
        "equal C": {"option_a": 0.10, "option_b": 0.10, "neither": 0.80},
        # Both option_a and option_b exceed min_frequency=0.40, yet only strict winner is True
        "close D": {"option_a": 0.48, "option_b": 0.45, "neither": 0.07},
        # Exact tie between option_a and option_b -> neither wins (both False)
        "tie E": {"option_a": 0.50, "option_b": 0.50, "neither": 0.00},
    }

    def handler(state: Any, key: str, question: Any) -> dict[str, float]:
        assert isinstance(state, dict)
        assert set(state.keys()) >= {"circle", "square"}
        assert isinstance(question, ChoiceQuestion)
        assert set(question.criteria.keys()) == {"option_a", "option_b", "neither"}
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
    client_true = CountingDecisionClient({"tone": "option_a"})
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
        and list(qs["tone"].criteria.keys()) == ["option_a", "option_b", "neither"]
        for state, qs, _ in client_true.requests
    )

    # Explicit circle_first=False
    client_false = CountingDecisionClient({"tone": "option_b"})
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
        and list(qs["tone"].criteria.keys()) == ["option_b", "option_a", "neither"]
        for state, qs, _ in client_false.requests
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
            return "option_a"
        return "option_b"

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
        {"warmth": {"option_a": 0.85, "option_b": 0.05, "neither": 0.10}}
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


def test_classify_differentiate_square_primary_label_polarity(tmp_path) -> None:
    """Square-primary directional pairs bind 'option_a' -> actual and 'option_b' -> inverted."""
    actual_key = "square entries emphasize cost more than circle entries_actual"
    inverted_key = "circle entries emphasize cost more than square entries_inverted"
    labels = {
        actual_key: "Square emphasizes cost more than Circle",
        inverted_key: "Circle emphasizes cost more than Square",
    }
    df = pd.DataFrame(
        {
            "c": ["focus on quality", "focus on price and budget"],
            "s": ["focus on price and budget", "focus on quality"],
        }
    )

    seen_questions: list[ChoiceQuestion] = []

    def handler(state: Any, key: str, question: Any) -> dict[str, float]:
        assert isinstance(question, ChoiceQuestion)
        seen_questions.append(question)
        assert set(question.criteria.keys()) == {"option_a", "option_b", "neither"}
        assert (
            question.criteria["option_a"] == "Square emphasizes cost more than Circle"
        )
        assert (
            question.criteria["option_b"] == "Circle emphasizes cost more than Square"
        )
        assert "feature" not in str(question.instructions).lower()
        assert "feature" not in str(question.criteria["neither"]).lower()
        # Evaluate faithfully against the propositions: option_a says Square mentions price
        if "price" in state["square"]:
            return {"option_a": 0.90, "option_b": 0.05, "neither": 0.05}
        return {"option_a": 0.05, "option_b": 0.85, "neither": 0.10}

    dc = CountingDecisionClient(handler=handler)
    res = run(
        classify(
            df,
            labels=labels,
            save_dir=str(tmp_path / "cls_sq_primary"),
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            include_probabilities=True,
            decision_client=dc,
        )
    )

    # Collapsed into 1 ChoiceQuestion per row (2 calls total, not 4)
    assert dc.calls == 2
    assert len(seen_questions) == 2
    # Row 0: Square genuinely emphasizes cost more than Circle -> actual=True, inverted=False
    assert bool(res[actual_key].iloc[0]) is True
    assert bool(res[inverted_key].iloc[0]) is False
    assert res[f"{actual_key}_prob"].iloc[0] == pytest.approx(0.90)
    assert res[f"{inverted_key}_prob"].iloc[0] == pytest.approx(0.05)
    # Row 1: Circle genuinely emphasizes cost more than Square -> actual=False, inverted=True
    assert bool(res[actual_key].iloc[1]) is False
    assert bool(res[inverted_key].iloc[1]) is True
    assert res[f"{actual_key}_prob"].iloc[1] == pytest.approx(0.05)
    assert res[f"{inverted_key}_prob"].iloc[1] == pytest.approx(0.85)


def test_classify_resume_rechunks_when_provider_cap_shrinks(tmp_path) -> None:
    """Resuming with a smaller max_questions_per_call re-chunks instead of violating the cap."""
    # 1. Single-column (noul) mode
    labels = {"a": "desc a", "b": "desc b", "c": "desc c"}
    save_single = str(tmp_path / "resume_cap_single")
    c_large = CountingDecisionClient(
        {"a": 0.9, "b": 0.2, "c": 0.8}, max_questions_per_call=64
    )
    run(
        classify(
            pd.DataFrame({"text": ["item_1"]}),
            "text",
            labels=labels,
            save_dir=save_single,
            decision_client=c_large,
        )
    )
    assert c_large.calls == 1
    assert len(c_large.requests[0][1]) == 3

    c_small = CountingDecisionClient(
        {"a": 0.9, "b": 0.2, "c": 0.8}, max_questions_per_call=2
    )
    res_single = run(
        classify(
            pd.DataFrame({"text": ["item_1", "item_2"]}),
            "text",
            labels=labels,
            save_dir=save_single,
            reset_files=False,
            decision_client=c_small,
        )
    )
    assert c_small.calls > 0
    assert all(len(qs) <= 2 for _, qs, _ in c_small.requests)
    assert list(res_single["a"]) == [True, True]
    assert list(res_single["b"]) == [False, False]
    assert list(res_single["c"]) == [True, True]

    # 2. Differentiate (choice) mode
    pair_labels = {
        "p1_actual": "Circle has p1",
        "p1_inverted": "Square has p1",
        "p2_actual": "Circle has p2",
        "p2_inverted": "Square has p2",
        "p3_actual": "Circle has p3",
        "p3_inverted": "Square has p3",
    }
    save_diff = str(tmp_path / "resume_cap_diff")
    cd_large = CountingDecisionClient(
        {"p1": "option_a", "p2": "option_b", "p3": "option_a"},
        max_questions_per_call=64,
    )
    run(
        classify(
            pd.DataFrame({"c": ["c1"], "s": ["s1"]}),
            labels=pair_labels,
            save_dir=save_diff,
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            decision_client=cd_large,
        )
    )
    assert cd_large.calls == 1
    assert len(cd_large.requests[0][1]) == 3

    cd_small = CountingDecisionClient(
        {"p1": "option_a", "p2": "option_b", "p3": "option_a"}, max_questions_per_call=2
    )
    res_diff = run(
        classify(
            pd.DataFrame({"c": ["c1", "c2"], "s": ["s1", "s2"]}),
            labels=pair_labels,
            save_dir=save_diff,
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            reset_files=False,
            decision_client=cd_small,
        )
    )
    assert cd_small.calls > 0
    assert all(len(qs) <= 2 for _, qs, _ in cd_small.requests)
    assert list(res_diff["p1_actual"]) == [True, True]
    assert list(res_diff["p2_inverted"]) == [True, True]
    assert list(res_diff["p3_actual"]) == [True, True]


def test_classify_systemone_missing_decision_model_fails_fast_without_disk_mutation(
    tmp_path, monkeypatch
) -> None:
    """SystemOneClient without decision_model / LEXIMETER_DECISION_MODEL fails fast before HTTP or disk resets."""
    import httpx

    from leximeter.client import SystemOneClient

    http_calls = 0

    def fail_transport(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        raise AssertionError(f"Unexpected HTTP call to {request.url}")

    save_dir = tmp_path / "cls_fail_fast"
    save_dir.mkdir(parents=True, exist_ok=True)
    meta_path = save_dir / "classify_responses_run_metadata.json"
    attrs_path = save_dir / "attributes.json"
    legacy_attrs_path = save_dir / "classify_responses_attrs.json"
    raw_path = save_dir / "classify_responses_raw_responses.csv"
    meta_sentinel = '{"model": "prev-model"}'
    attrs_sentinel = '{"old": "kept"}'
    raw_sentinel = "Identifier,Response,Successful,Error Log,Time Taken\n"
    meta_path.write_text(meta_sentinel, encoding="utf-8")
    attrs_path.write_text(attrs_sentinel, encoding="utf-8")
    legacy_attrs_path.write_text(attrs_sentinel, encoding="utf-8")
    raw_path.write_text(raw_sentinel, encoding="utf-8")

    df = pd.DataFrame({"text": ["hello world"]})
    labels = {"new_lbl": "new description"}
    sys1 = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=httpx.MockTransport(fail_transport),
    )

    for reset in (False, True):
        for bad_dm in (None, "   "):
            with pytest.raises(ValueError, match="decision_model"):
                run(
                    classify(
                        df,
                        "text",
                        labels=labels,
                        save_dir=str(save_dir),
                        model="gpt-5.4-mini",
                        decision_model=bad_dm,
                        decision_client=sys1,
                        reset_files=reset,
                    )
                )

    monkeypatch.setenv("LEXIMETER_DECISION_PROTOCOL", "systemone")
    monkeypatch.setenv("LEXIMETER_DECISION_BASE_URL", "https://api.typesafe.ai")
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)
    with pytest.raises(ValueError, match="decision_model"):
        run(
            classify(
                df,
                "text",
                labels=labels,
                save_dir=str(save_dir),
                model="gpt-5.4-mini",
                reset_files=True,
            )
        )

    assert http_calls == 0
    assert meta_path.read_text(encoding="utf-8") == meta_sentinel
    assert attrs_path.read_text(encoding="utf-8") == attrs_sentinel
    assert legacy_attrs_path.read_text(encoding="utf-8") == attrs_sentinel
    assert raw_path.read_text(encoding="utf-8") == raw_sentinel


def test_classify_differentiate_issue_18_comparatives_negations_and_multilingual(
    tmp_path,
) -> None:
    """Proposition-bound ChoiceQuestion avoids polarity inversion across arbitrary phrasing and languages (#18)."""
    labels = {
        # Exact Issue #18 reproduction (reference side named first)
        "evidence_actual": "Compared to square entries, circle entries cite more quantitative evidence.",
        "evidence_inverted": "Compared to circle entries, square entries cite more quantitative evidence.",
        # Plural-noun '-s' + relative-clause negation (Square lacks reliance on Circle)
        "autonomy_actual": "Square's new systems metrics that omit details do not rely on Circle",
        "autonomy_inverted": "Circle's new systems metrics that omit details do not rely on Square",
        # Chinese comparative naming 'square' before 'circle'
        "zh_evidence_actual": "与 square 相比，circle 引用了更多定量证据。",
        "zh_evidence_inverted": "与 circle 相比，square 引用了更多定量证据。",
        # Directional label stem with undirected shared description
        "Square is detailed": "uses concrete numbers",
        "Circle is detailed": "uses concrete numbers",
        # Undirected pair with empty descriptions
        "warmth_actual": "",
        "warmth_inverted": "",
        # Solo directional label (no explicit inverted key)
        "Unlike Square, Circle cites primary sources": "",
    }
    df = pd.DataFrame(
        {
            "c": ["Circle cites p < 0.01 and primary archive sources.", "Brief note."],
            "s": ["Brief note.", "Square cites 42 benchmarks independently."],
        }
    )

    def handler(state: Any, key: str, question: Any) -> dict[str, float]:
        assert isinstance(question, ChoiceQuestion)
        crit = question.criteria
        if key == "evidence":
            assert (
                crit["option_a"]
                == "Compared to square entries, circle entries cite more quantitative evidence."
            )
            assert (
                crit["option_b"]
                == "Compared to circle entries, square entries cite more quantitative evidence."
            )
        elif key == "zh_evidence":
            assert crit["option_a"] == "与 square 相比，circle 引用了更多定量证据。"
            assert crit["option_b"] == "与 circle 相比，square 引用了更多定量证据。"
        elif key == "Square is detailed":
            assert crit["option_a"] == "Square is detailed: uses concrete numbers"
            assert crit["option_b"] == "Circle is detailed: uses concrete numbers"
        elif key == "warmth":
            assert (
                "Circle clearly manifests 'warmth' more than Square."
                == crit["option_a"]
            )
            assert (
                "Square clearly manifests 'warmth' more than Circle."
                == crit["option_b"]
            )
        elif key == "Unlike Square, Circle cites primary sources":
            assert (
                crit["option_a"]
                == "The label 'Unlike Square, Circle cites primary sources' applies."
            )
            assert (
                crit["option_b"]
                == "The label 'Unlike Circle, Square cites primary sources' applies."
            )

        # Row 0: actual holds for Circle-favoring pairs; inverted holds for Square-favoring pairs
        if "p < 0.01" in state["circle"]:
            if key in {"autonomy", "Square is detailed"}:
                return {"option_a": 0.05, "option_b": 0.90, "neither": 0.05}
            return {"option_a": 0.90, "option_b": 0.05, "neither": 0.05}
        # Row 1: opposite
        if key in {"autonomy", "Square is detailed"}:
            return {"option_a": 0.88, "option_b": 0.06, "neither": 0.06}
        return {"option_a": 0.06, "option_b": 0.88, "neither": 0.06}

    dc = CountingDecisionClient(handler=handler)
    res = run(
        classify(
            df,
            labels=labels,
            save_dir=str(tmp_path / "cls_issue18"),
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            include_probabilities=True,
            decision_client=dc,
        )
    )

    assert dc.calls == 2
    assert list(res["evidence_actual"]) == [True, False]
    assert list(res["evidence_inverted"]) == [False, True]
    assert list(res["autonomy_actual"]) == [False, True]
    assert list(res["autonomy_inverted"]) == [True, False]
    assert list(res["zh_evidence_actual"]) == [True, False]
    assert list(res["zh_evidence_inverted"]) == [False, True]
    assert list(res["Square is detailed"]) == [False, True]
    assert list(res["Circle is detailed"]) == [True, False]
    assert list(res["warmth_actual"]) == [True, False]
    assert list(res["warmth_inverted"]) == [False, True]
    assert list(res["Unlike Square, Circle cites primary sources"]) == [True, False]


def test_classify_differentiate_legacy_v1_checkpoint_auto_invalidates(tmp_path) -> None:
    """Pre-v2 differentiate checkpoints with old 'circle'/'square' option keys auto-reset on resume."""
    import json

    from leximeter.tasks import _common

    save_dir = tmp_path / "cls_v1_upgrade"
    save_dir.mkdir(parents=True, exist_ok=True)
    ident = _common.hash_identifier("c1|s1")
    (save_dir / "classify_responses_run_metadata.json").write_text(
        json.dumps(
            {
                "model": "dummy-decision",
                "decision_provider": True,
                "n_attributes_per_run": 64,
                "attribute_batches": [["evidence_actual", "evidence_inverted"]],
            }
        ),
        encoding="utf-8",
    )
    legacy_resp = json.dumps(
        {
            "model": "dummy-decision",
            "answers": {
                "evidence": {
                    "type": "choice",
                    "choice": "circle",
                    "probabilities": {"circle": 0.95, "square": 0.03, "neither": 0.02},
                    "confidence": 0.95,
                    "refused": False,
                }
            },
            "input_tokens": 10,
            "output_tokens": 0,
        }
    )
    pd.DataFrame(
        [
            {
                "Identifier": f"{ident}_batch0",
                "Response": legacy_resp,
                "Successful": True,
                "Error Log": "",
                "Time Taken": 0.01,
            }
        ]
    ).to_csv(save_dir / "classify_responses_raw_responses.csv", index=False)

    dc = CountingDecisionClient(
        {"evidence": {"option_a": 0.85, "option_b": 0.10, "neither": 0.05}}
    )
    res = run(
        classify(
            pd.DataFrame({"c": ["c1"], "s": ["s1"]}),
            labels={
                "evidence_actual": "Compared to square, circle cites more data.",
                "evidence_inverted": "Compared to circle, square cites more data.",
            },
            save_dir=str(save_dir),
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            reset_files=False,
            include_probabilities=True,
            decision_client=dc,
        )
    )

    assert dc.calls == 1
    assert bool(res["evidence_actual"].iloc[0]) is True
    assert res["evidence_actual_prob"].iloc[0] == pytest.approx(0.85)
