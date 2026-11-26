"""Public API / task-seam tests for rank with DecisionClient.

Tests inject fake DecisionClient implementations into ``leximeter.rank``
and assert only external behavior: DataFrame columns (``{attr}``,
``{attr}_raw``, ``{attr}_se``), Bradley-Terry ordering, 4-way ``choice``
adjudication (``circle``, ``square``, ``draw``, ``insufficient_signal``),
chunking by ``max_questions_per_call`` and ``n_attributes_per_run``,
preserved ``circle_first`` randomization, graceful refusal handling,
round-CSV checkpoint resume with zero duplicate calls, and fallback to
``LLMClient`` when unconfigured.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import pytest
from conftest import CountingClient, CountingDecisionClient, DictDecisionClient

from leximeter import rank
from leximeter.client import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionClient,
    DecisionResult,
    DummyDecisionClient,
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_rank_without_decision_client_uses_llm_path_unchanged(
    tmp_path, monkeypatch
) -> None:
    """When no Decision Provider is configured, rank executes the LLMClient path."""
    monkeypatch.delenv("LEXIMETER_DECISION_PROTOCOL", raising=False)
    df = pd.DataFrame({"speech": ["alpha", "beta", "gamma"]})
    llm = CountingClient({"populism": "circle"})

    res = run(
        rank(
            df,
            "speech",
            attributes={"populism": "appeals to the common people against elites"},
            save_dir=str(tmp_path / "llm_fallback"),
            client=llm,
            n_rounds=1,
            matches_per_round=1,
        )
    )

    assert llm.calls > 0
    assert list(res.columns) == ["speech", "populism", "populism_raw", "populism_se"]
    assert len(res) == 3
    assert res["populism"].notna().all()
    assert res["populism_se"].notna().all()
    assert (res["populism_se"] > 0).all()


def test_rank_decision_client_four_way_choice_and_bt_ordering(tmp_path) -> None:
    """Injected DecisionClient adjudicates pairwise matches via 4-way choice and fits BT."""
    df = pd.DataFrame({"speech": ["low_item", "mid_item", "high_item"]})
    strength = {"low_item": 1, "mid_item": 2, "high_item": 3}

    def adjudicate(state: Any, key: str, question: Any) -> str:
        assert key == "clarity"
        assert isinstance(state, dict)
        assert set(state.keys()) >= {"circle", "square"}
        assert isinstance(question, ChoiceQuestion)
        assert set(question.criteria.keys()) == {
            "circle",
            "square",
            "draw",
            "insufficient_signal",
        }
        c_score = strength[state["circle"]]
        s_score = strength[state["square"]]
        if c_score > s_score:
            return "circle"
        if s_score > c_score:
            return "square"
        return "draw"

    decision_client = CountingDecisionClient(handler=adjudicate)
    llm_unused = CountingClient({"clarity": "circle"})
    save_dir = str(tmp_path / "rank_decision")

    res = run(
        rank(
            df,
            "speech",
            attributes={"clarity": "clear and easy to follow"},
            save_dir=save_dir,
            n_rounds=2,
            matches_per_round=2,
            decision_model="cloudflare/clef-flash",
            client=llm_unused,
            decision_client=decision_client,
        )
    )

    assert llm_unused.calls == 0
    assert decision_client.calls > 0
    assert all(m == "cloudflare/clef-flash" for _, _, m in decision_client.requests)

    # Exact rank DataFrame contract
    assert list(res.columns) == ["speech", "clarity", "clarity_raw", "clarity_se"]
    assert len(res) == 3
    assert (res["clarity_se"] > 0).all()
    assert res["clarity_se"].notna().all()
    assert res["clarity_raw"].mean() == pytest.approx(0.0, abs=1e-6)

    # Ground-truth Bradley-Terry ordering: high_item > mid_item > low_item
    by_speech = res.set_index("speech")
    assert (
        by_speech.loc["high_item", "clarity_raw"]
        > by_speech.loc["mid_item", "clarity_raw"]
    )
    assert (
        by_speech.loc["mid_item", "clarity_raw"]
        > by_speech.loc["low_item", "clarity_raw"]
    )
    assert by_speech.loc["high_item", "clarity"] > by_speech.loc["mid_item", "clarity"]
    assert by_speech.loc["mid_item", "clarity"] > by_speech.loc["low_item", "clarity"]

    # Round CSV checkpoints preserve Batch/Pair/IdA/IdB columns
    for rnd in (0, 1):
        round_csv = tmp_path / "rank_decision" / f"rankings_round{rnd}.csv"
        assert round_csv.exists()
        round_df = pd.read_csv(round_csv)
        assert {"Batch", "Pair", "IdA", "IdB"}.issubset(round_df.columns)


@pytest.mark.parametrize(
    "outcome",
    [
        "draw",
        "insufficient_signal",
        # Exact circle/square probability tie even when provider picks "circle" alphabetically
        ChoiceAnswer(
            choice="circle",
            probabilities={
                "circle": 0.5,
                "square": 0.5,
                "draw": 0.0,
                "insufficient_signal": 0.0,
            },
            confidence=0.5,
        ),
    ],
)
def test_rank_draw_insufficient_signal_and_probability_ties(tmp_path, outcome) -> None:
    """draw, insufficient_signal, and exact circle/square probability ties yield balanced scores."""
    df = pd.DataFrame({"speech": ["item_a", "item_b", "item_c"]})
    client = DictDecisionClient({"tone": outcome})

    res = run(
        rank(
            df,
            "speech",
            attributes={"tone": "assertiveness"},
            save_dir=str(tmp_path / "rank_tie"),
            n_rounds=2,
            matches_per_round=2,
            circle_first=True,
            decision_client=client,
        )
    )

    assert list(res.columns) == ["speech", "tone", "tone_raw", "tone_se"]
    assert res["tone_raw"].abs().max() == pytest.approx(0.0, abs=1e-6)
    assert res["tone"].abs().max() == pytest.approx(0.0, abs=1e-6)
    assert res["tone_se"].notna().all()
    assert (res["tone_se"] > 0).all()


def test_rank_dummy_decision_client_and_model_resolution(tmp_path, monkeypatch) -> None:
    """DummyDecisionClient works out of the box; decision_model resolves explicit > env > model."""
    df = pd.DataFrame({"speech": ["a", "b", "c"]})

    # DummyDecisionClient out-of-the-box
    res = run(
        rank(
            df,
            "speech",
            attributes={"warmth": "friendly tone"},
            save_dir=str(tmp_path / "rank_dummy"),
            n_rounds=1,
            matches_per_round=2,
            decision_client=DummyDecisionClient(),
        )
    )
    assert list(res.columns) == ["speech", "warmth", "warmth_raw", "warmth_se"]
    assert res["warmth_se"].notna().all()
    assert (res["warmth_se"] > 0).all()

    # Env LEXIMETER_DECISION_MODEL wins when decision_model=None
    monkeypatch.setenv("LEXIMETER_DECISION_MODEL", "typesafe/jev-1.14")
    c_env = CountingDecisionClient({"warmth": "circle"})
    run(
        rank(
            df,
            "speech",
            attributes={"warmth": "friendly tone"},
            save_dir=str(tmp_path / "rank_env_model"),
            model="gpt-5.4-mini",
            n_rounds=1,
            matches_per_round=1,
            decision_client=c_env,
        )
    )
    assert c_env.calls > 0
    assert all(m == "typesafe/jev-1.14" for _, _, m in c_env.requests)

    # Fallback to cfg.model when both decision_model and env are unset
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)
    c_fallback = CountingDecisionClient({"warmth": "circle"})
    run(
        rank(
            df,
            "speech",
            attributes={"warmth": "friendly tone"},
            save_dir=str(tmp_path / "rank_fallback_model"),
            model="custom-fallback-model",
            n_rounds=1,
            matches_per_round=1,
            decision_client=c_fallback,
        )
    )
    assert c_fallback.calls > 0
    assert all(m == "custom-fallback-model" for _, _, m in c_fallback.requests)


def test_rank_chunks_attributes_by_max_questions_per_call_and_n_attributes_per_run(
    tmp_path,
) -> None:
    """Attributes are chunked by min(n_attributes_per_run, client.max_questions_per_call)."""
    df = pd.DataFrame({"speech": ["low", "mid", "high"]})
    attributes = {"a": "desc a", "b": "desc b", "c": "desc c"}
    strength = {"low": 1, "mid": 2, "high": 3}

    def adjudicate(state: Any, key: str, question: Any) -> str:
        assert isinstance(question, ChoiceQuestion)
        return (
            "circle"
            if strength[state["circle"]] > strength[state["square"]]
            else "square"
        )

    # 1. Provider cap = 2 splits 3 attributes into [2, 1] batches per pair
    c_cap = CountingDecisionClient(handler=adjudicate, max_questions_per_call=2)
    res_cap = run(
        rank(
            df,
            "speech",
            attributes=attributes,
            save_dir=str(tmp_path / "rank_cap"),
            n_rounds=1,
            matches_per_round=2,
            decision_client=c_cap,
        )
    )
    assert c_cap.calls > 0
    assert all(len(qs) <= 2 for _, qs, _ in c_cap.requests)
    assert {len(qs) for _, qs, _ in c_cap.requests} == {1, 2}
    assert list(res_cap.columns) == [
        "speech",
        "a",
        "a_raw",
        "a_se",
        "b",
        "b_raw",
        "b_se",
        "c",
        "c_raw",
        "c_se",
    ]
    for attr in ("a", "b", "c"):
        by_speech = res_cap.set_index("speech")
        assert (
            by_speech.loc["high", f"{attr}_raw"] > by_speech.loc["low", f"{attr}_raw"]
        )
        assert (res_cap[f"{attr}_se"] > 0).all()

    # 2. n_attributes_per_run=1 < provider cap=64 splits into 1-question calls
    c_attr = CountingDecisionClient(handler=adjudicate, max_questions_per_call=64)
    res_attr = run(
        rank(
            df,
            "speech",
            attributes=attributes,
            save_dir=str(tmp_path / "rank_attr_cap"),
            n_rounds=1,
            matches_per_round=1,
            n_attributes_per_run=1,
            decision_client=c_attr,
        )
    )
    assert c_attr.calls > 0
    assert all(len(qs) == 1 for _, qs, _ in c_attr.requests)
    assert res_attr["a_se"].notna().all()
    assert res_attr["c_se"].notna().all()

    # 3. Invalid n_attributes_per_run raises ValueError
    with pytest.raises(ValueError, match="n_attributes_per_run"):
        run(
            rank(
                df,
                "speech",
                attributes=attributes,
                save_dir=str(tmp_path / "rank_bad_n"),
                n_rounds=1,
                matches_per_round=1,
                n_attributes_per_run=0,
                decision_client=c_attr,
            )
        )


def test_rank_preserves_circle_first_randomization_and_additional_instructions(
    tmp_path,
) -> None:
    """circle_first=True/False/None controls Circle/Square ordering without inverting outcomes."""
    df = pd.DataFrame({"speech": ["low", "mid", "high"]})
    strength = {"low": 1, "mid": 2, "high": 3}

    def adjudicate(state: Any, key: str, question: Any) -> str:
        return (
            "circle"
            if strength[state["circle"]] > strength[state["square"]]
            else "square"
        )

    # Explicit circle_first=True + additional_instructions + modality="entity"
    c_true = CountingDecisionClient(handler=adjudicate)
    res_true = run(
        rank(
            df,
            "speech",
            attributes={"clarity": "clear expression"},
            save_dir=str(tmp_path / "cf_true"),
            n_rounds=1,
            matches_per_round=2,
            circle_first=True,
            modality="entity",
            additional_instructions="Focus strictly on literal wording.",
            decision_client=c_true,
        )
    )
    assert c_true.calls > 0
    for state, qs, _ in c_true.requests:
        assert isinstance(state, dict)
        assert list(state.keys())[:2] == ["circle", "square"]
        assert state["additional_instructions"] == "Focus strictly on literal wording."
        q = qs["clarity"]
        assert isinstance(q, ChoiceQuestion)
        assert list(q.criteria.keys())[:2] == ["circle", "square"]
    by_speech_t = res_true.set_index("speech")
    assert (
        by_speech_t.loc["high", "clarity_raw"] > by_speech_t.loc["low", "clarity_raw"]
    )

    # Explicit circle_first=False: keys reversed, BT ordering preserved identically!
    c_false = CountingDecisionClient(handler=adjudicate)
    res_false = run(
        rank(
            df,
            "speech",
            attributes={"clarity": "clear expression"},
            save_dir=str(tmp_path / "cf_false"),
            n_rounds=1,
            matches_per_round=2,
            circle_first=False,
            decision_client=c_false,
        )
    )
    assert c_false.calls > 0
    for state, qs, _ in c_false.requests:
        assert isinstance(state, dict)
        assert list(state.keys())[:2] == ["square", "circle"]
        q = qs["clarity"]
        assert isinstance(q, ChoiceQuestion)
        assert list(q.criteria.keys())[:2] == ["square", "circle"]
    by_speech_f = res_false.set_index("speech")
    assert (
        by_speech_f.loc["high", "clarity_raw"] > by_speech_f.loc["low", "clarity_raw"]
    )

    # Randomized circle_first=None produces both orderings across many pairs
    df_many = pd.DataFrame({"speech": [f"item_{i}" for i in range(8)]})
    c_rand = CountingDecisionClient({"clarity": "draw"})
    run(
        rank(
            df_many,
            "speech",
            attributes={"clarity": "clear expression"},
            save_dir=str(tmp_path / "cf_rand"),
            n_rounds=2,
            matches_per_round=3,
            circle_first=None,
            decision_client=c_rand,
        )
    )
    orderings = {
        tuple(list(state.keys())[:2])
        for state, _, _ in c_rand.requests
        if isinstance(state, dict)
    }
    assert orderings == {("circle", "square"), ("square", "circle")}


def test_rank_handles_per_question_refusals_and_call_errors_gracefully(
    tmp_path,
) -> None:
    """Per-question refusals and transient provider errors never crash a tournament round."""
    df = pd.DataFrame({"speech": ["low", "mid", "high"]})
    strength = {"low": 1, "mid": 2, "high": 3}
    call_counter = 0

    def handler(state: Any, key: str, question: Any) -> Any:
        nonlocal call_counter
        if key == "blocked_all":
            return ChoiceAnswer(
                choice=None, probabilities={}, confidence=None, refused=True
            )
        if key == "blocked_half":
            call_counter += 1
            if call_counter % 2 == 1:
                return ChoiceAnswer(
                    choice=None, probabilities={}, confidence=None, refused=True
                )
        return (
            "circle"
            if strength[state["circle"]] > strength[state["square"]]
            else "square"
        )

    decision_client = DictDecisionClient(handler=handler)
    res = run(
        rank(
            df,
            "speech",
            attributes={
                "safe": "non-sensitive construct",
                "blocked_half": "sometimes refused",
                "blocked_all": "always refused",
            },
            save_dir=str(tmp_path / "rank_refusal"),
            n_rounds=2,
            matches_per_round=2,
            decision_client=decision_client,
        )
    )

    assert len(res) == 3
    by_speech = res.set_index("speech")
    # Safe attribute unaffected: proper BT ordering and positive SEs
    assert by_speech.loc["high", "safe_raw"] > by_speech.loc["low", "safe_raw"]
    assert (res["safe_se"] > 0).all()
    # Partially refused attribute still fits BT on non-refused outcomes
    assert res["blocked_half_se"].notna().all()
    assert (res["blocked_half_se"] > 0).all()
    # Fully refused attribute degrades cleanly to 0.0 / NaN SE without crashing
    assert (res["blocked_all"] == 0.0).all()
    assert (res["blocked_all_raw"] == 0.0).all()
    assert res["blocked_all_se"].isna().all()

    # Whole-call failure on one pair also completes without crashing
    class FailFirstCallClient(DecisionClient):
        def __init__(self) -> None:
            self.calls = 0

        async def decide(self, state, questions, *, model, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient single-pair failure")
            return DecisionResult(
                answers={
                    k: ChoiceAnswer(
                        choice="circle",
                        probabilities={
                            "circle": 1.0,
                            "square": 0.0,
                            "draw": 0.0,
                            "insufficient_signal": 0.0,
                        },
                        confidence=1.0,
                    )
                    for k in questions
                },
                model=model,
            )

    res_err = run(
        rank(
            df,
            "speech",
            attributes={"safe": "construct"},
            save_dir=str(tmp_path / "rank_call_err"),
            n_rounds=1,
            matches_per_round=2,
            max_retries=0,
            decision_client=FailFirstCallClient(),
        )
    )
    assert len(res_err) == 3
    assert res_err["safe_se"].notna().all()


def test_rank_checkpoint_resume_interrupted_and_completed_zero_duplicate_calls(
    tmp_path,
) -> None:
    """Interrupted rank resumes from round CSVs with zero duplicate DecisionClient calls."""
    df = pd.DataFrame({"speech": ["item_1", "item_2", "item_3", "item_4"]})
    strength = {"item_1": 1, "item_2": 2, "item_3": 3, "item_4": 4}
    save_dir = str(tmp_path / "rank_resume")

    def ground_truth(state: Any, key: str, question: Any) -> str:
        return (
            "circle"
            if strength[state["circle"]] > strength[state["square"]]
            else "square"
        )

    # 1. Complete Round 0 (simulating interruption before Round 1 starts)
    c_r0 = CountingDecisionClient(handler=ground_truth)
    res_r0 = run(
        rank(
            df,
            "speech",
            attributes={"clarity": "clear writing"},
            save_dir=save_dir,
            n_rounds=1,
            matches_per_round=2,
            decision_client=c_r0,
        )
    )
    assert c_r0.calls > 0
    assert (tmp_path / "rank_resume" / "rankings_round0.csv").exists()
    assert (
        res_r0.set_index("speech").loc["item_4", "clarity_raw"]
        > (res_r0.set_index("speech").loc["item_1", "clarity_raw"])
    )

    # 2. Resume with n_rounds=2: Round 0 replayed from CSV (0 calls for Round 0);
    #    Round 1 returns all "draw", so item_4 > item_1 ONLY holds if Round 0's
    #    DecisionResult JSON was faithfully replayed!
    c_r1 = CountingDecisionClient({"clarity": "draw"})
    res_r1 = run(
        rank(
            df,
            "speech",
            attributes={"clarity": "clear writing"},
            save_dir=save_dir,
            n_rounds=2,
            matches_per_round=2,
            reset_files=False,
            decision_client=c_r1,
        )
    )
    round1_df = pd.read_csv(tmp_path / "rank_resume" / "rankings_round1.csv")
    assert c_r1.calls == len(
        round1_df
    )  # ONLY Round 1 called; zero duplicate Round 0 calls
    by_speech_r1 = res_r1.set_index("speech")
    assert (
        by_speech_r1.loc["item_4", "clarity_raw"]
        > by_speech_r1.loc["item_1", "clarity_raw"]
    )

    # 3. Delete final CSV (simulating crash before final checkpoint write) ->
    #    replays both round CSVs with ZERO DecisionClient calls and matches scores!
    (tmp_path / "rank_resume" / "rankings_final.csv").unlink()
    c_replay = CountingDecisionClient({"clarity": "square"})
    res_rebuilt = run(
        rank(
            df,
            "speech",
            attributes={"clarity": "clear writing"},
            save_dir=save_dir,
            n_rounds=2,
            matches_per_round=2,
            reset_files=False,
            decision_client=c_replay,
        )
    )
    assert c_replay.calls == 0
    pd.testing.assert_frame_equal(res_rebuilt, res_r1)

    # 4. Re-running completed tournament makes ZERO DecisionClient calls
    c_done = CountingDecisionClient({"clarity": "square"})
    res_done = run(
        rank(
            df,
            "speech",
            attributes={"clarity": "clear writing"},
            save_dir=save_dir,
            n_rounds=2,
            matches_per_round=2,
            reset_files=False,
            decision_client=c_done,
        )
    )
    assert c_done.calls == 0
    pd.testing.assert_frame_equal(res_done, res_r1)

    # 5. reset_files=True forces full recomputation (matches_per_round=3 on 4
    #    items is the regular complete graph K_4, so all draws yield exact 0.0)
    c_reset = CountingDecisionClient({"clarity": "draw"})
    res_reset = run(
        rank(
            df,
            "speech",
            attributes={"clarity": "clear writing"},
            save_dir=save_dir,
            n_rounds=1,
            matches_per_round=3,
            reset_files=True,
            decision_client=c_reset,
        )
    )
    assert c_reset.calls > 0
    assert res_reset["clarity_raw"].abs().max() == pytest.approx(0.0, abs=1e-6)


def test_rank_checkpoint_resume_ignores_incomplete_round_csv(tmp_path) -> None:
    """A partial round CSV missing Batch/IdA/IdB does not skip that round on resume."""
    df = pd.DataFrame({"speech": ["low", "mid", "high"]})
    strength = {"low": 1, "mid": 2, "high": 3}
    save_dir = str(tmp_path / "rank_partial")

    def ground_truth(state: Any, key: str, question: Any) -> str:
        return (
            "circle"
            if strength[state["circle"]] > strength[state["square"]]
            else "square"
        )

    # Complete Round 0
    c_r0 = CountingDecisionClient(handler=ground_truth)
    run(
        rank(
            df,
            "speech",
            attributes={"clarity": "clear"},
            save_dir=save_dir,
            n_rounds=1,
            matches_per_round=2,
            decision_client=c_r0,
        )
    )

    # Simulate an aborted mid-batch flush for Round 1 (no Batch/Pair/IdA/IdB columns)
    partial_round1 = tmp_path / "rank_partial" / "rankings_round1.csv"
    pd.DataFrame(
        [
            {
                "Identifier": "stale_partial_id",
                "Response": "",
                "Successful": False,
                "Error Log": "Interrupted",
                "Time Taken": 0.1,
            }
        ]
    ).to_csv(partial_round1, index=False)

    # Resume n_rounds=2 -> Round 0 replayed (0 calls), Round 1 actually runs and completes
    c_r1 = CountingDecisionClient(handler=ground_truth)
    res = run(
        rank(
            df,
            "speech",
            attributes={"clarity": "clear"},
            save_dir=save_dir,
            n_rounds=2,
            matches_per_round=2,
            reset_files=False,
            decision_client=c_r1,
        )
    )
    assert c_r1.calls > 0
    r1_df = pd.read_csv(partial_round1)
    assert {"Batch", "Pair", "IdA", "IdB"}.issubset(r1_df.columns)
    assert (
        res.set_index("speech").loc["high", "clarity_raw"]
        > (res.set_index("speech").loc["low", "clarity_raw"])
    )


def test_rank_checkpoint_resume_catchup_new_items_uses_decision_client(
    tmp_path,
) -> None:
    """New items added on resume catch up on completed rounds via DecisionClient."""
    df_init = pd.DataFrame({"speech": ["low", "mid", "high"]})
    df_more = pd.DataFrame({"speech": ["low", "mid", "high", "top"]})
    strength = {"low": 1, "mid": 2, "high": 3, "top": 4}
    save_dir = str(tmp_path / "rank_catchup")

    def ground_truth(state: Any, key: str, question: Any) -> str:
        return (
            "circle"
            if strength[state["circle"]] > strength[state["square"]]
            else "square"
        )

    c_r0 = CountingDecisionClient(handler=ground_truth)
    run(
        rank(
            df_init,
            "speech",
            attributes={"clarity": "clear"},
            save_dir=save_dir,
            n_rounds=1,
            matches_per_round=2,
            decision_client=c_r0,
        )
    )
    r0_initial_rows = len(
        pd.read_csv(tmp_path / "rank_catchup" / "rankings_round0.csv")
    )

    # Add "top" and resume for 2 rounds; LLMClient must never be called
    llm_unused = CountingClient({"clarity": "circle"})
    c_resume = CountingDecisionClient(handler=ground_truth)
    res = run(
        rank(
            df_more,
            "speech",
            attributes={"clarity": "clear"},
            save_dir=save_dir,
            n_rounds=2,
            matches_per_round=2,
            reset_files=False,
            client=llm_unused,
            decision_client=c_resume,
        )
    )

    assert llm_unused.calls == 0
    assert c_resume.calls > 0
    assert len(res) == 4
    assert (res["clarity_se"] > 0).all()
    by_speech = res.set_index("speech")
    assert by_speech.loc["top", "clarity_raw"] > by_speech.loc["low", "clarity_raw"]

    # Round 0 CSV retains metadata for both original and catch-up rows
    r0_after = pd.read_csv(tmp_path / "rank_catchup" / "rankings_round0.csv")
    assert len(r0_after) > r0_initial_rows
    assert r0_after["Batch"].notna().all()
    assert (r0_after["IdA"].astype(str) != "").all()
    assert (r0_after["IdB"].astype(str) != "").all()


def test_rank_end_to_end_with_systemone_mock_transport_non_ascii_and_refusal(
    tmp_path,
) -> None:
    """End-to-end rank + SystemOneClient handles non-ASCII attributes and OpenRouter refusals."""
    import json

    import httpx

    from leximeter.client import SystemOneClient

    strength = {"短句": 1, "中等清晰表达": 2, "极其清晰的论点": 3}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert list(body["questions"].keys()) == ["q_0", "q_1"]
        for q_payload in body["questions"].values():
            assert q_payload["type"] == "choice"
            assert set(q_payload["criteria"].keys()) == {
                "circle",
                "square",
                "draw",
                "insufficient_signal",
            }
        state = body["state"]
        winner = (
            "circle"
            if strength[state["circle"]] > strength[state["square"]]
            else "square"
        )
        probs = {
            opt: (0.9 if opt == winner else 0.033)
            for opt in ("circle", "square", "draw", "insufficient_signal")
        }
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14.0",
                "answers": {
                    "q_0": {
                        "type": "choice",
                        "choice": winner,
                        "probabilities": probs,
                        "confidence": 0.9,
                    },
                    "q_1": {"type": "refusal"},
                },
                "usage": {"input_tokens": 50, "output_tokens": 0},
            },
        )

    client = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    df = pd.DataFrame({"text": ["短句", "中等清晰表达", "极其清晰的论点"]})
    k1 = "清晰度 (> 80%)"
    k2 = "敏感话题: 风险"

    res = run(
        rank(
            df,
            "text",
            attributes={k1: "表达清晰、逻辑明确", k2: "涉及敏感风险"},
            save_dir=str(tmp_path / "rank_e2e_sys1"),
            n_rounds=2,
            matches_per_round=2,
            decision_model="jev-latest",
            decision_client=client,
        )
    )

    assert list(res.columns) == [
        "text",
        k1,
        f"{k1}_raw",
        f"{k1}_se",
        k2,
        f"{k2}_raw",
        f"{k2}_se",
    ]
    by_text = res.set_index("text")
    assert by_text.loc["极其清晰的论点", f"{k1}_raw"] > by_text.loc["短句", f"{k1}_raw"]
    assert (res[f"{k1}_se"] > 0).all()
    assert (res[k2] == 0.0).all()
    assert res[f"{k2}_se"].isna().all()


def test_rank_resume_rechunks_when_provider_cap_shrinks(tmp_path) -> None:
    """Resuming rank with a smaller max_questions_per_call re-chunks instead of violating cap."""
    df = pd.DataFrame({"speech": ["low", "mid", "high"]})
    attributes = {"a": "desc a", "b": "desc b", "c": "desc c"}
    save_dir = str(tmp_path / "rank_resume_cap")

    c_large = CountingDecisionClient(
        {"a": "circle", "b": "square", "c": "draw"}, max_questions_per_call=64
    )
    run(
        rank(
            df,
            "speech",
            attributes=attributes,
            save_dir=save_dir,
            n_rounds=1,
            matches_per_round=2,
            decision_client=c_large,
        )
    )
    assert c_large.calls > 0
    assert all(len(qs) == 3 for _, qs, _ in c_large.requests)

    c_small = CountingDecisionClient(
        {"a": "circle", "b": "square", "c": "draw"}, max_questions_per_call=2
    )
    res = run(
        rank(
            df,
            "speech",
            attributes=attributes,
            save_dir=save_dir,
            n_rounds=2,
            matches_per_round=2,
            reset_files=False,
            decision_client=c_small,
        )
    )
    assert c_small.calls > 0
    assert all(len(qs) <= 2 for _, qs, _ in c_small.requests)
    assert {len(qs) for _, qs, _ in c_small.requests} == {1, 2}
    for attr in ("a", "b", "c"):
        assert res[f"{attr}_se"].notna().all()


def test_rank_systemone_missing_decision_model_fails_fast_without_disk_mutation(
    tmp_path, monkeypatch
) -> None:
    """SystemOneClient without decision_model fails fast before HTTP, disk resets, or empty/cached returns."""
    import httpx

    from leximeter.client import SystemOneClient

    http_calls = 0

    def fail_transport(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        raise AssertionError(f"Unexpected HTTP call to {request.url}")

    save_dir = tmp_path / "rank_fail_fast"
    save_dir.mkdir(parents=True, exist_ok=True)
    meta_path = save_dir / "rankings_run_metadata.json"
    attrs_path = save_dir / "attributes.json"
    legacy_attrs_path = save_dir / "rankings_attrs.json"
    round0_path = save_dir / "rankings_round0.csv"
    final_path = save_dir / "rankings_final.csv"

    meta_sentinel = '{"model": "prev-model"}'
    attrs_sentinel = '{"old": "kept"}'
    round0_sentinel = (
        "Batch,Pair,IdA,IdB,Attribute,Winner,Successful\n0,0,a,b,old,circle,True\n"
    )
    final_sentinel = "speech,old,old_raw,old_se\nitem1,0.0,0.0,0.1\n"
    meta_path.write_text(meta_sentinel, encoding="utf-8")
    attrs_path.write_text(attrs_sentinel, encoding="utf-8")
    legacy_attrs_path.write_text(attrs_sentinel, encoding="utf-8")
    round0_path.write_text(round0_sentinel, encoding="utf-8")
    final_path.write_text(final_sentinel, encoding="utf-8")

    sys1 = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=httpx.MockTransport(fail_transport),
    )
    df = pd.DataFrame({"speech": ["item1", "item2"]})
    empty_df = pd.DataFrame({"speech": pd.Series([], dtype="object")})

    for frame in (df, empty_df):
        for reset in (False, True):
            for bad_dm in (None, "   "):
                with pytest.raises(ValueError, match="decision_model"):
                    run(
                        rank(
                            frame,
                            "speech",
                            attributes={"clarity": "clear expression"},
                            save_dir=str(save_dir),
                            model="gpt-5.4-mini",
                            decision_model=bad_dm,
                            n_rounds=1,
                            matches_per_round=1,
                            decision_client=sys1,
                            reset_files=reset,
                        )
                    )

    monkeypatch.setenv("LEXIMETER_DECISION_PROTOCOL", "systemone")
    monkeypatch.setenv("LEXIMETER_DECISION_BASE_URL", "https://api.typesafe.ai")
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)
    with pytest.raises(ValueError, match="decision_model"):
        run(
            rank(
                df,
                "speech",
                attributes={"clarity": "clear expression"},
                save_dir=str(save_dir),
                model="gpt-5.4-mini",
                n_rounds=1,
                matches_per_round=1,
                reset_files=True,
            )
        )

    assert http_calls == 0
    assert meta_path.read_text(encoding="utf-8") == meta_sentinel
    assert attrs_path.read_text(encoding="utf-8") == attrs_sentinel
    assert legacy_attrs_path.read_text(encoding="utf-8") == attrs_sentinel
    assert round0_path.read_text(encoding="utf-8") == round0_sentinel
    assert final_path.read_text(encoding="utf-8") == final_sentinel
