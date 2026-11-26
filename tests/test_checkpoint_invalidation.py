"""Regression tests for Issue #15 checkpoint invalidation across tasks.

Verifies that provider toggles (``DecisionClient`` <-> ``LLMClient``),
``model`` / ``decision_model`` / ``LEXIMETER_DECISION_MODEL`` changes,
``Filter`` ``condition`` / ``additional_instructions`` changes, and
``reset_files=True`` cleanly invalidate and purge stale checkpoints across
``classify``, ``codify``, ``rank``, and ``filter`` — all offline.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from conftest import CountingClient, CountingDecisionClient

from leximeter import classify, codify, rank
from leximeter import filter as leximeter_filter
from leximeter.client import (
    LLMClient,
    NoulAnswer,
    SystemOneClient,
)
from leximeter.tasks._common import load_task_state


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class ScriptedLLMClient(LLMClient):
    """Offline LLMClient returning deterministic per-prompt JSON payloads."""

    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls = 0
        self.models: list[str] = []
        self.prompts: list[str] = []

    async def complete(
        self, prompt: str, *, model: str, json_mode: bool = False, **kwargs: Any
    ) -> str:
        del json_mode, kwargs
        self.calls += 1
        self.models.append(model)
        self.prompts.append(prompt)
        result = self._handler(prompt, model, self.calls)
        return result if isinstance(result, str) else json.dumps(result)

    def estimate_tokens(self, prompt: str, *, model: str) -> int:
        del prompt, model
        return 0

    def estimate_cost(self, prompt: str, *, model: str) -> float:
        del prompt, model
        return 0.0


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------


def test_classify_provider_toggle_bidirectional_with_same_model_name(
    tmp_path: Path,
) -> None:
    """DecisionClient <-> LLMClient toggles with identical model names reset checkpoints."""
    df = pd.DataFrame({"text": ["alpha item", "beta item"]})
    labels = {"pos": "positive sentiment", "neg": "negative sentiment"}
    save_dir = tmp_path / "cls_toggle_single"

    # 1. DecisionClient -> LLMClient -> DecisionClient (same model="shared-model")
    dc1 = CountingDecisionClient({"pos": 0.9, "neg": 0.1})
    res_dc1 = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(save_dir),
            model="shared-model",
            decision_model="shared-model",
            decision_client=dc1,
        )
    )
    assert dc1.calls == 2
    assert list(res_dc1["pos"]) == [True, True]
    assert list(res_dc1["neg"]) == [False, False]
    meta = json.loads(
        (save_dir / "classify_responses_run_metadata.json").read_text(encoding="utf-8")
    )
    assert meta["decision_provider"] is True
    assert meta["model"] == "shared-model"

    llm = CountingClient({"pos": False, "neg": True})
    res_llm = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(save_dir),
            model="shared-model",
            client=llm,
            decision_client=None,
            reset_files=False,
        )
    )
    assert llm.calls == 2
    assert list(res_llm["pos"]) == [False, False]
    assert list(res_llm["neg"]) == [True, True]
    meta = json.loads(
        (save_dir / "classify_responses_run_metadata.json").read_text(encoding="utf-8")
    )
    assert meta["decision_provider"] is False
    assert meta["model"] == "shared-model"

    dc2 = CountingDecisionClient({"pos": 0.8, "neg": 0.8})
    res_dc2 = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(save_dir),
            model="shared-model",
            decision_model="shared-model",
            decision_client=dc2,
            reset_files=False,
        )
    )
    assert dc2.calls == 2
    assert list(res_dc2["pos"]) == [True, True]
    assert list(res_dc2["neg"]) == [True, True]

    # 2. Differentiate mode with n_runs=2 -> n_runs=1 purges _full_disaggregated.csv
    pair_df = pd.DataFrame({"c": ["circle text"], "s": ["square text"]})
    pair_labels = {
        "tone_actual": "Circle is warmer",
        "tone_inverted": "Square is warmer",
    }
    diff_dir = tmp_path / "cls_toggle_diff"
    dc_diff = CountingDecisionClient({"tone": "option_a"})
    res_diff_dc = run(
        classify(
            pair_df,
            labels=pair_labels,
            save_dir=str(diff_dir),
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            n_runs=2,
            model="shared-model",
            decision_model="shared-model",
            decision_client=dc_diff,
        )
    )
    assert dc_diff.calls == 2
    assert list(res_diff_dc["tone_actual"]) == [True]
    assert list(res_diff_dc["tone_inverted"]) == [False]
    disagg_path = diff_dir / "classify_responses_full_disaggregated.csv"
    assert disagg_path.exists()

    llm_diff = CountingClient({"tone_actual": False, "tone_inverted": True})
    res_diff_llm = run(
        classify(
            pair_df,
            labels=pair_labels,
            save_dir=str(diff_dir),
            differentiate=True,
            circle_column_name="c",
            square_column_name="s",
            n_runs=1,
            model="shared-model",
            client=llm_diff,
            decision_client=None,
            reset_files=False,
        )
    )
    assert llm_diff.calls == 1
    assert list(res_diff_llm["tone_actual"]) == [False]
    assert list(res_diff_llm["tone_inverted"]) == [True]
    assert not disagg_path.exists()


def test_classify_model_and_env_change_invalidates_with_reset_files_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing model, decision_model, or LEXIMETER_DECISION_MODEL invalidates classify."""
    df = pd.DataFrame({"text": ["sample one"]})
    labels = {"flag": "contains flag"}

    # Decision model changes (explicit + env)
    dc_dir = tmp_path / "cls_dc_model"
    dc1 = CountingDecisionClient({"flag": 0.9})
    res1 = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(dc_dir),
            decision_model="dec-v1",
            decision_client=dc1,
        )
    )
    assert dc1.calls == 1
    assert list(res1["flag"]) == [True]

    dc2 = CountingDecisionClient({"flag": 0.1})
    res2 = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(dc_dir),
            decision_model="dec-v2",
            decision_client=dc2,
            reset_files=False,
        )
    )
    assert dc2.calls == 1
    assert list(res2["flag"]) == [False]

    monkeypatch.setenv("LEXIMETER_DECISION_MODEL", "dec-env-v3")
    dc3 = CountingDecisionClient({"flag": 0.95})
    res3 = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(dc_dir),
            decision_client=dc3,
            reset_files=False,
        )
    )
    assert dc3.calls == 1
    assert dc3.requests[0][2] == "dec-env-v3"
    assert list(res3["flag"]) == [True]

    # LLM model change
    llm_dir = tmp_path / "cls_llm_model"
    llm1 = CountingClient({"flag": True})
    res_l1 = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(llm_dir),
            model="gen-v1",
            client=llm1,
            decision_client=None,
        )
    )
    assert llm1.calls == 1
    assert list(res_l1["flag"]) == [True]

    llm2 = CountingClient({"flag": False})
    res_l2 = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(llm_dir),
            model="gen-v2",
            client=llm2,
            decision_client=None,
            reset_files=False,
        )
    )
    assert llm2.calls == 1
    assert list(res_l2["flag"]) == [False]


def test_classify_interrupted_reset_or_provider_toggle_does_not_leak_prior_rows(
    tmp_path: Path,
) -> None:
    """Interrupted reset/toggle purges prior raw CSV so Run 1 rows cannot leak into Run 3."""
    df = pd.DataFrame({"text": ["item_1", "item_2"]})
    labels = {"flag": "meets flag"}
    save_dir = tmp_path / "cls_interrupted_reset"

    # Run 1: completes on DecisionClient
    dc = CountingDecisionClient({"flag": 0.95})
    res1 = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(save_dir),
            model="shared-model",
            decision_model="shared-model",
            decision_client=dc,
        )
    )
    assert list(res1["flag"]) == [True, True]

    # Run 2: toggles to LLMClient, fails on item_2 (max_retries=0)
    class FlakyLLMClient(LLMClient):
        def __init__(self) -> None:
            self.fail_item_2 = True
            self.seen_items: list[str] = []

        async def complete(
            self, prompt: str, *, model: str, json_mode: bool = False, **kwargs: Any
        ) -> str:
            del model, json_mode, kwargs
            if "item_2" in prompt:
                self.seen_items.append("item_2")
                if self.fail_item_2:
                    raise RuntimeError("interrupted on item_2")
            else:
                self.seen_items.append("item_1")
            return json.dumps({"flag": False})

        def estimate_tokens(self, prompt: str, *, model: str) -> int:
            del prompt, model
            return 0

        def estimate_cost(self, prompt: str, *, model: str) -> float:
            del prompt, model
            return 0.0

    flaky = FlakyLLMClient()
    res2 = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(save_dir),
            model="shared-model",
            client=flaky,
            decision_client=None,
            reset_files=False,
            max_retries=0,
        )
    )
    assert bool(res2["flag"].iloc[0]) is False
    assert pd.isna(res2["flag"].iloc[1])

    # Run 3: resume LLMClient with reset_files=False -> only item_2 runs, no Run 1 True leaks
    flaky.fail_item_2 = False
    flaky.seen_items.clear()
    res3 = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(save_dir),
            model="shared-model",
            client=flaky,
            decision_client=None,
            reset_files=False,
            max_retries=0,
        )
    )
    assert flaky.seen_items == ["item_2"]
    assert list(res3["flag"]) == [False, False]


# ---------------------------------------------------------------------------
# Codify
# ---------------------------------------------------------------------------


def test_codify_reset_files_true_multi_round_purges_all_rounds_and_completion_checks(
    tmp_path: Path,
) -> None:
    """codify(..., reset_files=True, n_rounds=3) purges iter1+ and completion_iter1+."""
    df = pd.DataFrame(
        {
            "text": [
                "First alpha claim one. Second beta claim two. Third gamma claim three.",
            ]
        }
    )
    categories = {"alpha": "alpha claims"}
    save_dir = tmp_path / "codify_reset_3"

    def llm_run1(prompt: str, model: str, call_idx: int) -> dict[str, Any]:
        del model, call_idx
        if "First alpha" in prompt:
            return {
                "alpha": [
                    {"beginning excerpt": "First alpha", "ending excerpt": "claim one."}
                ]
            }
        if "Second beta" in prompt:
            return {
                "alpha": [
                    {"beginning excerpt": "Second beta", "ending excerpt": "claim two."}
                ]
            }
        return {
            "alpha": [
                {"beginning excerpt": "Third gamma", "ending excerpt": "claim three."}
            ]
        }

    dc_run1 = CountingDecisionClient({"alpha": 0.90})
    llm1 = ScriptedLLMClient(llm_run1)
    res1 = run(
        codify(
            df,
            "text",
            categories=categories,
            save_dir=str(save_dir),
            n_rounds=3,
            client=llm1,
            decision_client=dc_run1,
        )
    )
    assert res1["alpha"].iloc[0] == [
        "First alpha claim one.",
        "Second beta claim two.",
        "Third gamma claim three.",
    ]
    assert (save_dir / "coding_results.csv").exists()
    assert (save_dir / "coding_results_iter1.csv").exists()
    assert (save_dir / "coding_results_iter2.csv").exists()
    assert (
        save_dir
        / "completion_checks"
        / "coding_results_completion_iter0_raw_responses.csv"
    ).exists()
    assert (
        save_dir
        / "completion_checks"
        / "coding_results_completion_iter1_raw_responses.csv"
    ).exists()

    # Run 2 with reset_files=True, n_rounds=3, and new payloads (only iter0 extracts, dc stops loop)
    def llm_run2(prompt: str, model: str, call_idx: int) -> dict[str, Any]:
        del prompt, model, call_idx
        return {
            "alpha": [
                {"beginning excerpt": "Third gamma", "ending excerpt": "claim three."}
            ]
        }

    dc_run2 = CountingDecisionClient({"alpha": 0.10})
    llm2 = ScriptedLLMClient(llm_run2)
    res2 = run(
        codify(
            df,
            "text",
            categories=categories,
            save_dir=str(save_dir),
            n_rounds=3,
            client=llm2,
            decision_client=dc_run2,
            reset_files=True,
        )
    )
    assert llm2.calls == 1
    assert dc_run2.calls == 1
    assert res2["alpha"].iloc[0] == ["Third gamma claim three."]
    assert not (save_dir / "coding_results_iter1.csv").exists()
    assert not (save_dir / "coding_results_iter2.csv").exists()
    assert not (
        save_dir
        / "completion_checks"
        / "coding_results_completion_iter1_raw_responses.csv"
    ).exists()


def test_codify_interrupted_in_progress_resume_on_model_or_provider_change(
    tmp_path: Path,
) -> None:
    """Mid-run crash leaves status='in_progress' and invalidates checkpoints on provider/model change."""
    df = pd.DataFrame({"text": ["Alpha piece one. Beta piece two."]})
    categories = {"alpha": "alpha claims"}
    save_dir = tmp_path / "codify_in_progress"

    class SimulatedCrash(BaseException):
        pass

    crash_on_iter1 = True

    def llm_handler(prompt: str, model: str, call_idx: int) -> dict[str, Any]:
        del model, call_idx
        if "Alpha piece" in prompt:
            return {
                "alpha": [
                    {"beginning excerpt": "Alpha piece", "ending excerpt": "one."}
                ]
            }
        if crash_on_iter1:
            raise SimulatedCrash("crash inside iter1")
        return {
            "alpha": [{"beginning excerpt": "Beta piece", "ending excerpt": "two."}]
        }

    llm = ScriptedLLMClient(llm_handler)
    dc_v1 = CountingDecisionClient({"alpha": 0.9})
    with pytest.raises(SimulatedCrash, match="crash inside iter1"):
        run(
            codify(
                df,
                "text",
                categories=categories,
                save_dir=str(save_dir),
                model="gen-v1",
                decision_model="dec-v1",
                n_rounds=2,
                client=llm,
                decision_client=dc_v1,
                max_retries=0,
            )
        )

    state = load_task_state(str(save_dir), "codify")
    assert state is not None
    assert state["status"] == "in_progress"
    assert (save_dir / "coding_results.csv").exists()
    assert (
        save_dir
        / "completion_checks"
        / "coding_results_completion_iter0_raw_responses.csv"
    ).exists()

    # Resume with reset_files=False and changed decision_model -> purges and succeeds
    crash_on_iter1 = False
    dc_v2 = CountingDecisionClient({"alpha": 0.9})
    res = run(
        codify(
            df,
            "text",
            categories=categories,
            save_dir=str(save_dir),
            model="gen-v1",
            decision_model="dec-v2",
            n_rounds=2,
            client=llm,
            decision_client=dc_v2,
            reset_files=False,
        )
    )
    assert dc_v2.calls == 1
    assert res["alpha"].iloc[0] == ["Alpha piece one.", "Beta piece two."]
    done_state = load_task_state(str(save_dir), "codify")
    assert done_state is not None
    assert done_state["status"] == "complete"


def test_codify_unlinked_state_manifest_purges_completion_and_colliding_round_batches(
    tmp_path: Path,
) -> None:
    """Even if codify_state.json is missing, substage metadata resets completion_checks and iter1+."""
    df = pd.DataFrame({"text": ["Part one first. Part two second."]})
    categories = {"alpha": "alpha claims", "beta": "beta claims"}
    save_dir = tmp_path / "codify_unlinked_state"

    def extract_both(prompt: str, model: str, call_idx: int) -> dict[str, Any]:
        del model, call_idx
        if "Part one" in prompt:
            return {
                "alpha": [{"beginning excerpt": "Part one", "ending excerpt": "first."}]
            }
        if "alpha" in prompt and "beta" not in prompt:
            return {
                "alpha": [
                    {"beginning excerpt": "Part two", "ending excerpt": "second."}
                ]
            }
        if "beta" in prompt and "alpha" not in prompt:
            return {
                "beta": [{"beginning excerpt": "Part two", "ending excerpt": "second."}]
            }
        return {}

    llm1 = ScriptedLLMClient(extract_both)
    dc_v1 = CountingDecisionClient({"alpha": 0.9, "beta": 0.1})
    res1 = run(
        codify(
            df,
            "text",
            categories=categories,
            save_dir=str(save_dir),
            model="gen-v1",
            decision_model="dec-v1",
            n_rounds=2,
            client=llm1,
            decision_client=dc_v1,
        )
    )
    assert res1["alpha"].iloc[0] == ["Part one first.", "Part two second."]
    assert res1["beta"].iloc[0] == []
    assert (save_dir / "coding_results_iter1.csv").exists()

    # Simulate missing codify_state.json while coding_results*.csv and completion_checks/ remain
    (save_dir / "codify_state.json").unlink()

    # Resume with dc_v2 flagging "beta" instead of "alpha" (positional batch_idx=0 would collide!)
    llm2 = ScriptedLLMClient(extract_both)
    dc_v2 = CountingDecisionClient({"alpha": 0.1, "beta": 0.9})
    res2 = run(
        codify(
            df,
            "text",
            categories=categories,
            save_dir=str(save_dir),
            model="gen-v1",
            decision_model="dec-v2",
            n_rounds=2,
            client=llm2,
            decision_client=dc_v2,
            reset_files=False,
        )
    )
    assert dc_v2.calls == 1
    assert res2["alpha"].iloc[0] == ["Part one first."]
    assert res2["beta"].iloc[0] == ["Part two second."]

    # Unlink codify_state.json again and switch completion check to LLMClient + new gen model
    (save_dir / "codify_state.json").unlink()

    def llm3_handler(prompt: str, model: str, call_idx: int) -> dict[str, Any]:
        del model, call_idx
        if "BEGIN LABELS" in prompt:
            return {"alpha": True, "beta": False}
        if "Part one" in prompt:
            return {
                "alpha": [{"beginning excerpt": "Part one", "ending excerpt": "first."}]
            }
        return {
            "alpha": [{"beginning excerpt": "Part two", "ending excerpt": "second."}]
        }

    llm3 = ScriptedLLMClient(llm3_handler)
    res3 = run(
        codify(
            df,
            "text",
            categories=categories,
            save_dir=str(save_dir),
            model="gen-v2",
            n_rounds=2,
            client=llm3,
            decision_client=None,
            reset_files=False,
        )
    )
    assert llm3.calls >= 3  # iter0 + completion check + iter1
    assert res3["alpha"].iloc[0] == ["Part one first.", "Part two second."]
    assert res3["beta"].iloc[0] == []


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------


def test_rank_decision_model_and_env_change_bypasses_final_cache_and_purges_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rank invalidates final.csv cache and purges round*.csv on decision_model/env changes."""
    df = pd.DataFrame({"speech": ["item_a", "item_b", "item_c"]})
    attributes = {"clarity": "clear reasoning"}
    save_dir = tmp_path / "rank_model_purge"

    # Run 1: n_rounds=3 with dec-v1 -> creates round0, round1, round2, final
    def always_circle(state: Any, key: str, q: Any) -> str:
        del key, q
        # Prefer whichever text was presented first ("item_a" < "item_b" < "item_c")
        c = state["circle"]
        s = state["square"]
        return "circle" if c < s else "square"

    def always_square(state: Any, key: str, q: Any) -> str:
        del key, q
        c = state["circle"]
        s = state["square"]
        return "square" if c < s else "circle"

    c1 = CountingDecisionClient(handler=always_circle)
    res1 = run(
        rank(
            df,
            "speech",
            attributes=attributes,
            save_dir=str(save_dir),
            n_rounds=3,
            matches_per_round=2,
            circle_first=True,
            decision_model="dec-v1",
            decision_client=c1,
        )
    )
    assert c1.calls > 0
    assert (save_dir / "rankings_round0.csv").exists()
    assert (save_dir / "rankings_round1.csv").exists()
    assert (save_dir / "rankings_round2.csv").exists()
    by_text_1 = res1.set_index("speech")
    assert (
        by_text_1.loc["item_a", "clarity_raw"] > by_text_1.loc["item_c", "clarity_raw"]
    )

    # Run 2: decision_model="dec-v2" with n_rounds=1, reset_files=False -> must purge round1/round2
    c2 = CountingDecisionClient(handler=always_square)
    res2 = run(
        rank(
            df,
            "speech",
            attributes=attributes,
            save_dir=str(save_dir),
            n_rounds=1,
            matches_per_round=2,
            circle_first=True,
            decision_model="dec-v2",
            decision_client=c2,
            reset_files=False,
        )
    )
    assert c2.calls > 0
    assert (save_dir / "rankings_round0.csv").exists()
    assert not (save_dir / "rankings_round1.csv").exists()
    assert not (save_dir / "rankings_round2.csv").exists()
    by_text_2 = res2.set_index("speech")
    assert (
        by_text_2.loc["item_c", "clarity_raw"] > by_text_2.loc["item_a", "clarity_raw"]
    )

    # Run 3: LEXIMETER_DECISION_MODEL env change invalidates completed cache
    monkeypatch.setenv("LEXIMETER_DECISION_MODEL", "dec-env-v3")
    c3 = CountingDecisionClient(handler=always_circle)
    res3 = run(
        rank(
            df,
            "speech",
            attributes=attributes,
            save_dir=str(save_dir),
            n_rounds=1,
            matches_per_round=2,
            circle_first=True,
            decision_client=c3,
            reset_files=False,
        )
    )
    assert c3.calls > 0
    assert all(m == "dec-env-v3" for _, _, m in c3.requests)
    by_text_3 = res3.set_index("speech")
    assert (
        by_text_3.loc["item_a", "clarity_raw"] > by_text_3.loc["item_c", "clarity_raw"]
    )


def test_rank_provider_toggle_bidirectional_with_same_model_name_and_llm_model_change(
    tmp_path: Path,
) -> None:
    """Rank invalidates on DecisionClient <-> LLMClient (same model) and LLM model changes."""
    df = pd.DataFrame({"speech": ["alpha", "beta", "gamma"]})
    attributes = {"quality": "overall quality"}
    save_dir = tmp_path / "rank_provider_toggle"

    def llm_prefer_circle(prompt: str, model: str, call_idx: int) -> dict[str, str]:
        del prompt, model, call_idx
        return {"quality": "circle"}

    def llm_prefer_square(prompt: str, model: str, call_idx: int) -> dict[str, str]:
        del prompt, model, call_idx
        return {"quality": "square"}

    # 1. LLMClient (shared) -> DecisionClient (shared) partial/expanded n_rounds=1 -> 2
    llm1 = ScriptedLLMClient(llm_prefer_circle)
    run(
        rank(
            df,
            "speech",
            attributes=attributes,
            save_dir=str(save_dir),
            n_rounds=1,
            matches_per_round=2,
            circle_first=True,
            model="shared-model",
            client=llm1,
            decision_client=None,
        )
    )
    assert llm1.calls > 0
    meta = json.loads(
        (save_dir / "rankings_run_metadata.json").read_text(encoding="utf-8")
    )
    assert meta["decision_provider"] is False

    # Switch to DecisionClient with same model="shared-model", expanding n_rounds=2 -> must reset round0!
    dc = CountingDecisionClient({"quality": "square"})
    run(
        rank(
            df,
            "speech",
            attributes=attributes,
            save_dir=str(save_dir),
            n_rounds=2,
            matches_per_round=2,
            circle_first=True,
            model="shared-model",
            decision_model="shared-model",
            decision_client=dc,
            reset_files=False,
        )
    )
    # Two rounds with 3 pairs each -> 6 calls (Round 0 was NOT replayed from LLMClient)
    assert dc.calls == 6
    meta = json.loads(
        (save_dir / "rankings_run_metadata.json").read_text(encoding="utf-8")
    )
    assert meta["decision_provider"] is True

    # 2. DecisionClient (shared) -> LLMClient (shared) on completed run
    llm2 = ScriptedLLMClient(llm_prefer_square)
    run(
        rank(
            df,
            "speech",
            attributes=attributes,
            save_dir=str(save_dir),
            n_rounds=2,
            matches_per_round=2,
            circle_first=True,
            model="shared-model",
            client=llm2,
            decision_client=None,
            reset_files=False,
        )
    )
    assert llm2.calls == 6
    meta = json.loads(
        (save_dir / "rankings_run_metadata.json").read_text(encoding="utf-8")
    )
    assert meta["decision_provider"] is False

    # 3. LLMClient model change (shared-model -> gen-v2) bypasses final.csv cache
    llm3 = ScriptedLLMClient(llm_prefer_circle)
    run(
        rank(
            df,
            "speech",
            attributes=attributes,
            save_dir=str(save_dir),
            n_rounds=2,
            matches_per_round=2,
            circle_first=True,
            model="gen-v2",
            client=llm3,
            decision_client=None,
            reset_files=False,
        )
    )
    assert llm3.calls == 6


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def test_filter_decision_invalidates_on_model_env_condition_and_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DecisionClient filter invalidates on model, env, condition, or instructions changes."""
    df = pd.DataFrame({"item": ["apple", "banana"]})
    save_dir = tmp_path / "flt_dc_invalidation"

    c1 = CountingDecisionClient(default=0.9)
    res1 = run(
        leximeter_filter(
            df,
            "item",
            condition="is red",
            additional_instructions="None",
            save_dir=str(save_dir),
            shuffle=False,
            decision_model="dec-v1",
            decision_client=c1,
        )
    )
    assert c1.calls == 1
    assert list(res1["keep"]) == [True, True]

    # Whitespace-only additional_instructions vs None -> should NOT reset (0 calls)
    c_ws = CountingDecisionClient(default=0.1)
    run(
        leximeter_filter(
            df,
            "item",
            condition="is red",
            additional_instructions="None",
            save_dir=str(save_dir),
            shuffle=False,
            decision_model="dec-v1",
            decision_client=c_ws,
            reset_files=False,
        )
    )
    assert c_ws.calls == 0

    # Changing additional_instructions -> resets
    c2 = CountingDecisionClient(default=0.2)
    res2 = run(
        leximeter_filter(
            df,
            "item",
            condition="is red",
            additional_instructions="be strict",
            save_dir=str(save_dir),
            shuffle=False,
            decision_model="dec-v1",
            decision_client=c2,
            reset_files=False,
        )
    )
    assert c2.calls == 1
    assert list(res2["keep"]) == [False, False]

    # Changing from "be strict" to None vs "   " after setting None
    c3 = CountingDecisionClient(default=0.8)
    run(
        leximeter_filter(
            df,
            "item",
            condition="is red",
            additional_instructions=None,
            save_dir=str(save_dir),
            shuffle=False,
            decision_model="dec-v1",
            decision_client=c3,
            reset_files=False,
        )
    )
    assert c3.calls == 1

    c_empty_ws = CountingDecisionClient(default=0.1)
    run(
        leximeter_filter(
            df,
            "item",
            condition="is red",
            additional_instructions="   ",
            save_dir=str(save_dir),
            shuffle=False,
            decision_model="dec-v1",
            decision_client=c_empty_ws,
            reset_files=False,
        )
    )
    assert c_empty_ws.calls == 0

    # Changing condition -> resets
    c4 = CountingDecisionClient(default=0.3)
    res4 = run(
        leximeter_filter(
            df,
            "item",
            condition="is yellow",
            additional_instructions=None,
            save_dir=str(save_dir),
            shuffle=False,
            decision_model="dec-v1",
            decision_client=c4,
            reset_files=False,
        )
    )
    assert c4.calls == 1
    assert list(res4["keep"]) == [False, False]

    # Changing decision_model -> resets
    c5 = CountingDecisionClient(default=0.85)
    run(
        leximeter_filter(
            df,
            "item",
            condition="is yellow",
            save_dir=str(save_dir),
            shuffle=False,
            decision_model="dec-v2",
            decision_client=c5,
            reset_files=False,
        )
    )
    assert c5.calls == 1

    # Changing LEXIMETER_DECISION_MODEL env -> resets
    monkeypatch.setenv("LEXIMETER_DECISION_MODEL", "dec-env-v3")
    c6 = CountingDecisionClient(default=0.15)
    res6 = run(
        leximeter_filter(
            df,
            "item",
            condition="is yellow",
            save_dir=str(save_dir),
            shuffle=False,
            decision_client=c6,
            reset_files=False,
        )
    )
    assert c6.calls == 1
    assert c6.requests[0][2] == "dec-env-v3"
    assert list(res6["keep"]) == [False, False]


def test_filter_llm_invalidates_on_model_condition_and_instructions(
    tmp_path: Path,
) -> None:
    """LLMClient filter invalidates on model, condition, or instructions changes."""
    df = pd.DataFrame({"item": ["apple", "banana"]})
    save_dir = tmp_path / "flt_llm_invalidation"

    c1 = CountingClient({"entities meeting condition": ["apple"]})
    res1 = run(
        leximeter_filter(
            df,
            "item",
            condition="is red",
            additional_instructions=None,
            save_dir=str(save_dir),
            shuffle=False,
            model="gen-v1",
            client=c1,
            decision_client=None,
        )
    )
    assert c1.calls == 1
    assert list(res1["keep"]) == [True, False]

    # Equivalent whitespace instructions -> 0 calls
    c_ws = CountingClient({"entities meeting condition": []})
    run(
        leximeter_filter(
            df,
            "item",
            condition="is red",
            additional_instructions="   ",
            save_dir=str(save_dir),
            shuffle=False,
            model="gen-v1",
            client=c_ws,
            decision_client=None,
            reset_files=False,
        )
    )
    assert c_ws.calls == 0

    # Model change -> resets
    c2 = CountingClient({"entities meeting condition": ["banana"]})
    res2 = run(
        leximeter_filter(
            df,
            "item",
            condition="is red",
            save_dir=str(save_dir),
            shuffle=False,
            model="gen-v2",
            client=c2,
            decision_client=None,
            reset_files=False,
        )
    )
    assert c2.calls == 1
    assert list(res2["keep"]) == [False, True]

    # Condition change -> resets
    c3 = CountingClient({"entities meeting condition": ["apple", "banana"]})
    res3 = run(
        leximeter_filter(
            df,
            "item",
            condition="is fruit",
            save_dir=str(save_dir),
            shuffle=False,
            model="gen-v2",
            client=c3,
            decision_client=None,
            reset_files=False,
        )
    )
    assert c3.calls == 1
    assert list(res3["keep"]) == [True, True]

    # Additional instructions change -> resets
    c4 = CountingClient({"entities meeting condition": []})
    res4 = run(
        leximeter_filter(
            df,
            "item",
            condition="is fruit",
            additional_instructions="strict mode",
            save_dir=str(save_dir),
            shuffle=False,
            model="gen-v2",
            client=c4,
            decision_client=None,
            reset_files=False,
        )
    )
    assert c4.calls == 1
    assert list(res4["keep"]) == [False, False]


def test_filter_provider_toggle_bidirectional_with_same_model_name(
    tmp_path: Path,
) -> None:
    """Filter purges raw_responses.csv and flips decision_provider on same-model provider toggle."""
    df = pd.DataFrame({"item": ["apple", "banana"]})
    save_dir = tmp_path / "flt_provider_toggle"

    llm1 = CountingClient({"entities meeting condition": ["apple"]})
    res1 = run(
        leximeter_filter(
            df,
            "item",
            condition="is fruit",
            save_dir=str(save_dir),
            shuffle=False,
            model="shared-model",
            client=llm1,
            decision_client=None,
        )
    )
    assert llm1.calls == 1
    assert list(res1["keep"]) == [True, False]
    raw_df_1 = pd.read_csv(save_dir / "filter_responses_raw_responses.csv")
    assert len(raw_df_1) == 1
    meta = json.loads(
        (save_dir / "filter_responses_run_metadata.json").read_text(encoding="utf-8")
    )
    assert meta["decision_provider"] is False

    # LLMClient -> DecisionClient (same model name)
    dc = CountingDecisionClient(default=0.9)
    res2 = run(
        leximeter_filter(
            df,
            "item",
            condition="is fruit",
            save_dir=str(save_dir),
            shuffle=False,
            model="shared-model",
            decision_model="shared-model",
            decision_client=dc,
            reset_files=False,
        )
    )
    assert dc.calls == 1
    assert list(res2["keep"]) == [True, True]
    raw_df_2 = pd.read_csv(save_dir / "filter_responses_raw_responses.csv")
    assert len(raw_df_2) == 1
    assert "answers" in str(raw_df_2["Response"].iloc[0])
    meta = json.loads(
        (save_dir / "filter_responses_run_metadata.json").read_text(encoding="utf-8")
    )
    assert meta["decision_provider"] is True

    # DecisionClient -> LLMClient (same model name)
    llm2 = CountingClient({"entities meeting condition": []})
    res3 = run(
        leximeter_filter(
            df,
            "item",
            condition="is fruit",
            save_dir=str(save_dir),
            shuffle=False,
            model="shared-model",
            client=llm2,
            decision_client=None,
            reset_files=False,
        )
    )
    assert llm2.calls == 1
    assert list(res3["keep"]) == [False, False]
    raw_df_3 = pd.read_csv(save_dir / "filter_responses_raw_responses.csv")
    assert len(raw_df_3) == 1
    assert "entities meeting condition" in str(raw_df_3["Response"].iloc[0])
    meta = json.loads(
        (save_dir / "filter_responses_run_metadata.json").read_text(encoding="utf-8")
    )
    assert meta["decision_provider"] is False


def test_filter_preserves_saved_batch_size_when_unchanged_and_resets_when_changed(
    tmp_path: Path,
) -> None:
    """Filter preserves saved_batch_size when all 4 fields match, and resets on change or legacy metadata."""
    df = pd.DataFrame({"item": ["a", "b", "c", "d"]})
    save_dir = tmp_path / "flt_batch_size"

    # Run 1 with cap=2 -> saved_batch_size=2
    c_small = CountingDecisionClient(default=0.8, max_questions_per_call=2)
    run(
        leximeter_filter(
            df,
            "item",
            condition="is valid",
            save_dir=str(save_dir),
            shuffle=False,
            decision_model="dec-v1",
            decision_client=c_small,
        )
    )
    assert c_small.calls == 2

    # Run 2 with cap=64 and identical 4 fields -> saved_batch_size=2 preserved -> 0 calls
    c_large = CountingDecisionClient(default=0.1, max_questions_per_call=64)
    res_cached = run(
        leximeter_filter(
            df,
            "item",
            condition="is valid",
            save_dir=str(save_dir),
            shuffle=False,
            decision_model="dec-v1",
            decision_client=c_large,
            reset_files=False,
        )
    )
    assert c_large.calls == 0
    assert list(res_cached["keep"]) == [True, True, True, True]

    # Run 3 with condition changed and cap=64 -> resets and uses effective_batch_size=64 (1 call)
    res_changed = run(
        leximeter_filter(
            df,
            "item",
            condition="is different",
            save_dir=str(save_dir),
            shuffle=False,
            decision_model="dec-v1",
            decision_client=c_large,
            reset_files=False,
        )
    )
    assert c_large.calls == 1
    assert list(res_changed["keep"]) == [False, False, False, False]

    # Legacy metadata missing "condition" / "decision_provider" -> triggers clean reset
    meta_path = save_dir / "filter_responses_run_metadata.json"
    meta_path.write_text(
        json.dumps({"model": "dec-v1", "effective_batch_size": 2}),
        encoding="utf-8",
    )
    c_legacy = CountingDecisionClient(default=0.9, max_questions_per_call=64)
    res_legacy = run(
        leximeter_filter(
            df,
            "item",
            condition="is different",
            save_dir=str(save_dir),
            shuffle=False,
            decision_model="dec-v1",
            decision_client=c_legacy,
            reset_files=False,
        )
    )
    assert c_legacy.calls == 1
    assert list(res_legacy["keep"]) == [True, True, True, True]


# ---------------------------------------------------------------------------
# SystemOneClient Fail-Fast Preservation
# ---------------------------------------------------------------------------


def test_checkpoint_invalidation_preserves_systemone_fail_fast(
    tmp_path: Path,
) -> None:
    """Switching to SystemOneClient without decision_model raises ValueError before touching disk."""
    http_calls = 0

    def fail_transport(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        raise AssertionError(f"Unexpected HTTP call to {request.url}")

    sys1 = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=httpx.MockTransport(fail_transport),
    )

    # 1. Rank existing artifacts untouched
    rank_dir = tmp_path / "rank_sys1_preserve"
    rank_df = pd.DataFrame({"speech": ["a", "b"]})
    run(
        rank(
            rank_df,
            "speech",
            attributes={"clarity": "clear"},
            save_dir=str(rank_dir),
            n_rounds=1,
            matches_per_round=1,
            decision_model="dec-v1",
            decision_client=CountingDecisionClient({"clarity": "circle"}),
        )
    )
    rank_meta_before = (rank_dir / "rankings_run_metadata.json").read_text(
        encoding="utf-8"
    )
    rank_round0_before = (rank_dir / "rankings_round0.csv").read_text(encoding="utf-8")
    rank_final_before = (rank_dir / "rankings_final.csv").read_text(encoding="utf-8")

    for reset in (False, True):
        with pytest.raises(ValueError, match="decision_model"):
            run(
                rank(
                    rank_df,
                    "speech",
                    attributes={"clarity": "clear"},
                    save_dir=str(rank_dir),
                    n_rounds=1,
                    matches_per_round=1,
                    decision_model=None,
                    decision_client=sys1,
                    reset_files=reset,
                )
            )
    assert (rank_dir / "rankings_run_metadata.json").read_text(
        encoding="utf-8"
    ) == rank_meta_before
    assert (rank_dir / "rankings_round0.csv").read_text(
        encoding="utf-8"
    ) == rank_round0_before
    assert (rank_dir / "rankings_final.csv").read_text(
        encoding="utf-8"
    ) == rank_final_before

    # 2. Filter existing artifacts untouched
    flt_dir = tmp_path / "flt_sys1_preserve"
    flt_df = pd.DataFrame({"item": ["x", "y"]})
    run(
        leximeter_filter(
            flt_df,
            "item",
            condition="keep",
            save_dir=str(flt_dir),
            decision_model="dec-v1",
            decision_client=CountingDecisionClient(default=NoulAnswer(probability=0.9)),
        )
    )
    flt_meta_before = (flt_dir / "filter_responses_run_metadata.json").read_text(
        encoding="utf-8"
    )
    flt_raw_before = (flt_dir / "filter_responses_raw_responses.csv").read_text(
        encoding="utf-8"
    )
    flt_cleaned_before = (flt_dir / "filter_responses_cleaned.csv").read_text(
        encoding="utf-8"
    )

    for reset in (False, True):
        with pytest.raises(ValueError, match="decision_model"):
            run(
                leximeter_filter(
                    flt_df,
                    "item",
                    condition="keep",
                    save_dir=str(flt_dir),
                    decision_model=None,
                    decision_client=sys1,
                    reset_files=reset,
                )
            )
    assert (flt_dir / "filter_responses_run_metadata.json").read_text(
        encoding="utf-8"
    ) == flt_meta_before
    assert (flt_dir / "filter_responses_raw_responses.csv").read_text(
        encoding="utf-8"
    ) == flt_raw_before
    assert (flt_dir / "filter_responses_cleaned.csv").read_text(
        encoding="utf-8"
    ) == flt_cleaned_before
    assert http_calls == 0


def test_filter_legacy_untracked_raw_csv_invalidates_on_resume(tmp_path: Path) -> None:
    """Pre-metadata filter_responses_raw_responses.csv is invalidated rather than reused blindly."""
    from leximeter.tasks import _common

    save_dir = tmp_path / "flt_legacy_raw"
    save_dir.mkdir(parents=True, exist_ok=True)
    batch_hash = _common.hash_identifier(json.dumps(["apple"], ensure_ascii=False))
    pd.DataFrame(
        [
            {
                "Identifier": f"{batch_hash}_run1",
                "Response": json.dumps({"entities meeting condition": ["apple"]}),
                "Successful": True,
                "Error Log": "",
                "Time Taken": 0.01,
            }
        ]
    ).to_csv(save_dir / "filter_responses_raw_responses.csv", index=False)

    llm_reject = ScriptedLLMClient(lambda p, m, i: {"entities meeting condition": []})
    res = run(
        leximeter_filter(
            pd.DataFrame({"item": ["apple"]}),
            "item",
            condition="must be a mineral",
            save_dir=str(save_dir),
            model="gen-v1",
            reset_files=False,
            client=llm_reject,
        )
    )
    assert llm_reject.calls == 1
    assert list(res["keep"]) == [False]


def test_codify_early_exit_no_added_snippets_purges_trailing_rounds(
    tmp_path: Path,
) -> None:
    """Unlinked codify_state.json + early exit on 'not added' purges trailing round & completion files."""
    df = pd.DataFrame(
        {"text": ["First alpha claim one. Second beta remainder claim two."]}
    )
    categories = {"alpha": "alpha claims"}
    save_dir = tmp_path / "codify_no_added_exit"

    # Seed stale round 2 and completion_iter1 artifacts alongside missing codify_state.json
    save_dir.mkdir(parents=True, exist_ok=True)
    comp_dir = save_dir / "completion_checks"
    comp_dir.mkdir(parents=True, exist_ok=True)
    stale_iter2 = save_dir / "coding_results_iter2.csv"
    stale_comp1 = comp_dir / "coding_results_completion_iter1_raw_responses.csv"
    stale_iter2.write_text(
        "Identifier,Response,Successful,Error Log,Time Taken\n", encoding="utf-8"
    )
    stale_comp1.write_text(
        "Identifier,Response,Successful,Error Log,Time Taken\n", encoding="utf-8"
    )
    (comp_dir / "coding_results_completion_iter0_run_metadata.json").write_text(
        json.dumps({"model": "old-dec", "decision_provider": True}),
        encoding="utf-8",
    )

    def llm_no_new_in_round1(prompt: str, model: str, call_idx: int) -> dict[str, Any]:
        del prompt, model
        if call_idx == 1:
            return {
                "alpha": [
                    {"beginning excerpt": "First alpha", "ending excerpt": "claim one."}
                ]
            }
        # Round 1 returns empty -> triggers 'if not added: break'
        return {"alpha": []}

    llm = ScriptedLLMClient(llm_no_new_in_round1)
    dc = CountingDecisionClient({"alpha": 0.95})
    res = run(
        codify(
            df,
            "text",
            categories=categories,
            save_dir=str(save_dir),
            model="gen-v1",
            decision_model="new-dec",
            n_rounds=3,
            reset_files=False,
            client=llm,
            decision_client=dc,
        )
    )
    assert res["alpha"].iloc[0] == ["First alpha claim one."]
    assert not stale_iter2.exists()
    assert not stale_comp1.exists()
