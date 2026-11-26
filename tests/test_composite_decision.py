"""Public API / task-seam tests for composite tasks with DecisionClient.

Tests inject counting/recording ``LLMClient`` and ``DecisionClient`` fakes at
the public facade (``leximeter.discover``, ``leximeter.paraphrase``,
``leximeter.codify``, ``leximeter.debias``) and assert only external behavior:

- Generative stages run on ``LLMClient`` (``model``) while internal
  ``classify`` / ``rank`` stages delegate to ``DecisionClient``
  (``decision_model``).
- Completed-run manifests short-circuit unchanged re-runs with zero calls to
  both clients, and invalidate cleanly when ``decision_model`` (explicit or
  ``LEXIMETER_DECISION_MODEL``) or provider mode changes.
- When ``decision_client=None`` (and unconfigured in env), all four composite
  tasks fall back to ``LLMClient`` unchanged.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pandas as pd
import pytest
from conftest import CountingDecisionClient
from test_tasks import DebiasClient, StageClient

from leximeter import codify, debias, discover, paraphrase
from leximeter.client import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionClient,
    DecisionResult,
    LLMClient,
    NoulAnswer,
    NoulQuestion,
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class RecordingStageClient(StageClient):
    """StageClient that records every prompt and model passed to ``complete``."""

    def __init__(self, payloads: dict[str, Any]) -> None:
        super().__init__(payloads)
        self.calls = 0
        self.prompts: list[str] = []
        self.models: list[str] = []

    async def complete(
        self, prompt: str, *, model: str, json_mode: bool = False, **kwargs: Any
    ) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        self.models.append(model)
        return await super().complete(
            prompt, model=model, json_mode=json_mode, **kwargs
        )


def _discover_pair_df() -> pd.DataFrame:
    return pd.DataFrame({"circle": ["sunny", "rainy"], "square": ["cloudy", "clear"]})


def _discover_pair_payloads() -> dict[str, Any]:
    return {
        "compare": {"quality": "circle has more quality than square"},
        "bucket": {"quality": "a distinguishing feature"},
        "classify": {"quality": True, "quality (inverted)": False},
        "codify": {},
    }


def _discover_single_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                (
                    "This is a long passage. Hello world. "
                    "There is much more content here that remains uncoded."
                )
            ]
        }
    )


def _discover_single_payloads() -> dict[str, Any]:
    return {
        "codify": {
            "feature code": [
                {
                    "beginning excerpt": "Hello world",
                    "ending excerpt": "much more content",
                }
            ]
        },
        "bucket": {"feature code": "a distinguishing feature"},
        "classify": {"feature code": True},
        "compare": {},
    }


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


def test_discover_single_column_delegates_stage3_classify_to_decision_client(
    tmp_path,
) -> None:
    """Single-column discover runs codify/bucket on LLMClient and classify on DecisionClient."""
    df = _discover_single_df()
    llm = RecordingStageClient(_discover_single_payloads())

    def decide_noul(state: Any, key: str, question: Any) -> float:
        assert isinstance(question, NoulQuestion)
        assert key == "feature code"
        return 0.90

    dc = CountingDecisionClient(handler=decide_noul)

    res = run(
        discover(
            df,
            column_name="text",
            save_dir=str(tmp_path / "disc_single"),
            model="gen-model",
            decision_model="dec-model",
            bucket_count=1,
            additional_instructions="identify features",
            client=llm,
            decision_client=dc,
        )
    )

    assert llm.calls > 0
    assert all(m == "gen-model" for m in llm.models)
    # Stage 3 classify must NOT hit LLMClient
    assert not any("BEGIN LABELS" in p for p in llm.prompts)

    assert dc.calls == 1
    assert all(m == "dec-model" for _, _, m in dc.requests)

    assert res["buckets"] == {"feature code": "a distinguishing feature"}
    assert "codify" in res
    assert "summary" not in res
    assert list(res["classification"]["feature code"]) == [True]


def test_discover_pair_differentiate_delegates_stage3_choice_to_decision_client(
    tmp_path,
) -> None:
    """Paired discover runs compare/bucket on LLMClient and differentiate classify on DecisionClient."""
    df = _discover_pair_df()
    llm = RecordingStageClient(_discover_pair_payloads())

    def decide_pair(state: Any, key: str, question: Any) -> dict[str, float]:
        assert isinstance(question, ChoiceQuestion)
        assert set(question.criteria.keys()) == {"circle", "square", "neither"}
        assert isinstance(state, dict)
        # First row favours circle; second row favours square -> net_pct = 0.0
        if state["circle"] == "sunny":
            return {"circle": 0.85, "square": 0.05, "neither": 0.10}
        return {"circle": 0.05, "square": 0.85, "neither": 0.10}

    dc = CountingDecisionClient(handler=decide_pair)

    res = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=str(tmp_path / "disc_pair"),
            model="gen-model",
            decision_model="dec-model",
            bucket_count=1,
            client=llm,
            decision_client=dc,
        )
    )

    assert llm.calls > 0
    assert all(m == "gen-model" for m in llm.models)
    assert not any("BEGIN LABELS" in p for p in llm.prompts)

    assert dc.calls == 2
    assert all(m == "dec-model" for _, _, m in dc.requests)

    summary = res["summary"]
    assert summary["label"].iloc[0] == "quality"
    assert summary["actual_true"].iloc[0] == 1
    assert summary["inverted_true"].iloc[0] == 1
    assert summary["actual_pct"].iloc[0] == 50.0
    assert summary["inverted_pct"].iloc[0] == 50.0
    assert summary["net_pct"].iloc[0] == 0.0
    # Mutual exclusivity on differentiate output
    clf_df = res["classification"]
    assert not (clf_df["quality_actual"] & clf_df["quality_inverted"]).any()


def test_discover_completed_manifest_cache_and_decision_model_invalidation(
    tmp_path, monkeypatch
) -> None:
    """Unchanged rerun makes 0 calls to both clients; changing decision_model re-runs Stage 3."""
    monkeypatch.delenv("LEXIMETER_DECISION_PROTOCOL", raising=False)
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)
    df = _discover_pair_df()
    save_dir = str(tmp_path / "disc_sig")
    llm = RecordingStageClient(_discover_pair_payloads())

    # Run 1: explicit decision_model="dec-v1" -> circle always wins (net_pct = +100)
    dc_v1 = CountingDecisionClient(
        handler=lambda state, key, q: {"circle": 0.9, "square": 0.05, "neither": 0.05}
    )
    first = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            model="gen-model",
            decision_model="dec-v1",
            bucket_count=1,
            client=llm,
            decision_client=dc_v1,
        )
    )
    llm_after_v1 = llm.calls
    dc_after_v1 = dc_v1.calls
    assert llm_after_v1 > 0
    assert dc_after_v1 == 2
    assert first["summary"]["net_pct"].iloc[0] == 100.0

    # Run 2: unchanged -> 0 calls to BOTH clients
    second = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            model="gen-model",
            decision_model="dec-v1",
            bucket_count=1,
            client=llm,
            decision_client=dc_v1,
        )
    )
    assert llm.calls == llm_after_v1
    assert dc_v1.calls == dc_after_v1
    assert second["summary"]["net_pct"].iloc[0] == 100.0

    # Run 3: change decision_model via explicit arg -> invalidates manifest, re-runs Stage 3
    dc_v2 = CountingDecisionClient(
        handler=lambda state, key, q: {"circle": 0.05, "square": 0.9, "neither": 0.05}
    )
    third = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            model="gen-model",
            decision_model="dec-v2",
            bucket_count=1,
            client=llm,
            decision_client=dc_v2,
        )
    )
    assert llm.calls == llm_after_v1  # Stage 1 & 2 checkpoints reused
    assert dc_v2.calls == 2
    assert all(m == "dec-v2" for _, _, m in dc_v2.requests)
    assert third["summary"]["net_pct"].iloc[0] == -100.0

    # Run 4: change decision_model via LEXIMETER_DECISION_MODEL env var (decision_model=None)
    monkeypatch.setenv("LEXIMETER_DECISION_MODEL", "dec-env-v3")
    dc_v3 = CountingDecisionClient(
        handler=lambda state, key, q: {"circle": 0.9, "square": 0.05, "neither": 0.05}
    )
    fourth = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            model="gen-model",
            bucket_count=1,
            client=llm,
            decision_client=dc_v3,
        )
    )
    assert llm.calls == llm_after_v1
    assert dc_v3.calls == 2
    assert all(m == "dec-env-v3" for _, _, m in dc_v3.requests)
    assert fourth["summary"]["net_pct"].iloc[0] == 100.0

    # Run 5: switch decision_client off -> invalidates manifest & falls back to LLMClient
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)
    fifth = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            model="gen-model",
            bucket_count=1,
            client=llm,
            decision_client=None,
        )
    )
    assert llm.calls > llm_after_v1
    assert any("BEGIN LABELS" in p for p in llm.prompts[llm_after_v1:])
    assert fifth["summary"]["net_pct"].iloc[0] == 100.0


def test_discover_resumes_interrupted_decision_run_with_zero_extra_calls(
    tmp_path,
) -> None:
    """Removing discover_state.json + snapshot resumes all stages (incl. DecisionClient) with 0 calls."""
    df = _discover_pair_df()
    save_dir = str(tmp_path / "disc_resume")
    llm = RecordingStageClient(_discover_pair_payloads())
    dc = CountingDecisionClient(
        handler=lambda state, key, q: {"circle": 0.9, "square": 0.05, "neither": 0.05}
    )

    first = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            model="gen-model",
            decision_model="dec-model",
            bucket_count=1,
            client=llm,
            decision_client=dc,
        )
    )
    llm_after = llm.calls
    dc_after = dc.calls
    assert llm_after > 0
    assert dc_after == 2

    (tmp_path / "disc_resume" / "discover_state.json").unlink()
    (tmp_path / "disc_resume" / "discover_results_snapshot.json").unlink()

    resumed = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            model="gen-model",
            decision_model="dec-model",
            bucket_count=1,
            client=llm,
            decision_client=dc,
        )
    )
    assert llm.calls == llm_after
    assert dc.calls == dc_after
    assert resumed["buckets"] == first["buckets"]
    assert list(resumed["summary"]["net_pct"]) == list(first["summary"]["net_pct"])


def test_discover_without_decision_client_falls_back_to_llm_unchanged(
    tmp_path, monkeypatch
) -> None:
    """When decision_client is None and unconfigured in env, discover runs entirely on LLMClient."""
    monkeypatch.delenv("LEXIMETER_DECISION_PROTOCOL", raising=False)
    df = _discover_pair_df()
    llm = RecordingStageClient(_discover_pair_payloads())

    res = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=str(tmp_path / "disc_fallback"),
            model="gen-model",
            bucket_count=1,
            client=llm,
        )
    )
    assert llm.calls > 0
    assert any("BEGIN LABELS" in p for p in llm.prompts)
    assert res["summary"]["net_pct"].iloc[0] == 100.0


# ---------------------------------------------------------------------------
# Paraphrase
# ---------------------------------------------------------------------------


class RecordingSeqClient(LLMClient):
    """LLMClient returning scripted text responses while recording prompts & models."""

    def __init__(self, responses: list[str] | Any) -> None:
        self._responses = responses
        self.calls = 0
        self.prompts: list[str] = []
        self.models: list[str] = []

    async def complete(
        self, prompt: str, *, model: str, json_mode: bool = False, **kwargs: Any
    ) -> str:
        idx = self.calls
        self.calls += 1
        self.prompts.append(prompt)
        self.models.append(model)
        if callable(self._responses):
            return str(self._responses(prompt, idx))
        if idx < len(self._responses):
            return str(self._responses[idx])
        return str(self._responses[-1])

    def estimate_tokens(self, prompt: str, *, model: str) -> int:
        return 0

    def estimate_cost(self, prompt: str, *, model: str) -> float:
        return 0.0


def test_paraphrase_recursive_validation_delegates_to_decision_client(tmp_path) -> None:
    """Paraphrase (n_rounds > 1) generates on LLMClient and validates on DecisionClient."""
    df = pd.DataFrame({"text": ["Original technical sentence."]})
    llm = RecordingSeqClient(["draft_bad", "draft_good_1", "draft_good_2"])

    def validate(state: Any, key: str, question: Any) -> float:
        assert isinstance(question, NoulQuestion)
        assert key == "instructions_followed"
        text = state if isinstance(state, str) else str(state)
        return 0.90 if "draft_good" in text else 0.15

    dc = CountingDecisionClient(handler=validate)

    res = run(
        paraphrase(
            df,
            "text",
            instructions="simplify for a general reader",
            save_dir=str(tmp_path / "para_dec"),
            model="gen-model",
            decision_model="dec-model",
            n_rounds=3,
            n_runs=1,
            later_round_run_multiplier=2,
            client=llm,
            decision_client=dc,
        )
    )

    # Round 0 generated 1 candidate (rejected); Round 1 generated 2 candidates (accepted)
    assert llm.calls == 3
    assert all(m == "gen-model" for m in llm.models)
    assert not any("BEGIN LABELS" in p for p in llm.prompts)

    assert dc.calls == 3
    assert all(m == "dec-model" for _, _, m in dc.requests)

    assert res["text_revised"].iloc[0] == "draft_good_1"
    assert bool(res["text_revised_approved"].iloc[0]) is True


def test_paraphrase_validation_exhausts_rounds_custom_attr_and_refusal(
    tmp_path,
) -> None:
    """Custom validation_attribute forwards to DecisionClient; refusal passes as uncertain."""
    df = pd.DataFrame({"text": ["Original text."]})

    # Exhausting rounds -> approved=False with custom label key
    llm_fail = RecordingSeqClient(["bad_0", "bad_1", "bad_2"])
    dc_fail = CountingDecisionClient({"faithful": 0.10})
    res_fail = run(
        paraphrase(
            df,
            "text",
            instructions="rewrite formally",
            save_dir=str(tmp_path / "para_fail"),
            model="gen-model",
            decision_model="dec-model",
            n_rounds=2,
            n_runs=1,
            later_round_run_multiplier=2,
            validation_attribute={"faithful": "rewrite obeys formal tone"},
            client=llm_fail,
            decision_client=dc_fail,
        )
    )
    assert dc_fail.calls == 3
    assert all(list(qs.keys()) == ["faithful"] for _, qs, _ in dc_fail.requests)
    assert bool(res_fail["text_revised_approved"].iloc[0]) is False

    # Refusal -> treated as uncertain -> passes in round 0
    llm_ref = RecordingSeqClient(["rewritten_once"])
    dc_ref = CountingDecisionClient(
        {
            "instructions_followed": NoulAnswer(
                probability=None, confidence=None, refused=True
            )
        }
    )
    res_ref = run(
        paraphrase(
            df,
            "text",
            instructions="rewrite formally",
            save_dir=str(tmp_path / "para_ref"),
            model="gen-model",
            decision_model="dec-model",
            n_rounds=3,
            n_runs=1,
            client=llm_ref,
            decision_client=dc_ref,
        )
    )
    assert llm_ref.calls == 1
    assert dc_ref.calls == 1
    assert res_ref["text_revised"].iloc[0] == "rewritten_once"
    assert bool(res_ref["text_revised_approved"].iloc[0]) is True


def test_paraphrase_single_round_skips_decision_client(tmp_path) -> None:
    """When n_rounds == 1, paraphrase skips validation and makes 0 calls to DecisionClient."""
    df = pd.DataFrame({"text": ["Hello world"]})
    llm = RecordingSeqClient(["Rewritten hello"])
    dc = CountingDecisionClient({"instructions_followed": 0.95})

    res = run(
        paraphrase(
            df,
            "text",
            instructions="rephrase",
            save_dir=str(tmp_path / "para_r1"),
            model="gen-model",
            decision_model="dec-model",
            n_rounds=1,
            client=llm,
            decision_client=dc,
        )
    )
    assert llm.calls == 1
    assert llm.models == ["gen-model"]
    assert dc.calls == 0
    assert res["text_revised"].iloc[0] == "Rewritten hello"
    assert bool(res["text_revised_approved"].iloc[0]) is True


def test_paraphrase_env_decision_model_and_validation_reset_on_model_change(
    tmp_path, monkeypatch
) -> None:
    """Changing decision_model (incl. via env) invalidates stale validation checkpoint."""
    monkeypatch.delenv("LEXIMETER_DECISION_PROTOCOL", raising=False)
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)
    df = pd.DataFrame({"text": ["Original text"]})
    save_dir = str(tmp_path / "para_env")
    llm = RecordingSeqClient(["cand_0", "cand_1"])

    dc_v1 = CountingDecisionClient({"instructions_followed": 0.10})
    first = run(
        paraphrase(
            df,
            "text",
            instructions="rephrase",
            save_dir=save_dir,
            model="gen-model",
            decision_model="dec-v1",
            n_rounds=2,
            n_runs=1,
            later_round_run_multiplier=1,
            client=llm,
            decision_client=dc_v1,
        )
    )
    assert bool(first["text_revised_approved"].iloc[0]) is False
    assert dc_v1.calls == 2
    assert all(m == "dec-v1" for _, _, m in dc_v1.requests)

    # Re-run with LEXIMETER_DECISION_MODEL="dec-env-v2" -> validation must re-run
    monkeypatch.setenv("LEXIMETER_DECISION_MODEL", "dec-env-v2")
    dc_v2 = CountingDecisionClient({"instructions_followed": 0.95})
    second = run(
        paraphrase(
            df,
            "text",
            instructions="rephrase",
            save_dir=save_dir,
            model="gen-model",
            n_rounds=2,
            n_runs=1,
            later_round_run_multiplier=1,
            client=llm,
            decision_client=dc_v2,
        )
    )
    assert bool(second["text_revised_approved"].iloc[0]) is True
    assert dc_v2.calls == 1
    assert all(m == "dec-env-v2" for _, _, m in dc_v2.requests)


def test_paraphrase_without_decision_client_falls_back_to_llm_unchanged(
    tmp_path, monkeypatch
) -> None:
    """When decision_client=None and env unconfigured, paraphrase validates via LLMClient."""
    monkeypatch.delenv("LEXIMETER_DECISION_PROTOCOL", raising=False)
    df = pd.DataFrame({"text": ["hello"]})
    llm = RecordingSeqClient(
        lambda prompt, idx: (
            json.dumps({"instructions_followed": True})
            if "BEGIN LABELS" in prompt
            else "rewritten_text"
        )
    )

    res = run(
        paraphrase(
            df,
            "text",
            instructions="summarize",
            save_dir=str(tmp_path / "para_fallback"),
            model="gen-model",
            n_rounds=2,
            client=llm,
        )
    )
    assert llm.calls == 2
    assert any("BEGIN LABELS" in p for p in llm.prompts)
    assert res["text_revised"].iloc[0] == "rewritten_text"
    assert bool(res["text_revised_approved"].iloc[0]) is True


# ---------------------------------------------------------------------------
# Codify
# ---------------------------------------------------------------------------


def _codify_two_snippet_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                (
                    "First claim: revenue doubled. "
                    "Second claim: margins rose. "
                    "Unrelated closing remarks."
                )
            ]
        }
    )


def _codify_extract_response(prompt: str, idx: int) -> str:
    if "BEGIN LABELS" in prompt:
        return json.dumps({"finance": False})
    if "First claim" in prompt:
        return json.dumps(
            {
                "finance": [
                    {
                        "beginning excerpt": "First claim",
                        "ending excerpt": "revenue doubled.",
                    }
                ]
            }
        )
    if "Second claim" in prompt:
        return json.dumps(
            {
                "finance": [
                    {
                        "beginning excerpt": "Second claim",
                        "ending excerpt": "margins rose.",
                    }
                ]
            }
        )
    return json.dumps({"finance": []})


def test_codify_completion_check_delegates_to_decision_client_and_recodes_flagged(
    tmp_path,
) -> None:
    """Codify extracts snippets on LLMClient and delegates completion-check classify to DecisionClient."""
    df = _codify_two_snippet_df()
    llm = RecordingSeqClient(_codify_extract_response)

    def check_remaining(state: Any, key: str, question: Any) -> float:
        assert isinstance(question, NoulQuestion)
        assert key == "finance"
        text = state if isinstance(state, str) else str(state.get("text", ""))
        return 0.90 if "Second claim" in text else 0.10

    dc = CountingDecisionClient(handler=check_remaining)

    res = run(
        codify(
            df,
            "text",
            categories={"finance": "claims about financial performance"},
            save_dir=str(tmp_path / "codify_dec"),
            model="gen-model",
            decision_model="dec-model",
            n_rounds=3,
            client=llm,
            decision_client=dc,
        )
    )

    assert llm.calls == 2
    assert all(m == "gen-model" for m in llm.models)
    assert not any("BEGIN LABELS" in p for p in llm.prompts)

    assert dc.calls == 2
    assert all(m == "dec-model" for _, _, m in dc.requests)

    snippets = res["finance"].iloc[0]
    assert any("First claim" in s for s in snippets)
    assert any("Second claim" in s for s in snippets)


def test_codify_completion_check_refusal_treated_as_not_flagged(tmp_path) -> None:
    """A refused DecisionClient completion check safely stops without raising or false-flagging."""
    df = _codify_two_snippet_df()
    llm = RecordingSeqClient(_codify_extract_response)
    dc = CountingDecisionClient(
        {"finance": NoulAnswer(probability=None, confidence=None, refused=True)}
    )

    res = run(
        codify(
            df,
            "text",
            categories={"finance": "claims about financial performance"},
            save_dir=str(tmp_path / "codify_refusal"),
            model="gen-model",
            decision_model="dec-model",
            n_rounds=3,
            client=llm,
            decision_client=dc,
        )
    )

    assert llm.calls == 1
    assert dc.calls == 1
    snippets = res["finance"].iloc[0]
    assert len(snippets) == 1
    assert "First claim" in snippets[0]


def test_codify_single_round_and_dynamic_mode_skip_decision_client(tmp_path) -> None:
    """n_rounds == 1 and dynamic mode never invoke DecisionClient."""
    df = _codify_two_snippet_df()
    dc = CountingDecisionClient({"finance": 0.95})

    # Static n_rounds == 1
    llm_r1 = RecordingSeqClient(_codify_extract_response)
    res_r1 = run(
        codify(
            df,
            "text",
            categories={"finance": "financial claims"},
            save_dir=str(tmp_path / "codify_r1"),
            model="gen-model",
            decision_model="dec-model",
            n_rounds=1,
            client=llm_r1,
            decision_client=dc,
        )
    )
    assert llm_r1.calls == 1
    assert dc.calls == 0
    assert any("First claim" in s for s in res_r1["finance"].iloc[0])

    # Dynamic mode (categories=None)
    llm_dyn = RecordingSeqClient(_codify_extract_response)
    res_dyn = run(
        codify(
            df,
            "text",
            categories=None,
            additional_instructions="extract financial claims",
            save_dir=str(tmp_path / "codify_dyn"),
            model="gen-model",
            decision_model="dec-model",
            n_rounds=2,
            client=llm_dyn,
            decision_client=dc,
        )
    )
    assert llm_dyn.calls == 1
    assert dc.calls == 0
    assert "coded_passages" in res_dyn.columns


def test_codify_completed_manifest_cache_and_decision_model_invalidation(
    tmp_path, monkeypatch
) -> None:
    """Unchanged rerun makes 0 calls to both clients; decision_model change invalidates manifest."""
    monkeypatch.delenv("LEXIMETER_DECISION_PROTOCOL", raising=False)
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)
    df = _codify_two_snippet_df()
    save_dir = str(tmp_path / "codify_sig")
    llm = RecordingSeqClient(_codify_extract_response)

    # Run 1: dec-v1 does not flag -> only snippet 1 extracted
    dc_v1 = CountingDecisionClient({"finance": 0.10})
    first = run(
        codify(
            df,
            "text",
            categories={"finance": "financial claims"},
            save_dir=save_dir,
            model="gen-model",
            decision_model="dec-v1",
            n_rounds=2,
            client=llm,
            decision_client=dc_v1,
        )
    )
    llm_after_v1 = llm.calls
    dc_after_v1 = dc_v1.calls
    assert llm_after_v1 == 1
    assert dc_after_v1 == 1
    assert len(first["finance"].iloc[0]) == 1

    # Run 2: unchanged -> 0 calls to BOTH clients
    second = run(
        codify(
            df,
            "text",
            categories={"finance": "financial claims"},
            save_dir=save_dir,
            model="gen-model",
            decision_model="dec-v1",
            n_rounds=2,
            client=llm,
            decision_client=dc_v1,
        )
    )
    assert llm.calls == llm_after_v1
    assert dc_v1.calls == dc_after_v1
    assert second["finance"].iloc[0] == first["finance"].iloc[0]

    # Run 3: explicit decision_model="dec-v2" flags remaining text -> extracts snippet 2
    dc_v2 = CountingDecisionClient({"finance": 0.90})
    third = run(
        codify(
            df,
            "text",
            categories={"finance": "financial claims"},
            save_dir=save_dir,
            model="gen-model",
            decision_model="dec-v2",
            n_rounds=2,
            client=llm,
            decision_client=dc_v2,
        )
    )
    assert dc_v2.calls == 1
    assert all(m == "dec-v2" for _, _, m in dc_v2.requests)
    assert any("Second claim" in s for s in third["finance"].iloc[0])

    # Run 4: change via LEXIMETER_DECISION_MODEL env var -> invalidates manifest
    monkeypatch.setenv("LEXIMETER_DECISION_MODEL", "dec-env-v3")
    dc_v3 = CountingDecisionClient({"finance": 0.10})
    fourth = run(
        codify(
            df,
            "text",
            categories={"finance": "financial claims"},
            save_dir=save_dir,
            model="gen-model",
            n_rounds=2,
            client=llm,
            decision_client=dc_v3,
        )
    )
    assert dc_v3.calls == 1
    assert all(m == "dec-env-v3" for _, _, m in dc_v3.requests)
    assert len(fourth["finance"].iloc[0]) == 1

    # Run 5: switch decision_client off -> invalidates manifest & falls back to LLMClient
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)
    llm_before_fallback = llm.calls
    run(
        codify(
            df,
            "text",
            categories={"finance": "financial claims"},
            save_dir=save_dir,
            model="gen-model",
            n_rounds=2,
            client=llm,
            decision_client=None,
        )
    )
    assert llm.calls > llm_before_fallback
    assert any("BEGIN LABELS" in p for p in llm.prompts[llm_before_fallback:])


def test_codify_without_decision_client_falls_back_to_llm_unchanged(
    tmp_path, monkeypatch
) -> None:
    """When decision_client=None and env unconfigured, codify completion check uses LLMClient."""
    monkeypatch.delenv("LEXIMETER_DECISION_PROTOCOL", raising=False)
    df = _codify_two_snippet_df()
    llm = RecordingSeqClient(_codify_extract_response)

    res = run(
        codify(
            df,
            "text",
            categories={"finance": "financial claims"},
            save_dir=str(tmp_path / "codify_fallback"),
            model="gen-model",
            n_rounds=2,
            client=llm,
        )
    )
    assert llm.calls == 2
    assert any("BEGIN LABELS" in p for p in llm.prompts)
    assert any("First claim" in s for s in res["finance"].iloc[0])


# ---------------------------------------------------------------------------
# Debias
# ---------------------------------------------------------------------------


def _debias_df() -> pd.DataFrame:
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


class RecordingDebiasClient(DebiasClient):
    """DebiasClient extended with call recording plus paraphrase/classify/rank/extract support."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.prompts: list[str] = []
        self.models: list[str] = []

    async def complete(
        self, prompt: str, *, model: str, json_mode: bool = False, **kwargs: Any
    ) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        self.models.append(model)

        # Paraphrase generation prompt (non-JSON plain text rewrite)
        if not json_mode and "Rewrite the passage" in prompt:
            for brand, snippet in (
                ("Alpha", "Brand Alpha is great. Revenue is up."),
                ("Beta", "Brand Beta shines. Costs are low."),
                ("Gamma", "Brand Gamma excels. Growth is fast."),
                ("Delta", "Brand Delta grows. Margins are high."),
            ):
                if brand in prompt:
                    return snippet
            return "Stripped passage."

        # Classify fallback (brand or codify/paraphrase validation)
        if "BEGIN LABELS" in prompt:
            if "instructions_followed" in prompt:
                return json.dumps({"instructions_followed": True})
            if "mentions of the chief executive" in prompt:
                return json.dumps({"ceo": False})
            if "CEO" in prompt:
                val = "Brand Delta" not in prompt
                return json.dumps({"brand": val})
            val = "Brand Alpha" in prompt or "Brand Beta" in prompt
            return json.dumps({"brand": val})

        # Rank fallback (pairwise Circle vs Square)
        if "Circle" in prompt and "Square" in prompt and "brand" in prompt.lower():
            return json.dumps({"brand": "circle"})

        # Extract fallback
        if "BEGIN ATTRIBUTES TO EXTRACT" in prompt or (
            "brand" in prompt.lower()
            and "prevalence of ceo" not in prompt
            and "BEGIN CATEGORIES/ATTRIBUTES" not in prompt
            and "0-100" not in prompt
            and "rate" not in prompt.lower()
            and "score" not in prompt.lower()
        ):
            for brand, score in self._STRIPPED.items():
                if brand in prompt:
                    return json.dumps({f"Brand {brand}": {"brand": str(score)}})

        return await super().complete(
            prompt, model=model, json_mode=json_mode, **kwargs
        )


def test_debias_classify_mode_with_codify_delegates_classify_and_keeps_llm_stages(
    tmp_path,
) -> None:
    """debias(mode='classify') delegates classify & codify completion checks to DecisionClient."""
    df = _debias_df()
    llm = RecordingDebiasClient()

    def decide_classify(state: Any, key: str, question: Any) -> float:
        assert isinstance(question, NoulQuestion)
        text = state if isinstance(state, str) else str(state.get("text", ""))
        if key == "ceo":
            # Codify completion check -> stop after initial extraction
            return 0.10
        assert key == "brand"
        if "CEO" in text:
            # Original: Alpha/Beta/Gamma True, Delta False
            return 0.90 if "Delta" not in text else 0.10
        # Stripped: Alpha/Beta True, Gamma/Delta False
        return 0.90 if ("Alpha" in text or "Beta" in text) else 0.10

    dc = CountingDecisionClient(handler=decide_classify)

    res = run(
        debias(
            df,
            "text",
            save_dir=str(tmp_path / "db_cls"),
            mode="classify",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            remaining_signal=True,
            model="gen-model",
            decision_model="dec-model",
            measurement_kwargs={
                "min_frequency": 0.6,
                "decision_model": "dec-measure-model",
            },
            client=llm,
            decision_client=dc,
            verbose=False,
        )
    )

    # Generative LLMClient handled codify snippet extraction + remaining_signal rate ONLY
    assert llm.calls > 0
    assert all(m == "gen-model" for m in llm.models)
    assert not any("BEGIN LABELS" in p for p in llm.prompts)
    assert any("BEGIN CATEGORIES/ATTRIBUTES" in p for p in llm.prompts)
    assert any("prevalence of ceo" in p for p in llm.prompts)

    # DecisionClient handled baseline classify (4) + codify completion (4) + stripped classify (4)
    assert dc.calls == 12
    brand_models = [m for _, qs, m in dc.requests if "brand" in qs]
    ceo_models = [m for _, qs, m in dc.requests if "ceo" in qs]
    assert len(brand_models) == 8
    assert all(m == "dec-measure-model" for m in brand_models)
    assert len(ceo_models) == 4
    assert all(m == "dec-model" for m in ceo_models)

    results = res.results
    assert list(results["brand"]) == [True, True, True, False]
    assert list(results["brand (stripped)"]) == [True, True, False, False]
    assert "brand (raw - stripped)" in results.columns
    assert "brand (debiased)" in results.columns
    assert "prevalence of ceo (stripped)" in results.columns
    assert res.regression["stripped_100pct"].stage1_delta is not None


def test_debias_rank_mode_with_paraphrase_delegates_rank_and_validation(
    tmp_path,
) -> None:
    """debias(mode='rank', removal_method='paraphrase') delegates rank & validation to DecisionClient."""
    df = _debias_df()
    llm = RecordingDebiasClient()
    rank_order_orig = {"Alpha": 1, "Beta": 2, "Gamma": 3, "Delta": 4}
    rank_order_stripped = {"Alpha": 4, "Beta": 3, "Gamma": 2, "Delta": 1}

    def _brand_rank(text: str, table: dict[str, int]) -> int:
        for k, v in table.items():
            if k in text:
                return v
        return 0

    def decide_rank_or_val(state: Any, key: str, question: Any) -> Any:
        if key == "instructions_followed":
            assert isinstance(question, NoulQuestion)
            return 0.95
        assert key == "brand"
        assert isinstance(question, ChoiceQuestion)
        assert isinstance(state, dict)
        c_txt, s_txt = state["circle"], state["square"]
        table = rank_order_orig if "CEO" in c_txt else rank_order_stripped
        c_val = _brand_rank(c_txt, table)
        s_val = _brand_rank(s_txt, table)
        if c_val > s_val:
            return "circle"
        if s_val > c_val:
            return "square"
        return "draw"

    dc = CountingDecisionClient(handler=decide_rank_or_val)

    res = run(
        debias(
            df,
            "text",
            save_dir=str(tmp_path / "db_rank"),
            mode="rank",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="paraphrase",
            remaining_signal=True,
            model="gen-model",
            decision_model="dec-model",
            measurement_kwargs={
                "n_rounds": 1,
                "matches_per_round": 2,
                "decision_model": "dec-rank-model",
            },
            removal_kwargs={"n_rounds": 2, "later_round_run_multiplier": 1},
            client=llm,
            decision_client=dc,
            verbose=False,
        )
    )

    # LLMClient handled paraphrase rewriting + remaining_signal rate ONLY
    assert llm.calls == 8  # 4 paraphrases + 4 remaining_signal ratings
    assert all(m == "gen-model" for m in llm.models)
    assert not any("BEGIN LABELS" in p for p in llm.prompts)

    # DecisionClient handled baseline rank + stripped rank + paraphrase validation (4)
    rank_models = [m for _, qs, m in dc.requests if "brand" in qs]
    val_models = [m for _, qs, m in dc.requests if "instructions_followed" in qs]
    assert len(rank_models) > 0
    assert all(m == "dec-rank-model" for m in rank_models)
    assert len(val_models) == 4
    assert all(m == "dec-model" for m in val_models)

    results = res.results
    assert "brand" in results.columns
    assert "brand (stripped)" in results.columns
    assert "brand (raw - stripped)" in results.columns
    assert "brand (debiased)" in results.columns
    assert "prevalence of ceo (stripped)" in results.columns


def test_debias_rate_and_extract_stay_on_llm_and_pop_decision_model(tmp_path) -> None:
    """Rate and extract stages stay on LLMClient and defensively ignore decision_model kwargs."""
    df = _debias_df()
    dc = CountingDecisionClient({"brand": 0.9, "ceo": 0.1})

    # mode="rate" with codify n_rounds=1 -> 0 calls to DecisionClient, no RateConfig TypeError
    llm_rate = RecordingDebiasClient()
    res_rate = run(
        debias(
            df,
            "text",
            save_dir=str(tmp_path / "db_rate"),
            mode="rate",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            removal_kwargs={"n_rounds": 1},
            measurement_kwargs={"decision_model": "ignored-for-rate"},
            model="gen-model",
            decision_model="dec-model",
            client=llm_rate,
            decision_client=dc,
            verbose=False,
        )
    )
    assert llm_rate.calls > 0
    assert all(m == "gen-model" for m in llm_rate.models)
    assert dc.calls == 0
    assert list(res_rate.results["brand"]) == [70.0, 75.0, 80.0, 85.0]

    # mode="extract" with decision_model in measurement_kwargs -> no ExtractConfig TypeError
    llm_ext = RecordingDebiasClient()
    res_ext = run(
        debias(
            df,
            "text",
            save_dir=str(tmp_path / "db_ext"),
            mode="extract",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            remaining_signal=False,
            removal_kwargs={"n_rounds": 1},
            measurement_kwargs={"decision_model": "ignored-for-extract"},
            model="gen-model",
            decision_model="dec-model",
            client=llm_ext,
            decision_client=dc,
            verbose=False,
        )
    )
    assert llm_ext.calls > 0
    assert dc.calls == 0
    assert "brand" in res_ext.results.columns


def test_debias_completed_manifest_cache_and_decision_model_invalidation(
    tmp_path, monkeypatch
) -> None:
    """Unchanged rerun makes 0 calls to both clients; decision_model change invalidates & re-runs."""
    monkeypatch.delenv("LEXIMETER_DECISION_PROTOCOL", raising=False)
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)
    df = _debias_df()
    save_dir = str(tmp_path / "db_sig")
    llm = RecordingDebiasClient()

    def make_dc(invert_delta_alpha: bool) -> CountingDecisionClient:
        def handler(state: Any, key: str, question: Any) -> float:
            text = state if isinstance(state, str) else str(state.get("text", ""))
            if key == "ceo":
                return 0.10
            if "CEO" in text:
                if invert_delta_alpha:
                    return 0.10 if "Alpha" in text else 0.90
                return 0.10 if "Delta" in text else 0.90
            return 0.90 if ("Alpha" in text or "Beta" in text) else 0.10

        return CountingDecisionClient(handler=handler)

    # Run 1: explicit decision_model="dec-v1" (Alpha=True, Delta=False)
    dc_v1 = make_dc(invert_delta_alpha=False)
    first = run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            mode="classify",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            removal_kwargs={"n_rounds": 2},
            model="gen-model",
            decision_model="dec-v1",
            client=llm,
            decision_client=dc_v1,
            verbose=False,
        )
    )
    llm_after_v1 = llm.calls
    dc_after_v1 = dc_v1.calls
    assert llm_after_v1 > 0
    assert dc_after_v1 == 12
    assert list(first.results["brand"]) == [True, True, True, False]

    # Run 2: unchanged -> 0 calls to BOTH clients
    second = run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            mode="classify",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            removal_kwargs={"n_rounds": 2},
            model="gen-model",
            decision_model="dec-v1",
            client=llm,
            decision_client=dc_v1,
            verbose=False,
        )
    )
    assert llm.calls == llm_after_v1
    assert dc_v1.calls == dc_after_v1
    assert list(second.results["brand"]) == [True, True, True, False]

    # Run 3: explicit decision_model="dec-v2" (Alpha=False, Delta=True) -> invalidates & re-runs
    dc_v2 = make_dc(invert_delta_alpha=True)
    third = run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            mode="classify",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            removal_kwargs={"n_rounds": 2},
            model="gen-model",
            decision_model="dec-v2",
            client=llm,
            decision_client=dc_v2,
            verbose=False,
        )
    )
    assert dc_v2.calls == 12
    assert all(m == "dec-v2" for _, _, m in dc_v2.requests)
    assert list(third.results["brand"]) == [False, True, True, True]

    # Run 4: env LEXIMETER_DECISION_MODEL="dec-env-v3" (decision_model=None)
    monkeypatch.setenv("LEXIMETER_DECISION_MODEL", "dec-env-v3")
    dc_v3 = make_dc(invert_delta_alpha=False)
    fourth = run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            mode="classify",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            removal_kwargs={"n_rounds": 2},
            model="gen-model",
            client=llm,
            decision_client=dc_v3,
            verbose=False,
        )
    )
    assert dc_v3.calls == 12
    assert all(m == "dec-env-v3" for _, _, m in dc_v3.requests)
    assert list(fourth.results["brand"]) == [True, True, True, False]

    # Run 5: switch decision_client off -> falls back cleanly to LLMClient
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)
    llm_before_fb = llm.calls
    fifth = run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            mode="classify",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            removal_kwargs={"n_rounds": 2},
            model="gen-model",
            client=llm,
            decision_client=None,
            verbose=False,
        )
    )
    assert llm.calls > llm_before_fb
    assert any("BEGIN LABELS" in p for p in llm.prompts[llm_before_fb:])
    assert list(fifth.results["brand"]) == [True, True, True, False]


def test_debias_rank_decision_model_invalidation(tmp_path, monkeypatch) -> None:
    """Changing decision_model in debias(mode='rank') invalidates manifest and re-runs rank."""
    monkeypatch.delenv("LEXIMETER_DECISION_PROTOCOL", raising=False)
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)
    df = _debias_df()
    save_dir = str(tmp_path / "db_rank_sig")
    llm = RecordingDebiasClient()

    dc_v1 = CountingDecisionClient({"brand": "circle", "ceo": 0.10})
    run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            mode="rank",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            remaining_signal=False,
            measurement_kwargs={"n_rounds": 1, "matches_per_round": 2},
            removal_kwargs={"n_rounds": 1},
            model="gen-model",
            decision_model="dec-rank-v1",
            client=llm,
            decision_client=dc_v1,
            verbose=False,
        )
    )
    llm_after_v1 = llm.calls
    dc_v1_calls = dc_v1.calls
    assert dc_v1_calls > 0
    assert all(m == "dec-rank-v1" for _, _, m in dc_v1.requests)

    # Unchanged rerun -> 0 calls to BOTH clients
    run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            mode="rank",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            remaining_signal=False,
            measurement_kwargs={"n_rounds": 1, "matches_per_round": 2},
            removal_kwargs={"n_rounds": 1},
            model="gen-model",
            decision_model="dec-rank-v1",
            client=llm,
            decision_client=dc_v1,
            verbose=False,
        )
    )
    assert llm.calls == llm_after_v1
    assert dc_v1.calls == dc_v1_calls

    # Changed decision_model -> invalidates manifest and re-runs rank on dec-rank-v2
    dc_v2 = CountingDecisionClient({"brand": "square", "ceo": 0.10})
    run(
        debias(
            df,
            "text",
            save_dir=save_dir,
            mode="rank",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            remaining_signal=False,
            measurement_kwargs={"n_rounds": 1, "matches_per_round": 2},
            removal_kwargs={"n_rounds": 1},
            model="gen-model",
            decision_model="dec-rank-v2",
            client=llm,
            decision_client=dc_v2,
            verbose=False,
        )
    )
    assert dc_v2.calls > 0
    assert all(m == "dec-rank-v2" for _, _, m in dc_v2.requests)


def test_debias_without_decision_client_falls_back_to_llm_unchanged(
    tmp_path, monkeypatch
) -> None:
    """When decision_client=None and env unconfigured, debias classify & rank run on LLMClient."""
    monkeypatch.delenv("LEXIMETER_DECISION_PROTOCOL", raising=False)
    df = _debias_df()

    llm_cls = RecordingDebiasClient()
    res_cls = run(
        debias(
            df,
            "text",
            save_dir=str(tmp_path / "db_fb_cls"),
            mode="classify",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            remaining_signal=True,
            model="gen-model",
            client=llm_cls,
            verbose=False,
        )
    )
    assert llm_cls.calls > 0
    assert any("BEGIN LABELS" in p for p in llm_cls.prompts)
    assert list(res_cls.results["brand"]) == [True, True, True, False]

    llm_rank = RecordingDebiasClient()
    res_rank = run(
        debias(
            df,
            "text",
            save_dir=str(tmp_path / "db_fb_rank"),
            mode="rank",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            remaining_signal=False,
            measurement_kwargs={"n_rounds": 1, "matches_per_round": 2},
            removal_kwargs={"n_rounds": 1},
            model="gen-model",
            client=llm_rank,
            verbose=False,
        )
    )
    assert llm_rank.calls > 0
    assert "brand" in res_rank.results.columns
    assert "brand (stripped)" in res_rank.results.columns


def test_discover_pair_square_primary_label_yields_positive_net_pct(tmp_path) -> None:
    """Square-primary bucket label in paired discover yields actual=True and net_pct > 0."""
    label = "square entries emphasize cost more than circle entries"
    df = pd.DataFrame(
        {
            "circle": ["focus on design quality", "focus on craftsmanship"],
            "square": ["focus on low price and cost", "focus on budget savings"],
        }
    )
    llm = RecordingStageClient(
        {
            "compare": {label: "Square emphasizes cost more than Circle"},
            "bucket": {label: "Square emphasizes cost more than Circle"},
            "classify": {},
            "codify": {},
        }
    )

    def decide_cost(state: Any, key: str, question: Any) -> dict[str, float]:
        assert isinstance(question, ChoiceQuestion)
        assert question.criteria["square"] == "Square emphasizes cost more than Circle"
        assert question.criteria["circle"] == "circle emphasizes cost more than square"
        # Square genuinely emphasizes cost more than Circle in both rows
        return {"circle": 0.05, "square": 0.90, "neither": 0.05}

    dc = CountingDecisionClient(handler=decide_cost)
    res = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=str(tmp_path / "disc_sq_primary"),
            model="gen-model",
            decision_model="dec-model",
            bucket_count=1,
            client=llm,
            decision_client=dc,
        )
    )

    clf_df = res["classification"]
    assert list(clf_df[f"{label}_actual"]) == [True, True]
    assert list(clf_df[f"{label}_inverted"]) == [False, False]
    summary = res["summary"]
    assert summary["label"].iloc[0] == label
    assert summary["actual_pct"].iloc[0] == 100.0
    assert summary["inverted_pct"].iloc[0] == 0.0
    assert summary["net_pct"].iloc[0] == 100.0


def test_discover_mid_stage_classify_crash_resumes_with_zero_duplicate_calls(
    tmp_path, monkeypatch
) -> None:
    """Mid-stage crash in discover Stage 3 classify preserves checkpointed items on resume."""
    import leximeter.engine.batch as batch_mod

    monkeypatch.setattr(batch_mod, "_SAVE_EVERY", 1)

    class _SimulatedCrash(BaseException):
        pass

    class CrashOnSecondRowClient(DecisionClient):
        def __init__(self) -> None:
            self.seen_circles: list[str] = []
            self.should_crash = True

        async def decide(self, state, questions, *, model, **kwargs):
            circle_text = state["circle"]
            self.seen_circles.append(circle_text)
            if self.should_crash and circle_text == "rainy":
                raise _SimulatedCrash("simulated mid-stage process crash")
            qkey = next(iter(questions))
            return DecisionResult(
                answers={
                    qkey: ChoiceAnswer(
                        choice="circle",
                        probabilities={"circle": 0.9, "square": 0.05, "neither": 0.05},
                        confidence=0.9,
                    )
                },
                model=model,
            )

    df = _discover_pair_df()
    save_dir = str(tmp_path / "disc_mid_crash")
    llm = RecordingStageClient(_discover_pair_payloads())
    dc = CrashOnSecondRowClient()

    with pytest.raises(_SimulatedCrash):
        run(
            discover(
                df,
                circle_column_name="circle",
                square_column_name="square",
                save_dir=save_dir,
                model="gen-model",
                decision_model="dec-model",
                bucket_count=1,
                n_parallels=1,
                client=llm,
                decision_client=dc,
            )
        )

    assert dc.seen_circles == ["sunny", "rainy"]
    meta_path = (
        tmp_path
        / "disc_mid_crash"
        / "classify"
        / "classify_responses_run_metadata.json"
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["decision_provider"] is True
    assert meta["model"] == "dec-model"

    llm_after_crash = llm.calls
    dc.should_crash = False
    dc.seen_circles.clear()

    resumed = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=save_dir,
            model="gen-model",
            decision_model="dec-model",
            bucket_count=1,
            n_parallels=1,
            reset_files=False,
            client=llm,
            decision_client=dc,
        )
    )

    assert llm.calls == llm_after_crash  # Stages 1 & 2 reused from checkpoints
    assert dc.seen_circles == ["rainy"]  # "sunny" reused with ZERO duplicate calls
    assert resumed["summary"]["net_pct"].iloc[0] == 100.0


def test_debias_mid_stage_rank_crash_resumes_with_zero_duplicate_calls(
    tmp_path,
) -> None:
    """Mid-stage crash in debias(mode='rank') preserves Round 0 checkpoint on resume."""

    class _SimulatedCrash(BaseException):
        pass

    save_dir = tmp_path / "db_rank_mid_crash"
    round0_csv = save_dir / "debias_brand" / "original" / "rankings_round0.csv"
    round1_csv = save_dir / "debias_brand" / "original" / "rankings_round1.csv"

    class CrashOnRound1Client(DecisionClient):
        def __init__(self) -> None:
            self.should_crash = True
            self.orig_calls = 0

        async def decide(self, state, questions, *, model, **kwargs):
            if "CEO" in state["circle"]:
                if self.should_crash and round0_csv.exists():
                    raise _SimulatedCrash("simulated crash at start of Round 1")
                self.orig_calls += 1
            qkey = next(iter(questions))
            return DecisionResult(
                answers={
                    qkey: ChoiceAnswer(
                        choice="circle",
                        probabilities={
                            "circle": 0.9,
                            "square": 0.05,
                            "draw": 0.025,
                            "insufficient_signal": 0.025,
                        },
                        confidence=0.9,
                    )
                },
                model=model,
            )

    df = _debias_df()
    llm = RecordingDebiasClient()
    dc = CrashOnRound1Client()

    with pytest.raises(_SimulatedCrash):
        run(
            debias(
                df,
                "text",
                save_dir=str(save_dir),
                mode="rank",
                attributes={"brand": "overall brand strength"},
                signal_dictionary={"ceo": "mentions of the chief executive"},
                removal_method="codify",
                remaining_signal=False,
                measurement_kwargs={"n_rounds": 2, "matches_per_round": 2},
                removal_kwargs={"n_rounds": 1},
                model="gen-model",
                decision_model="dec-rank-model",
                client=llm,
                decision_client=dc,
                verbose=False,
            )
        )

    assert round0_csv.exists()
    meta = json.loads(
        (
            save_dir / "debias_brand" / "original" / "rankings_run_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert meta["decision_provider"] is True
    assert meta["model"] == "dec-rank-model"

    dc.should_crash = False
    dc.orig_calls = 0

    res = run(
        debias(
            df,
            "text",
            save_dir=str(save_dir),
            mode="rank",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            remaining_signal=False,
            measurement_kwargs={"n_rounds": 2, "matches_per_round": 2},
            removal_kwargs={"n_rounds": 1},
            model="gen-model",
            decision_model="dec-rank-model",
            reset_files=False,
            client=llm,
            decision_client=dc,
            verbose=False,
        )
    )

    assert round1_csv.exists()
    r1_rows = len(pd.read_csv(round1_csv))
    # Only Round 1 executed for original rank; Round 0 replayed with ZERO duplicate calls!
    assert dc.orig_calls == r1_rows
    assert "brand" in res.results.columns
    assert "brand (stripped)" in res.results.columns


def test_composite_tasks_systemone_missing_decision_model_fails_fast(
    tmp_path,
) -> None:
    """discover, paraphrase(n_rounds>1), codify(n_rounds>1), and decision-active debias fail fast before LLM/HTTP/disk resets."""
    import httpx

    from leximeter.client import SystemOneClient

    http_calls = 0

    def fail_transport(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        raise AssertionError(f"Unexpected HTTP call to {request.url}")

    class FailIfCalledLLM(RecordingStageClient):
        def __init__(self) -> None:
            super().__init__({})

        async def complete(
            self, prompt: str, *, model: str, json_mode: bool = False, **kwargs: Any
        ) -> str:
            self.calls += 1
            raise AssertionError("LLMClient.complete must not be called")

    sys1 = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=httpx.MockTransport(fail_transport),
    )
    llm = FailIfCalledLLM()

    # 1. discover
    disc_dir = tmp_path / "disc_fail"
    disc_dir.mkdir()
    disc_state = disc_dir / "discover_state.json"
    disc_state.write_text('{"status": "complete"}', encoding="utf-8")
    with pytest.raises(ValueError, match="decision_model"):
        run(
            discover(
                _discover_single_df(),
                column_name="text",
                save_dir=str(disc_dir),
                client=llm,
                decision_client=sys1,
                reset_files=True,
            )
        )
    assert disc_state.read_text(encoding="utf-8") == '{"status": "complete"}'

    # 2. paraphrase (n_rounds > 1)
    with pytest.raises(ValueError, match="decision_model"):
        run(
            paraphrase(
                pd.DataFrame({"text": ["sample text"]}),
                "text",
                instructions="Rewrite concisely.",
                save_dir=str(tmp_path / "para_fail"),
                n_rounds=2,
                client=llm,
                decision_client=sys1,
                reset_files=True,
            )
        )

    # 3. codify (static categories + n_rounds > 1)
    cod_dir = tmp_path / "cod_fail"
    cod_dir.mkdir()
    cod_state = cod_dir / "codify_state.json"
    cod_state.write_text('{"status": "complete"}', encoding="utf-8")
    with pytest.raises(ValueError, match="decision_model"):
        run(
            codify(
                pd.DataFrame({"text": ["sample text"]}),
                "text",
                categories={"cat": "desc"},
                save_dir=str(cod_dir),
                n_rounds=2,
                client=llm,
                decision_client=sys1,
                reset_files=True,
            )
        )
    assert cod_state.read_text(encoding="utf-8") == '{"status": "complete"}'

    # 4. debias (mode="classify" and mode="rate" with paraphrase n_rounds=2)
    db_cls_dir = tmp_path / "db_cls_fail" / "debias_brand"
    db_cls_dir.mkdir(parents=True)
    db_cls_state = db_cls_dir / "debias_state.json"
    db_cls_state.write_text('{"status": "complete"}', encoding="utf-8")
    with pytest.raises(ValueError, match="decision_model"):
        run(
            debias(
                _debias_df(),
                "text",
                save_dir=str(tmp_path / "db_cls_fail"),
                mode="classify",
                attributes={"brand": "overall brand strength"},
                signal_dictionary={"ceo": "mentions of the chief executive"},
                removal_method="codify",
                removal_kwargs={"n_rounds": 1},
                client=llm,
                decision_client=sys1,
                reset_files=True,
                verbose=False,
            )
        )
    assert db_cls_state.read_text(encoding="utf-8") == '{"status": "complete"}'

    with pytest.raises(ValueError, match="decision_model"):
        run(
            debias(
                _debias_df(),
                "text",
                save_dir=str(tmp_path / "db_para_fail"),
                mode="rate",
                attributes={"brand": "overall brand strength"},
                signal_dictionary={"ceo": "mentions of the chief executive"},
                removal_method="paraphrase",
                removal_kwargs={"n_rounds": 2},
                client=llm,
                decision_client=sys1,
                reset_files=True,
                verbose=False,
            )
        )

    assert llm.calls == 0
    assert http_calls == 0


def test_composite_and_debias_non_decision_paths_ignore_unset_systemone_model(
    tmp_path, monkeypatch
) -> None:
    """When no DecisionClient stage is active or stages override decision_model/None, SystemOne env without decision_model does not fail."""
    import httpx

    from leximeter.client import SystemOneClient

    http_calls = 0

    def fail_if_called_transport(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        raise AssertionError(f"Unexpected HTTP call to {request.url}")

    monkeypatch.setenv("LEXIMETER_DECISION_PROTOCOL", "systemone")
    monkeypatch.setenv("LEXIMETER_DECISION_BASE_URL", "https://api.typesafe.ai")
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)

    sys1_fail = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=httpx.MockTransport(fail_if_called_transport),
    )

    # 1. paraphrase n_rounds=1 -> LLM-only
    para_llm = RecordingSeqClient(["concise rewrite"])
    para_res = run(
        paraphrase(
            pd.DataFrame({"text": ["hello world"]}),
            "text",
            instructions="Rewrite concisely.",
            save_dir=str(tmp_path / "para_1round"),
            n_runs=1,
            n_rounds=1,
            client=para_llm,
            decision_client=sys1_fail,
        )
    )
    assert len(para_res) == 1
    assert para_llm.calls == 1
    assert para_res["text_revised"].iloc[0] == "concise rewrite"

    # 2. codify n_rounds=1 and dynamic codify n_rounds=2 -> LLM-only
    cod_llm = RecordingSeqClient(lambda prompt, idx: json.dumps({"cat": []}))
    run(
        codify(
            pd.DataFrame({"text": ["hello world"]}),
            "text",
            categories={"cat": "desc"},
            save_dir=str(tmp_path / "cod_1round"),
            n_rounds=1,
            client=cod_llm,
            decision_client=sys1_fail,
        )
    )
    run(
        codify(
            pd.DataFrame({"text": ["hello world"]}),
            "text",
            categories=None,
            additional_instructions="Extract key topics.",
            save_dir=str(tmp_path / "cod_dyn_2round"),
            n_rounds=2,
            client=cod_llm,
            decision_client=sys1_fail,
        )
    )
    assert cod_llm.calls == 2

    # 3. debias(mode="rate", codify n_rounds=1) -> succeeds and ignores LEXIMETER_DECISION_MODEL changes in cache
    db_df = _debias_df()
    db_rate_llm = RecordingDebiasClient()
    db_rate_dir = str(tmp_path / "db_rate_1round")
    run(
        debias(
            db_df,
            "text",
            save_dir=db_rate_dir,
            mode="rate",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            removal_kwargs={"n_rounds": 1},
            remaining_signal=False,
            client=db_rate_llm,
            verbose=False,
        )
    )
    calls_after_rate = db_rate_llm.calls
    assert calls_after_rate > 0
    monkeypatch.setenv("LEXIMETER_DECISION_MODEL", "unrelated-dec-model")
    run(
        debias(
            db_df,
            "text",
            save_dir=db_rate_dir,
            mode="rate",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            removal_kwargs={"n_rounds": 1},
            remaining_signal=False,
            client=db_rate_llm,
            verbose=False,
        )
    )
    assert db_rate_llm.calls == calls_after_rate  # cache hit!
    monkeypatch.delenv("LEXIMETER_DECISION_MODEL", raising=False)

    # 4. debias(mode="extract", paraphrase n_rounds=1) -> succeeds
    db_ext_llm = RecordingDebiasClient()
    run(
        debias(
            db_df,
            "text",
            save_dir=str(tmp_path / "db_ext_1round"),
            mode="extract",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="paraphrase",
            removal_kwargs={"n_rounds": 1},
            remaining_signal=False,
            client=db_ext_llm,
            verbose=False,
        )
    )
    assert db_ext_llm.calls > 0

    # 5. debias(mode="classify") with per-stage decision_client=None overrides -> LLM-only
    db_none_llm = RecordingDebiasClient()
    run(
        debias(
            db_df,
            "text",
            save_dir=str(tmp_path / "db_stage_none"),
            mode="classify",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            measurement_kwargs={"decision_client": None},
            removal_kwargs={"n_rounds": 2, "decision_client": None},
            remaining_signal=False,
            client=db_none_llm,
            verbose=False,
        )
    )
    assert any("BEGIN LABELS" in p for p in db_none_llm.prompts)
    assert http_calls == 0

    # 6. debias(mode="classify") with measurement_kwargs={"decision_model": "dec-m"} & removal n_rounds=1
    def ok_sys1_transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "dec-m"
        return httpx.Response(
            200,
            json={
                "model": "dec-m",
                "answers": {"q_0": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        )

    sys1_ok = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=httpx.MockTransport(ok_sys1_transport),
    )
    db_meas_llm = RecordingDebiasClient()
    run(
        debias(
            db_df,
            "text",
            save_dir=str(tmp_path / "db_meas_override"),
            mode="classify",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            measurement_kwargs={"decision_model": "dec-m"},
            removal_kwargs={"n_rounds": 1},
            remaining_signal=False,
            client=db_meas_llm,
            decision_client=sys1_ok,
            verbose=False,
        )
    )

    # 7. debias(mode="rate", removal n_rounds=2 with decision_model) + cache invalidation on removal decision_model change
    dc_rem = CountingDecisionClient({"ceo": 0.1})
    db_rem_llm = RecordingDebiasClient()
    db_rem_dir = str(tmp_path / "db_rem_override")
    run(
        debias(
            db_df,
            "text",
            save_dir=db_rem_dir,
            mode="rate",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            removal_kwargs={"n_rounds": 2, "decision_model": "dec-r1"},
            remaining_signal=False,
            client=db_rem_llm,
            decision_client=dc_rem,
            verbose=False,
        )
    )
    assert dc_rem.calls == 4
    assert all(m == "dec-r1" for _, _, m in dc_rem.requests)

    dc_rem2 = CountingDecisionClient({"ceo": 0.1})
    run(
        debias(
            db_df,
            "text",
            save_dir=db_rem_dir,
            mode="rate",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            removal_kwargs={"n_rounds": 2, "decision_model": "dec-r2"},
            remaining_signal=False,
            client=db_rem_llm,
            decision_client=dc_rem2,
            verbose=False,
        )
    )
    assert dc_rem2.calls == 4
    assert all(m == "dec-r2" for _, _, m in dc_rem2.requests)


def test_debias_measurement_kwargs_systemone_client_and_deterministic_cache(
    tmp_path,
) -> None:
    """SystemOneClient in measurement_kwargs works (fresh, reused, explicit http_client) and caches deterministically."""
    import httpx

    from leximeter.client import SystemOneClient

    def make_transport(counter: list[int], expected_model: str) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            counter[0] += 1
            body = json.loads(request.content.decode("utf-8"))
            assert body["model"] == expected_model
            qkey = next(iter(body["questions"]))
            score = 0.9 if "Delta" not in str(body.get("state", "")) else 0.1
            return httpx.Response(
                200,
                json={
                    "model": expected_model,
                    "answers": {qkey: {"type": "noul", "noul": score}},
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            )

        return httpx.MockTransport(handler)

    df = _debias_df()

    # 1. Fresh SystemOneClient in measurement_kwargs["decision_client"]
    fresh_dir = tmp_path / "db_meas_fresh"
    fresh_calls = [0]
    s1_fresh = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=make_transport(fresh_calls, "dec-meas"),
    )
    llm = RecordingDebiasClient()

    async def _run_fresh() -> Any:
        try:
            return await debias(
                df,
                "text",
                save_dir=str(fresh_dir),
                mode="classify",
                attributes={"brand": "overall brand strength"},
                signal_dictionary={"ceo": "mentions of the chief executive"},
                removal_method="codify",
                measurement_kwargs={
                    "decision_client": s1_fresh,
                    "decision_model": "dec-meas",
                },
                removal_kwargs={"n_rounds": 1},
                remaining_signal=False,
                client=llm,
                verbose=False,
            )
        finally:
            await s1_fresh.aclose()

    res_fresh = run(_run_fresh())
    assert fresh_calls[0] == 8  # 4 baseline + 4 stripped
    llm_calls_after_fresh = llm.calls
    assert llm_calls_after_fresh > 0
    assert list(res_fresh.results["brand"]) == [True, True, True, False]

    state_path = fresh_dir / "debias_brand" / "debias_state.json"
    meta_path = fresh_dir / "debias_brand" / "debias_metadata.json"
    state_1 = json.loads(state_path.read_text(encoding="utf-8"))
    meta_1 = json.loads(meta_path.read_text(encoding="utf-8"))
    assert state_1["status"] == "complete"
    assert (
        meta_1["config"]["measurement_kwargs"]["decision_client"] == "SystemOneClient"
    )
    assert json.dumps(res_fresh.to_dict())

    # 2. Second fresh SystemOneClient instance on same save_dir -> deterministic cache hit (0 calls)
    second_calls = [0]
    s1_second = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=make_transport(second_calls, "dec-meas"),
    )
    res_cached = run(
        debias(
            df,
            "text",
            save_dir=str(fresh_dir),
            mode="classify",
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            removal_method="codify",
            measurement_kwargs={
                "decision_client": s1_second,
                "decision_model": "dec-meas",
            },
            removal_kwargs={"n_rounds": 1},
            remaining_signal=False,
            client=llm,
            verbose=False,
        )
    )
    assert second_calls[0] == 0
    assert llm.calls == llm_calls_after_fresh
    state_2 = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_2["input_signature"] == state_1["input_signature"]
    assert list(res_cached.results["brand"]) == [True, True, True, False]

    # 3. Already-used SystemOneClient (opened _http_client) in measurement_kwargs
    reused_dir = tmp_path / "db_meas_reused"
    reused_calls = [0]
    s1_reused = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=make_transport(reused_calls, "dec-meas"),
    )
    llm_reused = RecordingDebiasClient()

    async def _run_reused() -> Any:
        try:
            await s1_reused.decide(
                "warmup",
                {"brand": NoulQuestion(instructions="brand strength")},
                model="dec-meas",
            )
            assert s1_reused._http_client is not None
            return await debias(
                df,
                "text",
                save_dir=str(reused_dir),
                mode="classify",
                attributes={"brand": "overall brand strength"},
                signal_dictionary={"ceo": "mentions of the chief executive"},
                removal_method="codify",
                measurement_kwargs={
                    "decision_client": s1_reused,
                    "decision_model": "dec-meas",
                },
                removal_kwargs={"n_rounds": 1},
                remaining_signal=False,
                client=llm_reused,
                verbose=False,
            )
        finally:
            await s1_reused.aclose()

    res_reused = run(_run_reused())
    assert reused_calls[0] == 9  # 1 warmup + 8 debias
    assert list(res_reused.results["brand"]) == [True, True, True, False]
    assert (reused_dir / "debias_brand" / "debias_state.json").exists()

    # 4. Explicit http_client=httpx.AsyncClient() SystemOneClient in measurement_kwargs
    ext_dir = tmp_path / "db_meas_ext_http"
    ext_calls = [0]
    llm_ext = RecordingDebiasClient()

    async def _run_external_http() -> Any:
        ext_http = httpx.AsyncClient(transport=make_transport(ext_calls, "dec-meas"))
        s1_ext = SystemOneClient(
            base_url="https://api.typesafe.ai",
            api_key="test-key",
            http_client=ext_http,
        )
        try:
            return await debias(
                df,
                "text",
                save_dir=str(ext_dir),
                mode="classify",
                attributes={"brand": "overall brand strength"},
                signal_dictionary={"ceo": "mentions of the chief executive"},
                removal_method="codify",
                measurement_kwargs={
                    "decision_client": s1_ext,
                    "decision_model": "dec-meas",
                },
                removal_kwargs={"n_rounds": 1},
                remaining_signal=False,
                client=llm_ext,
                verbose=False,
            )
        finally:
            await ext_http.aclose()

    res_ext = run(_run_external_http())
    assert ext_calls[0] == 8
    assert list(res_ext.results["brand"]) == [True, True, True, False]
    assert (ext_dir / "debias_brand" / "debias_state.json").exists()


def test_debias_removal_kwargs_systemone_client_no_deepcopy_crash(tmp_path) -> None:
    """SystemOneClient in removal_kwargs works for fresh, already-used, and explicit http_client instances."""
    import httpx

    from leximeter.client import SystemOneClient

    def make_transport(counter: list[int], expected_model: str) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            counter[0] += 1
            body = json.loads(request.content.decode("utf-8"))
            assert body["model"] == expected_model
            qkey = next(iter(body["questions"]))
            return httpx.Response(
                200,
                json={
                    "model": expected_model,
                    "answers": {qkey: {"type": "noul", "noul": 0.1}},
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            )

        return httpx.MockTransport(handler)

    df = _debias_df()

    # 1. Fresh SystemOneClient in removal_kwargs
    fresh_dir = tmp_path / "db_rem_fresh"
    fresh_calls = [0]
    s1_fresh = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=make_transport(fresh_calls, "dec-rem"),
    )
    llm_fresh = RecordingDebiasClient()

    async def _run_fresh() -> Any:
        try:
            return await debias(
                df,
                "text",
                save_dir=str(fresh_dir),
                mode="rate",
                attributes={"brand": "overall brand strength"},
                signal_dictionary={"ceo": "mentions of the chief executive"},
                removal_method="codify",
                removal_kwargs={
                    "decision_client": s1_fresh,
                    "decision_model": "dec-rem",
                    "n_rounds": 2,
                },
                remaining_signal=False,
                client=llm_fresh,
                verbose=False,
            )
        finally:
            await s1_fresh.aclose()

    res_fresh = run(_run_fresh())
    assert fresh_calls[0] == 4
    meta_fresh = json.loads(
        (fresh_dir / "debias_brand" / "debias_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        meta_fresh["config"]["removal_kwargs"]["decision_client"] == "SystemOneClient"
    )
    assert json.dumps(res_fresh.to_dict())
    assert (fresh_dir / "debias_brand" / "debias_state.json").exists()

    # 2. Already-used SystemOneClient (opened _http_client) in removal_kwargs
    reused_dir = tmp_path / "db_rem_reused"
    reused_calls = [0]
    s1_reused = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=make_transport(reused_calls, "dec-rem"),
    )
    llm_reused = RecordingDebiasClient()

    async def _run_reused() -> Any:
        try:
            await s1_reused.decide(
                "warmup",
                {"ceo": NoulQuestion(instructions="ceo mention")},
                model="dec-rem",
            )
            assert s1_reused._http_client is not None
            return await debias(
                df,
                "text",
                save_dir=str(reused_dir),
                mode="rate",
                attributes={"brand": "overall brand strength"},
                signal_dictionary={"ceo": "mentions of the chief executive"},
                removal_method="codify",
                removal_kwargs={
                    "decision_client": s1_reused,
                    "decision_model": "dec-rem",
                    "n_rounds": 2,
                },
                remaining_signal=False,
                client=llm_reused,
                verbose=False,
            )
        finally:
            await s1_reused.aclose()

    run(_run_reused())
    assert reused_calls[0] == 5  # 1 warmup + 4 codify completion
    assert (reused_dir / "debias_brand" / "debias_state.json").exists()

    # 3. Explicit http_client=httpx.AsyncClient() SystemOneClient in removal_kwargs
    ext_dir = tmp_path / "db_rem_ext"
    ext_calls = [0]
    llm_ext = RecordingDebiasClient()

    async def _run_ext() -> Any:
        ext_http = httpx.AsyncClient(transport=make_transport(ext_calls, "dec-rem"))
        s1_ext = SystemOneClient(
            base_url="https://api.typesafe.ai",
            api_key="test-key",
            http_client=ext_http,
        )
        try:
            return await debias(
                df,
                "text",
                save_dir=str(ext_dir),
                mode="rate",
                attributes={"brand": "overall brand strength"},
                signal_dictionary={"ceo": "mentions of the chief executive"},
                removal_method="codify",
                removal_kwargs={
                    "decision_client": s1_ext,
                    "decision_model": "dec-rem",
                    "n_rounds": 2,
                },
                remaining_signal=False,
                client=llm_ext,
                verbose=False,
            )
        finally:
            await ext_http.aclose()

    run(_run_ext())
    assert ext_calls[0] == 4
    assert (ext_dir / "debias_brand" / "debias_state.json").exists()
