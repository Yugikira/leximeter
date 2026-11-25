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
from conftest import CountingDecisionClient
from test_tasks import DebiasClient, StageClient

from leximeter import codify, debias, discover, paraphrase
from leximeter.client import ChoiceQuestion, LLMClient, NoulAnswer, NoulQuestion


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
