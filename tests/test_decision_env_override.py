"""Regression tests for Issue #13: decision env override & test isolation.

Verifies that:
1. ``tests/conftest.py`` isolates ``LEXIMETER_DECISION_*`` env vars per test.
2. Pure-decision tasks (``Classify``, ``Filter``, ``Rank``) respect an explicit
   ``client=<LLMClient>`` and ignore ``LEXIMETER_DECISION_PROTOCOL`` when
   ``decision_client`` is omitted, while still resolving env ``DecisionClient``
   when ``client`` is omitted.
3. Explicit ``decision_client`` beats both explicit ``client`` and env.
4. Hybrid/composite tasks (``Discover``, ``Paraphrase``, ``Codify``, ``Debias``)
   preserve env ``DecisionClient`` resolution when ``client`` is injected and
   ``decision_client`` is omitted.
5. Explicit ``decision_client=None`` opts out of env resolution across top-level
   tasks, child tasks, and ``Debias`` ``measurement_kwargs`` / ``removal_kwargs``.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import pandas as pd
import pytest
from conftest import CountingClient, CountingDecisionClient

from leximeter import (
    classify,
    codify,
    debias,
    discover,
    paraphrase,
    rank,
)
from leximeter import (
    filter as leximeter_filter,
)
from leximeter.client.base import LLMClient
from leximeter.client.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionQuestion,
    DecisionResult,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from leximeter.client.systemone import SystemOneClient
from leximeter.tasks.classify import Classify, ClassifyConfig
from leximeter.tasks.codify import Codify, CodifyConfig
from leximeter.tasks.debias import Debias, DebiasConfig
from leximeter.tasks.discover import Discover, DiscoverConfig
from leximeter.tasks.filter import Filter, FilterConfig
from leximeter.tasks.paraphrase import Paraphrase, ParaphraseConfig
from leximeter.tasks.rank import Rank, RankConfig


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def _set_systemone_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEXIMETER_DECISION_PROTOCOL", "systemone")
    monkeypatch.setenv("LEXIMETER_DECISION_BASE_URL", "https://decision.invalid/api")
    monkeypatch.setenv("LEXIMETER_DECISION_API_KEY", "sk-dec-test")
    monkeypatch.setenv("LEXIMETER_DECISION_MODEL", "env-dec-model")


def _install_failing_systemone(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail_decide(
        self: SystemOneClient,
        state: str | dict[str, Any],
        questions: dict[str, DecisionQuestion],
        *,
        model: str = "dummy",
        **kwargs: Any,
    ) -> DecisionResult:
        del self, state, questions, model, kwargs
        pytest.fail("SystemOneClient.decide must not be called")

    monkeypatch.setattr(SystemOneClient, "decide", _fail_decide)


def _install_spy_systemone(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str | None, str, dict[str, DecisionQuestion]]]:
    calls: list[tuple[str, str | None, str, dict[str, DecisionQuestion]]] = []

    async def _spy_decide(
        self: SystemOneClient,
        state: str | dict[str, Any],
        questions: dict[str, DecisionQuestion],
        *,
        model: str = "dummy",
        **kwargs: Any,
    ) -> DecisionResult:
        del state, kwargs
        calls.append((self.url, self._api_key, model, dict(questions)))
        answers: dict[str, DecisionAnswer] = {}
        for key, q in questions.items():
            if isinstance(q, NoulQuestion):
                answers[key] = NoulAnswer(
                    probability=0.9, confidence=0.8, refused=False
                )
            elif isinstance(q, ChoiceQuestion):
                opts = list(q.criteria.keys())
                chosen = opts[0]
                probs = {
                    o: (0.9 if o == chosen else 0.1 / max(1, len(opts) - 1))
                    for o in opts
                }
                answers[key] = ChoiceAnswer(
                    choice=chosen, probabilities=probs, confidence=0.9, refused=False
                )
            elif isinstance(q, ScoreQuestion):
                answers[key] = ScoreAnswer(
                    score=1.0, probabilities={1: 1.0}, confidence=1.0, refused=False
                )
        return DecisionResult(answers=answers, model=model)

    monkeypatch.setattr(SystemOneClient, "decide", _spy_decide)
    return calls


class _RecordingHybridClient(LLMClient):
    """Offline LLMClient handling all pure, composite, and debias prompts."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.models: list[str] = []

    async def complete(
        self, prompt: str, *, model: str, json_mode: bool = False, **kwargs: Any
    ) -> str:
        del kwargs
        self.calls += 1
        self.prompts.append(prompt)
        self.models.append(model)

        # 1. Classify (single or differentiate) / Paraphrase validation / Codify completion check
        if "BEGIN LABELS" in prompt:
            if "instructions_followed" in prompt:
                return json.dumps({"instructions_followed": True})
            if "mentions of the chief executive" in prompt:
                return json.dumps({"ceo": False})
            if "finance" in prompt:
                return json.dumps({"finance": False})
            if "quality (inverted)" in prompt:
                return json.dumps({"quality": True, "quality (inverted)": False})
            if "feature code" in prompt:
                return json.dumps({"feature code": True})
            if "brand" in prompt:
                return json.dumps({"brand": "Alpha" in prompt or "Beta" in prompt})
            return json.dumps({"world": True, "sports": False})

        # 2. Filter
        if (
            "entities meeting condition" in prompt.lower()
            or "condition" in prompt.lower()
            and "iPhone" in prompt
        ):
            return json.dumps({"entities meeting condition": ["iPhone"]})

        # 3. Rank (pairwise Circle vs Square)
        if "BEGIN ATTRIBUTES" in prompt and "BEGIN ENTRY CIRCLE" in prompt:
            if "brand" in prompt.lower():
                return json.dumps({"brand": "circle"})
            return json.dumps({"clarity": "circle"})

        # 4. Paraphrase generation (non-JSON plain text rewrite)
        if not json_mode and "Rewrite the passage" in prompt:
            for brand in ("Alpha", "Beta", "Gamma", "Delta"):
                if brand in prompt:
                    return f"Rewritten clean passage for {brand}."
            return "Rewritten clean passage."

        # 5. Discover bucket
        if "BEGIN RAW TERMS" in prompt:
            return json.dumps({"quality": "a distinguishing feature"})

        # 6. Codify static branch
        if "BEGIN CATEGORIES/ATTRIBUTES" in prompt:
            if "ceo" in prompt.lower():
                return json.dumps(
                    {
                        "ceo": [
                            {
                                "beginning excerpt": "CEO John leads",
                                "ending excerpt": "firm.",
                            }
                        ]
                    }
                )
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

        # 7. Remaining signal (rate) in debias
        if "prevalence of ceo" in prompt:
            val = 50 if "Alpha" in prompt else 40
            return json.dumps({"prevalence of ceo": val})

        # 8. Compare in discover pair
        if "sunny" in prompt or "cloudy" in prompt or "circle" in prompt.lower():
            return json.dumps({"quality": "circle has more quality than square"})

        return json.dumps({})

    def estimate_tokens(self, prompt: str, *, model: str) -> int:
        del prompt, model
        return 0

    def estimate_cost(self, prompt: str, *, model: str) -> float:
        del prompt, model
        return 0.0


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


def _codify_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"text": ["First claim: revenue doubled. Second claim: margins rose. End."]}
    )


def _discover_pair_df() -> pd.DataFrame:
    return pd.DataFrame({"circle": ["sunny", "rainy"], "square": ["cloudy", "clear"]})


# ---------------------------------------------------------------------------
# Test 1: conftest isolation sanity
# ---------------------------------------------------------------------------


def test_conftest_clears_decision_env_by_default() -> None:
    """conftest autouse fixture removes all LEXIMETER_DECISION_* env vars."""
    assert all(not k.startswith("LEXIMETER_DECISION_") for k in os.environ)


# ---------------------------------------------------------------------------
# Test 2: Pure-decision api.* with explicit client ignores env protocol
# ---------------------------------------------------------------------------


def test_pure_decision_api_with_explicit_client_ignores_env_decision_protocol(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """classify, filter, rank via api.py ignore env SystemOne when client= is given."""
    _set_systemone_env(monkeypatch)
    _install_failing_systemone(monkeypatch)

    # classify (single-column)
    cls_client = CountingClient({"world": True, "sports": False})
    cls_res = run(
        classify(
            pd.DataFrame({"text": ["global summit", "local match"]}),
            "text",
            labels={"world": "international affairs", "sports": "athletics"},
            save_dir=str(tmp_path / "cls_single"),
            client=cls_client,
        )
    )
    assert cls_client.calls == 2
    assert list(cls_res["world"]) == [True, True]
    assert list(cls_res["sports"]) == [False, False]

    # classify (differentiate=True)
    diff_client = CountingClient({"quality": True})
    diff_res = run(
        classify(
            _discover_pair_df(),
            labels={"quality": "quality difference"},
            save_dir=str(tmp_path / "cls_diff"),
            differentiate=True,
            circle_column_name="circle",
            square_column_name="square",
            client=diff_client,
        )
    )
    assert diff_client.calls == 2
    assert list(diff_res["quality"]) == [True, True]

    # filter
    flt_client = CountingClient({"entities meeting condition": ["iPhone"]})
    flt_res = run(
        leximeter_filter(
            pd.DataFrame({"item": ["iPhone", "banana"]}),
            "item",
            condition="is a phone",
            save_dir=str(tmp_path / "flt"),
            client=flt_client,
            shuffle=False,
        )
    )
    assert flt_client.calls == 1
    assert list(flt_res["keep"]) == [True, False]

    # rank
    rnk_client = CountingClient({"clarity": "circle"})
    rnk_res = run(
        rank(
            pd.DataFrame({"speech": ["alpha", "beta", "gamma"]}),
            "speech",
            attributes={"clarity": "clear writing"},
            save_dir=str(tmp_path / "rnk"),
            client=rnk_client,
            n_rounds=1,
            matches_per_round=1,
        )
    )
    assert rnk_client.calls > 0
    assert list(rnk_res.columns) == ["speech", "clarity", "clarity_raw", "clarity_se"]


# ---------------------------------------------------------------------------
# Test 3: Pure-decision direct constructors ignore env when client= is injected
# ---------------------------------------------------------------------------


def test_pure_decision_direct_task_constructors_ignore_env_decision_protocol(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct Classify, Filter, Rank constructors ignore env when client= is injected."""
    _set_systemone_env(monkeypatch)
    _install_failing_systemone(monkeypatch)

    cls_client = CountingClient({"world": True})
    cls_task = Classify(
        ClassifyConfig(labels={"world": "affairs"}, save_dir=str(tmp_path / "cls")),
        client=cls_client,
    )
    cls_res = run(cls_task.run(pd.DataFrame({"text": ["summit"]}), "text"))
    assert cls_client.calls == 1
    assert list(cls_res["world"]) == [True]

    flt_client = CountingClient({"entities meeting condition": ["iPhone"]})
    flt_task = Filter(
        FilterConfig(condition="is a phone", save_dir=str(tmp_path / "flt")),
        client=flt_client,
    )
    flt_res = run(flt_task.run(pd.DataFrame({"item": ["iPhone", "apple"]}), "item"))
    assert flt_client.calls == 1
    assert list(flt_res["keep"]) == [True, False]

    rnk_client = CountingClient({"clarity": "circle"})
    rnk_task = Rank(
        RankConfig(
            attributes={"clarity": "clear"},
            save_dir=str(tmp_path / "rnk"),
            n_rounds=1,
            matches_per_round=1,
        ),
        client=rnk_client,
    )
    rnk_res = run(rnk_task.run(pd.DataFrame({"speech": ["a", "b", "c"]}), "speech"))
    assert rnk_client.calls > 0
    assert "clarity" in rnk_res.columns


# ---------------------------------------------------------------------------
# Test 4: Pure-decision tasks resolve env SystemOneClient when client= is omitted
# ---------------------------------------------------------------------------


def test_pure_decision_tasks_without_client_resolve_env_systemone(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When client is omitted and decision_client is omitted, env SystemOne is used."""
    _set_systemone_env(monkeypatch)
    calls = _install_spy_systemone(monkeypatch)

    # API facade calls
    run(
        classify(
            pd.DataFrame({"text": ["hello"]}),
            "text",
            labels={"world": "affairs"},
            save_dir=str(tmp_path / "api_cls"),
        )
    )
    run(
        leximeter_filter(
            pd.DataFrame({"item": ["iPhone"]}),
            "item",
            condition="is a phone",
            save_dir=str(tmp_path / "api_flt"),
        )
    )
    run(
        rank(
            pd.DataFrame({"speech": ["a", "b"]}),
            "speech",
            attributes={"clarity": "clear"},
            save_dir=str(tmp_path / "api_rnk"),
            n_rounds=1,
            matches_per_round=1,
        )
    )

    # Direct task constructors
    run(
        Classify(
            ClassifyConfig(
                labels={"world": "affairs"}, save_dir=str(tmp_path / "dir_cls")
            )
        ).run(pd.DataFrame({"text": ["hello"]}), "text")
    )
    run(
        Filter(
            FilterConfig(condition="is a phone", save_dir=str(tmp_path / "dir_flt"))
        ).run(pd.DataFrame({"item": ["iPhone"]}), "item")
    )
    run(
        Rank(
            RankConfig(
                attributes={"clarity": "clear"},
                save_dir=str(tmp_path / "dir_rnk"),
                n_rounds=1,
                matches_per_round=1,
            )
        ).run(pd.DataFrame({"speech": ["a", "b"]}), "speech")
    )

    assert len(calls) == 6
    for url, api_key, model, _ in calls:
        assert url == "https://decision.invalid/api/v1/systemone"
        assert api_key == "sk-dec-test"
        assert model == "env-dec-model"


# ---------------------------------------------------------------------------
# Test 5: Explicit client + explicit decision_client precedence
# ---------------------------------------------------------------------------


def test_explicit_decision_client_beats_both_explicit_client_and_env(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit decision_client wins over both explicit client and env SystemOne."""
    _set_systemone_env(monkeypatch)
    _install_failing_systemone(monkeypatch)

    # Pure-decision tasks: llm untouched, dc handles all calls
    llm_unused = CountingClient({})
    dc_pure = CountingDecisionClient(
        handler=lambda state, key, q: (
            {"circle": 0.9, "square": 0.05, "draw": 0.05, "insufficient_signal": 0.0}
            if isinstance(q, ChoiceQuestion)
            else 0.9
        )
    )
    run(
        classify(
            pd.DataFrame({"text": ["hello"]}),
            "text",
            labels={"world": "affairs"},
            save_dir=str(tmp_path / "p_cls"),
            client=llm_unused,
            decision_client=dc_pure,
        )
    )
    run(
        leximeter_filter(
            pd.DataFrame({"item": ["iPhone"]}),
            "item",
            condition="is a phone",
            save_dir=str(tmp_path / "p_flt"),
            client=llm_unused,
            decision_client=dc_pure,
        )
    )
    run(
        rank(
            pd.DataFrame({"speech": ["a", "b"]}),
            "speech",
            attributes={"clarity": "clear"},
            save_dir=str(tmp_path / "p_rnk"),
            n_rounds=1,
            matches_per_round=1,
            client=llm_unused,
            decision_client=dc_pure,
        )
    )
    assert llm_unused.calls == 0
    assert dc_pure.calls == 3

    # Hybrid tasks: llm handles generative stages, dc handles judgment substages
    for name, coro_fn in (
        (
            "discover",
            lambda llm, dc: discover(
                _discover_pair_df(),
                circle_column_name="circle",
                square_column_name="square",
                save_dir=str(tmp_path / "h_disc"),
                bucket_count=1,
                client=llm,
                decision_client=dc,
            ),
        ),
        (
            "paraphrase",
            lambda llm, dc: paraphrase(
                pd.DataFrame({"text": ["hello"]}),
                "text",
                instructions="rephrase",
                save_dir=str(tmp_path / "h_para"),
                n_rounds=2,
                client=llm,
                decision_client=dc,
            ),
        ),
        (
            "codify",
            lambda llm, dc: codify(
                _codify_df(),
                "text",
                categories={"finance": "financial claims"},
                save_dir=str(tmp_path / "h_cod"),
                n_rounds=2,
                client=llm,
                decision_client=dc,
            ),
        ),
        (
            "debias",
            lambda llm, dc: debias(
                _debias_df(),
                "text",
                save_dir=str(tmp_path / "h_db"),
                mode="classify",
                attributes={"brand": "brand strength"},
                signal_dictionary={"ceo": "mentions of CEO"},
                removal_method="codify",
                removal_kwargs={"n_rounds": 2},
                client=llm,
                decision_client=dc,
                verbose=False,
            ),
        ),
    ):
        llm = _RecordingHybridClient()
        dc = CountingDecisionClient(
            handler=lambda state, key, q: (
                {"option_a": 0.9, "option_b": 0.05, "neither": 0.05}
                if isinstance(q, ChoiceQuestion)
                else (0.1 if key in {"ceo", "finance"} else 0.9)
            )
        )
        run(coro_fn(llm, dc))
        assert dc.calls > 0, f"{name} should use explicit decision_client"
        assert llm.calls > 0, f"{name} should run generative stages on llm"
        assert not any("BEGIN LABELS" in p for p in llm.prompts), (
            f"{name} must not fall back to LLMClient for classify"
        )


# ---------------------------------------------------------------------------
# Test 6: Hybrid tasks with explicit client preserve env Decision Provider
# ---------------------------------------------------------------------------


def test_hybrid_tasks_with_explicit_client_preserve_env_decision_provider(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hybrid tasks preserve env SystemOneClient when client= is given & decision_client omitted."""
    _set_systemone_env(monkeypatch)
    calls = _install_spy_systemone(monkeypatch)

    def _run_and_assert(label: str, llm: _RecordingHybridClient, coro: Any) -> None:
        before = len(calls)
        run(coro)
        new_calls = calls[before:]
        assert len(new_calls) > 0, f"{label} should hit env SystemOneClient"
        assert llm.calls > 0, f"{label} should run generative stages on llm"
        assert not any("BEGIN LABELS" in p for p in llm.prompts), (
            f"{label} must not route classify to llm"
        )
        assert not any(
            "BEGIN ATTRIBUTES" in p and "BEGIN ENTRY CIRCLE" in p for p in llm.prompts
        ), f"{label} must not route rank to llm"
        for url, api_key, model, _ in new_calls:
            assert url == "https://decision.invalid/api/v1/systemone"
            assert api_key == "sk-dec-test"
            assert model == "env-dec-model"

    # Via api.*
    llm1 = _RecordingHybridClient()
    _run_and_assert(
        "api.discover",
        llm1,
        discover(
            _discover_pair_df(),
            circle_column_name="circle",
            square_column_name="square",
            save_dir=str(tmp_path / "api_disc"),
            bucket_count=1,
            client=llm1,
        ),
    )

    llm2 = _RecordingHybridClient()
    _run_and_assert(
        "api.paraphrase",
        llm2,
        paraphrase(
            pd.DataFrame({"text": ["hello"]}),
            "text",
            instructions="rephrase",
            save_dir=str(tmp_path / "api_para"),
            n_rounds=2,
            client=llm2,
        ),
    )

    llm3 = _RecordingHybridClient()
    _run_and_assert(
        "api.codify",
        llm3,
        codify(
            _codify_df(),
            "text",
            categories={"finance": "financial claims"},
            save_dir=str(tmp_path / "api_cod"),
            n_rounds=2,
            client=llm3,
        ),
    )

    llm4 = _RecordingHybridClient()
    _run_and_assert(
        "api.debias(classify)",
        llm4,
        debias(
            _debias_df(),
            "text",
            save_dir=str(tmp_path / "api_db_cls"),
            mode="classify",
            attributes={"brand": "brand strength"},
            signal_dictionary={"ceo": "mentions of CEO"},
            removal_method="codify",
            removal_kwargs={"n_rounds": 2},
            client=llm4,
            verbose=False,
        ),
    )

    llm5 = _RecordingHybridClient()
    _run_and_assert(
        "api.debias(rank)",
        llm5,
        debias(
            _debias_df(),
            "text",
            save_dir=str(tmp_path / "api_db_rnk"),
            mode="rank",
            attributes={"brand": "brand strength"},
            signal_dictionary={"ceo": "mentions of CEO"},
            removal_method="codify",
            removal_kwargs={"n_rounds": 2},
            measurement_kwargs={"n_rounds": 1, "matches_per_round": 1},
            client=llm5,
            verbose=False,
        ),
    )

    # Via direct Task constructors
    llm6 = _RecordingHybridClient()
    _run_and_assert(
        "Discover()",
        llm6,
        Discover(
            DiscoverConfig(save_dir=str(tmp_path / "dir_disc"), bucket_count=1),
            client=llm6,
        ).run(
            _discover_pair_df(),
            circle_column_name="circle",
            square_column_name="square",
        ),
    )

    llm7 = _RecordingHybridClient()
    _run_and_assert(
        "Paraphrase()",
        llm7,
        Paraphrase(
            ParaphraseConfig(
                instructions="rephrase",
                save_dir=str(tmp_path / "dir_para"),
                n_rounds=2,
            ),
            client=llm7,
        ).run(pd.DataFrame({"text": ["hello"]}), "text"),
    )

    llm8 = _RecordingHybridClient()
    _run_and_assert(
        "Codify()",
        llm8,
        Codify(
            CodifyConfig(save_dir=str(tmp_path / "dir_cod"), n_rounds=2),
            client=llm8,
        ).run(
            _codify_df(),
            "text",
            categories={"finance": "financial claims"},
        ),
    )

    llm9 = _RecordingHybridClient()
    _run_and_assert(
        "Debias()",
        llm9,
        Debias(
            DebiasConfig(
                save_dir=str(tmp_path / "dir_db"),
                mode="classify",
                attributes={"brand": "brand strength"},
                signal_dictionary={"ceo": "mentions of CEO"},
                removal_method="codify",
                removal_kwargs={"n_rounds": 2},
                verbose=False,
            ),
            client=llm9,
        ).run(_debias_df(), "text"),
    )


# ---------------------------------------------------------------------------
# Test 7: Explicit decision_client=None opt-out and child propagation
# ---------------------------------------------------------------------------


def test_explicit_decision_client_none_opts_out_and_propagates_to_children(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit decision_client=None disables env DecisionClient across parents & children."""
    _set_systemone_env(monkeypatch)
    _install_failing_systemone(monkeypatch)

    # Direct constructor _get_decision_client() check on all 7 task classes
    assert (
        Classify(
            ClassifyConfig(labels={"a": "b"}, save_dir=str(tmp_path / "u1")),
            decision_client=None,
        )._get_decision_client()
        is None
    )
    assert (
        Filter(
            FilterConfig(condition="c", save_dir=str(tmp_path / "u2")),
            decision_client=None,
        )._get_decision_client()
        is None
    )
    assert (
        Rank(
            RankConfig(attributes={"a": "b"}, save_dir=str(tmp_path / "u3")),
            decision_client=None,
        )._get_decision_client()
        is None
    )
    assert (
        Discover(
            DiscoverConfig(save_dir=str(tmp_path / "u4")),
            decision_client=None,
        )._get_decision_client()
        is None
    )
    assert (
        Paraphrase(
            ParaphraseConfig(instructions="i", save_dir=str(tmp_path / "u5")),
            decision_client=None,
        )._get_decision_client()
        is None
    )
    assert (
        Codify(
            CodifyConfig(save_dir=str(tmp_path / "u6")),
            decision_client=None,
        )._get_decision_client()
        is None
    )
    assert (
        Debias(
            DebiasConfig(
                save_dir=str(tmp_path / "u7"),
                attributes={"a": "b"},
                signal_dictionary={"s": "d"},
            ),
            decision_client=None,
        )._get_decision_client()
        is None
    )

    # Pure-decision tasks with client=llm AND client=None when decision_client=None
    llm_pure_explicit = _RecordingHybridClient()
    run(
        classify(
            pd.DataFrame({"text": ["hello"]}),
            "text",
            labels={"world": "affairs"},
            save_dir=str(tmp_path / "explicit_llm_cls"),
            client=llm_pure_explicit,
            decision_client=None,
        )
    )
    run(
        leximeter_filter(
            pd.DataFrame({"item": ["iPhone"]}),
            "item",
            condition="is a phone",
            save_dir=str(tmp_path / "explicit_llm_flt"),
            client=llm_pure_explicit,
            decision_client=None,
        )
    )
    run(
        rank(
            pd.DataFrame({"speech": ["a", "b"]}),
            "speech",
            attributes={"clarity": "clear"},
            save_dir=str(tmp_path / "explicit_llm_rnk"),
            n_rounds=1,
            matches_per_round=1,
            client=llm_pure_explicit,
            decision_client=None,
        )
    )
    assert llm_pure_explicit.calls >= 3

    llm_pure_env = _RecordingHybridClient()
    with monkeypatch.context() as mp:
        mp.setattr("leximeter.client.resolve.resolve_client", lambda **kw: llm_pure_env)
        run(
            classify(
                pd.DataFrame({"text": ["hello"]}),
                "text",
                labels={"world": "affairs"},
                save_dir=str(tmp_path / "env_llm_cls"),
                client=None,
                decision_client=None,
            )
        )
        run(
            leximeter_filter(
                pd.DataFrame({"item": ["iPhone"]}),
                "item",
                condition="is a phone",
                save_dir=str(tmp_path / "env_llm_flt"),
                client=None,
                decision_client=None,
            )
        )
        run(
            rank(
                pd.DataFrame({"speech": ["a", "b"]}),
                "speech",
                attributes={"clarity": "clear"},
                save_dir=str(tmp_path / "env_llm_rnk"),
                n_rounds=1,
                matches_per_round=1,
                client=None,
                decision_client=None,
            )
        )
    assert llm_pure_env.calls >= 3

    # Composite tasks with decision_client=None propagate None to child tasks
    llm_disc = _RecordingHybridClient()
    run(
        discover(
            _discover_pair_df(),
            circle_column_name="circle",
            square_column_name="square",
            save_dir=str(tmp_path / "none_disc"),
            bucket_count=1,
            client=llm_disc,
            decision_client=None,
        )
    )
    assert any("BEGIN LABELS" in p for p in llm_disc.prompts)

    llm_para = _RecordingHybridClient()
    run(
        paraphrase(
            pd.DataFrame({"text": ["hello"]}),
            "text",
            instructions="rephrase",
            save_dir=str(tmp_path / "none_para"),
            n_rounds=2,
            client=llm_para,
            decision_client=None,
        )
    )
    assert any("BEGIN LABELS" in p for p in llm_para.prompts)

    llm_cod = _RecordingHybridClient()
    run(
        codify(
            _codify_df(),
            "text",
            categories={"finance": "financial claims"},
            save_dir=str(tmp_path / "none_cod"),
            n_rounds=2,
            client=llm_cod,
            decision_client=None,
        )
    )
    assert any("BEGIN LABELS" in p for p in llm_cod.prompts)

    # Debias top-level decision_client=None propagates to Classify/Rank + Codify/Paraphrase
    llm_db_cls = _RecordingHybridClient()
    run(
        debias(
            _debias_df(),
            "text",
            save_dir=str(tmp_path / "none_db_cls"),
            mode="classify",
            attributes={"brand": "brand strength"},
            signal_dictionary={"ceo": "mentions of CEO"},
            removal_method="codify",
            removal_kwargs={"n_rounds": 2},
            client=llm_db_cls,
            decision_client=None,
            verbose=False,
        )
    )
    assert any("BEGIN LABELS" in p and "brand" in p for p in llm_db_cls.prompts)
    assert any("BEGIN LABELS" in p and "ceo" in p for p in llm_db_cls.prompts)

    llm_db_rnk = _RecordingHybridClient()
    run(
        debias(
            _debias_df(),
            "text",
            save_dir=str(tmp_path / "none_db_rnk"),
            mode="rank",
            attributes={"brand": "brand strength"},
            signal_dictionary={"ceo": "mentions of CEO"},
            removal_method="paraphrase",
            removal_kwargs={"n_rounds": 2},
            measurement_kwargs={"n_rounds": 1, "matches_per_round": 1},
            client=llm_db_rnk,
            decision_client=None,
            verbose=False,
        )
    )
    assert any(
        "BEGIN ATTRIBUTES" in p and "BEGIN ENTRY CIRCLE" in p
        for p in llm_db_rnk.prompts
    )
    assert any(
        "BEGIN LABELS" in p and "instructions_followed" in p for p in llm_db_rnk.prompts
    )

    # Debias per-stage kwargs opt-out: measurement_kwargs={"decision_client": None}
    # and removal_kwargs={"decision_client": None} must NOT fall back to env SystemOne
    llm_db_kw_cod = _RecordingHybridClient()
    run(
        debias(
            _debias_df(),
            "text",
            save_dir=str(tmp_path / "kw_none_cod"),
            mode="classify",
            attributes={"brand": "brand strength"},
            signal_dictionary={"ceo": "mentions of CEO"},
            removal_method="codify",
            measurement_kwargs={"decision_client": None},
            removal_kwargs={"n_rounds": 2, "decision_client": None},
            client=llm_db_kw_cod,
            verbose=False,
        )
    )
    assert any("BEGIN LABELS" in p and "brand" in p for p in llm_db_kw_cod.prompts)
    assert any("BEGIN LABELS" in p and "ceo" in p for p in llm_db_kw_cod.prompts)

    llm_db_kw_para = _RecordingHybridClient()
    run(
        debias(
            _debias_df(),
            "text",
            save_dir=str(tmp_path / "kw_none_para"),
            mode="rank",
            attributes={"brand": "brand strength"},
            signal_dictionary={"ceo": "mentions of CEO"},
            removal_method="paraphrase",
            measurement_kwargs={
                "n_rounds": 1,
                "matches_per_round": 1,
                "decision_client": None,
            },
            removal_kwargs={"n_rounds": 2, "decision_client": None},
            client=llm_db_kw_para,
            verbose=False,
        )
    )
    assert any(
        "BEGIN ATTRIBUTES" in p and "BEGIN ENTRY CIRCLE" in p
        for p in llm_db_kw_para.prompts
    )
    assert any(
        "BEGIN LABELS" in p and "instructions_followed" in p
        for p in llm_db_kw_para.prompts
    )
