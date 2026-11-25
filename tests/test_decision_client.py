"""Tests for DecisionClient, SystemOneClient (wire seam), and DummyDecisionClient.

All tests run offline using ``httpx.MockTransport`` — zero API keys, zero network.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

from leximeter.client.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAPIError,
    DecisionResult,
    DummyDecisionClient,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from leximeter.client.systemone import SystemOneClient


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_systemone_sanitizes_keys_and_normalizes_primitives() -> None:
    """Labels with spaces, punctuation, and non-ASCII round-trip via q_0..q_n."""
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content.decode("utf-8"))
        captured.append(body)
        # Provider receives only clean ASCII wire keys q_0..q_2
        assert list(body["questions"].keys()) == ["q_0", "q_1", "q_2"]
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14.0",
                "answers": {
                    "q_0": {"type": "noul", "noul": 0.87},
                    "q_1": {
                        "type": "choice",
                        "choice": "technical",
                        "confidence": 0.92,
                        "probabilities": {"billing": 0.05, "technical": 0.95},
                    },
                    "q_2": {
                        "type": "score",
                        "score": 1.4,
                        "confidence": 0.6,
                        "legend": {"0": "Low", "1": "Medium", "2": "High"},
                        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
                    },
                },
                "usage": {"input_tokens": 120, "output_tokens": 0},
            },
        )

    client = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )

    k_noul = "Cost > $100 (USD)?"
    k_choice = "Routing: team / queue"
    k_score = "情绪强度 (中文)"

    result = run(
        client.decide(
            state="Our checkout API has returned 500 errors since noon.",
            questions={
                k_noul: NoulQuestion(
                    instructions="Does this mention a cost above $100?",
                    true_criteria="Explicitly states a cost exceeding $100",
                    false_criteria="No such cost mentioned",
                ),
                k_choice: ChoiceQuestion(
                    instructions="Which team should handle this?",
                    criteria={
                        "billing": "Payment or invoice issues",
                        "technical": "Bugs or API outages",
                    },
                ),
                k_score: ScoreQuestion(
                    instructions="How intense is the emotion?",
                    criteria=["Calm", "Frustrated", "Furious"],
                ),
            },
            model="jev-latest",
        )
    )

    # Wire request shape
    req = captured[0]
    assert req["model"] == "jev-latest"
    assert "instructions" not in req
    assert req["questions"]["q_0"]["type"] == "noul"
    assert req["questions"]["q_0"]["criteria"] == {
        "true": "Explicitly states a cost exceeding $100",
        "false": "No such cost mentioned",
    }
    assert req["questions"]["q_1"]["type"] == "choice"
    assert req["questions"]["q_2"]["type"] == "score"
    assert req["questions"]["q_2"]["criteria"] == ["Calm", "Frustrated", "Furious"]

    # Answers mapped back to verbatim caller keys
    assert set(result.answers.keys()) == {k_noul, k_choice, k_score}
    assert result.model == "jev-1.14.0"
    assert result.input_tokens == 120
    assert result.output_tokens == 0
    assert result.cost_usd is None

    ans_noul = result.answers[k_noul]
    assert isinstance(ans_noul, NoulAnswer)
    assert ans_noul.probability == 0.87
    assert ans_noul.confidence is None
    assert ans_noul.refused is False

    ans_choice = result.answers[k_choice]
    assert isinstance(ans_choice, ChoiceAnswer)
    assert ans_choice.choice == "technical"
    assert ans_choice.probabilities == {"billing": 0.05, "technical": 0.95}
    assert ans_choice.confidence == 0.92
    assert ans_choice.refused is False

    ans_score = result.answers[k_score]
    assert isinstance(ans_score, ScoreAnswer)
    assert ans_score.score == 1.4
    assert ans_score.probabilities == {0: 0.1, 1: 0.4, 2: 0.5}
    assert ans_score.confidence == 0.6
    assert ans_score.refused is False


def test_systemone_d3_strict_mode_omits_noul_criteria_and_top_level_instructions() -> (
    None
):
    """AMD Token Factory d3/d3-flash rejects noul criteria and top-level instructions."""
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # base_url ending in /v1 must not duplicate /v1/v1/systemone
        assert (
            str(request.url) == "https://developer.amd.com.cn/radeon/api/v1/systemone"
        )
        body = json.loads(request.content.decode("utf-8"))
        captured.append(body)
        return httpx.Response(
            200,
            json={
                "id": "decision-123",
                "object": "structured_decision",
                "model": "d3",
                "answers": {
                    "q_0": {
                        "type": "noul",
                        "noul": 0.9967,
                        "probabilities": {"yes": 0.9967, "no": 0.0033},
                        "confidence": 0.9967,
                    }
                },
                "usage": {"input_tokens": 50, "output_tokens": 0},
                "diagnostics": {"q_0": {"label_mass": 1.0, "argmax_is_label": True}},
            },
        )

    client = SystemOneClient(
        base_url="https://developer.amd.com.cn/radeon/api/v1",
        api_key="rc-test",
        send_noul_criteria=False,
        transport=httpx.MockTransport(handler),
    )

    result = run(
        client.decide(
            state="System down across all regions.",
            questions={
                "urgent": NoulQuestion(
                    instructions="Is this urgent?",
                    true_criteria="Critical outage",
                    false_criteria="Routine",
                )
            },
            model="d3",
        )
    )

    body = captured[0]
    assert "instructions" not in body
    assert "criteria" not in body["questions"]["q_0"]
    assert body["questions"]["q_0"]["instructions"] == "Is this urgent?"
    ans = result.answers["urgent"]
    assert isinstance(ans, NoulAnswer)
    assert ans.probability == pytest.approx(0.9967)
    assert ans.confidence == pytest.approx(0.9967)
    assert ans.refused is False


def test_systemone_openrouter_refusal_preserves_siblings_and_parses_cost() -> None:
    """Per-question refusal normalizes to refused=True without failing sibling questions."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
        assert request.headers["typesafe-danger-zone-long-context"] == "true"
        return httpx.Response(
            200,
            json={
                "id": "or-dec-1",
                "provider": "OpenAI",
                "model": "openai/gpt-6-luna-decisions",
                "answers": {
                    "q_0": {"type": "refusal"},
                    "q_1": {"type": "noul", "noul": 0.82},
                    "q_2": {"type": "refusal"},
                    "q_3": {"type": "refusal"},
                },
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 0,
                    "cost": 0.00002,
                },
            },
        )

    client = SystemOneClient(
        base_url="https://openrouter.ai",
        endpoint_path="/api/alpha/decisions",
        api_key="or-test",
        long_context=True,
        transport=httpx.MockTransport(handler),
    )

    result = run(
        client.decide(
            state="Sensitive passage",
            questions={
                "blocked_noul": NoulQuestion(instructions="Blocked check"),
                "allowed_noul": NoulQuestion(instructions="Allowed check"),
                "blocked_choice": ChoiceQuestion(
                    instructions="Pick one", criteria={"a": "A", "b": "B"}
                ),
                "blocked_score": ScoreQuestion(
                    instructions="Rate", criteria=["Low", "High"]
                ),
            },
            model="openai/gpt-6-luna-decisions",
        )
    )

    assert result.cost_usd == pytest.approx(0.00002)

    a0 = result.answers["blocked_noul"]
    assert isinstance(a0, NoulAnswer)
    assert a0.refused is True
    assert a0.probability is None

    a1 = result.answers["allowed_noul"]
    assert isinstance(a1, NoulAnswer)
    assert a1.refused is False
    assert a1.probability == pytest.approx(0.82)

    a2 = result.answers["blocked_choice"]
    assert isinstance(a2, ChoiceAnswer)
    assert a2.refused is True
    assert a2.choice is None
    assert a2.probabilities == {}

    a3 = result.answers["blocked_score"]
    assert isinstance(a3, ScoreAnswer)
    assert a3.refused is True
    assert a3.score is None
    assert a3.probabilities == {}


def test_systemone_retries_429_honoring_retry_after() -> None:
    """HTTP 429 retries after the Retry-After interval and succeeds."""
    timestamps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timestamps.append(time.monotonic())
        if len(timestamps) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0.05"},
                json={"error": "rate_limited"},
            )
        return httpx.Response(
            200,
            json={
                "model": "jev-latest",
                "answers": {"q_0": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        )

    client = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        max_retries=2,
        transport=httpx.MockTransport(handler),
    )

    result = run(
        client.decide(
            state="hello",
            questions={"ok": NoulQuestion(instructions="Is greeting?")},
            model="jev-latest",
        )
    )
    assert len(timestamps) == 2
    assert timestamps[1] - timestamps[0] >= 0.04
    ans = result.answers["ok"]
    assert isinstance(ans, NoulAnswer)
    assert ans.probability == pytest.approx(0.9)


def test_systemone_does_not_retry_non_retryable_400() -> None:
    """Client errors like HTTP 400 fail fast on the first attempt."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "context length exceeded"})

    client = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        max_retries=3,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(DecisionAPIError) as exc_info:
        run(
            client.decide(
                state="too long",
                questions={"k": NoulQuestion(instructions="q")},
                model="d3",
            )
        )
    assert calls == 1
    assert exc_info.value.status_code == 400
    assert exc_info.value.retryable is False


def test_dummy_decision_client_and_result_json_roundtrip() -> None:
    """DummyDecisionClient works offline across all primitives and round-trips JSON."""
    dummy = DummyDecisionClient(noul_probability=0.75, default_choice="square")
    result = run(
        dummy.decide(
            state="sample text",
            questions={
                "flag": NoulQuestion(instructions="Applies?"),
                "dir": ChoiceQuestion(
                    instructions="Which side?",
                    criteria={"circle": "C", "square": "S", "neither": "N"},
                ),
                "intensity": ScoreQuestion(
                    instructions="How strong?",
                    criteria=["None", "Mild", "Strong"],
                ),
            },
            model="dummy-decision",
        )
    )

    restored = DecisionResult.from_json(result.to_json())
    assert restored == result
    assert isinstance(restored.answers["flag"], NoulAnswer)
    assert restored.answers["flag"].probability == pytest.approx(0.75)
    assert isinstance(restored.answers["dir"], ChoiceAnswer)
    assert restored.answers["dir"].choice == "square"
    assert isinstance(restored.answers["intensity"], ScoreAnswer)
    assert restored.answers["intensity"].score == pytest.approx(1.0)


def test_batch_decide_checkpoints_and_resumes_with_zero_duplicate_calls(
    tmp_path,
) -> None:
    """batch_decide writes normalized DecisionResult JSON and skips done IDs on resume."""
    from conftest import CountingDecisionClient

    from leximeter.engine import batch_decide

    save_path = str(tmp_path / "decisions.csv")
    client = CountingDecisionClient({"positive": 0.85})
    q = {"positive": NoulQuestion(instructions="Is it positive?")}

    # First run: 2 items
    df1 = run(
        batch_decide(
            states=["great day", "bad day"],
            question_sets=[q, q],
            identifiers=["id_1", "id_2"],
            client=client,
            model="clef-flash",
            save_path=save_path,
            n_parallels=2,
        )
    )
    assert client.calls == 2
    assert len(df1) == 2
    assert df1["Successful"].all()

    # Check raw probability persisted in checkpoint Response JSON
    parsed = DecisionResult.from_json(
        df1.loc[df1["Identifier"] == "id_1", "Response"].iloc[0]
    )
    assert parsed is not None
    assert isinstance(parsed.answers["positive"], NoulAnswer)
    assert parsed.answers["positive"].probability == pytest.approx(0.85)

    # Resume with 1 new item added: only the new item triggers a DecisionClient call
    df2 = run(
        batch_decide(
            states=["great day", "bad day", "mixed day"],
            question_sets=[q, q, q],
            identifiers=["id_1", "id_2", "id_3"],
            client=client,
            model="clef-flash",
            save_path=save_path,
            n_parallels=2,
            reset_files=False,
        )
    )
    assert client.calls == 3  # only id_3 was evaluated
    assert set(df2["Identifier"]) == {"id_1", "id_2", "id_3"}

    # Re-run completed set: zero additional calls
    run(
        batch_decide(
            states=["great day", "bad day", "mixed day"],
            question_sets=[q, q, q],
            identifiers=["id_1", "id_2", "id_3"],
            client=client,
            model="clef-flash",
            save_path=save_path,
            reset_files=False,
        )
    )
    assert client.calls == 3


def test_batch_decide_retries_honoring_retry_after_and_fails_fast_on_400(
    tmp_path,
) -> None:
    """decide_with_retry honors Retry-After on retryable errors and skips retries on 400."""
    from leximeter.client.decision import DecisionClient
    from leximeter.engine import batch_decide

    class FlakyDecisionClient(DecisionClient):
        def __init__(self) -> None:
            self.timestamps: dict[str, list[float]] = {"retry_me": [], "bad_req": []}

        async def decide(self, state, questions, *, model, **kwargs):
            key = str(state)
            self.timestamps[key].append(time.monotonic())
            if key == "bad_req":
                raise DecisionAPIError(
                    "HTTP 400 invalid input", status_code=400, retryable=False
                )
            if len(self.timestamps[key]) == 1:
                raise DecisionAPIError(
                    "HTTP 429 rate limited",
                    status_code=429,
                    retry_after=0.05,
                    retryable=True,
                )
            return DecisionResult(
                answers={"ok": NoulAnswer(probability=0.9)}, model=model
            )

    flaky = FlakyDecisionClient()
    q = {"ok": NoulQuestion(instructions="ok?")}
    df = run(
        batch_decide(
            states=["retry_me", "bad_req"],
            question_sets=[q, q],
            identifiers=["id_retry", "id_bad"],
            client=flaky,
            model="d3",
            max_retries=2,
            save_path=str(tmp_path / "retry.csv"),
        )
    )
    row_retry = df.loc[df["Identifier"] == "id_retry"].iloc[0]
    row_bad = df.loc[df["Identifier"] == "id_bad"].iloc[0]
    assert bool(row_retry["Successful"]) is True
    assert bool(row_bad["Successful"]) is False
    assert len(flaky.timestamps["retry_me"]) == 2
    assert flaky.timestamps["retry_me"][1] - flaky.timestamps["retry_me"][0] >= 0.04
    assert len(flaky.timestamps["bad_req"]) == 1  # failed fast, no wasted retries


def test_batch_decide_with_systemone_no_nested_retry_amplification(tmp_path) -> None:
    """SystemOneClient x batch_decide executes a single retry layer (never N x M)."""
    from leximeter.engine import batch_decide

    q = {"ok": NoulQuestion(instructions="ok?")}

    # 1. Default SystemOneClient() has max_retries=0; batch_decide(max_retries=2) -> 3 HTTP calls
    default_calls = 0

    def always_429_default(request: httpx.Request) -> httpx.Response:
        nonlocal default_calls
        default_calls += 1
        return httpx.Response(
            429, headers={"Retry-After": "0"}, json={"error": "rate_limited"}
        )

    client_default = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        transport=httpx.MockTransport(always_429_default),
    )
    assert client_default._max_retries == 0

    df_default = run(
        batch_decide(
            states=["sample"],
            question_sets=[q],
            identifiers=["id_1"],
            client=client_default,
            model="jev-latest",
            max_retries=2,
            save_path=str(tmp_path / "single_layer_default.csv"),
        )
    )
    assert default_calls == 3  # 1 initial + 2 retries (never 9 or 12)
    assert bool(df_default["Successful"].iloc[0]) is False

    # 2. Explicit SystemOneClient(max_retries=2) inside batch_decide(max_retries=3) -> still 3 HTTP calls
    explicit_calls = 0

    def always_503_explicit(request: httpx.Request) -> httpx.Response:
        nonlocal explicit_calls
        explicit_calls += 1
        return httpx.Response(
            503, headers={"Retry-After": "0"}, json={"error": "unavailable"}
        )

    client_explicit = SystemOneClient(
        base_url="https://api.typesafe.ai",
        api_key="test-key",
        max_retries=2,
        backoff_base=0.0,
        transport=httpx.MockTransport(always_503_explicit),
    )
    df_explicit = run(
        batch_decide(
            states=["sample"],
            question_sets=[q],
            identifiers=["id_1"],
            client=client_explicit,
            model="jev-latest",
            max_retries=3,
            save_path=str(tmp_path / "single_layer_explicit.csv"),
        )
    )
    assert (
        explicit_calls == 3
    )  # client layer handles retries; batch_decide does not stack
    assert bool(df_explicit["Successful"].iloc[0]) is False
