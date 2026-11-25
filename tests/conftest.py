"""Shared fake providers for the test suite."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from leximeter.client.base import LLMClient
from leximeter.client.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionClient,
    DecisionQuestion,
    DecisionResult,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)


class DictClient(LLMClient):
    """Return a fixed JSON payload for every prompt."""

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
        return json.dumps(self._payload)

    def estimate_tokens(self, prompt, *, model) -> int:
        return 0

    def estimate_cost(self, prompt, *, model) -> float:
        return 0.0


class CountingClient(DictClient):
    """DictClient that also counts how many completions were actually made."""

    def __init__(self, payload: dict[str, Any]):
        super().__init__(payload)
        self.calls = 0

    async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
        self.calls += 1
        return await super().complete(
            prompt, model=model, json_mode=json_mode, **kwargs
        )


class RecordingClient(DictClient):
    """DictClient that records every rendered prompt in dispatch order."""

    def __init__(self, payload: dict[str, Any]):
        super().__init__(payload)
        self.prompts: list[str] = []

    async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
        self.prompts.append(prompt)
        return await super().complete(
            prompt, model=model, json_mode=json_mode, **kwargs
        )


class TextClient(LLMClient):
    """Return a fixed plain-text string for every prompt."""

    def __init__(self, text: str):
        self._text = text

    async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
        return self._text

    def estimate_tokens(self, prompt, *, model) -> int:
        return 0

    def estimate_cost(self, prompt, *, model) -> float:
        return 0.0


def _coerce_decision_answer(
    spec: Any,
    question: DecisionQuestion,
    state: str | dict[str, Any],
    key: str,
) -> DecisionAnswer:
    if callable(spec):
        spec = spec(state, key, question)
    if isinstance(spec, (NoulAnswer, ChoiceAnswer, ScoreAnswer)):
        return spec
    if isinstance(question, NoulQuestion):
        prob = float(spec) if spec is not None else None
        return NoulAnswer(
            probability=prob,
            confidence=abs(2.0 * prob - 1.0) if prob is not None else None,
            refused=prob is None,
        )
    if isinstance(question, ChoiceQuestion):
        options = list(question.criteria.keys())
        if isinstance(spec, dict):
            probs = {str(k): float(v) for k, v in spec.items()}
            chosen = max(probs, key=lambda k: probs[k]) if probs else None
            conf = probs[chosen] if chosen is not None else None
            return ChoiceAnswer(choice=chosen, probabilities=probs, confidence=conf)
        if isinstance(spec, str):
            probs = {opt: (1.0 if opt == spec else 0.0) for opt in options}
            return ChoiceAnswer(choice=spec, probabilities=probs, confidence=1.0)
        if spec is None:
            return ChoiceAnswer(choice=None, probabilities={}, refused=True)
    if isinstance(question, ScoreQuestion):
        n_levels = len(question.criteria)
        if isinstance(spec, dict):
            probs_s = {int(k): float(v) for k, v in spec.items()}
            score = sum(k * v for k, v in probs_s.items())
            return ScoreAnswer(score=score, probabilities=probs_s)
        if spec is not None:
            val = float(spec)
            probs_s = {i: (1.0 / n_levels) for i in range(n_levels)} if n_levels else {}
            return ScoreAnswer(score=val, probabilities=probs_s)
        return ScoreAnswer(score=None, probabilities={}, refused=True)
    raise TypeError(f"Cannot coerce {spec!r} for question {question!r}")


class DictDecisionClient(DecisionClient):
    """Configurable offline DecisionClient for task-seam tests."""

    def __init__(
        self,
        answers: dict[str, Any] | None = None,
        *,
        default: Any = 0.5,
        handler: Callable[[str | dict[str, Any], str, DecisionQuestion], Any]
        | None = None,
        max_questions_per_call: int = 64,
    ) -> None:
        self._answers = dict(answers or {})
        self._default = default
        self._handler = handler
        self.max_questions_per_call = max(1, int(max_questions_per_call))

    async def decide(
        self,
        state: str | dict[str, Any],
        questions: dict[str, DecisionQuestion],
        *,
        model: str = "dummy-decision",
        **kwargs: Any,
    ) -> DecisionResult:
        del kwargs
        out: dict[str, DecisionAnswer] = {}
        for key, question in questions.items():
            if self._handler is not None:
                spec = self._handler(state, key, question)
            elif key in self._answers:
                spec = self._answers[key]
            else:
                spec = self._default
            out[key] = _coerce_decision_answer(spec, question, state, key)
        return DecisionResult(answers=out, model=model)


class CountingDecisionClient(DictDecisionClient):
    """DictDecisionClient that counts calls and records every request."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.requests: list[
            tuple[str | dict[str, Any], dict[str, DecisionQuestion], str]
        ] = []

    async def decide(
        self,
        state: str | dict[str, Any],
        questions: dict[str, DecisionQuestion],
        *,
        model: str = "dummy-decision",
        **kwargs: Any,
    ) -> DecisionResult:
        self.calls += 1
        self.requests.append((state, dict(questions), model))
        return await super().decide(state, questions, model=model, **kwargs)
