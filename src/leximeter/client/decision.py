"""Decision Provider protocol abstraction (separate from the generative LLM client).

Models typed text-only decision primitives (``noul``, ``choice``, ``score``)
evaluated in parallel against a shared ``state`` in a single forward pass,
returning calibrated probabilities without prose generation (ADR-0004).
"""

from __future__ import annotations

import email.utils
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Typed question primitives (text-only per ADR-0002)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoulQuestion:
    """Binary yes/no decision question."""

    instructions: str | dict[str, Any]
    true_criteria: str | None = None
    false_criteria: str | None = None


@dataclass(frozen=True)
class ChoiceQuestion:
    """Single-select categorical decision question."""

    instructions: str | dict[str, Any]
    criteria: dict[str, str | None]


@dataclass(frozen=True)
class ScoreQuestion:
    """Ordinal rubric rating decision question (0-indexed levels, low -> high)."""

    instructions: str | dict[str, Any]
    criteria: list[str]


DecisionQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion

# ---------------------------------------------------------------------------
# Normalized typed answers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoulAnswer:
    """Normalized answer for a :class:`NoulQuestion`."""

    probability: float | None
    confidence: float | None = None
    refused: bool = False


@dataclass(frozen=True)
class ChoiceAnswer:
    """Normalized answer for a :class:`ChoiceQuestion`."""

    choice: str | None
    probabilities: dict[str, float]
    confidence: float | None = None
    refused: bool = False


@dataclass(frozen=True)
class ScoreAnswer:
    """Normalized answer for a :class:`ScoreQuestion`."""

    score: float | None
    probabilities: dict[int, float]
    confidence: float | None = None
    refused: bool = False


DecisionAnswer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class DecisionResult:
    """Normalized response from a single :meth:`DecisionClient.decide` call."""

    answers: dict[str, DecisionAnswer]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict preserving raw probabilities."""
        serialized_answers: dict[str, dict[str, Any]] = {}
        for key, ans in self.answers.items():
            if isinstance(ans, NoulAnswer):
                serialized_answers[key] = {
                    "type": "noul",
                    "probability": ans.probability,
                    "confidence": ans.confidence,
                    "refused": ans.refused,
                }
            elif isinstance(ans, ChoiceAnswer):
                serialized_answers[key] = {
                    "type": "choice",
                    "choice": ans.choice,
                    "probabilities": dict(ans.probabilities),
                    "confidence": ans.confidence,
                    "refused": ans.refused,
                }
            elif isinstance(ans, ScoreAnswer):
                serialized_answers[key] = {
                    "type": "score",
                    "score": ans.score,
                    "probabilities": {
                        str(level): prob for level, prob in ans.probabilities.items()
                    },
                    "confidence": ans.confidence,
                    "refused": ans.refused,
                }
            else:
                raise TypeError(f"Unsupported DecisionAnswer type: {type(ans)!r}")
        payload: dict[str, Any] = {
            "model": self.model,
            "answers": serialized_answers,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
        if self.cost_usd is not None:
            payload["cost_usd"] = self.cost_usd
        return payload

    def to_json(self) -> str:
        """Serialize to a JSON string for checkpoint persistence."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecisionResult:
        """Reconstruct a :class:`DecisionResult` from :meth:`to_dict` output."""
        raw_answers = data.get("answers")
        answers: dict[str, DecisionAnswer] = {}
        if isinstance(raw_answers, Mapping):
            for key, item in raw_answers.items():
                if not isinstance(item, Mapping):
                    continue
                kind = item.get("type")
                refused = bool(item.get("refused", False))
                conf_raw = item.get("confidence")
                conf = float(conf_raw) if conf_raw is not None else None
                if kind == "noul":
                    prob_raw = item.get("probability")
                    prob = float(prob_raw) if prob_raw is not None else None
                    answers[str(key)] = NoulAnswer(
                        probability=prob, confidence=conf, refused=refused
                    )
                elif kind == "choice":
                    choice_raw = item.get("choice")
                    choice = str(choice_raw) if choice_raw is not None else None
                    probs_raw = item.get("probabilities") or {}
                    probs = {str(k): float(v) for k, v in probs_raw.items()}
                    answers[str(key)] = ChoiceAnswer(
                        choice=choice,
                        probabilities=probs,
                        confidence=conf,
                        refused=refused,
                    )
                elif kind == "score":
                    score_raw = item.get("score")
                    score = float(score_raw) if score_raw is not None else None
                    probs_raw = item.get("probabilities") or {}
                    score_probs = {int(k): float(v) for k, v in probs_raw.items()}
                    answers[str(key)] = ScoreAnswer(
                        score=score,
                        probabilities=score_probs,
                        confidence=conf,
                        refused=refused,
                    )
        cost_raw = data.get("cost_usd")
        return cls(
            answers=answers,
            model=str(data.get("model") or ""),
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            cost_usd=float(cost_raw) if cost_raw is not None else None,
        )

    @classmethod
    def from_json(
        cls, raw: str | bytes | Mapping[str, Any] | None
    ) -> DecisionResult | None:
        """Parse a checkpointed :class:`DecisionResult`; ``None`` on empty/invalid input."""
        if raw is None:
            return None
        if isinstance(raw, Mapping):
            return cls.from_dict(raw)
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            loaded = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(loaded, Mapping):
            return None
        return cls.from_dict(loaded)


# ---------------------------------------------------------------------------
# Errors & wire-protocol helpers
# ---------------------------------------------------------------------------


class DecisionAPIError(RuntimeError):
    """Error raised by a Decision Provider HTTP/wire call."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = retryable


def encode_wire_keys(
    questions: Mapping[str, DecisionQuestion],
) -> tuple[dict[str, DecisionQuestion], dict[str, str]]:
    """Map arbitrary caller question keys to safe ASCII IDs ``q_0..q_{n-1}``.

    Returns ``(wire_questions, wire_to_user)`` so responses can be mapped back
    to verbatim caller keys regardless of spaces, punctuation, or non-ASCII
    characters.
    """
    wire_questions: dict[str, DecisionQuestion] = {}
    wire_to_user: dict[str, str] = {}
    for idx, (user_key, question) in enumerate(questions.items()):
        wire_key = f"q_{idx}"
        wire_questions[wire_key] = question
        wire_to_user[wire_key] = user_key
    return wire_questions, wire_to_user


def parse_retry_after(value: str | None) -> float | None:
    """Parse an HTTP ``Retry-After`` header (seconds or HTTP-date) into seconds."""
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return max(0.0, float(cleaned))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(cleaned)
    except (TypeError, ValueError):
        return None
    return max(0.0, dt.timestamp() - time.time())


# ---------------------------------------------------------------------------
# Protocol ABC & deterministic offline dummy
# ---------------------------------------------------------------------------


class DecisionClient(ABC):
    """Minimal protocol for Decision Providers (System One and compatible APIs)."""

    max_questions_per_call: int = 64
    requires_decision_model: bool = False

    @abstractmethod
    async def decide(
        self,
        state: str | dict[str, Any],
        questions: dict[str, DecisionQuestion],
        *,
        model: str,
        **kwargs: Any,
    ) -> DecisionResult:
        """Evaluate ``questions`` in parallel against ``state``."""
        raise NotImplementedError


class DummyDecisionClient(DecisionClient):
    """Deterministic, zero-network Decision Provider for tests and dry runs."""

    def __init__(
        self,
        *,
        noul_probability: float = 0.5,
        default_choice: str | None = None,
        max_questions_per_call: int = 64,
    ) -> None:
        self._noul_probability = float(noul_probability)
        self._default_choice = default_choice
        self.max_questions_per_call = max(1, int(max_questions_per_call))

    async def decide(
        self,
        state: str | dict[str, Any],
        questions: dict[str, DecisionQuestion],
        *,
        model: str = "dummy",
        **kwargs: Any,
    ) -> DecisionResult:
        del state, kwargs
        answers: dict[str, DecisionAnswer] = {}
        for key, question in questions.items():
            if isinstance(question, NoulQuestion):
                prob = self._noul_probability
                answers[key] = NoulAnswer(
                    probability=prob,
                    confidence=abs(2.0 * prob - 1.0),
                )
            elif isinstance(question, ChoiceQuestion):
                options = list(question.criteria.keys())
                if not options:
                    answers[key] = ChoiceAnswer(
                        choice=None, probabilities={}, confidence=None
                    )
                else:
                    chosen = (
                        self._default_choice
                        if self._default_choice in question.criteria
                        else options[0]
                    )
                    probs = {opt: (1.0 if opt == chosen else 0.0) for opt in options}
                    answers[key] = ChoiceAnswer(
                        choice=chosen, probabilities=probs, confidence=1.0
                    )
            elif isinstance(question, ScoreQuestion):
                n_levels = len(question.criteria)
                if n_levels == 0:
                    answers[key] = ScoreAnswer(
                        score=None, probabilities={}, confidence=None
                    )
                else:
                    uniform = 1.0 / n_levels
                    probs_score = {i: uniform for i in range(n_levels)}
                    answers[key] = ScoreAnswer(
                        score=(n_levels - 1) / 2.0,
                        probabilities=probs_score,
                        confidence=0.0,
                    )
            else:
                raise TypeError(f"Unsupported question type: {type(question)!r}")
        return DecisionResult(answers=answers, model=model)


__all__ = [
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecisionAPIError",
    "DecisionAnswer",
    "DecisionClient",
    "DecisionQuestion",
    "DecisionResult",
    "DummyDecisionClient",
    "NoulAnswer",
    "NoulQuestion",
    "ScoreAnswer",
    "ScoreQuestion",
    "encode_wire_keys",
    "parse_retry_after",
]
