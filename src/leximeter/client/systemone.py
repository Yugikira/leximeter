"""System One protocol Decision Provider client (``POST /v1/systemone``).

Speaks the open System One wire protocol implemented by TypeSafe Jev,
OpenRouter's Decisions gateway (``/api/v1/systemone`` and
``/api/alpha/decisions``), AMD Radeon Cloud Token Factory (``d3``,
``d3-flash``, ``Clef``), and self-hosted vLLM / SGLang servers.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionAPIError,
    DecisionClient,
    DecisionQuestion,
    DecisionResult,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    encode_wire_keys,
    parse_retry_after,
)

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_ENDPOINT_PATH = "/v1/systemone"
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _segment_overlap(base_segs: list[str], path_segs: list[str]) -> int:
    """Return the longest suffix-prefix segment overlap between ``base_segs`` and ``path_segs``."""
    max_k = min(len(base_segs), len(path_segs))
    for k in range(max_k, 0, -1):
        if base_segs[-k:] == path_segs[:k]:
            return k
    return 0


def _join_endpoint_url(base_url: str, endpoint_path: str) -> str:
    """Combine ``base_url`` and ``endpoint_path`` without duplicating path prefixes."""
    parsed = urlsplit(base_url.strip())
    ep_parsed = urlsplit(endpoint_path.strip())
    base_segs = [seg for seg in parsed.path.split("/") if seg]
    path_segs = [seg for seg in ep_parsed.path.split("/") if seg]

    if path_segs:
        overlap = _segment_overlap(base_segs, path_segs)
        if (
            overlap == 0
            and base_segs
            and base_segs[-1] == "v1"
            and (
                path_segs[0] == "alpha"
                or _segment_overlap(base_segs[:-1], path_segs) > 0
            )
        ):
            base_segs = base_segs[:-1]
            overlap = _segment_overlap(base_segs, path_segs)
        joined_segs = [*base_segs, *path_segs[overlap:]]
    else:
        joined_segs = base_segs

    joined_path = "/" + "/".join(joined_segs)
    query = ep_parsed.query or parsed.query
    return urlunsplit(
        (parsed.scheme, parsed.netloc, joined_path, query, parsed.fragment)
    )


def _serialize_question(
    question: DecisionQuestion, *, send_noul_criteria: bool
) -> dict[str, Any]:
    if isinstance(question, NoulQuestion):
        payload: dict[str, Any] = {
            "type": "noul",
            "instructions": question.instructions,
        }
        if send_noul_criteria and (
            question.true_criteria is not None or question.false_criteria is not None
        ):
            criteria: dict[str, str] = {}
            if question.true_criteria is not None:
                criteria["true"] = question.true_criteria
            if question.false_criteria is not None:
                criteria["false"] = question.false_criteria
            payload["criteria"] = criteria
        return payload
    if isinstance(question, ChoiceQuestion):
        return {
            "type": "choice",
            "instructions": question.instructions,
            "criteria": dict(question.criteria),
        }
    if isinstance(question, ScoreQuestion):
        return {
            "type": "score",
            "instructions": question.instructions,
            "criteria": list(question.criteria),
        }
    raise TypeError(f"Unsupported question type: {type(question)!r}")


def _refused_answer(question: DecisionQuestion) -> DecisionAnswer:
    if isinstance(question, NoulQuestion):
        return NoulAnswer(probability=None, confidence=None, refused=True)
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            choice=None, probabilities={}, confidence=None, refused=True
        )
    if isinstance(question, ScoreQuestion):
        return ScoreAnswer(score=None, probabilities={}, confidence=None, refused=True)
    raise TypeError(f"Unsupported question type: {type(question)!r}")


def _parse_answer(raw: Mapping[str, Any], question: DecisionQuestion) -> DecisionAnswer:
    kind = str(raw.get("type") or "").strip().lower()
    if kind == "refusal":
        return _refused_answer(question)

    conf_raw = raw.get("confidence")
    confidence = float(conf_raw) if conf_raw is not None else None

    if kind == "noul" or (not kind and isinstance(question, NoulQuestion)):
        prob_raw = raw.get("noul")
        prob = float(prob_raw) if prob_raw is not None else None
        return NoulAnswer(probability=prob, confidence=confidence, refused=False)

    if kind == "choice" or (not kind and isinstance(question, ChoiceQuestion)):
        choice_raw = raw.get("choice")
        choice = str(choice_raw) if choice_raw is not None else None
        probs_raw = raw.get("probabilities") or {}
        probs = {str(k): float(v) for k, v in probs_raw.items()}
        return ChoiceAnswer(
            choice=choice, probabilities=probs, confidence=confidence, refused=False
        )

    if kind == "score" or (not kind and isinstance(question, ScoreQuestion)):
        score_raw = raw.get("score")
        score = float(score_raw) if score_raw is not None else None
        probs_raw = raw.get("probabilities") or {}
        score_probs = {int(k): float(v) for k, v in probs_raw.items()}
        return ScoreAnswer(
            score=score, probabilities=score_probs, confidence=confidence, refused=False
        )

    raise DecisionAPIError(f"Unknown System One answer type: {kind!r}")


class SystemOneClient(DecisionClient):
    """HTTP client for the System One Decision Provider wire protocol."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        endpoint_path: str = DEFAULT_ENDPOINT_PATH,
        max_questions_per_call: int = 64,
        send_noul_criteria: bool = True,
        long_context: bool = False,
        timeout: float = 60.0,
        max_retries: int = 0,
        backoff_base: float = 0.5,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_base = (
            base_url or os.getenv("LEXIMETER_DECISION_BASE_URL") or DEFAULT_BASE_URL
        )
        self._url = _join_endpoint_url(resolved_base, endpoint_path)
        self._api_key = (
            api_key if api_key is not None else os.getenv("LEXIMETER_DECISION_API_KEY")
        )
        self.max_questions_per_call = max(1, int(max_questions_per_call))
        self.send_noul_criteria = send_noul_criteria
        self._long_context = long_context
        self._timeout = timeout
        self._max_retries = max(0, int(max_retries))
        self._backoff_base = backoff_base
        self._extra_headers = dict(headers) if headers else {}
        self._transport = transport
        self._external_http_client = http_client
        self._http_client: httpx.AsyncClient | None = None
        self._http_client_loop: asyncio.AbstractEventLoop | None = None

    @property
    def url(self) -> str:
        """Resolved endpoint URL for System One POST requests."""
        return self._url

    def _build_headers(self) -> dict[str, str]:
        hdrs: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            hdrs["Authorization"] = f"Bearer {self._api_key}"
        if self._long_context:
            hdrs["TypeSafe-Danger-Zone-Long-Context"] = "true"
        hdrs.update(self._extra_headers)
        return hdrs

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._external_http_client is not None:
            return self._external_http_client
        loop = asyncio.get_running_loop()
        if (
            self._http_client is None
            or self._http_client.is_closed
            or self._http_client_loop is not loop
        ):
            self._http_client = httpx.AsyncClient(
                transport=self._transport,
                timeout=self._timeout,
            )
            self._http_client_loop = loop
        return self._http_client

    async def aclose(self) -> None:
        """Close the internally managed ``httpx.AsyncClient`` if open."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
        self._http_client = None
        self._http_client_loop = None

    async def decide(
        self,
        state: str | dict[str, Any],
        questions: dict[str, DecisionQuestion],
        *,
        model: str,
        **kwargs: Any,
    ) -> DecisionResult:
        if not questions:
            return DecisionResult(answers={}, model=model)

        wire_questions, wire_to_user = encode_wire_keys(questions)
        payload: dict[str, Any] = {
            "model": model,
            "state": state,
            "questions": {
                wire_key: _serialize_question(
                    question, send_noul_criteria=self.send_noul_criteria
                )
                for wire_key, question in wire_questions.items()
            },
        }
        request_timeout = kwargs.get("timeout", self._timeout)
        headers = self._build_headers()
        client = self._get_http_client()

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(
                    self._url,
                    json=payload,
                    headers=headers,
                    timeout=request_timeout,
                )
            except httpx.HTTPError as exc:
                last_exc = DecisionAPIError(
                    f"System One transport error: {exc}", retryable=True
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(min(self._backoff_base * (2**attempt), 30.0))
                    continue
                raise last_exc from exc

            if response.status_code == 200:
                return self._parse_response(
                    response.json(),
                    wire_to_user=wire_to_user,
                    questions=questions,
                    fallback_model=model,
                )

            retry_after = parse_retry_after(response.headers.get("Retry-After"))
            retryable = response.status_code in _RETRYABLE_STATUS_CODES
            err = DecisionAPIError(
                f"System One HTTP {response.status_code}: {response.text[:300]}",
                status_code=response.status_code,
                retry_after=retry_after,
                retryable=retryable,
            )
            last_exc = err
            if retryable and attempt < self._max_retries:
                delay = (
                    retry_after
                    if retry_after is not None
                    else min(self._backoff_base * (2**attempt), 30.0)
                )
                await asyncio.sleep(delay)
                continue
            raise err

        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _parse_response(
        data: Mapping[str, Any],
        *,
        wire_to_user: Mapping[str, str],
        questions: Mapping[str, DecisionQuestion],
        fallback_model: str,
    ) -> DecisionResult:
        raw_answers = data.get("answers")
        if not isinstance(raw_answers, Mapping):
            raise DecisionAPIError("System One response missing 'answers' object")

        answers: dict[str, DecisionAnswer] = {}
        for wire_key, raw_ans in raw_answers.items():
            user_key = wire_to_user.get(str(wire_key))
            if user_key is None or not isinstance(raw_ans, Mapping):
                continue
            answers[user_key] = _parse_answer(raw_ans, questions[user_key])

        usage = data.get("usage")
        input_tokens = 0
        output_tokens = 0
        cost_usd: float | None = None
        if isinstance(usage, Mapping):
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            if usage.get("cost") is not None:
                cost_usd = float(usage["cost"])

        resp_model = str(data.get("model") or fallback_model)
        return DecisionResult(
            answers=answers,
            model=resp_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )


__all__ = ["DEFAULT_BASE_URL", "DEFAULT_ENDPOINT_PATH", "SystemOneClient"]
