"""Question planning, key resolution, sampling, and CSV codecs for ``Poll``."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PollQuestionInput = str | Sequence[str] | Mapping[str, str]

_LEADING_BULLET_RE = re.compile(r"^[-*•]+\s+")
_BRACKET_TAG_RE = re.compile(r"^\[([^\[\]]{1,40})\]\s*[:\-–—.]?\s*(.*)$")
_CODE_PREFIX_RE = re.compile(
    r"^([A-Za-z]{0,8}[Qq]\d+[A-Za-z0-9_.-]*)\s*[:.)\-–—]\s*(.+)$"
)
_NUMBER_PREFIX_RE = re.compile(r"^(\d+)\s*[.)]\s+(.+)$")
_QUOTE_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
)


@dataclass(frozen=True)
class QuestionSpec:
    """Single survey question mapping an output column name to prompt text."""

    column: str
    text: str


@dataclass(frozen=True)
class QuestionPlan:
    """Normalized sequence of survey questions and their keying mode."""

    items: tuple[QuestionSpec, ...]
    use_ids: bool = False

    @classmethod
    def from_input(cls, questions: PollQuestionInput | None) -> QuestionPlan:
        if questions is None:
            return cls((), use_ids=False)
        if isinstance(questions, Mapping):
            return cls(
                tuple(QuestionSpec(str(k), str(v)) for k, v in questions.items()),
                use_ids=True,
            )
        if isinstance(questions, str):
            return cls((QuestionSpec(questions, questions),), use_ids=False)
        return cls(
            tuple(QuestionSpec(str(q), str(q)) for q in questions),
            use_ids=False,
        )

    @property
    def columns(self) -> list[str]:
        return [q.column for q in self.items]

    @property
    def signature_payload(self) -> dict[str, str] | list[str]:
        if self.use_ids:
            return {q.column: q.text for q in self.items}
        return self.columns

    def batches(self, batch_size: int) -> list[QuestionPlan]:
        step = max(1, int(batch_size))
        return [
            QuestionPlan(self.items[i : i + step], use_ids=self.use_ids)
            for i in range(0, len(self.items), step)
        ]

    @property
    def rendered_prompt_block(self) -> str:
        if self.use_ids:
            return "\n".join(f"- [{q.column}] {q.text}" for q in self.items)
        return "\n".join(f"- {q.text}" for q in self.items)

    @property
    def batch_hash_source(self) -> str:
        if self.use_ids:
            return "|".join(f"{q.column}\x1f{q.text}" for q in self.items)
        return "|".join(q.column for q in self.items)


def serialize_cell(value: Any) -> Any:
    """Encode list/dict cells as JSON strings so CSV persistence preserves types."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def restore_cell(value: Any) -> Any:
    """Restore JSON-encoded list/dict cells read back from CSV."""
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except ValueError:
        return value
    return parsed if isinstance(parsed, (list, dict)) else value


def _norm_basic(text: str) -> str:
    cleaned = text.translate(_QUOTE_TRANSLATION).strip()
    cleaned = _LEADING_BULLET_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip("?.!:")
    return cleaned.casefold()


def _extract_leading_tag(text: str) -> tuple[str | None, str]:
    cleaned = text.translate(_QUOTE_TRANSLATION).strip()
    cleaned = _LEADING_BULLET_RE.sub("", cleaned).strip()
    m_bracket = _BRACKET_TAG_RE.match(cleaned)
    if m_bracket:
        return m_bracket.group(1).strip().casefold(), m_bracket.group(2).strip()
    m_code = _CODE_PREFIX_RE.match(cleaned)
    if m_code:
        return m_code.group(1).strip().casefold(), m_code.group(2).strip()
    m_num = _NUMBER_PREFIX_RE.match(cleaned)
    if m_num:
        return m_num.group(1).strip(), m_num.group(2).strip()
    return None, cleaned


def _strip_trailing_parens(text: str) -> str:
    cur = text.strip().rstrip("?.!:")
    while cur.endswith(")"):
        depth = 0
        open_idx = -1
        for i in range(len(cur) - 1, -1, -1):
            ch = cur[i]
            if ch == ")":
                depth += 1
            elif ch == "(":
                depth -= 1
                if depth == 0:
                    open_idx = i
                    break
        if open_idx <= 0:
            break
        prefix = cur[:open_idx].strip().rstrip("?.!:")
        if not prefix:
            break
        cur = prefix
    return cur


def _norm_no_trailing_parens(text: str) -> str:
    return _norm_basic(_strip_trailing_parens(text))


def _core_stem(text: str) -> str:
    _, remainder = _extract_leading_tag(text)
    return _norm_no_trailing_parens(remainder)


def resolve_batch_answers(
    obj: dict[Any, Any],
    batch: QuestionPlan,
) -> dict[str, Any]:
    """Match JSON response keys to ``batch`` questions without stealing or guessing."""
    resolved: dict[str, Any] = {}
    unclaimed_q: list[int] = []
    claimed_keys: set[Any] = set()

    # Tier 0a: exact column / question_id match across the entire batch first
    for idx, q in enumerate(batch.items):
        if q.column in obj and q.column not in claimed_keys:
            resolved[q.column] = obj[q.column]
            claimed_keys.add(q.column)
        else:
            unclaimed_q.append(idx)

    # Tier 0b: exact full question_text match when questions were supplied as a mapping
    if batch.use_ids and unclaimed_q:
        all_text_counts = Counter(q.text for q in batch.items)
        still_unclaimed: list[int] = []
        for idx in unclaimed_q:
            q = batch.items[idx]
            if (
                all_text_counts[q.text] == 1
                and q.text in obj
                and q.text not in claimed_keys
            ):
                resolved[q.column] = obj[q.text]
                claimed_keys.add(q.text)
            else:
                still_unclaimed.append(idx)
        unclaimed_q = still_unclaimed

    unclaimed_k: list[Any] = [k for k in obj if k not in claimed_keys]
    if not unclaimed_q or not unclaimed_k:
        return resolved

    expected_tags: list[set[str]] = []
    all_tags_in_batch: set[str] = set()
    all_no_parens: list[set[str]] = []
    all_stems: list[str] = []
    for q in batch.items:
        tags: set[str] = set()
        if batch.use_ids:
            cid = _norm_basic(q.column).strip("[]")
            if cid:
                tags.add(cid)
        q_tag, _ = _extract_leading_tag(q.text)
        if q_tag:
            tags.add(q_tag)
        expected_tags.append(tags)
        all_tags_in_batch.update(tags)

        no_parens_targets = {_norm_no_trailing_parens(q.text)}
        if batch.use_ids:
            no_parens_targets.add(_norm_no_trailing_parens(f"[{q.column}] {q.text}"))
        no_parens_targets.discard("")
        all_no_parens.append(no_parens_targets)

        all_stems.append(_core_stem(q.text))

    all_stem_counts = Counter(s for s in all_stems if s)
    all_tag_counts: Counter[str] = Counter()
    for tags in expected_tags:
        for t in tags:
            all_tag_counts[t] += 1
    all_no_parens_counts: Counter[str] = Counter()
    for targets in all_no_parens:
        for t in targets:
            all_no_parens_counts[t] += 1

    def _has_tag_conflict(qi: int, rk_tag: str | None) -> bool:
        if rk_tag is None:
            return False
        if expected_tags[qi]:
            return rk_tag not in expected_tags[qi]
        return rk_tag in all_tags_in_batch

    def run_tier(predicate: Callable[[int, str], bool]) -> None:
        nonlocal unclaimed_q, unclaimed_k
        if not unclaimed_q or not unclaimed_k:
            return
        edges: list[tuple[int, int]] = []
        deg_q: Counter[int] = Counter()
        deg_k: Counter[int] = Counter()
        for qi in unclaimed_q:
            for ki, raw_k in enumerate(unclaimed_k):
                if predicate(qi, str(raw_k)):
                    edges.append((qi, ki))
                    deg_q[qi] += 1
                    deg_k[ki] += 1
        matched_q: set[int] = set()
        matched_k: set[int] = set()
        for qi, ki in edges:
            if deg_q[qi] == 1 and deg_k[ki] == 1:
                resolved[batch.items[qi].column] = obj[unclaimed_k[ki]]
                matched_q.add(qi)
                matched_k.add(ki)
        if matched_q:
            unclaimed_q = [qi for qi in unclaimed_q if qi not in matched_q]
            unclaimed_k = [
                rk for ki, rk in enumerate(unclaimed_k) if ki not in matched_k
            ]

    # Tier 1: basic normalized match (leading bullet, whitespace, quotes, trailing punctuation, case)
    def tier_basic(qi: int, rk: str) -> bool:
        q = batch.items[qi]
        nk = _norm_basic(rk)
        if not nk:
            return False
        targets = {_norm_basic(q.column), _norm_basic(q.text)}
        if batch.use_ids:
            targets.add(_norm_basic(f"[{q.column}] {q.text}"))
        return nk in targets

    # Tier 2: trailing parenthetical hints stripped while preserving any leading tag/number
    def tier_no_trailing_parens(qi: int, rk: str) -> bool:
        rk_tag, _ = _extract_leading_tag(rk)
        if _has_tag_conflict(qi, rk_tag):
            return False
        nk = _norm_no_trailing_parens(rk)
        return bool(nk) and nk in all_no_parens[qi] and all_no_parens_counts[nk] == 1

    # Tier 3: bare question ID or leading bracket/code/number tag match
    def tier_tag(qi: int, rk: str) -> bool:
        if not expected_tags[qi]:
            return False
        rk_tag, rk_rem = _extract_leading_tag(rk)
        rk_bare = _norm_basic(rk).strip("[]")
        if rk_bare in expected_tags[qi] and all_tag_counts[rk_bare] == 1:
            return True
        return (
            rk_tag is not None
            and not _norm_basic(rk_rem)
            and rk_tag in expected_tags[qi]
            and all_tag_counts[rk_tag] == 1
        )

    # Tier 4: core stem match (both leading tag and trailing parentheticals stripped)
    def tier_stem(qi: int, rk: str) -> bool:
        rk_tag, _ = _extract_leading_tag(rk)
        if _has_tag_conflict(qi, rk_tag):
            return False
        sk = _core_stem(rk)
        sq = all_stems[qi]
        if not sk or sk != sq:
            return False
        if rk_tag is not None and rk_tag in expected_tags[qi]:
            return all_tag_counts[rk_tag] == 1
        return all_stem_counts[sq] == 1

    for tier_fn in (tier_basic, tier_no_trailing_parens, tier_tag, tier_stem):
        run_tier(tier_fn)

    return resolved


def _stable_uniform(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _coerce_weight(weight: Any) -> float | None:
    if isinstance(weight, bool) or weight is None:
        return None
    try:
        val = float(weight)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val) or val < 0.0:
        return None
    return val


def _pick_uniform(candidates: Sequence[Any], u: float) -> Any:
    return candidates[min(int(u * len(candidates)), len(candidates) - 1)]


def sample_distribution(distribution: Any, *, sample_key: str = "") -> Any:
    """Draw a deterministic probability-weighted answer from ``distribution``."""
    if isinstance(distribution, dict):
        if (
            len(distribution) == 1
            and "options" in distribution
            and isinstance(distribution["options"], list)
        ):
            distribution = distribution["options"]
        else:
            return distribution

    if not isinstance(distribution, list):
        return distribution
    if not distribution:
        return None

    valid_pairs: list[tuple[Any, float]] = []
    saw_pair_shape = False
    fallback_candidates: list[Any] = []
    for entry in distribution:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            saw_pair_shape = True
            ans, raw_weight = entry
            fallback_candidates.append(ans)
            w = _coerce_weight(raw_weight)
            if w is not None:
                valid_pairs.append((ans, w))

    u = _stable_uniform(sample_key)
    if valid_pairs:
        positive = [(ans, w) for ans, w in valid_pairs if w > 0.0]
        if positive:
            total = sum(w for _, w in positive)
            threshold = u * total
            cumulative = 0.0
            for ans, w in positive:
                cumulative += w
                if threshold < cumulative:
                    return ans
            return positive[-1][0]
        return _pick_uniform([ans for ans, _ in valid_pairs], u)

    candidates = fallback_candidates if saw_pair_shape else list(distribution)
    if not candidates:
        return None
    return _pick_uniform(candidates, u)
