"""Poll: seed a population, expand into personas, and survey them.

Ported and modified from openai/GABRIEL's ``tasks/poll.py`` (Apache-2.0).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ...engine.batch import batch_execute
from ...templates import load_prompt
from ...utils.parsing import safe_json
from .. import _common
from ..base import Task

_LEADING_BULLET_RE = re.compile(r"^(?:[-*•]+|\d+[.)])\s+")
_BRACKET_TAG_RE = re.compile(r"^\[([^\[\]]{1,40})\]\s*[:\-–—.]?\s*(.*)$")
_CODE_PREFIX_RE = re.compile(
    r"^([A-Za-z]{0,8}[Qq]\d+[A-Za-z0-9_.-]*)\s*[:.)\-–—]\s*(.+)$"
)
_TRAILING_PARENS_RE = re.compile(r"\s*\([^()]*\)\s*[?:.]*\s*$")
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


def normalize_poll_questions(
    questions: str | Sequence[str] | Mapping[str, str] | None,
) -> list[str] | dict[str, str]:
    """Normalize caller-supplied ``questions`` without losing mapping values."""
    if questions is None:
        return []
    if isinstance(questions, Mapping):
        return {str(k): str(v) for k, v in questions.items()}
    if isinstance(questions, str):
        return [questions]
    return [str(q) for q in questions]


@dataclass
class PollConfig:
    population_description: str | None = None
    questions: list[str] | dict[str, str] = field(default_factory=list)
    save_dir: str = "poll"
    file_name: str = "poll_results.csv"
    seed_model: str = "gpt-5.5"
    persona_model: str = "gpt-5.5"
    poll_model: str = "gpt-5.5"
    n_parallels: int = 50
    num_personas: int = 1000
    entities_per_generation: int = 50
    n_questions_per_run: int = 8
    random_seed: int | None = None


class Poll(Task):
    def __init__(self, cfg: PollConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._persona_tpl = load_prompt("persona_prompt.jinja2")
        self._poll_tpl = load_prompt("poll_prompt.jinja2")

    async def run(
        self,
        df: pd.DataFrame | None = None,
        column_name: str | None = None,
        *,
        population_description: str | None = None,
        questions: Any | None = None,
        random_seed: int | None = None,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        raw_questions = questions if questions is not None else self.cfg.questions
        normalized_questions = normalize_poll_questions(raw_questions)
        use_question_ids = isinstance(normalized_questions, Mapping)
        question_items: list[tuple[str, str]] = (
            list(normalized_questions.items())
            if isinstance(normalized_questions, Mapping)
            else [(q, q) for q in normalized_questions]
        )
        qs = [col for col, _ in question_items]
        desc = population_description or self.cfg.population_description
        effective_seed = self.cfg.random_seed if random_seed is None else random_seed

        # Input signature covers everything that shapes the run: the seed
        # source (input frame, or population description), the questions
        # (including full text when keyed by ID), the per-stage models, and
        # the sampling seed.  A completed run with the same signature is
        # reloaded instead of re-executed (zero calls).
        seed_key = (
            (df, column_name) if df is not None else (desc, self.cfg.num_personas)
        )
        questions_sig: Any = dict(question_items) if use_question_ids else qs
        signature = _common.compute_input_signature(
            seed_key,
            questions_sig,
            self.cfg.seed_model,
            self.cfg.persona_model,
            self.cfg.poll_model,
            self.cfg.n_questions_per_run,
            self.cfg.entities_per_generation,
            self.cfg.file_name,
            effective_seed,
        )
        loaded = self._load_completed(signature, qs, reset_files=reset_files)
        if loaded is not None:
            return loaded

        if df is not None:
            seeds = [str(v) for v in df[column_name].tolist()]
        else:
            from ..seed import Seed, SeedConfig

            seed_cfg = SeedConfig(
                instructions=self._seed_instructions(desc),
                save_dir=os.path.join(self.cfg.save_dir, "seed"),
                model=self.cfg.seed_model,
                num_entities=self.cfg.num_personas,
                entities_per_generation=self.cfg.entities_per_generation,
            )
            seed_df = await Seed(seed_cfg, client=self._client).run(
                reset_files=reset_files, **kwargs
            )
            seeds = seed_df["entity"].tolist()

        seed_ids = [_common.hash_identifier(s) for s in seeds]
        persona_prompts = [
            self._persona_tpl.render(seed=s, population_description=desc) for s in seeds
        ]
        persona_resp = await batch_execute(
            persona_prompts,
            seed_ids,
            client=self._get_client(),
            model=self.cfg.persona_model,
            json_mode=False,
            n_parallels=self.cfg.n_parallels,
            save_path=os.path.join(self.cfg.save_dir, "personas_raw.csv"),
            reset_files=reset_files,
            **kwargs,
        )
        persona_map = dict(zip(persona_resp["Identifier"], persona_resp["Response"]))

        out = pd.DataFrame(
            {"seed": seeds, "persona": [persona_map.get(i, "") for i in seed_ids]}
        )

        for i in range(0, len(question_items), self.cfg.n_questions_per_run):
            batch_items = question_items[i : i + self.cfg.n_questions_per_run]
            batch_cols = [col for col, _ in batch_items]
            rendered_questions = (
                "\n".join(f"- [{qid}] {qtext}" for qid, qtext in batch_items)
                if use_question_ids
                else "\n".join(f"- {qtext}" for _, qtext in batch_items)
            )
            batch_hash_source = (
                "|".join(f"{qid}\x1f{qtext}" for qid, qtext in batch_items)
                if use_question_ids
                else "|".join(batch_cols)
            )
            ans_prompts = [
                self._poll_tpl.render(
                    persona=persona_map.get(seed_ids[j], ""),
                    questions=rendered_questions,
                    use_question_ids=use_question_ids,
                )
                for j in range(len(seeds))
            ]
            ans_ids = [
                f"{seed_ids[j]}_q{_common.hash_identifier(batch_hash_source)}"
                for j in range(len(seeds))
            ]
            ans_resp = await batch_execute(
                ans_prompts,
                ans_ids,
                client=self._get_client(),
                model=self.cfg.poll_model,
                json_mode=True,
                n_parallels=self.cfg.n_parallels,
                save_path=os.path.join(self.cfg.save_dir, "answers_raw.csv"),
                reset_files=reset_files,
                **kwargs,
            )
            ans_map = dict(zip(ans_resp["Identifier"], ans_resp["Response"]))
            for q in batch_cols:
                if q not in out.columns:
                    out[q] = pd.Series([None] * len(seeds), dtype=object)
            for j, s in enumerate(seeds):
                obj = safe_json(ans_map.get(ans_ids[j]))
                if isinstance(obj, dict):
                    resolved = self._resolve_batch_answers(
                        obj, batch_items, use_question_ids=use_question_ids
                    )
                    for q in batch_cols:
                        if q in resolved:
                            sample_key = f"{effective_seed}:{j}:{seed_ids[j]}:{q}"
                            out.at[j, q] = self._sample(
                                resolved[q], sample_key=sample_key
                            )

        self._save(out, qs)
        _common.save_task_state(
            self.cfg.save_dir,
            "poll",
            input_signature=signature,
            result_artifacts=[self.cfg.file_name],
        )
        return out

    def _load_completed(
        self,
        signature: str,
        question_cols: list[str],
        *,
        reset_files: bool = False,
    ) -> pd.DataFrame | None:
        """Return the saved result when inputs match and it is complete."""
        out = _common.load_completed_csv(
            self.cfg.save_dir,
            "poll",
            signature=signature,
            file_name=self.cfg.file_name,
            reset_files=reset_files,
        )
        if out is None:
            return None
        for col in question_cols:
            if col in out.columns:
                out[col] = out[col].map(self._restore_structured_cell)
        path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        print(f"[Poll] Found completed result in {path}; loading {len(out)} responses.")
        return out

    @staticmethod
    def _serialize_structured_cell(value: Any) -> Any:
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return value

    @staticmethod
    def _restore_structured_cell(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        return parsed if isinstance(parsed, (list, dict)) else value

    @staticmethod
    def _norm_basic(text: str) -> str:
        cleaned = text.translate(_QUOTE_TRANSLATION).strip()
        cleaned = _LEADING_BULLET_RE.sub("", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(":")
        return cleaned.casefold()

    @classmethod
    def _extract_leading_tag(cls, text: str) -> tuple[str | None, str]:
        cleaned = text.translate(_QUOTE_TRANSLATION).strip()
        cleaned = _LEADING_BULLET_RE.sub("", cleaned).strip()
        m_bracket = _BRACKET_TAG_RE.match(cleaned)
        if m_bracket:
            return m_bracket.group(1).strip().casefold(), m_bracket.group(2).strip()
        m_code = _CODE_PREFIX_RE.match(cleaned)
        if m_code:
            return m_code.group(1).strip().casefold(), m_code.group(2).strip()
        return None, cleaned

    @staticmethod
    def _strip_trailing_parens(text: str) -> str:
        cur = text.strip()
        while True:
            stripped = _TRAILING_PARENS_RE.sub("", cur).strip()
            if stripped == cur or len(stripped) < 3:
                break
            cur = stripped
        return cur

    @classmethod
    def _core_stem(cls, text: str) -> str:
        _, remainder = cls._extract_leading_tag(text)
        without_parens = cls._strip_trailing_parens(remainder)
        return cls._norm_basic(without_parens).rstrip("?.!:")

    @classmethod
    def _resolve_batch_answers(
        cls,
        obj: dict[Any, Any],
        batch_items: Sequence[tuple[str, str]],
        *,
        use_question_ids: bool = False,
    ) -> dict[str, Any]:
        """Match JSON response keys to ``batch_items`` without stealing or guessing."""
        resolved: dict[str, Any] = {}
        unclaimed_q: list[int] = []
        claimed_keys: set[Any] = set()

        # Tier 0a: exact column/ID key match across the whole batch first
        for idx, (col_key, _) in enumerate(batch_items):
            if col_key in obj and col_key not in claimed_keys:
                resolved[col_key] = obj[col_key]
                claimed_keys.add(col_key)
            else:
                unclaimed_q.append(idx)

        # Tier 0b: exact full question_text match when questions were supplied as a mapping
        if use_question_ids and unclaimed_q:
            still_unclaimed: list[int] = []
            text_counts = Counter(batch_items[i][1] for i in unclaimed_q)
            for idx in unclaimed_q:
                col_key, qtext = batch_items[idx]
                if (
                    text_counts[qtext] == 1
                    and qtext in obj
                    and qtext not in claimed_keys
                ):
                    resolved[col_key] = obj[qtext]
                    claimed_keys.add(qtext)
                else:
                    still_unclaimed.append(idx)
            unclaimed_q = still_unclaimed

        unclaimed_k: list[Any] = [k for k in obj if k not in claimed_keys]
        if not unclaimed_q or not unclaimed_k:
            return resolved

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
                    col_key, _ = batch_items[qi]
                    resolved[col_key] = obj[unclaimed_k[ki]]
                    matched_q.add(qi)
                    matched_k.add(ki)
            if matched_q:
                unclaimed_q = [qi for qi in unclaimed_q if qi not in matched_q]
                unclaimed_k = [
                    rk for ki, rk in enumerate(unclaimed_k) if ki not in matched_k
                ]

        # Tier 1: basic normalized match (leading bullet, whitespace, smart quotes, case)
        def tier_basic(qi: int, rk: str) -> bool:
            col_key, qtext = batch_items[qi]
            nk = cls._norm_basic(rk)
            if not nk:
                return False
            targets = {cls._norm_basic(col_key), cls._norm_basic(qtext)}
            if use_question_ids:
                targets.add(cls._norm_basic(f"[{col_key}] {qtext}"))
            return nk in targets

        # Tier 2: trailing parenthetical hints stripped while preserving any leading [ID] tag
        def tier_no_trailing_parens(qi: int, rk: str) -> bool:
            col_key, qtext = batch_items[qi]
            nk = cls._norm_basic(cls._strip_trailing_parens(rk)).rstrip("?.!:")
            nq = cls._norm_basic(cls._strip_trailing_parens(qtext)).rstrip("?.!:")
            if not nk or not nq:
                return False
            if nk == nq:
                return True
            if use_question_ids:
                combo = cls._norm_basic(
                    cls._strip_trailing_parens(f"[{col_key}] {qtext}")
                ).rstrip("?.!:")
                return nk == combo
            return False

        # Tier 3: explicit question ID or leading bracket/code tag match
        def tier_tag(qi: int, rk: str) -> bool:
            col_key, qtext = batch_items[qi]
            rk_tag, _ = cls._extract_leading_tag(rk)
            rk_bare = cls._norm_basic(rk).strip("[]")
            expected_tags: set[str] = set()
            if use_question_ids:
                cid = cls._norm_basic(col_key).strip("[]")
                if cid:
                    expected_tags.add(cid)
            q_tag, _ = cls._extract_leading_tag(qtext)
            if q_tag:
                expected_tags.add(q_tag)
            if not expected_tags:
                return False
            return rk_bare in expected_tags or (
                rk_tag is not None and rk_tag in expected_tags
            )

        # Tier 4: core stem match (both leading [ID] tag and trailing parentheticals stripped)
        def tier_stem(qi: int, rk: str) -> bool:
            _, qtext = batch_items[qi]
            sk = cls._core_stem(rk)
            sq = cls._core_stem(qtext)
            return len(sk) >= 4 and sk == sq

        # Tier 5: unambiguous long stem prefix match (>= 15 chars)
        def tier_prefix(qi: int, rk: str) -> bool:
            _, qtext = batch_items[qi]
            sk = cls._core_stem(rk)
            sq = cls._core_stem(qtext)
            if min(len(sk), len(sq)) < 15:
                return False
            return sq.startswith(sk) or sk.startswith(sq)

        for tier_fn in (
            tier_basic,
            tier_no_trailing_parens,
            tier_tag,
            tier_stem,
            tier_prefix,
        ):
            run_tier(tier_fn)

        return resolved

    @staticmethod
    def _stable_uniform(key: str) -> float:
        digest = hashlib.sha1(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    @staticmethod
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

    @classmethod
    def _sample(cls, distribution: Any, *, sample_key: str = "") -> Any:
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
                w = cls._coerce_weight(raw_weight)
                if w is not None:
                    valid_pairs.append((ans, w))
            elif isinstance(entry, dict) and ("answer" in entry or "option" in entry):
                saw_pair_shape = True
                ans = entry.get("answer", entry.get("option"))
                fallback_candidates.append(ans)
                raw_weight = entry.get(
                    "probability", entry.get("prob", entry.get("weight"))
                )
                w = cls._coerce_weight(raw_weight)
                if w is not None:
                    valid_pairs.append((ans, w))

        u = cls._stable_uniform(sample_key)
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
            candidates = [ans for ans, _ in valid_pairs]
            return candidates[min(int(u * len(candidates)), len(candidates) - 1)]

        candidates = fallback_candidates if saw_pair_shape else list(distribution)
        if not candidates:
            return None
        return candidates[min(int(u * len(candidates)), len(candidates) - 1)]

    @staticmethod
    def _seed_instructions(desc: str | None) -> str:
        return (
            f"Generate representative people matching this population: {desc}"
            if desc
            else "Generate a representative set of people."
        )

    def _save(self, out: pd.DataFrame, question_cols: list[str]) -> None:
        out_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        to_write = out.copy()
        for col in question_cols:
            if col in to_write.columns:
                to_write[col] = to_write[col].map(self._serialize_structured_cell)
        try:
            to_write.to_csv(out_path, index=False)
        except OSError:
            pass
