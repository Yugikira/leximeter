"""Filter task: high-throughput boolean screening by natural-language condition.

Ported and modified from openai/GABRIEL's ``tasks/filter.py`` (Apache-2.0).
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..client.decision import (
    DecisionClient,
    DecisionQuestion,
    DecisionResult,
    NoulAnswer,
    NoulQuestion,
)
from ..templates import load_prompt
from ..utils.parsing import safe_json
from . import _common
from .base import Task


def _build_filter_state(
    condition: str, additional_instructions: str | None
) -> dict[str, Any]:
    state: dict[str, Any] = {"condition": condition}
    if additional_instructions:
        state["additional_instructions"] = additional_instructions
    return state


def _noul_instruction(item: str) -> str:
    return f"Does the following item meet the screening condition?\nItem: {item}"


@dataclass
class FilterConfig:
    condition: str
    save_dir: str = "filter"
    file_name: str = "filter_responses.csv"
    model: str = "gpt-5.4-nano"
    decision_model: str | None = None
    n_parallels: int = 50
    entities_per_call: int = 150
    shuffle: bool = True
    random_seed: int = 42
    n_runs: int = 1
    threshold: float = 0.5
    additional_instructions: str | None = None

    def __post_init__(self) -> None:
        if self.additional_instructions is not None:
            cleaned = str(self.additional_instructions).strip()
            self.additional_instructions = cleaned or None


class Filter(Task):
    """Keep items whose natural-language ``condition`` holds."""

    _prefer_explicit_llm_client = True

    def __init__(self, cfg: FilterConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("filter_prompt.jinja2")

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        decision_client = self._get_decision_client()
        if decision_client is not None:
            return await self._run_with_decision_client(
                df,
                column_name,
                decision_client=decision_client,
                reset_files=reset_files,
                **kwargs,
            )

        # 1. unique entities, optionally shuffled with a fixed seed
        values = [str(v) for v in df[column_name].tolist()]
        unique = list(dict.fromkeys(values))
        if self.cfg.shuffle:
            rnd = random.Random(self.cfg.random_seed)
            rnd.shuffle(unique)

        # 2. batch into groups
        batches = [
            unique[i : i + self.cfg.entities_per_call]
            for i in range(0, len(unique), self.cfg.entities_per_call)
        ]

        # 3. jobs: batch x run
        n_runs = max(1, int(self.cfg.n_runs))
        prompts: list[str] = []
        ids: list[str] = []
        meta: list[tuple[int, int]] = []
        for batch_idx, batch in enumerate(batches):
            batch_id = _common.hash_identifier("\x1f".join(sorted(batch)))
            prompt = self._template.render(
                condition=self.cfg.condition,
                entities="\n".join(batch),
                additional_instructions=self.cfg.additional_instructions,
            )
            for run_idx in range(1, n_runs + 1):
                prompts.append(prompt)
                ids.append(f"{batch_id}_run{run_idx}")
                meta.append((batch_idx, run_idx))

        # 4. execute
        resp_df = await self._execute(prompts, ids, reset_files=reset_files, **kwargs)
        resp_lookup = dict(zip(resp_df["Identifier"], resp_df["Response"]))

        # 5. tally "meets condition" per entity across runs
        marks: dict[str, int] = {}
        for (batch_idx, _run_idx), ident in zip(meta, ids):
            marked = self._parse_entities(resp_lookup.get(ident))
            for entity in batches[batch_idx]:
                if entity in marked:
                    marks[entity] = marks.get(entity, 0) + 1

        # 6. keep/score per entity, mapped back onto the input frame
        score = {entity: marks.get(entity, 0) / n_runs for entity in unique}
        out = df.copy()
        out["keep"] = [score[str(v)] >= self.cfg.threshold for v in df[column_name]]
        out["score"] = [score[str(v)] for v in df[column_name]]
        self._save(out)
        return out

    def _parse_entities(self, raw: Any) -> set:
        obj = safe_json(raw)
        if not isinstance(obj, dict):
            return set()
        meeting = obj.get("entities meeting condition")
        if not isinstance(meeting, list):
            return set()
        return {str(item) for item in meeting}

    async def _execute(self, prompts, ids, *, reset_files, **kwargs):
        from ..engine.batch import batch_execute

        return await batch_execute(
            prompts,
            ids,
            client=self._get_client(),
            model=self.cfg.model,
            json_mode=True,
            n_parallels=self.cfg.n_parallels,
            save_path=os.path.join(
                self.cfg.save_dir, f"{self._base_name()}_raw_responses.csv"
            ),
            reset_files=reset_files,
            **kwargs,
        )

    def _save(self, out: pd.DataFrame) -> None:
        out_path = os.path.join(self.cfg.save_dir, f"{self._base_name()}_cleaned.csv")
        try:
            out.to_csv(out_path, index=False)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Decision Provider execution path (inverted fan-out)
    # ------------------------------------------------------------------
    async def _run_with_decision_client(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        decision_client: DecisionClient,
        reset_files: bool,
        **kwargs: Any,
    ) -> pd.DataFrame:
        from ..engine.batch import batch_decide

        requested_per_call = int(self.cfg.entities_per_call)
        if requested_per_call < 1:
            raise ValueError("entities_per_call must be an integer >= 1")

        effective_model = (
            _common.effective_decision_model(
                decision_client,
                self.cfg.decision_model,
                fallback=self.cfg.model,
            )
            or self.cfg.model
        )

        provider_cap = max(
            1, int(getattr(decision_client, "max_questions_per_call", 64))
        )
        base_name = self._base_name()
        run_metadata = _common.load_run_metadata(
            self.cfg.save_dir, base_name, reset_files=reset_files
        )
        effective_batch_size = min(requested_per_call, provider_cap)
        saved_batch_size = run_metadata.get("effective_batch_size")
        if (
            not reset_files
            and isinstance(saved_batch_size, int)
            and not isinstance(saved_batch_size, bool)
            and 1 <= saved_batch_size <= effective_batch_size
        ):
            effective_batch_size = saved_batch_size

        _common.update_run_metadata(
            self.cfg.save_dir,
            base_name,
            model=effective_model,
            effective_batch_size=effective_batch_size,
        )

        values = [str(v) for v in df[column_name].tolist()]
        unique = list(dict.fromkeys(values))
        if self.cfg.shuffle:
            rnd = random.Random(self.cfg.random_seed)
            rnd.shuffle(unique)

        batches = [
            unique[i : i + effective_batch_size]
            for i in range(0, len(unique), effective_batch_size)
        ]

        shared_state = _build_filter_state(
            self.cfg.condition, self.cfg.additional_instructions
        )

        n_runs = max(1, int(self.cfg.n_runs))
        states: list[str | dict[str, Any]] = []
        question_sets: list[dict[str, DecisionQuestion]] = []
        ids: list[str] = []
        meta: list[tuple[int, int]] = []
        for batch_idx, batch in enumerate(batches):
            batch_id = _common.hash_identifier(
                json.dumps(sorted(batch), ensure_ascii=False)
            )
            qmap: dict[str, DecisionQuestion] = {
                item: NoulQuestion(instructions=_noul_instruction(item))
                for item in batch
            }
            for run_idx in range(1, n_runs + 1):
                states.append(shared_state)
                question_sets.append(qmap)
                ids.append(f"{batch_id}_run{run_idx}")
                meta.append((batch_idx, run_idx))

        resp_df = await batch_decide(
            states=states,
            question_sets=question_sets,
            identifiers=ids,
            client=decision_client,
            model=effective_model,
            n_parallels=self.cfg.n_parallels,
            save_path=os.path.join(
                self.cfg.save_dir, f"{self._base_name()}_raw_responses.csv"
            ),
            reset_files=reset_files,
            **kwargs,
        )
        resp_lookup = dict(zip(resp_df["Identifier"], resp_df["Response"]))

        probs_by_item: dict[str, list[float]] = {item: [] for item in unique}
        for (batch_idx, _run_idx), ident in zip(meta, ids):
            parsed = DecisionResult.from_json(resp_lookup.get(ident))
            if parsed is None:
                continue
            for item in batches[batch_idx]:
                ans = parsed.answers.get(item)
                if (
                    isinstance(ans, NoulAnswer)
                    and not ans.refused
                    and ans.probability is not None
                ):
                    probs_by_item[item].append(float(ans.probability))

        scores: dict[str, float] = {
            item: (sum(vals) / len(vals) if vals else math.nan)
            for item, vals in probs_by_item.items()
        }

        out = df.copy()
        out["keep"] = pd.Series(
            [
                not math.isnan(scores[str(v)]) and scores[str(v)] >= self.cfg.threshold
                for v in df[column_name]
            ],
            index=out.index,
            dtype=bool,
        )
        out["score"] = pd.Series(
            [scores[str(v)] for v in df[column_name]],
            index=out.index,
            dtype=float,
        )
        self._save(out)
        return out
