"""Poll: seed a population, expand into personas, and survey them.

Ported and modified from openai/GABRIEL's ``tasks/poll.py`` (Apache-2.0).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ...engine.batch import batch_execute
from ...templates import load_prompt
from ...utils.parsing import safe_json
from .. import _common
from ..base import Task


@dataclass
class PollConfig:
    population_description: str | None = None
    questions: list[str] = field(default_factory=list)
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
        qs: list[str] = (
            [questions]
            if isinstance(questions, str)
            else list(questions or self.cfg.questions)
        )
        desc = population_description or self.cfg.population_description
        effective_seed = self.cfg.random_seed if random_seed is None else random_seed

        # Input signature covers everything that shapes the run: the seed
        # source (input frame, or population description), the questions,
        # the per-stage models, and the sampling seed.  A completed run with
        # the same signature is reloaded instead of re-executed (zero calls).
        seed_key = (
            (df, column_name) if df is not None else (desc, self.cfg.num_personas)
        )
        signature = _common.compute_input_signature(
            seed_key,
            qs,
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

        for i in range(0, len(qs), self.cfg.n_questions_per_run):
            batch = qs[i : i + self.cfg.n_questions_per_run]
            ans_prompts = [
                self._poll_tpl.render(
                    persona=persona_map.get(seed_ids[j], ""),
                    questions="\n".join(f"- {q}" for q in batch),
                )
                for j in range(len(seeds))
            ]
            ans_ids = [
                f"{seed_ids[j]}_q{_common.hash_identifier('|'.join(batch))}"
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
            for q in batch:
                if q not in out.columns:
                    out[q] = pd.Series([None] * len(seeds), dtype=object)
            for j, s in enumerate(seeds):
                obj = safe_json(ans_map.get(ans_ids[j]))
                if isinstance(obj, dict):
                    for q in batch:
                        sample_key = f"{effective_seed}:{j}:{seed_ids[j]}:{q}"
                        out.at[j, q] = self._sample(obj.get(q), sample_key=sample_key)

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
