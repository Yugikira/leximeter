"""Poll: seed a population, expand into personas, and survey them.

Ported and modified from openai/GABRIEL's ``tasks/poll.py`` (Apache-2.0).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ...engine.batch import batch_execute
from ...templates import load_prompt
from ...utils.parsing import safe_json
from .. import _common
from ..base import Task
from .parsing import (
    PollQuestionInput,
    QuestionPlan,
    resolve_batch_answers,
    restore_cell,
    sample_distribution,
    serialize_cell,
)


@dataclass
class PollConfig:
    population_description: str | None = None
    questions: PollQuestionInput = field(default_factory=list)
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
        questions: PollQuestionInput | None = None,
        random_seed: int | None = None,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        plan = QuestionPlan.from_input(
            questions if questions is not None else self.cfg.questions
        )
        question_cols = plan.columns
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
        signature = _common.compute_input_signature(
            seed_key,
            plan.signature_payload,
            self.cfg.seed_model,
            self.cfg.persona_model,
            self.cfg.poll_model,
            self.cfg.n_questions_per_run,
            self.cfg.entities_per_generation,
            self.cfg.file_name,
            effective_seed,
        )
        loaded = self._load_completed(signature, question_cols, reset_files=reset_files)
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

        for batch in plan.batches(self.cfg.n_questions_per_run):
            rendered_questions = batch.rendered_prompt_block
            batch_hash = _common.hash_identifier(batch.batch_hash_source)
            ans_prompts = [
                self._poll_tpl.render(
                    persona=persona_map.get(seed_ids[j], ""),
                    questions=rendered_questions,
                    use_question_ids=batch.use_ids,
                )
                for j in range(len(seeds))
            ]
            ans_ids = [f"{seed_ids[j]}_q{batch_hash}" for j in range(len(seeds))]
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
            for col in batch.columns:
                if col not in out.columns:
                    out[col] = pd.Series([None] * len(seeds), dtype=object)
            for j in range(len(seeds)):
                obj = safe_json(ans_map.get(ans_ids[j]))
                if isinstance(obj, dict):
                    resolved = resolve_batch_answers(obj, batch)
                    for col in batch.columns:
                        if col in resolved:
                            sample_key = f"{effective_seed}:{j}:{seed_ids[j]}:{col}"
                            out.at[j, col] = self._sample(
                                resolved[col], sample_key=sample_key
                            )

        self._save(out, question_cols)
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
                out[col] = out[col].map(restore_cell)
        path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        print(f"[Poll] Found completed result in {path}; loading {len(out)} responses.")
        return out

    @staticmethod
    def _sample(distribution: Any, *, sample_key: str = "") -> Any:
        return sample_distribution(distribution, sample_key=sample_key)

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
                to_write[col] = to_write[col].map(serialize_cell)
        try:
            to_write.to_csv(out_path, index=False)
        except OSError:
            pass
