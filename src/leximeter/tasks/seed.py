"""Seed task: generate a representative / diverse set of entities."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..templates import load_prompt
from ..utils.parsing import safe_json
from . import _common
from .base import Task


@dataclass
class SeedConfig:
    instructions: str
    save_dir: str = "seed"
    file_name: str = "seed_entities.csv"
    model: str = "gpt-5.5"
    n_parallels: int = 50
    num_entities: int = 1000
    entities_per_generation: int = 50
    existing_entities_cap: int = 100
    deduplicate: bool = False
    deduplicate_sample_seed: int = 42
    max_retries: int = 3


class Seed(Task):
    """Generate ``num_entities`` distinct entities per ``instructions``."""

    def __init__(self, cfg: SeedConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("seed.jinja2")

    async def run(
        self,
        existing_entities: list[str] | None = None,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        signature = _common.compute_input_signature(
            self.cfg.instructions,
            self.cfg.num_entities,
            self.cfg.entities_per_generation,
            existing_entities,
            self.cfg.model,
            self.cfg.file_name,
        )
        loaded = self._load_completed(signature, reset_files=reset_files)
        if loaded is not None:
            return loaded

        entities: list[str] = list(existing_entities or [])
        rng = random.Random(self.cfg.deduplicate_sample_seed)

        # Generation budget: a provider that keeps returning partial results
        # must give up instead of looping forever.  The nominal batch count
        # plus two extra attempts bounds the loop.
        max_batches = max(
            2,
            math.ceil(self.cfg.num_entities / self.cfg.entities_per_generation) + 2,
        )
        batch_idx = 0
        while len(entities) < self.cfg.num_entities and batch_idx < max_batches:
            sample = (
                rng.sample(entities, min(self.cfg.existing_entities_cap, len(entities)))
                if entities
                else None
            )
            prompt = self._template.render(
                instructions=self.cfg.instructions,
                entities_per_generation=self.cfg.entities_per_generation,
                existing_entities="\n".join(sample) if sample else None,
            )
            from ..engine.batch import complete_with_retry

            text = await complete_with_retry(
                self._get_client(),
                prompt,
                model=self.cfg.model,
                json_mode=True,
                max_retries=self.cfg.max_retries,
                timeout=None,
                **kwargs,
            )
            new = self._parse(text)
            if not new:
                break
            entities.extend(new)
            batch_idx += 1

        if len(entities) < self.cfg.num_entities:
            print(
                f"[Seed] Stopped after {batch_idx} generation batches with "
                f"{len(entities)}/{self.cfg.num_entities} entities."
            )
        entities = entities[: self.cfg.num_entities]
        out = pd.DataFrame(
            {
                "entity": entities,
                "entity_id": [f"entity-{i:05d}" for i in range(len(entities))],
            }
        )
        self._save(out)
        # Only a full generation is reusable; partial results must regenerate
        # on the next run instead of silently locking in a shortfall.
        if len(entities) >= self.cfg.num_entities:
            _common.save_task_state(
                self.cfg.save_dir,
                "seed",
                input_signature=signature,
                result_artifacts=[self.cfg.file_name],
            )
        return out

    def _load_completed(
        self, signature: str, *, reset_files: bool = False
    ) -> pd.DataFrame | None:
        """Return the saved seed frame when inputs match and it is complete."""
        out = _common.load_completed_csv(
            self.cfg.save_dir,
            "seed",
            signature=signature,
            file_name=self.cfg.file_name,
            reset_files=reset_files,
        )
        if out is None or "entity" not in out.columns or len(out) < self.cfg.num_entities:
            return None
        path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        print(
            f"[Seed] Found completed seed in {path}; loading {len(out)} entities."
        )
        return out

    @staticmethod
    def _parse(raw: Any) -> list[str]:
        obj = safe_json(raw)
        if isinstance(obj, dict):
            return [str(v) for v in obj.values() if str(v).strip()]
        return []

    def _save(self, out: pd.DataFrame) -> None:
        path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        try:
            out.to_csv(path, index=False)
        except OSError:
            pass
