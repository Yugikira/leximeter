"""Seed task: generate a representative / diverse set of entities."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..templates import load_prompt
from ..utils.parsing import safe_json
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
        entities: list[str] = list(existing_entities or [])
        rng = random.Random(self.cfg.deduplicate_sample_seed)

        batch_idx = 0
        while len(entities) < self.cfg.num_entities:
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
            text = await self._get_client().complete(
                prompt, model=self.cfg.model, json_mode=True, **kwargs
            )
            new = self._parse(text)
            if not new:
                break
            entities.extend(new)
            batch_idx += 1

        entities = entities[: self.cfg.num_entities]
        out = pd.DataFrame(
            {
                "entity": entities,
                "entity_id": [f"entity-{i:05d}" for i in range(len(entities))],
            }
        )
        self._save(out)
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
