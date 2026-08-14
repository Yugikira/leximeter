"""Deduplicate task: map equivalent terms to one representative."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from ..templates import load_prompt
from ..utils.parsing import safe_json
from . import _common
from .base import Task


@dataclass
class DeduplicateConfig:
    save_dir: str = "deduplicate"
    file_name: str = "deduplicate_responses.csv"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    additional_instructions: Optional[str] = None
    group_size: int = 500
    modality: str = "entity"


class Deduplicate(Task):
    """Map conceptually-equivalent terms to one representative term."""

    def __init__(self, cfg: DeduplicateConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("deduplicate_prompt.jinja2")

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        values = [str(v) for v in df[column_name].tolist()]
        unique = list(dict.fromkeys(values))
        groups = [
            unique[i : i + self.cfg.group_size]
            for i in range(0, len(unique), self.cfg.group_size)
        ]

        mapping: Dict[str, str] = {}
        prompts, ids, groups_meta = [], [], []
        for g_idx, group in enumerate(groups):
            ids.append(_common.hash_identifier("\x1f".join(sorted(group))))
            prompts.append(
                self._template.render(
                    raw_terms="\n".join(group),
                    modality=self.cfg.modality,
                    additional_instructions=self.cfg.additional_instructions,
                )
            )
            groups_meta.append(group)

        resp_df = await self._execute(prompts, ids, reset_files=reset_files, **kwargs)
        resp_lookup = dict(zip(resp_df["Identifier"], resp_df["Response"]))

        for group, ident in zip(groups_meta, ids):
            obj = safe_json(resp_lookup.get(ident))
            if isinstance(obj, dict):
                for rep, raws in obj.items():
                    if not isinstance(raws, list):
                        continue
                    for raw in raws:
                        mapping[str(raw)] = str(rep)

        out = df.copy()
        out[f"mapped_{column_name}"] = [
            mapping.get(str(v), str(v)) for v in df[column_name]
        ]
        self._save(out)
        return out

    async def _execute(self, prompts, ids, *, reset_files, **kwargs):
        from ..engine.batch import batch_execute

        return await batch_execute(
            prompts,
            ids,
            client=self._get_client(),
            model=self.cfg.model,
            json_mode=True,
            n_parallels=self.cfg.n_parallels,
            save_path=os.path.join(self.cfg.save_dir, self.cfg.file_name),
            reset_files=reset_files,
            **kwargs,
        )

    def _save(self, out: pd.DataFrame) -> None:
        out_path = os.path.join(self.cfg.save_dir, f"{self._base_name()}_cleaned.csv")
        try:
            out.to_csv(out_path, index=False)
        except Exception:
            pass
