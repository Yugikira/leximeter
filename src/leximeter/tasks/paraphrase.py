"""Paraphrase task: rewrite text consistently per instructions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd

from ..templates import load_prompt
from . import _common
from .base import Task


@dataclass
class ParaphraseConfig:
    instructions: str
    save_dir: str = "paraphrase"
    file_name: str = "paraphrase_responses.csv"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    modality: str = "text"
    json_mode: bool = False
    revised_column_name: Optional[str] = None


class Paraphrase(Task):
    """Rewrite each item per ``instructions`` (single round in v1)."""

    def __init__(self, cfg: ParaphraseConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("paraphrase_prompt.jinja2")

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        values = [str(v) for v in df[column_name].tolist()]
        ids = [_common.hash_identifier(v) for v in values]
        prompts = [
            self._template.render(
                text=v,
                instructions=self.cfg.instructions,
                modality=self.cfg.modality,
                json_mode=self.cfg.json_mode,
            )
            for v in values
        ]

        resp_df = await self._execute(prompts, ids, reset_files=reset_files, **kwargs)
        resp_lookup = dict(zip(resp_df["Identifier"], resp_df["Response"]))

        out_col = self.cfg.revised_column_name or f"{column_name}_paraphrased"
        out = df.copy()
        out[out_col] = [resp_lookup.get(ident, "") for ident in ids]
        self._save(out)
        return out

    async def _execute(self, prompts: List[str], ids: List[str], *, reset_files, **kwargs):
        from ..engine.batch import batch_execute

        return await batch_execute(
            prompts,
            ids,
            client=self._get_client(),
            model=self.cfg.model,
            json_mode=self.cfg.json_mode,
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
        except Exception:
            pass
