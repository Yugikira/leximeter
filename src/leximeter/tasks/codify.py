"""Codify task: passage coding — mark snippets matching qualitative codes."""

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
class CodifyConfig:
    categories: Optional[Dict[str, str]] = None
    save_dir: str = "codify"
    file_name: str = "coding_results.csv"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    additional_instructions: str = ""
    modality: str = "text"


class Codify(Task):
    """Mark snippets that match each qualitative code (static categories in v1)."""

    def __init__(self, cfg: CodifyConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("codify_prompt.jinja2")

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        categories: Optional[Dict[str, str]] = None,
        additional_instructions: str = "",
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        cats = categories if categories is not None else self.cfg.categories
        values = [str(v) for v in df[column_name].tolist()]
        ids = [_common.hash_identifier(v) for v in values]
        prompts = [
            self._template.render(
                text=v,
                categories=cats,
                modality=self.cfg.modality,
                additional_instructions=additional_instructions or self.cfg.additional_instructions,
            )
            for v in values
        ]

        resp_df = await self._execute(prompts, ids, reset_files=reset_files, **kwargs)
        resp_lookup = dict(zip(resp_df["Identifier"], resp_df["Response"]))

        out = df.copy()
        out["coded_passages"] = [safe_json(resp_lookup.get(ident)) for ident in ids]
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
            save_path=os.path.join(
                self.cfg.save_dir, f"{self._base_name()}_raw_responses.csv"
            ),
            reset_files=reset_files,
            **kwargs,
        )

    def _save(self, out: pd.DataFrame) -> None:
        out_path = os.path.join(self.cfg.save_dir, "coded_passages.csv")
        try:
            out.to_csv(out_path, index=False)
        except Exception:
            pass
