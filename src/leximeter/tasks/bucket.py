"""Bucket task: build a taxonomy of ``bucket_count`` buckets from many terms."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from ..templates import load_prompt
from ..utils.parsing import safe_json
from . import _common
from .base import Task


@dataclass
class BucketConfig:
    bucket_count: int = 10
    save_dir: str = "buckets"
    file_name: str = "bucket_definitions.csv"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    additional_instructions: Optional[str] = None
    n_terms_per_prompt: int = 250


class Bucket(Task):
    """Propose ``bucket_count`` buckets that group the input terms."""

    def __init__(self, cfg: BucketConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("bucket_prompt.jinja2")

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        terms = [str(v) for v in df[column_name].tolist()]
        unique = list(dict.fromkeys(terms))
        chunks = [
            unique[i : i + self.cfg.n_terms_per_prompt]
            for i in range(0, len(unique), self.cfg.n_terms_per_prompt)
        ]

        prompts, ids = [], []
        for chunk in chunks:
            ids.append(_common.hash_identifier("\x1f".join(sorted(chunk))))
            prompts.append(
                self._template.render(
                    bucket_count=self.cfg.bucket_count,
                    terms="\n".join(chunk),
                    additional_instructions=self.cfg.additional_instructions,
                )
            )

        resp_df = await self._execute(prompts, ids, reset_files=reset_files, **kwargs)
        resp_lookup = dict(zip(resp_df["Identifier"], resp_df["Response"]))

        rows = []
        for chunk, ident in zip(chunks, ids):
            obj = safe_json(resp_lookup.get(ident))
            if isinstance(obj, dict):
                for bucket, definition in obj.items():
                    rows.append({"bucket": bucket, "definition": definition})

        out = pd.DataFrame(rows)
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
        out_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        try:
            out.to_csv(out_path, index=False)
        except Exception:
            pass
