"""Compare task: list similarities/differences between paired items.

Ported and modified from openai/GABRIEL's ``tasks/compare.py`` (Apache-2.0).
"""

from __future__ import annotations

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
class CompareConfig:
    save_dir: str = "comparison"
    file_name: str = "comparison_responses.csv"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    differentiate: bool = True
    additional_instructions: str | None = None
    modality: str = "text"
    circle_first: bool | None = None


class Compare(Task):
    """Identify similarities/differences between paired entries."""

    def __init__(self, cfg: CompareConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("comparison_prompt.jinja2")

    async def run(
        self,
        df: pd.DataFrame,
        circle_column_name: str,
        square_column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        pairs: list[tuple[str, str]] = []
        for circle, square in zip(df[circle_column_name], df[square_column_name]):
            if pd.isna(circle) or pd.isna(square):
                continue
            pairs.append((str(circle), str(square)))

        # dedup identical pairs (same pair -> same comparison)
        unique = list(dict.fromkeys(pairs))

        prompts: list[str] = []
        ids: list[str] = []
        for circle, square in unique:
            ident = _common.hash_identifier(f"{circle}\x1f{square}")
            ids.append(ident)
            circle_first_flag = (
                self.cfg.circle_first
                if self.cfg.circle_first is not None
                else random.random() < 0.5
            )
            prompts.append(
                self._template.render(
                    entry_circle=circle,
                    entry_square=square,
                    modality=self.cfg.modality,
                    circle_first=circle_first_flag,
                    differentiate=self.cfg.differentiate,
                    additional_instructions=self.cfg.additional_instructions,
                )
            )

        resp_df = await self._execute_or_reuse(
            prompts, ids, reset_files=reset_files, **kwargs
        )
        resp_lookup = dict(zip(resp_df["Identifier"], resp_df["Response"]))

        rows: list[dict] = []
        for (circle, square), ident in zip(unique, ids):
            obj = safe_json(resp_lookup.get(ident))
            if isinstance(obj, dict):
                for term, explanation in obj.items():
                    rows.append(
                        {
                            "circle": circle,
                            "square": square,
                            "attribute": term,
                            "explanation": explanation,
                        }
                    )
        return pd.DataFrame(rows)

    async def _execute_or_reuse(self, prompts, ids, *, reset_files, **kwargs):
        """Reuse the batch checkpoint when every identifier is already done.

        Compare identifiers are content hashes of the pair, so a fully
        completed checkpoint is self-validating: same pairs -> same ids.
        This short-circuit makes reruns fully deterministic with zero
        provider calls (no prompt building, no dispatch).
        """
        if not reset_files and ids:
            from ..engine import checkpoint as _ckpt

            checkpoint = _ckpt.load_checkpoint(
                os.path.join(
                    self.cfg.save_dir, f"{self._base_name()}_raw_responses.csv"
                )
            )
            done = _ckpt.completed_identifiers(checkpoint)
            if all(ident in done for ident in ids):
                print(
                    f"[Compare] All {len(ids)} comparisons already complete; "
                    "reusing checkpoint."
                )
                return checkpoint
        return await self._execute(prompts, ids, reset_files=reset_files, **kwargs)

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
