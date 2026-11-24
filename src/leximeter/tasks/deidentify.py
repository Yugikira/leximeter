"""Deidentify task: replace PII with consistent, realistic fake PII.

Ported and modified from openai/GABRIEL's ``tasks/deidentify.py`` (Apache-2.0).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..templates import load_prompt
from ..utils.parsing import safe_json
from .base import Task


@dataclass
class DeidentifyConfig:
    save_dir: str = "deidentify"
    file_name: str = "deidentified.csv"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    max_words_per_call: int = 7500
    additional_instructions: str | None = None
    n_passes: int = 1
    use_existing_mappings_only: bool = False


class Deidentify(Task):
    """Replace identifying details with consistent casted substitutes."""

    def __init__(self, cfg: DeidentifyConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("deidentification_prompt.jinja2")

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        grouping_column: str | None = None,
        mapping_column: str | None = None,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        out = df.copy()
        deid_text: dict[int, str] = {}
        mapping_store: dict[int, dict[str, Any]] = {}

        if grouping_column is not None:
            groups = {
                key: list(idxs)
                for key, idxs in df.groupby(grouping_column).indices.items()
            }
        else:
            groups = {i: [i] for i in range(len(df))}

        for row_idxs in groups.values():
            mapping: dict[str, Any] = {}
            if mapping_column and mapping_column in df.columns:
                cell = df[mapping_column].iloc[row_idxs[0]]
                if isinstance(cell, dict):
                    mapping = cell

            if not self.cfg.use_existing_mappings_only:
                for idx in row_idxs:
                    text = str(df[column_name].iloc[idx])
                    for chunk in self._chunk_words(text, self.cfg.max_words_per_call):
                        mapping = await self._update_mapping(chunk, mapping, **kwargs)

            for idx in row_idxs:
                text = str(df[column_name].iloc[idx])
                deid_text[idx] = self._apply_mapping(text, mapping)
                mapping_store[idx] = mapping

        out["deidentified_text"] = [
            deid_text.get(i, str(df[column_name].iloc[i])) for i in range(len(df))
        ]
        out["mapping"] = [mapping_store.get(i, {}) for i in range(len(df))]
        self._save(out)
        return out

    @staticmethod
    def _chunk_words(text: str, max_words: int) -> list[str]:
        words = text.split()
        if max_words <= 0:
            return [text]
        return [
            " ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)
        ]

    async def _update_mapping(
        self, chunk: str, current_map: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        prompt = self._template.render(
            text=chunk,
            current_map=json.dumps(current_map, ensure_ascii=False, indent=2),
            additional_instructions=self.cfg.additional_instructions,
        )
        text = await self._get_client().complete(
            prompt, model=self.cfg.model, json_mode=True, **kwargs
        )
        parsed = safe_json(text)
        return parsed if isinstance(parsed, dict) else current_map

    @staticmethod
    def _apply_mapping(text: str, mapping: dict[str, Any]) -> str:
        for entry in mapping.values():
            if not isinstance(entry, dict):
                continue
            casted = entry.get("casted form")
            if not casted:
                continue
            for form in entry.get("real forms", []):
                form = str(form)
                if not form:
                    continue
                pattern = re.compile(rf"(?<!\w){re.escape(form)}(?!\w)", re.IGNORECASE)
                text = pattern.sub(casted, text)
        return text

    def _save(self, out: pd.DataFrame) -> None:
        out_path = os.path.join(self.cfg.save_dir, f"{self._base_name()}_cleaned.csv")
        try:
            out.to_csv(out_path, index=False)
        except OSError:
            pass
