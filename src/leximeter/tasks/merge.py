"""Merge task: fuzzy-join two tables via LLM matching."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ..templates import load_prompt
from ..utils.parsing import safe_json
from ..utils.similarity import top_k_similar
from . import _common
from .base import Task


@dataclass
class MergeConfig:
    save_dir: str = "merge"
    file_name: str = "merge_responses.csv"
    model: str = "gpt-5.4-nano"
    n_parallels: int = 50
    additional_instructions: Optional[str] = None
    short_list_len: int = 16
    long_list_len: int = 256
    use_embeddings: bool = True


class Merge(Task):
    """Match each short (left) term to the best long (right) term."""

    def __init__(self, cfg: MergeConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("merge_prompt.jinja2")

    async def run(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        *,
        on: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        how: str = "left",
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        left_col = left_on or on
        right_col = right_on or on
        if not left_col or not right_col:
            raise ValueError("provide `on` or both `left_on` and `right_on`")

        left_terms = [str(v) for v in df_left[left_col].tolist()]
        right_terms = [str(v) for v in df_right[right_col].tolist()]

        mapping: Dict[str, str] = {}
        if self.cfg.use_embeddings and left_terms and right_terms:
            try:
                mapping = await self._embedding_match(
                    left_terms, right_terms, reset_files, **kwargs
                )
            except Exception as exc:  # noqa: BLE001 - optional dependency may be absent
                print(
                    f"[Merge] embedding prefilter unavailable ({exc}); "
                    "falling back to batched matching."
                )
                mapping = await self._batched_match(
                    left_terms, right_terms, reset_files, **kwargs
                )
        else:
            mapping = await self._batched_match(
                left_terms, right_terms, reset_files, **kwargs
            )

        out = df_left.copy()
        out["_match"] = [mapping.get(str(v)) for v in df_left[left_col]]
        merged = out.merge(
            df_right,
            left_on="_match",
            right_on=right_col,
            how=how,
            suffixes=("", "_right"),
        ).drop(columns=["_match"])
        self._save(merged)
        return merged

    async def _embedding_match(self, left_terms, right_terms, reset_files, **kwargs):
        left_emb = await self._get_embedding_client().embed(left_terms)
        right_emb = await self._get_embedding_client().embed(right_terms)
        shortlists = top_k_similar(left_emb, right_emb, self.cfg.long_list_len)

        prompts, ids, metas = [], [], []
        for i, term in enumerate(left_terms):
            candidates = [right_terms[j] for j in shortlists[i]]
            ids.append(_common.hash_identifier(term))
            prompts.append(
                self._template.render(
                    short_list={term: ""},
                    long_list=candidates,
                    additional_instructions=self.cfg.additional_instructions,
                )
            )
            metas.append(term)

        resp_df = await self._execute(prompts, ids, reset_files=reset_files, **kwargs)
        resp_lookup = dict(zip(resp_df["Identifier"], resp_df["Response"]))

        mapping: Dict[str, str] = {}
        for term, ident in zip(metas, ids):
            obj = safe_json(resp_lookup.get(ident))
            if isinstance(obj, dict):
                for short, long in obj.items():
                    if long != "no certain match":
                        mapping[str(short)] = str(long)
        return mapping

    async def _batched_match(self, left_terms, right_terms, reset_files, **kwargs):
        long_terms = right_terms[: self.cfg.long_list_len]
        mapping: Dict[str, str] = {}
        prompts, ids, batches = [], [], []
        for i in range(0, len(left_terms), self.cfg.short_list_len):
            batch = list(dict.fromkeys(left_terms[i : i + self.cfg.short_list_len]))
            ids.append(_common.hash_identifier("\x1f".join(sorted(batch))))
            prompts.append(
                self._template.render(
                    short_list={t: "" for t in batch},
                    long_list=long_terms,
                    additional_instructions=self.cfg.additional_instructions,
                )
            )
            batches.append(batch)

        resp_df = await self._execute(prompts, ids, reset_files=reset_files, **kwargs)
        resp_lookup = dict(zip(resp_df["Identifier"], resp_df["Response"]))

        for batch, ident in zip(batches, ids):
            obj = safe_json(resp_lookup.get(ident))
            if isinstance(obj, dict):
                for short, long in obj.items():
                    if long != "no certain match" and str(short) in batch:
                        mapping[str(short)] = str(long)
        return mapping

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
