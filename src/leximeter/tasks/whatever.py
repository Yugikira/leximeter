"""Whatever task: run arbitrary prompts through the provider-agnostic engine."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import pandas as pd

from ..utils.parsing import safe_json
from . import _common
from .base import Task


@dataclass
class WhateverConfig:
    save_dir: str = "whatever"
    file_name: str = "custom_prompt_responses.csv"
    model: str = "gpt-5.4-mini"
    json_mode: bool = False
    n_parallels: int = 50


class Whatever(Task):
    """Run a batch of arbitrary prompts, reusing concurrency + checkpointing."""

    async def run(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        *,
        df: Optional[pd.DataFrame] = None,
        column_name: Optional[str] = None,
        identifiers: Optional[List[str]] = None,
        identifier_column: Optional[str] = None,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        prompts_list, identifiers_list = self._normalize(
            prompts, df=df, column_name=column_name,
            identifiers=identifiers, identifier_column=identifier_column,
        )

        from ..engine.batch import batch_execute

        resp_df = await batch_execute(
            prompts_list,
            identifiers_list,
            client=self._get_client(),
            model=self.cfg.model,
            json_mode=self.cfg.json_mode,
            n_parallels=self.cfg.n_parallels,
            save_path=os.path.join(self.cfg.save_dir, self.cfg.file_name),
            reset_files=reset_files,
            **kwargs,
        )

        cols = ["Identifier", "Response"]
        if self.cfg.json_mode:
            resp_df["Response JSON"] = resp_df["Response"].apply(safe_json)
            cols.append("Response JSON")
        return resp_df[cols].reset_index(drop=True)

    def _normalize(
        self,
        prompts: Optional[Union[str, List[str]]],
        *,
        df: Optional[pd.DataFrame],
        column_name: Optional[str],
        identifiers: Optional[List[str]],
        identifier_column: Optional[str],
    ) -> tuple[List[str], List[str]]:
        if df is not None:
            if column_name is None or column_name not in df.columns:
                raise ValueError("column_name must name a column in df")
            prompts_list = [str(v) for v in df[column_name].tolist()]
            if identifier_column is not None:
                idents = [str(i) for i in df[identifier_column].tolist()]
            else:
                idents = [_common.hash_identifier(p) for p in prompts_list]
        elif isinstance(prompts, str):
            prompts_list = [prompts]
            idents = identifiers or [_common.hash_identifier(prompts)]
        elif prompts is not None:
            prompts_list = [str(p) for p in prompts]
            idents = identifiers or [_common.hash_identifier(p) for p in prompts_list]
        else:
            raise ValueError("provide prompts or df")

        if len(idents) != len(prompts_list):
            raise ValueError("identifiers length must match prompts")
        return prompts_list, idents
