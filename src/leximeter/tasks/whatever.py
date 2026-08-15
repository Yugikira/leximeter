"""Whatever task: run arbitrary prompts through the provider-agnostic engine.

Mirrors GABRIEL's ``whatever`` surface: ``web_search`` / ``web_search_filters`` /
``search_context_size`` are forwarded as retrieval options.  When web search is
enabled, each prompt is first expanded with retrieved context (via the injected
``search_tool``), then passed to :func:`leximeter.engine.batch_execute`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..utils.parsing import safe_json
from . import _common
from .base import Task

logger = logging.getLogger(__name__)


@dataclass
class WhateverConfig:
    save_dir: str = "whatever"
    file_name: str = "custom_prompt_responses.csv"
    model: str = "gpt-5.4-mini"
    json_mode: bool = False
    n_parallels: int = 50
    web_search: bool | None = None
    web_search_filters: dict[str, Any] | None = None
    search_context_size: str = "medium"


class Whatever(Task):
    """Run a batch of arbitrary prompts, reusing concurrency + checkpointing."""

    async def run(
        self,
        prompts: str | list[str] | None = None,
        *,
        df: pd.DataFrame | None = None,
        column_name: str | None = None,
        identifiers: list[str] | None = None,
        identifier_column: str | None = None,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        prompts_list, identifiers_list = self._normalize(
            prompts, df=df, column_name=column_name,
            identifiers=identifiers, identifier_column=identifier_column,
        )

        web_search_flag = self.cfg.web_search
        if web_search_flag is None:
            web_search_flag = bool(self.cfg.web_search_filters)
        if web_search_flag:
            prompts_list = await self._augment_with_search(prompts_list)

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

    # ------------------------------------------------------------------
    # Web-search augmentation (GABRIEL's ``whatever`` web_search mode)
    # ------------------------------------------------------------------
    def _get_search_tool(self) -> Any | None:
        """Resolve the search tool, forwarding GABRIEL-style config knobs."""
        if self._search_tool is None:
            from ..client.resolve import resolve_search_tool

            self._search_tool = resolve_search_tool(
                search_context_size=self.cfg.search_context_size,
                web_search_filters=self.cfg.web_search_filters,
            )
        return self._search_tool

    async def _augment_with_search(
        self, prompts: list[str]
    ) -> list[str]:
        """Prepend retrieved context to each prompt when a search tool is available.

        A prompt is used verbatim when retrieval yields nothing (no tool
        configured, empty result, or a failed call) — search augments but never
        drops a prompt.  Concurrency is capped by ``n_parallels``.
        """
        tool = self._get_search_tool()
        if tool is None:
            logger.warning(
                "web_search is enabled but no search tool is configured; "
                "running prompts without retrieval."
            )
            return prompts

        sem = asyncio.Semaphore(max(1, self.cfg.n_parallels))

        async def search_one(prompt: str) -> str:
            async with sem:
                try:
                    return await tool.search(prompt)
                except Exception as exc:  # noqa: BLE001 - provider errors
                    logger.warning("web search failed for %r: %s", prompt, exc)
                    return ""

        contexts = await asyncio.gather(*(search_one(p) for p in prompts))
        return [
            f"{ctx.strip()}\n\n{prompt}" if ctx and ctx.strip() else prompt
            for ctx, prompt in zip(contexts, prompts)
        ]

    def _normalize(
        self,
        prompts: str | list[str] | None,
        *,
        df: pd.DataFrame | None,
        column_name: str | None,
        identifiers: list[str] | None,
        identifier_column: str | None,
    ) -> tuple[list[str], list[str]]:
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
