"""Task base classes: eliminate cross-task copy-paste via the template method.

Every task shares the same infrastructure (client injection, ``save_dir``
expansion, identifier hashing, attribute persistence).  :class:`Task` holds
that common ground; :class:`AttributeTask` adds the shared
"measure N attributes per item" pipeline.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..client.base import LLMClient
from ..client.embedding import EmbeddingClient
from ..client.search import SearchTool
from . import _common


class Task(ABC):
    """Common infrastructure every task shares.

    Providers are injected at construction (for tests / per-call override) or
    resolved lazily from the environment on first use.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        client: LLMClient | None = None,
        embedding_client: EmbeddingClient | None = None,
        search_tool: SearchTool | None = None,
    ) -> None:
        self.cfg = cfg
        self._client = client
        self._embedding_client = embedding_client
        self._search_tool = search_tool
        self._expand_save_dir()

    # ------------------------------------------------------------------
    # Providers (lazy: dummy-only tests need no key / network / deps)
    # ------------------------------------------------------------------
    def _get_client(self) -> LLMClient:
        if self._client is None:
            from ..client.resolve import resolve_client

            self._client = resolve_client()
        return self._client

    def _get_embedding_client(self) -> EmbeddingClient | None:
        if self._embedding_client is None:
            from ..client.resolve import resolve_embedding_client

            self._embedding_client = resolve_embedding_client()
        return self._embedding_client

    def _get_search_tool(self) -> SearchTool | None:
        if self._search_tool is None:
            from ..client.resolve import resolve_search_tool

            self._search_tool = resolve_search_tool()
        return self._search_tool

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------
    def _expand_save_dir(self) -> None:
        save_dir = getattr(self.cfg, "save_dir", None)
        if save_dir:
            expanded = Path(os.path.expandvars(os.path.expanduser(save_dir)))
            expanded.mkdir(parents=True, exist_ok=True)
            self.cfg.save_dir = str(expanded)

    def _base_name(self) -> str:
        return os.path.splitext(self.cfg.file_name)[0]

    def _build_ids(self, texts: Sequence[str]) -> tuple[list[str], list[str], dict[str, str]]:
        """Hash + dedup: ``row_ids`` align to input rows; ``base_ids`` are unique."""
        row_ids: list[str] = []
        base_ids: list[str] = []
        id_to_text: dict[str, str] = {}
        for text in texts:
            ident = _common.hash_identifier(text)
            row_ids.append(ident)
            if ident not in id_to_text:
                id_to_text[ident] = text
                base_ids.append(ident)
        return row_ids, base_ids, id_to_text

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the task."""
        raise NotImplementedError


class AttributeTask(Task):
    """Base for tasks that score a set of attributes per item.

    Subclasses implement three variant hooks; the invariant pipeline below is
    defined once and reused by every attribute-measurement task.
    """

    # ---- variant hooks (subclass) ------------------------------------
    @abstractmethod
    def render_prompt(self, text: str, attributes: dict[str, str]) -> str:
        """Render one prompt for ``text`` against a batch of ``attributes``."""

    @abstractmethod
    async def parse_response(
        self, raw: Any, attributes: Sequence[str]
    ) -> dict[str, Any]:
        """Parse one raw completion into ``{attribute: value}``."""

    def _get_attributes(self) -> dict[str, str]:
        """Return the attribute (or label) dict for this task."""
        return dict(self.cfg.attributes)

    def _set_attributes(self, attributes: dict[str, str]) -> None:
        self.cfg.attributes = attributes

    def _attribute_names(self) -> list[str]:
        return list(self._get_attributes().keys())

    def aggregate(self, records: pd.DataFrame, df_proc: pd.DataFrame) -> pd.DataFrame:
        """Combine per-run records into final per-item values (default: mean)."""
        attrs = self._attribute_names()
        agg = records.groupby("id")[attrs].mean()
        return df_proc.merge(agg, left_on="_gid", right_index=True, how="left").drop(
            columns=["_gid"]
        )

    # ---- invariant pipeline (template method) ------------------------
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        df_proc = df.reset_index(drop=True).copy()
        texts = [str(v) for v in df_proc[column_name].tolist()]

        # 1. stable identifiers (dedup identical content)
        row_ids, base_ids, id_to_text = self._build_ids(texts)
        df_proc["_gid"] = row_ids

        # 2. attributes, persisted for reproducibility
        attributes = _common.load_persisted_attributes(
            self.cfg.save_dir,
            self._get_attributes(),
            reset_files=reset_files,
            task_name=self.__class__.__name__,
        )
        self._set_attributes(attributes)

        # 3. batch attributes into prompts (reuse saved plan on resume)
        base_name = self._base_name()
        run_metadata = _common.load_run_metadata(
            self.cfg.save_dir, base_name, reset_files=reset_files
        )
        attr_items = list(attributes.items())
        batches, n_per_run = _common.resolve_attribute_batches(
            attr_items,
            requested_n=getattr(self.cfg, "n_attributes_per_run", None),
            metadata=run_metadata,
            reset_files=reset_files,
        )
        _common.update_run_metadata(
            self.cfg.save_dir,
            base_name,
            model=self.cfg.model,
            n_attributes_per_run=n_per_run,
            attribute_batches=[[name for name, _ in b] for b in batches],
        )

        # 4. build jobs: (batch x unique item x run)
        n_runs = max(1, int(getattr(self.cfg, "n_runs", 1)))
        prompts: list[str] = []
        ids: list[str] = []
        meta: list[tuple[str, int, int]] = []  # (base_id, batch_idx, run_idx)
        for batch_idx, batch_items in enumerate(batches):
            batch_attrs = dict(batch_items)
            for ident in base_ids:
                for run_idx in range(1, n_runs + 1):
                    ids.append(f"{ident}_batch{batch_idx}_run{run_idx}")
                    prompts.append(self.render_prompt(id_to_text[ident], batch_attrs))
                    meta.append((ident, batch_idx, run_idx))

        # 5. execute through the provider-agnostic engine
        resp_df = await self._execute(prompts, ids, reset_files=reset_files, **kwargs)
        resp_lookup = dict(zip(resp_df["Identifier"], resp_df["Response"]))

        # 6. parse each response into per-run records
        records = await self._parse_all(meta, ids, resp_lookup, batches, id_to_text)

        # 7. aggregate + merge back onto the input frame
        result = self.aggregate(records, df_proc)
        self._save(result, records)
        return result

    # ---- internal steps ---------------------------------------------
    async def _execute(
        self,
        prompts: list[str],
        ids: list[str],
        *,
        reset_files: bool,
        **kwargs: Any,
    ) -> pd.DataFrame:
        from ..engine.batch import batch_execute

        return await batch_execute(
            prompts,
            ids,
            client=self._get_client(),
            model=self.cfg.model,
            json_mode=True,
            n_parallels=getattr(self.cfg, "n_parallels", 50),
            save_path=os.path.join(
                self.cfg.save_dir, f"{self._base_name()}_raw_responses.csv"
            ),
            reset_files=reset_files,
            **kwargs,
        )

    async def _parse_all(
        self,
        meta: list[tuple[str, int, int]],
        ids: list[str],
        resp_lookup: dict[str, Any],
        batches: Sequence[Sequence[tuple[str, Any]]],
        id_to_text: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for (base_id, batch_idx, run_idx), ident in zip(meta, ids):
            names = [name for name, _ in batches[batch_idx]]
            parsed = await self.parse_response(resp_lookup.get(ident), names)
            records.append(
                {
                    "id": base_id,
                    "run": run_idx,
                    "text": (id_to_text or {}).get(base_id, ""),
                    **parsed,
                }
            )
        return pd.DataFrame(records)

    def _save(
        self, result: pd.DataFrame, records: pd.DataFrame | None = None
    ) -> None:
        from ..utils.file_utils import save_dataframe_with_fallback

        label = self.__class__.__name__
        out_path = os.path.join(self.cfg.save_dir, f"{self._base_name()}_cleaned.csv")
        save_dataframe_with_fallback(result, out_path, index=False, label=label)

        # Keep the disaggregated (id, run) level when n_runs > 1 for later
        # reliability analysis.
        n_runs = max(1, int(getattr(self.cfg, "n_runs", 1)))
        if records is not None and n_runs > 1 and len(records):
            full_df = records.set_index(["id", "run"])
            disagg_path = os.path.join(
                self.cfg.save_dir, f"{self._base_name()}_full_disaggregated.csv"
            )
            save_dataframe_with_fallback(
                full_df, disagg_path, index=True, label=label
            )
