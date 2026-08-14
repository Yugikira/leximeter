"""Task base classes: eliminate cross-task copy-paste via the template method.

Every task shares the same infrastructure (client injection, ``save_dir``
expansion, identifier hashing, attribute persistence).  :class:`Task` holds
that common ground; :class:`AttributeTask` adds the shared
"measure N attributes per item" pipeline.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        search_tool: Optional[SearchTool] = None,
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

    def _get_embedding_client(self) -> EmbeddingClient:
        if self._embedding_client is None:
            from ..client.resolve import resolve_embedding_client

            self._embedding_client = resolve_embedding_client()
        return self._embedding_client

    def _get_search_tool(self) -> Optional[SearchTool]:
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

    def _build_ids(self, texts: Sequence[str]) -> Tuple[List[str], List[str], Dict[str, str]]:
        """Hash + dedup: ``row_ids`` align to input rows; ``base_ids`` are unique."""
        row_ids: List[str] = []
        base_ids: List[str] = []
        id_to_text: Dict[str, str] = {}
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
    def render_prompt(self, text: str, attributes: Dict[str, str]) -> str:
        """Render one prompt for ``text`` against a batch of ``attributes``."""

    @abstractmethod
    async def parse_response(
        self, raw: Any, attributes: Sequence[str]
    ) -> Dict[str, Any]:
        """Parse one raw completion into ``{attribute: value}``."""

    def _attribute_names(self) -> List[str]:
        return list(self.cfg.attributes.keys())

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
            dict(self.cfg.attributes),
            reset_files=reset_files,
            task_name=self.__class__.__name__,
        )
        self.cfg.attributes = attributes

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
        prompts: List[str] = []
        ids: List[str] = []
        meta: List[Tuple[str, int, int]] = []  # (base_id, batch_idx, run_idx)
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
        records = await self._parse_all(meta, ids, resp_lookup, batches)

        # 7. aggregate + merge back onto the input frame
        result = self.aggregate(records, df_proc)
        self._save(result)
        return result

    # ---- internal steps ---------------------------------------------
    async def _execute(
        self,
        prompts: List[str],
        ids: List[str],
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
        meta: List[Tuple[str, int, int]],
        ids: List[str],
        resp_lookup: Dict[str, Any],
        batches: Sequence[Sequence[Tuple[str, Any]]],
    ) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for (base_id, batch_idx, run_idx), ident in zip(meta, ids):
            names = [name for name, _ in batches[batch_idx]]
            parsed = await self.parse_response(resp_lookup.get(ident), names)
            records.append({"id": base_id, "run": run_idx, **parsed})
        return pd.DataFrame(records)

    def _save(self, result: pd.DataFrame) -> None:
        out_path = os.path.join(self.cfg.save_dir, f"{self._base_name()}_cleaned.csv")
        try:
            result.to_csv(out_path, index=False)
        except Exception:
            pass
