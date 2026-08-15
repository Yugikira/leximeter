"""Deduplicate task: map equivalent terms to one representative."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

from ..client import EmbeddingClient
from ..templates import load_prompt
from ..utils.parsing import safe_json
from . import _common
from .base import Task


@dataclass
class DeduplicateConfig:
    save_dir: str = "deduplicate"
    file_name: str = "deduplicate_responses.csv"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    additional_instructions: str | None = None
    group_size: int = 500
    modality: str = "entity"
    use_embeddings: bool = True


class Deduplicate(Task):
    """Map conceptually-equivalent terms to one representative term."""

    def __init__(self, cfg: DeduplicateConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("deduplicate_prompt.jinja2")

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        values = [str(v) for v in df[column_name].tolist()]
        unique = list(dict.fromkeys(values))
        groups = await self._group_terms(unique, reset_files=reset_files)

        mapping: dict[str, str] = {}
        prompts, ids, groups_meta = [], [], []
        for group in groups:
            ids.append(_common.hash_identifier("\x1f".join(sorted(group))))
            prompts.append(
                self._template.render(
                    raw_terms="\n".join(group),
                    modality=self.cfg.modality,
                    additional_instructions=self.cfg.additional_instructions,
                )
            )
            groups_meta.append(group)

        resp_df = await self._execute(prompts, ids, reset_files=reset_files, **kwargs)
        resp_lookup = dict(zip(resp_df["Identifier"], resp_df["Response"]))

        for group, ident in zip(groups_meta, ids):
            obj = safe_json(resp_lookup.get(ident))
            if isinstance(obj, dict):
                for rep, raws in obj.items():
                    if not isinstance(raws, list):
                        continue
                    for raw in raws:
                        mapping[str(raw)] = str(rep)

        out = df.copy()
        out[f"mapped_{column_name}"] = [
            mapping.get(str(v), str(v)) for v in df[column_name]
        ]
        self._save(out)
        return out

    async def _embed_with_cache(
        self, texts: list[str], client: EmbeddingClient, *, reset_files: bool
    ) -> dict[str, list[float]]:
        """Embed ``texts`` keyed by text, checkpointing to a JSON cache.

        Mirrors GABRIEL's ``get_all_embeddings`` checkpoint contract
        (``openai_utils.py:2688``): on resume only texts missing from the
        cache are re-embedded.  The cache lives at
        ``{save_dir}/deduplicate_embeddings.json`` and file IO runs in a
        worker thread so it does not block the event loop.
        """
        cache_path = os.path.join(self.cfg.save_dir, "deduplicate_embeddings.json")
        vectors: dict[str, list[float]] = {}
        if not reset_files:
            cached = await asyncio.to_thread(_common.load_json_cache, cache_path)
            if cached:
                vectors = {
                    str(k): [float(x) for x in v]
                    for k, v in cached.items()
                    if isinstance(v, list)
                }
        missing = [t for t in texts if t not in vectors]
        if missing:
            embedded = await client.embed(missing)
            vectors.update(
                {str(t): [float(x) for x in v] for t, v in zip(missing, embedded)}
            )
        await asyncio.to_thread(_common.write_json_cache, cache_path, vectors)
        return vectors

    async def _group_terms(
        self, unique: list[str], *, reset_files: bool
    ) -> list[list[str]]:
        """Split unique terms into LLM-sized groups.

        With embeddings enabled and ``len(unique) >= group_size``, terms are
        clustered with ``kmeans2`` (``k = ceil(n / group_size)``) then
        re-chunked to ``group_size`` per group — GABRIEL's contract
        (deduplicate.py:145-164).  Vectors are checkpointed per term (see
        ``_embed_with_cache``).  Otherwise (or when the embedding client is
        unavailable) fall back to plain chunking.
        """
        client = self._get_embedding_client()
        if (
            client is not None
            and self.cfg.use_embeddings
            and len(unique) >= self.cfg.group_size
        ):
            try:
                emb = await self._embed_with_cache(
                    unique, client, reset_files=reset_files
                )
                arr = np.array([emb[t] for t in unique], dtype=float)
                k = max(1, int(np.ceil(len(unique) / self.cfg.group_size)))
                _, labels = kmeans2(arr, k, minit="points")
                clusters: list[list[str]] = [[] for _ in range(k)]
                for term, lbl in zip(unique, labels):
                    clusters[int(lbl)].append(term)
                groups: list[list[str]] = []
                current: list[str] = []
                for cluster in clusters:
                    for term in cluster:
                        current.append(term)
                        if len(current) >= self.cfg.group_size:
                            groups.append(current)
                            current = []
                if current:
                    groups.append(current)
                if groups:
                    return groups
            except Exception as exc:  # noqa: BLE001 - optional dependency may be absent
                print(
                    f"[Deduplicate] embedding prefilter unavailable ({exc}); "
                    "falling back to chunking."
                )
        return [
            unique[i : i + self.cfg.group_size]
            for i in range(0, len(unique), self.cfg.group_size)
        ]

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
        except OSError:
            pass
