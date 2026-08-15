"""Extract task: pull structured facts out of each item.

Ported from GABRIEL's ``tasks/extract.py``: a standalone task (not an
``AttributeTask``) because responses can contain **multiple entities**, each
expanding into its own output row.  Supports per-batch attribute prompts,
multiple runs, type coercion, and legacy ``attributes.json`` persistence.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..templates import load_prompt
from ..utils.file_utils import save_dataframe_with_fallback
from ..utils.parsing import safest_json
from . import _common
from .base import Task


@dataclass
class ExtractConfig:
    attributes: dict[str, str]
    save_dir: str = "extraction"
    file_name: str = "extraction.csv"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    n_runs: int = 1
    n_attributes_per_run: int | None = None
    additional_instructions: str | None = None
    modality: str = "entity"
    types: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.additional_instructions is not None:
            cleaned = str(self.additional_instructions).strip()
            self.additional_instructions = cleaned or None


class Extract(Task):
    """Extract attributes from passages using an LLM."""

    def __init__(self, cfg: ExtractConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("extraction_prompt.jinja2")

    def render_prompt(self, text: str, attributes: dict[str, str]) -> str:
        return self._template.render(
            text=text,
            attributes=attributes,
            modality=self.cfg.modality,
            additional_instructions=self.cfg.additional_instructions,
        )

    # ------------------------------------------------------------------
    # Parsing: a response may name several entities; each becomes a row.
    # ------------------------------------------------------------------
    async def _parse(
        self, raw: Any, attrs: list[str]
    ) -> list[tuple[str | None, dict[str, str]]]:
        obj = await safest_json(raw)
        attr_names = list(attrs)

        def _default_attr_map() -> dict[str, str]:
            return {attr: "unknown" for attr in attr_names}

        def _clean_name(name: Any) -> str | None:
            if isinstance(name, str):
                cleaned = name.strip()
                return cleaned or None
            if name is None:
                return None
            text = str(name).strip()
            return text or None

        def _build_entry(
            entity_name: str | None, payload: dict[str, Any] | None
        ) -> tuple[str | None, dict[str, str]]:
            values = _default_attr_map()
            if isinstance(payload, dict):
                for attr in attr_names:
                    val = payload.get(attr)
                    values[attr] = str(val) if val is not None else "unknown"
            return (_clean_name(entity_name), values)

        entries: list[tuple[str | None, dict[str, str]]] = []

        if isinstance(obj, dict):
            attr_keys = set(attr_names)
            nested_candidates = [
                (key, val)
                for key, val in obj.items()
                if isinstance(val, dict)
                and (not attr_keys or key not in attr_keys)
            ]
            if nested_candidates:
                for name, payload in nested_candidates:
                    entries.append(_build_entry(name, payload))
                if entries:
                    return entries
            entries.append(_build_entry(None, obj))
            return entries

        if isinstance(obj, list):
            for item in obj:
                if not isinstance(item, dict):
                    continue
                attr_payload: dict[str, Any] | None = None
                if isinstance(item.get("attributes"), dict):
                    attr_payload = item.get("attributes")  # type: ignore[assignment]
                else:
                    attr_payload = item
                name = item.get("entity_name") or item.get("entity") or item.get("name")
                entries.append(_build_entry(name, attr_payload))
            if entries:
                return entries

        return [
            _build_entry(None, None),
        ]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        types: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        from ..engine.batch import batch_execute

        df_proc = df.reset_index(drop=True).copy()
        input_columns = list(df_proc.columns)
        if types is None:
            types = self.cfg.types
        base_name = self._base_name()
        csv_path = os.path.join(self.cfg.save_dir, f"{base_name}_raw_responses.csv")
        run_metadata = _common.load_run_metadata(
            self.cfg.save_dir, base_name, reset_files=reset_files
        )

        # Attributes, persisted for reproducibility (legacy filename supported).
        self.cfg.attributes = _common.load_persisted_attributes(
            save_dir=self.cfg.save_dir,
            incoming=self.cfg.attributes,
            reset_files=reset_files,
            task_name="Extract",
            item_name="attributes",
            legacy_filename=f"{base_name}_attrs.json",
        )

        values = df_proc[column_name].tolist()
        texts = [str(v) for v in values]

        base_ids: list[str] = []
        id_to_rows: defaultdict[str, list[int]] = defaultdict(list)
        id_to_val: dict[str, Any] = {}
        prompt_texts: dict[str, str] = {}
        row_ids: list[str] = []

        for row, (passage, orig) in enumerate(zip(texts, values)):
            ident = _common.hash_identifier(passage)
            row_ids.append(ident)
            id_to_rows[ident].append(row)
            if len(id_to_rows[ident]) > 1:
                continue
            id_to_val[ident] = orig
            prompt_texts[ident] = passage
            base_ids.append(ident)

        df_proc["_gid"] = row_ids

        if not base_ids:
            out_path = os.path.join(self.cfg.save_dir, f"{base_name}_cleaned.csv")
            result = df_proc.drop(columns=["_gid"])
            result["entity_name"] = pd.NA
            for attr in self.cfg.attributes:
                result[attr] = pd.NA
            save_dataframe_with_fallback(result, out_path, index=False, label="Extract")
            return result

        attr_items = list(self.cfg.attributes.items())
        attr_batch_items, effective_n_attributes_per_run = (
            _common.resolve_attribute_batches(
                attr_items,
                requested_n=self.cfg.n_attributes_per_run,
                metadata=run_metadata,
                reset_files=reset_files,
            )
        )
        attr_batches: list[dict[str, str]] = [dict(batch) for batch in attr_batch_items]
        attr_count = len(attr_items)
        if (
            effective_n_attributes_per_run is not None
            and attr_count > effective_n_attributes_per_run
        ):
            batches = (
                attr_count + effective_n_attributes_per_run - 1
            ) // effective_n_attributes_per_run
            print(
                f"[Extract] {attr_count} attributes provided. n_attributes_per_run={effective_n_attributes_per_run}. "
                f"Splitting into {batches} prompt batches; set n_attributes_per_run=None to process them together."
            )
        _common.update_run_metadata(
            self.cfg.save_dir,
            base_name,
            model=self.cfg.model,
            n_attributes_per_run=effective_n_attributes_per_run,
            attribute_batches=[[name for name, _ in b] for b in attr_batch_items],
        )

        prompts: list[str] = []
        ids: list[str] = []
        for batch_idx, batch_attrs in enumerate(attr_batches):
            for ident in base_ids:
                prompts.append(self.render_prompt(prompt_texts[ident], batch_attrs))
                ids.append(f"{ident}_batch{batch_idx}")

        if not isinstance(self.cfg.n_runs, int) or self.cfg.n_runs < 1:
            raise ValueError("n_runs must be an integer >= 1")

        existing_ids: set[str] = set()
        if not reset_files and os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path, usecols=["Identifier"])
                existing_ids = set(existing_df["Identifier"].astype(str))
            except (OSError, ValueError):
                existing_ids = set()

        run_identifier_lists: list[list[str]] = []
        for run_idx in range(1, self.cfg.n_runs + 1):
            run_ids: list[str] = []
            for ident in ids:
                if run_idx == 1:
                    legacy_ident = f"{ident}_run1"
                    run_ids.append(legacy_ident if legacy_ident in existing_ids else ident)
                else:
                    run_ids.append(f"{ident}_run{run_idx}")
            run_identifier_lists.append(run_ids)

        prompts_all: list[str] = []
        ids_all: list[str] = []
        for run_ids in run_identifier_lists:
            prompts_all.extend(prompts)
            ids_all.extend(run_ids)

        df_resp_all = await batch_execute(
            prompts=prompts_all,
            identifiers=ids_all,
            client=self._get_client(),
            model=self.cfg.model,
            json_mode=self.cfg.modality != "audio",
            n_parallels=self.cfg.n_parallels,
            save_path=csv_path,
            reset_files=reset_files,
            **kwargs,
        )

        df_resps = []
        for run_idx, run_ids in enumerate(run_identifier_lists, start=1):
            suffix = f"_run{run_idx}"
            sub = df_resp_all[df_resp_all.Identifier.isin(run_ids)].copy()
            sub.Identifier = sub.Identifier.str.replace(suffix + "$", "", regex=True)
            df_resps.append(sub)

        full_records: list[dict[str, Any]] = []
        base_attrs = list(self.cfg.attributes.keys())
        for run_idx, df_resp in enumerate(df_resps, start=1):
            id_to_entity_vals: dict[str, dict[str | None, dict[str, str]]] = {
                ident: {} for ident in base_ids
            }
            for ident_batch, raw in zip(df_resp.Identifier, df_resp.Response):
                if "_batch" not in ident_batch:
                    continue
                base_ident, batch_part = ident_batch.rsplit("_batch", 1)
                batch_idx = int(batch_part)
                attrs = list(attr_batches[batch_idx].keys())
                parsed_entities = await self._parse(raw, attrs)
                entity_store = id_to_entity_vals.setdefault(base_ident, {})
                for entity_name, entity_attrs in parsed_entities:
                    key = entity_name if entity_name is not None else None
                    if key not in entity_store:
                        entity_store[key] = {attr: "unknown" for attr in base_attrs}
                    for attr in attrs:
                        entity_store[key][attr] = entity_attrs.get(attr, "unknown")
            for ident in base_ids:
                entity_map = id_to_entity_vals.get(ident) or {
                    None: {attr: "unknown" for attr in base_attrs}
                }
                for entity_name, attr_values in entity_map.items():
                    rec: dict[str, Any] = {
                        "id": ident,
                        "entity_name": entity_name,
                        "text": id_to_val[ident],
                        "run": run_idx,
                    }
                    rec.update({attr: attr_values.get(attr, "unknown") for attr in base_attrs})
                    full_records.append(rec)

        full_df = pd.DataFrame(full_records).set_index(["id", "entity_name", "run"])
        if self.cfg.n_runs > 1:
            disagg_path = os.path.join(
                self.cfg.save_dir, f"{base_name}_full_disaggregated.csv"
            )
            save_dataframe_with_fallback(
                full_df,
                disagg_path,
                index=True,
                label="Extract",
            )

        def _pick_first(s: pd.Series) -> str:
            for val in s.dropna():
                if str(val).strip().lower() != "unknown":
                    return str(val)
            return "unknown"

        entity_index = full_df.index.droplevel("run").unique()
        if base_attrs:
            agg_series = {
                attr: full_df[attr]
                .groupby(level=["id", "entity_name"], sort=False)
                .apply(_pick_first)
                for attr in base_attrs
            }
            agg_df = pd.DataFrame(agg_series)
        else:
            agg_df = pd.DataFrame(index=entity_index)
        agg_df = agg_df.reindex(entity_index)

        unknown_counts = {attr: (agg_df[attr] == "unknown").sum() for attr in base_attrs}

        out_path = os.path.join(self.cfg.save_dir, f"{base_name}_cleaned.csv")
        agg_reset = agg_df.reset_index()
        result = df_proc.merge(agg_reset, left_on="_gid", right_on="id", how="left")
        drop_cols = [col for col in ("_gid", "id") if col in result.columns]
        if drop_cols:
            result = result.drop(columns=drop_cols)

        original_cols = [col for col in df_proc.columns if col != "_gid"]
        final_order: list[str] = []
        for col in original_cols:
            if col in result.columns:
                final_order.append(col)
        if "entity_name" in result.columns and "entity_name" not in final_order:
            final_order.append("entity_name")
        for attr in base_attrs:
            if attr in result.columns:
                final_order.append(attr)
        remaining = [col for col in result.columns if col not in final_order]
        if remaining:
            final_order.extend(remaining)
        result = result[final_order]

        save_dataframe_with_fallback(result, out_path, index=False, label="Extract")

        result = result.replace("unknown", pd.NA)

        duplicate_rows = len(result) - len(df_proc)

        if types:
            coerced = result.copy()
            fail_logs: dict[str, int] = {}
            for col, typ in types.items():
                if col not in coerced:
                    continue
                orig = coerced[col]
                non_null = orig.notna()
                target = str(typ).lower()
                if target in {"datetime", "date"}:
                    conv = pd.to_datetime(orig, errors="coerce")
                else:
                    conv = pd.to_numeric(orig, errors="coerce")
                    if target in {"int", "int64"}:
                        conv = conv.round().astype("Int64")
                coerced[col] = conv
                fail_logs[col] = int((non_null & conv.isna()).sum())
            coerced_path = os.path.join(
                self.cfg.save_dir, f"{base_name}_cleaned_coerced.csv"
            )
            save_dataframe_with_fallback(
                coerced, coerced_path, index=False, label="Extract"
            )
            for col, n_fail in fail_logs.items():
                print(f"[Extract] Failed to coerce {n_fail} values in column '{col}'.")
            result = coerced

        total = len(agg_df)
        print("\n=== Extraction coverage ===")
        for attr in base_attrs:
            known = total - unknown_counts[attr]
            print(f"{attr:<55s}: {known:5d} extracted, {unknown_counts[attr]:5d} unknown")
        print("============================\n")

        if duplicate_rows > 0:
            subset_hint = ", ".join(f"'{col}'" for col in input_columns)
            print(
                "[Extract] Multiple entity names were returned for at least one input.\n"
                f"          {duplicate_rows} additional row(s) were created so the cleaned file now has {len(result)} rows versus {len(df_proc)} inputs.\n"
                "          If you only need one row per original input, deduplicate on your source columns\n"
                f"          (e.g. `result = result.drop_duplicates(subset=[{subset_hint}], keep='first')`).\n"
            )

        return result
