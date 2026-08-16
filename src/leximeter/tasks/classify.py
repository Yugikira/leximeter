"""Classify task: decide whether each label applies to each item.

Ported from GABRIEL's ``tasks/classify.py``.  Two input modes:

* **Single column** (``column_name``) — every item is judged against the
  labels as a standalone passage.
* **Differentiate** (``circle_column_name`` + ``square_column_name``) —
  pairs are judged against directional label pairs (e.g. ``X_actual`` /
  ``X_inverted``); the prompt asks for at most one of each pair to be True.

Per-run responses are aggregated with a minimum-frequency rule
(``true_count / n_runs >= min_frequency``; all-missing → ``None``).  The
output DataFrame is the input frame merged back on the item values with one
boolean column per label plus a ``predicted_classes`` summary column.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..templates import load_prompt
from ..utils.file_utils import save_dataframe_with_fallback
from ..utils.parsing import safest_json
from . import _common
from .base import Task

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _collect_predictions(row: pd.Series) -> list[str]:
    """Return labels whose values evaluate to ``True``."""
    return [lab for lab, val in row.items() if bool(val)]


@dataclass
class ClassifyConfig:
    labels: dict[str, str]  # {"label_name": "description", ...}
    save_dir: str = "classifier"
    file_name: str = "classify_responses.csv"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    n_runs: int = 1
    min_frequency: float = 0.6
    additional_instructions: str | None = None
    modality: str = "text"
    n_attributes_per_run: int | None = None
    differentiate: bool = False
    circle_first: bool | None = None

    def __post_init__(self) -> None:
        if self.additional_instructions is not None:
            cleaned = str(self.additional_instructions).strip()
            self.additional_instructions = cleaned or None


class Classify(Task):
    """Robust passage classifier using an LLM."""

    def __init__(self, cfg: ClassifyConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("classification_prompt.jinja2")

    # ------------------------------------------------------------------
    # Parsing raw model output
    # ------------------------------------------------------------------
    @staticmethod
    def _regex(raw: str, labels: list[str]) -> dict[str, bool | None]:
        out: dict[str, bool | None] = {}
        for lab in labels:
            pat = re.compile(
                rf'\s*"?\s*{re.escape(lab)}\s*"?\s*:\s*(true|false)',
                re.IGNORECASE | re.DOTALL,
            )
            m = pat.search(raw)
            out[lab] = None if not m else m.group(1).lower() == "true"
        return out

    async def _parse(self, resp: Any, labels: list[str]) -> dict[str, bool | None]:
        # unwrap common response containers (list-of-one, bytes, fenced blocks)
        if isinstance(resp, list) and len(resp) == 1:
            resp = resp[0]
        if isinstance(resp, (bytes, bytearray)):
            resp = resp.decode()
        data: Any | None = None
        if isinstance(resp, str):
            m = _FENCE_RE.search(resp)
            if m:
                resp = m.group(1).strip()

            data = await safest_json(resp)
        elif isinstance(resp, dict):
            data = resp
        if isinstance(data, dict):
            norm = {
                k.strip().lower(): (
                    True
                    if str(v).strip().lower() in {"true", "yes", "1"}
                    else (
                        False
                        if str(v).strip().lower() in {"false", "no", "0"}
                        else None
                    )
                )
                for k, v in data.items()
            }
            return {lab: norm.get(lab.lower(), None) for lab in labels}

        # fallback to regex extraction
        return self._regex(str(resp), labels)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str | None = None,
        *,
        circle_column_name: str | None = None,
        square_column_name: str | None = None,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Classify items and return ``df`` with label columns."""
        from ..engine.batch import batch_execute

        if self.cfg.differentiate:
            if circle_column_name is None or square_column_name is None:
                raise ValueError(
                    "circle_column_name and square_column_name are required when differentiate is True"
                )
        elif column_name is None:
            raise ValueError("column_name is required when differentiate is False")

        df_proc = df.reset_index(drop=True).copy()
        base_name = os.path.splitext(self.cfg.file_name)[0]
        csv_path = os.path.join(self.cfg.save_dir, f"{base_name}_raw_responses.csv")
        run_metadata = _common.load_run_metadata(
            self.cfg.save_dir, base_name, reset_files=reset_files
        )

        self.cfg.labels = _common.load_persisted_attributes(
            save_dir=self.cfg.save_dir,
            incoming=self.cfg.labels,
            reset_files=reset_files,
            task_name="Classify",
            item_name="labels",
            legacy_filename=f"{base_name}_attrs.json",
        )

        label_items = list(self.cfg.labels.items())
        label_batch_items, effective_n_attributes_per_run = (
            _common.resolve_attribute_batches(
                items=label_items,
                requested_n=self.cfg.n_attributes_per_run,
                metadata=run_metadata,
                reset_files=reset_files,
            )
        )
        label_batches: list[dict[str, str]] = [
            dict(batch) for batch in label_batch_items
        ]
        label_count = len(label_items)
        if (
            effective_n_attributes_per_run is not None
            and label_count > effective_n_attributes_per_run
        ):
            batches = (
                label_count + effective_n_attributes_per_run - 1
            ) // effective_n_attributes_per_run
            print(
                f"[Classify] {label_count} labels provided. n_attributes_per_run={effective_n_attributes_per_run}. "
                f"Splitting into {batches} prompt batches; set n_attributes_per_run=None to process them together."
            )
        _common.update_run_metadata(
            self.cfg.save_dir,
            base_name,
            model=self.cfg.model,
            n_attributes_per_run=effective_n_attributes_per_run,
            attribute_batches=[[name for name, _ in b] for b in label_batch_items],
        )

        prompts: list[str] = []
        ids: list[str] = []
        base_ids: list[str] = []
        id_to_circle_first: dict[str, bool] = {}
        id_to_rows: defaultdict[str, list[int]] = defaultdict(list)
        id_to_val: dict[str, Any] = {}
        prompt_texts: dict[str, str] = {}
        prompt_circles: dict[str, str] = {}
        prompt_squares: dict[str, str] = {}

        if self.cfg.differentiate:
            circles = df_proc[circle_column_name].tolist()  # type: ignore[index]
            squares = df_proc[square_column_name].tolist()  # type: ignore[index]
            for row, (circ, sq) in enumerate(zip(circles, squares)):
                clean = " ".join(str(circ).split()) + "|" + " ".join(str(sq).split())
                ident = _common.hash_identifier(clean)
                id_to_rows[ident].append(row)
                if len(id_to_rows[ident]) > 1:
                    continue
                id_to_val[ident] = (circ, sq)
                prompt_circles[ident] = (
                    circ if self.cfg.modality in {"text", "entity", "web"} else ""
                )
                prompt_squares[ident] = (
                    sq if self.cfg.modality in {"text", "entity", "web"} else ""
                )
                circle_first_flag = (
                    self.cfg.circle_first
                    if self.cfg.circle_first is not None
                    else random.random() < 0.5
                )
                id_to_circle_first[ident] = circle_first_flag
                base_ids.append(ident)
            for batch_idx, batch_labels in enumerate(label_batches):
                for ident in base_ids:
                    prompts.append(
                        self._template.render(
                            entry_circle=prompt_circles[ident],
                            entry_square=prompt_squares[ident],
                            attributes=batch_labels,
                            additional_instructions=self.cfg.additional_instructions,
                            differentiate=True,
                            modality=self.cfg.modality,
                            circle_first=id_to_circle_first[ident],
                        )
                    )
                    ids.append(f"{ident}_batch{batch_idx}")
        else:
            values = df_proc[column_name].tolist()  # type: ignore[index]
            for row, val in enumerate(values):
                clean = " ".join(str(val).split())
                ident = _common.hash_identifier(clean)
                id_to_rows[ident].append(row)
                if len(id_to_rows[ident]) > 1:
                    continue
                id_to_val[ident] = val
                prompt_texts[ident] = (
                    str(val) if self.cfg.modality in {"text", "entity", "web"} else ""
                )
                base_ids.append(ident)
            for batch_idx, batch_labels in enumerate(label_batches):
                for ident in base_ids:
                    prompts.append(
                        self._template.render(
                            text=prompt_texts[ident],
                            attributes=batch_labels,
                            additional_instructions=self.cfg.additional_instructions,
                            modality=self.cfg.modality,
                        )
                    )
                    ids.append(f"{ident}_batch{batch_idx}")

        if not isinstance(self.cfg.n_runs, int) or self.cfg.n_runs < 1:
            raise ValueError("n_runs must be an integer >= 1")

        existing_ids: set = set()
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
                    run_ids.append(
                        legacy_ident if legacy_ident in existing_ids else ident
                    )
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
            json_mode=True,
            n_parallels=self.cfg.n_parallels,
            save_path=csv_path,
            reset_files=reset_files,
            **kwargs,
        )
        if not isinstance(df_resp_all, pd.DataFrame):
            raise TypeError("batch_execute returned no DataFrame")

        df_resps = []
        for run_idx, run_ids in enumerate(run_identifier_lists, start=1):
            suffix = f"_run{run_idx}"
            sub = df_resp_all[df_resp_all.Identifier.isin(run_ids)].copy()
            sub.Identifier = sub.Identifier.str.replace(suffix + "$", "", regex=True)
            df_resps.append(sub)

        # parse each run and construct disaggregated records
        full_records: list[dict[str, Any]] = []
        total_orphans = 0
        all_labels = list(self.cfg.labels.keys())
        for run_idx, df_resp in enumerate(df_resps, start=1):
            id_to_labels: dict[str, dict[str, bool | None]] = {
                ident: {lab: None for lab in all_labels} for ident in base_ids
            }
            orphans = 0
            for ident_batch, raw in zip(df_resp.Identifier, df_resp.Response):
                if "_batch" not in ident_batch:
                    continue
                base_ident, batch_part = ident_batch.rsplit("_batch", 1)
                if base_ident not in id_to_rows:
                    orphans += 1
                    continue
                batch_idx = int(batch_part)
                labs = list(label_batches[batch_idx].keys())
                parsed = await self._parse(raw, labs)
                for lab in labs:
                    id_to_labels[base_ident][lab] = parsed.get(lab)
            total_orphans += orphans
            for ident in base_ids:
                parsed = id_to_labels.get(ident, {lab: None for lab in all_labels})
                if self.cfg.differentiate:
                    circ_val, sq_val = id_to_val[ident]
                    rec = {"circle": circ_val, "square": sq_val, "run": run_idx}
                else:
                    rec = {"text": id_to_val[ident], "run": run_idx}
                rec.update({lab: parsed.get(lab) for lab in all_labels})
                full_records.append(rec)

        if total_orphans:
            print(
                f"[Classify] WARNING: {total_orphans} response(s) had no matching passage this run."
            )

        if self.cfg.differentiate:
            full_df = pd.DataFrame(full_records).set_index(["circle", "square", "run"])
            group_cols = ["circle", "square"]
        else:
            full_df = pd.DataFrame(full_records).set_index(["text", "run"])
            group_cols = ["text"]
        if self.cfg.n_runs > 1:
            disagg_path = os.path.join(
                self.cfg.save_dir, f"{base_name}_full_disaggregated.csv"
            )
            save_dataframe_with_fallback(
                full_df,
                disagg_path,
                index=True,
                label="Classify",
            )

        # aggregate across runs using a minimum frequency threshold
        def _min_freq(s: pd.Series) -> bool | None:
            s_bool = s.astype("boolean")
            if s_bool.notna().sum() == 0:
                return None
            true_count = int(s_bool.sum(skipna=True))
            prop = true_count / self.cfg.n_runs
            return prop >= self.cfg.min_frequency

        agg_df = pd.DataFrame(
            {
                lab: full_df[lab].groupby(group_cols).apply(_min_freq)
                for lab in self.cfg.labels
            }
        )

        out_path = os.path.join(self.cfg.save_dir, f"{base_name}_cleaned.csv")
        if self.cfg.differentiate:
            result = df_proc.merge(
                agg_df,
                left_on=[circle_column_name, square_column_name],
                right_index=True,
                how="left",
            )
        else:
            result = df_proc.merge(
                agg_df, left_on=column_name, right_index=True, how="left"
            )

        label_cols = list(self.cfg.labels.keys())

        if not self.cfg.differentiate and column_name in result.columns:
            cols = result.columns.tolist()
            cols.remove(column_name)
            cols.insert(0, column_name)
            result = result[cols]

        result.insert(
            1,
            "predicted_classes",
            result[label_cols].apply(_collect_predictions, axis=1),
        )

        result_to_save = result.copy()
        result_to_save["predicted_classes"] = result_to_save["predicted_classes"].apply(
            json.dumps
        )
        save_dataframe_with_fallback(
            result_to_save,
            out_path,
            index=False,
            label="Classify",
        )

        return result
