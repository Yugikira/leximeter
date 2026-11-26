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

from ..client.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionClient,
    DecisionQuestion,
    DecisionResult,
    NoulAnswer,
    NoulQuestion,
)
from ..templates import load_prompt
from ..utils.file_utils import save_dataframe_with_fallback
from ..utils.parsing import safest_json
from . import _common
from .base import Task

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_CIRCLE_SQUARE_RE = re.compile(r"(?i)circle|square")
_CIRCLE_SQUARE_WORD_RE = re.compile(
    r"(?<![A-Za-z])(circle|square)(?![A-Za-z])", re.IGNORECASE
)


def _is_square_primary(actual_label: str, actual_desc: str) -> bool:
    """Return ``True`` when the actual label or definition describes Square over Circle."""
    for text in (actual_label, actual_desc):
        if not text:
            continue
        match = _CIRCLE_SQUARE_WORD_RE.search(str(text))
        if match is not None:
            return match.group(1).lower() == "square"
    return False


@dataclass(frozen=True)
class _DifferentiatePair:
    """One directional Circle/Square label pair evaluated as a 3-way choice."""

    question_key: str
    actual_label: str
    actual_desc: str
    inverted_label: str | None
    inverted_desc: str | None
    square_primary: bool = False

    @property
    def circle_label(self) -> str | None:
        return self.inverted_label if self.square_primary else self.actual_label

    @property
    def square_label(self) -> str | None:
        return self.actual_label if self.square_primary else self.inverted_label

    @property
    def circle_desc(self) -> str:
        raw = self.inverted_desc if self.square_primary else self.actual_desc
        return str(raw).strip() if raw is not None else ""

    @property
    def square_desc(self) -> str:
        raw = self.actual_desc if self.square_primary else self.inverted_desc
        return str(raw).strip() if raw is not None else ""


def _collect_predictions(row: pd.Series) -> list[str]:
    """Return labels whose values evaluate to ``True``."""
    return [lab for lab, val in row.items() if pd.notna(val) and bool(val)]


def _noul_instruction(label: str, definition: str, *, modality: str) -> str:
    noun = "entity" if modality == "entity" else "passage"
    desc = str(definition).strip() if definition is not None else ""
    if desc:
        return f"Does the label '{label}' apply to this {noun}? Definition: {desc}"
    return f"Does the label '{label}' apply to this {noun}?"


def _swap_circle_square(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.lower() == "circle":
            return "Square" if word[0].isupper() else "square"
        return "Circle" if word[0].isupper() else "circle"

    return _CIRCLE_SQUARE_RE.sub(_repl, text)


def _group_differentiate_labels(labels: dict[str, str]) -> list[_DifferentiatePair]:
    """Group directional label pairs into 3-way ChoiceQuestion specs.

    Recognizes three pairing conventions used across Leximeter:
    1. ``<base>_actual`` / ``<base>_inverted``
    2. ``<base>`` / ``<base> (inverted)``
    3. ``circle`` <-> ``square`` swapped label names (from :class:`Discover`)
    Any remaining unpaired label is evaluated as a solo 3-way choice.
    """
    lower_lookup = {k.lower(): k for k in labels}
    visited: set[str] = set()
    used_qkeys: set[str] = set()
    pairs: list[_DifferentiatePair] = []

    def _unique_key(candidate: str, fallback: str) -> str:
        if candidate not in used_qkeys:
            used_qkeys.add(candidate)
            return candidate
        if fallback not in used_qkeys:
            used_qkeys.add(fallback)
            return fallback
        idx = 1
        while f"{fallback}_{idx}" in used_qkeys:
            idx += 1
        chosen = f"{fallback}_{idx}"
        used_qkeys.add(chosen)
        return chosen

    for key in labels:
        if key in visited:
            continue

        actual_key: str = key
        inverted_key: str | None = None
        base_name: str = key

        if key.endswith("_actual"):
            base = key[: -len("_actual")]
            inv_cand = f"{base}_inverted"
            if inv_cand in labels and inv_cand not in visited:
                actual_key, inverted_key, base_name = key, inv_cand, base
            else:
                swapped_base = _swap_circle_square(base)
                if swapped_base.lower() != base.lower():
                    swapped_inv = f"{swapped_base}_inverted"
                    swapped_inv_key = (
                        swapped_inv
                        if swapped_inv in labels
                        else lower_lookup.get(swapped_inv.lower())
                    )
                    if swapped_inv_key is not None and swapped_inv_key not in visited:
                        actual_key, inverted_key, base_name = (
                            key,
                            swapped_inv_key,
                            base,
                        )
        elif key.endswith("_inverted"):
            base = key[: -len("_inverted")]
            act_cand = f"{base}_actual"
            if act_cand in labels and act_cand not in visited:
                actual_key, inverted_key, base_name = act_cand, key, base
            else:
                swapped_base = _swap_circle_square(base)
                if swapped_base.lower() != base.lower():
                    swapped_act = f"{swapped_base}_actual"
                    swapped_act_key = (
                        swapped_act
                        if swapped_act in labels
                        else lower_lookup.get(swapped_act.lower())
                    )
                    if swapped_act_key is not None and swapped_act_key not in visited:
                        actual_key, inverted_key, base_name = (
                            swapped_act_key,
                            key,
                            swapped_act_key[: -len("_actual")],
                        )
        elif f"{key} (inverted)" in labels and f"{key} (inverted)" not in visited:
            actual_key, inverted_key, base_name = key, f"{key} (inverted)", key
        elif key.endswith(" (inverted)"):
            base = key[: -len(" (inverted)")]
            if base in labels and base not in visited:
                actual_key, inverted_key, base_name = base, key, base
        else:
            swapped = _swap_circle_square(key)
            if swapped.lower() != key.lower():
                swapped_key = (
                    swapped if swapped in labels else lower_lookup.get(swapped.lower())
                )
                if (
                    swapped_key is not None
                    and swapped_key != key
                    and swapped_key not in visited
                ):
                    actual_key, inverted_key, base_name = key, swapped_key, key

        visited.add(actual_key)
        if inverted_key is not None:
            visited.add(inverted_key)

        pairs.append(
            _DifferentiatePair(
                question_key=_unique_key(base_name, actual_key),
                actual_label=actual_key,
                actual_desc=labels[actual_key],
                inverted_label=inverted_key,
                inverted_desc=(
                    labels[inverted_key] if inverted_key is not None else None
                ),
                square_primary=_is_square_primary(actual_key, labels[actual_key]),
            )
        )
    return pairs


def _build_differentiate_choice(
    pair: _DifferentiatePair, *, circle_first: bool
) -> ChoiceQuestion:
    circle_desc = pair.circle_desc
    square_desc = pair.square_desc
    if not circle_desc and square_desc:
        swapped = _swap_circle_square(square_desc)
        circle_desc = swapped if swapped.lower() != square_desc.lower() else ""
    if not square_desc and circle_desc:
        swapped = _swap_circle_square(circle_desc)
        square_desc = swapped if swapped.lower() != circle_desc.lower() else ""

    if pair.square_primary:
        circle_fallback = f"The label '{pair.circle_label or _swap_circle_square(pair.question_key)}' applies."
        square_fallback = (
            f"The label '{pair.square_label or pair.question_key}' applies."
        )
    else:
        circle_fallback = (
            f"Circle clearly manifests '{pair.question_key}' more than Square."
        )
        square_fallback = (
            f"Square clearly manifests '{pair.question_key}' more than Circle."
        )
    circle_text = circle_desc or circle_fallback
    square_text = square_desc or square_fallback
    neither_text = (
        "Neither applies: the label does not distinguish Circle from Square "
        "(absent from both, or equally present in both)."
    )
    if circle_first:
        criteria: dict[str, str | None] = {
            "circle": circle_text,
            "square": square_text,
            "neither": neither_text,
        }
    else:
        criteria = {
            "square": square_text,
            "circle": circle_text,
            "neither": neither_text,
        }
    instructions = (
        f"Compare Circle and Square on the label '{pair.question_key}'. "
        "Choose 'circle' if the 'circle' description applies clearly, "
        "'square' if the 'square' description applies clearly, "
        "or 'neither' if neither applies or both are equal."
    )
    return ChoiceQuestion(instructions=instructions, criteria=criteria)


def _choice_answer_probs(ans: ChoiceAnswer) -> tuple[float, float, float] | None:
    if ans.refused:
        return None
    if ans.probabilities:
        return (
            float(ans.probabilities.get("circle", 0.0)),
            float(ans.probabilities.get("square", 0.0)),
            float(ans.probabilities.get("neither", 0.0)),
        )
    if ans.choice == "circle":
        return (1.0, 0.0, 0.0)
    if ans.choice == "square":
        return (0.0, 1.0, 0.0)
    if ans.choice == "neither":
        return (0.0, 0.0, 1.0)
    return None


def _resolve_pair_batches(
    pairs: list[_DifferentiatePair],
    *,
    effective_limit: int,
    metadata: dict[str, Any],
    reset_files: bool,
    max_batch_size: int | None = None,
) -> list[list[_DifferentiatePair]]:
    """Chunk differentiate pairs while reusing any valid saved batch plan on resume."""
    if not pairs:
        return []
    limit = (
        min(effective_limit, max_batch_size)
        if max_batch_size is not None
        else effective_limit
    )
    if not reset_files and isinstance(metadata.get("attribute_batches"), list):
        pair_by_actual = {pair.actual_label: pair for pair in pairs}
        rebuilt: list[list[_DifferentiatePair]] = []
        seen_pairs: list[_DifferentiatePair] = []
        valid = True
        for saved_batch in metadata["attribute_batches"]:
            if not isinstance(saved_batch, list):
                valid = False
                break
            saved_set = set(saved_batch)
            batch_pairs: list[_DifferentiatePair] = []
            for name in saved_batch:
                pair = pair_by_actual.get(name)
                if pair is None:
                    continue
                if (
                    pair.inverted_label is not None
                    and pair.inverted_label not in saved_set
                ):
                    valid = False
                    break
                batch_pairs.append(pair)
                seen_pairs.append(pair)
            if not valid or (
                max_batch_size is not None and len(batch_pairs) > max_batch_size
            ):
                valid = False
                break
            rebuilt.append(batch_pairs)
        if valid and seen_pairs == pairs:
            return rebuilt
    return _common.chunk_items(pairs, limit)


@dataclass
class ClassifyConfig:
    labels: dict[str, str]  # {"label_name": "description", ...}
    save_dir: str = "classifier"
    file_name: str = "classify_responses.csv"
    model: str = "gpt-5.4-mini"
    decision_model: str | None = None
    n_parallels: int = 50
    n_runs: int = 1
    min_frequency: float = 0.6
    additional_instructions: str | None = None
    modality: str = "text"
    n_attributes_per_run: int | None = None
    differentiate: bool = False
    circle_first: bool | None = None
    include_probabilities: bool = False
    order_strategy: _common.OrderStrategy = "per_run"

    def __post_init__(self) -> None:
        if self.additional_instructions is not None:
            cleaned = str(self.additional_instructions).strip()
            self.additional_instructions = cleaned or None
        _common.validate_order_strategy(self.order_strategy)


class Classify(Task):
    """Robust passage classifier using an LLM."""

    _prefer_explicit_llm_client = True

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

        decision_client = self._get_decision_client()
        if decision_client is not None:
            return await self._run_with_decision_client(
                df,
                decision_client=decision_client,
                column_name=column_name,
                circle_column_name=circle_column_name,
                square_column_name=square_column_name,
                reset_files=reset_files,
                **kwargs,
            )

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
            decision_provider=False,
            n_attributes_per_run=effective_n_attributes_per_run,
            attribute_batches=[[name for name, _ in b] for b in label_batch_items],
        )

        if not isinstance(self.cfg.n_runs, int) or self.cfg.n_runs < 1:
            raise ValueError("n_runs must be an integer >= 1")

        order_strategy = _common.validate_order_strategy(self.cfg.order_strategy)

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

        existing_ids: set[str] = set()
        if not reset_files and os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path, usecols=["Identifier"])
                existing_ids = set(existing_df["Identifier"].astype(str))
            except (OSError, ValueError):
                existing_ids = set()

        # Run-major dispatch: batch -> run -> item
        prompts_all: list[str] = []
        ids_all: list[str] = []
        run_identifier_lists: list[list[str]] = [[] for _ in range(self.cfg.n_runs)]
        for batch_idx, batch_labels in enumerate(label_batches):
            for run_idx in range(1, self.cfg.n_runs + 1):
                next_labels = _common.make_mapping_orderer(
                    batch_labels,
                    strategy=order_strategy,
                    run_idx=run_idx,
                    batch_idx=batch_idx,
                )
                for ident in base_ids:
                    call_labels = next_labels()
                    ident_batch = f"{ident}_batch{batch_idx}"
                    if run_idx == 1:
                        legacy_ident = f"{ident_batch}_run1"
                        full_ident = (
                            legacy_ident
                            if legacy_ident in existing_ids
                            else ident_batch
                        )
                    else:
                        full_ident = f"{ident_batch}_run{run_idx}"
                    run_identifier_lists[run_idx - 1].append(full_ident)
                    ids_all.append(full_ident)
                    if self.cfg.differentiate:
                        prompts_all.append(
                            self._template.render(
                                entry_circle=prompt_circles[ident],
                                entry_square=prompt_squares[ident],
                                attributes=call_labels,
                                additional_instructions=self.cfg.additional_instructions,
                                differentiate=True,
                                modality=self.cfg.modality,
                                circle_first=id_to_circle_first[ident],
                            )
                        )
                    else:
                        prompts_all.append(
                            self._template.render(
                                text=prompt_texts[ident],
                                attributes=call_labels,
                                additional_instructions=self.cfg.additional_instructions,
                                modality=self.cfg.modality,
                            )
                        )

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

    # ------------------------------------------------------------------
    # Decision Provider execution path
    # ------------------------------------------------------------------
    async def _run_with_decision_client(
        self,
        df: pd.DataFrame,
        *,
        decision_client: DecisionClient,
        column_name: str | None,
        circle_column_name: str | None,
        square_column_name: str | None,
        reset_files: bool,
        **kwargs: Any,
    ) -> pd.DataFrame:
        from ..engine.batch import batch_decide

        if not isinstance(self.cfg.n_runs, int) or self.cfg.n_runs < 1:
            raise ValueError("n_runs must be an integer >= 1")
        if self.cfg.n_attributes_per_run is not None and (
            not isinstance(self.cfg.n_attributes_per_run, int)
            or self.cfg.n_attributes_per_run < 1
        ):
            raise ValueError("n_attributes_per_run must be None or an integer >= 1")

        effective_model = (
            _common.effective_decision_model(
                decision_client,
                self.cfg.decision_model,
                fallback=self.cfg.model,
            )
            or self.cfg.model
        )

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

        provider_cap = max(
            1, int(getattr(decision_client, "max_questions_per_call", 64))
        )
        effective_limit = (
            min(self.cfg.n_attributes_per_run, provider_cap)
            if self.cfg.n_attributes_per_run is not None
            else provider_cap
        )

        all_labels = list(self.cfg.labels.keys())
        row_ids: list[str] = []
        base_ids: list[str] = []
        id_to_val: dict[str, Any] = {}
        item_states: dict[str, str | dict[str, Any]] = {}
        id_to_circle_first: dict[str, bool] = {}

        if self.cfg.differentiate:
            circles = df_proc[circle_column_name].tolist()  # type: ignore[index]
            squares = df_proc[square_column_name].tolist()  # type: ignore[index]
            for circ, sq in zip(circles, squares):
                clean = " ".join(str(circ).split()) + "|" + " ".join(str(sq).split())
                ident = _common.hash_identifier(clean)
                row_ids.append(ident)
                if ident in id_to_val:
                    continue
                id_to_val[ident] = (circ, sq)
                circ_text = (
                    str(circ) if self.cfg.modality in {"text", "entity", "web"} else ""
                )
                sq_text = (
                    str(sq) if self.cfg.modality in {"text", "entity", "web"} else ""
                )
                circle_first_flag = (
                    self.cfg.circle_first
                    if self.cfg.circle_first is not None
                    else random.random() < 0.5
                )
                id_to_circle_first[ident] = circle_first_flag
                pair_state: dict[str, Any] = (
                    {"circle": circ_text, "square": sq_text}
                    if circle_first_flag
                    else {"square": sq_text, "circle": circ_text}
                )
                if self.cfg.additional_instructions:
                    pair_state["additional_instructions"] = (
                        self.cfg.additional_instructions
                    )
                item_states[ident] = pair_state
                base_ids.append(ident)
        else:
            values = df_proc[column_name].tolist()  # type: ignore[index]
            for val in values:
                clean = " ".join(str(val).split())
                ident = _common.hash_identifier(clean)
                row_ids.append(ident)
                if ident in id_to_val:
                    continue
                id_to_val[ident] = val
                text = (
                    str(val) if self.cfg.modality in {"text", "entity", "web"} else ""
                )
                if self.cfg.additional_instructions:
                    content_key = "entity" if self.cfg.modality == "entity" else "text"
                    item_states[ident] = {
                        content_key: text,
                        "additional_instructions": self.cfg.additional_instructions,
                    }
                else:
                    item_states[ident] = text
                base_ids.append(ident)

        df_proc["_gid"] = row_ids

        # Build question batches & jobs
        base_states: list[str | dict[str, Any]] = []
        base_qsets: list[dict[str, DecisionQuestion]] = []
        base_job_ids: list[str] = []

        label_batches: list[dict[str, str]] = []
        pair_batches: list[list[_DifferentiatePair]] = []

        if self.cfg.differentiate:
            pairs = _group_differentiate_labels(self.cfg.labels)
            pair_batches = _resolve_pair_batches(
                pairs,
                effective_limit=effective_limit,
                metadata=run_metadata,
                reset_files=reset_files,
                max_batch_size=provider_cap,
            )
            _common.update_run_metadata(
                self.cfg.save_dir,
                base_name,
                model=effective_model,
                decision_provider=True,
                n_attributes_per_run=effective_limit,
                attribute_batches=[
                    [
                        lab
                        for pair in batch
                        for lab in (
                            (pair.actual_label, pair.inverted_label)
                            if pair.inverted_label is not None
                            else (pair.actual_label,)
                        )
                    ]
                    for batch in pair_batches
                ],
            )
            for batch_idx, pair_batch in enumerate(pair_batches):
                for ident in base_ids:
                    c_first = id_to_circle_first[ident]
                    qmap: dict[str, DecisionQuestion] = {
                        pair.question_key: _build_differentiate_choice(
                            pair, circle_first=c_first
                        )
                        for pair in pair_batch
                    }
                    base_states.append(item_states[ident])
                    base_qsets.append(qmap)
                    base_job_ids.append(f"{ident}_batch{batch_idx}")
        else:
            label_items = list(self.cfg.labels.items())
            label_batch_items, effective_n_per_run = _common.resolve_attribute_batches(
                items=label_items,
                requested_n=effective_limit,
                metadata=run_metadata,
                reset_files=reset_files,
                max_batch_size=provider_cap,
            )
            label_batches = [dict(b) for b in label_batch_items]
            _common.update_run_metadata(
                self.cfg.save_dir,
                base_name,
                model=effective_model,
                decision_provider=True,
                n_attributes_per_run=effective_n_per_run,
                attribute_batches=[[name for name, _ in b] for b in label_batch_items],
            )
            batch_question_maps: list[dict[str, DecisionQuestion]] = [
                {
                    lab: NoulQuestion(
                        instructions=_noul_instruction(
                            lab, desc, modality=self.cfg.modality
                        )
                    )
                    for lab, desc in batch_labels.items()
                }
                for batch_labels in label_batches
            ]
            for batch_idx, qmap in enumerate(batch_question_maps):
                for ident in base_ids:
                    base_states.append(item_states[ident])
                    base_qsets.append(qmap)
                    base_job_ids.append(f"{ident}_batch{batch_idx}")

        states_all: list[str | dict[str, Any]] = []
        qsets_all: list[dict[str, DecisionQuestion]] = []
        ids_all: list[str] = []
        run_identifier_lists: list[list[str]] = []
        for run_idx in range(1, self.cfg.n_runs + 1):
            run_ids = [
                ident if run_idx == 1 else f"{ident}_run{run_idx}"
                for ident in base_job_ids
            ]
            run_identifier_lists.append(run_ids)
            states_all.extend(base_states)
            qsets_all.extend(base_qsets)
            ids_all.extend(run_ids)

        df_resp_all = await batch_decide(
            states=states_all,
            question_sets=qsets_all,
            identifiers=ids_all,
            client=decision_client,
            model=effective_model,
            n_parallels=self.cfg.n_parallels,
            save_path=csv_path,
            reset_files=reset_files,
            **kwargs,
        )
        resp_lookup = dict(zip(df_resp_all["Identifier"], df_resp_all["Response"]))

        agg_rows: list[dict[str, Any]] = []
        full_records: list[dict[str, Any]] = []

        if self.cfg.differentiate:
            all_pairs = [pair for batch in pair_batches for pair in batch]
            id_pair_runs: dict[
                str, dict[str, list[tuple[float, float, float] | None]]
            ] = {
                ident: {pair.question_key: [] for pair in all_pairs}
                for ident in base_ids
            }
            for run_idx, run_ids in enumerate(run_identifier_lists, start=1):
                run_triples: dict[str, dict[str, tuple[float, float, float] | None]] = {
                    ident: {pair.question_key: None for pair in all_pairs}
                    for ident in base_ids
                }
                for job_id, base_job_id in zip(run_ids, base_job_ids):
                    base_ident, batch_part = base_job_id.rsplit("_batch", 1)
                    batch_idx = int(batch_part)
                    parsed = DecisionResult.from_json(resp_lookup.get(job_id))
                    if parsed is None:
                        continue
                    for pair in pair_batches[batch_idx]:
                        ans = parsed.answers.get(pair.question_key)
                        if isinstance(ans, ChoiceAnswer):
                            run_triples[base_ident][pair.question_key] = (
                                _choice_answer_probs(ans)
                            )
                for ident in base_ids:
                    circ_val, sq_val = id_to_val[ident]
                    rec: dict[str, Any] = {
                        "circle": circ_val,
                        "square": sq_val,
                        "run": run_idx,
                    }
                    for pair in all_pairs:
                        triple = run_triples[ident][pair.question_key]
                        id_pair_runs[ident][pair.question_key].append(triple)
                        if triple is None:
                            if pair.circle_label is not None:
                                rec[pair.circle_label] = None
                                rec[f"{pair.circle_label}_prob"] = None
                            if pair.square_label is not None:
                                rec[pair.square_label] = None
                                rec[f"{pair.square_label}_prob"] = None
                        else:
                            p_c, p_s, p_n = triple
                            if pair.circle_label is not None:
                                rec[pair.circle_label] = bool(
                                    p_c > p_s
                                    and p_c > p_n
                                    and p_c >= self.cfg.min_frequency
                                )
                                rec[f"{pair.circle_label}_prob"] = p_c
                            if pair.square_label is not None:
                                rec[pair.square_label] = bool(
                                    p_s > p_c
                                    and p_s > p_n
                                    and p_s >= self.cfg.min_frequency
                                )
                                rec[f"{pair.square_label}_prob"] = p_s
                    full_records.append(rec)

            if self.cfg.n_runs > 1 and full_records:
                full_df = pd.DataFrame(full_records).set_index(
                    ["circle", "square", "run"]
                )
                disagg_path = os.path.join(
                    self.cfg.save_dir, f"{base_name}_full_disaggregated.csv"
                )
                save_dataframe_with_fallback(
                    full_df, disagg_path, index=True, label="Classify"
                )

            for ident in base_ids:
                bool_vals: dict[str, bool | None] = {}
                prob_vals: dict[str, float | None] = {}
                for pair in all_pairs:
                    valid_t = [
                        t
                        for t in id_pair_runs[ident][pair.question_key]
                        if t is not None
                    ]
                    if not valid_t:
                        if pair.circle_label is not None:
                            bool_vals[pair.circle_label] = None
                            prob_vals[pair.circle_label] = None
                        if pair.square_label is not None:
                            bool_vals[pair.square_label] = None
                            prob_vals[pair.square_label] = None
                    else:
                        p_c = sum(t[0] for t in valid_t) / len(valid_t)
                        p_s = sum(t[1] for t in valid_t) / len(valid_t)
                        p_n = sum(t[2] for t in valid_t) / len(valid_t)
                        if pair.circle_label is not None:
                            bool_vals[pair.circle_label] = bool(
                                p_c > p_s
                                and p_c > p_n
                                and p_c >= self.cfg.min_frequency
                            )
                            prob_vals[pair.circle_label] = p_c
                        if pair.square_label is not None:
                            bool_vals[pair.square_label] = bool(
                                p_s > p_c
                                and p_s > p_n
                                and p_s >= self.cfg.min_frequency
                            )
                            prob_vals[pair.square_label] = p_s
                row_data: dict[str, Any] = {"_gid": ident}
                for lab in all_labels:
                    row_data[lab] = bool_vals.get(lab)
                if self.cfg.include_probabilities:
                    for lab in all_labels:
                        row_data[f"{lab}_prob"] = prob_vals.get(lab)
                agg_rows.append(row_data)
        else:
            id_run_probs: dict[str, dict[str, list[float | None]]] = {
                ident: {lab: [] for lab in all_labels} for ident in base_ids
            }
            for run_idx, run_ids in enumerate(run_identifier_lists, start=1):
                run_lookup: dict[str, dict[str, float | None]] = {
                    ident: {lab: None for lab in all_labels} for ident in base_ids
                }
                for job_id, base_job_id in zip(run_ids, base_job_ids):
                    base_ident, batch_part = base_job_id.rsplit("_batch", 1)
                    batch_idx = int(batch_part)
                    parsed = DecisionResult.from_json(resp_lookup.get(job_id))
                    if parsed is None:
                        continue
                    for lab in label_batches[batch_idx]:
                        ans = parsed.answers.get(lab)
                        if (
                            isinstance(ans, NoulAnswer)
                            and not ans.refused
                            and ans.probability is not None
                        ):
                            run_lookup[base_ident][lab] = float(ans.probability)
                for ident in base_ids:
                    rec = {"text": id_to_val[ident], "run": run_idx}
                    for lab in all_labels:
                        prob = run_lookup[ident][lab]
                        id_run_probs[ident][lab].append(prob)
                        rec[lab] = (
                            None
                            if prob is None
                            else bool(prob >= self.cfg.min_frequency)
                        )
                        rec[f"{lab}_prob"] = prob
                    full_records.append(rec)

            if self.cfg.n_runs > 1 and full_records:
                full_df = pd.DataFrame(full_records).set_index(["text", "run"])
                disagg_path = os.path.join(
                    self.cfg.save_dir, f"{base_name}_full_disaggregated.csv"
                )
                save_dataframe_with_fallback(
                    full_df, disagg_path, index=True, label="Classify"
                )

            for ident in base_ids:
                row_data = {"_gid": ident}
                probs_for_row: dict[str, float | None] = {}
                for lab in all_labels:
                    valid_p = [p for p in id_run_probs[ident][lab] if p is not None]
                    if not valid_p:
                        row_data[lab] = None
                        probs_for_row[lab] = None
                    else:
                        mean_prob = sum(valid_p) / len(valid_p)
                        row_data[lab] = bool(mean_prob >= self.cfg.min_frequency)
                        probs_for_row[lab] = mean_prob
                if self.cfg.include_probabilities:
                    for lab in all_labels:
                        row_data[f"{lab}_prob"] = probs_for_row[lab]
                agg_rows.append(row_data)

        agg_cols = ["_gid", *all_labels] + (
            [f"{lab}_prob" for lab in all_labels]
            if self.cfg.include_probabilities
            else []
        )
        agg_df = pd.DataFrame(agg_rows, columns=agg_cols)
        for lab in all_labels:
            if not agg_df.empty and agg_df[lab].notna().all():
                agg_df[lab] = agg_df[lab].astype(bool)

        result = df_proc.merge(agg_df, on="_gid", how="left").drop(columns=["_gid"])
        for lab in all_labels:
            if not result.empty and result[lab].notna().all():
                result[lab] = result[lab].astype(bool)

        if not self.cfg.differentiate and column_name in result.columns:
            cols = result.columns.tolist()
            cols.remove(column_name)
            cols.insert(0, column_name)
            result = result[cols]

        result.insert(
            1,
            "predicted_classes",
            result[all_labels].apply(_collect_predictions, axis=1),
        )

        out_path = os.path.join(self.cfg.save_dir, f"{base_name}_cleaned.csv")
        result_to_save = result.copy()
        result_to_save["predicted_classes"] = result_to_save["predicted_classes"].apply(
            json.dumps
        )
        save_dataframe_with_fallback(
            result_to_save, out_path, index=False, label="Classify"
        )
        return result
