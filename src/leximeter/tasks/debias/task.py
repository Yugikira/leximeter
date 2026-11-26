"""Debias: remove inference bias from a measurement.

Ported from GABRIEL's ``tasks/debias.py``.  The pipeline measures an
attribute on the original text, removes the confounding signal (codify
snippets stripped deterministically, or paraphrase rewrites), re-measures
the stripped variants, measures the remaining signal, and then computes:

* **Simple difference debiasing** — ``{attr} (raw - stripped)``
* **Two-step debiasing** — ``{attr} (debiased) = y - s + delta * r`` where
  ``delta`` is the stage-1 regression coefficient of the stripped score on
  the remaining-signal proxy.

Regression diagnostics use :func:`leximeter.utils.regression.fit_ols`
(statsmodels with robust HC3 covariance, HC1 fallback, or a NumPy solver).
Results are saved to ``debias_results.csv`` and ``debias_metadata.json``
under ``save_dir/run_name``.

Differences from GABRIEL: the provider is injected via ``client=`` instead
of ``response_fn``/``use_dummy`` (those kwargs are ignored when passed), and
the binned-scatter PDF plots are skipped (returning ``None``).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import warnings
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

import numpy as np
import pandas as pd

from ...utils.file_utils import save_dataframe_with_fallback
from ...utils.regression import _safe_fit_ols
from .. import _common
from ..base import _UNSET, Task, _UnsetType
from ..classify import Classify, ClassifyConfig
from ..codify import Codify, CodifyConfig
from ..extract import Extract, ExtractConfig
from ..paraphrase import Paraphrase, ParaphraseConfig
from ..rank import Rank, RankConfig
from ..rate import Rate, RateConfig

RemovalMethod = Literal["codify", "paraphrase"]
MeasurementMode = Literal["rate", "classify", "extract", "rank"]


def _dump_json(path: str, payload: dict, *, indent: int | None = None) -> None:
    """Write JSON to disk (sync; run via ``asyncio.to_thread``)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)


def _to_float_series(series: pd.Series) -> pd.Series:
    """Coerce numeric or boolean values to float64 (safe for subtraction & OLS)."""
    return pd.to_numeric(series, errors="coerce").astype(float)


@dataclass
class DebiasRegressionResult:
    """Container for regression diagnostics of a debiasing run."""

    variant: str
    display_name: str
    strip_percentage: int | None
    correlation: float | None
    mean_original: float | None
    mean_stripped: float | None
    attenuation_pct: float | None = None
    diff_regression: dict[str, Any] | None = None
    twostep_regression: dict[str, Any] | None = None
    stage1_regression: dict[str, Any] | None = None
    stage1_delta: float | None = None
    debiased_columns: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable representation of the result."""

        def _convert(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            if isinstance(value, np.generic):
                return value.item()
            return value

        return {
            "variant": self.variant,
            "display_name": self.display_name,
            "strip_percentage": self.strip_percentage,
            "correlation": _convert(self.correlation),
            "mean_original": _convert(self.mean_original),
            "mean_stripped": _convert(self.mean_stripped),
            "attenuation_pct": _convert(self.attenuation_pct),
            "diff_regression": _convert(self.diff_regression),
            "twostep_regression": _convert(self.twostep_regression),
            "stage1_regression": _convert(self.stage1_regression),
            "stage1_delta": _convert(self.stage1_delta),
            "debiased_columns": _convert(self.debiased_columns),
        }


@dataclass
class DebiasConfig:
    """Configuration for :class:`Debias`."""

    mode: MeasurementMode = "rate"
    measurement_attribute: str | None = None
    removal_attribute: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    signal_dictionary: dict[str, str] = field(default_factory=dict)
    removal_method: RemovalMethod = "codify"
    remaining_signal: bool = True
    remaining_signal_attribute: str | None = None
    remaining_signal_description: str | None = None
    save_dir: str = "debias"
    run_name: str | None = None
    strip_percentages: list[int] | None = None
    categories_to_strip: list[str] | None = None
    model: str = "gpt-5.4-mini"
    decision_model: str | None = None
    n_parallels: int = 50
    measurement_kwargs: dict[str, Any] = field(default_factory=dict)
    removal_kwargs: dict[str, Any] = field(default_factory=dict)
    robust_regression: bool = True
    random_seed: int = 12345
    verbose: bool = True


@dataclass
class DebiasResult:
    """Return object for :func:`leximeter.debias`."""

    results: pd.DataFrame
    metadata: dict[str, Any]
    regression: dict[str, DebiasRegressionResult]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result metadata for convenient inspection."""

        return {
            "metadata": self.metadata,
            "regression": {k: v.as_dict() for k, v in self.regression.items()},
        }


class Debias(Task):
    """Coordinate debiasing runs that rely on core leximeter tasks."""

    def __init__(self, cfg: DebiasConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._validate_config()
        base_dir = os.path.expandvars(os.path.expanduser(self.cfg.save_dir))
        os.makedirs(base_dir, exist_ok=True)
        run_name = self.cfg.run_name or self._default_run_name()
        self.run_dir = os.path.join(base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _default_run_name(self) -> str:
        base_name = (
            self.cfg.measurement_attribute or self.cfg.removal_attribute or "signal"
        )
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", base_name)
        prefix = "debias"
        cleaned = cleaned.strip("_")
        if cleaned:
            return f"{prefix}_{cleaned}"
        return prefix

    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        valid_modes = {"rate", "classify", "extract", "rank"}
        if self.cfg.mode not in valid_modes:
            raise ValueError(
                "mode must be one of {'rate', 'classify', 'extract', 'rank'}"
            )
        self.cfg.attributes = dict(self.cfg.attributes or {})
        if not self.cfg.attributes:
            raise ValueError("attributes must be supplied for the selected mode")

        self.cfg.signal_dictionary = dict(self.cfg.signal_dictionary or {})
        if not self.cfg.signal_dictionary:
            raise ValueError("signal_dictionary must describe the signal to remove")
        attr_keys = list(self.cfg.attributes.keys())
        first_attr = attr_keys[0]
        measurement_attr = self.cfg.measurement_attribute
        if measurement_attr is not None and measurement_attr not in self.cfg.attributes:
            raise ValueError(
                f"Measurement attribute '{measurement_attr}' must be a key in attributes"
            )
        if measurement_attr is None:
            measurement_attr = first_attr
            if self.cfg.verbose:
                msg = (
                    "[Debias] measurement_attribute not provided; "
                    f"defaulting to '{measurement_attr}'."
                )
                if len(attr_keys) > 1:
                    msg += " Debiasing will use the first attribute provided."
                print(msg)
        self.cfg.measurement_attribute = measurement_attr

        removal_attr = self.cfg.removal_attribute
        if removal_attr is None:
            if measurement_attr in self.cfg.signal_dictionary:
                removal_attr = measurement_attr
                if self.cfg.verbose:
                    print(
                        "[Debias] removal_attribute not provided; "
                        f"defaulting to measurement attribute '{removal_attr}'."
                    )
            else:
                removal_attr = next(iter(self.cfg.signal_dictionary))
                if self.cfg.verbose:
                    print(
                        "[Debias] removal_attribute not provided; "
                        f"defaulting to '{removal_attr}'."
                    )
        elif removal_attr not in self.cfg.signal_dictionary:
            raise ValueError(
                f"Removal attribute '{removal_attr}' must be a key in signal_dictionary"
            )
        self.cfg.removal_attribute = removal_attr
        if self.cfg.removal_method not in {"codify", "paraphrase"}:
            raise ValueError("removal_method must be 'codify' or 'paraphrase'")

        if not bool(self.cfg.remaining_signal):
            self.cfg.remaining_signal_attribute = None
            self.cfg.remaining_signal_description = None
        else:
            attr_name = (self.cfg.remaining_signal_attribute or "").strip()
            if not attr_name:
                default_attr = f"prevalence of {self.cfg.removal_attribute}"
                default_desc = (
                    "Prevalence of any direct mentions or allusions to concepts related to "
                    f"{self.cfg.removal_attribute} anywhere in the text."
                )
                self.cfg.remaining_signal_attribute = default_attr
                self.cfg.remaining_signal_description = default_desc
                if self.cfg.verbose:
                    print(
                        "[Debias] remaining_signal_attribute not provided; "
                        f"defaulting to '{default_attr}'."
                    )
            elif not self.cfg.remaining_signal_description:
                raise ValueError(
                    "remaining_signal_description must be provided when remaining_signal_attribute is set"
                )

        if self.cfg.categories_to_strip is None:
            if self.cfg.removal_method == "codify":
                self.cfg.categories_to_strip = [self.cfg.removal_attribute]
            else:
                self.cfg.categories_to_strip = []
        else:
            categories: list[str] = []
            for name in self.cfg.categories_to_strip:
                if name in self.cfg.signal_dictionary and name not in categories:
                    categories.append(name)
            if self.cfg.removal_method == "codify" and not categories:
                raise ValueError(
                    "categories_to_strip must contain at least one key from signal_dictionary when using codify"
                )
            self.cfg.categories_to_strip = categories

        if self.cfg.strip_percentages is None:
            percentages: list[int] = [100]
        else:
            percentages = []
            for pct in self.cfg.strip_percentages:
                try:
                    val = int(pct)
                except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                    raise ValueError(f"Invalid strip percentage: {pct!r}") from exc
                val = max(0, min(100, val))
                if val not in percentages:
                    percentages.append(val)
            if not percentages:
                percentages = [100]
        percentages.sort()
        self.cfg.strip_percentages = percentages

        self.cfg.measurement_kwargs = dict(self.cfg.measurement_kwargs or {})
        self.cfg.removal_kwargs = dict(self.cfg.removal_kwargs or {})

    # ------------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
    ) -> DebiasResult:
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")

        decision_client = self._get_decision_client()
        effective_decision_model = _common.effective_decision_model(
            decision_client,
            self.cfg.decision_model,
            fallback=self.cfg.model,
        )

        # Input signature covers the whole input frame (the result embeds
        # every input column) plus every config field that shapes the run
        # (including the effective decision model / provider mode); a
        # completed run with the same signature is reloaded from disk
        # (zero provider calls) instead of re-measuring.
        cfg_for_sig = {
            k: v
            for k, v in asdict(self.cfg).items()
            if k not in ("save_dir", "run_name")
        }
        cfg_for_sig["decision_model"] = effective_decision_model
        signature = _common.compute_input_signature(
            df,
            column_name,
            cfg_for_sig,
        )
        if reset_files:
            _common.clear_task_state(self.run_dir, "debias")
        else:
            loaded = self._load_completed(signature)
            if loaded is not None:
                return loaded

        df_master = df.copy().reset_index(drop=True)
        row_index = pd.RangeIndex(
            start=0, stop=len(df_master), step=1, name="__debias_row_id"
        )
        df_master.index = row_index

        if self.cfg.verbose:
            print(f"[Debias] Running debiasing pipeline on {len(df_master)} rows.")

        attr_keys = list(self.cfg.attributes.keys())
        if self.cfg.verbose and len(attr_keys) > 1:
            print(
                "[Debias] Multiple attributes detected; debiasing will focus on "
                f"'{self.cfg.measurement_attribute}' while measuring all attributes "
                "on both raw and stripped text."
            )

        existing_raw_attrs = [attr for attr in attr_keys if attr in df_master.columns]
        missing_raw_attrs = [
            attr for attr in attr_keys if attr not in df_master.columns
        ]

        if existing_raw_attrs and self.cfg.verbose:
            print(
                "[Debias] Using existing raw measurement columns: "
                + ", ".join(existing_raw_attrs)
            )

        if missing_raw_attrs:
            if self.cfg.verbose:
                print(
                    "[Debias] Measuring baseline signals for missing attributes: "
                    + ", ".join(missing_raw_attrs)
                )
            base_measure = await self._run_measurement(
                df_master,
                column_name=column_name,
                mode=self.cfg.mode,
                save_label="original",
                attributes={k: self.cfg.attributes[k] for k in missing_raw_attrs},
                extra_kwargs=self.cfg.measurement_kwargs,
                default_model=self.cfg.model,
                reset_files=reset_files,
            )
            self._attach_measurement(
                df_master,
                base_measure,
                missing_raw_attrs,
                variant_key="original",
                display_name="original",
            )
        elif self.cfg.verbose:
            print(
                "[Debias] All raw measurement attributes already exist; skipping baseline rating step."
            )

        if self.cfg.removal_method == "codify":
            variant_info = await self._prepare_codify_variants(
                df_master, column_name, reset_files=reset_files
            )
        else:
            variant_info = await self._prepare_paraphrase_variant(
                df_master, column_name, reset_files=reset_files
            )

        if variant_info and self.cfg.verbose:
            print("[Debias] Measuring stripped variants...")

        variant_count = len(variant_info)
        for info in variant_info.values():
            info["label_suffix"] = self._variant_label_suffix(info, variant_count)

        for key in list(variant_info.keys()):
            info = variant_info[key]
            measure_df = await self._run_measurement(
                df_master,
                column_name=info["text_column"],
                mode=self.cfg.mode,
                save_label=key,
                attributes=self.cfg.attributes,
                extra_kwargs=self.cfg.measurement_kwargs,
                default_model=self.cfg.model,
                reset_files=reset_files,
            )
            column_map = self._attach_measurement(
                df_master,
                measure_df,
                attr_keys,
                variant_key=key,
                display_name=info["label_suffix"],
            )
            info["measurement_columns"] = column_map
            self._print_average_attenuation(
                df_master,
                attr_keys,
                column_map,
                label_suffix=info["label_suffix"],
                display_name=info["display"],
            )

        regression_info: dict[str, DebiasRegressionResult] = {}
        for key, info in variant_info.items():
            stripped_column = info["measurement_columns"].get(
                self.cfg.measurement_attribute
            )
            if not stripped_column:
                continue
            remaining_signal_column = await self._measure_remaining_signal(
                df_master,
                info=info,
                save_label=key,
                label_suffix=info["label_suffix"],
                reset_files=reset_files,
            )
            summary = self._run_debiasing(
                df_master,
                original_column=self.cfg.measurement_attribute,
                stripped_column=stripped_column,
                remaining_signal_column=remaining_signal_column,
                variant_key=key,
                display_name=info["display"],
                strip_percentage=info.get("strip_percentage"),
                label_suffix=info["label_suffix"],
            )
            regression_info[key] = summary

        results_path = os.path.join(self.run_dir, "debias_results.csv")
        metadata: dict[str, Any] = {
            "config": self._serialise_config(),
            "variants": [regression_info[k].as_dict() for k in regression_info],
            "result_path": results_path,
        }
        metadata_path = os.path.join(self.run_dir, "debias_metadata.json")
        await asyncio.to_thread(_dump_json, metadata_path, metadata, indent=2)
        metadata["metadata_path"] = metadata_path

        results_df = df_master.reset_index(drop=True)
        save_dataframe_with_fallback(
            results_df, results_path, index=False, label="Debias"
        )
        if self.cfg.verbose:
            print(f"[Debias] Results saved to {results_path}")

        _common.save_task_state(
            self.run_dir,
            "debias",
            input_signature=signature,
            result_artifacts=["debias_results.csv", "debias_metadata.json"],
        )

        return DebiasResult(
            results=results_df,
            metadata=metadata,
            regression=regression_info,
        )

    # ------------------------------------------------------------------
    def _load_completed(self, signature: str) -> DebiasResult | None:
        """Rebuild the result when inputs match and the run is complete."""
        state = _common.load_task_state(self.run_dir, "debias")
        if (
            not state
            or state.get("status") != "complete"
            or state.get("input_signature") != signature
        ):
            return None
        results_path = os.path.join(self.run_dir, "debias_results.csv")
        metadata_path = os.path.join(self.run_dir, "debias_metadata.json")
        if not os.path.exists(results_path) or not os.path.exists(metadata_path):
            return None
        try:
            results_df = pd.read_csv(results_path)
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(metadata, dict) or results_df.empty:
            return None
        regression: dict[str, DebiasRegressionResult] = {}
        try:
            for variant in metadata.get("variants") or []:
                if isinstance(variant, dict) and "variant" in variant:
                    regression[variant["variant"]] = DebiasRegressionResult(**variant)
        except (TypeError, KeyError, ValueError):
            # metadata from an incompatible format: fall back to a full rerun
            return None
        # mirror the fresh-run in-memory metadata, which carries its path
        metadata = dict(metadata)
        metadata["metadata_path"] = metadata_path
        if self.cfg.verbose:
            print(
                f"[Debias] Found completed result in {results_path}; "
                f"loading {len(results_df)} rows."
            )
        return DebiasResult(
            results=results_df,
            metadata=metadata,
            regression=regression,
        )

    # ------------------------------------------------------------------
    async def _run_measurement(
        self,
        df: pd.DataFrame,
        *,
        column_name: str,
        mode: MeasurementMode,
        save_label: str,
        attributes: dict[str, str] | None,
        extra_kwargs: dict[str, Any] | None,
        default_model: str,
        reset_files: bool,
    ) -> pd.DataFrame:
        kwargs = dict(extra_kwargs or {})
        # GABRIEL injects providers via response_fn/use_dummy; leximeter uses client=.
        kwargs.pop("response_fn", None)
        kwargs.pop("get_all_responses_fn", None)
        kwargs.pop("use_dummy", None)
        kwargs.pop("template_path", None)
        save_dir = kwargs.pop("save_dir", os.path.join(self.run_dir, save_label))
        os.makedirs(save_dir, exist_ok=True)
        reset_files_local = bool(kwargs.pop("reset_files", False)) or bool(reset_files)
        df_reset = df[[column_name]].reset_index()
        if mode == "rate":
            kwargs.pop("decision_model", None)
            kwargs.pop("decision_client", None)
            rate_cfg = RateConfig(
                attributes=attributes or {},
                save_dir=save_dir,
                model=kwargs.pop("model", default_model),
                n_parallels=kwargs.pop("n_parallels", self.cfg.n_parallels),
                **kwargs,
            )
            rate_runner = Rate(rate_cfg, client=self._get_client())
            result = await rate_runner.run(
                df_reset,
                column_name,
                reset_files=reset_files_local,
            )
        elif mode == "classify":
            dc_override = kwargs.pop("decision_client", _UNSET)
            decision_client = (
                self._get_decision_client()
                if isinstance(dc_override, _UnsetType)
                else dc_override
            )
            model_local = kwargs.pop("model", default_model)
            decision_model_local = kwargs.pop("decision_model", self.cfg.decision_model)
            classify_cfg = ClassifyConfig(
                labels=attributes or {},
                save_dir=save_dir,
                model=model_local,
                decision_model=decision_model_local,
                n_parallels=kwargs.pop("n_parallels", self.cfg.n_parallels),
                **kwargs,
            )
            base_name = os.path.splitext(classify_cfg.file_name)[0]
            expected_model = (
                _common.effective_decision_model(
                    decision_client, decision_model_local, fallback=model_local
                )
                or model_local
            )
            reset_files_local = _common.should_reset_substage_checkpoint(
                save_dir,
                base_name,
                expected_model=expected_model,
                decision_provider=decision_client is not None,
                reset_files=reset_files_local,
            )
            classify_runner = Classify(
                classify_cfg,
                client=self._get_client(),
                decision_client=decision_client,
            )
            result = await classify_runner.run(
                df_reset,
                column_name,
                reset_files=reset_files_local,
            )
        elif mode == "extract":
            kwargs.pop("decision_model", None)
            kwargs.pop("decision_client", None)
            extract_cfg = ExtractConfig(
                attributes=attributes or {},
                save_dir=save_dir,
                model=kwargs.pop("model", default_model),
                n_parallels=kwargs.pop("n_parallels", self.cfg.n_parallels),
                **kwargs,
            )
            extract_runner = Extract(extract_cfg, client=self._get_client())
            result = await extract_runner.run(
                df_reset,
                column_name,
                reset_files=reset_files_local,
            )
        elif mode == "rank":
            dc_override = kwargs.pop("decision_client", _UNSET)
            decision_client = (
                self._get_decision_client()
                if isinstance(dc_override, _UnsetType)
                else dc_override
            )
            model_local = kwargs.pop("model", default_model)
            decision_model_local = kwargs.pop("decision_model", self.cfg.decision_model)
            rank_cfg = RankConfig(
                attributes=attributes or {},
                save_dir=save_dir,
                model=model_local,
                decision_model=decision_model_local,
                n_parallels=kwargs.pop("n_parallels", self.cfg.n_parallels),
                **kwargs,
            )
            base_name = os.path.splitext(rank_cfg.file_name)[0]
            expected_model = (
                _common.effective_decision_model(
                    decision_client, decision_model_local, fallback=model_local
                )
                or model_local
            )
            reset_files_local = _common.should_reset_substage_checkpoint(
                save_dir,
                base_name,
                expected_model=expected_model,
                decision_provider=decision_client is not None,
                reset_files=reset_files_local,
            )
            rank_runner = Rank(
                rank_cfg,
                client=self._get_client(),
                decision_client=decision_client,
            )
            result = await rank_runner.run(
                df_reset,
                column_name,
                reset_files=reset_files_local,
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported mode: {mode}")

        if "__debias_row_id" in result.columns:
            result = result.set_index("__debias_row_id")
        elif "identifier" in result.columns:
            result = result.set_index("identifier")
        else:
            result.index = df.index
        result.index.name = "__debias_row_id"
        return result

    # ------------------------------------------------------------------
    def _attach_measurement(
        self,
        df_master: pd.DataFrame,
        measurement_df: pd.DataFrame,
        attributes: list[str],
        *,
        variant_key: str,
        display_name: str,
    ) -> dict[str, str]:
        column_map: dict[str, str] = {}
        for attr in attributes:
            if attr not in measurement_df.columns:
                continue
            if variant_key == "original":
                target_name = attr
            else:
                target_name = f"{attr} ({display_name})"
            df_master[target_name] = measurement_df[attr].reindex(df_master.index)
            column_map[attr] = target_name
        return column_map

    # ------------------------------------------------------------------
    def _remaining_signal_attributes(self) -> dict[str, str]:
        if not self.cfg.remaining_signal_attribute:
            return {}
        return {
            self.cfg.remaining_signal_attribute: str(
                self.cfg.remaining_signal_description
            )
        }

    # ------------------------------------------------------------------
    async def _measure_remaining_signal(
        self,
        df_master: pd.DataFrame,
        *,
        info: dict[str, Any],
        save_label: str,
        label_suffix: str,
        reset_files: bool,
    ) -> str | None:
        attrs = self._remaining_signal_attributes()
        if not attrs:
            return None
        measurement_attr = self.cfg.remaining_signal_attribute
        assert measurement_attr is not None
        target_column = f"{measurement_attr} ({label_suffix})"
        if target_column in df_master.columns:
            if self.cfg.verbose:
                print(
                    f"[Debias] Using existing remaining-signal column: {target_column}"
                )
            return target_column
        if self.cfg.verbose:
            print(
                "[Debias] Measuring remaining-signal prevalence on "
                f"variant '{info['display']}'."
            )
        rate_allowed = {f.name for f in fields(RateConfig)} | {"reset_files"}
        remaining_kwargs = {
            k: v
            for k, v in (self.cfg.measurement_kwargs or {}).items()
            if k in rate_allowed
        }
        remaining_df = await self._run_measurement(
            df_master,
            column_name=info["text_column"],
            mode="rate",
            save_label=f"{save_label}_remaining_signal",
            attributes=attrs,
            extra_kwargs=remaining_kwargs,
            default_model=self.cfg.model,
            reset_files=reset_files,
        )
        column_map = self._attach_measurement(
            df_master,
            remaining_df,
            [measurement_attr],
            variant_key=f"{save_label}_remaining_signal",
            display_name=label_suffix,
        )
        return column_map.get(measurement_attr, target_column)

    # ------------------------------------------------------------------
    def _variant_label_suffix(self, info: dict[str, Any], variant_count: int) -> str:
        pct = info.get("strip_percentage")
        if pct is None:
            return "stripped"
        if pct == 100 and variant_count == 1:
            return "stripped"
        return f"stripped {pct}%"

    # ------------------------------------------------------------------
    async def _prepare_codify_variants(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool,
    ) -> dict[str, dict[str, Any]]:
        kwargs = dict(self.cfg.removal_kwargs or {})
        kwargs.pop("response_fn", None)
        kwargs.pop("get_all_responses_fn", None)
        kwargs.pop("use_dummy", None)
        kwargs.pop("template_path", None)
        reset_files_local = bool(kwargs.pop("reset_files", False)) or bool(reset_files)
        save_dir = kwargs.pop("save_dir", os.path.join(self.run_dir, "codify"))
        os.makedirs(save_dir, exist_ok=True)
        additional_instructions = kwargs.pop("additional_instructions", "")
        kwargs.setdefault("n_rounds", 3)
        dc_override = kwargs.pop("decision_client", _UNSET)
        decision_client = (
            self._get_decision_client()
            if isinstance(dc_override, _UnsetType)
            else dc_override
        )
        cfg = CodifyConfig(
            save_dir=save_dir,
            model=kwargs.pop("model", self.cfg.model),
            decision_model=kwargs.pop("decision_model", self.cfg.decision_model),
            n_parallels=kwargs.pop("n_parallels", self.cfg.n_parallels),
            **kwargs,
        )
        runner = Codify(
            cfg,
            client=self._get_client(),
            decision_client=decision_client,
        )
        codify_df = await runner.run(
            df.reset_index(),
            column_name,
            categories=self.cfg.signal_dictionary,
            additional_instructions=additional_instructions,
            reset_files=reset_files_local,
        )
        variants: dict[str, dict[str, Any]] = {}
        categories = self.cfg.categories_to_strip or []
        for pct in self.cfg.strip_percentages:
            if pct <= 0:
                continue
            key = f"stripped_{pct:03d}pct"
            stripped_base = f"{column_name} (stripped)"
            single_full_strip = (
                len(self.cfg.strip_percentages or []) == 1 and pct == 100
            )
            display = "stripped" if single_full_strip else f"stripped {pct}%"
            new_col = (
                stripped_base if single_full_strip else f"{column_name} ({display})"
            )
            df[new_col] = [
                self._strip_passages(
                    original_text=str(df.at[idx, column_name]),
                    snippets=self._collect_snippets(codify_df, idx, categories),
                    pct_strip=pct,
                    row_idx=int(idx),
                )
                for idx in df.index
            ]
            variants[key] = {
                "text_column": new_col,
                "display": display,
                "strip_percentage": pct,
            }
        return variants

    # ------------------------------------------------------------------
    async def _prepare_paraphrase_variant(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool,
    ) -> dict[str, dict[str, Any]]:
        kwargs = dict(self.cfg.removal_kwargs or {})
        kwargs.pop("response_fn", None)
        kwargs.pop("get_all_responses_fn", None)
        kwargs.pop("use_dummy", None)
        kwargs.pop("template_path", None)
        reset_files_local = bool(kwargs.pop("reset_files", False)) or bool(reset_files)
        save_dir = kwargs.pop("save_dir", os.path.join(self.run_dir, "paraphrase"))
        os.makedirs(save_dir, exist_ok=True)
        revised_name = f"{column_name} (stripped)"
        instructions = (
            kwargs.pop("instructions", None) or self._build_paraphrase_instructions()
        )
        response_kwargs: dict[str, Any] = {}
        if "n_rounds" in kwargs:
            response_kwargs["n_rounds"] = kwargs.pop("n_rounds")
        if "completion_max_rounds" in kwargs and "n_rounds" not in response_kwargs:
            replacement = kwargs.pop("completion_max_rounds")
            warnings.warn(
                "completion_max_rounds is deprecated; use n_rounds instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if replacement is not None:
                response_kwargs["n_rounds"] = replacement
        response_kwargs.setdefault("n_rounds", 3)
        dc_override = kwargs.pop("decision_client", _UNSET)
        decision_client = (
            self._get_decision_client()
            if isinstance(dc_override, _UnsetType)
            else dc_override
        )
        cfg = ParaphraseConfig(
            instructions=instructions,
            revised_column_name=revised_name,
            save_dir=save_dir,
            model=kwargs.pop("model", self.cfg.model),
            decision_model=kwargs.pop("decision_model", self.cfg.decision_model),
            n_parallels=kwargs.pop("n_parallels", self.cfg.n_parallels),
            **kwargs,
        )
        runner = Paraphrase(
            cfg,
            client=self._get_client(),
            decision_client=decision_client,
        )
        paraphrased = await runner.run(
            df.reset_index(),
            column_name,
            reset_files=reset_files_local,
            **response_kwargs,
        )
        df[revised_name] = paraphrased[revised_name].reindex(df.index)
        return {
            "paraphrase": {
                "text_column": revised_name,
                "display": "stripped",
                "strip_percentage": None,
            }
        }

    # ------------------------------------------------------------------
    def _build_paraphrase_instructions(self) -> str:
        lines = [
            "Rewrite the passage so that every reference to the following signal(s) is removed while preserving all other content:",
        ]
        for key, desc in self.cfg.signal_dictionary.items():
            lines.append(f"- {key}: {desc}")
        lines.append(
            "Keep the remaining content identical. Be exhaustive: the final output must have absolutely no content which references, mentions, manifests, or alludes to the aformentioned signal(s). All such content must be stripped from the text; ensure nothing at all relevant remains. When in doubt, remove it. The stripping must be perfect; don't output anything related to / manifesting the signal(s). Your output is the original text, with everything else unchanged, but all this content completely and entirely removed."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _collect_snippets(
        self,
        codify_df: pd.DataFrame,
        row_idx: int,
        categories: list[str],
    ) -> list[str]:
        snippets: list[str] = []
        for cat in categories:
            if cat not in codify_df.columns:
                continue
            raw = codify_df.at[row_idx, cat]
            if isinstance(raw, list):
                snippets.extend(str(s) for s in raw if s)
            elif isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        snippets.extend(str(s) for s in parsed if s)
                except Exception:  # noqa: BLE001 - malformed output falls back to raw
                    if raw:
                        snippets.append(str(raw))
        return snippets

    # ------------------------------------------------------------------
    def _strip_passages(
        self,
        *,
        original_text: str,
        snippets: list[str],
        pct_strip: int,
        row_idx: int,
    ) -> str:
        if pct_strip <= 0 or not snippets:
            return self._normalise_ws(original_text)
        unique: list[str] = []
        seen = set()
        for snippet in snippets:
            snippet = snippet.strip()
            if not snippet:
                continue
            if snippet not in seen:
                seen.add(snippet)
                unique.append(snippet)
        if not unique:
            return self._normalise_ws(original_text)
        n_total = len(unique)
        n_remove = max(0, min(n_total, round(n_total * (pct_strip / 100.0))))
        if n_remove == 0:
            return self._normalise_ws(original_text)
        seed_val = f"{self.cfg.random_seed}:{row_idx}:{pct_strip}:{n_total}"
        rng = random.Random()
        rng.seed(seed_val)
        to_remove = rng.sample(unique, n_remove)
        cleaned = original_text
        for snippet in sorted(to_remove, key=len, reverse=True):
            cleaned = cleaned.replace(snippet, " ")
        return self._normalise_ws(cleaned)

    # ------------------------------------------------------------------
    def _run_debiasing(
        self,
        df: pd.DataFrame,
        original_column: str,
        stripped_column: str,
        *,
        remaining_signal_column: str | None,
        variant_key: str,
        display_name: str,
        strip_percentage: int | None,
        label_suffix: str,
    ) -> DebiasRegressionResult:
        cols = [original_column, stripped_column]
        if remaining_signal_column:
            cols.append(remaining_signal_column)
        measurement_attr = self.cfg.measurement_attribute
        reg_df = df[cols].apply(_to_float_series)
        reg_df = reg_df.dropna(subset=[original_column, stripped_column])
        if len(reg_df) < 3:
            if self.cfg.verbose:
                print(
                    f"[Debias] Not enough observations for debiasing on variant '{display_name}'."
                )
            return DebiasRegressionResult(
                variant=variant_key,
                display_name=display_name,
                strip_percentage=strip_percentage,
                correlation=None,
                mean_original=None,
                mean_stripped=None,
            )

        y_series = reg_df[original_column]
        s_series = reg_df[stripped_column]

        if np.allclose(np.var(s_series.values), 0.0):
            if self.cfg.verbose:
                print(
                    f"[Debias] Stripped column has no variation for variant '{display_name}'."
                )
            return DebiasRegressionResult(
                variant=variant_key,
                display_name=display_name,
                strip_percentage=strip_percentage,
                correlation=None,
                mean_original=float(y_series.mean()),
                mean_stripped=float(s_series.mean()),
            )

        attenuation_pct = self._attenuation_pct(y_series, s_series)
        if self.cfg.verbose and attenuation_pct is not None:
            print(
                "[Debias] Mean attenuation from original to stripped "
                f"for '{display_name}': {attenuation_pct:.2f}%"
            )

        diff_col = (
            f"{measurement_attr} (raw - stripped)"
            if label_suffix == "stripped"
            else f"{measurement_attr} (raw - {label_suffix})"
        )
        df.loc[reg_df.index, diff_col] = y_series - s_series
        pretty_diff_col = diff_col

        mean_pct_diff, mean_abs_pct_diff = self._avg_pct_diff(
            y_series,
            df.loc[reg_df.index, diff_col],
        )
        if self.cfg.verbose:
            print(
                "[Debias] Simple difference debiasing percent change "
                f"(signed / absolute): {mean_pct_diff:.2f}% / {mean_abs_pct_diff:.2f}%"
            )

        diff_reg = self._regress_against_original(
            df,
            original_column=original_column,
            debiased_column=diff_col,
            title="original ~ (original - stripped)",
        )

        twostep_col = (
            f"{measurement_attr} (debiased)"
            if label_suffix == "stripped"
            else f"{measurement_attr} (debiased, {label_suffix})"
        )
        pretty_twostep_col = twostep_col
        stage1_reg: dict[str, Any] | None = None
        delta: float | None = None
        if remaining_signal_column and remaining_signal_column in df.columns:
            stage1_reg, delta = self._stage1_remaining_signal_regression(
                df,
                stripped_column=stripped_column,
                remaining_signal_column=remaining_signal_column,
                display_name=display_name,
            )
            if delta is not None:
                valid_idx, y_vals, s_vals, r_vals = self._aligned_series(
                    df,
                    original_column,
                    stripped_column,
                    remaining_signal_column,
                )
                df.loc[valid_idx, twostep_col] = y_vals - s_vals + delta * r_vals
                df.loc[valid_idx, pretty_twostep_col] = df.loc[valid_idx, twostep_col]
                mean_pct_two, mean_abs_pct_two = self._avg_pct_diff(
                    y_vals,
                    df.loc[valid_idx, twostep_col],
                )
                if self.cfg.verbose:
                    print(
                        "[Debias] Two-step debiasing percent change "
                        f"(signed / absolute): {mean_pct_two:.2f}% / {mean_abs_pct_two:.2f}%"
                    )
        else:
            if self.cfg.verbose and self.cfg.remaining_signal_attribute:
                print(
                    "[Debias] Remaining-signal attribute configured but no remaining-signal "
                    f"column was available for variant '{display_name}'."
                )

        twostep_reg: dict[str, Any] | None = None
        if twostep_col in df.columns and df[twostep_col].notna().sum() >= 3:
            twostep_reg = self._regress_against_original(
                df,
                original_column=original_column,
                debiased_column=twostep_col,
                title="original ~ debiased (two-step)",
            )

        correlation = float(y_series.corr(s_series))
        summary = DebiasRegressionResult(
            variant=variant_key,
            display_name=display_name,
            strip_percentage=strip_percentage,
            correlation=correlation,
            mean_original=float(y_series.mean()),
            mean_stripped=float(s_series.mean()),
            attenuation_pct=attenuation_pct,
            diff_regression=diff_reg,
            twostep_regression=twostep_reg,
            stage1_regression=stage1_reg,
            stage1_delta=delta,
            debiased_columns={
                "diff": pretty_diff_col,
                "twostep": pretty_twostep_col,
                "remaining_signal": remaining_signal_column or "",
            },
        )
        return summary

    # ------------------------------------------------------------------
    def _aligned_series(
        self,
        df: pd.DataFrame,
        original_column: str,
        stripped_column: str,
        remaining_signal_column: str,
    ) -> tuple[pd.Index, pd.Series, pd.Series, pd.Series]:
        tmp = df[[original_column, stripped_column, remaining_signal_column]].apply(
            _to_float_series
        )
        tmp = tmp.dropna()
        return (
            tmp.index,
            tmp[original_column],
            tmp[stripped_column],
            tmp[remaining_signal_column],
        )

    # ------------------------------------------------------------------
    def _print_average_attenuation(
        self,
        df: pd.DataFrame,
        attributes: list[str],
        column_map: dict[str, str],
        *,
        label_suffix: str,
        display_name: str,
    ) -> None:
        if not self.cfg.verbose:
            return
        attenuations: list[float] = []
        for attr in attributes:
            stripped_col = column_map.get(attr)
            if not stripped_col or attr not in df.columns:
                continue
            raw_series = _to_float_series(df[attr])
            stripped_series = _to_float_series(df[stripped_col])
            tmp = pd.concat([raw_series, stripped_series], axis=1).dropna()
            if tmp.empty:
                continue
            raw_mean = float(tmp.iloc[:, 0].mean())
            stripped_mean = float(tmp.iloc[:, 1].mean())
            if np.isclose(raw_mean, 0.0):
                continue
            pct = (stripped_mean - raw_mean) / raw_mean * 100.0
            attenuations.append(float(pct))
        if not attenuations:
            print(
                "[Debias] Unable to compute average percent attenuation from raw to stripped "
                f"for variant '{display_name}'."
            )
            return
        avg_pct = float(np.mean(attenuations))
        print(
            "[Debias] Average percent attenuation from raw to "
            f"'{label_suffix}' across measurement attributes: {avg_pct:.2f}%"
        )

    # ------------------------------------------------------------------
    def _attenuation_pct(
        self,
        original: pd.Series,
        stripped: pd.Series,
    ) -> float | None:
        mean_original = float(_to_float_series(original).mean())
        mean_stripped = float(_to_float_series(stripped).mean())
        if np.isclose(mean_original, 0.0):
            return None
        return float((mean_original - mean_stripped) / mean_original * 100.0)

    # ------------------------------------------------------------------
    def _avg_pct_diff(
        self,
        raw: pd.Series,
        debiased: pd.Series,
    ) -> tuple[float, float]:
        raw_num = _to_float_series(raw)
        deb_num = _to_float_series(debiased)
        pct = (deb_num - raw_num) / raw_num * 100.0
        pct = pct.replace([np.inf, -np.inf], np.nan)
        pct = pct[raw_num != 0]
        return float(pct.mean()), float(pct.abs().mean())

    # ------------------------------------------------------------------
    def _stage1_remaining_signal_regression(
        self,
        df: pd.DataFrame,
        *,
        stripped_column: str,
        remaining_signal_column: str,
        display_name: str,
    ) -> tuple[dict[str, Any] | None, float | None]:
        stage1_df = (
            df[[stripped_column, remaining_signal_column]]
            .apply(_to_float_series)
            .dropna()
        )
        if len(stage1_df) < 3:
            if self.cfg.verbose:
                print(
                    "[Debias] Not enough observations for remaining-signal regression "
                    f"on variant '{display_name}'."
                )
            return None, None

        r_vals = stage1_df[remaining_signal_column].values
        if np.allclose(np.var(r_vals), 0.0):
            if self.cfg.verbose:
                print(
                    "[Debias] Remaining-signal proxy has no variation for "
                    f"variant '{display_name}'."
                )
            return None, None

        s_vals = stage1_df[stripped_column].values
        X1 = np.column_stack([np.ones(len(stage1_df)), r_vals])
        reg = _safe_fit_ols(
            s_vals,
            X1,
            robust=self.cfg.robust_regression,
            varnames=["Intercept", remaining_signal_column],
        )
        if self.cfg.verbose:
            self._print_generic_regression_table(
                reg,
                names=["Intercept", remaining_signal_column],
                title=f"Stage-1 regression: stripped ~ remaining signal [{display_name}]",
            )
        coef = reg["coef"]
        coef_vals = coef.to_numpy() if isinstance(coef, pd.Series) else np.asarray(coef)
        delta = float(coef_vals[1])
        return self._regression_dict(reg, ["Intercept", remaining_signal_column]), delta

    # ------------------------------------------------------------------
    def _regress_against_original(
        self,
        df: pd.DataFrame,
        *,
        original_column: str,
        debiased_column: str,
        title: str,
    ) -> dict[str, Any]:
        reg_df = df[[original_column, debiased_column]].apply(_to_float_series).dropna()
        if len(reg_df) < 3:
            return {}
        y_vals = reg_df[original_column].values
        x_vals = reg_df[debiased_column].values
        X = np.column_stack([np.ones(len(reg_df)), x_vals])
        reg_res = _safe_fit_ols(
            y_vals,
            X,
            robust=self.cfg.robust_regression,
            varnames=["Intercept", debiased_column],
        )
        if self.cfg.verbose:
            self._print_generic_regression_table(
                reg_res,
                names=["Intercept", debiased_column],
                title=title,
            )
        return self._regression_dict(reg_res, ["Intercept", debiased_column]) or {}

    # ------------------------------------------------------------------
    def _print_generic_regression_table(
        self,
        result: dict[str, Any],
        *,
        names: list[str],
        title: str,
    ) -> None:
        table = pd.DataFrame(
            {
                "coef": result["coef"],
                "se": result["se"],
                "t": result["t"],
                "p": result["p"],
            },
            index=names,
        )
        print(f"\n{title}")
        print(table.round(6).to_string())
        print(
            f"R^2 = {result['r2']:.4f}, adj. R^2 = {result['adj_r2']:.4f}, n = {result['n']}"
        )

    # ------------------------------------------------------------------
    def _regression_dict(
        self,
        result: dict[str, Any] | None,
        names: list[str],
    ) -> dict[str, Any] | None:
        if result is None:
            return None
        return {
            "coef": {name: float(val) for name, val in zip(names, result["coef"])},
            "se": {name: float(val) for name, val in zip(names, result["se"])},
            "t": {name: float(val) for name, val in zip(names, result["t"])},
            "p": {name: float(val) for name, val in zip(names, result["p"])},
            "r2": float(result["r2"]),
            "adj_r2": float(result["adj_r2"]),
            "n": int(result["n"]),
        }

    # ------------------------------------------------------------------
    def _serialise_config(self) -> dict[str, Any]:
        def _convert(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [_convert(v) for v in value]
            if isinstance(value, np.generic):
                return value.item()
            if callable(value):
                name = getattr(value, "__name__", None)
                return name or repr(value)
            try:
                json.dumps(value)
                return value
            except TypeError:
                return repr(value)

        cfg_dict = asdict(self.cfg)
        cfg_dict["save_dir"] = os.path.expandvars(os.path.expanduser(self.cfg.save_dir))
        cfg_dict["measurement_kwargs"] = _convert(self.cfg.measurement_kwargs)
        cfg_dict["removal_kwargs"] = _convert(self.cfg.removal_kwargs)
        return cfg_dict

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_ws(text: str) -> str:
        return " ".join(str(text or "").split())
