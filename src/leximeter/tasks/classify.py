"""Classify task: decide whether each label applies to each item."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from ..templates import load_prompt
from ..utils.parsing import safe_json
from .base import AttributeTask


def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    return bool(value)


def _true_proportion(series: pd.Series) -> Any:
    vals = series.dropna()
    if len(vals) == 0:
        return pd.NA
    return float(vals.astype(bool).mean())


@dataclass
class ClassifyConfig:
    labels: Dict[str, str]
    save_dir: str = "classifier"
    file_name: str = "classify_responses.csv"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    n_runs: int = 1
    min_frequency: float = 0.6
    additional_instructions: Optional[str] = None
    modality: str = "text"
    n_attributes_per_run: Optional[int] = None


class Classify(AttributeTask):
    """Decide whether each label applies to each item (single-column mode)."""

    def __init__(self, cfg: ClassifyConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("classification_prompt.jinja2")

    # classify's attribute dict lives on ``labels`` rather than ``attributes``
    def _get_attributes(self) -> Dict[str, str]:
        return dict(self.cfg.labels)

    def _set_attributes(self, attributes: Dict[str, str]) -> None:
        self.cfg.labels = attributes

    def render_prompt(self, text: str, attributes: Dict[str, str]) -> str:
        return self._template.render(
            text=text,
            attributes=attributes,
            modality=self.cfg.modality,
            additional_instructions=self.cfg.additional_instructions,
            differentiate=False,
            circle_first=None,
        )

    async def parse_response(
        self, raw: Any, attributes: Sequence[str]
    ) -> Dict[str, Optional[bool]]:
        obj = safe_json(raw)
        out: Dict[str, Optional[bool]] = {}
        if isinstance(obj, dict):
            for label in attributes:
                out[label] = _to_bool(obj.get(label))
        else:
            out = {label: None for label in attributes}
        return out

    def aggregate(self, records: pd.DataFrame, df_proc: pd.DataFrame) -> pd.DataFrame:
        """True per label if the True proportion across runs >= min_frequency."""
        labels = self._attribute_names()
        grouped = records.groupby("id")
        columns: Dict[str, pd.Series] = {}
        for label in labels:
            prop = grouped[label].apply(_true_proportion)
            columns[label] = prop >= self.cfg.min_frequency
        agg = pd.DataFrame(columns)
        return df_proc.merge(agg, left_on="_gid", right_index=True, how="left").drop(
            columns=["_gid"]
        )
