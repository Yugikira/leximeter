"""Extract task: pull structured facts out of each item."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from ..templates import load_prompt
from ..utils.parsing import safe_json
from .base import AttributeTask


@dataclass
class ExtractConfig:
    attributes: Dict[str, str]
    save_dir: str = "extraction"
    file_name: str = "extraction.csv"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    n_runs: int = 1
    n_attributes_per_run: Optional[int] = None
    additional_instructions: Optional[str] = None
    modality: str = "entity"
    types: Optional[Dict[str, Any]] = None


class Extract(AttributeTask):
    """Extract structured facts per item (single-entity output in v1)."""

    def __init__(self, cfg: ExtractConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("extraction_prompt.jinja2")

    def render_prompt(self, text: str, attributes: Dict[str, str]) -> str:
        return self._template.render(
            text=text,
            attributes=attributes,
            modality=self.cfg.modality,
            additional_instructions=self.cfg.additional_instructions,
        )

    async def parse_response(
        self, raw: Any, attributes: Sequence[str]
    ) -> Dict[str, Any]:
        obj = safe_json(raw)
        # The model returns {entity: {attr: value}}; flatten the first entity.
        if (
            isinstance(obj, dict)
            and obj
            and all(isinstance(v, dict) for v in obj.values())
        ):
            obj = next(iter(obj.values()))
        out: Dict[str, Any] = {}
        if isinstance(obj, dict):
            for attr in attributes:
                out[attr] = self._coerce(obj.get(attr), attr)
        else:
            out = {attr: None for attr in attributes}
        return out

    def _coerce(self, value: Any, attr: str) -> Any:
        if value is None or value == "unknown":
            return None
        typ = (self.cfg.types or {}).get(attr)
        if typ is None:
            return value
        try:
            return typ(value)
        except (TypeError, ValueError):
            return value

    def aggregate(self, records: pd.DataFrame, df_proc: pd.DataFrame) -> pd.DataFrame:
        """Take the first non-null value across runs (values are not averaged)."""
        attrs = self._attribute_names()

        def first_non_null(series: pd.Series) -> Any:
            non_null = series.dropna()
            return non_null.iloc[0] if len(non_null) else pd.NA

        agg = records.groupby("id")[attrs].agg(first_non_null)
        return df_proc.merge(agg, left_on="_gid", right_index=True, how="left").drop(
            columns=["_gid"]
        )
