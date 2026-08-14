"""Debias: remove inference bias from a measurement (simplified v1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..base import Task


@dataclass
class DebiasConfig:
    save_dir: str = "debias"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    measurement_attribute: Optional[str] = None
    removal_attribute: Optional[str] = None
    attributes: Optional[Dict[str, str]] = None


class Debias(Task):
    """Measure a signal, strip a confounder, re-measure, and regress out bias."""

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from ..codify import Codify, CodifyConfig
        from ..rate import Rate, RateConfig

        attrs = self.cfg.attributes or {}
        measure_attr = self.cfg.measurement_attribute or next(iter(attrs), None)
        if not measure_attr or not self.cfg.removal_attribute:
            raise ValueError("debias needs measurement_attribute and removal_attribute")

        # 1. measure original
        rate_cfg = RateConfig(
            attributes=attrs, save_dir=f"{self.cfg.save_dir}/original",
            model=self.cfg.model, n_parallels=self.cfg.n_parallels,
        )
        original = await Rate(rate_cfg, client=self._client).run(
            df, column_name, reset_files=reset_files, **kwargs
        )

        # 2. locate + strip the removal attribute's passages
        cod_cfg = CodifyConfig(
            categories={self.cfg.removal_attribute: self.cfg.removal_attribute},
            save_dir=f"{self.cfg.save_dir}/codify",
            model=self.cfg.model, n_parallels=self.cfg.n_parallels,
        )
        coded = await Codify(cod_cfg, client=self._client).run(
            df, column_name, reset_files=reset_files, **kwargs
        )
        stripped_texts = [
            self._strip(text, passages)
            for text, passages in zip(
                df[column_name].astype(str), coded["coded_passages"]
            )
        ]

        # 3. re-measure on stripped text
        stripped_df = df.copy()
        stripped_df["_stripped"] = stripped_texts
        stripped_cfg = RateConfig(
            attributes={measure_attr: attrs[measure_attr]},
            save_dir=f"{self.cfg.save_dir}/stripped",
            model=self.cfg.model, n_parallels=self.cfg.n_parallels,
        )
        remeasured = await Rate(stripped_cfg, client=self._client).run(
            stripped_df, "_stripped", reset_files=reset_files, **kwargs
        )

        # 4. attenuation + regression
        y = original[measure_attr].astype(float).to_numpy()
        x = remeasured[measure_attr].astype(float).to_numpy()
        mask = ~(np.isnan(y) | np.isnan(x))
        attenuation = float(np.mean(y[mask] - x[mask])) if mask.any() else float("nan")
        coef = self._ols(y[mask], x[mask])

        result = original.copy()
        result[f"{measure_attr} (stripped)"] = remeasured[measure_attr]
        return {
            "results": result,
            "metadata": {
                "attenuation": attenuation,
                "regression_coef": coef,
                "n": int(mask.sum()),
            },
        }

    @staticmethod
    def _strip(text: str, passages: Any) -> str:
        if not isinstance(passages, dict):
            return text
        out = text
        for snippets in passages.values():
            if not isinstance(snippets, list):
                continue
            for snippet in snippets:
                if isinstance(snippet, dict) and snippet.get("beginning excerpt"):
                    out = out.replace(str(snippet["beginning excerpt"]), "")
        return out

    @staticmethod
    def _ols(y: np.ndarray, x: np.ndarray) -> Optional[float]:
        if len(x) < 2:
            return None
        X = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(coef[1])
