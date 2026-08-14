"""Discover: compare two classes, bucket the differences into a taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from ..base import Task


@dataclass
class DiscoverConfig:
    save_dir: str = "discover"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    bucket_count: int = 10
    additional_instructions: Optional[str] = None


class Discover(Task):
    """Contrast two columns and bucket the distinguishing features."""

    async def run(
        self,
        df: pd.DataFrame,
        *,
        column_name: Optional[str] = None,
        circle_column_name: Optional[str] = None,
        square_column_name: Optional[str] = None,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from ..bucket import Bucket, BucketConfig
        from ..compare import Compare, CompareConfig
        from ..codify import Codify, CodifyConfig

        # 1. candidate discovery
        term_defs: Dict[str, str] = {}
        if circle_column_name and square_column_name:
            cmp_cfg = CompareConfig(
                save_dir=f"{self.cfg.save_dir}/compare",
                model=self.cfg.model,
                n_parallels=self.cfg.n_parallels,
                additional_instructions=self.cfg.additional_instructions,
            )
            cmp_df = await Compare(cmp_cfg, client=self._client).run(
                df, circle_column_name, square_column_name, reset_files=reset_files, **kwargs
            )
            for attr, expl in zip(cmp_df["attribute"], cmp_df["explanation"]):
                if pd.notna(attr) and attr not in term_defs:
                    term_defs[str(attr)] = str(expl) if pd.notna(expl) else ""
        elif column_name:
            cod_cfg = CodifyConfig(
                save_dir=f"{self.cfg.save_dir}/codify",
                model=self.cfg.model,
                n_parallels=self.cfg.n_parallels,
            )
            cod_df = await Codify(cod_cfg, client=self._client).run(
                df, column_name, categories=None, reset_files=reset_files, **kwargs
            )
            for entry in cod_df["coded_passages"].dropna():
                if isinstance(entry, dict):
                    for k, v in entry.items():
                        if k not in term_defs:
                            term_defs[str(k)] = str(v[0]) if isinstance(v, list) and v else str(v)
        else:
            raise ValueError("provide column_name or circle/square columns")

        candidate_df = pd.DataFrame({"term": list(term_defs.keys())})

        # 2. bucket
        if candidate_df.empty:
            bucket_df = pd.DataFrame(columns=["bucket", "definition"])
        else:
            bkt_cfg = BucketConfig(
                bucket_count=self.cfg.bucket_count,
                save_dir=f"{self.cfg.save_dir}/bucket",
                model=self.cfg.model,
                n_parallels=self.cfg.n_parallels,
                additional_instructions=self.cfg.additional_instructions,
            )
            bucket_df = await Bucket(bkt_cfg, client=self._client).run(
                candidate_df, "term", reset_files=reset_files, **kwargs
            )

        return {"candidates": candidate_df, "buckets": bucket_df}
