"""Rank task: pairwise tournaments yielding relative z-scores."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ...templates import load_prompt
from ...utils.parsing import safe_json
from .. import _common
from ..base import Task
from .bt import fit_bt


@dataclass
class RankConfig:
    attributes: Dict[str, str]
    save_dir: str = "rank"
    file_name: str = "rankings"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    n_rounds: int = 5
    matches_per_round: int = 3
    learning_rate: float = 0.1
    additional_instructions: Optional[str] = None
    modality: str = "text"


class Rank(Task):
    """Compare items pairwise and output centered Bradley–Terry scores."""

    def __init__(self, cfg: RankConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("rankings_prompt.jinja2")

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        texts = [str(v) for v in df[column_name].tolist()]
        unique = list(dict.fromkeys(texts))
        item_ids = [_common.hash_identifier(t) for t in unique]
        id_to_text = dict(zip(item_ids, unique))
        attrs = list(self.cfg.attributes.keys())

        outcomes: Dict[str, List[Tuple[str, str]]] = {a: [] for a in attrs}
        prompts: List[str] = []
        ids: List[str] = []
        meta: List[Tuple[str, str]] = []  # (winner_if_circle, winner_if_square)

        for rnd in range(self.cfg.n_rounds):
            pairs = self._pair(item_ids, self.cfg.matches_per_round, rnd)
            for a, b in pairs:
                circle_first = self._circle_first(a, b, rnd)
                circle, square = (a, b) if circle_first else (b, a)
                ids.append(f"{a}|{b}|rnd{rnd}")
                prompts.append(
                    self._template.render(
                        entry_circle=id_to_text[circle],
                        entry_square=id_to_text[square],
                        attributes=self.cfg.attributes,
                        additional_instructions=self.cfg.additional_instructions,
                        modality=self.cfg.modality,
                        circle_first=circle_first,
                    )
                )
                meta.append((circle, square))

        resp_df = await self._execute(prompts, ids, reset_files=reset_files, **kwargs)
        resp_lookup = dict(zip(resp_df["Identifier"], resp_df["Response"]))

        for (circle, square), ident in zip(meta, ids):
            obj = safe_json(resp_lookup.get(ident))
            if not isinstance(obj, dict):
                continue
            for attr in attrs:
                winner = str(obj.get(attr, "")).strip().lower()
                if winner == "circle":
                    outcomes[attr].append((circle, square))
                elif winner == "square":
                    outcomes[attr].append((square, circle))

        scores_by_attr: Dict[str, Dict[str, float]] = {}
        for attr in attrs:
            raw = fit_bt(item_ids, outcomes[attr], pseudo=self.cfg.learning_rate)
            vals = [raw.get(i, 0.0) for i in item_ids]
            mean = sum(vals) / len(vals)
            scores_by_attr[attr] = {i: v - mean for i, v in zip(item_ids, vals)}

        out = df.copy()
        id_by_text = {t: i for t, i in zip(unique, item_ids)}
        for attr in attrs:
            out[attr] = [
                scores_by_attr[attr].get(id_by_text[str(v)], 0.0) for v in df[column_name]
            ]
        self._save(out)
        return out

    def _pair(self, item_ids: List[str], matches_per_round: int, rnd: int) -> List[Tuple[str, str]]:
        rng = random.Random(1000 + rnd)
        pairs: List[Tuple[str, str]] = []
        for a in item_ids:
            opponents = [b for b in item_ids if b != a]
            rng.shuffle(opponents)
            for b in opponents[:matches_per_round]:
                if (a, b) not in pairs and (b, a) not in pairs:
                    pairs.append((a, b))
        return pairs

    @staticmethod
    def _circle_first(a: str, b: str, rnd: int) -> bool:
        rng = random.Random(hash(a) ^ hash(b) ^ rnd)
        return rng.random() < 0.5

    async def _execute(self, prompts, ids, *, reset_files, **kwargs):
        from ...engine.batch import batch_execute

        return await batch_execute(
            prompts,
            ids,
            client=self._get_client(),
            model=self.cfg.model,
            json_mode=True,
            n_parallels=self.cfg.n_parallels,
            save_path=os.path.join(
                self.cfg.save_dir, f"{self._base_name()}_raw_responses.csv"
            ),
            reset_files=reset_files,
            **kwargs,
        )

    def _save(self, out: pd.DataFrame) -> None:
        out_path = os.path.join(self.cfg.save_dir, f"{self._base_name()}_final.csv")
        try:
            out.to_csv(out_path, index=False)
        except Exception:
            pass
