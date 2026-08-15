"""Rate task: score each item 0-100 on natural-language attributes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..templates import load_prompt
from ..utils.parsing import safe_json
from .base import AttributeTask


@dataclass
class RateConfig:
    attributes: dict[str, str]
    save_dir: str = "ratings"
    file_name: str = "ratings.csv"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    n_runs: int = 1
    n_attributes_per_run: int | None = None
    rating_scale: str | None = None
    additional_instructions: str | None = None
    modality: str = "text"  # "text" or "entity"


class Rate(AttributeTask):
    """Score each item on the configured attributes (0-100)."""

    def __init__(self, cfg: RateConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("ratings_prompt.jinja2")

    def render_prompt(self, text: str, attributes: dict[str, str]) -> str:
        return self._template.render(
            text=text,
            attributes=attributes,
            scale=self.cfg.rating_scale,
            additional_instructions=self.cfg.additional_instructions,
            modality=self.cfg.modality,
        )

    async def parse_response(
        self, raw: Any, attributes: Sequence[str]
    ) -> dict[str, float | None]:
        obj = safe_json(raw)
        out: dict[str, float | None] = {}
        if isinstance(obj, dict):
            for attr in attributes:
                try:
                    out[attr] = float(obj.get(attr)) if obj.get(attr) is not None else None
                except (TypeError, ValueError):
                    out[attr] = None
        else:
            out = {attr: None for attr in attributes}
        return out
