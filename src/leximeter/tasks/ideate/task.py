"""Ideate: generate novel ideas/theories on a topic."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ...engine.batch import batch_execute
from ...templates import load_prompt
from .. import _common
from ..base import Task


@dataclass
class IdeateConfig:
    save_dir: str = "ideate"
    file_name: str = "ideation.csv"
    model: str = "gpt-5.4-mini"
    n_parallels: int = 50
    n_ideas: int = 1000
    scientific_theory: bool = True
    additional_instructions: str | None = None


class Ideate(Task):
    """Generate ``n_ideas`` reports on a topic and parse them into fields."""

    def __init__(self, cfg: IdeateConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("ideation_prompt.jinja2")

    async def run(
        self,
        topic: str,
        *,
        additional_instructions: str | None = None,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        instructions = additional_instructions or self.cfg.additional_instructions
        signature = _common.compute_input_signature(
            topic,
            self.cfg.n_ideas,
            self.cfg.scientific_theory,
            instructions,
            self.cfg.model,
            self.cfg.file_name,
        )
        if reset_files:
            _common.clear_task_state(self.cfg.save_dir, "ideate")
        else:
            # Identifiers are positional (idea-00000...), so the raw batch
            # checkpoint does not encode the topic: when the inputs change,
            # stale completions must be discarded or they would be reused.
            state = _common.load_task_state(self.cfg.save_dir, "ideate")
            if state is not None and state.get("input_signature") != signature:
                _common.remove_file(
                    os.path.join(self.cfg.save_dir, f"{self._base_name()}_raw.csv")
                )
            loaded = self._load_completed(signature)
            if loaded is not None:
                return loaded

        prompts = [
            self._template.render(
                topic=topic,
                scientific_theory=self.cfg.scientific_theory,
                seed=None,
                additional_instructions=instructions,
            )
            for _ in range(self.cfg.n_ideas)
        ]
        ids = [f"idea-{i:05d}" for i in range(self.cfg.n_ideas)]

        resp_df = await batch_execute(
            prompts,
            ids,
            client=self._get_client(),
            model=self.cfg.model,
            json_mode=False,
            n_parallels=self.cfg.n_parallels,
            save_path=os.path.join(self.cfg.save_dir, f"{self._base_name()}_raw.csv"),
            reset_files=reset_files,
            **kwargs,
        )
        resp_map = dict(zip(resp_df["Identifier"], resp_df["Response"]))

        rows = []
        for ident in ids:
            report = resp_map.get(ident, "")
            parsed = self._parse_sections(report)
            parsed["idea_id"] = ident
            parsed["topic"] = topic
            parsed["report_text"] = report
            rows.append(parsed)

        out = pd.DataFrame(rows)
        self._save(out)
        _common.save_task_state(
            self.cfg.save_dir,
            "ideate",
            input_signature=signature,
            result_artifacts=[self.cfg.file_name],
        )
        return out

    def _load_completed(self, signature: str) -> pd.DataFrame | None:
        """Return the saved result when inputs match and it is complete."""
        out = _common.load_completed_csv(
            self.cfg.save_dir,
            "ideate",
            signature=signature,
            file_name=self.cfg.file_name,
        )
        if out is None:
            return None
        path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        print(f"[Ideate] Found completed result in {path}; loading {len(out)} ideas.")
        return out

    @staticmethod
    def _parse_sections(text: str) -> dict[str, str | None]:
        headers = {
            "title": "title",
            "in a nutshell": "in_a_nutshell",
            "in one paragraph": "in_one_paragraph",
            "illustrative examples": "illustrative_examples",
            "testable predictions": "testable_predictions",
            "the full thinking": "full_thinking",
        }
        result: dict[str, str | None] = {v: None for v in headers.values()}
        current: str | None = None
        buffer: list[str] = []
        for line in text.splitlines():
            lowered = line.strip().lower()
            matched = None
            for header, key in headers.items():
                if lowered.startswith(f"{header}:"):
                    if current is not None:
                        result[current] = "\n".join(buffer).strip() or None
                    buffer = [line.split(":", 1)[1].strip()] if ":" in line else []
                    current = key
                    matched = True
                    break
            if not matched and current is not None:
                buffer.append(line)
        if current is not None:
            result[current] = "\n".join(buffer).strip() or None
        return result

    def _save(self, out: pd.DataFrame) -> None:
        out_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        try:
            out.to_csv(out_path, index=False)
        except OSError:
            pass
