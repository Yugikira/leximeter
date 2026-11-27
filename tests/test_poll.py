"""Behavioral tests for ``leximeter.poll`` through its public interface."""

from __future__ import annotations

import asyncio
import json

import pandas as pd

from leximeter import poll
from leximeter.client.base import LLMClient


def run(coro):
    return asyncio.run(coro)


class SurveyStubClient(LLMClient):
    """Fake LLMClient returning per-seed persona bios and canned survey payloads."""

    def __init__(self, responses_by_seed: dict[str, dict]) -> None:
        self._responses_by_seed = responses_by_seed
        self.calls = 0
        self.prompts: list[str] = []

    async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if "BEGIN SURVEY QUESTIONS" in prompt:
            for seed_name, payload in self._responses_by_seed.items():
                if f"BIO_FOR_{seed_name}" in prompt:
                    return json.dumps(payload)
            return json.dumps({})
        for seed_name in self._responses_by_seed:
            if seed_name in prompt:
                return f"BIO_FOR_{seed_name}"
        return "BIO_FOR_UNKNOWN"

    def estimate_tokens(self, prompt, *, model) -> int:
        return 0

    def estimate_cost(self, prompt, *, model) -> float:
        return 0.0


def test_poll_preserves_list_and_dict_answers_across_run_and_reload(tmp_path):
    df = pd.DataFrame({"seed": ["Alice", "Bob", "Carol"]})
    questions = [
        "Which foods do you cold-smoke? (Select all that apply)",
        "Rank your top two priorities",
        "Any open-ended note?",
    ]
    responses = {
        "Alice": {
            questions[0]: [[["Salmon", "Cheese"], 1.0]],
            questions[1]: [[{"first": "temp_control", "second": "safety"}, 1.0]],
            questions[2]: [["[unclosed bracket note", 1.0]],
        },
        "Bob": {
            questions[0]: [[["Brisket"], 1.0]],
            questions[1]: [[{"first": "clean_smoke", "second": "capacity"}, 1.0]],
            questions[2]: [["Plain text answer", 1.0]],
        },
        "Carol": {
            questions[0]: [[[], 1.0]],
            questions[1]: [[["safety", "price", "size", "brand"], 1.0]],
            questions[2]: [[42, 1.0]],
        },
    }

    save_dir = str(tmp_path / "poll_lists")
    client = SurveyStubClient(responses)
    first = run(
        poll(
            df,
            "seed",
            questions=questions,
            save_dir=save_dir,
            client=client,
        )
    )

    assert list(first[questions[0]]) == [["Salmon", "Cheese"], ["Brisket"], []]
    assert list(first[questions[1]]) == [
        {"first": "temp_control", "second": "safety"},
        {"first": "clean_smoke", "second": "capacity"},
        ["safety", "price", "size", "brand"],
    ]
    assert first[questions[2]].iloc[0] == "[unclosed bracket note"
    assert first[questions[2]].iloc[1] == "Plain text answer"
    assert first[questions[2]].iloc[2] == 42

    calls_after_first = client.calls
    reloaded = run(
        poll(
            df,
            "seed",
            questions=questions,
            save_dir=save_dir,
            client=client,
            reset_files=False,
        )
    )
    assert client.calls == calls_after_first
    assert list(reloaded[questions[0]]) == [["Salmon", "Cheese"], ["Brisket"], []]
    assert list(reloaded[questions[1]]) == [
        {"first": "temp_control", "second": "safety"},
        {"first": "clean_smoke", "second": "capacity"},
        ["safety", "price", "size", "brand"],
    ]
    assert reloaded[questions[2]].iloc[0] == "[unclosed bracket note"
    assert reloaded[questions[2]].iloc[1] == "Plain text answer"
    assert int(reloaded[questions[2]].iloc[2]) == 42
