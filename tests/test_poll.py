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


def test_poll_weighted_sampling_reproducibility_and_seed_resampling(tmp_path):
    seeds = [f"respondent_{i}" for i in range(60)]
    df = pd.DataFrame({"seed": seeds})
    q_scale = "How satisfied are you? (1-5)"
    q_guard = "Edge case guard"
    q_wrapped = "Wrapped options"
    questions = [q_scale, q_guard, q_wrapped]

    # Canonical ascending order where option 1 has only 5% mass and option 5 has 95%.
    payload = {
        q_scale: [[1, 0.05], [5, 0.95]],
        q_guard: [["bool_weight", True], ["neg_weight", -0.4], ["valid_pick", 0.5]],
        q_wrapped: {"options": [["wrapped_yes", 1.0], ["wrapped_no", 0.0]]},
    }
    responses = {s: payload for s in seeds}

    dir_a = tmp_path / "run_a"
    client_a = SurveyStubClient(responses)
    res_a = run(
        poll(
            df,
            "seed",
            questions=questions,
            save_dir=str(dir_a),
            client=client_a,
        )
    )

    # 1. Weighted draw: option 5 (0.95 mass) must dominate over first-listed option 1 (0.05)
    counts = res_a[q_scale].value_counts().to_dict()
    assert counts.get(5, 0) >= 45
    assert set(res_a[q_scale]).issubset({1, 5})

    # 2. Malformed boolean/negative weights are skipped; wrapped {"options": ...} resolves
    assert list(res_a[q_guard]) == ["valid_pick"] * 60
    assert list(res_a[q_wrapped]) == ["wrapped_yes"] * 60

    # 3. Fresh run in a separate directory with default random_seed=None is deterministic
    dir_b = tmp_path / "run_b"
    client_b = SurveyStubClient(responses)
    res_b = run(
        poll(
            df,
            "seed",
            questions=questions,
            save_dir=str(dir_b),
            client=client_b,
        )
    )
    assert list(res_b[q_scale]) == list(res_a[q_scale])

    # 4. Interrupted resume (poll_state.json missing) reproduces identical draws with 0 calls
    (dir_a / "poll_state.json").unlink()
    calls_before_resume = client_a.calls
    res_resumed = run(
        poll(
            df,
            "seed",
            questions=questions,
            save_dir=str(dir_a),
            client=client_a,
            reset_files=False,
        )
    )
    assert client_a.calls == calls_before_resume
    assert list(res_resumed[q_scale]) == list(res_a[q_scale])

    # 5. Changing random_seed invalidates manifest, re-samples cached answers_raw.csv with 0 calls
    res_seed_1 = run(
        poll(
            df,
            "seed",
            questions=questions,
            save_dir=str(dir_a),
            client=client_a,
            random_seed=101,
            reset_files=False,
        )
    )
    res_seed_2 = run(
        poll(
            df,
            "seed",
            questions=questions,
            save_dir=str(dir_a),
            client=client_a,
            random_seed=202,
            reset_files=False,
        )
    )
    assert client_a.calls == calls_before_resume
    assert list(res_seed_1[q_scale]) != list(res_seed_2[q_scale])
    assert res_seed_1[q_scale].value_counts().to_dict().get(5, 0) >= 45
    assert res_seed_2[q_scale].value_counts().to_dict().get(5, 0) >= 45


def test_poll_resolves_shortened_and_normalized_question_keys_safely(tmp_path):
    df = pd.DataFrame({"seed": ["Alice", "Bob"]})
    q_bullet = "How often do you grill outdoors?"
    q_tagged = (
        "[Q14] How interested are you in cold smoking at home? "
        "(Select one: Very interested / Somewhat interested / Not interested)"
    )
    q_exact = "How old are you?"
    q_ambig_a = "[Q54] Rate continuous smoke output (If present)"
    q_ambig_b = "[Q55] Rate continuous smoke output (If absent)"
    questions = [q_bullet, q_tagged, q_exact, q_ambig_a, q_ambig_b]

    responses = {
        "Alice": {
            # Echoed leading prompt bullet + smart punctuation/case variation
            "- how often do you grill outdoors? ": [["Weekly", 1.0]],
            # Dropped both leading [Q14] tag and trailing (Select one: ...) hint
            "How interested are you in cold smoking at home?": [
                ["Very interested", 1.0]
            ],
            # Exact match must never be stolen
            "How old are you?": [[34, 1.0]],
            # Unambiguous when leading ID tags are kept even if trailing hints drop
            "[Q54] Rate continuous smoke output": [["Must-be", 1.0]],
            "[Q55] Rate continuous smoke output": [["Dislike", 1.0]],
        },
        "Bob": {
            # Keyed by leading [Q14] ID alone
            "Q14": [["Somewhat interested", 1.0]],
            "- How often do you grill outdoors?": [["Monthly", 1.0]],
            "How old are you?": [[49, 1.0]],
            # Ambiguous bare stem matches BOTH Q54 and Q55 -> must NOT cross-contaminate
            "Rate continuous smoke output": [["Ambiguous", 1.0]],
        },
    }

    res = run(
        poll(
            df,
            "seed",
            questions=questions,
            save_dir=str(tmp_path / "poll_keys"),
            client=SurveyStubClient(responses),
        )
    )

    assert list(res[q_bullet]) == ["Weekly", "Monthly"]
    assert list(res[q_tagged]) == ["Very interested", "Somewhat interested"]
    assert list(res[q_exact]) == [34, 49]
    assert res[q_ambig_a].iloc[0] == "Must-be"
    assert res[q_ambig_b].iloc[0] == "Dislike"
    assert pd.isna(res[q_ambig_a].iloc[1])
    assert pd.isna(res[q_ambig_b].iloc[1])


def test_poll_supports_mapping_questions_with_short_ids_and_signature(tmp_path):
    df = pd.DataFrame({"seed": ["Alice", "Bob"]})
    questions_v1 = {
        "smoke_interest": (
            "How interested are you in cold smoking salmon or cheese? "
            "(1 = Not at all, 5 = Extremely)"
        ),
        "preferred_fuel": "Which fuel type do you prefer? (Pellets / Wood chips)",
    }

    responses = {
        # Alice's completion keys by short question_id (prompt contract)
        "Alice": {
            "smoke_interest": [[5, 1.0]],
            "preferred_fuel": [["Pellets", 1.0]],
        },
        # Bob's completion echoes full question text on one item and ID on the other
        "Bob": {
            "How interested are you in cold smoking salmon or cheese?": [[4, 1.0]],
            "preferred_fuel": [["Wood chips", 1.0]],
        },
    }

    save_dir = str(tmp_path / "poll_map")
    client = SurveyStubClient(responses)
    res = run(
        poll(
            df,
            "seed",
            questions=questions_v1,
            save_dir=save_dir,
            client=client,
        )
    )

    assert list(res.columns) == ["seed", "persona", "smoke_interest", "preferred_fuel"]
    assert list(res["smoke_interest"]) == [5, 4]
    assert list(res["preferred_fuel"]) == ["Pellets", "Wood chips"]

    # Prefix-first invariant preserved: question IDs + texts precede BEGIN PERSONA
    survey_prompts = [p for p in client.prompts if "BEGIN SURVEY QUESTIONS" in p]
    prefixes = [p.split("BEGIN PERSONA")[0] for p in survey_prompts]
    assert len(set(prefixes)) == 1
    assert "smoke_interest" in prefixes[0]
    assert "cold smoking salmon or cheese" in prefixes[0]

    # Changing question wording under the same question_id must invalidate cache
    calls_after_v1 = client.calls
    questions_v2 = {
        **questions_v1,
        "smoke_interest": "How interested are you in hot smoking brisket? (1-5)",
    }
    run(
        poll(
            df,
            "seed",
            questions=questions_v2,
            save_dir=save_dir,
            client=client,
            reset_files=False,
        )
    )
    assert client.calls > calls_after_v1
