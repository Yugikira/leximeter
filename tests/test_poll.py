"""Behavioral tests for ``leximeter.poll`` through its public interface."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pandas as pd
import pytest
from conftest import CountingClient

from leximeter import poll


def run(coro):
    return asyncio.run(coro)


class SurveyStubClient(CountingClient):
    """CountingClient returning per-seed persona bios and canned survey payloads."""

    def __init__(
        self,
        responses_by_seed: dict[str, dict[str, Any]],
        *,
        fail_on_survey_substring: str | None = None,
    ) -> None:
        super().__init__({})
        self._responses_by_seed = responses_by_seed
        self.fail_on_survey_substring = fail_on_survey_substring
        self.prompts: list[str] = []

    async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if "BEGIN SURVEY QUESTIONS" in prompt:
            if (
                self.fail_on_survey_substring is not None
                and self.fail_on_survey_substring in prompt
            ):
                raise asyncio.CancelledError("transient survey interruption")
            for seed_name, payload in self._responses_by_seed.items():
                if f"BIO_FOR_{seed_name}" in prompt:
                    return json.dumps(payload)
            return json.dumps({})
        for seed_name in self._responses_by_seed:
            if seed_name in prompt:
                return f"BIO_FOR_{seed_name}"
        return "BIO_FOR_UNKNOWN"


def test_poll_preserves_list_and_dict_answers_across_run_and_reload(tmp_path):
    df = pd.DataFrame({"seed": ["Alice", "Bob", "Carol"]})
    questions = [
        "Which foods do you cold-smoke? (Select all that apply)",
        "Rank your top two priorities",
        "Unweighted dict candidate",
        "Any open-ended note?",
    ]
    responses = {
        "Alice": {
            questions[0]: [[["Salmon", "Cheese"], 1.0]],
            questions[1]: [[{"first": "temp_control", "second": "safety"}, 1.0]],
            # Flat unweighted list holding a structured dict with "answer"/"option" keys
            questions[2]: [{"answer": "temp", "option": "smoke"}],
            questions[3]: [["[unclosed bracket note", 1.0]],
        },
        "Bob": {
            questions[0]: [[["Brisket"], 1.0]],
            questions[1]: [[{"first": "clean_smoke", "second": "capacity"}, 1.0]],
            questions[2]: [{"answer": "safety", "option": "size"}],
            questions[3]: [["Plain text answer", 1.0]],
        },
        "Carol": {
            questions[0]: [[[], 1.0]],
            questions[1]: [[["safety", "price", "size", "brand"], 1.0]],
            questions[2]: [{"answer": "price", "option": "brand"}],
            questions[3]: [[42, 1.0]],
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
    assert list(first[questions[2]]) == [
        {"answer": "temp", "option": "smoke"},
        {"answer": "safety", "option": "size"},
        {"answer": "price", "option": "brand"},
    ]
    assert first[questions[3]].iloc[0] == "[unclosed bracket note"
    assert first[questions[3]].iloc[1] == "Plain text answer"
    assert first[questions[3]].iloc[2] == 42

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
    assert list(reloaded[questions[2]]) == [
        {"answer": "temp", "option": "smoke"},
        {"answer": "safety", "option": "size"},
        {"answer": "price", "option": "brand"},
    ]
    assert reloaded[questions[3]].iloc[0] == "[unclosed bracket note"
    assert reloaded[questions[3]].iloc[1] == "Plain text answer"
    assert int(reloaded[questions[3]].iloc[2]) == 42


def test_poll_weighted_sampling_reproducibility_and_seed_resampling(tmp_path):
    seeds = [f"persona_{i}" for i in range(60)]
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
            n_questions_per_run=2,
        )
    )

    # 1. Weighted draw: option 5 (0.95 mass) must dominate over first-listed option 1 (0.05)
    counts = res_a[q_scale].value_counts().to_dict()
    assert counts.get(5, 0) >= 45
    assert set(res_a[q_scale]).issubset({1, 5})

    # 2. Malformed boolean/negative weights are skipped; wrapped {"options": ...} resolves
    assert list(res_a[q_guard]) == ["valid_pick"] * 60
    assert list(res_a[q_wrapped]) == ["wrapped_yes"] * 60

    # 3. Interrupted run driven through client= seam (batch 2 fails on attempt 1, succeeds on resume)
    dir_interrupted = tmp_path / "run_interrupted"
    flaky_client = SurveyStubClient(responses, fail_on_survey_substring=q_wrapped)
    with pytest.raises(asyncio.CancelledError):
        run(
            poll(
                df,
                "seed",
                questions=questions,
                save_dir=str(dir_interrupted),
                client=flaky_client,
                n_questions_per_run=2,
                max_retries=0,
            )
        )
    # Batch 1 (60 personas + 60 batch-1 survey calls) succeeded; only batch 2 (60 calls) remains
    flaky_client.fail_on_survey_substring = None
    calls_before_resume = flaky_client.calls
    res_resumed = run(
        poll(
            df,
            "seed",
            questions=questions,
            save_dir=str(dir_interrupted),
            client=flaky_client,
            n_questions_per_run=2,
            reset_files=False,
        )
    )
    assert flaky_client.calls - calls_before_resume == 60
    assert list(res_resumed[q_scale]) == list(res_a[q_scale])
    assert list(res_resumed[q_wrapped]) == ["wrapped_yes"] * 60

    # 4. Changing random_seed invalidates manifest, re-samples cached answers_raw.csv with 0 calls
    calls_before_reseed = client_a.calls
    res_seed_1 = run(
        poll(
            df,
            "seed",
            questions=questions,
            save_dir=str(dir_a),
            client=client_a,
            n_questions_per_run=2,
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
            n_questions_per_run=2,
            random_seed=202,
            reset_files=False,
        )
    )
    assert client_a.calls == calls_before_reseed
    assert list(res_seed_1[q_scale]) != list(res_seed_2[q_scale])
    assert res_seed_1[q_scale].value_counts().to_dict().get(5, 0) >= 45
    assert res_seed_2[q_scale].value_counts().to_dict().get(5, 0) >= 45


def test_poll_resolves_shortened_and_normalized_question_keys_safely(tmp_path):
    df = pd.DataFrame({"seed": ["Alice", "Bob", "Carol"]})
    q_bullet = "How often do you grill outdoors?"
    q_nested = (
        "[Q14] How interested are you in cold smoking at home? "
        "(Scale: 1 (Not at all) to 5 (Extremely))"
    )
    q_short_en = "[Q1] Age? (in years)"
    q_short_zh = "[Q2] 年龄 (周岁)"
    q_num_a = "1. Rate continuous smoke output (If present)"
    q_num_b = "2. Rate continuous smoke output (If absent)"
    q_ambig_a = "[Q54] Rate chamber insulation (If present)"
    q_ambig_b = "[Q55] Rate chamber insulation (If absent)"
    questions = [
        q_bullet,
        q_nested,
        q_short_en,
        q_short_zh,
        q_num_a,
        q_num_b,
        q_ambig_a,
        q_ambig_b,
    ]

    responses = {
        "Alice": {
            "- how often do you grill outdoors? ": [["Weekly", 1.0]],
            # Dropped [Q14] and nested parenthetical hint
            "How interested are you in cold smoking at home?": [[5, 1.0]],
            # Short English and CJK stems
            "Age": [[34, 1.0]],
            "年龄": [[34, 1.0]],
            # Numbered siblings with trailing parentheticals dropped
            "1. Rate continuous smoke output": [["Must-be", 1.0]],
            "2. Rate continuous smoke output": [["Dislike", 1.0]],
            # Bracket-tagged siblings with trailing parentheticals dropped
            "[Q54] Rate chamber insulation": [["Attractive", 1.0]],
            "[Q55] Rate chamber insulation": [["Neutral", 1.0]],
        },
        "Bob": {
            "- How often do you grill outdoors.": [["Monthly", 1.0]],
            # Keyed by leading ID tag alone
            "Q14.": [[3, 1.0]],
            "Q1": [[49, 1.0]],
            "Q2": [[49, 1.0]],
            # Ambiguous bare stem matches BOTH Q54 and Q55 -> must NOT cross-contaminate
            "Rate chamber insulation": [["Ambiguous", 1.0]],
            # Wrong tag [Q99] must NEVER match Q54 or Q55
            "[Q99] Rate continuous smoke output": [["WrongTag", 1.0]],
        },
        "Carol": {
            "How often do you grill outdoors?": [["Daily", 1.0]],
            "[Q14] How interested are you in cold smoking at home?": [[4, 1.0]],
            "[Q1] Age? (in years)": [[28, 1.0]],
            "[Q2] 年龄 (周岁)": [[28, 1.0]],
            "1. Rate continuous smoke output (If present)": [["Like", 1.0]],
            "2. Rate continuous smoke output (If absent)": [["Live with", 1.0]],
            # Q54 matched exact, Q55 omitted, plus stray bare stem & duplicate Q54 key -> Q55 stays None
            q_ambig_a: [["ExactQ54", 1.0]],
            "[Q54] Rate chamber insulation": [["DupQ54", 1.0]],
            "Rate chamber insulation": [["StrayBareStem", 1.0]],
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

    assert list(res[q_bullet]) == ["Weekly", "Monthly", "Daily"]
    assert list(res[q_nested]) == [5, 3, 4]
    assert list(res[q_short_en]) == [34, 49, 28]
    assert list(res[q_short_zh]) == [34, 49, 28]
    assert res[q_num_a].iloc[0] == "Must-be"
    assert res[q_num_b].iloc[0] == "Dislike"
    assert pd.isna(res[q_num_a].iloc[1])
    assert pd.isna(res[q_num_b].iloc[1])
    assert res[q_ambig_a].iloc[0] == "Attractive"
    assert res[q_ambig_b].iloc[0] == "Neutral"
    assert pd.isna(res[q_ambig_a].iloc[1])
    assert pd.isna(res[q_ambig_b].iloc[1])
    assert res[q_ambig_a].iloc[2] == "ExactQ54"
    assert pd.isna(res[q_ambig_b].iloc[2])


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
        # Alice keys by short question_id (including trailing period normalization)
        "Alice": {
            "smoke_interest.": [[5, 1.0]],
            "preferred_fuel": [["Pellets", 1.0]],
        },
        # Bob echoes full question stem on one item and ID on the other
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


def test_poll_tier_basic_prevents_key_stealing_and_mapping_order_updates_signature(
    tmp_path,
):
    # 1. Mapping mode: duplicate question text cannot be stolen by q2 in Tier 1
    df = pd.DataFrame({"seed": ["Alice"]})
    map_dup = {"q1": "Rate insulation", "q2": "Rate insulation"}
    map_res = run(
        poll(
            df,
            "seed",
            questions=map_dup,
            save_dir=str(tmp_path / "map_dup"),
            client=SurveyStubClient(
                {
                    "Alice": {
                        "q1": [["A", 1.0]],
                        "Rate insulation": [["Ambiguous", 1.0]],
                    }
                }
            ),
        )
    )
    assert map_res["q1"].iloc[0] == "A"
    assert pd.isna(map_res["q2"].iloc[0])

    # 2. List mode: normalized punctuation duplicate cannot be stolen by second question
    list_dup = ["Rate insulation?", "Rate insulation."]
    list_res = run(
        poll(
            df,
            "seed",
            questions=list_dup,
            save_dir=str(tmp_path / "list_dup"),
            client=SurveyStubClient(
                {
                    "Alice": {
                        "Rate insulation?": [["A", 1.0]],
                        "rate insulation": [["Dup", 1.0]],
                    }
                }
            ),
        )
    )
    assert list_res[list_dup[0]].iloc[0] == "A"
    assert pd.isna(list_res[list_dup[1]].iloc[0])

    # 3. Reordering Mapping keys invalidates poll_state.json and updates column order
    order_dir = str(tmp_path / "map_order")
    order_client = SurveyStubClient(
        {"Alice": {"a": [["Alpha", 1.0]], "b": [["Beta", 1.0]]}}
    )
    first = run(
        poll(
            df,
            "seed",
            questions={"b": "Question B?", "a": "Question A?"},
            save_dir=order_dir,
            client=order_client,
        )
    )
    assert list(first.columns) == ["seed", "persona", "b", "a"]

    reordered = run(
        poll(
            df,
            "seed",
            questions={"a": "Question A?", "b": "Question B?"},
            save_dir=order_dir,
            client=order_client,
            reset_files=False,
        )
    )
    assert list(reordered.columns) == ["seed", "persona", "a", "b"]
    assert reordered["a"].iloc[0] == "Alpha"
    assert reordered["b"].iloc[0] == "Beta"
