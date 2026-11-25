"""Offline behavioral tests for KV-cache-friendly prefix-first prompts,
configurable key ordering (``order_strategy``), and run-major dispatch.

All tests inject fake recording ``LLMClient`` implementations at the public
task API seam (zero network, zero API keys).
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
from typing import Any

import pandas as pd
import pytest
from conftest import RecordingClient

from leximeter import classify, codify, extract, rank, rate
from leximeter.tasks.classify import Classify, ClassifyConfig


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def _extract_json_block_keys(prompt: str, begin: str, end: str) -> list[str]:
    """Parse key order directly from a rendered BEGIN..END JSON block."""
    start = prompt.index(begin) + len(begin)
    stop = prompt.index(end, start)
    block = prompt[start:stop].strip()
    parsed = json.loads(block)
    assert isinstance(parsed, dict)
    return list(parsed.keys())


def _extract_reminder_keys(prompt: str, line_prefix: str) -> list[str]:
    """Parse key order from the trailing Python-list reminder line."""
    m = re.search(rf"{re.escape(line_prefix)}\s*(\[[^\n]+\])", prompt)
    assert m is not None, f"Missing reminder line {line_prefix!r} in prompt"
    parsed = ast.literal_eval(m.group(1))
    assert isinstance(parsed, list)
    return [str(k) for k in parsed]


def _prefix_before(prompt: str, marker: str) -> str:
    idx = prompt.index(marker)
    return prompt[:idx]


# ---------------------------------------------------------------------------
# Issue #10 — Slice 10a: rate (AttributeTask)
# ---------------------------------------------------------------------------


def test_rate_prefix_first_layout_shared_across_items(tmp_path):
    df = pd.DataFrame(
        {
            "passage": [
                "ITEM_ALPHA discusses fiscal policy.",
                "ITEM_BETA discusses marine biology.",
                "ITEM_GAMMA discusses classical music.",
            ]
        }
    )
    attrs = {
        "clarity": "how clear the writing is",
        "urgency": "how urgent the tone is",
        "rigor": "how analytical the argument is",
    }
    client = RecordingClient({"clarity": 50, "urgency": 40, "rigor": 60})

    res = run(
        rate(
            df,
            "passage",
            attributes=attrs,
            save_dir=str(tmp_path / "rate_prefix"),
            client=client,
            additional_instructions="Calibrate carefully against academic prose.",
        )
    )

    assert len(client.prompts) == 3
    assert list(res.columns[-3:]) == ["clarity", "urgency", "rigor"]

    # Static preamble + rubric + additional_instructions precede the dynamic item
    prefixes = [_prefix_before(p, "BEGIN TEXT ENTRY") for p in client.prompts]
    assert len(set(prefixes)) == 1
    shared = prefixes[0]
    assert "BEGIN ATTRIBUTES" in shared
    assert "BEGIN RATING SCALE" in shared
    assert "BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS" in shared
    assert "ITEM_ALPHA" not in shared
    assert "ITEM_BETA" not in shared
    assert "ITEM_GAMMA" not in shared

    # Rubric and trailing reminder list preserve caller order on run 1
    expected_keys = ["clarity", "urgency", "rigor"]
    for prompt in client.prompts:
        assert (
            _extract_json_block_keys(prompt, "BEGIN ATTRIBUTES", "END ATTRIBUTES")
            == expected_keys
        )
        assert (
            _extract_reminder_keys(
                prompt, "Attributes you are measuring in the content are:"
            )
            == expected_keys
        )


def test_rate_order_strategy_per_run_fixed_and_per_item(tmp_path):
    df = pd.DataFrame({"text": ["ITEM_ONE", "ITEM_TWO", "ITEM_THREE"]})
    attrs = {f"attr_{i}": f"definition {i}" for i in range(6)}
    caller_order = list(attrs.keys())
    payload = {k: 50 for k in attrs}

    # 1. Default "per_run": run 1 = caller order, run 2 & run 3 = distinct deterministic
    #    orders, identical across all items within a run, and matching rubric/reminder.
    client_per_run = RecordingClient(payload)
    run(
        rate(
            df,
            "text",
            attributes=attrs,
            save_dir=str(tmp_path / "per_run"),
            client=client_per_run,
            n_runs=3,
        )
    )
    assert len(client_per_run.prompts) == 9
    run_orders: list[list[str]] = []
    for run_idx in range(3):
        run_prompts = client_per_run.prompts[run_idx * 3 : (run_idx + 1) * 3]
        prefixes = [_prefix_before(p, "BEGIN TEXT ENTRY") for p in run_prompts]
        assert len(set(prefixes)) == 1
        keys_in_run = _extract_json_block_keys(
            run_prompts[0], "BEGIN ATTRIBUTES", "END ATTRIBUTES"
        )
        for p in run_prompts:
            assert (
                _extract_json_block_keys(p, "BEGIN ATTRIBUTES", "END ATTRIBUTES")
                == keys_in_run
            )
            assert (
                _extract_reminder_keys(
                    p, "Attributes you are measuring in the content are:"
                )
                == keys_in_run
            )
        run_orders.append(keys_in_run)

    assert run_orders[0] == caller_order
    assert run_orders[1] != caller_order
    assert run_orders[2] != caller_order
    assert run_orders[1] != run_orders[2]

    # Deterministic across independent calls
    client_per_run_repeat = RecordingClient(payload)
    run(
        rate(
            df,
            "text",
            attributes=attrs,
            save_dir=str(tmp_path / "per_run_repeat"),
            client=client_per_run_repeat,
            n_runs=3,
        )
    )
    assert client_per_run_repeat.prompts == client_per_run.prompts

    # 2. "fixed": caller order across every item and every run (identical prefix across runs)
    client_fixed = RecordingClient(payload)
    run(
        rate(
            df,
            "text",
            attributes=attrs,
            save_dir=str(tmp_path / "fixed"),
            client=client_fixed,
            n_runs=3,
            order_strategy="fixed",
        )
    )
    fixed_prefixes = [
        _prefix_before(p, "BEGIN TEXT ENTRY") for p in client_fixed.prompts
    ]
    assert len(set(fixed_prefixes)) == 1
    assert (
        _extract_json_block_keys(
            client_fixed.prompts[0], "BEGIN ATTRIBUTES", "END ATTRIBUTES"
        )
        == caller_order
    )

    # 3. "per_item": legacy per-call shuffle, while rubric and reminder still match per call
    many_df = pd.DataFrame({"text": [f"ITEM_{i}" for i in range(8)]})
    client_per_item = RecordingClient(payload)
    run(
        rate(
            many_df,
            "text",
            attributes=attrs,
            save_dir=str(tmp_path / "per_item"),
            client=client_per_item,
            n_runs=1,
            order_strategy="per_item",
        )
    )
    observed_orders = []
    for p in client_per_item.prompts:
        rubric_keys = _extract_json_block_keys(p, "BEGIN ATTRIBUTES", "END ATTRIBUTES")
        reminder_keys = _extract_reminder_keys(
            p, "Attributes you are measuring in the content are:"
        )
        assert rubric_keys == reminder_keys
        observed_orders.append(tuple(rubric_keys))
    # With 6! = 720 permutations over 8 calls, per-item shuffle produces >1 distinct order
    assert len(set(observed_orders)) > 1

    # 4. Invalid order_strategy raises ValueError
    with pytest.raises(ValueError, match="order_strategy"):
        run(
            rate(
                df,
                "text",
                attributes=attrs,
                save_dir=str(tmp_path / "invalid"),
                client=RecordingClient(payload),
                order_strategy="bogus",  # type: ignore[arg-type]
            )
        )


def test_rate_dispatches_run_major_batch_then_run_then_item(tmp_path):
    df = pd.DataFrame({"text": ["ITEM_A", "ITEM_B"]})
    attrs = {"a1": "d1", "a2": "d2", "b1": "d3", "b2": "d4"}
    client = RecordingClient({"a1": 10, "a2": 20, "b1": 30, "b2": 40})

    run(
        rate(
            df,
            "text",
            attributes=attrs,
            save_dir=str(tmp_path / "rate_run_major"),
            client=client,
            n_runs=2,
            n_attributes_per_run=2,
            n_parallels=1,
        )
    )

    # Expected order: batch0(run1: A, B; run2: A, B), batch1(run1: A, B; run2: A, B)
    assert len(client.prompts) == 8
    observed = [
        (
            tuple(_extract_json_block_keys(p, "BEGIN ATTRIBUTES", "END ATTRIBUTES")),
            "ITEM_A" if "ITEM_A" in p else "ITEM_B",
        )
        for p in client.prompts
    ]
    assert observed == [
        (("a1", "a2"), "ITEM_A"),
        (("a1", "a2"), "ITEM_B"),
        (("a2", "a1"), "ITEM_A"),
        (("a2", "a1"), "ITEM_B"),
        (("b1", "b2"), "ITEM_A"),
        (("b1", "b2"), "ITEM_B"),
        (("b2", "b1"), "ITEM_A"),
        (("b2", "b1"), "ITEM_B"),
    ]


# ---------------------------------------------------------------------------
# Issue #10 — Slice 10b: classify (single & differentiate paired mode)
# ---------------------------------------------------------------------------


def test_classify_prefix_first_order_strategy_and_run_major(tmp_path):
    df = pd.DataFrame({"text": ["DOC_ONE", "DOC_TWO"]})
    labels = {
        "l1": "label 1",
        "l2": "label 2",
        "l3": "label 3",
        "l4": "label 4",
    }
    payload = {k: True for k in labels}

    # Run-major dispatch (batch -> run -> item) + default "per_run" ordering
    client = RecordingClient(payload)
    res = run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(tmp_path / "cls_run_major"),
            client=client,
            n_runs=2,
            n_attributes_per_run=2,
            n_parallels=1,
            additional_instructions="Mark True only on explicit evidence.",
        )
    )
    assert list(res["l1"]) == [True, True]
    assert len(client.prompts) == 8

    observed = []
    for p in client.prompts:
        shared = _prefix_before(p, "BEGIN TEXT ENTRY")
        assert "BEGIN LABELS" in shared
        assert "BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS" in shared
        rubric_keys = _extract_json_block_keys(p, "BEGIN LABELS", "END LABELS")
        reminder_keys = _extract_reminder_keys(p, "Labels you are evaluating are:")
        assert rubric_keys == reminder_keys
        observed.append(
            (tuple(rubric_keys), "DOC_ONE" if "DOC_ONE" in p else "DOC_TWO")
        )

    assert observed == [
        (("l1", "l2"), "DOC_ONE"),
        (("l1", "l2"), "DOC_TWO"),
        (("l2", "l1"), "DOC_ONE"),
        (("l2", "l1"), "DOC_TWO"),
        (("l3", "l4"), "DOC_ONE"),
        (("l3", "l4"), "DOC_TWO"),
        (("l4", "l3"), "DOC_ONE"),
        (("l4", "l3"), "DOC_TWO"),
    ]
    # Identical prefix across items within each (batch, run)
    for group_start in range(0, 8, 2):
        assert _prefix_before(
            client.prompts[group_start], "BEGIN TEXT ENTRY"
        ) == _prefix_before(client.prompts[group_start + 1], "BEGIN TEXT ENTRY")

    # "fixed" keeps identical prefix across runs
    client_fixed = RecordingClient(payload)
    run(
        classify(
            df,
            "text",
            labels=labels,
            save_dir=str(tmp_path / "cls_fixed"),
            client=client_fixed,
            n_runs=2,
            order_strategy="fixed",
        )
    )
    assert (
        len({_prefix_before(p, "BEGIN TEXT ENTRY") for p in client_fixed.prompts}) == 1
    )

    # "per_item" shuffles per call while rubric and reminder still match
    many_df = pd.DataFrame({"text": [f"DOC_{i}" for i in range(8)]})
    six_labels = {f"lbl_{i}": f"def {i}" for i in range(6)}
    client_per_item = RecordingClient({k: False for k in six_labels})
    run(
        classify(
            many_df,
            "text",
            labels=six_labels,
            save_dir=str(tmp_path / "cls_per_item"),
            client=client_per_item,
            order_strategy="per_item",
        )
    )
    orders = []
    for p in client_per_item.prompts:
        rk = _extract_json_block_keys(p, "BEGIN LABELS", "END LABELS")
        assert rk == _extract_reminder_keys(p, "Labels you are evaluating are:")
        orders.append(tuple(rk))
    assert len(set(orders)) > 1


def test_classify_differentiate_circle_first_varies_only_in_suffix(tmp_path):
    df = pd.DataFrame(
        {
            "circle": ["CIRCLE_TEXT_1", "CIRCLE_TEXT_2"],
            "square": ["SQUARE_TEXT_1", "SQUARE_TEXT_2"],
        }
    )
    labels = {
        "circle more urgent": "circle shows more urgency than square",
        "square more urgent": "square shows more urgency than circle",
    }
    payload = {k: False for k in labels}

    client_true = RecordingClient(payload)
    run(
        Classify(
            ClassifyConfig(
                labels=labels,
                save_dir=str(tmp_path / "cls_diff_true"),
                differentiate=True,
                circle_first=True,
                additional_instructions="Focus on direct urgency cues.",
            ),
            client=client_true,
        ).run(df, circle_column_name="circle", square_column_name="square")
    )

    client_false = RecordingClient(payload)
    run(
        Classify(
            ClassifyConfig(
                labels=labels,
                save_dir=str(tmp_path / "cls_diff_false"),
                differentiate=True,
                circle_first=False,
                additional_instructions="Focus on direct urgency cues.",
            ),
            client=client_false,
        ).run(df, circle_column_name="circle", square_column_name="square")
    )

    marker = "Consider the following two distinct text entries:"
    prefixes_true = [_prefix_before(p, marker) for p in client_true.prompts]
    prefixes_false = [_prefix_before(p, marker) for p in client_false.prompts]
    assert len(set(prefixes_true + prefixes_false)) == 1
    shared = prefixes_true[0]
    assert "BEGIN LABELS" in shared
    assert "BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS" in shared
    assert "You are specifically looking at features that distinguish" in shared

    # Suffix flips circle/square order
    suffix_true = client_true.prompts[0][client_true.prompts[0].index(marker) :]
    suffix_false = client_false.prompts[0][client_false.prompts[0].index(marker) :]
    assert suffix_true != suffix_false
    assert suffix_true.index("BEGIN ENTRY CIRCLE") < suffix_true.index(
        "BEGIN ENTRY SQUARE"
    )
    assert suffix_false.index("BEGIN ENTRY SQUARE") < suffix_false.index(
        "BEGIN ENTRY CIRCLE"
    )


# ---------------------------------------------------------------------------
# Issue #10 — Slice 10c: extract
# ---------------------------------------------------------------------------


def test_extract_prefix_first_order_strategy_and_run_major(tmp_path):
    df = pd.DataFrame({"product": ["ENTITY_ONE", "ENTITY_TWO"]})
    attrs = {
        "company": "manufacturer name",
        "year": "launch year",
        "country": "country of origin",
        "price": "launch price",
    }
    payload = {
        "ENTITY_ONE": {
            "company": "Acme",
            "year": "2020",
            "country": "US",
            "price": "10",
        }
    }

    # Default modality="entity", n_runs=2, 2 attribute batches -> run-major dispatch
    client = RecordingClient(payload)
    res = run(
        extract(
            df,
            "product",
            attributes=attrs,
            save_dir=str(tmp_path / "ext_run_major"),
            client=client,
            n_runs=2,
            n_attributes_per_run=2,
            n_parallels=1,
            additional_instructions="Prefer primary headquarters.",
        )
    )
    assert list(res.columns) == [
        "product",
        "entity_name",
        "company",
        "year",
        "country",
        "price",
    ]
    assert len(client.prompts) == 8

    observed = []
    for p in client.prompts:
        shared = _prefix_before(p, "Entity: ")
        assert "BEGIN ATTRIBUTES" in shared
        assert "BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS" in shared
        rk = _extract_json_block_keys(p, "BEGIN ATTRIBUTES", "END ATTRIBUTES")
        assert rk == _extract_reminder_keys(
            p, "Attributes you are extracting info on are:"
        )
        observed.append(
            (tuple(rk), "ENTITY_ONE" if "ENTITY_ONE" in p else "ENTITY_TWO")
        )

    assert observed == [
        (("company", "year"), "ENTITY_ONE"),
        (("company", "year"), "ENTITY_TWO"),
        (("year", "company"), "ENTITY_ONE"),
        (("year", "company"), "ENTITY_TWO"),
        (("country", "price"), "ENTITY_ONE"),
        (("country", "price"), "ENTITY_TWO"),
        (("price", "country"), "ENTITY_ONE"),
        (("price", "country"), "ENTITY_TWO"),
    ]
    for group_start in range(0, 8, 2):
        assert _prefix_before(
            client.prompts[group_start], "Entity: "
        ) == _prefix_before(client.prompts[group_start + 1], "Entity: ")

    # "fixed" keeps identical prefix across runs (text modality)
    client_fixed = RecordingClient(payload)
    run(
        extract(
            df,
            "product",
            attributes=attrs,
            save_dir=str(tmp_path / "ext_fixed"),
            client=client_fixed,
            n_runs=2,
            modality="text",
            order_strategy="fixed",
        )
    )
    assert (
        len({_prefix_before(p, "BEGIN TEXT ENTRY") for p in client_fixed.prompts}) == 1
    )

    # "per_item" shuffles per call while keeping rubric/reminder aligned
    many_df = pd.DataFrame({"product": [f"ENT_{i}" for i in range(8)]})
    six_attrs = {f"f_{i}": f"desc {i}" for i in range(6)}
    client_per_item = RecordingClient({k: "val" for k in six_attrs})
    run(
        extract(
            many_df,
            "product",
            attributes=six_attrs,
            save_dir=str(tmp_path / "ext_per_item"),
            client=client_per_item,
            order_strategy="per_item",
        )
    )
    orders = []
    for p in client_per_item.prompts:
        rk = _extract_json_block_keys(p, "BEGIN ATTRIBUTES", "END ATTRIBUTES")
        assert rk == _extract_reminder_keys(
            p, "Attributes you are extracting info on are:"
        )
        orders.append(tuple(rk))
    assert len(set(orders)) > 1


# ---------------------------------------------------------------------------
# Issue #10 — Slice 10d: rank
# ---------------------------------------------------------------------------


def test_rank_prefix_first_circle_first_suffix_and_order_strategy(tmp_path):
    df = pd.DataFrame({"speech": ["SPEECH_A", "SPEECH_B", "SPEECH_C", "SPEECH_D"]})
    attrs = {f"dim_{i}": f"definition {i}" for i in range(5)}
    caller_order = list(attrs.keys())
    payload = {k: "circle" for k in attrs}
    marker = "Consider the following two distinct text entries:"

    # 1. circle_first=True vs False: identical prefix, suffix-only flip
    client_true = RecordingClient(payload)
    run(
        rank(
            df,
            "speech",
            attributes=attrs,
            save_dir=str(tmp_path / "rank_cf_true"),
            client=client_true,
            n_rounds=1,
            matches_per_round=1,
            circle_first=True,
            additional_instructions="Compare rhetoric directly.",
        )
    )
    client_false = RecordingClient(payload)
    run(
        rank(
            df,
            "speech",
            attributes=attrs,
            save_dir=str(tmp_path / "rank_cf_false"),
            client=client_false,
            n_rounds=1,
            matches_per_round=1,
            circle_first=False,
            additional_instructions="Compare rhetoric directly.",
        )
    )
    assert len(client_true.prompts) >= 2
    assert len(client_false.prompts) >= 2
    all_prefixes = [_prefix_before(p, marker) for p in client_true.prompts] + [
        _prefix_before(p, marker) for p in client_false.prompts
    ]
    assert len(set(all_prefixes)) == 1
    shared = all_prefixes[0]
    assert "BEGIN ATTRIBUTES" in shared
    assert "BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS" in shared

    suf_t = client_true.prompts[0][client_true.prompts[0].index(marker) :]
    suf_f = client_false.prompts[0][client_false.prompts[0].index(marker) :]
    assert suf_t.index("BEGIN ENTRY CIRCLE") < suf_t.index("BEGIN ENTRY SQUARE")
    assert suf_f.index("BEGIN ENTRY SQUARE") < suf_f.index("BEGIN ENTRY CIRCLE")

    # 2. Default "per_run" across 3 rounds: distinct deterministic order per round,
    #    shared across all pairs within a round, matching rubric/reminder.
    client_per_run = RecordingClient(payload)
    run(
        rank(
            df,
            "speech",
            attributes=attrs,
            save_dir=str(tmp_path / "rank_per_run"),
            client=client_per_run,
            n_rounds=3,
            matches_per_round=1,
            n_parallels=1,
        )
    )
    round_orders: list[tuple[str, ...]] = []
    for p in client_per_run.prompts:
        rk = _extract_json_block_keys(p, "BEGIN ATTRIBUTES", "END ATTRIBUTES")
        assert rk == _extract_reminder_keys(p, "Attributes you are adjudicating are:")
        tup = tuple(rk)
        if not round_orders or round_orders[-1] != tup:
            round_orders.append(tup)
    assert len(round_orders) == 3
    assert list(round_orders[0]) == caller_order
    assert list(round_orders[1]) == list(reversed(caller_order))
    assert round_orders[2] != round_orders[0]
    assert round_orders[2] != round_orders[1]

    # 3. "fixed": identical prefix across all rounds
    client_fixed = RecordingClient(payload)
    run(
        rank(
            df,
            "speech",
            attributes=attrs,
            save_dir=str(tmp_path / "rank_fixed"),
            client=client_fixed,
            n_rounds=3,
            matches_per_round=1,
            order_strategy="fixed",
        )
    )
    assert len({_prefix_before(p, marker) for p in client_fixed.prompts}) == 1

    # 4. "per_item": shuffles per pair call while keeping rubric/reminder aligned
    client_per_item = RecordingClient(payload)
    run(
        rank(
            df,
            "speech",
            attributes=attrs,
            save_dir=str(tmp_path / "rank_per_item"),
            client=client_per_item,
            n_rounds=2,
            matches_per_round=2,
            order_strategy="per_item",
        )
    )
    per_item_orders = []
    for p in client_per_item.prompts:
        rk = _extract_json_block_keys(p, "BEGIN ATTRIBUTES", "END ATTRIBUTES")
        assert rk == _extract_reminder_keys(p, "Attributes you are adjudicating are:")
        per_item_orders.append(tuple(rk))
    assert len(set(per_item_orders)) > 1


# ---------------------------------------------------------------------------
# Issue #10 — Slice 10e: codify (static & dynamic modes)
# ---------------------------------------------------------------------------


def test_codify_prefix_first_order_strategy_and_batch_grouping(tmp_path):
    df = pd.DataFrame(
        {
            "text": [
                "PASSAGE_ONE mentions taxes and clean water policies in detail.",
                "PASSAGE_TWO mentions inflation and river conservation efforts.",
            ]
        }
    )
    categories = {
        "economy": "economic or fiscal topics",
        "environment": "environmental or ecological topics",
        "health": "healthcare topics",
        "education": "schooling topics",
    }
    payload = {
        "economy": [{"beginning excerpt": "mentions taxes", "ending excerpt": ""}],
        "environment": [],
        "health": [],
        "education": [],
    }

    # 1. Static mode, 2 category batches, n_rounds=1: batch-major + prefix-first + matching JSON/reminder
    client = RecordingClient(payload)
    res = run(
        codify(
            df,
            "text",
            categories=categories,
            additional_instructions="Mark concise phrase-level spans.",
            save_dir=str(tmp_path / "codify_static"),
            client=client,
            n_rounds=1,
            max_categories_per_call=2,
            n_parallels=1,
        )
    )
    assert list(categories.keys()) == list(res.columns[-4:])
    assert len(client.prompts) == 4
    observed = []
    for p in client.prompts:
        shared = _prefix_before(p, "BEGIN TEXT ENTRY")
        assert "BEGIN CATEGORIES/ATTRIBUTES" in shared
        assert "BEGIN ADDITIONAL INSTRUCTIONS" in shared
        rk = _extract_json_block_keys(
            p, "BEGIN CATEGORIES/ATTRIBUTES", "END CATEGORIES/ATTRIBUTES"
        )
        assert rk == _extract_reminder_keys(
            p, "Again: the categories you are marking are:"
        )
        observed.append(
            (tuple(rk), "PASSAGE_ONE" if "PASSAGE_ONE" in p else "PASSAGE_TWO")
        )
    assert observed == [
        (("economy", "environment"), "PASSAGE_ONE"),
        (("economy", "environment"), "PASSAGE_TWO"),
        (("health", "education"), "PASSAGE_ONE"),
        (("health", "education"), "PASSAGE_TWO"),
    ]
    assert _prefix_before(client.prompts[0], "BEGIN TEXT ENTRY") == _prefix_before(
        client.prompts[1], "BEGIN TEXT ENTRY"
    )
    assert _prefix_before(client.prompts[2], "BEGIN TEXT ENTRY") == _prefix_before(
        client.prompts[3], "BEGIN TEXT ENTRY"
    )

    # 2. Multi-round completion sweep: "per_run" reverses order on iteration 1 vs "fixed" keeps caller order
    class TwoRoundCodifyClient(RecordingClient):
        async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
            self.prompts.append(prompt)
            if "BEGIN LABELS" in prompt:
                return json.dumps({"c1": True, "c2": True, "c3": True})
            return json.dumps(
                {
                    "c1": [{"beginning excerpt": "Hello world", "ending excerpt": ""}],
                    "c2": [],
                    "c3": [],
                }
            )

    long_df = pd.DataFrame(
        {
            "text": [
                "Intro text. Hello world. Remaining uncoded discussion continues here."
            ]
        }
    )
    cats3 = {"c1": "d1", "c2": "d2", "c3": "d3"}

    c_per_run = TwoRoundCodifyClient({})
    run(
        codify(
            long_df,
            "text",
            categories=cats3,
            save_dir=str(tmp_path / "codify_per_run"),
            client=c_per_run,
            n_rounds=2,
            order_strategy="per_run",
        )
    )
    codify_prompts_pr = [
        p for p in c_per_run.prompts if "BEGIN CATEGORIES/ATTRIBUTES" in p
    ]
    assert len(codify_prompts_pr) == 2
    assert _extract_json_block_keys(
        codify_prompts_pr[0], "BEGIN CATEGORIES/ATTRIBUTES", "END CATEGORIES/ATTRIBUTES"
    ) == ["c1", "c2", "c3"]
    assert _extract_json_block_keys(
        codify_prompts_pr[1], "BEGIN CATEGORIES/ATTRIBUTES", "END CATEGORIES/ATTRIBUTES"
    ) == ["c3", "c2", "c1"]

    c_fixed = TwoRoundCodifyClient({})
    run(
        codify(
            long_df,
            "text",
            categories=cats3,
            save_dir=str(tmp_path / "codify_fixed"),
            client=c_fixed,
            n_rounds=2,
            order_strategy="fixed",
        )
    )
    codify_prompts_fx = [
        p for p in c_fixed.prompts if "BEGIN CATEGORIES/ATTRIBUTES" in p
    ]
    assert len(codify_prompts_fx) == 2
    assert _prefix_before(codify_prompts_fx[0], "BEGIN TEXT ENTRY") == _prefix_before(
        codify_prompts_fx[1], "BEGIN TEXT ENTRY"
    )

    # 3. "per_item" shuffles per call while keeping JSON rubric & reminder aligned
    many_df = pd.DataFrame({"text": [f"TEXT_UNIT_{i} body" for i in range(8)]})
    six_cats = {f"code_{i}": f"def {i}" for i in range(6)}
    c_per_item = RecordingClient({k: [] for k in six_cats})
    run(
        codify(
            many_df,
            "text",
            categories=six_cats,
            save_dir=str(tmp_path / "codify_per_item"),
            client=c_per_item,
            n_rounds=1,
            order_strategy="per_item",
        )
    )
    orders = []
    for p in c_per_item.prompts:
        rk = _extract_json_block_keys(
            p, "BEGIN CATEGORIES/ATTRIBUTES", "END CATEGORIES/ATTRIBUTES"
        )
        assert rk == _extract_reminder_keys(
            p, "Again: the categories you are marking are:"
        )
        orders.append(tuple(rk))
    assert len(set(orders)) > 1

    # 4. Dynamic mode: shared prefix across items
    c_dyn = RecordingClient({"discovered": []})
    run(
        codify(
            df,
            "text",
            categories=None,
            additional_instructions="Discover policy domains.",
            save_dir=str(tmp_path / "codify_dynamic"),
            client=c_dyn,
            n_rounds=1,
        )
    )
    dyn_prefixes = [_prefix_before(p, "BEGIN TEXT ENTRY") for p in c_dyn.prompts]
    assert len(set(dyn_prefixes)) == 1
    assert "BEGIN USER INSTRUCTIONS" in dyn_prefixes[0]
