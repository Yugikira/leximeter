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

from leximeter import (
    bucket,
    classify,
    codify,
    compare,
    deduplicate,
    deidentify,
    extract,
    merge,
    paraphrase,
    poll,
    rank,
    rate,
    seed,
)
from leximeter import (
    filter as lex_filter,
)
from leximeter.tasks._common import order_mapping
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


# ---------------------------------------------------------------------------
# Issue #11 — Slice 11a: compare & paraphrase
# ---------------------------------------------------------------------------


def test_compare_prefix_first_and_circle_first_suffix_only(tmp_path):
    df = pd.DataFrame(
        {
            "circle": ["CIRCLE_A", "CIRCLE_B", "CIRCLE_C"],
            "square": ["SQUARE_A", "SQUARE_B", "SQUARE_C"],
        }
    )
    marker = "Consider the following two distinct text entries:"
    payload = {"circle mentions trade more than square": "Direct evidence."}

    c_true = RecordingClient(payload)
    run(
        compare(
            df,
            "circle",
            "square",
            save_dir=str(tmp_path / "cmp_true"),
            client=c_true,
            differentiate=True,
            circle_first=True,
            additional_instructions="Focus on trade policy.",
        )
    )
    c_false = RecordingClient(payload)
    run(
        compare(
            df,
            "circle",
            "square",
            save_dir=str(tmp_path / "cmp_false"),
            client=c_false,
            differentiate=True,
            circle_first=False,
            additional_instructions="Focus on trade policy.",
        )
    )

    prefixes = [_prefix_before(p, marker) for p in c_true.prompts + c_false.prompts]
    assert len(set(prefixes)) == 1
    shared = prefixes[0]
    assert "Output JSON only" in shared
    assert "BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS" in shared
    assert "CIRCLE_A" not in shared
    assert "SQUARE_A" not in shared

    suf_t = c_true.prompts[0][c_true.prompts[0].index(marker) :]
    suf_f = c_false.prompts[0][c_false.prompts[0].index(marker) :]
    assert suf_t.index("BEGIN ENTRY CIRCLE") < suf_t.index("BEGIN ENTRY SQUARE")
    assert suf_f.index("BEGIN ENTRY SQUARE") < suf_f.index("BEGIN ENTRY CIRCLE")

    # circle_first=None randomizes across pairs while keeping the shared prefix unbroken
    many_df = pd.DataFrame(
        {
            "circle": [f"C_{i}" for i in range(16)],
            "square": [f"S_{i}" for i in range(16)],
        }
    )
    c_rand = RecordingClient(payload)
    run(
        compare(
            many_df,
            "circle",
            "square",
            save_dir=str(tmp_path / "cmp_rand"),
            client=c_rand,
            differentiate=False,
            circle_first=None,
        )
    )
    rand_prefixes = [_prefix_before(p, marker) for p in c_rand.prompts]
    assert len(set(rand_prefixes)) == 1
    assert "similar to entry" in rand_prefixes[0]
    orientations = {
        p.index("BEGIN ENTRY CIRCLE") < p.index("BEGIN ENTRY SQUARE")
        for p in c_rand.prompts
    }
    assert orientations == {True, False}


def test_paraphrase_prefix_first_and_no_stale_multimodal_guidance(tmp_path):
    df = pd.DataFrame({"text": ["RAW_PASSAGE_1", "RAW_PASSAGE_2", "RAW_PASSAGE_3"]})
    client = RecordingClient({})

    run(
        paraphrase(
            df,
            "text",
            instructions="Rewrite in neutral academic prose.",
            save_dir=str(tmp_path / "para_text"),
            client=client,
            modality="text",
        )
    )
    assert len(client.prompts) == 3
    prefixes = [_prefix_before(p, "BEGIN TEXT ENTRY") for p in client.prompts]
    assert len(set(prefixes)) == 1
    shared = prefixes[0]
    assert "BEGIN INSTRUCTIONS" in shared
    assert "Rewrite in neutral academic prose." in shared
    assert "RAW_PASSAGE_1" not in shared

    # ADR-0002 text-only scope: stale multimodal lines must be absent
    for prompt in client.prompts:
        assert "image/audio/pdf" not in prompt.lower()
        assert "- web:" not in prompt.lower()
        assert "search term" not in prompt.lower()

    # Entity modality also places Entity: in the trailing suffix
    ent_df = pd.DataFrame({"entity": ["Apple Inc.", "Alphabet Inc."]})
    ent_client = RecordingClient({})
    run(
        paraphrase(
            ent_df,
            "entity",
            instructions="Standardize company names.",
            save_dir=str(tmp_path / "para_entity"),
            client=ent_client,
            modality="entity",
            json_mode=True,
        )
    )
    ent_prefixes = [_prefix_before(p, "Entity: ") for p in ent_client.prompts]
    assert len(set(ent_prefixes)) == 1
    assert "Always output JSON only" in ent_prefixes[0]


# ---------------------------------------------------------------------------
# Issue #11 — Slice 11b: poll (persona + survey) & deidentify
# ---------------------------------------------------------------------------


def test_poll_persona_and_survey_prompts_are_prefix_first(tmp_path):
    df = pd.DataFrame({"seed": ["SEED_NURSE_OHIO", "SEED_FARMER_IOWA", "SEED_DEV_WA"]})

    class PollRecordingClient(RecordingClient):
        async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
            self.prompts.append(prompt)
            if "BEGIN SURVEY QUESTIONS" in prompt:
                return json.dumps({"Trust in media?": [["medium", 1.0]]})
            # Return distinct persona bio per seed
            seed_tag = next(
                (s for s in df["seed"] if s in prompt),
                "UNKNOWN_SEED",
            )
            return f"BIO_FOR_{seed_tag}"

    client = PollRecordingClient({})
    res = run(
        poll(
            df,
            "seed",
            population_description="US adults aged 25-65",
            questions=["Trust in media?"],
            save_dir=str(tmp_path / "poll_prefix"),
            client=client,
            n_questions_per_run=8,
        )
    )
    assert list(res.columns) == ["seed", "persona", "Trust in media?"]
    assert len(client.prompts) == 6

    persona_prompts = client.prompts[:3]
    survey_prompts = client.prompts[3:]

    # 1. Persona prompts: static guidelines + population + Jenna Perez example + template precede BEGIN SEED
    persona_prefixes = [_prefix_before(p, "BEGIN SEED") for p in persona_prompts]
    assert len(set(persona_prefixes)) == 1
    p_shared = persona_prefixes[0]
    assert "US adults aged 25-65" in p_shared
    assert "Name: <insert name here>" in p_shared
    assert "SEED_NURSE_OHIO" not in p_shared

    # 2. Survey prompts: instructions + questions + rules + output format precede BEGIN PERSONA
    survey_prefixes = [_prefix_before(p, "BEGIN PERSONA") for p in survey_prompts]
    assert len(set(survey_prefixes)) == 1
    s_shared = survey_prefixes[0]
    assert "BEGIN SURVEY QUESTIONS" in s_shared
    assert "Trust in media?" in s_shared
    assert "BEGIN OUTPUT FORMAT" in s_shared
    assert "BIO_FOR_" not in s_shared


def test_deidentify_prefix_first_and_single_current_map_block(tmp_path):
    df = pd.DataFrame(
        {
            "group": ["g1", "g1"],
            "text": [
                "Alice lives in Springfield and works at Acme Labs.",
                "Later Alice visited Bob in Springfield near Acme Labs.",
            ],
        }
    )

    class GrowingMapClient(RecordingClient):
        async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
            self.prompts.append(prompt)
            if "Bob" in prompt:
                return json.dumps(
                    {
                        "person's name": {
                            "real forms": ["Alice"],
                            "casted form": "Clara",
                        },
                        "second person's name": {
                            "real forms": ["Bob"],
                            "casted form": "David",
                        },
                    }
                )
            return json.dumps(
                {"person's name": {"real forms": ["Alice"], "casted form": "Clara"}}
            )

    client = GrowingMapClient({})
    res = run(
        deidentify(
            df,
            "text",
            grouping_column="group",
            save_dir=str(tmp_path / "deid_prefix"),
            client=client,
            additional_instructions="Also anonymize local lab names.",
        )
    )
    assert len(client.prompts) == 2
    assert "Clara" in res["deidentified_text"].iloc[0]
    assert "David" in res["deidentified_text"].iloc[1]

    # Static preamble + examples + rules + additional_instructions precede dynamic current_map & text
    prefixes = [_prefix_before(p, "BEGIN CURRENT MAPPING") for p in client.prompts]
    assert len(set(prefixes)) == 1
    shared = prefixes[0]
    assert "Example mapping snippet:" in shared
    assert "Olmsted Medical Center" in shared
    assert "BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS" in shared
    assert "Also anonymize local lab names." in shared
    assert "Alice" not in shared
    assert "Clara" not in shared

    # Chunk 2 carries the updated mapping strictly once in the trailing suffix
    chunk2_suffix = client.prompts[1][
        client.prompts[1].index("BEGIN CURRENT MAPPING") :
    ]
    assert chunk2_suffix.count('"casted form": "Clara"') == 1
    assert chunk2_suffix.index("BEGIN CURRENT MAPPING") < chunk2_suffix.index(
        "BEGIN TEXT"
    )
    # No duplicate mid-prompt "Current mapping:" / "existing mapping is:" breaks
    assert "Again, the existing mapping is:" not in client.prompts[1]


# ---------------------------------------------------------------------------
# Issue #11 — Slice 11c: deduplicate, bucket, merge, filter
# ---------------------------------------------------------------------------


def test_deduplicate_prefix_first_across_groups(tmp_path):
    df = pd.DataFrame({"term": ["Alpha", "Alpha Inc", "Beta", "Beta Corp"]})
    client = RecordingClient({})
    run(
        deduplicate(
            df,
            "term",
            save_dir=str(tmp_path / "dedup_prefix"),
            client=client,
            group_size=2,
            use_embeddings=False,
            additional_instructions="Preserve legal suffixes when meaningful.",
        )
    )
    assert len(client.prompts) == 2
    prefixes = [_prefix_before(p, "BEGIN RAW TERMS") for p in client.prompts]
    assert len(set(prefixes)) == 1
    shared = prefixes[0]
    assert "Output JSON only" in shared
    assert "BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS" in shared
    assert "Alpha" not in shared
    assert "Beta" not in shared


def test_bucket_prefix_first_across_generate_reduce_and_finalize(tmp_path):
    df = pd.DataFrame({"term": ["t1", "t2", "t3", "t4", "t5", "t6"]})

    class BucketRecordingClient(RecordingClient):
        async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
            self.prompts.append(prompt)
            if "BEGIN BUCKET CANDIDATES" not in prompt:
                return json.dumps({f"bucket_{i}": f"definition {i}" for i in range(6)})
            if "BEGIN ALREADY FINALIZED BUCKETS" in prompt:
                return json.dumps({"bucket_1": "complementary"})
            return json.dumps({"bucket_0": "best", "bucket_1": "second"})

    client = BucketRecordingClient({})
    res = run(
        bucket(
            df,
            "term",
            save_dir=str(tmp_path / "bucket_prefix"),
            client=client,
            bucket_count=2,
            n_terms_per_prompt=3,
            repeat_bucketing=2,
            repeat_voting=2,
            top_k_per_round=1,
            additional_instructions="Keep buckets domain-specific.",
        )
    )
    assert len(res) == 2

    gen_prompts = [p for p in client.prompts if "BEGIN BUCKET CANDIDATES" not in p]
    vote_no_sel = [
        p
        for p in client.prompts
        if "BEGIN BUCKET CANDIDATES" in p and "BEGIN ALREADY FINALIZED BUCKETS" not in p
    ]
    vote_with_sel = [
        p
        for p in client.prompts
        if "BEGIN BUCKET CANDIDATES" in p and "BEGIN ALREADY FINALIZED BUCKETS" in p
    ]

    assert len(gen_prompts) >= 2
    assert len(vote_no_sel) >= 2
    assert len(vote_with_sel) >= 2

    # 1. Generation: static rules + JSON format + additional_instructions precede BEGIN RAW TERMS
    gen_prefixes = [_prefix_before(p, "BEGIN RAW TERMS") for p in gen_prompts]
    assert len(set(gen_prefixes)) == 1
    assert "Output JSON only" in gen_prefixes[0]
    assert "BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS" in gen_prefixes[0]

    # 2. Reduce / first finalize voting: static rules + additional_instructions precede BEGIN BUCKET CANDIDATES
    v0_prefixes = [_prefix_before(p, "BEGIN BUCKET CANDIDATES") for p in vote_no_sel]
    assert len(set(v0_prefixes)) == 1
    assert "BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS" in v0_prefixes[0]
    assert vote_no_sel[0].index("BEGIN BUCKET CANDIDATES") < vote_no_sel[0].index(
        "BEGIN RAW TERMS"
    )

    # 3. Finalize with selected_buckets: finalized buckets sit inside shared static prefix before dynamic candidates/terms
    v1_prefixes = [_prefix_before(p, "BEGIN BUCKET CANDIDATES") for p in vote_with_sel]
    assert len(set(v1_prefixes)) == 1
    assert "BEGIN ALREADY FINALIZED BUCKETS" in v1_prefixes[0]
    assert "BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS" in v1_prefixes[0]


def test_merge_prefix_first_and_order_strategy(tmp_path):
    df_left = pd.DataFrame(
        {"company": ["Acme A", "Acme B", "Acme C", "Beta A", "Beta B", "Beta C"]}
    )
    df_right = pd.DataFrame(
        {
            "name": [
                "Acme Alpha",
                "Acme Beta",
                "Acme Gamma",
                "Beta Alpha",
                "Beta Beta",
                "Beta Gamma",
            ],
            "id": [1, 2, 3, 4, 5, 6],
        }
    )
    payload = {
        "Acme A": "Acme Alpha",
        "Acme B": "Acme Beta",
        "Acme C": "Acme Gamma",
        "Beta A": "Beta Alpha",
        "Beta B": "Beta Beta",
        "Beta C": "Beta Gamma",
    }

    # 1. Default "per_run" across 2 runs and 2 short-list clusters
    client = RecordingClient(payload)
    res = run(
        merge(
            df_left,
            df_right,
            left_on="company",
            right_on="name",
            save_dir=str(tmp_path / "merge_per_run"),
            client=client,
            short_list_len=3,
            n_runs=2,
            max_attempts=1,
            use_embeddings=False,
            additional_instructions="Match legal entity names strictly.",
            n_parallels=1,
        )
    )
    assert list(res["id"]) == [1, 2, 3, 4, 5, 6]
    assert len(client.prompts) == 4

    # Static preamble + additional_instructions precede dynamic LONG LIST and SHORT LIST
    static_prefixes = [_prefix_before(p, "BEGIN LONG LIST") for p in client.prompts]
    assert len(set(static_prefixes)) == 1
    shared = static_prefixes[0]
    assert "Output JSON only" in shared
    assert "BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS" in shared

    # Shared long_list across clusters also sits before BEGIN SHORT LIST
    long_prefixes = [_prefix_before(p, "BEGIN SHORT LIST") for p in client.prompts]
    assert len(set(long_prefixes)) == 1

    # Matching order between BEGIN SHORT LIST and trailing reminder; run 2 reverses run 1
    run_orders = []
    for p in client.prompts:
        rk = _extract_json_block_keys(p, "BEGIN SHORT LIST", "END SHORT LIST")
        assert rk == _extract_reminder_keys(
            p, "Again, the short list terms you are finding matches for are:"
        )
        run_orders.append(rk)
    # Run-major: run 1 clusters [0, 1], then run 2 clusters [0, 1] reversed
    assert run_orders[0] == ["Acme A", "Acme B", "Acme C"]
    assert run_orders[1] == ["Beta A", "Beta B", "Beta C"]
    assert run_orders[2] == ["Acme C", "Acme B", "Acme A"]
    assert run_orders[3] == ["Beta C", "Beta B", "Beta A"]

    # 2. "fixed": identical short_list order across runs
    c_fixed = RecordingClient(payload)
    run(
        merge(
            df_left,
            df_right,
            left_on="company",
            right_on="name",
            save_dir=str(tmp_path / "merge_fixed"),
            client=c_fixed,
            short_list_len=3,
            n_runs=2,
            max_attempts=1,
            use_embeddings=False,
            order_strategy="fixed",
            n_parallels=1,
        )
    )
    assert c_fixed.prompts[0] == c_fixed.prompts[2]
    assert c_fixed.prompts[1] == c_fixed.prompts[3]

    # 3. "per_item": shuffles short_list per call while keeping JSON & reminder aligned
    df_left_6 = pd.DataFrame({"company": [f"Co_{i}" for i in range(6)]})
    df_right_6 = pd.DataFrame(
        {"name": [f"Co_{i}" for i in range(6)], "id": list(range(6))}
    )
    c_per_item = RecordingClient({f"Co_{i}": f"Co_{i}" for i in range(6)})
    run(
        merge(
            df_left_6,
            df_right_6,
            left_on="company",
            right_on="name",
            save_dir=str(tmp_path / "merge_per_item"),
            client=c_per_item,
            short_list_len=6,
            n_runs=8,
            max_attempts=1,
            use_embeddings=False,
            order_strategy="per_item",
        )
    )
    orders = []
    for p in c_per_item.prompts:
        rk = _extract_json_block_keys(p, "BEGIN SHORT LIST", "END SHORT LIST")
        assert rk == _extract_reminder_keys(
            p, "Again, the short list terms you are finding matches for are:"
        )
        orders.append(tuple(rk))
    assert len(set(orders)) > 1


def test_filter_prefix_first_across_entity_chunks(tmp_path):
    df = pd.DataFrame(
        {"entity": ["ENT_1", "ENT_2", "ENT_3", "ENT_4", "ENT_5", "ENT_6"]}
    )
    client = RecordingClient({"entities meeting condition": ["ENT_1"]})
    res = run(
        lex_filter(
            df,
            "entity",
            condition="is a semiconductor manufacturer",
            save_dir=str(tmp_path / "filter_prefix"),
            client=client,
            entities_per_call=2,
            additional_instructions="Include fabless designers.",
        )
    )
    assert len(res) == 6
    assert len(client.prompts) == 3

    prefixes = [_prefix_before(p, "BEGIN ENTITY LIST") for p in client.prompts]
    assert len(set(prefixes)) == 1
    shared = prefixes[0]
    assert "BEGIN CONDITION" in shared
    assert "is a semiconductor manufacturer" in shared
    assert "Output in following JSON format:" in shared
    assert "BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS" in shared
    assert "ENT_1" not in shared
    assert "ENT_6" not in shared


def test_order_mapping_per_run_distinct_permutations(tmp_path):
    # len(keys) == 3 -> all 3! = 6 permutations strictly distinct across run_idx 1..6
    m3 = {"a": 1, "b": 2, "c": 3}
    orders3 = [
        tuple(order_mapping(m3, strategy="per_run", run_idx=r, batch_idx=0).keys())
        for r in range(1, 7)
    ]
    assert len(set(orders3)) == 6
    assert orders3[0] == ("a", "b", "c")
    assert orders3[1] == ("c", "b", "a")
    # Deterministic across repeats and wraps cleanly after k!
    assert (
        tuple(order_mapping(m3, strategy="per_run", run_idx=7, batch_idx=0).keys())
        == orders3[0]
    )
    assert [
        tuple(order_mapping(m3, strategy="per_run", run_idx=r, batch_idx=0).keys())
        for r in range(1, 7)
    ] == orders3

    # len(keys) == 4 -> 10 strictly distinct orders across run_idx 1..10 (and all 24 across 1..24)
    m4 = {f"k{i}": i for i in range(4)}
    orders4_10 = [
        tuple(order_mapping(m4, strategy="per_run", run_idx=r, batch_idx=0).keys())
        for r in range(1, 11)
    ]
    assert len(set(orders4_10)) == 10
    assert orders4_10[0] == ("k0", "k1", "k2", "k3")
    assert orders4_10[1] == ("k3", "k2", "k1", "k0")
    orders4_all = [
        tuple(order_mapping(m4, strategy="per_run", run_idx=r, batch_idx=0).keys())
        for r in range(1, 25)
    ]
    assert len(set(orders4_all)) == 24

    # len(keys) > 6 (7 keys) -> 10 strictly distinct deterministic orders
    m7 = {f"k{i}": i for i in range(7)}
    orders7 = [
        tuple(order_mapping(m7, strategy="per_run", run_idx=r, batch_idx=2).keys())
        for r in range(1, 11)
    ]
    assert len(set(orders7)) == 10
    assert orders7[0] == tuple(m7.keys())
    assert orders7[1] == tuple(reversed(m7.keys()))
    assert [
        tuple(order_mapping(m7, strategy="per_run", run_idx=r, batch_idx=2).keys())
        for r in range(1, 11)
    ] == orders7

    # End-to-end: rank with 3 attributes and default n_rounds=5 gets 5 distinct orders
    df = pd.DataFrame({"speech": ["SPEECH_A", "SPEECH_B", "SPEECH_C", "SPEECH_D"]})
    attrs3 = {"clarity": "d1", "depth": "d2", "novelty": "d3"}
    client = RecordingClient({k: "circle" for k in attrs3})
    run(
        rank(
            df,
            "speech",
            attributes=attrs3,
            save_dir=str(tmp_path / "rank_3attrs_5rounds"),
            client=client,
            n_rounds=5,
            matches_per_round=1,
            n_parallels=1,
        )
    )
    seen_round_orders: list[tuple[str, ...]] = []
    for p in client.prompts:
        tup = tuple(_extract_json_block_keys(p, "BEGIN ATTRIBUTES", "END ATTRIBUTES"))
        if not seen_round_orders or seen_round_orders[-1] != tup:
            seen_round_orders.append(tup)
    assert len(seen_round_orders) == 5
    assert len(set(seen_round_orders)) == 5


def test_seed_prefix_first_across_generation_batches(tmp_path):
    class BatchSeedClient(RecordingClient):
        def __init__(self) -> None:
            super().__init__({})
            self._batch = 0

        async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
            self.prompts.append(prompt)
            b = self._batch
            self._batch += 1
            return json.dumps(
                {
                    "entity 1": f"ENTITY_B{b}_ONE",
                    "entity 2": f"ENTITY_B{b}_TWO",
                }
            )

    client = BatchSeedClient()
    res = run(
        seed(
            "Representative US small-business owners",
            save_dir=str(tmp_path / "seed_prefix"),
            client=client,
            num_entities=6,
            entities_per_generation=2,
        )
    )
    assert len(res) == 6
    assert len(client.prompts) == 3

    # Batch 0 (no existing entities) carries static instructions + output format + count rule
    batch0 = client.prompts[0]
    assert "BEGIN ENTITY INSTRUCTIONS" in batch0
    assert "BEGIN OUTPUT FORMAT" in batch0
    assert "Ensure you output exactly 2 unique entities." in batch0
    assert "BEGIN EXISTING ENTITIES" not in batch0

    # Batches 1 and 2 begin with batch 0's exact static prefix, share an identical prefix
    # right up to BEGIN EXISTING ENTITIES, and place dynamic existing entities at the suffix.
    later_prompts = client.prompts[1:]
    for p in later_prompts:
        assert p.startswith(batch0)
        assert p.index("BEGIN OUTPUT FORMAT") < p.index("BEGIN EXISTING ENTITIES")
        assert p.rstrip().endswith("END EXISTING ENTITIES")

    later_prefixes = [
        _prefix_before(p, "BEGIN EXISTING ENTITIES") for p in later_prompts
    ]
    assert len(set(later_prefixes)) == 1
    shared_later = later_prefixes[0]
    assert "BEGIN ENTITY INSTRUCTIONS" in shared_later
    assert "BEGIN OUTPUT FORMAT" in shared_later
    assert "ENTITY_B0_ONE" not in shared_later
    assert "ENTITY_B1_ONE" not in shared_later
