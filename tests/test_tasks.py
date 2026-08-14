"""End-to-end tests for the ported tasks, using fake (no-network) providers."""

from __future__ import annotations

import asyncio

import pandas as pd

from leximeter import classify, extract, filter, rate, whatever

from conftest import CountingClient, DictClient


def run(coro):
    return asyncio.run(coro)


def test_rate_scores_and_averages(tmp_path):
    df = pd.DataFrame({"entity": ["turkey", "pumpkin pie", "cornbread"]})
    client = DictClient({"savory taste": 50, "sweet taste": 30})
    res = run(
        rate(
            df,
            "entity",
            attributes={"savory taste": "", "sweet taste": ""},
            save_dir=str(tmp_path / "rate"),
            client=client,
            modality="entity",
            n_runs=2,
        )
    )
    assert list(res.columns[-2:]) == ["savory taste", "sweet taste"]
    assert (res["savory taste"] == 50.0).all()
    assert (res["sweet taste"] == 30.0).all()


def test_rate_resumes_from_checkpoint(tmp_path):
    df = pd.DataFrame({"entity": ["a", "b", "c"]})
    save = str(tmp_path / "rate")
    client = CountingClient({"savory": 50})
    run(rate(df, "entity", attributes={"savory": ""}, save_dir=save, client=client))
    calls_after_first = client.calls
    assert calls_after_first == 3  # 3 unique items, n_runs=1
    run(rate(df, "entity", attributes={"savory": ""}, save_dir=save, client=client))
    assert client.calls == calls_after_first  # resumed, no new calls


def test_classify_min_frequency(tmp_path):
    df = pd.DataFrame({"text": ["h1", "h2"]})
    client = DictClient({"world": True, "sports": False})
    res = run(
        classify(
            df,
            "text",
            labels={"world": "", "sports": ""},
            save_dir=str(tmp_path / "cls"),
            client=client,
            n_runs=3,
            min_frequency=0.6,
        )
    )
    assert res["world"].dtype == bool
    assert (res["world"] == True).all()  # noqa: E712
    assert (res["sports"] == False).all()  # noqa: E712


def test_extract_flattens_entity_and_coerces(tmp_path):
    df = pd.DataFrame({"product": ["Walkman"]})
    client = DictClient(
        {"Walkman": {"company": "Sony", "year of release": "1979", "unit price": "unknown"}}
    )
    res = run(
        extract(
            df,
            "product",
            attributes={"company": "", "year of release": "", "unit price": ""},
            save_dir=str(tmp_path / "ext"),
            client=client,
            types={"year of release": int},
        )
    )
    assert res["company"].iloc[0] == "Sony"
    assert res["year of release"].iloc[0] == 1979
    assert pd.isna(res["unit price"].iloc[0])


def test_filter_keep_score(tmp_path):
    df = pd.DataFrame({"tech": ["iPhone", "iPad", "banana", "F-16"]})
    client = DictClient({"entities meeting condition": ["iPhone", "iPad"]})
    res = run(
        filter(
            df,
            "tech",
            condition="is a technology",
            save_dir=str(tmp_path / "flt"),
            client=client,
            entities_per_call=2,
        )
    )
    assert list(res["keep"]) == [True, True, False, False]
    assert list(res["score"]) == [1.0, 1.0, 0.0, 0.0]


def test_whatever_json_mode(tmp_path):
    client = DictClient({"ok": True})
    res = run(
        whatever(
            ["define AI", "what is a NN"],
            save_dir=str(tmp_path / "w"),
            client=client,
            json_mode=True,
        )
    )
    assert list(res.columns) == ["Identifier", "Response", "Response JSON"]
    assert list(res["Response JSON"]) == [{"ok": True}, {"ok": True}]
