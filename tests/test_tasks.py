"""End-to-end tests for the ported tasks, using fake (no-network) providers."""

from __future__ import annotations

import asyncio

import pandas as pd

from leximeter import (
    bucket,
    classify,
    codify,
    compare,
    deduplicate,
    deidentify,
    extract,
    filter,
    ideate,
    load,
    merge,
    paraphrase,
    rank,
    rate,
    seed,
    whatever,
)

from conftest import CountingClient, DictClient, TextClient


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


def test_compare(tmp_path):
    df = pd.DataFrame({"a": ["x", "y"], "b": ["x2", "y2"]})
    client = DictClient({"circle has more": "justification"})
    res = run(
        compare(df, "a", "b", save_dir=str(tmp_path / "cmp"), client=client)
    )
    assert set(res.columns) == {"circle", "square", "attribute", "explanation"}
    assert list(res["attribute"]) == ["circle has more", "circle has more"]


def test_paraphrase(tmp_path):
    df = pd.DataFrame({"text": ["hello", "world"]})
    client = TextClient("REWRITTEN")
    res = run(
        paraphrase(
            df,
            "text",
            instructions="summarize",
            save_dir=str(tmp_path / "para"),
            client=client,
        )
    )
    assert "text_paraphrased" in res.columns
    assert list(res["text_paraphrased"]) == ["REWRITTEN", "REWRITTEN"]


def test_seed(tmp_path):
    client = DictClient({"entity 1": "person A", "entity 2": "person B"})
    res = run(
        seed(
            "generate people",
            save_dir=str(tmp_path / "seed"),
            client=client,
            num_entities=2,
            entities_per_generation=2,
        )
    )
    assert list(res.columns) == ["entity", "entity_id"]
    assert len(res) == 2


def test_deidentify(tmp_path):
    df = pd.DataFrame({"text": ["Alice went to school"]})
    client = DictClient(
        {"person's name": {"real forms": ["Alice"], "casted form": "Zoe"}}
    )
    res = run(deidentify(df, "text", save_dir=str(tmp_path / "deid"), client=client))
    assert res["deidentified_text"].iloc[0] == "Zoe went to school"


def test_merge(tmp_path):
    df_left = pd.DataFrame({"company": ["Google Inc."]})
    df_right = pd.DataFrame({"name": ["Gooogle Incorporated"], "id": [1]})
    client = DictClient({"Google Inc.": "Gooogle Incorporated"})
    res = run(
        merge(
            df_left,
            df_right,
            left_on="company",
            right_on="name",
            save_dir=str(tmp_path / "m"),
            client=client,
        )
    )
    assert res["id"].iloc[0] == 1


def test_deduplicate(tmp_path):
    df = pd.DataFrame({"tech": ["iPhone", "Iphone 6", "Android"]})
    client = DictClient({"iPhone": ["iPhone", "Iphone 6"]})
    res = run(
        deduplicate(df, "tech", save_dir=str(tmp_path / "d"), client=client)
    )
    assert list(res["mapped_tech"]) == ["iPhone", "iPhone", "Android"]


def test_bucket(tmp_path):
    df = pd.DataFrame({"tech": ["iPhone", "iPad"]})
    client = DictClient({"mobile device": "a definition"})
    res = run(
        bucket(df, "tech", save_dir=str(tmp_path / "b"), client=client, bucket_count=1)
    )
    assert set(res.columns) == {"bucket", "definition"}
    assert res["bucket"].iloc[0] == "mobile device"


def test_codify(tmp_path):
    df = pd.DataFrame({"text": ["hello world"]})
    client = DictClient(
        {"code1": [{"beginning excerpt": "hello world", "ending excerpt": ""}]}
    )
    res = run(
        codify(
            df,
            "text",
            categories={"code1": "def"},
            save_dir=str(tmp_path / "c"),
            client=client,
        )
    )
    assert "coded_passages" in res.columns
    assert res["coded_passages"].iloc[0]["code1"][0]["beginning excerpt"] == "hello world"


def test_load(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world", encoding="utf-8")
    res = load(str(tmp_path))
    assert list(res["name"]) == ["a.txt"]
    assert res["text"].iloc[0] == "hello world"


def test_rank(tmp_path):
    df = pd.DataFrame({"speech": ["a", "b", "c"]})
    client = DictClient({"populism": "circle"})
    res = run(
        rank(
            df,
            "speech",
            attributes={"populism": ""},
            save_dir=str(tmp_path / "r"),
            client=client,
            n_rounds=1,
            matches_per_round=1,
        )
    )
    assert "populism" in res.columns
    assert len(res) == 3


def test_ideate(tmp_path):
    client = TextClient("Title: Test\nIn a nutshell: A nutshell.")
    res = run(
        ideate("some topic", save_dir=str(tmp_path / "i"), client=client, n_ideas=2)
    )
    assert "title" in res.columns
    assert res["title"].iloc[0] == "Test"
    assert len(res) == 2
