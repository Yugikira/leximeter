"""End-to-end tests for the ported tasks, using fake (no-network) providers."""

from __future__ import annotations

import asyncio
import json
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from conftest import CountingClient, DictClient, TextClient

from leximeter import (
    bucket,
    classify,
    codify,
    compare,
    debias,
    deduplicate,
    deidentify,
    discover,
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
from leximeter.client import DummyEmbeddingClient


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
    assert (res["world"] == True).all()
    assert (res["sports"] == False).all()


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


def test_extract_expands_multiple_entities(tmp_path):
    df = pd.DataFrame({"product": ["Walkman"]})
    client = DictClient(
        {
            "Walkman": {"company": "Sony", "year of release": "1979"},
            "Walkman II": {"company": "Sony", "year of release": "1981"},
        }
    )
    res = run(
        extract(
            df,
            "product",
            attributes={"company": "", "year of release": ""},
            save_dir=str(tmp_path / "ext"),
            client=client,
        )
    )
    assert len(res) == 2  # one extra row per additional entity
    assert list(res["entity_name"]) == ["Walkman", "Walkman II"]
    assert list(res["company"]) == ["Sony", "Sony"]
    assert list(res["year of release"]) == ["1979", "1981"]


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


class CapturingSearchTool:
    """Fake search tool that returns canned context and records queries."""

    def __init__(self, context: str):
        self._context = context
        self.queries: list[str] = []

    async def search(self, query: str) -> str:
        self.queries.append(query)
        return self._context


def test_whatever_web_search_injects_context(tmp_path):
    class RecordingClient(DictClient):
        def __init__(self, payload):
            super().__init__(payload)
            self.prompts: list[str] = []

        async def complete(self, prompt, *, model, json_mode=False, **kwargs) -> str:
            self.prompts.append(prompt)
            return await super().complete(prompt, model=model, json_mode=json_mode, **kwargs)

    client = RecordingClient({"ok": True})
    search = CapturingSearchTool("RETRIEVED CONTEXT")
    res = run(
        whatever(
            ["define AI", "what is a NN"],
            save_dir=str(tmp_path / "w2"),
            client=client,
            search_tool=search,
            web_search=True,
        )
    )
    # each prompt was expanded with retrieved context, then sent to the model
    assert search.queries == ["define AI", "what is a NN"]
    assert client.prompts == [
        "RETRIEVED CONTEXT\n\ndefine AI",
        "RETRIEVED CONTEXT\n\nwhat is a NN",
    ]
    assert list(res.columns) == ["Identifier", "Response"]
    assert list(res["Response"]) == ['{"ok": true}', '{"ok": true}']


def test_whatever_web_search_missing_tool_falls_back(tmp_path):
    # web_search requested but no tool configured -> prompts sent verbatim.
    client = DictClient({"ok": True})
    res = run(
        whatever(
            ["define AI"],
            save_dir=str(tmp_path / "w3"),
            client=client,
            search_tool=None,  # no tool injected
            web_search=True,
        )
    )
    assert list(res["Response"]) == ['{"ok": true}']


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
    assert "text_revised" in res.columns
    assert "text_revised_approved" in res.columns
    assert list(res["text_revised"]) == ["REWRITTEN", "REWRITTEN"]
    assert (res["text_revised_approved"] == True).all()


def test_paraphrase_recursive_validation(tmp_path):
    # Classifier sees instructions_followed=True -> the first round's
    # candidates pass and no further rounds are scheduled.
    df = pd.DataFrame({"text": ["hello"]})
    client = DictClient({"instructions_followed": True})
    res = run(
        paraphrase(
            df,
            "text",
            instructions="summarize",
            save_dir=str(tmp_path / "p2"),
            client=client,
            n_rounds=3,
            n_runs=1,
        )
    )
    assert res["text_revised"].iloc[0] == '{"instructions_followed": true}'
    assert res["text_revised_approved"].iloc[0] == True
    # round-0 generation checkpoint landed on disk
    assert (tmp_path / "p2" / "paraphrase_responses_round0.csv").exists()


def test_paraphrase_validation_exhausts_rounds(tmp_path):
    # Classifier always answers False -> every round fails, the round
    # budget is exhausted, and a last-round candidate is kept with
    # approval=False.  (Unparseable responses count as "uncertain" and
    # pass under GABRIEL semantics, so the failing payload must be an
    # explicit False.)
    df = pd.DataFrame({"text": ["hello"]})
    client = DictClient({"instructions_followed": False})
    res = run(
        paraphrase(
            df,
            "text",
            instructions="summarize",
            save_dir=str(tmp_path / "p3"),
            client=client,
            n_rounds=2,
            n_runs=1,
            later_round_run_multiplier=2,
        )
    )
    assert res["text_revised"].iloc[0] == '{"instructions_followed": false}'
    assert res["text_revised_approved"].iloc[0] == False
    # both rounds were generated and checkpointed
    assert (tmp_path / "p3" / "paraphrase_responses_round0.csv").exists()
    assert (tmp_path / "p3" / "paraphrase_responses_round1.csv").exists()


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
            use_embeddings=False,
        )
    )
    assert res["id"].iloc[0] == 1


def test_merge_embeddings(tmp_path):
    df_left = pd.DataFrame({"company": ["Google Inc."]})
    df_right = pd.DataFrame({"name": ["Gooogle Incorporated"], "id": [1]})
    client = DictClient({"Google Inc.": "Gooogle Incorporated"})
    res = run(
        merge(
            df_left,
            df_right,
            left_on="company",
            right_on="name",
            save_dir=str(tmp_path / "m2"),
            client=client,
            embedding_client=DummyEmbeddingClient(),
        )
    )
    assert res["id"].iloc[0] == 1


def test_deduplicate(tmp_path):
    df = pd.DataFrame({"tech": ["iPhone", "Iphone 6", "Android"]})
    client = DictClient({"iPhone": ["iPhone", "Iphone 6"]})
    res = run(
        deduplicate(
            df, "tech", save_dir=str(tmp_path / "d"), client=client, use_embeddings=False
        )
    )
    assert list(res["mapped_tech"]) == ["iPhone", "iPhone", "Android"]


def test_deduplicate_embeddings(tmp_path):
    # 3 unique < group_size: embedding path not gated in, exercises fallback
    df = pd.DataFrame({"tech": ["iPhone", "Iphone 6", "Android"]})
    client = DictClient({"iPhone": ["iPhone", "Iphone 6"]})
    res = run(
        deduplicate(
            df,
            "tech",
            save_dir=str(tmp_path / "d2"),
            client=client,
            embedding_client=DummyEmbeddingClient(),
        )
    )
    assert list(res["mapped_tech"]) == ["iPhone", "iPhone", "Android"]


class CountingEmbeddingClient(DummyEmbeddingClient):
    """DummyEmbeddingClient that counts embed() calls."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return await super().embed(texts)


@pytest.mark.filterwarnings("ignore:One of the clusters is empty:UserWarning")
def test_deduplicate_embeddings_kmeans(tmp_path):
    """600 unique terms hit the kmeans2 prefilter: groups capped at group_size."""
    terms = [f"term_{i:03d}" for i in range(600)]
    df = pd.DataFrame({"tech": terms})
    client = CountingClient({})
    emb_client = CountingEmbeddingClient()
    res = run(
        deduplicate(
            df,
            "tech",
            save_dir=str(tmp_path / "d3"),
            client=client,
            embedding_client=emb_client,
        )
    )
    assert emb_client.calls == 1  # embedding path gated in (600 >= group_size)
    assert client.calls == 2  # k = ceil(600/500) = 2 -> two <=500 groups
    assert list(res["mapped_tech"]) == terms  # empty payload: identity mapping


@pytest.mark.filterwarnings("ignore:One of the clusters is empty:UserWarning")
def test_deduplicate_embeddings_checkpoint(tmp_path):
    """Embedding vectors are checkpointed: rerun hits the cache, reset recomputes."""
    terms = [f"term_{i:03d}" for i in range(600)]
    df = pd.DataFrame({"tech": terms})
    save = str(tmp_path / "d4")
    client = CountingClient({})
    emb_client = CountingEmbeddingClient()
    run(deduplicate(df, "tech", save_dir=save, client=client, embedding_client=emb_client))
    assert emb_client.calls == 1
    assert (tmp_path / "d4" / "deduplicate_embeddings.json").exists()
    run(deduplicate(df, "tech", save_dir=save, client=client, embedding_client=emb_client))
    assert emb_client.calls == 1  # cache hit: no re-embedding
    run(
        deduplicate(
            df,
            "tech",
            save_dir=save,
            client=client,
            embedding_client=emb_client,
            reset_files=True,
        )
    )
    assert emb_client.calls == 2  # reset_files: recompute


def test_bucket(tmp_path):
    df = pd.DataFrame({"tech": ["iPhone", "iPad"]})
    client = DictClient({"mobile device": "a definition"})
    res = run(
        bucket(df, "tech", save_dir=str(tmp_path / "b"), client=client, bucket_count=1)
    )
    assert set(res.columns) == {"bucket", "definition"}
    assert res["bucket"].iloc[0] == "mobile device"
    # three-stage checkpoints land on disk
    assert (tmp_path / "b" / "bucket_generation.csv").exists()
    assert (tmp_path / "b" / "bucket_state.json").exists()
    assert (tmp_path / "b" / "bucket_definitions.csv").exists()


def test_bucket_caches_completed_run(tmp_path):
    df = pd.DataFrame({"tech": ["iPhone", "iPad"]})
    save = str(tmp_path / "b2")
    client = CountingClient({"mobile device": "a definition"})
    run(bucket(df, "tech", save_dir=save, client=client, bucket_count=1))
    calls_after_first = client.calls
    assert calls_after_first > 0
    run(bucket(df, "tech", save_dir=save, client=client, bucket_count=1))
    assert client.calls == calls_after_first  # cached, no new calls


def test_bucket_reduces_many_candidates(tmp_path):
    # 3 candidates >= 3 * bucket_count triggers the iterative reduction
    # stage before final selection narrows to a single bucket.
    df = pd.DataFrame({"tech": ["iPhone", "iPad", "Mac", "Apple TV"]})
    client = DictClient({"phone": "def", "computer": "def", "tv": "def"})
    res = run(
        bucket(
            df,
            "tech",
            save_dir=str(tmp_path / "b3"),
            client=client,
            bucket_count=1,
            repeat_voting=3,
        )
    )
    assert len(res) == 1
    assert res["bucket"].iloc[0] in {"phone", "computer", "tv"}


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
    # static mode: one column per category holding the verbatim snippets
    assert "code1" in res.columns
    assert res["code1"].iloc[0] == ["hello world"]
    # final output lands on the hardcoded coded_passages.csv
    assert (tmp_path / "c" / "coded_passages.csv").exists()


def test_codify_dynamic_mode(tmp_path):
    df = pd.DataFrame({"text": ["hello world"]})
    client = DictClient(
        {"code1": [{"beginning excerpt": "hello world", "ending excerpt": ""}]}
    )
    res = run(
        codify(
            df,
            "text",
            categories=None,
            additional_instructions="code passages",
            save_dir=str(tmp_path / "c2"),
            client=client,
        )
    )
    # dynamic mode: discovered categories collapse into coded_passages
    assert "coded_passages" in res.columns
    assert res["coded_passages"].iloc[0]["code1"] == ["hello world"]


def test_codify_completion_loop(tmp_path):
    # Long text: the first pass leaves remaining uncoded content, so the
    # completion sweep (_classify_remaining) flags code1 again, re-codes the
    # stripped text, and stops once nothing new is found.
    df = pd.DataFrame(
        {
            "text": [
                (
                    "This is a long passage. Hello world. "
                    "There is much more content here that remains uncoded."
                )
            ]
        }
    )
    client = DictClient(
        {"code1": [{"beginning excerpt": "Hello world", "ending excerpt": "much more content"}]}
    )
    res = run(
        codify(
            df,
            "text",
            categories={"code1": "def"},
            save_dir=str(tmp_path / "c3"),
            client=client,
        )
    )
    assert "code1" in res.columns
    assert any("Hello world" in s for s in res["code1"].iloc[0])
    # the completion sweep checkpoint was written
    check_dir = tmp_path / "c3" / "completion_checks"
    assert list(check_dir.glob("*completion_iter0*"))


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
    # z-score, centred BT estimate and standard error per attribute
    assert "populism" in res.columns
    assert "populism_raw" in res.columns
    assert "populism_se" in res.columns
    assert len(res) == 3
    # round checkpoint exists with Batch/Pair/IdA/IdB columns
    round_csv = tmp_path / "r" / "rankings_round0.csv"
    assert round_csv.exists()
    round_df = pd.read_csv(round_csv)
    assert {"Batch", "Pair", "IdA", "IdB"}.issubset(round_df.columns)


def test_rank_se_finite_and_positive(tmp_path):
    # A tournament with recorded outcomes must yield finite, positive
    # standard errors (the BT Fisher-information projection).
    df = pd.DataFrame({"speech": ["a", "b", "c", "d"]})
    client = DictClient({"populism": "circle"})
    res = run(
        rank(
            df,
            "speech",
            attributes={"populism": ""},
            save_dir=str(tmp_path / "r2"),
            client=client,
            n_rounds=2,
            matches_per_round=2,
        )
    )
    assert (res["populism_se"] > 0).all()
    assert res["populism_se"].notna().all()
    assert res["populism"].notna().all()


def test_rank_resumes_from_rounds(tmp_path):
    df = pd.DataFrame({"speech": ["a", "b", "c"]})
    save = str(tmp_path / "r3")
    client = CountingClient({"populism": "circle"})
    run(
        rank(
            df,
            "speech",
            attributes={"populism": ""},
            save_dir=save,
            client=client,
            n_rounds=1,
            matches_per_round=1,
        )
    )
    calls_after_first = client.calls
    assert calls_after_first > 0
    # resume: final output exists and covers all items -> no new calls
    run(
        rank(
            df,
            "speech",
            attributes={"populism": ""},
            save_dir=save,
            client=client,
            n_rounds=1,
            matches_per_round=1,
        )
    )
    assert client.calls == calls_after_first


class StageClient(DictClient):
    """Return a different payload per discovery stage, keyed by prompt markers."""

    def __init__(self, payloads):
        self._payloads = payloads
        super().__init__({})

    async def complete(self, prompt, *, model, json_mode=False, **kwargs):
        if "BEGIN LABELS" in prompt:
            payload = self._payloads["classify"]
        elif "BEGIN RAW TERMS" in prompt:
            payload = self._payloads["bucket"]
        elif "BEGIN USER INSTRUCTIONS" in prompt:  # dynamic codify branch
            payload = self._payloads["codify"]
        else:
            payload = self._payloads["compare"]
        return json.dumps(payload)


def test_discover_pair_summary_net_pct(tmp_path):
    # Pair pipeline: compare -> bucket -> classify (differentiate).  The
    # classify stage sees quality_actual=True / quality (inverted)=False,
    # so the summary reports actual 100% / inverted 0% / net_pct 100.
    df = pd.DataFrame(
        {"circle": ["sunny", "rainy"], "square": ["cloudy", "clear"]}
    )
    client = StageClient(
        {
            "compare": {"quality": "circle has more quality than square"},
            "bucket": {"quality": "a distinguishing feature"},
            "classify": {"quality": True, "quality (inverted)": False},
            "codify": {},
        }
    )
    res = run(
        discover(
            df,
            circle_column_name="circle",
            square_column_name="square",
            save_dir=str(tmp_path / "d"),
            client=client,
            bucket_count=1,
        )
    )
    assert "summary" in res
    summary = res["summary"]
    assert list(summary.columns) == [
        "label",
        "actual_true",
        "inverted_true",
        "total",
        "actual_pct",
        "inverted_pct",
        "net_pct",
    ]
    assert summary["label"].iloc[0] == "quality"
    assert summary["actual_true"].iloc[0] == 2
    assert summary["inverted_true"].iloc[0] == 0
    assert summary["net_pct"].iloc[0] == 100.0
    assert res["buckets"] == {"quality": "a distinguishing feature"}
    assert "compare" in res
    assert "classification" in res
    # snapshot + zip exports land on disk
    assert (tmp_path / "d" / "discover_results_snapshot.json").exists()
    assert (tmp_path / "d" / "discover_results_export.zip").exists()


def test_discover_single_codify_pipeline(tmp_path):
    # Single-column pipeline: dynamic codify -> bucket -> classify.
    df = pd.DataFrame(
        {
            "text": [
                (
                    "This is a long passage. Hello world. "
                    "There is much more content here that remains uncoded."
                )
            ]
        }
    )
    client = StageClient(
        {
            "codify": {
                "feature code": [
                    {"beginning excerpt": "Hello world", "ending excerpt": "much more content"}
                ]
            },
            "bucket": {"feature code": "a distinguishing feature"},
            "classify": {"feature code": True},
            "compare": {},
        }
    )
    res = run(
        discover(
            df,
            column_name="text",
            save_dir=str(tmp_path / "d2"),
            client=client,
            bucket_count=1,
            additional_instructions="identify features",
        )
    )
    assert res["buckets"] == {"feature code": "a distinguishing feature"}
    assert "codify" in res
    assert "summary" not in res
    assert (res["classification"]["feature code"] == True).all()


class DebiasClient(DictClient):
    """Dispatch debias sub-task payloads by prompt content markers.

    The pipeline calls, in order: Rate (baseline) -> Codify (snippet
    extraction) -> Rate (stripped) -> Rate (remaining signal).  Payloads are
    chosen from the passage content so that original / stripped / remaining
    measurements return different numeric values.
    """

    _SNIPPETS: ClassVar[dict[str, str]] = {
        "John": "CEO John leads the firm.",
        "Mary": "CEO Mary runs the show.",
        "Alex": "CEO Alex drives the plan.",
        "Sam": "CEO Sam owns the market.",
    }
    _ORIG: ClassVar[dict[str, int]] = {"John": 70, "Mary": 75, "Alex": 80, "Sam": 85}
    _STRIPPED: ClassVar[dict[str, int]] = {"Alpha": 40, "Beta": 45, "Gamma": 50, "Delta": 55}
    _REMAINING: ClassVar[dict[str, int]] = {"Alpha": 50, "Beta": 45, "Gamma": 40, "Delta": 35}

    def __init__(self):
        super().__init__({})

    def _key(self, prompt, table):
        return next((k for k in table if k in prompt), next(iter(table)))

    async def complete(self, prompt, *, model, json_mode=False, **kwargs):
        if "BEGIN LABELS" in prompt:  # codify completion check (classify)
            return json.dumps({"ceo": False})
        if "BEGIN CATEGORIES/ATTRIBUTES" in prompt:  # codify static branch
            name = self._key(prompt, self._SNIPPETS)
            return json.dumps(
                {
                    "ceo": [
                        {
                            "beginning excerpt": self._SNIPPETS[name],
                            "ending excerpt": "",
                        }
                    ]
                }
            )
        if "prevalence of ceo" in prompt:  # remaining-signal measurement
            key = self._key(prompt, self._REMAINING)
            return json.dumps({"prevalence of ceo": self._REMAINING[key]})
        if "CEO" in prompt:  # baseline measurement on the original text
            name = self._key(prompt, self._ORIG)
            return json.dumps({"brand": self._ORIG[name]})
        # measurement on the stripped text (snippets already removed)
        key = self._key(prompt, self._STRIPPED)
        return json.dumps({"brand": self._STRIPPED[key]})


def test_debias_twostep_columns(tmp_path):
    # codify removal at 100%: snippets (CEO sentences) are stripped, both
    # simple-difference and two-step debiased columns are produced, and the
    # stage-1 remaining-signal regression drives ``{attr} (debiased)``.
    df = pd.DataFrame(
        {
            "text": [
                "Brand Alpha is great. CEO John leads the firm. Revenue is up.",
                "Brand Beta shines. CEO Mary runs the show. Costs are low.",
                "Brand Gamma excels. CEO Alex drives the plan. Growth is fast.",
                "Brand Delta grows. CEO Sam owns the market. Margins are high.",
            ]
        }
    )
    client = DebiasClient()
    res = run(
        debias(
            df,
            "text",
            save_dir=str(tmp_path / "db"),
            attributes={"brand": "overall brand strength"},
            signal_dictionary={"ceo": "mentions of the chief executive"},
            client=client,
            verbose=False,
        )
    )
    results = res.results
    assert "brand" in results.columns
    assert "brand (stripped)" in results.columns
    assert "brand (raw - stripped)" in results.columns
    assert "brand (debiased)" in results.columns
    assert "prevalence of ceo (stripped)" in results.columns
    assert list(results["brand"]) == [70.0, 75.0, 80.0, 85.0]
    assert list(results["brand (stripped)"]) == [40.0, 45.0, 50.0, 55.0]
    assert results["brand (debiased)"].notna().sum() == 4
    reg = res.regression["stripped_100pct"]
    assert reg.stage1_delta is not None
    assert reg.diff_regression["n"] == 4
    assert reg.twostep_regression["n"] == 4
    run_dir = tmp_path / "db" / "debias_brand"
    assert (run_dir / "debias_results.csv").exists()
    assert (run_dir / "debias_metadata.json").exists()


def test_fit_ols_robust_hc3():
    from leximeter.utils.regression import fit_ols

    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, 60)
    y = 2.0 + 3.0 * x + rng.normal(0, 1, 60)
    X = np.column_stack([np.ones(60), x])
    res = fit_ols(y, X, robust=True, varnames=["Intercept", "x"])
    assert np.allclose(res["coef"], [2.0, 3.0], atol=0.5)
    assert (res["se"] > 0).all()
    assert res["r2"] > 0.9
    assert res["n"] == 60
    assert res["k"] == 1


def test_ideate(tmp_path):
    client = TextClient("Title: Test\nIn a nutshell: A nutshell.")
    res = run(
        ideate("some topic", save_dir=str(tmp_path / "i"), client=client, n_ideas=2)
    )
    assert "title" in res.columns
    assert res["title"].iloc[0] == "Test"
    assert len(res) == 2
