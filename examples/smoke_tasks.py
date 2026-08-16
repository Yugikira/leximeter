"""
Leximeter 真实 provider 冒烟脚本（rate 已实测，其余任务按批次覆盖）。

用法：
    uv run python examples/smoke_tasks.py            # 默认跑第一批（classify extract）
    uv run python examples/smoke_tasks.py rank merge # 指定任务（零参数=全跑）

依赖项目根目录 .env（import leximeter 时自动加载）或环境变量。
每个任务独立 try/except：单个失败不阻塞后续任务，末尾打印 PASS/FAIL 汇总。
数据刻意用小样本控制成本；跑完检查列名、类型、非空率等关键契约。
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback

import pandas as pd

import leximeter

MODEL = os.getenv("LEXIMETER_MODEL") or "deepseek-v4-flash"

PASSED: list[str] = []
FAILED: list[str] = []


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def verdict(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"\n>>> [{name}] {'PASS' if ok else 'FAIL'} {detail}")


def check(name: str, cond: bool, detail: str = "") -> None:
    """Accumulate a sub-check; call ``verdict`` once per task at the end."""
    if not cond:
        raise AssertionError(f"{name}: {detail or 'check failed'}")


# ─────────────────────────────────────────────────────────────────────────────
async def smoke_classify() -> None:
    section("classify — 新闻标题多标签分类")
    df = pd.DataFrame({"text": [
        "I heard that the new president instituted a law on gun control.",
        "My dog decided to eat food at 3am.",
        "Congress passes amended budget for 2026",
        "Why are you so late for class today? Did your dog eat your homework?",
        "Local activists rally outside city hall demanding climate reform.",
        "Scientists discover new species of frog in the Amazon rainforest.",
    ]})
    labels = {
        "discusses politics": "The text makes reference to political figures, institutions, or issues.",
        "news headline": "The text is written in the style of a journalistic news headline.",
        "discusses animals": "The text includes mention of animals, whether real or metaphorical.",
        "first person perspective": "The text is written from a personal point of view.",
    }
    res = await leximeter.classify(
        df, "text", labels=labels, save_dir="runs/smoke_classify",
        model=MODEL, modality="text", n_runs=1, reset_files=True,
    )
    print(res.to_string(index=False))
    check("columns", list(res.columns) == ["text", "predicted_classes"] + list(labels),
          f"got {list(res.columns)}")
    label_cols = list(labels)
    check("boolean dtype", all(res[c].dtype == bool for c in label_cols),
          f"dtypes {[str(res[c].dtype) for c in label_cols]}")
    check("some positives", res[label_cols].any().any(), "no label ever True")
    check("row count", len(res) == len(df), f"got {len(res)} rows")
    verdict("classify", True, f"{len(df)}×{len(labels)} labels")


async def smoke_extract() -> None:
    section("extract — 产品信息抽取（多实体 + types 强转）")
    products = pd.DataFrame({"product": [
        "Walkman", "iPad", "Nintendo Switch", "Instant Pot",
        "GoPro", "Rubik's Cube",
    ]})
    attrs = {
        "company": "The name of the company that released this product.",
        "year of release": "The year the first iteration of this product was released.",
        "unit price": "The unit price of the product at the time it was first released.",
    }
    res = await leximeter.extract(
        products, "product", attributes=attrs, save_dir="runs/smoke_extract",
        model=MODEL, modality="entity",
        types={"year of release": "int", "unit price": "float"},
        reset_files=True,
    )
    print(res.to_string(index=False))
    check("columns", {"product", "company", "year of release", "unit price"} <= set(res.columns),
          f"got {list(res.columns)}")
    check("year coerced", res["year of release"].dtype == "Int64",
          f"dtype {res['year of release'].dtype}")
    check("price coerced", res["unit price"].dtype == "float64",
          f"dtype {res['unit price'].dtype}")
    known = res["company"].notna().sum()
    check("company coverage", known >= len(products) // 2,
          f"only {known}/{len(res)} companies extracted")
    verdict("extract", True, f"{len(res)} rows (multi-entity expansion included)")


async def smoke_compare() -> None:
    section("compare — 两两对比差异")
    df = pd.DataFrame({
        "circle": [
            "The new CEO announced a 20% R&D budget cut at the all-hands.",
            "Our team ships weekly; velocity is up 30% since Q1.",
            "The study surveyed 400 patients over two years.",
            "Ticket prices rose 15% after the merger.",
        ],
        "square": [
            "The CFO proposed reallocating funds to marketing instead.",
            "Sprint cycles are now two weeks; scope creep remains an issue.",
            "The follow-up used focus groups with 25 participants.",
            "Ticket prices held steady despite the merger.",
        ],
    })
    res = await leximeter.compare(
        df, "circle", "square", save_dir="runs/smoke_compare",
        model=MODEL, differentiate=True, reset_files=True,
    )
    print(res.to_string(index=False))
    check("columns", {"circle", "square", "attribute", "explanation"} <= set(res.columns),
          f"got {list(res.columns)}")
    check("row count", len(res) >= len(df), f"got {len(res)} rows")
    verdict("compare", True, f"{len(res)} attribute rows")


async def smoke_paraphrase() -> None:
    section("paraphrase — 按指令改写")
    df = pd.DataFrame({"text": [
        "u up? we should totally hit that new spot tmrw",
        "the boss is a total tool, no cap",
        "idk why they kept the old server up tbh",
        "lowkey hyped for the on-device models tho",
    ]})
    res = await leximeter.paraphrase(
        df, "text", instructions="Rewrite in a formal register, no slang.",
        save_dir="runs/smoke_paraphrase", model=MODEL,
        n_rounds=1, n_runs=1, reset_files=True,
    )
    print(res.to_string(index=False))
    check("revised col", "text_revised" in res.columns, f"got {list(res.columns)}")
    check("approved col", "text_revised_approved" in res.columns)
    nonempty = res["text_revised"].fillna("").str.len().gt(0).sum()
    check("revisions produced", nonempty == len(df), f"{nonempty}/{len(df)}")
    verdict("paraphrase", True, f"{nonempty}/{len(df)} revised")


async def smoke_codify() -> None:
    section("codify — 静态类别编码（逐字 snippet 定位）")
    df = pd.DataFrame({"text": [
        "The product team shipped the dashboard late, but the quality held up.",
        "Funding ran dry and the startup froze all hiring.",
        "She praised the review process yet questioned the timeline.",
        "Quarterly targets were met; the board approved the expansion.",
    ]})
    cats = {
        "positive sentiment": "Positive or approving statements about work.",
        "resource constraints": "Mentions of limited money, staff, or time.",
    }
    res = await leximeter.codify(
        df, "text", categories=cats, save_dir="runs/smoke_codify",
        model=MODEL, n_rounds=1, reset_files=True,
    )
    print(res.to_string(index=False))
    check("columns", {"text", "positive sentiment", "resource constraints"} <= set(res.columns),
          f"got {list(res.columns)}")
    total = res["positive sentiment"].map(len).sum() + res["resource constraints"].map(len).sum()
    check("snippets found", total > 0, f"{total} snippets total")
    verdict("codify", True, f"{total} verbatim snippets across {len(df)} rows")


async def smoke_bucket() -> None:
    section("bucket — 术语归桶（多轮投票，压缩轮数控制成本）")
    terms = pd.DataFrame({"term": [
        "smartphone", "laptop", "tablet", "smartwatch", "desktop computer",
        "wireless earbuds", "over-ear headphones", "bluetooth speaker",
        "fitness tracker", "e-reader",
    ]})
    res = await leximeter.bucket(
        terms, "term", save_dir="runs/smoke_bucket", model=MODEL,
        bucket_count=3, repeat_bucketing=2, repeat_voting=3, reset_files=True,
    )
    print(res.to_string(index=False))
    check("columns", {"bucket", "definition"} <= set(res.columns),
          f"got {list(res.columns)}")
    check("bucket count", len(res) == 3, f"got {len(res)} buckets")
    check("nonempty names", res["bucket"].fillna("").str.len().gt(0).all())
    verdict("bucket", True, f"{len(res)} buckets")


async def smoke_rank() -> None:
    section("rank — 两两配对 BT 拟合")
    df = pd.DataFrame({"text": [
        "While your conclusion is careless, the deployment will likely improve consumer surplus.",
        "R&D intensity rose 12% to 18% YoY; margins may compress before gains materialize.",
        "That take is clown shoes—learn the basics before posting.",
        "Low-key hyped for on-device models—privacy win without the lag.",
        "@mina you're right; the vendor ghosted us again—unacceptable.",
        "I concur with @lee: the policy update meaningfully strengthens user control.",
        "Empirical evidence suggests platform concentration exacerbates rent extraction.",
        "New chips slap; battery life finally sane.",
    ]})
    attrs = {
        "optimism": "Positive outlook on technology's future or impact.",
        "formality": "Professional or academic style and structure.",
    }
    res = await leximeter.rank(
        df, "text", attributes=attrs, save_dir="runs/smoke_rank",
        model=MODEL, modality="text", n_rounds=2, matches_per_round=4,
        reset_files=True,
    )
    print(res[["text", "optimism", "optimism_raw", "optimism_se",
               "formality", "formality_raw", "formality_se"]].to_string(index=False))
    for a in attrs:
        for suffix in ("", "_raw", "_se"):
            col = f"{a}{suffix}"
            check(f"column {col}", col in res.columns, f"got {list(res.columns)}")
            check(f"finite {col}", res[col].dropna().apply(
                lambda v: isinstance(v, float) and pd.notna(v)).all(),
                "non-float or NaN")
    check("se nonneg", (res["optimism_se"] >= 0).all() and (res["formality_se"] >= 0).all())
    check("row count", len(res) == len(df), f"got {len(res)} rows")
    verdict("rank", True, f"{len(df)} items × {len(attrs)} attrs, {res['optimism'].std():.3f} z-std")


async def smoke_merge() -> None:
    section("merge — 模糊合并（真实 embedding 预筛，long≥64 门槛）")
    bases = ["Google", "Microsoft", "Apple", "Amazon", "Meta", "Netflix", "Tesla", "Uber"]
    df_left = pd.DataFrame({
        "company_id": range(1, len(bases) + 1),
        "company_name": [f"{b} Inc." for b in bases],
    })
    # long 表：8 基础名 × 8 变体 + 16 个干扰项 ≈ 80 项（≥ long_list_len=64）
    variants = [
        "{b} Incorporated", "{b}, Inc.", "{b} Corp", "{b} Corporation",
        "{b} inc", "{b}.inc", "The {b} Company", "{b} LLC",
    ]
    long_names = [v.format(b=b) for b in bases for v in variants]
    long_names += [f"Distractor Company {i}" for i in range(16)]
    df_right = pd.DataFrame({"record_id": range(1, len(long_names) + 1), "name": long_names})
    res = await leximeter.merge(
        df_left, df_right, left_on="company_name", right_on="name", how="left",
        save_dir="runs/smoke_merge", model=MODEL, use_embeddings=True,
        long_list_len=64, max_attempts=2, reset_files=True,
    )
    print(res.to_string(index=False))
    check("row count", len(res) == len(df_left), f"got {len(res)} rows")
    check("record_id matched", "record_id" in res.columns)
    matched = res["record_id"].notna().sum()
    check("matches found", matched >= len(df_left) // 2,
          f"{matched}/{len(df_left)} left keys matched")
    verdict("merge", True, f"{matched}/{len(df_left)} matched via embedding prefilter")


async def smoke_deduplicate() -> None:
    section("deduplicate — 技术名词去重（真实 embedding + kmeans2 预筛）")
    bases = [
        "iPhone", "Bluetooth", "GPS", "PlayStation", "Fitbit", "AirPods", "Kindle",
        "Roomba", "GoPro", "Android", "Chromebook", "Tesla", "Spotify", "Netflix",
        "Zoom", "Slack", "Docker", "Kubernetes", "Python", "Linux", "Arduino",
        "Raspberry Pi", "Firebase", "Stripe", "Shopify",
    ]
    variants = [
        "{}", "{} ".strip().lower(), "{}".upper(), "{}".replace(" ", ""),
        "{} 5G", "the {}", "new {}", "{} 2.0", "{} pro", "{} max", "{} lite", "{}s",
        "{} 2026", "{} v2", "{} edition", "{} plus", "{} mini", "{} ultra",
        "super {}", "deluxe {}", "{} classic", "{} mkII", "{} (gen 1)",
    ]
    terms: list[str] = []
    for b in bases:
        for tmpl in variants:
            terms.append(tmpl.format(b))
    unique = sorted(set(terms))  # 去重后仍远大于 group_size
    df = pd.DataFrame({"tech": unique})
    res = await leximeter.deduplicate(
        df, "tech", save_dir="runs/smoke_deduplicate", model=MODEL,
        group_size=100, use_embeddings=True, reset_files=True,
    )
    print(res.head(12).to_string(index=False))
    check("mapped col", "mapped_tech" in res.columns, f"got {list(res.columns)}")
    nonnull = res["mapped_tech"].notna().sum()
    check("mapped nonnull", nonnull == len(res), f"{nonnull}/{len(res)}")
    n_groups = res["mapped_tech"].nunique()
    check("merge happened", n_groups < len(unique), f"{n_groups} groups from {len(unique)} terms")
    verdict("deduplicate", True, f"{len(unique)} terms → {n_groups} representatives")


async def smoke_filter() -> None:
    section("filter — 自然语言条件过滤")
    df = pd.DataFrame({"text": [
        "Apple announces iPhone 17 with foldable display.",
        "I made toast this morning and it was fine.",
        "Tesla begins deliveries of the Cybertruck in Europe.",
        "Weather forecast calls for rain on Tuesday.",
        "OpenAI ships GPT-6 API in public beta.",
        "My cat slept through the entire meeting.",
    ]})
    res = await leximeter.filter(
        df, "text", condition="The text announces a product launch or shipment.",
        save_dir="runs/smoke_filter", model=MODEL, threshold=0.5,
        n_runs=1, reset_files=True,
    )
    print(res.to_string(index=False))
    check("keep col", "keep" in res.columns and "score" in res.columns,
          f"got {list(res.columns)}")
    check("keep bool", res["keep"].dtype == bool, f"dtype {res['keep'].dtype}")
    check("score range", res["score"].between(0, 1).all())
    kept = res["keep"].sum()
    check("some kept", 0 < kept < len(res), f"{kept}/{len(res)} kept")
    verdict("filter", True, f"{kept}/{len(res)} kept (score range {res['score'].min():.2f}-{res['score'].max():.2f})")


async def smoke_deidentify() -> None:
    section("deidentify — PII 脱敏")
    df = pd.DataFrame({"text": [
        "John Smith called from 555-0142 to discuss the lease with Mary Jones.",
        "You can reach our team at support@acme.example; ask for Dr. Emily Chen.",
        "Alice Brown lives in Chicago, IL 60601 and her SSN ends in 4412.",
    ]})
    res = await leximeter.deidentify(
        df, "text", save_dir="runs/smoke_deidentify", model=MODEL,
        n_passes=1, reset_files=True,
    )
    print(res.to_string(index=False))
    check("columns", {"deidentified_text", "mapping"} <= set(res.columns),
          f"got {list(res.columns)}")
    nonempty = res["deidentified_text"].fillna("").str.len().gt(0).sum()
    check("outputs produced", nonempty == len(df), f"{nonempty}/{len(df)}")
    verdict("deidentify", True, f"{len(df)} rows deidentified")


async def smoke_seed() -> None:
    section("seed — 生成代表性实体")
    res = await leximeter.seed(
        "Generate fictional startup founders working on AI infrastructure.",
        save_dir="runs/smoke_seed", model=MODEL,
        num_entities=8, entities_per_generation=4, reset_files=True,
    )
    print(res.to_string(index=False))
    check("columns", {"entity", "entity_id"} <= set(res.columns),
          f"got {list(res.columns)}")
    check("row count", len(res) == 8, f"got {len(res)} rows")
    check("entities nonempty", res["entity"].fillna("").str.len().gt(0).all())
    verdict("seed", True, f"{len(res)} entities")


async def smoke_whatever() -> None:
    section("whatever — 任意提示词")
    res = await leximeter.whatever(
        ["Summarize in one sentence: transformers beat RNNs on long sequences.",
         "Name one risk of fine-tuning on small datasets.",
         "Rewrite politely: 'that claim is wrong'."],
        save_dir="runs/smoke_whatever", model=MODEL, reset_files=True,
    )
    print(res.to_string(index=False))
    check("columns", {"Identifier", "Response"} <= set(res.columns),
          f"got {list(res.columns)}")
    nonempty = res["Response"].fillna("").str.len().gt(0).sum()
    check("responses produced", nonempty == 3, f"{nonempty}/3")
    verdict("whatever", True, f"{nonempty}/3 responses")


async def smoke_ideate() -> None:
    section("ideate — 生成想法/理论")
    res = await leximeter.ideate(
        "What explains the persistence of income inequality across generations?",
        save_dir="runs/smoke_ideate", model=MODEL,
        n_ideas=5, reset_files=True,
    )
    print(res[["idea_id", "title", "in_a_nutshell"]].to_string(index=False))
    for col in ["idea_id", "topic", "report_text", "title", "in_a_nutshell"]:
        check(f"column {col}", col in res.columns, f"got {list(res.columns)}")
    nonempty = res["title"].fillna("").str.len().gt(0).sum()
    check("titles produced", nonempty == len(res), f"{nonempty}/{len(res)}")
    verdict("ideate", True, f"{len(res)} ideas × {len(res.columns)} cols")


async def smoke_poll() -> None:
    section("poll — 合成人群调查")
    res = await leximeter.poll(
        population_description="A representative sample of US adults who use social media daily.",
        questions=[
            "How many hours per day do you spend on social media? (1-10)",
            "Do you think social media is mostly good or mostly bad for society? (good/bad)",
        ],
        save_dir="runs/smoke_poll",
        seed_model=MODEL, persona_model=MODEL, poll_model=MODEL,
        num_personas=6, entities_per_generation=6, reset_files=True,
    )
    print(res.to_string(index=False))
    check("columns", {"seed", "persona"} <= set(res.columns),
          f"got {list(res.columns)}")
    for q in res.columns:
        if q not in {"seed", "persona"}:
            nonempty = res[q].fillna("").astype(str).str.len().gt(0).sum()
            check(f"question {q!r} answered", nonempty >= len(res) // 2,
                  f"{nonempty}/{len(res)}")
    verdict("poll", True, f"{len(res)} personas × {len(res.columns)-2} questions")


async def smoke_discover() -> None:
    section("discover — 对立两类文本特征发现（程序化生成）")
    optimistic = [
        "AI assistants handle routine support tickets, freeing staff for creative work.",
        "Renewable energy costs keep falling; solar is now the cheapest power source.",
        "Telemedicine reaches rural patients who previously had no local specialists.",
        "Open-source models let small teams build tools that were once enterprise-only.",
        "Self-driving trucks promise safer highways and cheaper goods for everyone.",
        "Gene therapies are curing diseases that were untreatable a decade ago.",
        "Remote work unlocked opportunities for parents and people with disabilities.",
        "Precision agriculture cuts water use while boosting yields for small farms.",
        "Smart grids smooth demand spikes and cut blackouts during heat waves.",
        "Digital literacy programs are narrowing the gap in underserved schools.",
    ]
    pessimistic = [
        "Algorithmic feeds deepen polarization and erode public discourse daily.",
        "Mass surveillance tools quietly expand with each new security law.",
        "Automation is hollowing out middle-skill jobs faster than retraining helps.",
        "Plastic waste keeps accumulating; recycling rates have barely improved.",
        "Social media feeds amplify misinformation about vaccines and elections.",
        "Data brokers sell sensitive health records with almost no oversight.",
        "Convenience apps exploit gig workers through opaque scheduling algorithms.",
        "Fast fashion's carbon footprint keeps growing despite green marketing.",
        "Autonomous weapons are racing ahead of any meaningful international rule.",
        "Facial recognition errors routinely flag innocent citizens at checkpoints.",
    ]
    df = pd.DataFrame({"circle": optimistic, "square": pessimistic})
    res = await leximeter.discover(
        df, circle_column_name="circle", square_column_name="square",
        save_dir="runs/smoke_discover", model=MODEL,
        bucket_count=3, repeat_bucketing=2, repeat_voting=3, reset_files=True,
    )
    print(type(res).__name__)
    for key in ("candidates", "buckets", "classification", "summary"):
        print(f"  {key}: {type(res.get(key)).__name__}")
        if key == "summary" and res.get(key) is not None:
            print(res["summary"].to_string(index=False))
    check("dict return", isinstance(res, dict), f"got {type(res).__name__}")
    check("keys", {"candidates", "buckets", "classification", "summary"} <= set(res),
          f"got {sorted(res)}")
    summary = res.get("summary")
    check("summary frame", summary is not None and len(summary) > 0)
    net = summary["net_pct"].dropna()
    check("net_pct computed", len(net) > 0, "all net_pct None")
    check("net_pct range", net.abs().le(100).all(), f"out of range: {net.tolist()}")
    verdict("discover", True,
            f"summary {len(summary)} labels; net_pct {net.tolist()}")


async def smoke_debias() -> None:
    section("debias — 两步去偏（小样本：自带 raw 测量列，只测链路连通）")
    names = ["John", "Mary", "Alex", "Sam", "Lee", "Ana", "Ravi", "Mia",
             "Tom", "Eve", "Ben", "Zoe", "Ken", "Ava", "Pia", "Leo",
             "Ida", "Ned", "Uma", "Cal"]
    rows = []
    for i, name in enumerate(names):
        # 100% strip removes the CEO sentence; revenue mentions vary per row
        # so the remaining-signal proxy keeps variation after stripping.
        revenue = " Revenue is up." * (1 + i % 3)
        rows.append({
            "text": f"Brand {chr(65+i)} is great. CEO {name} leads the firm.{revenue}",
            # pre-measured raw attribute (skips the baseline rating stage)
            "brand": 50.0 + i * 2,
        })
    df = pd.DataFrame(rows)
    res = await leximeter.debias(
        df, "text", save_dir="runs/smoke_debias",
        attributes={"brand": "overall brand strength"},
        signal_dictionary={"ceo": "mentions of the chief executive"},
        removal_method="codify",
        remaining_signal=True,
        remaining_signal_attribute="prevalence of revenue discussions",
        remaining_signal_description="mentions of revenue or financial performance",
        model=MODEL, reset_files=True,
    )
    results = res.results
    print(results.columns.tolist())
    print(results[["text", "brand", "brand (stripped)", "brand (raw - stripped)",
                   "brand (debiased)",
                   "prevalence of revenue discussions (stripped)"]].head().to_string(index=False))
    for col in ["brand (stripped)", "brand (raw - stripped)", "brand (debiased)"]:
        check(f"column {col}", col in results.columns, f"got {list(results.columns)}")
    deb = results["brand (debiased)"]
    check("debiased nonnull", deb.notna().sum() >= len(results) // 2,
          f"{deb.notna().sum()}/{len(results)}")
    diff = (results["brand"] - results["brand (debiased)"]).abs()
    check("debiased differs", diff.max() > 0.5, "no adjustment applied")
    reg = res.regression.get("stripped_100pct")
    check("stage1 delta", reg is not None and reg.stage1_delta is not None)
    verdict("debias", True, f"{len(results)} rows; delta={reg.stage1_delta:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
TASKS: dict[str, object] = {
    "classify": smoke_classify,
    "extract": smoke_extract,
    "compare": smoke_compare,
    "paraphrase": smoke_paraphrase,
    "codify": smoke_codify,
    "bucket": smoke_bucket,
    "rank": smoke_rank,
    "merge": smoke_merge,
    "deduplicate": smoke_deduplicate,
    "filter": smoke_filter,
    "deidentify": smoke_deidentify,
    "seed": smoke_seed,
    "whatever": smoke_whatever,
    "ideate": smoke_ideate,
    "poll": smoke_poll,
    "discover": smoke_discover,
    "debias": smoke_debias,
}


async def main() -> None:
    selected = sys.argv[1:] or list(TASKS)
    unknown = [t for t in selected if t not in TASKS]
    if unknown:
        print(f"Unknown task(s): {unknown}; available: {list(TASKS)}")
        sys.exit(2)
    for name in selected:
        try:
            await TASKS[name]()
        except Exception:  # noqa: BLE001 - smoke runner reports and continues
            FAILED.append(name)
            print(f"\n>>> [{name}] FAIL (exception)")
            traceback.print_exc()
    print("\n" + "=" * 64)
    print(f"SUMMARY: {len(PASSED)} passed, {len(FAILED)} failed — "
          f"passed={PASSED} failed={FAILED}")
    print("=" * 64)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    asyncio.run(main())
