"""
Leximeter 参考示例 —— 对应 GABRIEL tutorial notebook 的文本部分。

运行前配置 provider（三选一）：

  1) 环境变量（推荐）：
       export LEXIMETER_PROTOCOL=chat          # 或 responses
       export LEXIMETER_BASE_URL=https://api.deepseek.com
       export LEXIMETER_API_KEY=sk-...
       export LEXIMETER_MODEL=deepseek-chat

  2) 或显式注入 client（打开下方 CLIENT 注释）。

  3) 不配置则默认 OpenAI Responses API，需 OPENAI_API_KEY。

运行： python examples/reference.py
"""

import asyncio
import os

import pandas as pd

import leximeter

# —— 可选：显式注入 client（不配置 env 时打开这段）——
# from leximeter.client import ChatCompletionsClient
# CLIENT = ChatCompletionsClient(api_key="sk-...", base_url="https://api.deepseek.com")
CLIENT = None  # None = 从环境变量解析

MODEL = os.getenv("LEXIMETER_MODEL", "gpt-5.4-mini")


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


async def main() -> None:
    # ── 1. rate：给菜品按口味打分（0–100）─────────────────────────────
    section("1. rate — 给菜品打口味分")
    toy_data = pd.DataFrame(
        {"entity": ["turkey", "pumpkin pie", "pecan pie", "green bean casserole", "mashed potatoes", "cornbread"]}
    )
    attributes = {
        "savory taste": "How savory the dish is",
        "sweet taste": "Dessert-like sweetness",
        "bitter taste": "Notes of tartness or acidity",
    }
    rate_res = await leximeter.rate(
        toy_data, "entity", attributes=attributes, save_dir="runs/toy_rate",
        model=MODEL, n_runs=1, modality="entity", reset_files=True, client=CLIENT,
    )
    print(rate_res.to_string(index=False))

    # ── 2. classify：给新闻标题打标签 ─────────────────────────────────
    section("2. classify — 新闻标题分类")
    headlines = pd.DataFrame(
        {"text": [
            "Breaking: Supreme Court rules on digital privacy",
            "Scientists discover new frog species in the Amazon",
            "Congress passes amended budget for 2026",
            "Local team wins the championship in overtime",
        ]}
    )
    labels = {"world": "global/politics story", "sports": "sports story",
              "sci/tech": "science or technology story", "business": "business story"}
    cls_res = await leximeter.classify(
        headlines, "text", labels=labels, save_dir="runs/toy_classify",
        model=MODEL, n_runs=1, reset_files=True, client=CLIENT,
    )
    print(cls_res.to_string(index=False))

    # ── 3. extract：抽取产品信息 ──────────────────────────────────────
    section("3. extract — 抽取产品公司/年份/价格")
    products = pd.DataFrame({"product": ["Walkman", "iPad", "Roomba", "Nintendo Switch"]})
    attrs = {
        "company": "The company that released this product.",
        "year of release": "The year the first iteration was released.",
        "unit price": "The unit price at first release.",
    }
    ext_res = await leximeter.extract(
        products, "product", attributes=attrs, save_dir="runs/toy_extract",
        model=MODEL, modality="entity", reset_files=True, client=CLIENT,
    )
    print(ext_res.to_string(index=False))

    # ── 4. rank：两两比较的相对排序 ───────────────────────────────────
    section("4. rank — 演讲的两两排序")
    speeches = pd.DataFrame(
        {"speech": [
            "We must stand together as one nation, indivisible.",
            "The economy is booming and unemployment is at record lows.",
            "Our workers are the backbone of this great country.",
            "We will reduce taxes and cut burdensome regulations.",
        ]}
    )
    rank_res = await leximeter.rank(
        speeches, "speech", attributes={"populism": "people-vs-elite framing"},
        save_dir="runs/toy_rank", model=MODEL, n_rounds=2, matches_per_round=1,
        reset_files=True, client=CLIENT,
    )
    print(rank_res.to_string(index=False))

    # ── 5. merge：跨表模糊匹配 ────────────────────────────────────────
    section("5. merge — 公司名模糊合并")
    df_left = pd.DataFrame({"company_name": ["Google Inc.", "Microsoft Corporation", "Apple Inc", "Amazon.com"]})
    df_right = pd.DataFrame({"name": ["Gooogle Incorporated", "MSFT", "Apple Incorporated", "Amazn.com"]})
    merge_res = await leximeter.merge(
        df_left, df_right, left_on="company_name", right_on="name", how="left",
        save_dir="runs/toy_merge", model=MODEL, reset_files=True, client=CLIENT,
    )
    print(merge_res.to_string(index=False))

    # ── 6. deduplicate：语义去重 ──────────────────────────────────────
    section("6. deduplicate — 技术名词去重")
    tech = pd.DataFrame({"technology": ["iPhone", "Iphone 6", "iPhone 6s", "Apple smartphone", "F-16 Fighting Falcon", "F16"]})
    dedup_res = await leximeter.deduplicate(
        tech, "technology", save_dir="runs/toy_dedup", model=MODEL,
        modality="entity", reset_files=True, client=CLIENT,
    )
    print(dedup_res.to_string(index=False))

    # ── 7. filter：自然语言条件筛选 ───────────────────────────────────
    section("7. filter — 筛出冠军运动员")
    athletes = pd.DataFrame(
        {"athlete": ["Tom Brady", "Serena Williams", "Tom Cruise", "Leonardo DiCaprio", "Usain Bolt"]}
    )
    condition = (
        "The person must be an athlete who has won the top award in their sport "
        "(e.g. Super Bowl, Olympic gold) at least once. Championships only."
    )
    flt_res = await leximeter.filter(
        athletes, "athlete", condition=condition, save_dir="runs/toy_filter",
        model=MODEL, entities_per_call=5, reset_files=True, client=CLIENT,
    )
    print(flt_res[["athlete", "keep", "score"]].to_string(index=False))

    # ── 8. deidentify：PII 脱敏 ───────────────────────────────────────
    section("8. deidentify — 隐私脱敏")
    deid_data = pd.DataFrame(
        {"text": ["Elizabeth Green presented her findings in Manchester on 2 February 2023."]}
    )
    deid_res = await leximeter.deidentify(
        deid_data, "text", save_dir="runs/toy_deidentify", model=MODEL,
        reset_files=True, client=CLIENT,
    )
    print(deid_res["deidentified_text"].to_string(index=False))

    # ── 9. paraphrase：按指令改写 ──────────────────────────────────────
    section("9. paraphrase — 总结演讲")
    paraphrase_data = pd.DataFrame(
        {"text": ["The state of our union is strong. We have created millions of jobs and lifted wages."]}
    )
    para_res = await leximeter.paraphrase(
        paraphrase_data, "text", instructions="Summarize into one short sentence.",
        save_dir="runs/toy_paraphrase", model=MODEL, reset_files=True, client=CLIENT,
    )
    print(para_res["text_paraphrased"].to_string(index=False))

    # ── 10. seed：生成代表性人物 ──────────────────────────────────────
    section("10. seed — 生成代表性人物")
    seed_res = await leximeter.seed(
        "Each entity is a short description of a plausible US adult (age, job, region).",
        save_dir="runs/toy_seed", model=MODEL, num_entities=4,
        entities_per_generation=4, reset_files=True, client=CLIENT,
    )
    print(seed_res.to_string(index=False))

    # ── 11. whatever：任意自定义提示词 ────────────────────────────────
    section("11. whatever — 任意提示词")
    whatever_res = await leximeter.whatever(
        ["Define artificial intelligence.", "Explain the Turing test."],
        save_dir="runs/toy_whatever", model=MODEL, reset_files=True, client=CLIENT,
    )
    print(whatever_res[["Identifier", "Response"]].to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
