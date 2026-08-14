"""
Leximeter 参考示例 —— 与 GABRIEL tutorial notebook 文本任务一一对应。

每个 section 标注了 notebook 的 cell 编号，玩具数据与参数和原项目完全一致，
可直接运行后与原项目的结果对比（列名、数值语义均对齐 GABRIEL）。

运行前配置 provider（三选一）：

  1) 项目根目录放 .env（推荐，参考 .env.example）：
       LEXIMETER_PROTOCOL=chat
       LEXIMETER_BASE_URL=https://api.deepseek.com
       LEXIMETER_API_KEY=sk-...
       LEXIMETER_MODEL=deepseek-chat
     （或 LEXIMETER_PROTOCOL=responses + OPENAI_API_KEY 走 OpenAI）

  2) 或直接用环境变量 export 同名键（优先级高于 .env）。

  3) 或显式注入 client / search_tool（打开下方 CLIENT / SEARCH_TOOL 注释）。

  4) 不配置则默认 OpenAI Responses API，需 OPENAI_API_KEY。

联网搜索（最后两节）：需要 TAVILY_API_KEY + 安装可选依赖
    pip install -e '.[search]'
    LEXIMETER_SEARCH_PROVIDER=tavily

运行： python examples/reference.py
"""

import asyncio
import os

import pandas as pd

import leximeter

# —— 可选：显式注入 client（不配置 env 时打开这段）——
# from leximeter.client import ChatCompletionsClient
# CLIENT = ChatCompletionsClient(api_key="sk-...", base_url="https://api.deepseek.com")
CLIENT = None  # None = 从环境变量 / .env 解析（import leximeter 时已加载 .env）

# —— 可选：显式注入搜索工具（不配置 LEXIMETER_SEARCH_PROVIDER 时打开这段）——
# from leximeter.client.search_tavily import TavilySearchTool
# SEARCH_TOOL = TavilySearchTool()
SEARCH_TOOL = None  # None = 从 LEXIMETER_SEARCH_PROVIDER 解析；未配置则跳过检索

MODEL = os.getenv("LEXIMETER_MODEL") or "gpt-5.4-mini"


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


async def main() -> None:
    # ── 1. load：加载文件夹文本（notebook cell 59）────────────────────
    # 需要在本地有一个文本文件夹；这里给示例路径，可改成自己的目录。
    section("1. load — 加载文本文件夹")
    try:
        df_loaded = leximeter.load(
            "examples/data",
            modality="text",
            reset_files=True,
        )
        print(df_loaded[["name", "text"]].head().to_string(index=False))
    except Exception as exc:
        print(f"  (跳过：没有 examples/data 文件夹 —— {exc})")

    # ── 2. classify：新闻标题分类（notebook cell 65）─────────────────
    section("2. classify — 新闻标题分类")
    toy_data = pd.DataFrame({"text": [
        "I heard that the new president instituted a law on gun control.",
        "My dog decided to eat food at 3am.",
        "Congress passes amended budget for 2026",
        "Why are you so late for class today? Did your dog eat your homework?",
        "Local activists rally outside city hall demanding climate reform.",
        "Scientists discover new species of frog in the Amazon rainforest.",
        "Breaking: Supreme Court rules on landmark case affecting digital privacy.",
        "I just finished reading a book that completely changed how I think about economics.",
    ]})
    labels = {
        "discusses politics": "The text makes reference to political figures, institutions, or issues.",
        "news headline": "The text is written in the style of a journalistic news headline.",
        "discusses animals": "The text includes mention of animals, whether real or metaphorical.",
        "first person perspective": "The text is written from a personal point of view.",
        "focus is on gun control": "The text's primary subject is gun control, firearm regulations, or related debates.",
    }
    cls_res = await leximeter.classify(
        toy_data, "text", labels=labels, save_dir="runs/toy_classify",
        model=MODEL, modality="text", n_runs=1, reset_files=True, client=CLIENT,
    )
    print(cls_res.to_string(index=False))

    # ── 3. rate：对文本打分（notebook cell 82）────────────────────────
    section("3. rate — 文本评分")
    rate_texts = pd.DataFrame({"text": [
        "Following last quarter's review, I'm confident we can ship the analytics dashboard by Friday with the current headcount.",
        "I'm drowning in tickets; the build keeps failing and it's wearing me down.",
        "Today's walk by the bay shook off the fog in my head; I feel ready to try again.",
        "The rollout was mismanaged; we flagged the risk repeatedly and were summarily ignored.",
        "Preliminary analyses suggest the intervention achieved measurable improvements across key outcomes.",
        "My calendar is stacked, sleep is thin, but I think we can thread the needle if we cut scope.",
        "Let's merge—benchmarks beat baseline and the regressions look contained.",
        "Honestly, I'm lost; nothing I do seems to move the needle and it's exhausting.",
        "The exhibition balanced restraint with daring; a surprisingly confident curatorial voice.",
    ]})
    rate_attributes = {
        "focus on work life": "How much the text centers on jobs, projects, deadlines, meetings, or professional goals (0 = not about work, 100 = entirely about work).",
        "optimism": "How positive and forward-looking the text feels (near 0 = pessimistic/hopeless, near 100 = highly hopeful and upbeat).",
        "anger": "How much frustration, irritation, or hostility is expressed (near 0 = none, 100 = very angry).",
        "formality": "How formal, structured, and professional the language is (near 0 = very casual/slangy, near 100 = highly formal/academic).",
        "confidence": "How self-assured the speaker sounds about claims or outcomes (near 0 = doubtful/uncertain, near 100 = very certain/assertive).",
        "personal stress": "How strained, overwhelmed, or anxious the speaker sounds (near 0 = calm/relaxed, near 100 = extremely stressed).",
    }
    rate_res = await leximeter.rate(
        rate_texts, "text", attributes=rate_attributes, save_dir="runs/toy_rate",
        model=MODEL, modality="text", n_runs=1, reset_files=True, client=CLIENT,
    )
    print(rate_res.to_string(index=False))

    # ── 4. rate entity：给实体打分（notebook cell 88）────────────────
    section("4. rate — 实体评分")
    entity_data = pd.DataFrame({"entity": [
        "turkey", "pumpkin pie", "pecan pie", "green bean casserole",
        "mashed potatoes", "cornbread", "paneer makhani",
    ]})
    entity_attrs = {
        "savory taste": "",
        "sweet taste": "",
        "bitter taste": "",
        "salty taste": "",
        "sour taste": "",
        "cornbready taste": "",
    }
    entity_res = await leximeter.rate(
        entity_data, "entity", attributes=entity_attrs, save_dir="runs/entity_rate",
        model=MODEL, modality="entity", n_runs=3, reset_files=True, client=CLIENT,
    )
    print(entity_res.to_string(index=False))

    # ── 5. rank：两两比较相对排序（notebook cell 116）─────────────────
    section("5. rank — 两两排序")
    rank_data = pd.DataFrame({"text": [
        "While your conclusion is careless, the deployment will likely improve consumer surplus.",
        "R&D intensity rose from 12% to 18% YoY; margins may compress before productivity gains materialize.",
        "That take is clown shoes—learn the basics before posting.",
        "Low-key hyped for on-device models—privacy win without the lag.",
        "@mina you're right; the vendor ghosted us again—unacceptable.",
        "I concur with @lee: the policy update meaningfully strengthens user control.",
        "Subscription gating for security patches undermines trust.",
        "Empirical evidence suggests platform concentration exacerbates rent extraction absent interoperability mandates.",
        "New chips slap; battery life finally sane.",
        "Your deterministic framing is naive; nonetheless, the architecture promises substantial welfare gains.",
        "If inference cost drops below $0.001/query, long-tail apps become viable; otherwise, unit economics fail.",
        "@devon is right—bait-and-switch trials are predatory; users deserve defaults that respect consent.",
        "Benchmarks improved 12 to 19 on MTEB; small but real.",
        "Anyone shouting \"AI is over\" is farming engagement—spare us.",
        "The deployment evidences credible privacy-by-design, a salutary precedent for the sector.",
        "Love the features, hate the upsells—pick a lane.",
        "Claims of AGI progress conflate scaling curves with capability thresholds; the evidence remains circumstantial.",
        "Good thread, @arya—clear, fair, helpful.",
        "Yes, lock-in is gross, but cross-vendor APIs mean switching costs are falling.",
        "With respect, your argument misrepresents the data and borders on bad faith.",
    ]})
    rank_attrs = {
        "optimistic about technology": "Positive outlook on technology's future or impact.",
        "anger": "Expression of frustration or hostility.",
        "formality": "Professional or academic style and structure.",
        "toxicity towards other users": "Insults, belittling, or demeaning remarks aimed at other users or groups.",
        "use of logic": "Explicit reasoning or evidence (numbers, if-then claims, comparisons, citations).",
    }
    rank_res = await leximeter.rank(
        rank_data, "text", attributes=rank_attrs, save_dir="runs/toy_rank",
        model=MODEL, modality="text", n_rounds=3, matches_per_round=5,
        reset_files=True, client=CLIENT,
    )
    print(rank_res.to_string(index=False))

    # ── 6. extract：抽取产品信息（notebook cell 124）──────────────────
    section("6. extract — 抽取产品信息")
    consumer_products = [
        "Walkman", "iPad", "Roomba", "Kodak film camera", "Dyson vacuum", "Nintendo Switch",
        "Game Boy", "Instant Pot", "Sony Trinitron TV", "George Foreman Grill", "GoPro",
        "Polaroid camera", "Fitbit", "Google Chromecast", "Bose noise-canceling headphones",
        "BlackBerry", "Palm Pilot", "Apple Watch", "Maglite flashlight", "Keurig coffee maker",
        "PlayStation 5", "Super Soaker", "Lego Technic", "Nespresso machine",
        "Scrabble board game", "Vinyl record player", "Kindle e-reader", "Electric toothbrush",
        "Hoverboard", "Tamagotchi", "AirPods", "Cabbage Patch Kids", "Sony Discman",
        "Ring doorbell", "Tile tracker", "Rubik's Cube", "iMac G3", "Coleman cooler",
        "Fitbit Charge", "Roku Streaming Stick",
    ]
    product_df = pd.DataFrame({"product": consumer_products})
    extract_attrs = {
        "company": "The name of the company that released this product.",
        "year of release": "The year the first iteration of this product was released.",
        "ceo at time of release": "The CEO of the company (not at present necessarily, but when the product was first released).",
        "unit price": "The unit price of the product at the time it was first released.",
        "volume": "The approximate volume in cm^3 of the product.",
    }
    ext_res = await leximeter.extract(
        product_df, "product", attributes=extract_attrs, save_dir="runs/toy_extract",
        model=MODEL, modality="entity", reset_files=True, client=CLIENT,
    )
    print(ext_res.head().to_string(index=False))
    print(f"  … ({len(ext_res)} rows total)")

    # ── 7. merge：跨表模糊匹配（notebook cell 186）────────────────────
    section("7. merge — 公司名模糊合并")
    df1 = pd.DataFrame({
        "company_id": range(1, 11),
        "company_name": [
            "Google Inc.", "Microsoft Corporation", "Apple Inc", "Amazon.com", "Facebook",
            "Netflix LLC", "Tesla Motors", "Uber Technologies", "Airbnb", "OpenAI",
        ],
    })
    df2 = pd.DataFrame({
        "record_id": range(1, 15),
        "name": [
            "Gooogle Incorporated", "MSFT", "Apple Incorporated", "Amazn.com",
            "Meta Platforms", "Net Flix", "Tesla, Inc.", "Uber Tech", "Air BnB", "OpenAI",
            "Intl Business Machines", "Twiter", "GOOGL", "Microsft",
        ],
    })
    merge_res = await leximeter.merge(
        df1, df2, left_on="company_name", right_on="name", how="left",
        save_dir="runs/toy_merge", model=MODEL, use_embeddings=True,
        reset_files=True, client=CLIENT,
    )
    print(merge_res.to_string(index=False))

    # ── 8. deduplicate：技术名词去重（notebook cell 197）──────────────
    section("8. deduplicate — 技术名词去重")
    technologies = [
        "iPhone", "Iphone 6", "iPhone 6s", "Apple smartphone", "F-16 Fighting Falcon", "F16",
        "GeForce RTX 4090", "GPU card", "bluetooth", "Bluetooth 5", "Li-ion battery",
        "Lithium ion cell", "Steam engine", "rotative steam-engine", "Electric Vehicle",
        "Tesla Model 3", "wireless earbuds", "AirPods", "photovoltaic panel", "solar pannel",
        "GPS", "satnav", "gps", "PS5", "PlayStation 5", "PlayStation", "Roomba",
        "MRI scanner", "quantum computer", "drone quadcopter", "Nintendo Switch",
        "hydraulic press",
    ]
    tech_df = pd.DataFrame({"technology": technologies})
    dedup_instructions = (
        "Ensure all slight variants of the same technology are mapped to their best "
        "representative term from the list of technologies. For example, \"iPhone 4\", "
        "\"iPhone 6\", \"iPhone 15\" etc all map to \"iPhone\". All specific versions of a "
        "tech should be mapped to their generic tech name; avoid specificity."
    )
    dedup_res = await leximeter.deduplicate(
        tech_df, "technology", save_dir="runs/toy_deduplication",
        model=MODEL, additional_instructions=dedup_instructions,
        use_embeddings=True, modality="entity", reset_files=True, client=CLIENT,
    )
    print(dedup_res.to_string(index=False))

    # ── 9. filter：自然语言条件筛选（notebook cell 201）───────────────
    section("9. filter — 筛出冠军运动员")
    athletes = [
        "Dan Marino", "Jennifer Lopez", "Allen Iverson", "Tom Cruise", "Andrew Luck",
        "Tom Brady", "Serena Williams", "Pete Davidson", "Lando Norris", "Rory McIlroy",
        "Usain Bolt", "Chris Paul", "Ryan Gosling", "Randy Moss", "Gordon Ramsay",
        "Katie Ledecky", "James Harden", "Patrick Ewing", "Zendaya", "Neymar",
        "Michael Phelps", "Conor McGregor", "Emma Stone", "Barry Sanders",
        "Sachin Tendulkar", "Abby Wambach", "Virat Kohli", "Tobey Maguire",
        "Simone Biles", "Fernando Alonso", "Billie Eilish", "Charles Barkley",
        "Tiger Woods", "Kylian Mbappe", "Leonardo DiCaprio", "Ariana Grande",
        "Dirk Nowitzki", "Mike Trout", "Hank Aaron",
    ]
    athlete_df = pd.DataFrame({"athlete": athletes})
    condition = (
        "The person must be an athlete who has won the top award in their sport "
        "(e.g Super Bowl, Olympic gold, etc) at least once. Awards like MVP don't count; "
        "championships only. Only include championship winning athletes."
    )
    flt_res = await leximeter.filter(
        athlete_df, "athlete", condition=condition, save_dir="runs/toy_filtering",
        model=MODEL, entities_per_call=100, shuffle=True, n_runs=3, threshold=0.25,
        reset_files=True, client=CLIENT,
    )
    print(flt_res[["athlete", "keep", "score"]].to_string(index=False))

    # ── 10. deidentify：隐私脱敏（notebook cell 213）──────────────────
    section("10. deidentify — 隐私脱敏")
    deid_data = pd.DataFrame({
        "text": [
            "Elizabeth Green—known as Liz to most of her colleagues—presented her findings in Manchester on 2 February 2023 before catching the evening train back to York for her birthday dinner.",
            "While reviewing interview transcripts in Bilbao, research assistant John Carter realized that another participant, also named John but surnamed Alvarez, had been mixed into his April 2024 notes.",
            "Dr. Farah El-Hakim flew from Cairo to the equally bustling city of Karachi on 10 May 2025 to compare urban water-usage diaries collected by local volunteers.",
            "Please reach our field office at (415) 555-0771 if your luggage was misplaced during the overnight survey trip to Juneau on 17 December 2022.",
            "Barack Obama stopped by a public-policy round-table at the University of Chicago on 9 March 2024, a session later cited in anthropology student Talia Mendelsohn's thesis.",
            "After two weeks in the fishing village of Roscoff, Pierre-Louis Martin mailed handwritten questionnaires to participants in the similarly sized town of Dingle, Ireland, on 6 August 2023.",
            "Mikhail \"Misha\" Ivanov annotated sixteen oral histories in Novosibirsk on 12 January 2022, occasionally initialing pages simply \"M. Ivanov\" when pressed for time.",
            "Former classmates Jose Hernandez and Jose Luis Rey reunited in Cordoba on 21 October 2023, quickly realizing their migration stories had diverged since graduation.",
            "The focus-group transcripts note that Serena Williams—cited as an example of media representation—appeared on screen behind the moderator during the session in Melbourne on 28 January 2024.",
            "On 5 June 2025, research coordinator Chi-Hoon Park emailed the project list at chihoon.park@ricefieldlab.org describing her return to Seoul after a month of fieldwork in Ulaanbaatar and attaching a fifty-page observation log.",
        ]
    })
    deid_res = await leximeter.deidentify(
        deid_data, "text", save_dir="runs/toy_deidentify",
        model=MODEL, max_words_per_call=7500, n_passes=1,
        reset_files=True, client=CLIENT,
    )
    print(deid_res["deidentified_text"].to_string(index=False))

    # ── 11. paraphrase：按指令改写（notebook cell 240）────────────────
    section("11. paraphrase — 按指令改写")
    paraphrase_data = pd.DataFrame({
        "text": [
            "The state of our union is strong. We have created millions of jobs and lifted wages across the nation.",
            "Congress must act now to address the growing crisis in our healthcare system.",
        ]
    })
    para_instructions = (
        "Summarize the speech into a short two sentence easy to read summary followed by "
        "a limerick which captures the core content of the speech. Focus only on domestic "
        "policy, include nothing on foreign affairs in your output."
    )
    para_res = await leximeter.paraphrase(
        paraphrase_data, "text", instructions=para_instructions,
        save_dir="runs/toy_paraphrase", model=MODEL, n_rounds=1, n_runs=1,
        reset_files=True, client=CLIENT,
    )
    print(para_res.to_string(index=False))

    # ── 12. seed：生成代表性人物（notebook cell 223）──────────────────
    section("12. seed — 生成代表性人物")
    seed_instructions = (
        "Each entity is two sentences describing core demographic traits of a person "
        "(age, sex, hometown, race, education level, occupation, etc). The overall set "
        "must accurately represent the US population demographic distribution (including "
        "children) on all counts."
    )
    seed_res = await leximeter.seed(
        seed_instructions, save_dir="runs/toy_seed", model=MODEL,
        num_entities=10, entities_per_generation=5,
        existing_entities_cap=10, reset_files=True, client=CLIENT,
    )
    print(seed_res.to_string(index=False))

    # ── 13. whatever：任意提示词（notebook cell 256）──────────────────
    section("13. whatever — 任意提示词")
    sample_prompts = [
        "Define artificial intelligence.",
        "What are neural networks used for?",
        "Explain the Turing test.",
        "Describe the concept of machine learning.",
        "List applications of computer vision.",
    ]
    whatever_res = await leximeter.whatever(
        sample_prompts, save_dir="runs/toy_custom_prompts",
        model=MODEL, reset_files=True, client=CLIENT,
    )
    print(whatever_res[["Identifier", "Response"]].to_string(index=False))

    # ── 14. whatever + web_search：联网检索（notebook cell 266）───────
    # notebook 用原生 Responses web_search；leximeter 用独立 SearchTool 做检索增强。
    # 需要 TAVILY_API_KEY（或注入 SEARCH_TOOL）；未配置时按原样跑（无检索）。
    section("14. whatever — 联网检索（web_search）")
    regions = ["Monroe County, NY", "Travis County, TX", "King County, WA"]
    web_instructions = (
        "Investigate what is taught in local public school US history classes in this "
        "county. Focus on primary sources like school websites, course descriptions and "
        "syllabi. Summarize the specific topics and periods emphasized."
    )
    prompt_df = pd.DataFrame({
        "prompt": [
            f"Region: {region}\n\n{web_instructions}" for region in regions
        ],
        "identifier": [f"{region}|us-history" for region in regions],
        "region": regions,
    })
    web_res = await leximeter.whatever(
        df=prompt_df,
        save_dir="runs/toy_web",
        column_name="prompt",
        identifier_column="identifier",
        file_name="regional_responses.csv",
        model=MODEL,
        web_search=True,
        search_context_size="medium",
        search_tool=SEARCH_TOOL,
        reset_files=True,
        client=CLIENT,
    )
    print(web_res[["Identifier", "Response"]].to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
