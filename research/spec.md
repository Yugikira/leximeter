# Leximeter — Spec（PRD）

> 由 `/to-spec` 生成。描述 leximeter 要解决的问题、用户视角的方案、实现与测试决策。

## Problem Statement

社会科学研究者与分析人员需要从海量**定性文本**（访谈、演讲、评论、档案…）中提取**定量测量**（评分、分类、事实、排序）。上游库 GABRIEL 能胜任这件事，但作为使用方与维护方，它有三个痛点：

1. **难维护**：核心引擎 `openai_utils.py` 约 6900 行，把并发调度、限流、重试、成本估算、日志、检查点揉在同一个函数体里；`rank.py` 单文件 2200 行。改一处牵动全局，难以单测。
2. **厂商锁定**：只实现了 `OpenAIClient`，任务层直接 import 全局函数、请求参数硬编码为 OpenAI Responses API。想换 DeepSeek / MiniMax / 开源模型几乎无从下手。
3. **体积冗余**：捆绑了 image / audio / pdf / web 等多模态能力，但用户只需要文本。

用户想要一个**文本专用、厂商中立、易维护**的 fork，保留 GABRIEL 的全部文本功能，但把内部结构重做。

## Solution

**Leximeter**：GABRIEL 的重构 fork，把"厂商协议、通用编排、业务任务"切成三层，并补全 provider 抽象。

- **厂商中立**：`LLMClient.complete()` 单次补全协议 + 按协议划分的 `ResponsesClient` / `ChatCompletionsClient`；嵌入与联网检索作为独立 provider（`EmbeddingClient` / `SearchTool`）可单独配置。
- **易维护**：`openai_utils.py` 一劈为二（`client/` 协议 + `engine/` 编排）；`Task` / `AttributeTask` 模板方法基类消除各任务间的复制粘贴；复杂任务拆成子包。
- **文本专用**：移除多模态，只保留 text / entity；18 个任务（简单 + 复杂）全保留。
- **可测试**：所有 provider 可注入假实现，零密钥零联网跑通全链路。

## User Stories

1. As a researcher, I want to rate passages on natural-language attributes (0–100), so that I can quantify qualitative text.
2. As a researcher, I want to rank items pairwise into relative z-scores, so that I can order things that are hard to score absolutely.
3. As a researcher, I want to classify items into one or more labels, so that I can tag a corpus.
4. As a researcher, I want to extract structured facts per item, so that I can build tidy tables.
5. As a researcher, I want to discover which features distinguish two classes of text, so that I can learn what separates them.
6. As a researcher, I want to fuzzy-join two tables by meaning, so that I can link differently-spelled records.
7. As a researcher, I want to deduplicate equivalent terms to a representative, so that I can collapse variants.
8. As a researcher, I want to filter a large list by a natural-language condition, so that I can narrow candidates.
9. As a researcher, I want to deidentify PII, so that I can share interview corpora safely.
10. As a researcher, I want to seed a representative set of entities, so that I can initialize a diverse sample.
11. As a researcher, I want to ideate novel theories/ideas and get them structured, so that I can procure hypotheses.
12. As a researcher, I want to poll a synthetic population, so that I can simulate survey responses.
13. As a researcher, I want to paraphrase/rewrite text per instructions, so that I can transform a corpus.
14. As a researcher, I want to codify passages against qualitative codes, so that I can annotate text.
15. As a researcher, I want to compare two entries and list differences, so that I can contrast them.
16. As a researcher, I want to bucket many terms into a taxonomy, so that I can find emergent categories.
17. As a researcher, I want to debias a measurement against a confounder, so that I can trust the signal.
18. As a researcher, I want to run arbitrary prompts through the same infra, so that I can cover any bespoke step.
19. As a researcher, I want to load a folder of text files into a DataFrame, so that I can start a task.
20. As a developer using a non-OpenAI provider, I want to point at DeepSeek/MiniMax via a base_url, so that I can use my own model.
21. As a developer using a local model, I want to embed text with a local BGE model, so that I can dedupe/match without an external API.
22. As a developer, I want to configure providers via environment variables, so that I can switch backends without code changes.
23. As a developer, I want to override the provider per-call, so that I can mix providers in one script.
24. As a developer, I want to opt into web search via a pluggable tool, so that I can add retrieval when I need it.
25. As a user, I want interrupted runs to resume from a checkpoint, so that I don't re-spend on large corpora.
26. As a developer, I want to test every task without a network or key, so that the suite runs deterministically.
27. As a maintainer, I want to add a new provider without touching business logic, so that the library stays vendor-neutral.
28. As a maintainer, I want to add a new measurement task in a few lines, so that common logic isn't copy-pasted.

## Implementation Decisions

**架构分层**（依赖单向：`api → tasks → engine/client → utils`）：

- `client/` 厂商协议层；`engine/` 通用编排层（provider 无关）；`tasks/` 业务任务层；`api.py` 门面。

**Provider 抽象**：

- `LLMClient.complete(prompt, *, model, json_mode, **kw) -> str` 为最小原语；`estimate_tokens` / `estimate_cost` 为预留抽象方法（v1 不实现）。
- 实现按**协议**而非厂商：`ResponsesClient`（Responses API）、`ChatCompletionsClient`（`/v1/chat/completions`，走 openai SDK + `base_url`）、`DummyClient`（测试桩）。
- `EmbeddingClient.embed(texts) -> list[list[float]]`，默认 `LocalBgeClient`（`BAAI/bge-m3` 本地 CPU，`[embeddings]` 可选依赖、懒加载）。
- `SearchTool.search(query) -> str`，v1 实现 `TavilySearchTool`；未配置时 warning + 跳过。
- 选择机制：env 默认（`LEXIMETER_PROTOCOL` / `BASE_URL` / `API_KEY` / `EMBEDDING_PROVIDER` / `SEARCH_PROVIDER`）+ 每调用 `client=` / `embedding_client=` / `search_tool=` 覆盖。

**引擎**：

- `batch_execute(prompts, identifiers, *, client, model, json_mode, n_parallels, save_path, reset_files, max_retries, timeout, **kw) -> DataFrame`：信号量并发 + 指数退避重试 + CSV 检查点续跑（按 `Identifier` 去重 keep=last）。

**基类**：

- `Task` 注入 provider、展开 `save_dir`；`AttributeTask` 用模板方法实现"哈希标识符 → 属性持久化 → 分批 → 渲染 → 执行 → 解析 → 聚合 → 合并"骨架，子类只实现 `render_prompt` / `parse_response` / `aggregate`。
- 标识符 `{sha1_16(text)}_batch{b}_run{r}`；续跑靠确定性重推导。

**复杂任务**：

- `rank` / `debias` / `ideate` / `poll` / `discover` 拆成子包；`rank` 用 Zermelo 算法拟合 Bradley–Terry。

**嵌入预筛**：

- `deduplicate` 用嵌入相似度贪心聚类（阈值 0.85）后再 LLM 去重；`merge` 按 GABRIEL 契约走 kmeans2 聚类 + 簇内分段 + 窗口候选再 LLM 匹配（`use_embeddings` 需 long 表 ≥ 256 项才启用，否则纯 LLM 路径）；嵌入不可用时优雅回退。

**API 形态**：

- 参数显式化，去掉 GABRIEL 的 `**cfg_kwargs` + `inspect.signature` 自动路由魔法；额外参数经 `**engine_kwargs` 透传。

## Testing Decisions

- **测外部行为**：只断言返回的 DataFrame（列名、值、行数），不测内部实现细节。
- **单一接缝**：所有测试通过 `client=` / `embedding_client=` / `search_tool=` 注入假 provider（`DictClient` / `TextClient` / `CountingClient` / `DummyEmbeddingClient`）。这是全库唯一需要的测试接缝。
- **零网络零密钥**：`DummyClient` 返回确定性假响应，测试可离线、可复现。
- **覆盖**：32 个测试覆盖每个任务的端到端行为 + 检查点续跑（`CountingClient` 断言无重复调用）+ 嵌入/纯 LLM 两条路径 + fit_ols HC3 数值性质。
- **运行**：`uv run pytest`，pytest 配置 `pythonpath=["src"]`。

## Out of Scope

- 多模态（image / audio / pdf / 原生 web_search）。
- `view` 图形 UI（见 Backlog）。
- token 用量 / 成本估算（接口已预留，未实现）。

## 已恢复功能（2026-08-15）

> 上一轮重构把下列 GABRIEL 功能简化了；本轮按 GABRIEL 源码逐项恢复，identifier 格式、输出列、落盘文件均严格对齐上游。引擎沿用 leximeter 的 `batch_execute` + provider 注入（`client=` / `embedding_client=` / `search_tool=`），不做逐字复制。

| 功能 | 恢复内容 | 契约要点 |
|---|---|---|
| `extract` 多实体 | 独立 Task 重写 | 嵌套 dict 按 entity 展开多行 + `entity_name` 列；聚合取首个非 "unknown"；`types` 强转（int→`Int64`、datetime→`pd.to_datetime`）并落 `{base}_cleaned_coerced.csv`；`n_runs>1` 落 `_full_disaggregated.csv` |
| `rank` BT MLE + 标准误 | `bt.py` 补 `bt_standard_errors` | `fit_bt(return_info=True)` 输出 `n_ij/p_ij`；SE 经信息阵投影 + 岭正则（失败 `nan_to_num` 重试）；每轮用**全部历史 outcomes** 重拟合；输出 `{attr}` / `{attr}_raw` / `{attr}_se` 三列；每轮落 `{base}_round{rnd}.csv`，断点续跑按 round CSV 数推断 |
| `bucket` 多轮投票 | 三阶段重写 | 生成（repeat_bucketing=5）→ 归约（投票 tally 降序 + `keep=max(bucket_count, len·next_round_frac)`）→ 终选（`top_k_per_round` 补足）；`terms_signature`（sha1）变化清空 state；全程 `bucket_state.json` 续跑；落 `bucket_definitions.csv` |
| `codify` 补全循环 | dynamic_mode + `n_rounds>1` 补全 | 静态模式**每类别一列**（snippet 列表），动态 `coded_passages` 单列；补全循环用 Classify 判「残留可编码」，strip 已提取片段后只补 flagged 类别；snippet 经 word_matching 在原文逐字定位；最终落 `coded_passages.csv` |
| `paraphrase` 递归校验 | `n_rounds>1` 走 `_recursive_validate` | 候选 `round{r}_cand{c}`（后续轮 × `later_round_run_multiplier`）；Classify 判 `instructions_followed`；输出 `{base_col}`（+多 run `_{i}`）+ `{base_col}_approved`；轮数耗尽随机取末轮候选 |
| `discover` net_pct 汇总 | 完整链路（single→Codify / pair→Compare→Bucket→分类） | pair 时 `build_combined_metadata`（`{lab}_actual`/`{lab}_inverted`，swap_cs 整词互换）；summary 列 `label, actual_true, inverted_true, total, actual_pct, inverted_pct, net_pct`；返回 8 键；落 `discover_results_snapshot.json` + `discover_results_export.zip` |
| `merge` kmeans2 聚类 | kmeans2 + attempts 循环 | `use_embeddings = cfg.use_embeddings and len(long_uniques) >= 256`；kmeans2(`minit="points"`) 簇内分段 + 窗口候选；`max_attempts=4`（short_len 16→8→4→2）；自动匹配 `sims>=0.75`；identifier `merge_{attempt:02d}_{idx:05d}`；组文件 `merge_groups_attempt{attempt}.json` 断点续跑；`merge_progress.csv` |
| `debias` 稳健回归 | 完整 Debias 任务 | `fit_ols` 移植（statsmodels HC3 → HC1 回退 → `np.linalg.lstsq` 兜底，`robust_regression` 只影响 SE 不影响系数）；两步去偏 `{attr} (debiased) = y - s + δ·r` + 差异列 `{attr} (raw - stripped)`；`_strip_passages` 确定性 RNG（seed `f"{random_seed}:{row_idx}:{pct_strip}:{n_total}"`）；落 `debias_results.csv` + `debias_metadata.json` |

### 已知偏离与代价

- **identifier 格式变化破坏旧检查点**：extract / rank / bucket / codify / paraphrase / merge 的 identifier 已改为 GABRIEL 格式，旧版检查点（CSV / `bucket_state.json` / `merge_groups_attempt*.json` / `attributes.json`）无法续用。属版本演进，接受；`reset_files=True` 或删除 `save_dir` 即可重跑。
- **merge embedding 检查点为 JSON 而非二进制序列化**：GABRIEL 用 `short_embeddings.pkl` / `long_embeddings.pkl`；leximeter 因安全 hook 禁止该类序列化落盘，改用 `short_embeddings.json` / `long_embeddings.json`。存在即读、按文本命中缓存，语义完全一致。
- **debias 不移植 tqdm**（普通 for 循环，进度信息仍打印）；**绘图 PDF 不生成**（`_regress_against_original` 直接回归不画图，见 Backlog）。
- **`response_fn` / `get_all_responses_fn` / `template_path` / `use_dummy`**：GABRIEL 的引擎回调参数不从 kwargs 透传——leximeter 用 `client=` 注入 provider、子任务构造时显式 `client=self._get_client()`。

## Backlog（待办）

> 已确认本次不做、留待后续的事项。to-tickets skill 不可用，故以本表形式发布；如接入 issue tracker 可再逐条转录。

| # | 事项 | 说明 |
|---|---|---|
| B1 | 引擎自适应并发 | GABRIEL 的斜坡启动（ramp-up）、滚动限流、动态超时、风暴保护（storm guard）；当前 `batch_execute` 为固定 `n_parallels` + 指数退避重试 |
| B2 | `view` 图形 UI | GABRIEL 的 view.py（可视化测量结果），文本任务之外的工具 |
| B3 | rank 递归模式 / primer / `initial_rating_pass` | BT 拟合的进阶选项，配对策略本次只移植 `_pairs_info_gain` 核心 |
| B4 | debias 绘图 PDF | `_regress_against_original` 的 matplotlib 出图（当前直接回归）；可挂载 `fpdf`/`matplotlib` 后补 |
| B5 | debias rank mode 的 `id_column` 对齐 | rank 子任务目前不传 `id_column`，索引对齐用兜底（`result.index = df.index`） |
| B6 | `clean_json_df` 接线 | `utils/parsing.py` 已实现函数本体，任务层尚未接入 |
| B7 | token 用量 / 成本估算 | `estimate_tokens` / `estimate_cost` 为预留抽象方法，v1 未实现 |

## Further Notes

- 上游归属：fork 自 openai/GABRIEL，README 保留引用；引用论文见 NBER w34834。
- 移植参考见同目录 `gabriel-source-notes.md`（GABRIEL 源码逐任务契约）。
- 本仓库为本地 git，无 issue tracker 配置，故以 `research/spec.md` 文件形式发布；如需接入 tracker，可再补 `ready-for-agent` 标签。
