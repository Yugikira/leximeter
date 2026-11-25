# Leximeter

**文本专用、厂商中立**的 LLM 测量库——把杂乱无章的定性文本（访谈、演讲、评论、档案…）转化为可分析的结构化数据。

fork 自 [openai/GABRIEL](https://github.com/openai/GABRIEL)，保留其全部文本功能（含 8 项复杂行为：extract 多实体、rank BT 标准误、bucket 多轮投票、codify 补全循环、paraphrase 递归校验、discover `net_pct` 汇总、merge kmeans2 聚类、debias 两步去偏），重做内部结构，解决三个痛点：

- **解耦**：把 6900 行的 `openai_utils.py` 单体切成「协议层 `client/`」+「通用编排层 `engine/`」，并用基类 `Task`/`AttributeTask` 消除各任务间的复制粘贴。
- **厂商中立**：`LLMClient`（`ResponsesClient` / `ChatCompletionsClient`）可插拔地接 OpenAI / DeepSeek / MiniMax / Ollama 等任何兼容后端；**Decision Provider**（`DecisionClient` / `SystemOneClient`，`POST /v1/systemone`）支持 TypeSafe Jev、OpenRouter 44 款决策模型（Cloudflare Clef / Clef-Flash、Google `gemini-decision-1`、OpenAI `gpt-6-luna-decisions` 等）、AMD Radeon Token Factory（`d3` / `d3-flash`）及自托管 vLLM / SGLang；嵌入（本地 BGE）与联网搜索（Tavily）同样作为独立可选 provider 单独配置。
- **KV-Cache 友好**：全部 14 份提示词模板改为前缀固定（静态说明、量表与定义置顶，动态文本/配对置底），默认 `order_strategy="per_run"` + `batch_execute` 自适应首条预热；DeepSeek 实测缓存命中率 `61.5%–86.7%`、总费用下降 `23%–62%`，且与上游 GABRIEL 测量质量保持一致（详见 [research/kv-cache-experiment-results.md](research/kv-cache-experiment-results.md)）。
- **可测试**：所有 provider 可注入假实现（含 `DummyClient` / `DummyDecisionClient`），零密钥零联网跑通全链路（201 个测试全绿）；全量任务及 OpenRouter / AMD Radeon / DeepSeek 真实端点均已冒烟验证。

> **面向研究者**：使用指南见 [leximeter_wiki/index.html](leximeter_wiki/index.html)（含每个功能的示例代码、真实运行结果、算法与提示词讲解、全部参数说明）。
> 术语表见 [GLOSSARY.md](GLOSSARY.md)，架构决策见 [docs/adr/](docs/adr/)，设计规范见 [research/spec.md](research/spec.md)，研究与实测报告见 [research/classifier-models-research.md](research/classifier-models-research.md) 与 [research/kv-cache-experiment-results.md](research/kv-cache-experiment-results.md)，GABRIEL 源码移植笔记见 [research/gabriel-source-notes.md](research/gabriel-source-notes.md)。

## 安装

```bash
# 用 uv（推荐）
uv sync

# 或 pip
pip install -e .

# 可选依赖
pip install -e '.[embeddings]'   # 本地 BGE 嵌入
pip install -e '.[search]'      # Tavily 联网搜索
```

## 配置 Provider（.env）

所有任务底层都走 `LLMClient`。在项目根目录放一份 `.env`（可复制 `.env.example` 后填写；`uv sync` 会自动创建 venv，`.env` 由库导入时自动加载，无需手动 export）：

```bash
# .env —— 全部键可选；已存在的真实环境变量优先于 .env
LEXIMETER_PROTOCOL=chat                                        # responses | chat（默认 chat 时看下方端点）
LEXIMETER_BASE_URL=https://api.deepseek.com                    # Chat Completions 端点（DeepSeek/MiniMax/Ollama/vLLM…）
LEXIMETER_API_KEY=sk-...
LEXIMETER_MODEL=deepseek-chat

# 走 OpenAI Responses API 时（默认协议），只需：
OPENAI_API_KEY=sk-...

# 嵌入：local（默认，本地 BGE，需 [embeddings]）| openai | none（禁用，merge/deduplicate 自动走纯 LLM 路径）
LEXIMETER_EMBEDDING_PROVIDER=local
LEXIMETER_EMBEDDING_MODEL=BAAI/bge-m3
# openai 专用：嵌入端点/密钥与 LLM 相互独立（可用 SiliconFlow 等独立嵌入服务，
# 如 LEXIMETER_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1 + bge-m3 模型）；
# 不设则默认同 LEXIMETER_BASE_URL / LEXIMETER_API_KEY / OPENAI_BASE_URL
LEXIMETER_EMBEDDING_BASE_URL=
LEXIMETER_EMBEDDING_API_KEY=

# 联网搜索：tavily | none（默认）
LEXIMETER_SEARCH_PROVIDER=none
TAVILY_API_KEY=tvly-...

# 决策模型 provider（System One）：systemone | dummy | none（默认禁用，自动回退 LLM）
# 支持 TypeSafe Jev、OpenRouter（/api/v1/systemone 或 /api/alpha/decisions）、
# AMD Token Factory（d3 / d3-flash）及自托管 vLLM / SGLang
LEXIMETER_DECISION_PROTOCOL=none
LEXIMETER_DECISION_BASE_URL=https://openrouter.ai/api
LEXIMETER_DECISION_API_KEY=sk-or-...
LEXIMETER_DECISION_MODEL=cloudflare/clef-flash

# safest_json 解析兜底的 LLM 修复模型（默认 gpt-5.4-mini）
JSON_LLM_MODEL=gpt-5.4-mini
```

- 加载时机：`import leximeter` 时自动 `load_dotenv()`（不覆盖已有环境变量）——即 **真实环境变量 > `.env`**。
- 每次调用仍可用 `client=` / `decision_client=` / `embedding_client=` / `search_tool=` 覆盖，见下文「Decision Provider」与「Provider 覆盖」。

## 快速上手

```python
import pandas as pd
import leximeter

toy_data = pd.DataFrame({"entity": ["turkey", "pumpkin pie", "green bean casserole", "cornbread"]})
attributes = {
    "savory taste": "How savory the dish is",
    "sweet taste": "Dessert-like sweetness",
}

results = await leximeter.rate(
    toy_data, "entity", attributes=attributes, save_dir="toy_rate",
    modality="entity",
)
results.head()   # 一列一个属性（0–100）
```

## 功能一览

| 类别 | 函数 |
|---|---|
| **测量** | `rate`（0–100 评分）、`rank`（两两排序，BT 拟合 + z 分数 + 标准误）、`classify`（标签）、`extract`（事实抽取，支持多实体展开）、`discover`（发现判别特征，pair 模式出 `net_pct` 汇总） |
| **清洗** | `merge`（模糊合并，kmeans2 聚类）、`deduplicate`（去重）、`filter`（条件筛选）、`deidentify`（脱敏） |
| **生成/辅助** | `seed`、`ideate`、`poll`、`paraphrase`（递归校验）、`codify`（补全循环）、`compare`、`bucket`（多轮投票）、`debias`（两步去偏）、`whatever` |
| **数据** | `load`（文件夹 → DataFrame） |

## 逐任务参考

> 约定：`{base}` = `save_dir/{column_name}` 或 `save_dir/{n}_tokens` 等任务基名。`reset_files=False`（默认）时断点续跑，见「断点续跑」节。

### rate / classify / filter / deidentify / seed / ideate / poll（AttributeTask 系）

- `rate(df, column_name, *, attributes, save_dir, n_runs=1, n_parallels=50, modality="text", ...)` → 每属性一列（0–100）。
- `classify(df, column_name, *, labels, ...)` → 每标签一列（True/False）。
- `filter(df, column_name, *, condition, ...)` → 布尔列。
- `deidentify(df, column_name, *, grouping_column=None, mapping_column=None, ...)` → 打码列。
- `seed(instructions, *, num_entities=1000, ...)` / `ideate(topic, *, n_ideas=1000, ...)` / `poll(df=None, *, population_description, questions, ...)`。
- 落盘：`{base}_raw_responses.csv`（原始响应）、`{base}_cleaned.csv`（解析后聚合）、`n_runs>1` 时 `{base}_full_disaggregated.csv`（MultiIndex id×run 全量）；属性持久化 `{base}_attrs.json`。

### extract（多实体展开）

- `extract(df, column_name, *, attributes, types=None, ...)` → 每个实体一行：`entity_name` 列 + 每属性一列。
- 嵌套 dict 值自动按 entity 展开多行（打印 drop_duplicates 提示）；聚合取首个非 "unknown"。
- `types` 强转：`int` → `Int64`（四舍五入）、`datetime` → `pd.to_datetime`，失败计数打印；强转结果落 `{base}_cleaned_coerced.csv`。

### rank（BT MLE + 标准误）

- `rank(df, column_name, *, attributes, n_rounds=5, matches_per_round=3, ...)` → 每属性三列：`{attr}`（总体 z-score）、`{attr}_raw`（居中 BT 分）、`{attr}_se`（标准误）。
- 每轮结束用**全部历史 outcomes** 重拟合；每轮落 `{base}_round{rnd}.csv`（含 Batch/Pair/IdA/IdB 列），断点续跑按 round CSV 数推断。

### bucket（三阶段多轮投票）

- `bucket(df, column_name, *, bucket_count=10, repeat_bucketing=5, next_round_frac=0.5, top_k_per_round=3, raw_term_definitions=False, ...)` → `bucket_definitions.csv`（bucket + definition）与 DataFrame。
- 生成 → 归约（tally 降序 + `keep=max(bucket_count, len·next_round_frac)`）→ 终选（`top_k_per_round` 补足）三阶段；`bucket_state.json` 全程状态，**签名（sha1，含 term 定义）变化自动清空**；阶段文件 `bucket_generation.csv` / `vote_reduce{r}.csv` / `vote_final{i}.csv`。

### codify（补全循环）

- `codify(df, column_name, *, categories=None, dynamic_mode=False, n_rounds=1, ...)`
  - 静态模式（给定 `categories`）：**每类别一列**，值为 snippet 列表（GABRIEL 契约）。
  - 动态模式：`coded_passages` 单列。
- `n_rounds>1` 时补全循环：每轮 Classify 判「残留文本仍有可编码片段」→ strip 已提取片段 → 只补 flagged 类别；每轮落 `completion_checks/{stem}_completion_iter{i}.csv` + `{stem}_iter{i}.csv`；最终落 `coded_passages.csv`。

### paraphrase（递归校验）

- `paraphrase(df, column_name, *, instructions, n_rounds=1, later_round_run_multiplier=5, validation_attribute=None, ...)` → `{column_name}`（+多 run `_{i}`）+ 每列 `{column_name}_approved`。
- `n_rounds>1` 走递归校验：候选 `round{r}_cand{c}`，每轮 Classify 判 `instructions_followed`，失败进入下一轮；每轮落 `{stem}_round{r}.csv`。

### discover（net_pct 汇总）

- `discover(df, *, column_name | circle_column_name+square_column_name, bucket_count=10, max_retries=0, ...)` → 8 键 dict：`candidates, buckets, classification, bucket_df, summary, compare|codify`。
- pair 模式：`build_combined_metadata` 生成 `{lab}_actual`/`{lab}_inverted`（swap_cs 整词互换）；`summary` 列 `label, actual_true, inverted_true, total, actual_pct, inverted_pct, net_pct`。
- 落盘：`discover_results_snapshot.json` + `discover_results_export.zip`（每键一个 csv）；`discover_state.json` 记录输入签名——**同输入重跑直接加载 snapshot，零 provider 调用**（实测：fresh 1350s → resume 0.0s）。

### merge（kmeans2 + attempts 循环）

- `merge(df_left, df_right, *, on|left_on+right_on, how="left", short_list_len=16, long_list_len=256, use_embeddings=True, max_attempts=4, auto_match_threshold=0.75, ...)` → 左表按右键合并结果。
- short 侧在 `max_attempts` 轮内匹配（short_len 16→8→4→2）；剩余键 kmeans2 聚类（`use_embeddings` 需 long 表 ≥ 256 项，否则纯 LLM）+ 簇内分段 + 窗口候选；`sims >= auto_match_threshold` 唯一命中直接自动匹配。
- 落盘：`merge_responses.csv`、组计划 `merge_groups_attempt{n}.json`（断点续跑）、进度 `merge_progress.csv`、嵌入缓存 `short_embeddings.json` / `long_embeddings.json`。

### deduplicate

- `deduplicate(df, column_name, *, group_size=500, ...)` → 每项一行 + `{column_name}_deduplicated` 代表列。
- 落盘：`deduplicate_responses.csv`、`{base}_cleaned.csv`。

### debias（两步去偏 + 稳健回归）

- `debias(df, column_name, *, measurement_attribute, removal_attribute, attributes, signal_dictionary, removal_method="codify", remaining_signal=True, strip_percentages=None, robust_regression=True, random_seed=12345, run_name=None, max_retries=0, ...)` → `DebiasResult(results, metadata, regression)`。
- 流程：基线测量 → `_strip_passages`（确定性 RNG，seed 含 random_seed:row:pct:n）剥离 confounder 变体 → 剩余信号测量（恒 Rate）→ 两步去偏 `{attr} (debiased) = y - s + δ·r` + 差异列 `{attr} (raw - stripped)`。
- 回归：`fit_ols`（statsmodels HC3 → HC1 回退 → numpy 兜底）；`robust_regression` 只影响 SE 不影响系数。
- 落盘（`run_dir = save_dir/{run_name or debias_{attr}}`）：`debias_results.csv` + `debias_metadata.json` + `debias_state.json`（输入签名含**整个输入 df** 与全部关键参数——同输入重跑直接重建 `DebiasResult`，零 provider 调用）。

### whatever（自定义提示）

- `whatever(prompts | df+column_name, *, identifiers=None, save_dir, ...)` → DataFrame。任意提示走同一引擎与检查点设施。
- **联网检索（web_search）**：传 `web_search=True`（或提供 `web_search_filters`）时，每个 prompt 先用检索工具（Tavily）取回上下文、拼到 prompt 头部再交给模型——相当于 GABRIEL 教程里 `whatever` + 原生 web_search 的厂商中立版。参数对齐 GABRIEL：
  - `web_search: bool | None`（`None` 时由 `web_search_filters` 是否存在推导）
  - `web_search_filters: dict`：`{"allowed_domains": [...], "country": ...}` → Tavily `include_domains` / `country`
  - `search_context_size: "low"|"medium"|"high"` → Tavily `search_depth`（`small`/`large` 别名兼容）
  - 需要 `LEXIMETER_SEARCH_PROVIDER=tavily` + `TAVILY_API_KEY`（或注入 `search_tool=`）；未配置时 warning 并按原样跑。

## 断点续跑与 reset_files

- **逐请求检查点**：所有任务按确定性 identifier（文本哈希 + batch/run/iteration 段）写入检查点 CSV——已完成的样本不会重跑（`CountingClient` 在测试中断言无重复调用）。
- **任务级已完成验证**（TICKET-001）：`discover` / `debias` / `ideate` / `poll` / `seed` / `codify` 维护 `{task}_state.json`（`input_signature` = 输入数据 + 关键参数的 sha1）。**同输入重跑直接加载结果文件，零 provider 调用**；输入（含数据、`model` 等参数）变化则全量重跑，绝不误用旧结果。
- **任务级 `max_retries`**（`discover` / `debias` / `ideate` / `poll` / `seed`，默认 0）：整个 run 抛异常时自动从检查点续跑（重试强制 `reset_files=False`），最多重试 `max_retries` 次；参数错误等确定性异常不重试。单次请求的重试由 SDK 层负责（默认 600s 超时 + 2 次重试），不叠加。
- `reset_files=True` 时清空本任务的检查点文件与 state 后重跑；`save_dir` 与 column_name 相同即视为同一任务。
- **标识符为 GABRIEL 契约格式**：从旧版 leximeter 升级后 identifier 变化会使旧检查点失效——用 `reset_files=True` 或删除 `save_dir` 重跑（详见 spec「已知偏离与代价」）。

## Decision Provider（System One 决策模型，v0.2.0）

对于有界判断类任务，专用 **Decision Provider**（如 TypeSafe Jev、Cloudflare Clef / Clef-Flash、Google `gemini-decision-1`、AMD/vLLM-SR `d3` / `d3-flash`、OpenAI `gpt-6-luna-decisions`）可在**单次前向传播**内并行评估几十到上百个类型化问题（`noul` / `choice` / `score`），返回校准概率且 **$0 输出 token 计费**。`SystemOneClient` 实现开放协议 `POST /v1/systemone`，自动完成非 ASCII / 含空格标签键的安全转义、单层重试（遵循 `Retry-After`）与逐题 `refusal` 降级。

- **纯 Decision 路径**（传入 `decision_client=` 或配置 `LEXIMETER_DECISION_PROTOCOL=systemone` 时零 `LLMClient` 调用；未配置时自动回退 `LLMClient`）：
  - `classify`：单列走并行 `noul`（支持 `include_probabilities=True` 输出 `{label}_prob` 列，原始概率始终落盘检查点）；配对 `differentiate=True` 走 3 选 1 `choice`（`circle` / `square` / `neither`，结构性保证互斥并自动适配 Circle/Square 双向极性）。
  - `rank`：两两对决走 4 选 1 `choice`（`circle` / `square` / `draw` / `insufficient_signal`），保留 `circle_first` 随机化与 Bradley–Terry MLE + 标准误。
  - `filter`：反向扇出（`state = condition`，去重候选项批量作为并行 `noul`），单轮即得校准 `score` 与 `keep`。
- **混合路径**（生成阶段继续使用 `LLMClient` + `model`，内部判断子阶段自动委派给 `DecisionClient` + 独立 `decision_model`）：
  - `discover`（Stage 3 `classify`）、`paraphrase`（`n_rounds>1` 递归校验）、`codify`（`n_rounds>1` 补全检查）、`debias`（`mode="classify"` / `mode="rank"`）。

```python
import os
import leximeter
from leximeter.client import SystemOneClient

# 例 1：通过 OpenRouter 调用 Cloudflare Clef-Flash / TypeSafe Jev / Gemini Decision 1
or_decision = SystemOneClient(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
out = await leximeter.classify(
    df, "text", labels=labels, save_dir="run_cls",
    decision_client=or_decision,
    decision_model="cloudflare/clef-flash",
    include_probabilities=True,
)

# 例 2：通过 AMD Radeon Token Factory 调用 d3-flash（或混合搭配 DeepSeek 生成 + d3-flash 校验）
rad_decision = SystemOneClient(
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    api_key=os.environ["RADEON_API_KEY"],
    send_noul_criteria=False,  # d3 严格模式：noul 不附带 criteria
)
revised = await leximeter.paraphrase(
    df, "text", instructions="...", save_dir="run_para",
    model="deepseek-v4-flash",
    n_rounds=2,
    decision_client=rad_decision,
    decision_model="d3-flash",
)
```

## KV-Cache 友好提示词与 `order_strategy`（v0.2.0）

- **前缀固定布局**：全部 14 份提示词模板将静态指令、量表/定义、输出格式与 `additional_instructions` 置顶，动态文本（`Item` / `Circle & Square` / `Persona` / 增量映射）置底；配对任务的 `circle_first` 随机化仅发生在未缓存后缀中，兼顾位置去偏与前缀缓存命中。
- **键序策略 `order_strategy`**（支持于 `rate` / `classify` / `extract` / `rank` / `codify` / `merge`）：
  - `"per_run"`（默认）：按 run-major（`batch → run → item`）派发，同一 run 内所有样本共享固定顺序以最大化 KV-cache 命中，不同 run（`n_runs > 1` / `rank` 多轮）采用严格互异的确定性置换，抵消局部属性锚定偏差。
  - `"fixed"`：所有 run 复用调用方原始顺序，多次 run 可触发跨轮热缓存（实测高达 `86.7%` 命中率）。
  - `"per_item"`：保留上游 GABRIEL 逐条随机打乱行为。
- **自适应首条预热**：`batch_execute` 支持 `warmup_first: bool | None = None`（默认自动在 `len(pending) > max(8, n_parallels)` 时先顺序预热首条请求填充前缀缓存，再并发放行其余请求，小批次零额外串行开销）。
- 实测与质量对照报告见 [research/kv-cache-experiment-results.md](research/kv-cache-experiment-results.md) 与 [docs/adr/0005-prefix-first-prompts-and-per-run-ordering.md](docs/adr/0005-prefix-first-prompts-and-per-run-ordering.md)。

## Provider 覆盖

每次调用可用关键字参数注入 provider（覆盖 `.env` 默认）：

```python
import leximeter
from leximeter.client import ChatCompletionsClient, ResponsesClient, SystemOneClient

results = await leximeter.rate(
    df, "text", attributes=attrs, save_dir="run_a",
    client=ChatCompletionsClient(api_key="sk-...", base_url="https://api.deepseek.com"),  # 本调用走 DeepSeek
)
# 或逐条混用：
a = await leximeter.rate(..., client=ResponsesClient())   # OpenAI
b = await leximeter.classify(..., decision_client=SystemOneClient(...), decision_model="typesafe/jev-1.14")
```

自定义 provider 只需实现最小协议：

```python
class LLMClient:
    async def complete(self, prompt, *, model, json_mode, **kw) -> str: ...

class DecisionClient:
    async def decide(self, state, questions, *, model, **kw) -> DecisionResult: ...

class EmbeddingClient:
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

**测试桩**：`DummyClient` / `DummyDecisionClient` / `DummyEmbeddingClient` / `DummySearchTool` 无需密钥即可跑通全链路，`DictClient` / `DictDecisionClient` / `CountingClient` / `CountingDecisionClient` / `RecordingClient` 见 `tests/conftest.py`。

## 完整示例

[examples/reference.py](examples/reference.py) 与 GABRIEL 教程 notebook 的文本任务一一对应（14 节，每节标注 notebook cell 编号），玩具数据与参数和原项目完全一致，可直接跑完并对照 notebook 的输出（含 `whatever` 联网检索示例，需 `TAVILY_API_KEY`）。配置好 `.env` 后：

```bash
python examples/reference.py
```

## 测试

```bash
uv run pytest        # 零网络、零密钥，用假 provider 跑全链路（201 tests）
```

真实 provider 冒烟（读取 `.env` 配置，逐个任务独立运行，单任务失败不阻塞）：

```bash
uv run python examples/smoke_tasks.py            # 全跑 18 个任务
uv run python examples/smoke_tasks.py rank       # 指定单个任务
uv run python examples/smoke_tasks.py discover_checkpoint   # checkpoint 复验变体（跑两次：fresh → resume 零调用）
```

## FAQ

- **想换模型？** 改 `.env` 的 `LEXIMETER_MODEL`，或每次调用传 `model=` / `client=`。模型名不校验，协议由 `LEXIMETER_PROTOCOL` 决定。
- **怎么用本地模型？** 起一个 OpenAI 兼容网关（Ollama / vLLM / LM Studio），`LEXIMETER_PROTOCOL=chat` + `LEXIMETER_BASE_URL=http://localhost:11434/v1`。
- **嵌入不可用怎么办？** `LEXIMETER_EMBEDDING_PROVIDER=none`（或未装 `[embeddings]`）时，merge/deduplicate 自动回退纯 LLM 路径；merge 的 `use_embeddings=False` 可强制关闭。
- **运行中断了？** 直接重跑同一命令即可续跑：已完成的请求按检查点跳过；完整跑过的复杂任务（discover/debias/ideate/poll/seed/codify）重跑同输入直接加载结果（零 API，`{task}_state.json` 命中）；想强制重跑传 `reset_files=True`。
- **`.env` 不生效？** 确认文件在项目根目录（import leximeter 的进程 CWD 会依次向上找），且同名环境变量未设置（环境变量优先）。
- **输出列与 GABRIEL 不一致？** 列名严格对齐 GABRIEL 契约；本 fork 的差异只有两处：merge 嵌入缓存为 JSON（安全性）、`view` UI 未移植（见 spec Backlog）。

## 引用

本项目 fork 自 GABRIEL，请同时引用上游：

- Asirvatham, H., Mokski, E., and Shleifer, A. (2026). *GPT as a Measurement Tool*. NBER Working Paper No. 34834.
- Asirvatham, H. and Mokski, E. (2026). *GABRIEL*. https://github.com/openai/GABRIEL

## License

Apache License 2.0（同上游 [openai/GABRIEL](https://github.com/openai/GABRIEL)）。详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
