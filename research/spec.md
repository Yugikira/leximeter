# Leximeter 规范（Spec）

> 描述 leximeter 的架构与行为契约，作为后续开发、维护与扩展的权威参考。
> 生成于重构完成后（2026-08-14）。上游 GABRIEL 的移植笔记见同目录 `gabriel-source-notes.md`。

## 1. 目标与非目标

**目标**

- 一个**文本专用、厂商中立**的 LLM 测量库：把定性文本（访谈、演讲、评论…）转化为可分析的结构化数据。
- **解耦**：厂商协议、通用编排、业务任务三层分离，互不纠缠。
- **可维护**：基类消除复制粘贴；每个任务职责单一、可单测。

**非目标（v1 不包含）**

- 多模态（image / audio / pdf / web search 的原生实现）。
- `view` 图形 UI。
- token 用量 / 成本估算（接口已预留 `estimate_tokens` / `estimate_cost`，未实现）。
- bucket 多轮投票、codify 补全循环、paraphrase 递归校验（当前为单遍核心）。

## 2. 架构分层

```
leximeter/
├── api.py          # 公开门面：构造 config → 委托 task
├── client/         # 厂商协议层（协议，非厂商）
├── engine/         # 通用编排层（provider 无关）
├── tasks/          # 业务任务层（Config + Task 模式）
├── prompts/        # Jinja2 提示词模板（文本专用）
└── utils/          # 解析 / 相似度 / jinja 过滤器
```

**依赖方向（单向，不可反向）**：`api → tasks → engine/client → utils`。`engine` 只依赖 `client` 的协议，`tasks` 只依赖 `engine` 的 `batch_execute` 与 `client` 的协议，**任何层都不依赖具体厂商**。

## 3. Provider 契约

### 3.1 `LLMClient`（`client/base.py`）

```python
class LLMClient(ABC):
    async def complete(self, prompt: str, *, model: str, json_mode: bool = False, **generation_kwargs) -> str: ...
    def estimate_tokens(self, prompt: str, *, model: str) -> int: ...   # 预留，未实现
    def estimate_cost(self, prompt: str, *, model: str) -> float: ...   # 预留，未实现
```

**实现（按协议划分，非按厂商）**：

| 类 | 协议 | 说明 |
|---|---|---|
| `ResponsesClient` | OpenAI Responses API | 原生 OpenAI，或任何兼容 Responses 的 base_url |
| `ChatCompletionsClient` | `/v1/chat/completions` | DeepSeek / MiniMax / Ollama / vLLM 等兼容端点（openai SDK + base_url） |
| `DummyClient` | 无网络 | 确定性假响应，测试 / 干跑 |

### 3.2 `EmbeddingClient`（`client/embedding.py`）

```python
class EmbeddingClient(ABC):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

实现：`LocalBgeClient`（sentence-transformers，默认 `BAAI/bge-m3`，`[embeddings]` 可选依赖、懒加载）、`OpenAIEmbeddingClient`、`DummyEmbeddingClient`。

### 3.3 `SearchTool`（`client/search.py`）

```python
class SearchTool(ABC):
    async def search(self, query: str) -> str: ...   # 返回检索到的上下文文本
```

实现：`TavilySearchTool`（`[search]` 可选依赖）、`DummySearchTool`。**未配置时，调用方打 warning 并跳过检索**，不报错。

### 3.4 选择机制

- 环境变量默认：`LEXIMETER_PROTOCOL`（`responses`/`chat`）、`LEXIMETER_BASE_URL`、`LEXIMETER_API_KEY`、`LEXIMETER_EMBEDDING_PROVIDER`、`LEXIMETER_SEARCH_PROVIDER`。
- 每调用覆盖：所有任务接受 `client=` / `embedding_client=` / `search_tool=`。
- `resolve.py` 提供 `resolve_client` / `resolve_embedding_client` / `resolve_search_tool`。

## 4. Engine 契约（`engine/batch.py`）

```python
async def batch_execute(
    prompts, identifiers, *, client, model, json_mode=False,
    n_parallels=50, save_path=None, reset_files=False,
    max_retries=3, timeout=None, **generation_kwargs,
) -> pd.DataFrame  # 列：Identifier, Response, Successful, Error Log, Time Taken
```

**行为**：信号量并发（`n_parallels`）、指数退避重试（`max_retries`）、断点续跑（`save_path` 检查点，已完成 Identifier 跳过）。

**检查点语义**（`engine/checkpoint.py`）：CSV 按 `Identifier` 去重 `keep="last"`；`Successful` 判定完成；增量追加。

## 5. 基类契约（`tasks/base.py`）

### 5.1 `Task`

注入 `client` / `embedding_client` / `search_tool`（惰性解析），展开 `save_dir`，提供 `_hash_identifiers` 等公共助手。抽象 `run()`。

### 5.2 `AttributeTask(Task)` —— 模板方法

子类只实现三个变体钩子：

```python
def render_prompt(self, text, attributes) -> str: ...
async def parse_response(self, raw, attributes) -> dict: ...
def aggregate(self, records, df_proc) -> pd.DataFrame: ...   # 默认按属性取均值
```

基类实现不变骨架：哈希标识符 → 属性持久化 → 属性分批 → 渲染提示词 → `batch_execute` → 解析 → 聚合 → 合并回 df → 落盘。

**标识符约定**：`{sha1_16(text)}_batch{batch_idx}_run{run_idx}`；续跑靠确定性重推导（同 df + 同属性 + 同分批计划 → 同标识符）。

## 6. 公开 API

> 所有测量/清洗/生成函数为 `async`，返回 `DataFrame`（`discover`/`debias` 返回 `dict`）。所有函数接受 `client=` / `embedding_client=` / `search_tool=` 与 `**engine_kwargs`（透传 `max_output_tokens`、`temperature`、`timeout`、`max_retries` 等）。

### 测量

| 函数 | 输出 | 核心参数 |
|---|---|---|
| `rate(df, column_name, *, attributes, save_dir, ...)` | 每属性一列（0–100，跨 run 均值） | `n_runs`, `n_attributes_per_run`, `modality` |
| `rank(df, column_name, *, attributes, save_dir, ...)` | 每属性一列（中心化 BT z 分数） | `n_rounds`, `matches_per_round`, `learning_rate` |
| `classify(df, column_name, *, labels, save_dir, ...)` | 每标签一列（bool，`min_frequency` 聚合） | `n_runs`, `min_frequency` |
| `extract(df, column_name, *, attributes, save_dir, ...)` | 每字段一列（`unknown`→NA，`types` 强转） | `types`, `modality` |
| `discover(df, *, circle/square 或 column_name, save_dir, ...)` | `{"candidates", "buckets"}` | `bucket_count` |

### 清洗

| 函数 | 输出 | 核心参数 |
|---|---|---|
| `merge(df_left, df_right, *, on/left_on/right_on, save_dir, ...)` | 左表 join 右表（LLM 匹配键） | `use_embeddings`, `short_list_len`, `long_list_len` |
| `deduplicate(df, column_name, *, save_dir, ...)` | 加 `mapped_<col>` 列 | `use_embeddings`, `group_size`, `embedding_similarity_threshold` |
| `filter(df, column_name, *, condition, save_dir, ...)` | 加 `keep`/`score` 列 | `entities_per_call`, `n_runs`, `threshold` |
| `deidentify(df, column_name, *, save_dir, ...)` | 加 `deidentified_text`/`mapping` 列 | `grouping_column`, `mapping_column`, `max_words_per_call` |

### 生成 / 辅助

| 函数 | 输出 | 核心参数 |
|---|---|---|
| `seed(instructions, *, save_dir, ...)` | `entity` + `entity_id` 列 | `num_entities`, `entities_per_generation`, `existing_entities_cap` |
| `ideate(topic, *, save_dir, ...)` | 想法报告（`title`/`in_a_nutshell`/…） | `n_ideas`, `scientific_theory` |
| `poll(df=None, *, population_description, questions, save_dir, ...)` | `seed`+`persona`+每题一列 | `num_personas`, `n_questions_per_run`, 分阶段模型 |
| `paraphrase(df, column_name, *, instructions, save_dir, ...)` | 加改写列 | `json_mode`, `revised_column_name` |
| `codify(df, column_name, *, categories, save_dir, ...)` | 加 `coded_passages` 列 | `categories` |
| `compare(df, circle_col, square_col, *, save_dir, ...)` | `circle`/`square`/`attribute`/`explanation` | `differentiate` |
| `bucket(df, column_name, *, save_dir, ...)` | `bucket` + `definition` 列 | `bucket_count`, `n_terms_per_prompt` |
| `debias(df, column_name, *, measurement_attribute, removal_attribute, attributes, save_dir, ...)` | `{"results", "metadata"}` | `measurement_attribute`, `removal_attribute` |
| `whatever(prompts=None, *, df, column_name, save_dir, ...)` | `Identifier`+`Response`（+`Response JSON`） | `json_mode` |

### 数据加载

| 函数 | 输出 | 说明 |
|---|---|---|
| `load(folder_path, *, save_name, modality="text", ...)` | `path`+`name`+`text` 列 | 同步；读文件夹里的文本文件 |

## 7. 已知简化（可后续补）

- `merge`/`deduplicate` 嵌入预筛为**贪心聚类/top-k**（O(n²)），非 GABRIEL 的 kmeans2。
- `rank` 用 Zermelo BT 拟合（非完整 MLE + 标准误）。
- `debias` 用 `np.linalg.lstsq` 简化回归（非 statsmodels 稳健回归）。
- `discover` v1 只到「compare → bucket」两步，未做 paired classification 的 `net_pct` 汇总。
- `codify`/`bucket`/`paraphrase` 为单遍核心（未做多轮补全/投票/递归校验）。
- `extract` v1 单实体展开（多实体展开后置）。
