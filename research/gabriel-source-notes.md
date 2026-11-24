# GABRIEL 源码移植参考文档（面向 leximeter 重构）

> **适用范围**：本文档基于上游 `openai/GABRIEL` 一手源码精读而成，服务于 leximeter（openai/GABRIEL 的厂商无关、纯文本 fork）的重构。
> **路径约定**：除特别注明外，所有 `文件:行号` 均相对于 `openai/GABRIEL/src/gabriel/`。例如 `api.py:137` 指 `src/gabriel/api.py` 第 137 行。
> **范围声明**：本档只覆盖**文本**路径。image/audio/pdf 多模态实现细节一律不写；`web_search` 只记录其如何依赖 Responses API 原生 tool（leximeter 将用独立 SearchTool 替换）。**embeddings 必须覆盖**，因为 merge/deduplicate/seed/ideate 依赖 `get_embedding`/`get_all_embeddings`。
> **准确度声明**：每个可复现行为契约（签名、默认值、输出列名、落盘规则）均给出精确行号。极少数未能确证的地方标记「此处未深究」，不臆造行号。

---

## 目录

1. [总体架构与数据流](#1-总体架构与数据流)
2. [engine 层 `utils/openai_utils.py`](#2-engine-层-utilsopenai_utilspy)
3. [共享任务基础设施 `tasks/_run_utils.py` 与 `tasks/_attribute_utils.py`](#3-共享任务基础设施)
4. [逐任务参考](#4-逐任务参考)
5. [提示词模板清单](#5-提示词模板清单)
6. [文本工具 utils](#6-文本工具-utils)

---

## 1. 总体架构与数据流

### 1.1 分层

```
api.py（门面，async）            ── 用户唯一入口（gabriel.rate 等）
   │  构造 Config 数据类（dataclass）
   │  调用 TaskClass(cfg).run(...)
   ▼
tasks/<task>.py（任务实现）      ── Config dataclass + Task 类
   │  构造 prompts（Jinja2 渲染）
   │  标识符（Identifier）生成
   ▼
utils/openai_utils.py（engine）  ── get_all_responses / get_all_embeddings
   │  异步 worker 池 / 重试 / 检查点 / 断点续跑
   ▼
  解析 → 聚合 → save_dataframe_with_fallback 落盘 → 返回 DataFrame
```

### 1.2 一条典型调用链：`api.rate()`（`api.py:187`）

`api.py:272-304` 展示了所有门面函数的通用模式：

1. `save_dir` 先做 `expandvars/expanduser` 并 `makedirs`（`api.py:272-273`）。
2. **若 `df is None`**：加载缓存结果 `{base_name}_cleaned.csv`（`api.py:274-277`），委托给 `_load_cached_dataframe`（`api.py:80-124`）。该函数支持读取分片文件 `<stem>_{n}<ext>`（`api.py:85-99, 112-119`）。**注意：只有 text 类任务的最终落盘文件才能这样续跑**；discover 不落最终文件，`df is None` 时直接抛错（见 4.17）。
3. `_split_cfg_and_response_kwargs(RateConfig, dict(cfg_kwargs), task_name="rate")`（`api.py:278-282`）把用户多传的 kwargs 拆成两类。
4. 构造 `RateConfig(...)`（`api.py:283-296`），显式列出公共参数，剩余 `**cfg_kwargs` 透传。
5. `await Rate(cfg, template_path=template_path).run(df, column_name, reset_files=..., response_fn=..., get_all_responses_fn=..., **response_kwargs)`（`api.py:297-304`）。

### 1.3 `_split_cfg_and_response_kwargs` 的「魔法」（leximeter 决定删除，但必须记录）

- `_OPENAI_RESPONSE_OVERRIDES`（`api.py:137-152`）：取 `get_all_responses` 与 `get_response` **全部形参名的并集**，再减去 `{prompt, prompts, identifiers, prompt_images, prompt_audio, prompt_pdfs, prompt_web_search_filters, save_path}`。
- `_split_cfg_and_response_kwargs(cfg_cls, extra_kwargs, *, task_name)`（`api.py:155-184`）：
  - 对每个 key：在 Config 字段集合内 → `cfg_overrides`；在 `_OPENAI_RESPONSE_OVERRIDES` 内 → `response_overrides`；否则进 `unknown_keys`。
  - 有未知 key 就 `raise TypeError(f"Unknown keyword argument(s) for gabriel.{task_name}: ...")`（`api.py:176-182`）。
- **后果**：用户可以在 `gabriel.rate(...)` 里混传两类参数——Config 字段（如 `n_runs`）与 engine 参数（如 `image_detail`、`max_output_tokens`）。leximeter 若删除此机制，需决定引擎参数如何暴露（例如统一 `engine_kwargs=...`）。
- 由于该集合是**从签名动态推导**的，任何新增的 `get_all_responses` 参数会自动成为可透传白名单——移植后要小心签名漂移。

### 1.4 任务统一骨架

每个任务基本都重复以下步骤（以 `rate.py` 为最典型的样本，见第 4 节）：

1. `df_proc = df.reset_index(drop=True).copy()`（`rate.py:114`）。
2. 取 `base_name = os.path.splitext(file_name)[0]`；原始响应写 `<save_dir>/<base>_raw_responses.csv`（`rate.py:117-118`）。
3. 加载/写入 run metadata（`load_run_metadata` / `write_task_run_metadata`，`rate.py:119-127, 182-190`）。
4. **标识符**：默认 `hash_identifier(passage, bits=64)` 得到 16 位 sha1 十六进制；同一文本去重，`df_proc["_gid"]` 存哈希（`rate.py:138-149`）。
5. 渲染 prompt（`self.template.render(...)`），每条 id 追加 `_batch{batch_idx}`（`rate.py:194-210`）。
6. `await get_all_responses(...)`（`rate.py:297-311`），返回的 DataFrame 列含 `Identifier`/`Response` 等。
7. 解析（`_parse`，多用 `safest_json`）→ 按 `Identifier` 聚合（`rate.py:316-374`）。
8. `save_dataframe_with_fallback(result, out_path, index=False, label="Rate")` 落盘 `<base>_cleaned.csv`（`rate.py:376-379`），返回 `result`。

### 1.5 数据流关键约定

- **原始响应**：`<base>_raw_responses.csv`（`rate.py:118`）。
- **清洗结果**：`<base>_cleaned.csv`（`rate.py:376`）。
- **多 run 明细**（仅 `n_runs > 1` 时）：`<base>_full_disaggregated.csv`，`index=["id","run"]`（`rate.py:361-371`）。
- **元数据**：`<base>_run_metadata.json`（`_run_utils.py:26-27`）。
- **属性定义**：`attributes.json`（+ 兼容旧名 `<base>_attrs.json`，`_attribute_utils.py:24-26`）。
- **断点续跑语义**：`reset_files=False` 时，engine 会读取 `_raw_responses.csv`，只补跑 `Successful != True` 的行（见 2.3）。

---

## 2. engine 层 `utils/openai_utils.py`

> 文件 `utils/openai_utils.py` 是核心引擎，约 6600+ 行。以下只记录 text 相关的行为契约。

### 2.1 `get_response`（`openai_utils.py:1826`）

签名（keyword-only 众多）：

```python
async def get_response(
    prompt: str, *,
    model: str = "gpt-5.4-mini",
    n: int = 1,
    max_output_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    temperature: float = 0.9,
    json_mode: bool = False,
    expected_schema: Optional[Dict[str, Any]] = None,
    tools / tool_choice / web_search / web_search_filters / search_context_size = "medium",
    reasoning_effort / reasoning_summary / service_tier / include,
    use_dummy: bool = False,
    base_url / verbose / images / image_detail / audio / pdfs / return_raw,
    request_phase_callback / logging_level / background_mode / background_poll_interval = 10.0,
    **kwargs,
)  # -> ([text, ...], duration) 或 (..., raw) 当 return_raw=True
```

（签名逐参见 `openai_utils.py:1826-1858`；返回结构见 docstring `:1947-1952`）

行为要点：

- **返回** `(texts: List[str], duration: float)`；`return_raw=True` 时再带原始 SDK 对象（`openai_utils.py:2370-2372`）。text 来自 `resp.output_text`（`openai_utils.py:2366`）。
- **dummy 模式**：`use_dummy=True` 返回 `[f"DUMMY {prompt}"] * n`，duration 0.0，不发网络请求（`openai_utils.py:1966-1970`）。
- **客户端池**：`_get_client(base_url)` 按 base_url 缓存 `AsyncOpenAI` client，并在并行度提高时重建 httpx 连接池（`openai_utils.py:178-227`）。
- **背景模式/轮询**：当 timeout 为 None 或要求 background 时，走 `responses.create(background=True)` + `responses.retrieve` 轮询（`openai_utils.py:2016-2097`）。轮询失败重试，连续 5 次失败才放弃（`openai_utils.py:2095-2096`）。
- **legacy system instruction**：模型名以 `gpt-3`/`gpt-4` 开头时，构造 `[{"role":"system"},{"role":"user"}]` 双消息（`openai_utils.py:370-374, 2224-2240`）；否则单条 user 消息。temperature 仅对 legacy 生效（`openai_utils.py:1796-1809`）。
- **音频**：`audio` 非空时走 `chat.completions`（`openai_utils.py:2102-2217`）；text 场景不涉及。
- **异常**：`asyncio.TimeoutError` / `RateLimitError` / `APIConnectionError` 直接上抛由上层重试（`openai_utils.py:2281-2296`）。

### 2.2 `_build_params`（`openai_utils.py:1678`）

把高层参数编成 Responses API 的 payload：

- 基础 `{"model", "input", "truncation": "auto"}`（`openai_utils.py:1763-1767`）。
- `json_mode=True` 时设 `text.format = {"type": "json_schema", "schema": ...}` 或 `{"type": "json_object"}`（`openai_utils.py:1773-1778`）。
- `web_search=True` 时追加 `{"type": "web_search", "search_context_size": ...}`，`filters`/`user_location` 由 `_normalise_web_search_filters`（`openai_utils.py:1426`）规整（`openai_utils.py:1780-1791`）。
- `reasoning` 仅在非 legacy 模型时装配 `effort`/`summary`（`openai_utils.py:1799-1805`）。
- `web_search` 自动注入 `include=["web_search_call.action.sources"]`（`openai_utils.py:1810-1821`）。**这是 web_search 与 Responses API 原生 tool 的唯一耦合点**——leximeter 用独立 SearchTool 替换时，把这里换成自己的搜索工具调用即可。
- `search_context_size` 别名归一：`"small"→"low"`、`"large"→"high"`（`_normalise_search_context_size`，`openai_utils.py:1541`）。

### 2.3 `get_all_responses`（`openai_utils.py:3600`）

签名（keyword-only，注意第一组是位置参数）：

```python
async def get_all_responses(
    prompts: List[str],
    identifiers: Optional[List[str]] = None,
    prompt_images / prompt_audio / prompt_pdfs / prompt_web_search_filters = None,  # 位置参数
    *,
    model: str = "gpt-5.4-mini",
    modality: Optional[str] = None,
    image_detail / n: int = 1 / max_output_tokens / temperature = 0.9,
    json_mode: bool = False,
    estimated_output_tokens_per_prompt: int = 500,   # ESTIMATED_OUTPUT_TOKENS_PER_PROMPT, 行233
    tools / tool_choice / web_search / web_search_filters / search_context_size = "medium",
    reasoning_effort / reasoning_summary / service_tier / include,
    dummy_responses / use_dummy: bool = False,
    response_fn / get_all_responses_fn,
    base_url / api_key / print_example_prompt: bool = True,
    save_path: str = "responses.csv",
    reset_files: bool = False,
    skip_tail_fails: bool = True,
    n_parallels: int = 650,
    ramp_up_seconds: float = 15.0,
    ramp_up_start_fraction: float = 0.2,
    max_retries: int = 3,
    timeout_factor: float = 2.5,
    max_timeout / dynamic_timeout: bool = True,
    timeout_burst_window: float = 60.0,
    timeout_burst_cooldown: float = 60.0,
    timeout_burst_max_restarts: int = 3,
    background_mode / background_poll_interval: float = 2.0,
    cancel_existing_batch: bool = False,
    use_batch: bool = False,
    batch_completion_window: str = "24h",
    batch_poll_interval: int = 10,
    batch_wait_for_completion: bool = False,
    max_batch_requests: int = 50_000,
    max_batch_file_bytes: int = 100 * 1024 * 1024,
    save_every_x_responses: int = 100,
    verbose: bool = True,
    quiet: bool = False,
    global_cooldown: int = 15,
    rate_limit_window: float = 30.0,
    connection_error_window: float = 30.0,
    token_sample_size: int = 20,
    status_report_interval: Optional[float] = 120.0,
    planning_rate_limit_buffer: float = 0.85,   # PLANNING_RATE_LIMIT_BUFFER, 行241
    logging_level: Union[str, int] = "warning",
    **get_response_kwargs,
) -> pd.DataFrame
```

（签名逐参见 `openai_utils.py:3600-3674`）

**返回 DataFrame 的列**（`openai_utils.py:4242-4253`）：

| 列 | 含义 |
|---|---|
| `Identifier` | 每条 prompt 的标识符 |
| `Response` | 响应文本（存储时经 `_ser` JSON 序列化，读取时 `_de` 反序列化，`openai_utils.py:2375-2392`；续跑加载 `openai_utils.py:4182`） |
| `Web Search Sources` | web 搜索来源（无则 `pd.NA`，`openai_utils.py:4187-4188`） |
| `Time Taken` | 秒 |
| `Input Tokens` / `Reasoning Tokens` / `Output Tokens` | token 统计 |
| `Reasoning Effort` | 推理档位 |
| `Successful` | `True/False` 布尔（续跑断点判据） |
| `Error Log` | 错误日志（JSON 序列化存储） |

`reasoning_summary` 非 None 时在 `Reasoning Tokens` 后插入 `Reasoning Summary` 列（`openai_utils.py:4254-4255`）。

**断点续跑语义**：

- `reset_files=False` 且 `save_path` 已存在：`pd.read_csv` 后 **`drop_duplicates(subset=["Identifier"], keep="last")`**（`openai_utils.py:4179-4180`）——同一 Identifier 重跑时保留最后一次。
- `Successful` 归一化解析：布尔或字符串 `"true"/"1"/"yes"/"y"/"completed"/"succeeded"/"success"` 都算完成（`openai_utils.py:4208-4220`）；只补跑未完成的 Identifier（`openai_utils.py:4291`）。
- `skip_tail_fails=True` 且所有请求 id 都在检查点内时：若未完成行数 `<= max(0, floor(total * 0.0001 + 0.5))`（约 0.01% 尾部），直接返回检查点不重试（`openai_utils.py:4292-4314`）。
- 大批量自动放大检查点频率：`>=10k` 行每 500、`>=20k` 每 1000、`>=50k` 每 2000 条存一次（`openai_utils.py:4318-4333`）。
- `>100_000` 行打印建议分批提示（`openai_utils.py:4345-4350`）。

**并发与节流（leximeter 引擎层的重点契约）**：

- 初始并发上限 `n_parallels`（默认 650）；`web_search` 或 media 输入时自动**减半**（docstring `openai_utils.py:3741-3747`）。
- **斜坡启动**：`ramp_up_start_fraction`（默认 0.2）× 目标并发开始，在 `ramp_up_seconds`（默认 15s）内线性爬升到满并发；斜坡期遇 rate-limit/连接错误则停在当前并发，交还给自适应缩放（docstring `openai_utils.py:3727-3732`）。
- **滚动窗口限流**：`rate_limit_window` / `connection_error_window`（均默认 30s）内统计错误，只在大批量错误或连续长串错误时降并发；降档后要等满一个窗口才能再降；成功会重置计数并允许回弹（docstring `openai_utils.py:3712-3725`）。
- **动态超时**：先不限时，成功后取**观测时长的 p90** 作为超时（`dynamic_timeout=True`），可被 `max_timeout` 封顶；超时请求取消并重试（docstring `openai_utils.py:3699-3704`）。
- **超时风暴保护性重启**：在 `timeout_burst_window`（60s）内超时数达到「约 1.25 × 当前并发数」时触发（docstring `openai_utils.py:3734-3737`）。
- **cost 估算**：`_report_cost` 用 `MODEL_PRICING`（`openai_utils.py:418`）按 `input/1e6*pricing.input + (output+reasoning)/1e6*pricing.output * service_tier_multiplier` 计算，打印 `Actual total cost: ...; average per row: ...; average per 1000 rows: ...`（`openai_utils.py:4260-4289`）。默认无 `max_output_tokens` 时按 `estimated_output_tokens_per_prompt=500` 估（`openai_utils.py:229-233`）。

**依赖注入**：

- `response_fn`：替换**每次单 prompt 调用**，只转发其接受的 kwargs，返回可以是 `[resp]`、`(responses, duration)` 或 `(responses, duration, raw)`（docstring `openai_utils.py:3761-3771`）。
- `get_all_responses_fn`：**整体替换**整个 engine，必须接受 `prompts` 与 `identifiers`，返回含 `Identifier` + `Response` 两列的 DataFrame；优先于 `response_fn` 与内部所有处理（`openai_utils.py:3817-3876`）。**这是 leximeter 换成自己的引擎时的接缝**。
- `use_dummy` / `dummy_responses`：`DummyResponseSpec`（`openai_utils.py:270`）可精确指定每条 id 的假响应文本/duration/token/错误（docstring `openai_utils.py:3773-3778`）。

**Batch API**：`use_batch=True` 走 OpenAI Batch（24h 窗口，`batch_completion_window="24h"`）；音频输入不支持 batch，自动回落（`openai_utils.py:4336-4341`）。

### 2.4 `get_embedding`（`openai_utils.py:2596`）

```python
async def get_embedding(
    text: str, *,
    model: str = "text-embedding-3-small",
    timeout: Optional[float] = None,
    use_dummy: bool = False,
    base_url: Optional[str] = None,
    return_raw: bool = False,
    logging_level: Optional[Union[str, int]] = None,
    **kwargs,
) -> Tuple[List[float], float]   # (embedding, duration)
```

- dummy 模式返回 `[float(len(text))]`（`openai_utils.py:2643-2645`）。
- 调用 `client_async.embeddings.create(model=model, input=text, ...)`，取 `raw.data[0].embedding`（`openai_utils.py:2656, 2681`）。

### 2.5 `get_all_embeddings`（`openai_utils.py:2688`）

```python
async def get_all_embeddings(
    texts: List[str],
    identifiers: Optional[List[str]] = None,
    *,
    model: str = "text-embedding-3-small",
    save_path: str = "embeddings.pkl",
    reset_file: bool = False,
    n_parallels: int = 150,
    timeout: float = 30.0,
    save_every_x: int = 5000,
    use_dummy: bool = False,
    embedding_fn / get_all_embeddings_fn / dummy_embeddings,
    base_url / verbose / logging_level = "warning",
    max_retries: int = 3,
    global_cooldown: int = 15,
    **get_embedding_kwargs,
) -> Dict[str, List[float]]
```

（签名见 `openai_utils.py:2688-2708`）

行为要点：

- 返回 `{identifier: embedding}`，`identifiers is None` 时用 `texts` 本身作 key（`openai_utils.py:2776-2777`）。
- **检查点是 pickle**：`save_path` 默认 `embeddings.pkl`；每 `save_every_x`（默认 5000）条写一次（docstring `openai_utils.py:2712-2716`）。
- 超时/rate-limit 指数退避重试（`max_retries=3`），遇 rate-limit 全局限流 `global_cooldown=15s`（docstring `openai_utils.py:2761-2762`）；并发随成功率自适应（docstring `openai_utils.py:2713-2714`）。
- 超大数据集 split-pickle 回退：`_find_split_pickle_parts` 按 `<stem>_<n>.pkl` 分片（`openai_utils.py:2435-2447`），分片尺寸在调用处确定（merge 用 100_000 / 10_000，见 4.8）。
- `embedding_fn` 替换单文本嵌入；`get_all_embeddings_fn` 整体替换并返回 dict（`openai_utils.py:2779-2843`）。

---

## 3. 共享任务基础设施

### 3.1 `tasks/_run_utils.py`（318 行，全文核读）

常量：
- `DEFAULT_IDENTIFIER_HASH_BITS = 64`、`LEGACY_IDENTIFIER_HASH_BITS = 32`、`ATTRIBUTE_WARNING_THRESHOLD = 12`（`_run_utils.py:15-17`）。

函数：

| 函数 | 行号 | 契约 |
|---|---|---|
| `run_metadata_path(save_dir, base_name)` | 26-27 | 返回 `<save_dir>/<base>_run_metadata.json` |
| `load_run_metadata(save_dir, base_name, *, reset_files)` | 30-44 | `reset_files=True` 删除并返回 `{}`；否则读 JSON，非 dict 返回 `{}` |
| `update_run_metadata(save_dir, base_name, **updates)` | 47-65 | 合并更新；`metadata_version` 置 1；写失败静默 |
| `hash_identifier(value, *, bits=64)` | 68-70 | `sha1(str(value)).hexdigest()[:bits//4]`。64 位→16 hex；32 位→8 hex |
| `_identifier_bits(identifier)` | 73-79 | 正则识别 16 位 / 8 位十六进制 id（带 `_`/`-` 后缀） |
| `infer_identifier_hash_bits_from_paths(paths)` | 82-99 | 读 `Identifier` 列前 5000 行推断是 64 还是 32 位（见旧 id 兼容） |
| `resolve_identifier_hash_bits(task_name, metadata, reset_files, checkpoint_paths)` | 102-125 | 续跑时沿用 metadata 或推断的 bits；若识别到 legacy 32 位则打印警告并沿用，保证 save_dir 可续跑 |
| `_coerce_json_obj(raw)` | 128-134 | `safe_json`；list 单元素再解包一层；str 再解析 |
| `_collect_attribute_keys(obj, lookup)` | 137-149 | 递归收集 JSON 里命中的属性 key（大小写不敏感归一到 lookup） |
| `_batch_index(identifier, row=None)` | 152-166 | 从 `Batch` 列或 `_batch(\d+)` 正则取 batch 序号 |
| `infer_attribute_batches_from_paths(paths, attribute_names)` | 169-211 | 从检查点反推旧的多批属性划分，要求 batch 连续、属性全集一致，否则返回 None |
| `_chunk_items(items, n)` | 214-219 | `n=None` 时整块；否则按 n 切片 |
| `resolve_attribute_batches(task_name, items, requested_n, metadata, reset_files, checkpoint_paths)` | 222-288 | 决定属性如何分批：优先 metadata 已存批次 → 其次推断 → 其次 `n_attributes_per_run` 切片；>12 个属性时打印建议设 `n_attributes_per_run=12` |
| `metadata_attribute_batches(batches)` | 291-294 | 转成名字列表用于元数据 |
| `write_task_run_metadata(save_dir, base_name, task_name, model, identifier_hash_bits, n_attributes_per_run, attribute_batches)` | 297-318 | 写 `{task, output_base_name, model, identifier_hash_bits, n_attributes_per_run, attribute_count, attribute_batches}` |

**标识符约定（全库通用）**：默认 64 位 sha1；run 1 复用裸 id（`_run1` 后缀仅当续跑已存在才保留），run ≥2 追加 `_run{n}`（`rate.py:255-264`）；属性分批追加 `_batch{batch_idx}`（`rate.py:210`）。

### 3.2 `tasks/_attribute_utils.py`（69 行，全文核读）

- `load_persisted_attributes(*, save_dir, incoming, reset_files, task_name, item_name="attributes", legacy_filename=None)`（`_attribute_utils.py:6-14`）。
- **优先级**：`save_dir/attributes.json` → `legacy_filename`（如 `ratings_attrs.json`）→ 都不存在则把 `incoming` 同时写入两处（`_attribute_utils.py:24-69`）。
- `reset_files=True` 会删除已存的属性文件（`_attribute_utils.py:28-34`）。
- **坑**：同一 `save_dir` 复跑但改动了 attributes 时，会打印「Found saved attributes ... differ; set reset_files=True」并**使用磁盘旧值**（`_attribute_utils.py:49-58`）——leximeter 需保留或明确废弃此「属性版本一致性」机制。

---

## 4. 逐任务参考

> 每个任务给出：Config 字段 / 默认值 / run() 流程 / prompt 模板 / 解析 / 输出列与返回 / 依赖 / 移植坑。

---

### 4.1 `rate` — 文本评分

- 文件：`tasks/rate.py`（383 行，全文核读）。API 门面 `api.py:187`。
- **Config**（`rate.py:40-53`）：

| 字段 | 类型/默认 |
|---|---|
| `attributes` | `Dict[str, str]`（必填，属性名→定义） |
| `save_dir` | `"ratings"` |
| `file_name` | `"ratings.csv"` |
| `model` | `"gpt-5.4-mini"` |
| `n_parallels` | `650` |
| `n_runs` | `1` |
| `use_dummy` | `False` |
| `rating_scale` | `Optional[str] = None` |
| `additional_instructions` | `Optional[str] = None` |
| `modality` | `"text"` |
| `n_attributes_per_run` | `Optional[int] = None` |
| `reasoning_effort` | `Optional[str] = None` |
| `search_context_size` | `"medium"` |

- `__post_init__`：`additional_instructions` 去首尾空白，空串转 None（`rate.py:55-58`）。
- **run**：`async def run(df, column_name, *, debug=False, reset_files=False, **kwargs)`（`rate.py:103-111`）。
  - 流程：`reset_index` → `_gid = hash_identifier(text)` 去重 → 属性分批 → 渲染 prompts（id 形如 `<hash>_batch{i}`）→ `get_all_responses(json_mode=True)` → 每 run 解析 → 聚合。
  - **解析** `_parse`：`safest_json` → `{attr: float}`，非法值置 None（`rate.py:88-98`）。
  - **聚合**：`full_df.groupby("id")[attrs].mean()`，跨 run 取均值（`rate.py:374`）；`n_runs>1` 时另存 `_full_disaggregated.csv`（`rate.py:361-371`）。
  - **返回**：原 df + 每属性一列 float 均值（`rate.py:377-383`）。
  - `kwargs.setdefault("web_search", modality == "web")`（`rate.py:241`）。
- **模板**：`ratings_prompt.jinja2`。
- **移植坑**：同文本哈希去重意味着**重复文本只发一次请求**，但解析时按 `base_ids` 回填所有行——若 leximeter 去掉哈希去重，需保证输出行数与输入行数一致。另外 `_gid` 是内部列，最后会 drop（`rate.py:378`）。

---

### 4.2 `classify` — 布尔分类

- 文件：`tasks/classify.py`。API 门面 `api.py:712`。
- **Config**（`classify.py:55-72`）：

| 字段 | 类型/默认 |
|---|---|
| `labels` | `Dict[str, str]`（必填） |
| `save_dir` | `"classifier"` |
| `file_name` | `"classify_responses.csv"` |
| `model` | `"gpt-5.4-mini"` |
| `n_parallels` | `650` |
| `n_runs` | `1` |
| `min_frequency` | `0.6` |
| `additional_instructions` | `Optional[str] = None` |
| `use_dummy` | `False` |
| `modality` | `"text"` |
| `n_attributes_per_run` | `Optional[int] = None` |
| `reasoning_effort` | `Optional[str] = None` |
| `differentiate` | `False` |
| `circle_first` | `Optional[bool] = None` |
| `search_context_size` | `"medium"` |

- **run**：`async def run(df, column_name=None, *, circle_column_name=None, square_column_name=None, reset_files=False, **kwargs)`（`classify.py:160-169`）。`differentiate=True` 时必须提供 `circle_column_name`/`square_column_name`，否则 `ValueError`（`classify.py:172-176`）。
- **解析** `_parse`：`safest_json` → `{label: Optional[bool]}`（`classify.py:124, 136`）。
- **聚合**：跨 run 对每 label 计算 True 比例 `prop >= min_frequency`（`classify.py:540-551`）。注意教程里 `n_runs=3, min_frequency=0.3` 表示至少 1 次 True 即算 True；`min_frequency=0.6` 要求多数通过。
- 落盘 `<base>_cleaned.csv`（`classify.py:566`）。
- **模板**：`classification_prompt.jinja2`。
- **移植坑**：`min_frequency` 聚合语义是「True 比例 >= 阈值」，与 filter 的「求和 >= 阈值×runs」等价但实现不同；结果列是布尔（含 NaN 表示未判定）。

---

### 4.3 `extract` — 结构化抽取

- 文件：`tasks/extract.py`。API 门面 `api.py:306`。
- **Config**（`extract.py:33-44`）：`attributes`（必填）、`save_dir="extraction"`、`file_name="extraction.csv"`、`model="gpt-5.4-mini"`、`n_parallels=650`、`n_runs=1`、`use_dummy=False`、`additional_instructions=None`、`modality="entity"`、`n_attributes_per_run=None`、`reasoning_effort=None`。
- **run**：`async def run(df, column_name, *, reset_files=False, types=None, **kwargs)`（`extract.py:135-143`）。`types` 可对每属性做类型提示（如 `"year of release": int`）。
- **解析** `_parse`：`safest_json` → `{entity: {attr: value}}`（`extract.py:71, 74`）。
- **关键**：extract 支持**一输入多实体**——当 LLM 返回多个 entity 时按 entity 展开成多行，`groupby(level=["id", "entity_name"])` 聚合（`extract.py:411`）；缺失值置 `<NA>`（教程：`returns "<NA>" if GPT doesn't know`）。
- 落盘 `<base>_cleaned.csv`（`extract.py:192, 422`）。
- **模板**：`extraction_prompt.jinja2`，占位符 `"unknown"`。
- **移植坑**：输出**行数与输入不保证一致**（实体展开）。API 打印提示建议用户自行 `drop_duplicates(subset=[...], keep="first")`（`extract.py:486`）。

---

### 4.4 `rank` — 两两比较排序（Elo/BT）

- 文件：`tasks/rank.py`（2213 行，全文核读）。API 门面 `api.py:1132`。
- **Config 字段/默认**（详见前次会话核读，`rank.py` 顶部 dataclass）：
  `attributes`(Dict或List, 必填)、`n_rounds=5`、`matches_per_round=5`、`power_matching=True`、`learning_rate=0.1`、`model="gpt-5.4-mini"`、`n_parallels=650`、`use_dummy=False`、`save_dir="~/Documents/runs"`、`file_name="rankings"`、`additional_instructions=None`、`circle_first=None`、`modality="text"`、`n_attributes_per_run=None`、`reasoning_effort=None`、`recursive=False`、`recursive_fraction=1/3`、`recursive_min_remaining=30`、`recursive_final_round_multiplier=3`、`recursive_cut_attr`、`recursive_cut_side="top"`、`recursive_rate_first_round=True`、`recursive_rewrite_func`、`recursive_rewrite_text_col="text"`、`recursive_keep_stage_columns=True`、`recursive_add_stage_suffix=True`、`initial_rating_pass=True`、`rate_kwargs`、`primer_scores`、`primer_scale=1.0`、`primer_center=True`。
- **run**：`async def run(df, column_name, *, id_column, reset_files, n_runs)`。传入 `n_runs` 时**打印警告并忽略**（rank 是多次两两比较，没有 n_runs 概念）。
- 内部常量：`_EXPLORE_FRAC=0.2`、`_CANDIDATE_NEIGHBORS=20`、`_HIGH_SE_FRAC=0.25`、`_MAX_ITER=1000`、`_TOL=1e-6`、`_SE_RIDGE=1e-5`、`_MAX_CANDIDATE_PAIRS_PER_ROUND=200_000`。
- **拟合**：Bradley-Terry 定点迭代（pseudo count = learning_rate，`rank.py:480-549`）；标准误用 Fisher 信息伪逆（`rank.py:551-612`）。
- **两两对标识符**：`{rnd}|{batch_idx}|{pair_idx}|{id_a}|{id_b}` 再 `hash_identifier`（`rank.py:2020-2021`）。
- **输出列**（每属性）：`{attr}`（z-score）、`{attr}_raw`（居中 BT 分）、`{attr}_se`。
- 检查点 `<base>_round{rnd}.csv`；最终 `<base>_final.csv`。递归模式输出到 `<base>_recursive/stage{stage_idx}/...`，最终 `recursive_final.csv`，含 `overall_rank`、`exit_stage` 列。
- **模板**：`rankings_prompt.jinja2`。
- **移植坑**：**pair 标识符格式**是 `|` 分隔再哈希，续跑或 debug 时必须能反解；BT 拟合是数值迭代，leximeter 需决定是否引入 scipy。`id_column` 是必填关键词参数——rank 不会自己造 id。

---

### 4.5 `merge` — 模糊合并 / 交叉对照

- 文件：`tasks/merge.py`（全文核读）。API 门面 `api.py:2045`。
- **Config 字段/默认**：`save_dir="merge"`、`file_name="merge_responses.csv"`、`model="gpt-5.4-nano"`、`n_parallels=650`、`n_runs=1`、`use_dummy=False`、`additional_instructions=None`、`use_embeddings=True`、`short_list_len=16`、`long_list_len=256`、`max_attempts=4`、`short_list_multiplier=0.5`、`auto_match_threshold=0.75`、`use_best_auto_match=False`、`candidate_scan_chunks=5`、`verbose=True`。
- **run**：`async def run(df_left, df_right, *, on, left_on, right_on, how="left", reset_files, **kwargs)`。校验 `how ∈ {left, right}`。
- **归一化**：HTML unescape + NFKD + 去非字母数字（`merge.py:72-78`）。
- **embeddings**：调 `get_all_embeddings`，检查点 `{save_dir}/short_embeddings.pkl` 与 `long_embeddings.pkl`；cosine 相似度自动匹配，`auto_match_threshold=0.75`（`merge.py:176-190`）。
- **聚类**：`_build_groups` 用 `scipy.cluster.vq.kmeans2` 把长表聚成簇（`merge.py:197-209`），再逐簇做 LLM 短表↔长表匹配。
- 落盘：`{save_dir}/{file_name}` = `merge_responses.csv`；进度 `merge_progress.csv`；组文件 `merge_groups_attempt{attempt}.json`。
- 返回 merged DataFrame，**`drop_duplicates(subset=[short_key])`**。
- **模板**：`merge_prompt.jinja2`。
- **移植坑**：`left_on == right_on` 时会产生临时列冲突（merge 内部用临时列处理）；`use_embeddings=True` 依赖 embedding 模型，leximeter 需接自己的 embedding 实现（见 2.4/2.5）。kmeans2 依赖 scipy，可选 `use_embeddings=False` 纯 LLM 路径。

---

### 4.6 `deduplicate` — 去重映射

- 文件：`tasks/deduplicate.py`。API 门面 `api.py:1922`。
- **Config**（`deduplicate.py:34-48`）：`save_dir="deduplicate"`、`file_name="deduplicate_responses.csv"`、`model="gpt-5.4-mini"`、`n_parallels=650`、`n_runs=3`、`use_dummy=False`、`additional_instructions=None`、`use_embeddings=True`、`group_size=500`、`modality="entity"`、`max_words_per_text=500`、`verbose=True`。
- **run**：`async def run(df, *, column_name, reset_files=False, nruns=None, **kwargs)`（`deduplicate.py:247-255`）。`n_runs = nruns or cfg.n_runs`（`deduplicate.py:259`）。
- 检查点：多 run 时 `{base}_run{i+1}{ext}`，单 run 直接用 `file_name`（`deduplicate.py:262-266`）。
- 解析：`safest_json` / `safe_json`（`deduplicate.py:206-222`），LLM 输出 `{representative: [raw, ...]}` 映射表。
- **输出列**（`deduplicate.py:347-393`）：
  - `mapped_{column_name}_ids`：每个原值→代表项 id。
  - `mapped_{column_name}`：单 run 时每行映射后的代表文本。
  - `mapped_{column_name}_final`：多 run 时按出现次数/投票选最终代表。
  - `mapped_{column_name}_run{i+1}`：多 run 时每 run 的映射列。
- **模板**：`deduplicate_prompt.jinja2`。
- **移植坑**：列名 `mapped_<col>_*` 是既定契约；`use_embeddings=True` 时与 merge 一样依赖 embedding 做预筛。

---

### 4.7 `filter` — 自然语言条件筛选

- 文件：`tasks/filter.py`。API 门面 `api.py:2186`。
- **Config**（`filter.py:18-34`）：`save_dir="filter"`、`file_name="filter_responses.csv"`、`model="gpt-5.4-nano"`、`n_parallels=650`、`entities_per_call=150`、`shuffle=True`、`random_seed=42`、`n_runs=1`、`threshold=0.5`、`additional_instructions=None`、`use_dummy=False`、`fix_json_with_llm=False`、`json_fix_timeout=60.0`。
- **run**：`async def run(df, column_name, *, reset_files=False, **kwargs)`（`filter.py:62-69`）。
- 流程：去重保序 → 按 `entities_per_call` 分组 → 每组一个 prompt 问「哪些满足条件」→ 解析 `{"entities meeting condition": [...]}` → 聚合。
- **聚合**：`df_proc[run_cols].sum(axis=1) / n_runs >= threshold`（`filter.py:165`）——满足比例 ≥ threshold 才保留。
- **模板**：`filter_prompt.jinja2`。
- **移植坑**：`shuffle=True` 用固定 `random_seed=42` 保证可复现；`fix_json_with_llm=True` 时解析失败会用 LLM 修 JSON（`safest_json` 的 `use_llm_fallback`）。

---

### 4.8 `deidentify` — PII 脱敏

- 文件：`tasks/deidentify.py`（全文核读）。API 门面 `api.py:1022`。
- **Config 字段/默认**：`model="gpt-5.4-mini"`、`n_parallels=650`、`save_dir="deidentify"`、`file_name="deidentified.csv"`、`use_dummy=False`、`max_words_per_call=7500`、`additional_instructions=None`、`reasoning_effort=None`、`n_passes=1`、`use_existing_mappings_only=False`。
- **run**：`async def run(df, column_name, *, grouping_column, mapping_column, reset_files, **kwargs)`。
- 流程：加 `group_id`；按 `max_words_per_call` 切块；每 pass 每轮写 `<base>_raw_responses_pass{p}_round{r}.csv`；identifier `{gid}_pass{pass_idx}_seg_{rnd}`（`deidentify.py:243`）。
- **应用映射**：正则 `(?<!\w){escaped}(?!\w)` + `IGNORECASE`（`deidentify.py:303-307`），把 real forms 替换成 casted forms。
- **输出列**：`mapping`（dict）、`deidentified_text`。落盘 `<base>_cleaned.csv`。
- **模板**：`deidentification_prompt.jinja2`。
- **移植坑**：**正则词边界替换**对含空格/标点的 real forms 要小心；`grouping_column` 保证同一人的代称跨记录一致，`mapping_column` 支持 warm-start 已有映射。

---

### 4.9 `codify` — 段落编码（定性标注）

- 文件：`tasks/codify.py`（1029 行，全文核读）。API 门面 `api.py:1325`。
- **Config 字段/默认**：`save_dir`(必填)、`file_name="coding_results.csv"`、`model="gpt-5.4-mini"`、`n_parallels=650`、`max_words_per_call=1000`、`max_categories_per_call=8`、`debug_print=False`、`use_dummy=False`、`reasoning_effort=None`、`modality="text"`、`n_rounds=2`、`completion_classifier_instructions=None`。
- **run**：`async def run(df, column_name, *, categories, additional_instructions, reset_files, **kwargs)`。`categories=None` 时进入 **dynamic_mode**（从文本自己发现类别）。
- identifier：`row{row_idx}_iter{iteration}_chunk{chunk_idx}_batch{batch_idx}`。
- **输出**：dynamic → `coded_passages` 列；否则每类别一列（snippet 列表）。
- **落盘 `{save_dir}/coded_passages.csv`**（注意：**不是 `file_name`**，`file_name` 只控制原始响应文件名）。
- **补全循环**：`_completion_loop` 用 Classify 检查未覆盖信号，校验文件 `completion_checks/{stem}_completion_iter{iteration}.csv`。
- **模板**：`codify_prompt.jinja2`。
- **移植坑**：**输出文件名与 `file_name` 不同**，重构时别沿用「file_name 决定最终输出」的惯性；snippet 用「开头 8 词 + 结尾 8 词」定位，依赖原文 verbatim 匹配（见模板）。

---

### 4.10 `compare` — 成对相似/差异

- 文件：`tasks/compare.py`。API 门面 `api.py:1573`。
- **Config**（`compare.py:31-42`）：`save_dir="comparison"`、`file_name="comparison_responses.csv"`、`model="gpt-5.4-mini"`、`n_parallels=650`、`n_runs=1`、`use_dummy=False`、`differentiate=True`、`additional_instructions=None`、`modality="text"`、`reasoning_effort=None`、`circle_first=None`。
- **run**：`async def run(df, circle_column_name, square_column_name, *, reset_files=False, **kwargs)`（`compare.py:75-83`）。circle/square 任一为 NaN 的行跳过并打印计数（`compare.py:84-90`）。
- **解析** `_parse`：`safest_json` → `Dict[str, str]`（term → 证据/理由）（`compare.py:69-70`）。
- **模板**：`comparison_prompt.jinja2`。
- **移植坑**：`differentiate=True` 时输出是「差异」术语，`False` 时是「相似」术语；术语禁止下划线/连字符（模板约束）。

---

### 4.11 `bucket` — 桶化 / 分类体系

- 文件：`tasks/bucket.py`（424 行，全文核读）。API 门面 `api.py:1679`。
- **Config 字段/默认**：`bucket_count=10`、`save_dir="buckets"`、`file_name="bucket_definitions.csv"`、`model="gpt-5.4-mini"`、`n_parallels=650`、`use_dummy=False`、`additional_instructions=None`、`differentiate=False`、`n_terms_per_prompt=250`、`repeat_bucketing=5`、`repeat_voting=25`、`next_round_frac=0.25`、`top_k_per_round=1`、`raw_term_definitions=True`、`reasoning_effort=None`。
- **run**：`async def run(df, column_name, *, reset_files, **kwargs)`。
- 状态文件 `bucket_state.json`；terms 签名用 sha1。
- 三阶段：
  1. **生成**：identifier `gen|{rep}|{ci}`，落盘 `bucket_generation.csv`；
  2. **归约**：`vote|reduce{round_idx}|{rep}|{ci}`，落盘 `vote_reduce{round_idx}.csv`；
  3. **终选**：`vote_final{loop_idx}|...`，落盘 `vote_final{loop_idx}.csv`。
- 最终输出 `bucket_definitions.csv`，列 `bucket` + `definition`。
- 缓存检查：若缓存已含 `{"bucket","definition"}.issubset(cached.columns)` 直接返回缓存。
- **模板**：`bucket_prompt.jinja2`（`voting` 分支）。
- **移植坑**：多轮投票（repeat_bucketing × repeat_voting）是串行逻辑，成本高；`differentiate=True` 时桶名必须带「square/circle entries」方向语义（模板约束）。

---

### 4.12 `seed` — 种子实体生成（代表性/多样性）

- 文件：`tasks/seed.py`（全文核读）。API 门面 `api.py:423`。
- **Config 字段/默认**：`instructions`(必填)、`save_dir="~/Documents/runs"`、`file_name="seed_entities.csv"`、`model="gpt-5.5"`、`n_parallels=650`、`num_entities=1000`、`entities_per_generation=50`、`entity_batch_frac=0.25`、`existing_entities_cap=100`、`use_dummy=False`、`deduplicate=False`、`deduplicate_sample_seed=42`、`reasoning_effort=None`。
- **run**：`async def run(existing_entities, reset_files, **response_kwargs)`。`existing_entities` 为已生成的实体 DataFrame（用于「看已有集合不重复」）。
- 输出列：`entity`、`entity_id`（`f"entity-{idx:05d}"`）、`source_batch`、`source_identifier`。
- identifier：`seed|{batch_index}|{call_index}`（`seed.py:370`）。
- **归一化**：`_normalize_entity` 折叠空白 + 小写（`seed.py:459-461`）。
- 落盘 `{save_dir}/{file_name}`（`_finalize_entities`，`seed.py:299-313`）。
- **模板**：`seed.jinja2`。
- **移植坑**：`entity_batch_frac` 控制「一批生成量占总量比例」的分批策略；`existing_entities_cap` 限制 prompt 里能看到多少已生成实体。**batch 大小是自动计算不是直接指定**。

---

### 4.13 `paraphrase` — 改写

- 文件：`tasks/paraphrase.py`（637 行，全文核读）。API 门面 `api.py:1445`。
- **Config 字段/默认**：`instructions`(必填)、`revised_column_name=None`、`n_rounds=1`、`n_runs=1`、`later_round_run_multiplier=5`、`use_modified_source=False`、`validation_attribute=None`、`save_dir="paraphrase"`、`file_name="paraphrase_responses.csv"`、`model="gpt-5.4-mini"`、`json_mode=False`、`web_search=None`、`modality="text"`、`search_context_size="medium"`、`n_parallels=650`、`use_dummy=False`、`reasoning_effort=None`。
- **run**：`async def run(df, column_name, *, reset_files, **kwargs)`。
- identifier：`row_{idx}_rev{j}`。
- **输出列**：`{base_col}`（单 run）或 `{base_col}_{i}`（多 run），另加 `{base_col}_approved` / `{col}_approved` 校验标记。
- **递归校验**：`_recursive_validate` 用 Classify 判 `instructions_followed`；候选 identifier `row_{row}_rev{rev}_round{r}_cand{c}`；落盘 `{stem}_round{r}.csv`。最终写 `{stem}_cleaned.csv`。
- **模板**：`paraphrase_prompt.jinja2`。
- **移植坑**：`json_mode=False` 时输出是纯文本；`n_rounds>1` 的「改→判→再改」闭环依赖 classify，移植时两者都要有。

---

### 4.14 `whatever` — 任意 prompt 管道

- 文件：`tasks/whatever.py`（375 行，全文核读）。API 门面 `api.py:2461`。
- **Config 字段/默认**：`save_dir="whatever"`、`file_name="custom_prompt_responses.csv"`、`model="gpt-5.4-mini"`、`json_mode=False`、`web_search=None`、`web_search_filters=None`、`search_context_size="medium"`、`n_parallels=650`、`reasoning_effort=None`、`reasoning_summary=None`。
- **run**：`async def run(prompts, *, df, identifiers, column_name, identifier_column, image_column, audio_column, prompt_images, prompt_audio, web_search_filters, reset_files, return_original_columns=True, drop_prompts=True, parse_json=None, **kwargs)`。
- `_generate_identifiers`：prompt 的 64 位 hash；重复时后缀 `-{idx}`。
- `extract_json(df, text_column="Response", target_column="Response JSON")` 静态方法：把 `Response` 列解析成 JSON 并放入 `Response JSON` 列。
- 提供 `df` 时按 `_gid`/`Identifier` 合并回原 df。
- **每 prompt 的 web_search_filters** 可从 df 列值取（`allowed_domains` 会做类型强转）。
- 无 prompt 模板（直接用用户文本）。
- **移植坑**：`Response JSON` 是 `extract_json` 的固定列名；`drop_prompts=True` 默认删掉 prompt 列。

---

### 4.15 `view` — 查看器（`api.py:2622`）

- `view()` 委托给 `passage_viewer`（`api.py:2622-2676`），是**交互式 UI**（列展示 + 评分/coding 高亮）。
- text 场景只是 HTML 表格渲染，不涉及 LLM。
- **移植**：leximeter 若不做 UI，可以直接不移植；本档不做 UI 细节（非核心契约）。

---

### 4.16 `load` — 文件夹加载（`utils/file_utils.py:226`）

- 签名：`load(folder_path, extensions, *, tag_dict, save_name="gabriel_aggregated_content.csv", save_dir, reset_files, modality)`（`file_utils.py:226`）。
- 聚合文件夹内文件为 DataFrame，列 `name / path / layer_N / tag / text`。
- 常量：`TEXTUAL_MODALITIES={text, entity, web}`、`PATH_MODALITIES={image, audio, pdf}`，各有扩展名集合（`file_utils.py` 顶部常量，此处未深究具体行号）。
- text 场景：把文件内容读成 `text` 列。

---

### 4.17 `discover` — 特征发现（两 class 差异）

- 文件：`tasks/discover.py`（517 行，全文核读）。API 门面 `api.py:1779`，返回 `Dict[str, DataFrame]`。
- **Config 字段/默认**：`save_dir="discover"`、`model="gpt-5.4-mini"`、`n_parallels=650`、`n_runs=1`、`min_frequency=0.6`、`bucket_count=10`、`additional_instructions=None`、`differentiate=True`、`max_words_per_call=1000`、`max_categories_per_call=8`、`use_dummy=False`、`modality="text"`、`n_terms_per_prompt=250`、`repeat_bucketing=5`、`repeat_voting=25`、`next_round_frac=0.25`、`top_k_per_round=1`、`raw_term_definitions=True`、`reasoning_effort=None`。
- **run**：`async def run(df, *, column_name, circle_column_name, square_column_name, reset_files, **kwargs)`。
- 单列 → Codify dynamic；双列 → Compare。
- 桶化用 Bucket；分类用 Classify，label = 实际 + 反转。
- `build_combined_metadata`（`discover.py:324-338`）：对每个 label 建 `{lab}_actual` 与 `{lab}_inverted` 两列，`swap_cs`（`discover.py:318-322`）交换 text 里的 "circle"/"square" 词。
- `summary_df` 列：`label, actual_true, inverted_true, total, actual_pct, inverted_pct, net_pct`。
- 结果 dict 键：`candidates, buckets, classification, bucket_df, summary, compare, codify`。
- 持久化：`discover_results_snapshot.json` 与 `discover_results_export.zip`。
- **移植坑**：**circle/square 互换 + differentiate 方向处理**是核心正确性问题（教程里还要手动把 `circle entries` 替换成 `winning arguments` 再翻转负 net_pct 行，见 notebook code cell 181-182）；**`df is None` 会直接抛错**（无最终落盘文件可续跑，`api.py:1779` 起的 discover 门面）。

---

### 4.18 `ideate` — 科研创意生成 + 过滤

- 文件：`tasks/ideate.py`（827 行，全文核读）。API 门面 `api.py:840`。
- **Config 字段/默认**：`save_dir="~/Documents/runs"`、`file_name="ideation.csv"`、`model="gpt-5.4-mini"`、`ranking_model=None`、`n_parallels=650`、`n_ideas=1000`、`evaluation_mode="recursive_rank"`、`scientific_theory=True`、`attributes=None`、`rank_attribute=None`、`recursive_fraction=1/3`、`recursive_min_remaining=30`、`recursive_final_round_multiplier=3`、`recursive_cut_side="top"`、`recursive_rate_first_round=True`、`additional_instructions=None`、`use_dummy=False`、`web_search=False`、`reasoning_effort=None`、`use_seed_entities=True` + `seed_*` 参数 + `seed_deduplicate=True`、`deduplicate_ideas=True`。
- 默认 attributes：`"major new contribution to literature"` 或 `"overall idea quality"`。
- **run**：`async def run(topic, *, ...)`。缓存到 `{base}_final.csv`。
- 生成 identifier `idea-{idx:05d}`；加 `idea_id, topic, report_text, seed_text` 列。
- `_parse_reports` 提取 section：`title / in_a_nutshell / in_one_paragraph / illustrative_examples / testable_predictions / full_thinking / summary_preview / report_preview`。
- `_deduplicate_ideas` 用 Deduplicate 按 `report_text` 折叠，产出 `mapped_report_text`；`_sort_results` 加 `{attr_key}_rank` 列。
- **模板**：`ideation_prompt.jinja2`。
- **移植坑**：**评估链路是递归 rank（evaluation_mode="recursive_rank"）**——ideate 依赖 rank/rate/deduplicate 三个任务，移植顺序要靠后。

---

### 4.19 `poll` — 合成民意调查

- 文件：`tasks/poll.py`（842 行，全文核读）。API 门面 `api.py:540`。
- **Config 字段/默认**：`population_description`(必填)、`questions`(必填)、`save_dir="~/Documents/runs"`、`file_name="poll_results.csv"`、`seed_file_name="poll_seeds.csv"`、`persona_file_name="poll_personas.csv"`、`seed_model="gpt-5.5"`、`persona_model="gpt-5.5"`、`poll_model="gpt-5.5"`、`n_parallels=650`、`num_personas=1000`、`entities_per_generation=50`、`entity_batch_frac=0.25`、`existing_entities_cap=100`、`deduplicate=False`、`deduplicate_sample_seed=42`、`n_questions_per_run=8`、`seed_additional_instructions`、`additional_instructions`、`web_search=False`、`reasoning_effort`、`use_dummy=False`、`seed_column_name="seed"`、`persona_column_name="persona"`。
- **run**：`async def run(df, column_name, *, reset_files, response_fn, get_all_responses_fn, embedding_fn, get_all_embeddings_fn, **response_kwargs)`。
- 若 df 已有 `persona` 列则跳过 seed/persona 生成（`poll.py:841-842` `_has_existing_personas`）。
- 三阶段：
  1. **seed**（经 Seed）；
  2. **persona**：identifier = seed 文本的 hash；
  3. **answers**：`{ident}_batch{batch_idx}`，`json_mode=True`，落盘 `poll_answers_raw_responses.csv`。
- `_parse_answers` 处理分布型答案 `{options: [...]}`，用 `_stable_uniform`（sha1）采样。
- **输出**：每问题一列。
- 落盘 `seed_file_name`、`persona_file_name`、`file_name`。
- **模板**：`persona_prompt.jinja2`、`poll_prompt.jinja2`、`seed.jinja2`。
- **移植坑**：分布型答案的**确定性采样**（sha1）保证可复现；`n_questions_per_run` 控制每 prompt 问多少题。

---

### 4.20 `debias` — 测量去偏

- 文件：`tasks/debias.py`（1425 行，全文核读）。API 门面 `api.py:2297`。
- **Config 字段/默认**：`mode`（"rate"/"classify"/"extract"/"rank"）、`measurement_attribute`、`removal_attribute`、`attributes`、`signal_dictionary`、`removal_method`（"codify"/"paraphrase"）、`remaining_signal=True`、`remaining_signal_attribute`、`remaining_signal_description`、`save_dir="~/Documents/runs"`、`run_name`、`strip_percentages`、`categories_to_strip`、`template_path`、`model="gpt-5.4-mini"`、`n_parallels=650`、`measurement_kwargs`、`removal_kwargs`、`robust_regression=True`、`random_seed=12345`、`verbose=True`。
- **run**：`async def run(df, column_name, *, reset_files)` → 返回 `DebiasResult(results_df, metadata, regression)`；`regression` 是 `Dict[str, DebiasRegressionResult]`。
- **列命名（最大的坑）**：
  - `_attach_measurement`（`debias.py:689-708`）：原始变体保留 `{attr}` 名，变体用 `{attr} ({display_name})`；label_suffix `"stripped"` 或 `"stripped {pct}%"`。
  - `_prepare_codify_variants`：key `stripped_{pct:03d}pct`；单 100% 时新列 `{column_name} (stripped)`，否则 `{column_name} ({display})`。
  - 两步去偏产出 `{measurement_attr} (debiased)` 列；差异列 `{measurement_attr} (raw - stripped)`。
- `_strip_passages` 用种子 RNG `f"{random_seed}:{row_idx}:{pct_strip}:{n_total}"` 把 snippet 替换成空格（确定性）。
- 回归用 `fit_ols`（`utils/plot_utils`），`robust_regression=True` 时用稳健回归。
- 落盘 `debias_results.csv`、`debias_metadata.json`。
- **移植坑**：**变体列名**高度模板化（`{attr} (stripped)` / `{attr} ({display})` / `(debiased)` / `(raw - stripped)`），leximeter 若改命名会破坏下游；`mode="rank"` 时内部依赖 rank 的 `_raw`/`_se` 列。

---

## 5. 提示词模板清单

> 目录：`openai/GABRIEL/src/gabriel/prompts/`（glob 共 **18** 个文件）。加载器见 `utils/jinja.py:25-31`（`Environment(loader=FileSystemLoader(prompts_dir))`，过滤器 `shuffled`/`shuffled_dict`）。`core/prompt_template.py` 的 `resolve_template` 校验自定义模板的变量集与参考模板一致。

| 模板 | 用途（任务） | 主要变量 | 备注 |
|---|---|---|---|
| `snippets.jinja2` | 公共宏：`single_entry(modality, text)`、`pair_entries(modality, entry_square, entry_circle, circle_first)` | modality / text / entry_square / entry_circle / circle_first | 被多数模板 import |
| `ratings_prompt.jinja2` | rate | text, attributes, scale, additional_instructions, modality | 输出 `{attr: 0-100}`；`attributes \| shuffled_dict` |
| `rankings_prompt.jinja2` | rank | entry_square, entry_circle, circle_first, attributes, additional_instructions, modality | 输出 `"square"/"circle"/"draw"/"insufficient signal"` |
| `classification_prompt.jinja2` | classify | differentiate, entry_square, entry_circle, circle_first, text, attributes, additional_instructions, modality | 单/双条目两种模式；输出 `{label: True/False}` |
| `extraction_prompt.jinja2` | extract | text, attributes, additional_instructions, modality | 输出 `{entity: {attr: value}}`，占位 `"unknown"` |
| `comparison_prompt.jinja2` | compare | modality, entry_square, entry_circle, circle_first, differentiate, additional_instructions | 差异 vs 相似两种模式 |
| `merge_prompt.jinja2` | merge | short_list, long_list, additional_instructions | 短表每项找长表最佳匹配，可 `"no certain match"` |
| `deduplicate_prompt.jinja2` | deduplicate | raw_terms, additional_instructions | 输出 `{representative: [raw,...]}` |
| `filter_prompt.jinja2` | filter | condition, entities, additional_instructions | 输出 `{"entities meeting condition": [...]}` |
| `deidentification_prompt.jinja2` | deidentify | text, current_map, additional_instructions | 输出 mapping dict（real forms + casted form） |
| `codify_prompt.jinja2` | codify | text/entry_*, categories, additional_instructions, modality | 输出 `{category: [{beginning excerpt, ending excerpt}]}`；8 词首尾锚 |
| `bucket_prompt.jinja2` | bucket | terms, bucket_count, differentiate, voting, bucket_candidates, selected_buckets, additional_instructions | 生成 or 投票两分支 |
| `seed.jinja2` | seed | instructions, entities_per_generation, existing_entities | 输出 `{"entity 1": "...", ...}` |
| `persona_prompt.jinja2` | poll（persona 阶段） | seed, population_description | ~750 词人设传记 |
| `poll_prompt.jinja2` | poll（答题阶段） | persona, questions, additional_instructions | 输出 `{question: [[answer, prob], ...]}` |
| `ideation_prompt.jinja2` | ideate | topic, scientific_theory, seed, additional_instructions | 固定标题段落式 report |
| `regional_analysis_prompt.jinja2` | （whatever + web_search 的自制模板，教程 code cell 266） | region, topic, additional_instructions | 3000+ 词区域报告 |
| `paraphrase_prompt.jinja2` | paraphrase | modality, text, instructions, json_mode | 纯改写，无结构化输出 |

**要点**：
- 所有「attributes 列表」渲染前在 `PromptTemplate.render()` 里从 list 转 dict（`core/prompt_template.py`，此处未深究具体行号）。
- 模板大量使用 `| shuffled_dict` / `| shuffled`（无种子时用全局 random），目的是**打乱属性顺序避免位置偏差**；leximeter 要可复现需注入种子。
- 变量名 `circle`/`square` 在 `utils/prompt_utils.py:swap_circle_square` 中按大小写保留地互换（`prompt_utils.py` 42 行）。

---

## 6. 文本工具 utils

### 6.1 `utils/parsing.py`（311 行，全文核读）

- `_parse_json`（17-118）：严格 JSON + 清洗启发式——fence 代码块 `_JSON_FENCE_RE`、引号剥离、`json.loads`/`ast.literal_eval`、注释剥离、花括号/方括号提取、单引号列表解包、首尾括号定位。
- `safe_json(raw)`（121-131）：解析失败返回 `{}`（永不抛）。
- `parse_json_with_status(raw)`（134-140）：返回 `(parsed, bool)`。
- `safest_json(raw, *, use_llm_fallback=..., timeout=60.0, ...)`（143-187）：`async`；可选 LLM 兜底解析（`JSON_LLM_MODEL` env，默认 `"gpt-5.4-mini"`）；无 fallback 时失败返回 None。
- `clean_json_df(df, ..., id_col, ...)`（190-311）：产出 `<col>_cleaned` 列；`id_col` 必须唯一；identifier 为 `{id_val}__{col}`。
- **移植坑**：`safest_json` 是 async 且可能触发 LLM 调用（`use_llm_fallback=True` 时），所有任务解析都走它；leximeter 要保留「解析失败不炸」的契约。

### 6.2 `utils/file_utils.py`（638 行，全文核读）

- `save_dataframe_with_fallback(df, path, *, index=False, chunk_size=100_000, fallback_chunk_sizes=None, label=None)`（124-223）：
  - 先尝试 `df.to_csv` 整文件；失败则按 `<stem>_{i+1}{ext}` 分片写（分片尺寸逐级缩小，`fallback_chunk_sizes` 默认链），返回已写文件路径列表，并清理旧分片。
  - **这是所有任务落盘的统一入口**；`_load_cached_dataframe` 因此要能读回分片（见 1.2）。
- `load(folder_path, extensions, *, tag_dict, save_name, save_dir, reset_files, modality)`（226-414）：见 4.16。

### 6.3 `utils/word_matching.py`（158 行，全文核读）

- `normalize_whitespace`、`normalize_text_generous`（NFKC + 空白折叠）、`normalize_text_aggressive`（编码修复 + NFKD + unicode 修复 + 缩略词正则 + 空白折叠）、`letters_only`（小写 a-z）。
- `robust_find_improved(text, snippet)`：letters-only 匹配 + 回退，返回原文 excerpt 或 None。
- `strict_find(text, snippet) -> bool`：3 种策略。
- 被 codify/debias 等用于 snippet 定位。

### 6.4 `utils/logging.py`（86 行，全文核读）

- `LOG_LEVELS` 含 `"silent"`；`set_log_level`；`get_logger`。
- `announce_prompt_rendering(task, count)`（73-82）：打印 `[{task}] Rendering {count} prompts…` 并 flush。任务渲染前调用（如 `rate.py:197-200`）。

### 6.5 其它

- `utils/prompt_utils.py`（42 行）：`swap_circle_square(text)` 互换 circle/square 词（大小写保留、mask Jinja 变量名 `entry_circle`/`entry_square`），discover 用。
- `utils/jinja.py`（31 行）：`shuffled` / `shuffled_dict` / `get_env`，见第 5 节。
- `utils/plot_utils.py`：`fit_ols`（debias 回归用）、`regression_plot`/`bar_plot`/`box_plot`/`line_plot`（教程可视化用，非核心契约，此处未深究）。

---

## 附：移植时整体要注意的坑（汇总）

1. **`_split_cfg_and_response_kwargs` 删除后的参数通道**：engine 参数（`image_detail`、`max_output_tokens`、`web_search` 等）原先是靠「Config 字段 ∪ engine 签名白名单」自动路由的。leximeter 需要显式设计（如 `engine_kwargs`）。
2. **Identifier 是续跑命根**：`hash_identifier` 64 位 sha1 + `_batch{}/_run{}` 后缀是检查点 dedup 的 key。任何标识符格式改动都会破坏断点续跑；`_run_utils` 里专门有 legacy 32 位兼容逻辑。
3. **`save_dataframe_with_fallback` + `_load_cached_dataframe` 必须成对移植**：落盘可能分片，读取必须支持分片回读。
4. **`_attribute_utils.load_persisted_attributes` 的「磁盘优先」语义**：同一 save_dir 复跑会静默使用旧 attributes，容易让用户困惑——leximeter 应决定保留还是改为「显式覆盖」。
5. **`safest_json` async + LLM 兜底**：解析层不要简化为同步 `json.loads`。
6. **`use_dummy`/`response_fn`/`get_all_responses_fn`/`get_all_embeddings_fn` 注入点**：是离线测试与厂商替换的关键接缝，leximeter 引擎层务必保留等价物。
7. **tasks/__init__.py 里有死引用**：`EloRater`→`.elo`、`Regional`→`.regional`、`CountyCounter`→`.county_counter`（`tasks/__init__.py:11-12, 23-25`）指向不存在的模块——只在被 `getattr` 时才触发 ImportError，leximeter 不要照抄。
8. **`core/pipeline.py` 只是占位**（18 行），没有实际管道编排，不要误以为存在框架性 pipeline。

---

*文档生成日期：2026-08-14。所有行号基于对 `openai/GABRIEL/src/gabriel/` 源码的核读。*
