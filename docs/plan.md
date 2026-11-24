# Leximeter — 重构计划

> fork 自 [openai/GABRIEL](https://github.com/openai/GABRIEL)。目标：**文本专用、厂商中立、易维护**。

## 一、定位

GABRIEL 把"请 LLM 打分/分类/抽取"包装成一套测量仪器。它功能扎实，但存在两个结构性问题：

1. **单函数/单类职责耦合严重** —— `utils/openai_utils.py` 约 6900 行，把并发调度、限流退避、超时重试、成本估算、日志、检查点揉在一个函数体里；`rank.py` 约 2200 行。
2. **完全依赖 OpenAI Responses API** —— `LLMClient` 抽象形同虚设，任务层直接 import 全局函数，请求参数硬编码为 Responses API 字段。

Leximeter 保留 GABRIEL 的全部功能（文本部分），重做内部结构。

## 二、锁定共识

| # | 决策 | 结论 |
|---|---|---|
| 1 | 项目名 / 包名 | `leximeter`（`import leximeter`） |
| 2 | Provider 接口 | `LLMClient.complete(prompt, *, model, json_mode, **kw) -> str`；预留 `estimate_tokens()` / `estimate_cost()` 抽象方法（v1 不实现） |
| 3 | Provider 实现 | 按协议：`ResponsesClient`、`ChatCompletionsClient`（openai SDK）、`DummyClient` |
| 4 | 选择机制 | env 默认（`LEXIMETER_PROTOCOL/BASE_URL/API_KEY/MODEL`）+ 每调用 `client=` 覆盖 |
| 5 | 联网搜索 | 独立 `SearchTool`；v1 实现 `TavilySearchTool`（`ExaSearchTool` 占位）；Responses 用原生；未配置则 warning + 跳过 |
| 6 | 嵌入 | 独立 `EmbeddingClient`；默认 `LocalBgeClient`（`bge-m3` 本地 CPU）；`[embeddings]` 可选依赖、懒加载 |
| 7 | 去多模态 | 删 image/audio/pdf/web modality 及 5 个 utils；view/load 只留文本 |
| 8 | API | 函数名 + 心智模型保留；去 `**cfg_kwargs` / `inspect.signature` 魔法，参数显式化 |
| 9 | 模块结构 | `client/` + `engine/` + `tasks/` + `api.py`；`openai_utils.py` 一劈为二 |
| 10 | 基类 | `Task`（模板方法）+ `AttributeTask`（n_runs/均值聚合）；子类只实现 `render_prompt`/`parse_response`/`build_output` |
| 11 | 复杂任务拆分 | rank/debias/ideate/poll/discover 各拆子包；codify 保持单文件 |
| 12 | 推进方式 | 方案 B：重建骨架 + 逐个移植，`DummyClient` 边移植边测 |
| 13 | 脚手架 | `leximeter/`，新 git init，pytest |
| 14 | 协议客户端 | ChatCompletions 用 openai SDK |

## 三、目标结构

```
leximeter/
├── src/leximeter/
│   ├── __init__.py        # 公开 API 导出
│   ├── api.py             # 瘦身门面
│   ├── client/            # 厂商协议层
│   │   ├── base.py        #   LLMClient
│   │   ├── responses.py   #   ResponsesClient
│   │   ├── chat.py        #   ChatCompletionsClient
│   │   ├── dummy.py       #   DummyClient
│   │   ├── embedding.py   #   EmbeddingClient + LocalBgeClient + OpenAI + Dummy
│   │   ├── search.py      #   SearchTool + TavilySearchTool（Exa 占位）
│   │   └── resolve.py     #   env/config → client / embedding_client / search_tool
│   ├── engine/            # 通用编排层（provider 无关）
│   │   ├── batch.py       #   并发调度、限流退避、断点续跑
│   │   ├── retry.py       #   重试/退避/超时
│   │   ├── checkpoint.py  #   检查点读写
│   │   └── cost.py        #   成本进度（调 client.estimate_*）
│   ├── tasks/             # 业务任务（Config + Task 模式）
│   │   ├── base.py        #   Task + AttributeTask（模板方法）
│   │   ├── _common.py     #   属性持久化、标识符哈希、元数据
│   │   ├── rate.py  classify.py  extract.py  merge.py  ...（简单任务单文件）
│   │   └── rank/ debias/ ideate/ poll/ discover/（复杂任务子包）
│   ├── prompts/           # Jinja2 模板（只留文本）
│   └── utils/             # parsing / file_utils / logging / word_matching
└── tests/
```

## 四、里程碑

- **M0 脚手架**（当前）：目录、git、pyproject、`client` 抽象、`Task` 基类骨架。
- **M1 client 层**：`ResponsesClient` + `ChatCompletionsClient` + `resolve`（协议选择）。
- **M2 engine 层**：`batch`（并发/重试/限流/检查点），provider 无关。
- **M3 首个任务**：`rate`（参考实现）+ `AttributeTask` 完整化。
- **M4 简单任务**：classify/extract/merge/deduplicate/filter/deidentify/seed/paraphrase/codify/compare/bucket/whatever + view/load。
- **M5 复杂任务**：rank/discover/ideate/poll/debias（子包）。
- **M6 收尾**：测试覆盖、文档、清理。
