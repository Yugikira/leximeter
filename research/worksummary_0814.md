# Leximeter 工作小结（2026-08-14）

> 本次会话从 fork 决策一路做到重构交付。本文是交接文档，供下次根据测试反馈快速更正。

## 一、做了什么

1. **调研 GABRIEL**：精读源码，产出两份文档——`wiki/`（HTML code wiki，介绍 GABRIEL 功能/架构/组合原理/多模型适配）与 `research/gabriel-source-notes.md`（722 行，逐任务契约，带行号引用）。
2. **grilling 定共识**：项目命名 Leximeter、去多模态、provider 按协议抽象、基类模板方法、复杂任务拆子包、方案 B（重建骨架 + 逐个移植）。
3. **重构实现**：`client/`（厂商协议）+ `engine/`（并发/重试/检查点）+ `tasks/`（18 任务 + 基类）+ `api.py` 门面。
4. **嵌入预筛**：merge/dedup 用嵌入聚类/top-k，嵌入不可用时优雅回退纯 LLM。
5. **spec + 示例 + README**：`/to-spec` 规范、`examples/reference.py`、README。

## 二、当前状态

- **18 个任务 + `load` 全部移植**，19 个 pytest 用例全绿，工作树干净。
- **17 个 commit**，历史清晰，每里程碑一个。
- **尚未用真实 provider（DeepSeek/MiniMax）做端到端验证** —— 这是下一步最优先的事。

## 三、架构（已落地）

```
src/leximeter/
├── api.py        # 公开门面：构造 config → 委托 task
├── client/       # 厂商协议：ResponsesClient / ChatCompletionsClient / DummyClient
│                 #          + EmbeddingClient（本地 BGE）+ SearchTool（Tavily）+ resolve
├── engine/       # 通用编排：batch_execute（并发/重试/检查点）+ checkpoint
├── tasks/        # 业务任务：Config+Task 模板方法；复杂任务拆子包
│                 #   rank/{bt,task}  debias/  ideate/  poll/  discover/
├── prompts/      # Jinja2 模板（文本专用，17 个）
└── utils/        # parsing（safe_json）/ similarity / jinja
```

## 四、共识决策（grilling 定下）

- `LLMClient.complete(prompt, *, model, json_mode, **kw) -> str` 最小原语；`estimate_tokens`/`estimate_cost` 预留抽象方法（未实现）。
- provider **按协议而非厂商**：Responses API 与 Chat Completions 两套；DeepSeek/MiniMax 走 Chat Completions（openai SDK + base_url）。
- 选择机制：env 默认（`LEXIMETER_PROTOCOL`/`BASE_URL`/`API_KEY`/`MODEL`）+ 每调用 `client=`/`embedding_client=`/`search_tool=` 覆盖。
- 去多模态，只留 text/entity；嵌入本地 BGE `bge-m3`；搜索 Tavily（未配置则 warning + 跳过）。

## 五、如何测试

```bash
cd D:\claude\pyapps\sscitext\leximeter

# 单元测试（零网络零密钥，用假 provider）
uv run pytest

# 真实冒烟测试（先配 env）
export LEXIMETER_PROTOCOL=chat
export LEXIMETER_BASE_URL=https://api.deepseek.com   # 或 MiniMax 端点
export LEXIMETER_API_KEY=sk-...
export LEXIMETER_MODEL=deepseek-chat
uv run python examples/reference.py
```

## 六、待更正点（测试反馈最可能命中处，按优先级）

1. **provider 的 openai SDK 调用未实测**：`client/responses.py`、`client/chat.py`、`client/embedding_openai.py` 的 SDK 参数（`text={"format":...}`、`response_format`、`output_text`、`embeddings.create`）是照文档写的、没打过真实 API，最可能报错。
2. **`safe_json` 解析鲁棒性**：`utils/parsing.py` 的容错 JSON 解析，真实 LLM 输出（带 markdown/注释/不完整 JSON）可能解析失败 → 结果大量 `None`。GABRIEL 有 `safest_json` async + LLM 兜底，这里做了简化。
3. **有意的简化**（详见 `spec.md`「Out of Scope」）：bucket 多轮投票 / codify 补全循环 / paraphrase 递归校验 / debias 的 statsmodels 回归（现 numpy lstsq）/ rank 的完整 BT MLE+标准误（现 Zermelo）/ discover 的 net_pct 汇总 / extract 多实体展开 / merge 的 kmeans2（现贪心 top-k）。
4. **标识符/断点续跑**：`{sha1_16}_batch{b}_run{r}` 是检查点 key，改动会破坏续跑。
5. **pandas 3.0.5**：venv 是 pandas 3.x，`astype("boolean")`/`groupby.agg` 已按 3.x 验证通过。

## 七、下一步建议

1. **真实 provider 冒烟测试**（最优先）：先只跑 `examples/reference.py` 的 rate，确认 provider 链路通，再逐个任务试。
2. 按反馈补简化项（嵌入预筛的 kmeans2、discover net_pct 等）。
3. 补 `view` UI（v1 后置）。
