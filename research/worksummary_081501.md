# Leximeter 工作小结（2026-08-15 上午场）

> 本次会话：联网搜索工具（Tavily）调研 + 接入 `whatever` 单任务 + `reference.py` 对齐 GABRIEL notebook + DeepSeek 真实 provider 冒烟 + README 更新。本文是交接文档，供下次按「待完成」清单继续。

## 一、做了什么

1. **确认 web_search 现状**：`client/search_tavily.py` 的 `TavilySearchTool` 是真实实现（非仅抽象类），`resolve_search_tool` 已接线 `LEXIMETER_SEARCH_PROVIDER=tavily`；但 `tasks/base.py` 的 `_get_search_tool()` **定义了却从没被任何任务调用**、工具只用了 `max_results` 一个参数、且**无任何测试**。
2. **调研两套 API 差异**：下载 `uv.lock` 锁定的 tavily-python **0.7.27** wheel 直接读源码核实（v0.7.27 `search()` 仍返回纯 dict，`result.get("results")` 访问正确），确认 Tavily 响应结构为 `{results:[{title,url,content,score,...}], answer, query, response_time}`，顶层 `answer` 是 AI 摘要。
3. **`TavilySearchTool` 构造接口对齐 GABRIEL 调用面**（见「三、联网搜索设计」），去掉自造的 `topic`/`time_range`/`include_raw_content` 参数。
4. **`whatever` 单任务接入 web_search**：Config 加 `web_search`/`web_search_filters`/`search_context_size`，检索结果拼到 prompt 头部再走 `batch_execute`；`Whatever._get_search_tool()` 覆盖基类把旋钮喂进 `resolve_search_tool()`。
5. **`examples/reference.py` 重写**：与 GABRIEL tutorial notebook 文本任务一一对应（14 节，每节标注 cell 编号），玩具数据从 notebook 程序化提取保证逐字一致，含 `whatever` 联网检索示例（对齐 notebook cell 266）。
6. **DeepSeek 真实 provider 冒烟**：`rate` 在模型 `deepseek-v4-flash` 上分别用 Responses 与 Chat Completions 双协议跑通（3 菜品 × 2 属性，n_runs=1）。
7. **README 更新 + commit `58952f7`**：测试数 32→44、修正 Provider 覆盖示例 bug、补 `whatever` 联网检索文档；commit 收下全部 36 文件改动（8391 insertions）。

## 二、当前状态

- **44 个 pytest 全绿**（含 `test_search_tavily.py` 10 例 + `test_tasks.py` 的 whatever web_search 注入 2 例），工作树干净。
- **commit `58952f7`**：`feat: restore GABRIEL complex behaviors + pluggable web search`——含上一批「恢复 GABRIEL 复杂功能」（extract 多实体/rank BT SE/bucket 投票/codify 补全/paraphrase 递归/discover net_pct/merge kmeans2/debias 两步去偏）+ 本次联网搜索工具。因 `api.py`/`test_tasks.py` 混有两者改动且环境不支持交互式拆分，作单 commit。
- **rate 已真实实测**：DeepSeek `deepseek-v4-flash`，Responses（`client.responses.create` + `text.format=json_object`）与 ChatCompletions（`client.chat.completions.create` + `response_format=json_object`）双协议均 OK，结果列、数值语义一致。
- API key（用户提供 sk-0805…69fa）仅经环境变量传入测试脚本，**未落盘**；`.env` 已被 `.gitignore` 排除。

## 三、联网搜索设计（本次核心）

### 3.1 两套 API 返回结果差异（调研结论）

| 维度 | Responses API `web_search`（原生 tool） | Tavily Search（独立 API） |
|---|---|---|
| **执行位置** | 模型服务端内，搜索结果自动注入模型上下文 | 独立调用，模型调用前拿到结构化结果 |
| **返回内容** | 一条已由模型综合的答案（`output_text`）+ 引用 URL 列表（`web_search_call.action.sources`，需 `include` 显式请求） | 逐条结构化结果（`title/url/content/score`）+ 可选 AI 摘要 `answer` 字段 |
| **内容粒度** | 模型已消化内容，只给答案 + 来源 URL | ~300 字符 snippet / 可选全文 `raw_content` / 相关度 `score` |
| **控制旋钮** | `search_context_size`、`filters`、`user_location` | `search_depth`、`topic`、`time_range`、`include/exclude_domains`、`include_answer`、`include_raw_content`、`country` 等 |
| **集成方式** | 一次往返、自动 | 两次往返、需自己拼进 prompt（RAG）——正是 `SearchTool.search() -> str` 的设计 |

**关键结论**：Tavily 的顶层 `answer` 字段就是最接近 Responses API「模型综合答案」的东西——`TavilySearchTool.search()` 把它放在返回文本头部。

### 3.2 GABRIEL 调用面 → Tavily 映射

```python
TavilySearchTool(
    search_context_size="medium",            # low/medium/high（含 small→low、large→high 别名）
                                             #   → Tavily search_depth: fast/basic/advanced
    web_search_filters={
        "allowed_domains": ["arxiv.org"],    # → Tavily include_domains
        "country": "US",                     # 位置提示 → Tavily country
    },
)
```

返回文本 = `answer` 摘要（若 Tavily 给出）+ 每条 `- title — content (url)`。异常/空结果/缺 key/缺依赖一律返回 `""`（caller 跳过检索，不失败）。

### 3.3 `whatever` 接入语义（对齐 GABRIEL `whatever.py:318-322`）

- `web_search` 为 `None` 时由 `web_search_filters` 是否存在推导（`bool(filters)`）——不显式传 `web_search=True` 也能触发。
- run 时对每个 prompt `await search_tool.search(prompt)`，`n_parallels` 限流；结果非空则拼 `f"{ctx}\n\n{prompt}"`，空则原样用。
- 无工具/检索失败 → warning + 原样跑（augment 但不 drop）。

## 四、如何验证

```bash
cd D:\claude\pyapps\sscitext\leximeter

# 单元测试（零网络零密钥，假 provider）
uv run pytest                          # 44 passed

# 真实 provider 冒烟（需配 .env：协议/密钥/模型；web 需再加 TAVILY_API_KEY）
uv run python examples/reference.py    # 14 节，对照 notebook 输出

# 最小 rate 冒烟（本次用过的临时脚本，保留在 /tmp/smoke_rate.py）
DEEPSEEK_API_KEY=sk-... uv run python /tmp/smoke_rate.py
```

## 五、待完成：真实 provider 冒烟测试（最优先）

> memory 旧条目「provider 的 openai SDK 调用未实测」：`responses.py`/`chat.py` 已随 rate 实测解除；`embedding_openai.py` 仍未实测。

| 任务 | 真实 provider 冒烟状态 |
|---|---|
| `rate` | ✅ 已实测（Responses + ChatCompletions 双协议，deepseek-v4-flash） |
| `classify` / `extract` / `rank` | ⬜ 未实测 |
| `merge` / `deduplicate` / `filter` / `deidentify` | ⬜ 未实测（注意 embedding 路径） |
| `paraphrase` / `seed` / `whatever` | ⬜ 未实测 |
| `codify` / `bucket` / `compare` | ⬜ 未实测 |
| `discover` / `ideate` / `poll` / `debias` | ⬜ 未实测（复杂链路，成本高） |
| `embedding_openai.py`（OpenAI 嵌入） | ⬜ 未实测（LocalBge 本地嵌入已随 merge/dedup 测试覆盖逻辑） |

**建议顺序**：先 `classify`/`extract`（AttributeTask 系，与 rate 同构）→ 再 `rank`/`merge` 这些有数值/聚类逻辑的 → 最后复杂链路（discover/debias/poll）。

## 六、待完成：其它功能

### 6.1 联网搜索剩余项

- **搜索工具未用真实 Tavily 实测**：`TavilySearchTool` 只有假 client 的单元测试，未打过真实 Tavily API（需 `TAVILY_API_KEY`）。最可能踩的点：`include_answer=True` 时 Tavily 是否返回非空 `answer`、`search_depth` 映射是否符合预期。
- **web_search 只接了 `whatever` 单任务**：`rate`/`classify`/`extract`/`compare`/`rank` 的 `modality="web"`（notebook cell 262 用法）未接——这是用户本次拍板「whatever 单任务」的范围，后续如需可扩展。

### 6.2 spec Backlog（B1–B7，均未做）

| # | 事项 | 说明 |
|---|---|---|
| B1 | 引擎自适应并发 | GABRIEL 的斜坡启动/滚动限流/动态超时/风暴保护；当前 `batch_execute` 为固定 `n_parallels` + 指数退避重试 |
| B2 | `view` 图形 UI | GABRIEL 的 view.py（文本任务之外） |
| B3 | rank 递归模式 / primer / `initial_rating_pass` | BT 拟合进阶选项，本次只移植 `_pairs_info_gain` 核心 |
| B4 | debias 绘图 PDF | `_regress_against_original` 的 matplotlib 出图（当前直接回归） |
| B5 | debias rank mode 的 `id_column` 对齐 | rank 子任务不传 `id_column`，索引对齐用兜底 |
| B6 | `clean_json_df` 接线 | `utils/parsing.py` 已实现函数本体，任务层未接入 |
| B7 | token 用量 / 成本估算 | `estimate_tokens`/`estimate_cost` 预留抽象方法，v1 未实现 |

### 6.3 其它待办

- **`safe_json` 解析鲁棒性**：真实 LLM 输出（markdown/注释/不完整 JSON）可能解析失败 → 大量 `None`。GABRIEL 有 `safest_json` async + LLM 兜底，leximeter 做了简化。真实 provider 冒烟时留意。
- **identifier/断点续跑**：`{sha1_16}_batch{b}_run{r}` 格式是检查点 key，改动会破坏续跑（沿用 GABRIEL 契约，勿改）。

## 七、下一步建议

1. **按第五节顺序继续真实 provider 冒烟**（最优先）：`classify`/`extract` 开始，逐个解除 memory 里的「未实测」标注。
2. **真实 Tavily 冒烟**：配 `TAVILY_API_KEY` 后跑 reference.py 第 14 节，确认 `answer` 字段/`search_depth` 映射。
3. 按冒烟反馈补简化项（spec「Out of Scope」列全）或 Backlog 项。
