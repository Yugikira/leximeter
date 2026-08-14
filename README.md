# Leximeter

**文本专用、厂商中立**的 LLM 测量库——把杂乱无章的定性文本（访谈、演讲、评论、档案…）转化为可分析的结构化数据。

fork 自 [openai/GABRIEL](https://github.com/openai/GABRIEL)，保留其全部文本功能，重做内部结构，解决两个痛点：

- **解耦**：把 6900 行的 `openai_utils.py` 单体切成「厂商协议层 `client/`」+「通用编排层 `engine/`」，并用基类 `Task`/`AttributeTask` 消除各任务间的复制粘贴。
- **厂商中立**：`LLMClient` 抽象 + 按协议划分的 `ResponsesClient` / `ChatCompletionsClient`，可插拔地接 OpenAI / DeepSeek / MiniMax / Ollama 等任何兼容后端；嵌入（本地 BGE）、联网搜索（Tavily）同样作为独立 provider 可单独配置。

> 设计规范见 [research/spec.md](research/spec.md)，GABRIEL 源码移植笔记见 [research/gabriel-source-notes.md](research/gabriel-source-notes.md)。

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

## 配置 Provider

所有任务底层都走 `LLMClient`，用环境变量选择协议与端点（也可每次调用传 `client=` 覆盖）：

```bash
# 走传统 Chat Completions 协议（DeepSeek / MiniMax / Ollama / vLLM 等）
export LEXIMETER_PROTOCOL=chat
export LEXIMETER_BASE_URL=https://api.deepseek.com
export LEXIMETER_API_KEY=sk-...
export LEXIMETER_MODEL=deepseek-chat

# 或走 Responses API（原生 OpenAI）
export LEXIMETER_PROTOCOL=responses
export OPENAI_API_KEY=sk-...

# 嵌入默认用本地 BGE（需 [embeddings]），联网搜索默认关闭
export LEXIMETER_EMBEDDING_PROVIDER=local
export LEXIMETER_SEARCH_PROVIDER=tavily   # 需要 [search] + TAVILY_API_KEY
```

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

每次运行会在 `save_dir` 落盘原始响应与配置，`reset_files=False`（默认）时**断点续跑**，已完成的样本不会重跑。

## 功能一览

| 类别 | 函数 |
|---|---|
| **测量** | `rate`（0–100 评分）、`rank`（两两排序 z 分数）、`classify`（标签）、`extract`（事实抽取）、`discover`（发现判别特征） |
| **清洗** | `merge`（模糊合并）、`deduplicate`（去重）、`filter`（条件筛选）、`deidentify`（脱敏） |
| **生成/辅助** | `seed`、`ideate`、`poll`、`paraphrase`、`codify`、`compare`、`bucket`、`debias`、`whatever` |
| **数据** | `load`（文件夹 → DataFrame） |

所有函数签名与行为见 [research/spec.md](research/spec.md)。

## 完整示例

对应 GABRIEL 教程 notebook 文本部分的示例见 [examples/reference.py](examples/reference.py)，配置好 provider 后：

```bash
python examples/reference.py
```

## 测试

```bash
uv run pytest   # 零网络、零密钥，用假 provider 跑全链路
```

## 引用

本项目 fork 自 GABRIEL，请同时引用上游：

- Asirvatham, H., Mokski, E., and Shleifer, A. (2026). *GPT as a Measurement Tool*. NBER Working Paper No. 34834.
- Asirvatham, H. and Mokski, E. (2026). *GABRIEL*. https://github.com/openai/GABRIEL

## License

Apache-2.0（同上游）。
