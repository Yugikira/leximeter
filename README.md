# Leximeter

**Leximeter** 是一个**文本专用、厂商中立**的 LLM 测量库——把杂乱无章的定性文本（访谈、演讲、评论、档案…）转化为可分析的结构化数据。

它 fork 自 [openai/GABRIEL](https://github.com/openai/GABRIEL)，保留了后者的全部文本功能（测量、清洗、辅助、复杂编排），但重构了内部结构，解决两个痛点：

- **解耦**：把 6900 行的 `openai_utils.py` 单体拆成「厂商协议层 `client/`」+「通用编排层 `engine/`」，并用基类 `Task` 消除各任务间的复制粘贴。
- **厂商中立**：用 `LLMClient` 抽象 + 按协议划分的 `ResponsesClient` / `ChatCompletionsClient`，可插拔地接 OpenAI / DeepSeek / MiniMax / Ollama 等任何兼容后端；嵌入、联网搜索同样作为独立 provider 可单独配置。

> ⚠️ 状态：**开发中（WIP）**。计划见 [PLAN.md](PLAN.md)。原项目请见 [openai/GABRIEL](https://github.com/openai/GABRIEL)，论文见 NBER w34834。

## 引用

如果使用本项目，请同时引用上游 GABRIEL：

- Asirvatham, H., Mokski, E., and Shleifer, A. (2026). *GPT as a Measurement Tool*. NBER Working Paper No. 34834.
- Asirvatham, H. and Mokski, E. (2026). *GABRIEL*. https://github.com/openai/GABRIEL

## License

Apache-2.0（同上游）。
