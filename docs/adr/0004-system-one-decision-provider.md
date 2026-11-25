---
status: accepted
---

# System One protocol as the primary Decision Provider interface

Evaluating items for classification, pairwise ranking, and boolean filtering with generative chat models incurs unnecessary autoregressive latency, output-token cost, JSON parsing fragility, and multi-run voting overhead to approximate probabilities. We introduce an optional injectable `DecisionClient` abstraction with `SystemOneClient` (`POST /v1/systemone`) as the primary wire protocol, rather than coupling tasks to vendor-specific decision SDKs or chat-completion logprob hacks.

## Considered Options

- **Chat-completion logprobs / `logit_bias`**: Inconsistent support across chat providers, requires per-label prompt engineering, still susceptible to cross-label bleed, and cannot batch dozens of independent typed questions in a single forward pass.
- **Direct vendor SDKs first (Cloudflare, Google, OpenAI)**: Forces three divergent REST dialects upfront before proving task integration, whereas OpenRouter already normalizes 44 decision models (including Jev, Clef, Gemini Decision 1, and GPT-6 Luna) behind `/v1/systemone`, and AMD Token Factory, TypeSafe, vLLM, and SGLang speak `/v1/systemone` natively.

## Consequences

- One protocol client (`SystemOneClient`) unlocks hosted commercial gateways, open-weight cloud trials, and local self-hosted serving (`vllm serve`) while keeping tasks provider-neutral (ADR-0001) and offline-testable via `DummyDecisionClient` (ADR-0002).
- `DecisionClient` remains strictly optional (`None` by default): when unconfigured, every task falls back cleanly to `LLMClient`.
- Tasks with generative stages (such as `discover`, `codify`, `paraphrase`, and `debias`) pair `LLMClient` (`model`) with `DecisionClient` (`decision_model`) independently; direct Cloudflare/Google/OpenAI protocol adapters can be added later without changing task logic.
