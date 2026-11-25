---
status: accepted
---

# Protocol-based LLM clients instead of vendor adapters

Leximeter needs provider-neutral LLM execution across OpenAI, DeepSeek, MiniMax, local gateways (Ollama, vLLM), and future hosts without coupling tasks to vendor SDKs. We model LLM clients by wire protocol (`ResponsesClient` and `ChatCompletionsClient` behind a minimal `LLMClient.complete()` interface) rather than per-vendor adapters or a third-party router.

## Considered Options

- **Per-vendor client classes**: High maintenance churn as OpenAI-compatible providers proliferate, duplicating nearly identical HTTP/SDK wrappers.
- **Third-party router (e.g., LiteLLM)**: Adds a large dependency tree and opaque request translation between Leximeter and the underlying provider.

## Consequences

- Tasks and the execution engine depend solely on `LLMClient`, keeping provider neutrality enforceable.
- Switching between OpenAI-compatible providers requires only environment or per-call configuration (`LEXIMETER_PROTOCOL`, `LEXIMETER_BASE_URL`, `LEXIMETER_API_KEY`, `LEXIMETER_MODEL`); supporting a genuinely different wire protocol requires adding one new protocol client.
