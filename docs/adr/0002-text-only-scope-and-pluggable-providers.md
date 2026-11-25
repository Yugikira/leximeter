---
status: accepted
---

# Text-only scope and pluggable optional providers

Leximeter is forked from `openai/GABRIEL`, which bundles image, audio, and PDF modalities alongside OpenAI-native web search and dynamic `**cfg_kwargs` routing. We deliberately constrain Leximeter to text-only modalities (`text` and `entity`), make task parameters explicit, and decouple embeddings (`EmbeddingClient`) and web search (`SearchTool`) as optional injectable providers that degrade gracefully when unconfigured.

## Considered Options

- **Full upstream multimodal parity**: Preserves GABRIEL's full surface area, but pulls in multimodal dependencies, prompt branches, and vendor-coupled search hooks outside our text-research use case.

## Consequences

- Every task is testable offline with fake providers (zero network, zero API keys), and embedding- or search-enhanced tasks remain usable on LLM-only setups.
- Upstream multimodal features and OpenAI-only engine hooks (`response_fn`, `get_all_responses_fn`) are explicitly out of scope.
