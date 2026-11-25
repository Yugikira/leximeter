---
status: accepted
---

# Prefix-first prompt layout with per-run deterministic key ordering

Leximeter's upstream `openai/GABRIEL` prompt templates placed dynamic item content at the very start of each prompt and shuffled attribute, label, and code dictionaries independently on every call. That layout defeated automatic provider KV / prefix caching (`0.0%` single-run and `0.8–1.9%` repeat-run cache hit rates on DeepSeek) while leaving large concurrent batches exposed to cold-cache wave-1 misses. We move all static instructions, rubrics, formatting rules, and `additional_instructions` to an unbroken prompt prefix, place dynamic `Item`, `Circle / Square`, `Persona`, mapping, and chunk payloads at the trailing suffix, default key ordering to deterministic `"per_run"` permutation dispatched run-major (`batch -> run -> item`), and add an adaptive 1-call warmup to `batch_execute`.

## Considered Options

- **Keep upstream item-first prompts with per-item key shuffling**: Preserves byte-level upstream prompt fidelity, but forfeits `30–60%` automatic input/total cost savings on DeepSeek, OpenAI, Gemini, and vLLM/SGLang prefix caches.
- **Prefix-first with `"fixed"` key order as the sole mode**: Maximizes cross-run hot-cache reuse (`86.7%` cache hit rate on repeat runs in DeepSeek testing), but our empirical A/B benchmark (`216` calls on `deepseek-flash`; see [kv-cache-experiment-results.md](../../research/kv-cache-experiment-results.md)) measured a small but statistically detectable `+0.92` MAE localized construct-anchoring shift on `rate` when reversing attribute order (`p = 0.029`), whereas layout drift alone was indistinguishable from baseline noise (`p = 0.235`, with zero degradation on `classify` and `codify`).
- **Explicit `system` / `user` message splitting or vendor cache-control headers**: Couples prompts to protocol-specific message schemas and vendor cache directives (e.g., Anthropic `cache_control`), whereas a single prefix-first user prompt already unlocks automatic prefix caching across provider-neutral `LLMClient` implementations.

## Consequences

- Single-run batches achieve `~61–63%` KV-cache hit rates automatically (`-23%` to `-44%` total cost) with `"per_run"` keeping caller order on run 1 and deterministic distinct permutations on runs `2..k` so multi-run averaging cancels localized order drift while preserving per-run prefix reuse.
- Callers can opt into `order_strategy="fixed"` for maximum cross-run hot-cache reuse (`~87%` cache hit rate on repeat runs) or `order_strategy="per_item"` for legacy upstream shuffling, while pairwise `circle_first` randomization stays intact inside the trailing suffix at zero cache penalty.
- Output DataFrame schemas and checkpoint `Identifier` contracts remain unchanged, and `batch_execute` automatically runs a 1-call sequential warmup only when `len(pending) > max(8, n_parallels)` so small batches pay no serial latency overhead.
