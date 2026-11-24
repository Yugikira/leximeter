# Leximeter

Text-only, provider-neutral LLM measurement library forked and refactored from `openai/GABRIEL`.

## Architecture & Docs

- `src/leximeter/client/` — Provider protocol layer (`LLMClient`, `ResponsesClient`, `ChatCompletionsClient`, `DummyClient`, embedding & search clients).
- `src/leximeter/engine/` — Generic orchestration, concurrency, rate limiting, retries, checkpoints.
- `src/leximeter/tasks/` — Measurement tasks built on `Task` / `AttributeTask` base classes.
- `src/leximeter/api.py` — Thin public facade.
- `research/spec.md` — Design specification and backlog.
- `docs/plan.md` — Refactoring plan and locked decisions.
- `leximeter_wiki/index.html` — Researcher-facing usage guide.

## Commands (`uv`)

```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## Invariants

- **Provider neutrality:** Task layers depend only on the `LLMClient` abstraction (plus optional `EmbeddingClient` / `SearchTool`), never on OpenAI-specific SDK calls or Responses-only fields.
- **Offline testability:** Every provider is injectable; the full test suite must pass with `DummyClient` (zero API keys, zero network).
- **Env precedence:** `import leximeter` calls `load_dotenv()` without overriding existing environment variables (`env > .env`). Never commit `.env` or `runs/`.
