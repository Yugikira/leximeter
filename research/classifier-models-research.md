# Dedicated Classifier & Decision ("System One") Models — Primary-Source Research & Leximeter Integration Analysis

> **Research Date:** November 25, 2026  
> **Scope:** Primary-source verification of dedicated classifier / decision LLM models, wire protocols, pricing, latency, limits, and architectural fit for [Leximeter](../AGENTS.md).  
> **Epistemic policy:** Every specification, endpoint URL, JSON schema, price, and limit below is verified directly against first-party documentation, OpenAPI specs, Hugging Face model cards, or live API catalog responses. Items that could **not** be verified from primary sources are explicitly flagged.

---

## Executive Summary

Between September and November 2026, a distinct new model category—**System One / Decision Models**—has matured rapidly across both commercial APIs and Apache-2.0 open-weight releases. Rather than generating autoregressive prose or JSON tokens that must be parsed, decision models ingest a shared context (`state` / `input`) alongside a batch of typed questions (`noul` / yes-no, `choice`, `score`) and return **calibrated probability distributions in a single forward pass with $0 output-token billing**.

### Key Findings

1. **A de facto open wire standard has emerged: `POST /v1/systemone`.** Pioneered by TypeSafe's **Jev**, the `{model, state, questions} → {model, answers, usage}` protocol is implemented almost identically by **Cloudflare Clef** (self-hosted / SGLang / vLLM), **AMD Radeon Cloud Token Factory (`d3`, `d3-flash`, `Clef`)**, **vLLM native serving** (PR #61565), and **OpenRouter** (`POST /api/v1/systemone` and `POST /api/alpha/decisions`). Two major cloud providers expose slightly divergent REST dialects—**Google Gemini Decisions API** (`POST /v1beta/decisions`) and **OpenAI Decisions API** (`POST /v1/decisions`, `gpt-6-luna`)—both of which OpenRouter already normalizes behind its unified System One / Decisions schema.
2. **OpenRouter is already a 44-model Decisions gateway.** Querying OpenRouter's live catalog (`GET /api/v1/models?output_modalities=decisions`) on 2026-11-25 returns **44 decision models**, including TypeSafe Jev (`typesafe/jev-1.14`), Cloudflare Clef (`cloudflare/clef`, `cloudflare/clef-flash`, `cloudflare/clef-omni`), Google Gemini Decision 1 (`google/gemini-decision-1`), and OpenAI GPT-6 Luna Decisions (`openai/gpt-6-luna-decisions`). **AMD / vLLM Semantic Router `d3` and `d3-flash` are not hosted on OpenRouter** as of this date, though both are Apache-2.0 open-weight models self-hostable via vLLM or testable free on AMD Token Factory (China site).
3. **Economics and latency are 10×–50× better than chat LLMs for classification/adjudication.** Input pricing ranges from **$0.01–$0.10 per 1M input tokens** (and $0.00 output tokens), with `clef-flash` at **$0.038/M** (~39 ms median) and `jev-1.14` at **$0.042/M**; shared text is prefilled once across dozens or hundreds of independent questions.
4. **Leximeter fit is strong but asymmetric across tasks:**
   - **Drop-in Tier 1 wins:** `classify` (single-column via parallel `noul`s; `differentiate` circle/square mode via 3-way `choice`), `rank` (pairwise 4-way `choice`), plus the internal `Classify` gates inside `codify` (completion check), `paraphrase` (recursive validation), `discover` (Stage 3), and `debias` (measurement stages).
   - **Nuanced Tier 2 fits:** `filter` (best via **inverted fan-out**: put `condition` in `state` and batch $K$ entities as parallel `noul` questions—avoiding both naive 1-HTTP-call-per-entity overhead and generative LLM list-recall decay) and `rate` (viable via anchored $L$-level `score` expectation rescaled to 0–100, **not** `noul × 100`, with rubric-synthesis caveats).
   - **Hard non-fits for pure decision models:** `codify` primary verbatim snippet extraction, `extract`, `paraphrase` generation, `compare`, `bucket` synthesis, `deidentify`, `seed`, `ideate`. *(Bonus discovery: Cloudflare's preview extraction model **Clairic** on Workers AI & OpenRouter `/api/alpha/extractions` returns verbatim `[start, end]` character spans—a separate extraction protocol worth watching for static-mode `codify`/`extract`.)*
5. **Minimal Leximeter architecture:** Adding **one** protocol-based `DecisionClient` ABC (consistent with [ADR-0001](../docs/adr/0001-protocol-based-llm-clients.md) and [ADR-0002](../docs/adr/0002-text-only-scope-and-pluggable-providers.md)) with a primary **`SystemOneClient`** (`POST /v1/systemone`—immediately covering TypeSafe, OpenRouter's 44 models, AMD Token Factory, and self-hosted vLLM/SGLang) plus `DummyDecisionClient` and optional thin adapters for Cloudflare Workers AI REST / Gemini Decisions / OpenAI Decisions gives Leximeter broad provider-neutral coverage with minimal surface area.

---

## 1. Cross-Provider Comparison Matrix

| Provider / Model | Exact Model ID(s) | Params / Base | License / Self-Host | Wire Endpoint | Question Primitives | Max Questions / Call | Context Window (Input) | Input Price (/ 1M tok) | Output Price | Latency (Primary / Cited) | On OpenRouter? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **TypeSafe Jev** | `jev-1.14.0`, `jev-1.13.0`, aliases `jev-latest`, `jev-preview` | Undisclosed (closed; trained via RLCD) | Proprietary, managed API only | `POST https://api.typesafe.ai/v1/systemone` | `noul`, `choice` (1–255), `score` (2–10), `box` (1.14+, images) | Unbounded within context | 64k req / 32k per-Q (262k via opt-in header) | **$0.042** | $0 *(reports non-zero internal count)* | Not officially published *(CF bench: 524 ms p50; AA v2.1: 92 ms/task)* | **Yes** (`typesafe/jev-1.14`, `~typesafe/jev-latest`, $0.042/M) |
| **Cloudflare Clef** | `@cf/cloudflare/clef` (`Cloudflare/clef`) | 27B (`Qwen3.8-27B` + JointSchemaHead) | **Apache-2.0** (vLLM, SGLang, PyTorch) | `POST .../accounts/{id}/ai/run/@cf/cloudflare/clef` *(or `/v1/systemone` self-hosted)* | `noul`, `choice` (2–255), `score` (2–10) | **64** (Workers AI) | 65,536 (Workers AI); 16,384 (AMD/default `encode_record`) | **$0.240** (Workers AI); **$0.192** (OpenRouter) | $0 | **209.3 ms p50 / 238.6 ms p95** (CF Decision Index 0.2.1) | **Yes** (`cloudflare/clef`, $0.192/M) |
| **Cloudflare Clef-Flash** | `@cf/cloudflare/clef-flash` (`Cloudflare/clef-flash`) | 9B (`Qwen3.5-9B` + JointSchemaHead) | **Apache-2.0** (vLLM, SGLang, PyTorch) | `POST .../accounts/{id}/ai/run/@cf/cloudflare/clef-flash` | `noul`, `choice` (2–255), `score` (2–10) | **64** (Workers AI) | 24,576 (Workers AI) | **$0.038** *(3,455 Neurons/M)* | $0 | **38.8 ms p50 / 122.4 ms p95** (CF Decision Index 0.2.1) | **Yes** (`cloudflare/clef-flash`, $0.038/M) |
| **Cloudflare Clef-Omni** | `@cf/cloudflare/clef-omni` (`Cloudflare/clef-omni`) | 30B MoE (3B active) + JointSchemaHead | **Apache-2.0** (vLLM) | `POST .../accounts/{id}/ai/run/@cf/cloudflare/clef-omni` | `noul`, `choice` (2–255), `score` (2–10) | **64** (Workers AI) | 64,000 (Workers AI); 65,536 (OpenRouter) | **$0.150** *(13,636 Neurons/M)* | $0 | Not officially published | **Yes** (`cloudflare/clef-omni`, $0.150/M) |
| **Google Gemini Decision 1** | `gemini-decision-1` | Undisclosed (closed; Gemini family) | Proprietary, Gemini Dev API `v1beta` only | `POST https://generativelanguage.googleapis.com/v1beta/decisions` | `noul` (0.01–0.99), `choice` (2–255), `score` (**2–255**) | **1,000** | **262,144** | **$0.100** *(Free tier available)* | $0 (`total_output_tokens: 0`) | Not officially published *(secondary: ~150–250 ms)* | **Yes** (`google/gemini-decision-1`, $0.10/M) |
| **AMD Token Factory / vLLM-SR `d3`** | `d3` (`vllm-sr/d3`) | 26.09B (`Qwen3.8-27B`, bidirectional attn) | **Apache-2.0** (vLLM PR #61565, `vllm-sr`, HF) | `POST https://developer.amd.com.cn/radeon/api/v1/systemone` | `noul`, `choice`, `score` | Bounded by context / 5 MB body | **32,768 / question** (Token Factory; 262k config) | **Free trial** *(daily pts cap; not for prod)* | $0 (`output_tokens: 0`) | Token Factory: **~350 ms** (3-Q text); 1× MI325X: **54.4 ms p50** (27 req/s) | **No** |
| **AMD Token Factory / vLLM-SR `d3-flash`** | `d3-flash` (`vllm-sr/d3-flash`) | 8.39B (`Qwen3.5-9B`, bidirectional attn) | **Apache-2.0** (vLLM, `vllm-sr`, HF) | `POST https://developer.amd.com.cn/radeon/api/v1/systemone` | `noul`, `choice`, `score` | Bounded by context / 5 MB body | **32,768 / question** (Token Factory) | **Free trial** *(daily pts cap; not for prod)* | $0 (`output_tokens: 0`) | Token Factory: **~100 ms** (3-Q text); 1× MI325X: **19.2 ms p50** (95 req/s) | **No** |
| **OpenAI Decisions API** *(landscape)* | `gpt-6-luna` | Undisclosed (closed; GPT-6 Luna) | Proprietary, managed API | `POST https://api.openai.com/v1/decisions` | `predicate` (noul), `choice`, `score` (2–10) | **400** | Not officially published *(OpenRouter lists 1,050,000)* | **$0.100** *(free prompt cache)* | $0 | Not officially published *(AA v2.1: 206 ms/task)* | **Yes** (`openai/gpt-6-luna-decisions`, $0.10/M) |
| **OpenRouter Gateway** | 44 models (see §5) | Multi-vendor | Managed router | `POST https://openrouter.ai/api/v1/systemone` & `POST .../api/alpha/decisions` | `noul`, `choice`, `score`, `box`, + `refusal` | Depends on upstream model | 8k – 1.05M depending on model | **$0.00 (free)** to **$0.192** | $0 | Depends on upstream provider | *(Is OpenRouter)* |

---

## 2. Detailed Provider & Model Analysis

### 2.1 TypeSafe AI — Jev

**Primary sources:**
- Models & pricing: [https://docs.typesafe.ai/models](https://docs.typesafe.ai/models) (`.md` fetched directly)
- OpenAPI 3.1.0 spec (`POST /v1/systemone`): [https://docs.typesafe.ai/api/system-one](https://docs.typesafe.ai/api/system-one)
- Quickstart & SDKs: [https://docs.typesafe.ai/introduction/quickstart](https://docs.typesafe.ai/introduction/quickstart)
- Primitives & confidence math: [Choice](https://docs.typesafe.ai/primitives/choice), [Noul](https://docs.typesafe.ai/primitives/noul), [Score](https://docs.typesafe.ai/primitives/score), [Box](https://docs.typesafe.ai/primitives/box), [Confidence](https://docs.typesafe.ai/confidence), [State](https://docs.typesafe.ai/concepts/state)

#### Architecture & Capabilities
- **Company & model spelling:** **TypeSafe AI** (`typesafe.ai`), flagship model **Jev**—the first "System One" decision model (launched Sept 15, 2026 with `jev-1.13.0`; current release `jev-1.14.0` launched Oct 14, 2026; aliases `jev-latest` and `jev-preview` both resolve to `jev-1.14.0`).
- **Training & weights:** Closed-weight, managed API only; no self-hosting and no per-account fine-tuning/LoRA. TypeSafe documents that Jev is trained with **RLCD (Reinforcement Learning for Calibrated Decisions)** so probabilities reflect empirical frequencies; parameter count and architecture internals are not published.
- **Supported primitives:**
  - **`noul`** (binary yes/no): returns a single calibrated probability $p \in [0, 1]$ (`"noul"`). Multi-label classification is performed by sending $K$ independent `noul` questions in one request. Supports optional `criteria: {"true": ..., "false": ...}`. **Does not return a `confidence` field.**
  - **`choice`** (single-select categorical): 1 to 255 unordered options mapped in `criteria: {option_name: description_or_null}`. Returns winning `choice`, full `probabilities` map summing to 1, and `confidence`.
  - **`score`** (ordinal rubric rating): 2 to 10 ordered level descriptions (`minItems: 1, maxItems: 10` in OpenAPI) in `criteria: [...]`, 0-indexed. Returns continuous expected score $\sum_{i=0}^{L-1} i \cdot p_i$, `legend`, `probabilities`, and `confidence`.
  - **`box`** (`jev-1.14.0`+, "Dreamer-ready" preview): returns normalized image bounding box `{top, bottom, left, right}` + `confidence`; requires header `TypeSafe-Danger-Zone-Images: true`.
- **Non-capabilities:** Cannot generate free-form text, summaries, or verbatim text spans/excerpts. Its `state`, `instructions`, and `criteria` accept plain strings, nested JSON objects, or arrays—allowing structured context and structured instructions (e.g., passing a candidate record per `noul` question).
- **Documented failure modes ("jaggedness"):** TypeSafe explicitly warns that Jev reads literally, struggles with arithmetic/counting and date comparison, degrades on double negatives/indirection, and requires **concrete descriptive levels** for `score` (each level is evaluated independently without seeing its index or neighbors—passing bare numbers `["0", "1", "2"]` degrades calibration drastically).

#### Exact Wire Protocol (`POST https://api.typesafe.ai/v1/systemone`)
- **Authentication:** `Authorization: Bearer <TYPESAFE_API_KEY>`
- **Optional headers:** `TypeSafe-Danger-Zone-Long-Context: true` (expands context to 262k), `TypeSafe-Danger-Zone-Images: true` (enables `assets` and `box`).
- **Content-Type:** `application/json` (or `multipart/form-data` for binary images).

**Request JSON:**
```json
{
  "model": "jev-latest",
  "state": "Our API integration started returning 500 errors on every request about 20 minutes ago.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this",
      "criteria": {
        "billing": "Payment or subscription issues",
        "technical": "Bugs or integration problems",
        "sales": "Pricing or account questions"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated the customer appears",
      "criteria": [
        "Calm, just stating facts",
        "Frustrated but civil",
        "Very angry, strong language"
      ]
    },
    "is_urgent": {
      "type": "noul",
      "instructions": "The message conveys urgency or time-sensitivity",
      "criteria": {
        "true": "Blocking or time-critical problem",
        "false": "Routine inquiry"
      }
    }
  }
}
```

**Response JSON:**
```json
{
  "model": "jev-1.14.0",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "technical",
      "confidence": 1.0,
      "probabilities": { "billing": 0.0, "technical": 1.0, "sales": 0.0 }
    },
    "frustration": {
      "type": "score",
      "score": 0.59,
      "confidence": 0.38,
      "legend": {
        "0": "Calm, just stating facts",
        "1": "Frustrated but civil",
        "2": "Very angry, strong language"
      },
      "probabilities": { "0": 0.41, "1": 0.59, "2": 0.0 }
    },
    "is_urgent": {
      "type": "noul",
      "noul": 0.99
    }
  },
  "usage": {
    "input_tokens": 428,
    "output_tokens": 73
  }
}
```

> **Confidence formula note (TypeSafe):** Unlike `d3` and Gemini (which return $p_{\max}$), TypeSafe computes `choice` confidence as $\frac{p_{\max} - 1/n}{1 - 1/n}$ and `score` confidence as $\max\left(0,\ 1 - \frac{\sum_i p_i |i - m|}{\text{MAD}_{\text{unif}}}\right)$, where $m = \arg\max_i p_i$. Also note that Jev reports a non-zero `usage.output_tokens` for internal scoring tokens even though output tokens are billed at **$0**.

#### Pricing, Limits, Latency & Hosting
- **Pricing:** **$0.042 per 1M input tokens** ($42 / Btok); output tokens **$0**.
- **Context limits:** Default **64,000 tokens per request** (`state` + all questions) and **32,000 tokens per question** (`state` + longest single question). With `TypeSafe-Danger-Zone-Long-Context: true`, both rise to **262,000 tokens** (same price/token, slower and lower accuracy). No fixed cap on question count beyond context.
- **Latency:** *Not formally published in TypeSafe's API docs.* Independent/competitor measurements: **524.1 ms median / 536.0 ms p95** in Cloudflare's Decision Index 0.2.1 benchmark run ([Cloudflare/clef README](https://huggingface.co/Cloudflare/clef)); **92 ms/task** reported from Artificial Analysis Decision Index v2.1 via [systemonemodels.org](https://systemonemodels.org/models/openai-decisions-api/).
- **Rate limits & data:** Per-org RPS and TPS limits visible in `console.typesafe.ai/rate-limits` (`429` with `retry-after`). Customer requests/responses are not used for training; enterprise ZDR available.
- **Hosted providers:** Direct (`api.typesafe.ai`), OpenRouter (`typesafe/jev-1.14`, `typesafe/jev-1.13`, `~typesafe/jev-latest`). Official SDKs: `typesafe-sdk` (Python), `@typesafe-ai/sdk` (JS/TS).

---

### 2.2 Cloudflare — Clef, Clef-Flash, Clef-Omni (+ Clairic)

**Primary sources:**
- Workers AI model docs & raw JSON schemas: [clef](https://developers.cloudflare.com/workers-ai/models/clef/) ([input schema](https://developers.cloudflare.com/workers-ai/models/clef/schema-input.json), [output schema](https://developers.cloudflare.com/workers-ai/models/clef/schema-output.json)), [clef-flash](https://developers.cloudflare.com/workers-ai/models/clef-flash/), [clef-omni](https://developers.cloudflare.com/workers-ai/models/clef-omni/), [clairic](https://developers.cloudflare.com/workers-ai/models/clairic/)
- Workers AI pricing: [https://developers.cloudflare.com/workers-ai/platform/pricing/](https://developers.cloudflare.com/workers-ai/platform/pricing/)
- Hugging Face model cards: [Cloudflare/clef](https://huggingface.co/Cloudflare/clef), [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), [Cloudflare/clef-omni](https://huggingface.co/Cloudflare/clef-omni)

#### Models, Architecture & Capabilities
All three Clef decision models are **Apache-2.0 open-weight** models that append a **Joint Schema Head** (a lightweight transformer head reading the backbone's final hidden states to score all options of all questions jointly in one forward pass, outputting one logit per option with zero autoregressive decoding):

| Model ID (Workers AI) | HF Repo | Backbone / Size | Modalities | Context Window | Workers AI Pricing | OpenRouter Pricing | Latency (Decision Index 0.2.1) |
|---|---|---|---|---|---|---|---|
| `@cf/cloudflare/clef` | `Cloudflare/clef` | `Qwen/Qwen3.8-27B` (27B) | Text, JSON, Images (video in HF) | 65,536 tokens | **$0.240 / 1M input** (21,818 Neurons) | **$0.192 / 1M** (`cloudflare/clef`) | **209.3 ms p50 / 238.6 ms p95** |
| `@cf/cloudflare/clef-flash` | `Cloudflare/clef-flash` | `Qwen/Qwen3.5-9B` (9B) | Text, JSON, Images | 24,576 tokens | **$0.038 / 1M input** (3,455 Neurons) | **$0.038 / 1M** (`cloudflare/clef-flash`) | **38.8 ms p50 / 122.4 ms p95** |
| `@cf/cloudflare/clef-omni` | `Cloudflare/clef-omni` | 30B MoE (3B active) | Text, JSON, Images, Audio, Video | 64,000 tokens (65,536 OR) | **$0.150 / 1M input** (13,636 Neurons) | **$0.150 / 1M** (`cloudflare/clef-omni`) | Not published |

- **Zero Data Retention (ZDR):** Yes on all three models on Workers AI.
- **Free tier:** Workers AI includes **10,000 Neurons/day free** (resets 00:00 UTC) on both Free and Paid plans—equivalent to ~**2.89M input tokens/day** on `clef-flash` ($0.011 / 1,000 Neurons paid rate).
- **Primitives & limits (verified from `schema-input.json`):**
  - `noul` (`instructions` required, optional `criteria: {"true": ..., "false": ...}`), returns `{"type": "noul", "noul": float}` (no `confidence`).
  - `choice` (`instructions` + `criteria` map, **2 to 255 options**), returns `choice`, `probabilities`, `confidence`.
  - `score` (`instructions` + `criteria` array, **2 to 10 levels**, 0-indexed), returns `score`, `legend`, `probabilities`, `confidence`.
  - **Max questions per request:** **1 to 64** (`maxProperties: 64`). Question IDs may use letters, digits, `_`, `.`, `-` (max 100 chars).
  - **Truncation behavior:** Long text `state` is **silently truncated** to fit the remaining context window after questions and media tokens (unlike `d3` and Gemini, which return HTTP 400).
  - **No span/snippet extraction** in Clef.

#### Exact Wire Protocol
1. **Cloudflare Workers AI REST API:**
   - `POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/cloudflare/{clef|clef-flash|clef-omni}`
   - Auth: `Authorization: Bearer <CLOUDFLARE_AUTH_TOKEN>`
   - Request payload matches TypeSafe's `/v1/systemone` schema, **except** `model` must match `^\s*(clef|clef-flash)\s*$` (or `clef-omni`) and images use a top-level `images` array (base64/data URLs, max 4) rather than `assets`.
   - Output payload (`schema-output.json`) matches TypeSafe's `{model, answers, usage: {input_tokens, output_tokens}}` directly inside Workers (`env.AI.run(...)`); Cloudflare's v4 REST API wraps model outputs in `{"result": <schema-output>, "success": true, "errors": [], "messages": []}`.
2. **Self-hosting (exact `/v1/systemone`):**
   - **SGLang:** `lmsysorg/sglang:dev-clef` (`sglang serve --model-path Cloudflare/clef`) exposes `POST /v1/systemone`.
   - **vLLM:** Native support for `ClefFlashForJointSchema` (`Cloudflare/clef-flash`) and `ClefOmniForJointSchema` (`Cloudflare/clef-omni`) on `POST /v1/systemone` ([vLLM Structured Decisions Models](https://docs.vllm.ai/en/latest/models/structured_decisions_models/)).
3. **Third-party hosts:** Hosted on **OpenRouter** (`cloudflare/clef`, `cloudflare/clef-flash`, `cloudflare/clef-omni`) and **AMD Radeon Cloud Token Factory** (`model: "Clef"` at `/v1/systemone`, state truncated to 16,384 tokens).

#### Bonus Discovery: Cloudflare Clairic (Extraction System One Model)
Cloudflare also offers **`@cf/cloudflare/clairic`** (preview, 27B, 65,536 ctx, $0.10/M promo until Dec 1, 2026 then $0.24/M) and **`@cf/cloudflare/clairic-flash`** ($0.10/M), also routed via OpenRouter's `POST /api/alpha/extractions` (`cloudflare/clairic`). Clairic uses a `fields` schema (`text`, `number`, `compute`, `date`, `time`, `datetime`, `duration`, `boolean`, `enum`, `object`, `list`, `estimate`, `forecast`) and returns **exact `[start, end]` character spans into `state`** (`span` for `text`, `spans` for `list` of `text`) with `p_null` and `confidence` ([schema-output.json](https://developers.cloudflare.com/workers-ai/models/clairic/schema-output.json)). Crucially, Clairic executes **each `text` field in its own pass**, re-billing `state` input tokens once per pass (`usage.passes`).

---

### 2.3 Google — `gemini-decision-1` (Gemini Decisions API)

**Primary sources:**
- Model card: [https://aistudio.google.com/docs/models/gemini-decision-1](https://aistudio.google.com/docs/models/gemini-decision-1) (`.md` fetched directly)
- Developer guide: [https://aistudio.google.com/docs/decisions-api](https://aistudio.google.com/docs/decisions-api) (`.md` fetched directly)
- REST API Reference: [https://aistudio.google.com/api-reference/decisions/create-decision](https://aistudio.google.com/api-reference/decisions/create-decision) (`.md` fetched directly)
- Pricing & Rate limits: [Pricing](https://ai.google.dev/gemini-api/docs/pricing), [Rate limits](https://aistudio.google.com/docs/rate-limits)

#### Capabilities & Constraints
- **Model ID:** `gemini-decision-1` (launched November 2026; closed-weight).
- **Platform availability:** Explicitly documented under Limitations as **"Available on the Gemini Developer API (`v1beta`) with Gemini Decision 1 only."** *Vertex AI GA endpoint / model ID could not be verified in Google Cloud primary documentation as of 2026-11-25.*
- **Supported primitives:**
  - **`noul`**: Yes/No probability **clamped to `[0.01, 0.99]`** (docs warn: *"a check above 0.99 never matches"*), plus `confidence = max(noul, 1 - noul)`. Optional `criteria` uses **`true_description` / `false_description`** (NOT `true` / `false`!).
  - **`choice`**: 2 to 255 options mapped to descriptions (`""` allowed if self-explanatory). **Critical quirk:** option keys are **sorted alphabetically in UTF-8 byte order** before evaluation, and exact ties break alphabetically (avoid positional keys like `"none_of_the_above"`).
  - **`score`**: **2 to 255** ordered non-empty strings (2–10 recommended—much higher ceiling than Jev/Clef's 10). Returns probability-weighted expected `score`, `legend`, `probabilities`, and `confidence`.
- **Scale & modalities:** Up to **1,000 independent questions** in one forward pass over up to **262,144 tokens** of shared multimodal input (text, images, audio, video, PDFs). Text-only `instructions` per question; JSON objects must be serialized to strings via `json.dumps()` rather than passed as native JSON objects.
- **Explicitly unsupported features:** Text generation, structured JSON outputs, span extraction, Batch API, context caching, Flex/Priority inference, function calling, thinking, and Google Search grounding (though OpenRouter's catalog metadata lists a web_search price field, Google's own model page marks Search Grounding "not supported").

#### Exact Wire Protocol (`POST https://generativelanguage.googleapis.com/v1beta/decisions`)
- **Authentication:** `x-goog-api-key: $GEMINI_API_KEY` (or Python `google-genai >= 3.2`: `genai.Client(http_options={"api_version": "v1beta"}).decisions.create(...)`). **Not OpenAI-compatible and not `/v1/systemone`.**
- **Key validation limits:** Question keys must match `[A-Za-z0-9_-]+` (1–64 ASCII chars). Option keys 1–128 chars (≤512 UTF-8 bytes), no leading/trailing spaces or non-ASCII spaces.

**Request JSON:**
```json
{
  "model": "gemini-decision-1",
  "input": "Customer returned the sweater because they preferred green over blue.",
  "questions": {
    "is_defect": {
      "type": "noul",
      "instructions": "Is the customer reporting a product defect?",
      "criteria": {
        "true_description": "The product itself is broken or faulty.",
        "false_description": "Preference, sizing, shipping, or pricing complaint."
      }
    },
    "queue": {
      "type": "choice",
      "instructions": "Which support queue should handle this message?",
      "criteria": {
        "billing": "Payment, invoice, or refund questions.",
        "replacements": "Damaged, missing, or wrong items.",
        "general": "Product questions and account updates."
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is this issue?",
      "criteria": [
        "Cosmetic: no lost functionality.",
        "Workaround available.",
        "Fully blocked: no workaround."
      ]
    }
  }
}
```

**Response JSON:**
```json
{
  "answers": {
    "is_defect": { "type": "noul", "noul": 0.03, "confidence": 0.97 },
    "queue": {
      "type": "choice",
      "choice": "general",
      "probabilities": { "billing": 0.02, "general": 0.96, "replacements": 0.02 },
      "confidence": 0.96
    },
    "severity": {
      "type": "score",
      "score": 0.12,
      "legend": {
        "0": "Cosmetic: no lost functionality.",
        "1": "Workaround available.",
        "2": "Fully blocked: no workaround."
      },
      "probabilities": { "0": 0.89, "1": 0.10, "2": 0.01 },
      "confidence": 0.89
    }
  },
  "usage": {
    "total_input_tokens": 210,
    "input_tokens_by_modality": [{ "modality": "text", "tokens": 210 }],
    "total_output_tokens": 0,
    "total_tokens": 210
  }
}
```
- **Error / Safety handling:** Input safety filter blocks return HTTP `400` with `{"error": {"code": "safety" | "prohibited_content" | "blocklist" | "image_safety" | "content_blocked", "message": "..."}}` (or `prompt_feedback.block_reason` in schema).

#### Pricing, Rate Limits, Latency & Hosting
- **Pricing:** **$0.10 / 1M input tokens** across all modalities (text, image, audio, video, PDF); **$0 output** (`total_output_tokens` is always `0`). Shared input is billed once per request regardless of question count.
- **Rate Limits (Interactive Standard):**
  - **Free Tier:** 200 RPM / 50,000 TPM / 10,000 RPD *(data may be used to improve Google products)*.
  - **Paid Tier 1:** 1,500 RPM / 1,000,000 TPM / 50,000 RPD *(data NOT used to improve products)*.
  - **Paid Tier 2:** 5,000 RPM / 4,000,000 TPM / Unlimited RPD.
  - **Paid Tier 3:** 15,000 RPM / 10,000,000 TPM / Unlimited RPD.
- **Latency:** *Not published in Google's official docs.* Secondary write-ups report ~150–250 ms end-to-end.
- **Third-party gateways:** Hosted on **OpenRouter** as `google/gemini-decision-1` at the same **$0.10/M input** price, translated into OpenRouter's unified System One schema.

---

### 2.4 AMD Radeon Cloud / Token Factory — `d3` & `d3-flash` (+ hosted `Clef`)

**Primary sources (directly fetched):**
- AMD Radeon Cloud Docs: [models/d3](https://amd-aim.github.io/radeon-cloud-docs/models/d3/), [models/d3-flash](https://amd-aim.github.io/radeon-cloud-docs/models/d3-flash/), [models/clef](https://amd-aim.github.io/radeon-cloud-docs/models/clef/), [guides/token-factory](https://amd-aim.github.io/radeon-cloud-docs/guides/token-factory/), [api/overview](https://amd-aim.github.io/radeon-cloud-docs/api/overview/), [api/models](https://amd-aim.github.io/radeon-cloud-docs/api/models/), [api/rate-limits](https://amd-aim.github.io/radeon-cloud-docs/api/rate-limits/), [api/usage](https://amd-aim.github.io/radeon-cloud-docs/api/usage/)
- Hugging Face model cards: [vllm-sr/d3](https://huggingface.co/vllm-sr/d3), [vllm-sr/d3-flash](https://huggingface.co/vllm-sr/d3-flash)
- vLLM serving docs: [vLLM Structured Decisions Models](https://docs.vllm.ai/en/latest/models/structured_decisions_models/)

#### Models & Capabilities
- **Origin & license:** `d3` and `d3-flash` are the flagship **Decision 3.0** multimodal foundation decision models of **vLLM Semantic Router** (`vllm-sr`), trained on AMD Instinct MI325X GPUs and released under **Apache-2.0**.
  - **`d3`** (`vllm-sr/d3`): **26.09B parameters** (including 0.46B vision encoder), built on `Qwen/Qwen3.8-27B` with **bidirectional attention** and a linear classification head (`D3ForSequenceClassification`). Ranks **#1 of 149** on the Jev Decision Index 0.3.2 text board (**66.3**) and **#1 of 33** on the vision board (**71.3**).
  - **`d3-flash`** (`vllm-sr/d3-flash`): **8.39B parameters** (including 0.46B vision encoder), built on `Qwen/Qwen3.5-9B`. Scores **55.8** on Jev Decision Index 0.3.2 text board (~3× faster than `d3`).
  - **Sibling models:** Hugging Face's Decision 3.0 collection includes smaller open-weight variants (`d3-mini`, `d3-nano`, `d3-lite`, `d3-edge`). AMD Token Factory also hosts Cloudflare's **`Clef`** (27B, model ID `"Clef"`) on the same `/v1/systemone` endpoint.
- **Supported primitives:** `choice`, `noul`, `score`. Inputs: `state` (string or any JSON value) plus optional `images` (PNG/JPEG/WebP data URLs, ≤1.6 MP) and `videos` (MP4/WebM/MOV/MKV data URLs, 2 fps, ≤32 frames). **No free-form text generation or span extraction.**

#### Exact Wire Protocol (`POST https://developer.amd.com.cn/radeon/api/v1/systemone`)
- **Explicitly NOT on `/v1/chat/completions`:** `messages`, `max_tokens`, `temperature`, and `stream` do not apply.
- **Geography & Auth:** Token Factory is on the **China site only** (`https://developer.amd.com.cn/radeon/api/v1`; `/api/v1/...` and `/v1/...` are equivalent). Bearer auth with `rc-...` key (`Authorization: Bearer $RADEON_API_KEY`).
- **Schema quirks specific to `d3` / `d3-flash`:**
  - Top-level `instructions` is **rejected with HTTP 400** (`"this model reads a fixed prompt, so it takes no instructions"`).
  - `noul` accepts **no `criteria`**—put the full question in `instructions`.
  - Oversized inputs (>32,768 tokens per question) are **refused with HTTP 400** (never silently truncated; body max 5 MB).
  - Note: Hosted `Clef` on the same endpoint behaves slightly differently—it accepts optional `criteria: {"true", "false"}` on `noul`, silently truncates `state` at 16,384 tokens, rejects unknown keys with HTTP 422 FastAPI `{"detail": [...]}` errors, and omits `confidence`/`probabilities` on `noul` and `diagnostics`.

**Response JSON (`d3` / `d3-flash`):**
```json
{
  "id": "decision-aa4a76300e0af8f5",
  "object": "structured_decision",
  "created": 1792166448,
  "model": "d3",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "auth",
      "probabilities": { "auth": 0.9993, "billing": 0.0000, "other": 0.0006 },
      "confidence": 0.9993
    },
    "urgent": {
      "type": "noul",
      "noul": 0.9967,
      "probabilities": { "yes": 0.9967, "no": 0.0033 },
      "confidence": 0.9967
    },
    "severity": {
      "type": "score",
      "score": 1.8641,
      "legend": { "0": "Low", "1": "Medium", "2": "High" },
      "probabilities": { "0": 0.0034, "1": 0.1292, "2": 0.8674 },
      "confidence": 0.8674
    }
  },
  "usage": { "input_tokens": 352, "output_tokens": 0 },
  "diagnostics": {
    "team": { "label_mass": 1.0, "argmax_is_label": true },
    "urgent": { "label_mass": 1.0, "argmax_is_label": true },
    "severity": { "label_mass": 1.0, "argmax_is_label": true }
  }
}
```
- **Unique bonus field — `diagnostics`:** Reports per-question `label_mass` (probability mass falling on valid answer labels) and `argmax_is_label`—useful for detecting out-of-distribution/malformed prompts. `confidence` equals $p_{\max}$ on all three question types.

#### Pricing, Rate Limits, Latency & Self-Hosting
- **Token Factory pricing & terms:** **Free trial / development service** (*"no credits spent"*, *"not for production"*, stability `experimental`). Metered in **Token Factory points (`pts`)** against a per-account daily cap resetting at 00:00 Beijing time (UTC+8), reported via `X-RateLimit-*-User-Daily-USD` headers and `GET /v1/usage`. *Exact `pts/1M` rate for `d3`/`d3-flash` is served dynamically by authenticated `GET /v1/models` and is not published on static documentation pages.*
- **Token Factory rate limits:** Platform admission: **30 RPM / 8 concurrent** per API key (120 RPM per IP). Gateway metering: **20 RPM** per account across chat, speech, and `/v1/systemone` combined.
- **Latency:**
  - *Hosted Token Factory:* `d3` ~**0.35 s** (3-question text) / ~**8 s** (13k-token input); `d3-flash` ~**0.10 s** (3-question text).
  - *Self-hosted on 1× AMD Instinct MI325X (HF model cards):* `d3` **54.4 ms p50** (1-Q text), **548 ms** (16-Q text), **27 req/s** with p99 < 1 s (vLLM Semantic Router batching); `d3-flash` **19.2 ms p50** (1-Q text), **171 ms** (16-Q text), **95 req/s** with p99 < 1 s.
- **Self-hosting options:**
  1. **vLLM native:** `vllm serve vllm-sr/d3 --max-model-len 32768` (on vLLM `main` since 2026-10-14 via PR [#61565](https://github.com/vllm-project/vllm/pull/61565); bidirectional attention disables chunked prefill, so `--max-model-len 32768` is needed to fit on an 80 GB GPU).
  2. **vLLM Semantic Router Engine mode:** `vllm-sr serve -e vllm-sr/d3` (`localhost:8899/v1/systemone`).
  3. **In-process Python:** `AutoModel.from_pretrained("vllm-sr/d3", trust_remote_code=True).system_one(...)` (61.8 GiB peak VRAM for `d3`; 23.8 GiB for `d3-flash`).

---

### 2.5 OpenRouter — Multi-Provider Decisions Gateway

**Primary sources (live API + official docs fetched directly):**
- Live models catalog: `GET https://openrouter.ai/api/v1/models?output_modalities=decisions` (queried 2026-11-25)
- OpenAPI specs: [`POST /api/alpha/decisions`](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request), [`POST /api/v1/systemone`](https://openrouter.ai/docs/api/api-reference/systemone/submit-a-system-one-request)
- Guides & SDKs: [TypeSafe SDK guide](https://openrouter.ai/docs/guides/community/typesafe-sdk), [Multimodal Decisions guide](https://openrouter.ai/docs/guides/community/multimodal-decisions), [Python SDK Alpha.Decisions](https://openrouter.ai/docs/client-sdks/python/sdks/decisions/README.md), [Alpha.Extractions](https://openrouter.ai/docs/client-sdks/python/sdks/extractions/README.md), [Rerank](https://openrouter.ai/docs/client-sdks/python/sdks/rerank/README.md)

#### Dedicated Endpoints (NOT just `/api/v1/chat/completions`!)
OpenRouter exposes **four** non-chat endpoints relevant to classification, extraction, and ranking:
1. **`POST https://openrouter.ai/api/v1/systemone`** — Drop-in compatible with TypeSafe's `/v1/systemone` API and SDKs (just set `TYPESAFE_BASE_URL=https://openrouter.ai/api` and `TYPESAFE_API_KEY=$OPENROUTER_API_KEY`; bare IDs like `jev-1.13` / `jev-latest` auto-map to `typesafe/`).
2. **`POST https://openrouter.ai/api/alpha/decisions`** — OpenRouter's canonical Decisions endpoint (same request/response schema as `/api/v1/systemone`, available in official OpenRouter Python/TS/Go SDKs as `open_router.alpha.decisions.create(...)`).
3. **`POST https://openrouter.ai/api/alpha/extractions`** — For extraction System One models (`cloudflare/clairic`, ScaleDown Extract) using `fields` instead of `questions`.
4. **`POST https://openrouter.ai/api/v1/rerank`** — Cohere-compatible reranking endpoint (`query`, `documents`, `top_n`).

> **Important schema additions on OpenRouter:** Responses add top-level `id`, `provider`, and `usage.cost` (in USD credits). Answers add a fifth union member **`{"type": "refusal"}`** when an upstream provider (e.g., OpenAI) refuses a single question while still answering others in the same request. Note also that `GET /api/v1/models` omits decision models by default unless filtered with `?output_modalities=decisions`.

#### Live Catalog Inventory (`GET /api/v1/models?output_modalities=decisions` — 44 Models on 2026-11-25)

| Target / Notable Model | OpenRouter Model ID | Context | Input $/1M | Output $/1M | Notes |
|---|---|---|---|---|---|
| **TypeSafe Jev 1.14 / Latest** | `typesafe/jev-1.14`, `~typesafe/jev-latest`, `typesafe/jev-1.13` | 262,144 | $0.042 | $0 | Matches direct TypeSafe pricing |
| **Cloudflare Clef** | `cloudflare/clef` | 65,537 | **$0.192** | $0 | Cheaper than Workers AI list ($0.240) |
| **Cloudflare Clef-Flash** | `cloudflare/clef-flash` | 24,576 | $0.038 | $0 | Matches Workers AI pricing |
| **Cloudflare Clef-Omni** | `cloudflare/clef-omni` | 65,536 | $0.150 | $0 | Text + image + audio + video |
| **Google Gemini Decision 1** | `google/gemini-decision-1` | 262,144 | $0.100 | $0 | Translated to SystemOne schema |
| **OpenAI GPT-6 Luna Decisions** | `openai/gpt-6-luna-decisions` | 1,050,000 | $0.100 | $0 | Translated from OpenAI `/v1/decisions` |
| **AMD / vLLM-SR `d3` / `d3-flash`** | *Not hosted* | — | — | — | Absent from live 44-model catalog |
| **Perplexity Decider v1.3** | `perplexity/pplx-decider-v1.3:free`, `...flash-v1.3:free` | 262,144 | **$0.000** | $0 | Free tier; paid `~perplexity/pplx-decider-latest` $0.02/M |
| **Liquid AI d1.2 / Ultra** | `liquid/d1.2`, `liquid/d1.2-ultra` | 65,536 | $0.020 / $0.042 | $0 | Strong benchmark scores (d1-ultra 70.6 on AA v2.1) |
| **Microsoft Decision 1.1** | `microsoft/microsoft-decision-1.1` | 32,768 | $0.042 | $0 | Qwen3.5-9B based |
| **ScaleDown Classify-1** | `scaledown/classify-1` | 1,000,000 | $0.040 | $0 | 1M context, text/image/file |
| **Upstage Solar Decide** | `upstage/solar-decide`, `...flash` | 524,288 | $0.050 | $0 | 512k context, text-only |
| **Baseten Decisions** | `baseten/joey-1` ($0.01), `groundhog-1` ($0.035), `bison-1` ($0.10), `mammoth-1` ($0.10) | 16k–32k | $0.010–$0.100 | $0 | Low-cost fast tier (`joey-1` at $0.01/M) |
| **Parallel Decisions Fast** | `parallel/decisions-fast` | 32,000 | $0 ($0.001/req) | $0 | Web-grounded flat-rate decisions |
| **Cloudflare Clairic** | `cloudflare/clairic` | 65,536 | $0.100 | $0 | Listed in catalog; called via `/api/alpha/extractions` |

*(Note: `typesafe/jev-router` appears in the standard `/api/v1/models` text catalog as a model-routing chat completion endpoint—it is **not** a decision endpoint.)*

---

### 2.6 Broader Protocol Landscape Check

| Protocol / API | Wire Shape | Zero-Shot NL Rubrics? | Calibrated Probabilities? | Status & Relevance to Leximeter |
|---|---|---|---|---|
| **System One (`POST /v1/systemone`)** | `{model, state, questions: {id: {type, instructions, criteria}}}` → `{answers, usage}` | **Yes** (arbitrary NL labels/rubrics per call) | **Yes** | **Primary target protocol.** Adopted by TypeSafe, OpenRouter (44 models), AMD Token Factory, vLLM native serving, SGLang (`Clef`), and `vllm-sr`. Even serves standard LLMs (`Qwen3`, `Gemma-4`) on vLLM via `NextTokenStrategy` + prefix caching! |
| **OpenAI Decisions (`POST /v1/decisions`)** | `{model: "gpt-6-luna", input, questions: [{name, type: "predicate"\|"choice"\|"score", ...}]}` | **Yes** | Unclaimed by OpenAI *(AA v2.1 ECE 0.110 vs Jev 0.054)* | Major vendor dialect (GA Oct 22, 2026; 400 Qs/call; $0.10/M). Differs by using `input`, `questions` array, and `predicate` instead of `noul`. Already accessible via OpenRouter's `/v1/systemone`. |
| **Google Gemini Decisions (`POST /v1beta/decisions`)** | `{model: "gemini-decision-1", input, questions: {id: ...}}` | **Yes** (up to 1,000 Qs; 255 score levels) | **Yes** (clamped `[0.01, 0.99]`) | Major vendor dialect (Nov 2026; $0.10/M). Minor deltas: `input`, `true_description`/`false_description`, alphabetical choice sort, 64-char ASCII keys. Already accessible via OpenRouter's `/v1/systemone`. |
| **vLLM Pooling (`/classify`, `/score`, `/rerank`)** | `/classify`: `{model, input}` (fixed head); `/score`: `{text_1, text_2}`; `/rerank`: `{query, documents}` | **No** for `/classify` (fixed trained classes); partial for cross-encoder `/score` & `/rerank` | Softmax/sigmoid logits | Useful only if running custom fine-tuned sequence classifiers or cross-encoder rerankers. For Leximeter's arbitrary natural-language constructs, vLLM's newer **`/v1/systemone`** endpoint supersedes `/classify`. |
| **Cohere `/v1/classify`** | `{inputs, examples}` (few-shot) | Few-shot only | Confidence scores | **Officially Deprecated** in Cohere v2 docs ([docs.cohere.com/reference/classify](https://docs.cohere.com/reference/classify)). Do **not** design around it. |
| **HuggingFace TEI (`/predict`, `/rerank`)** | `/predict`: `{inputs}`; `/rerank`: `{query, texts}` | **No** (fixed classifier head / relevance rerank) | Softmax probabilities | Same limitation as vLLM `/classify`—cannot evaluate arbitrary natural-language label/attribute definitions on the fly without NLI prompt templates. |
| **Chat Completions + `logprobs` / `logit_bias`** | `POST /v1/chat/completions` with `max_tokens=1, logprobs=true, top_logprobs=20` | **Yes** | Uncalibrated raw LM logprobs | Fragile across providers (reasoning models, Anthropic, Gemini often omit `logprobs`/`logit_bias`; single-token label encoding fails on multi-token labels). Better handled server-side by vLLM's `/v1/systemone` (`NextTokenStrategy`) or by Leximeter's existing `LLMClient` fallback. |

---

## 3. Wire Protocol Diff: Subtle Gotchas Across Decision Providers

Even among System One–family endpoints, **seven concrete wire incompatibilities** must be normalized inside any provider-neutral client:

| Dimension | TypeSafe `/v1/systemone` | OpenRouter `/api/v1/systemone` & `/api/alpha/decisions` | Cloudflare Workers AI `/ai/run/@cf/...` | AMD Token Factory (`d3` / `d3-flash`) | Google Gemini `/v1beta/decisions` | OpenAI `/v1/decisions` |
|---|---|---|---|---|---|---|
| **Content field** | `state` (str/obj/arr) | `state` (str/obj/arr) | `state` (str/obj/arr) | `state` (str/obj/arr) | **`input`** (str or `Content[]`; JSON must be `dumps()` string) | **`input`** (str, messages, or `input_json`) |
| **Questions container** | `dict[id, Q]` | `dict[id, Q]` | `dict[id, Q]` (max 64) | `dict[id, Q]` | `dict[id, Q]` (max 1,000) | **`list[{name, ...}]`** (max 400) |
| **Yes/No type name** | `"noul"` | `"noul"` | `"noul"` | `"noul"` | `"noul"` | **`"predicate"`** |
| **Noul criteria keys** | `{"true", "false"}` (optional) | `{"true", "false"}` (optional) | `{"true", "false"}` (optional) | **Not supported** (instructions only) | **`{"true_description", "false_description"}`** | N/A (`instructions`) |
| **Question ID rules** | Arbitrary string | Arbitrary string | `[A-Za-z0-9_.-]`, ≤100 chars | Arbitrary string | **`[A-Za-z0-9_-]+`, ≤64 chars** | `name` field |
| **`noul` answer fields** | `{type, noul}` *(no confidence)* | `{type, noul, confidence?}` | `{type, noul}` *(wrapped in `{result}` on REST)* | `{type, noul, probabilities: {yes,no}, confidence}` | `{type, noul [0.01..0.99], confidence}` | Predicate prob |
| **`confidence` semantics** | Normalized spread (`choice`: $(p_{\max}-1/n)/(1-1/n)$; `score`: $1 - \text{MAD}/\text{MAD}_{\text{unif}}$) | Passed through from upstream (optional) | TypeSafe spread formula | **$p_{\max}$** (top probability) | **$p_{\max}$** (or $\max(p, 1-p)$) | Model confidence |
| **Per-question refusal** | N/A | **`{"type": "refusal"}`** | N/A | N/A | HTTP 400 safety block (whole request) | Per-question refusal |
| **Oversized input** | HTTP 400 (64k/32k; 262k opt-in) | Upstream-dependent | **Silent `state` truncation** | HTTP 400 (>32,768 tok/Q) | HTTP 400 (>262,144 tok) | HTTP 400 |

---

## 4. Leximeter Task Fit & Integration Surface

Grounded in [GLOSSARY.md](../GLOSSARY.md), [ADR-0001](../docs/adr/0001-protocol-based-llm-clients.md), [ADR-0002](../docs/adr/0002-text-only-scope-and-pluggable-providers.md), [ADR-0003](../docs/adr/0003-checkpoint-backed-task-recovery.md), and `src/leximeter/tasks/*.py`.

### 4.1 Task-by-Task Fit Assessment

#### Tier 1 — Drop-In, High-Impact Fits

1. **`classify` ([`tasks/classify.py`](../src/leximeter/tasks/classify.py)) — Best overall fit**
   - **Single-column mode:** Each `(label, definition)` in `ClassifyConfig.labels` becomes an independent `noul` question in **one** decision call per unique item (`state = item_text`).
     - *Why better than chat LLM:* Decision models evaluate every question in strict isolation against the shared prefill—mechanically enforcing `classification_prompt.jinja2`'s hardest rule (*"Independently judge each label; no cross-label inference"*) with zero JSON-parsing failures (`_parse` / `_regex` fallback eliminated) and zero output tokens.
     - *Calibration & `n_runs`:* Because `noul` returns a calibrated probability $p \in [0, 1]$, `n_runs=1` with a threshold (reusing `min_frequency` or a `decision_threshold: float = 0.5`—default `0.5` or configurable `0.6`/`0.7`) replaces repeated stochastic voting runs (`n_runs=3..5`). Leximeter can optionally emit `{label}_prob` columns (or store raw probabilities in the raw checkpoint CSV) while preserving the exact boolean + `predicted_classes` output contract!
     - *Batching:* Respect provider question caps (e.g., 64 on Cloudflare Workers AI) via `n_attributes_per_run` / `max_questions_per_call`.
     - *Caveat:* When `additional_instructions` explicitly asks for *"only the best label"* (mutually exclusive multi-class, see `classification_prompt.jinja2:46`), a single `choice` question (plus optional companion `noul` for "any label applies") is superior to independent `noul`s. If `additional_instructions` contains general calibration notes or few-shot examples, include them in `state` (as `{"guidelines": ..., "text": ...}`) or question `instructions`.
   - **`differentiate` (circle/square) mode:** Currently `discover` expands each label into `{lab}_actual` and `{lab}_inverted` and asks an LLM across 10 lines of prompt rules to never mark both True.
     - With a Decision model, pass `state = {"circle": entry_circle, "square": entry_square}` (respecting `circle_first` key ordering for positional-bias control!) and emit **one 3-way `choice` question per base label** with options `{"circle": "<circle manifests label more than square>", "square": "<square manifests label more than circle>", "neither": "<absent or equal in both>"}`.
     - This **structurally guarantees mutual exclusivity**, halves the number of evaluated questions, and eliminates contradictory `(True, True)` outputs! (Note: in standalone `Classify(differentiate=True)` where caller passes pre-expanded pairs rather than base labels, either detect `_actual`/`_inverted` pairs or run individual `noul` questions with strict instructions.)

2. **`rank` ([`tasks/rank/task.py`](../src/leximeter/tasks/rank/task.py)) — Ideal fit**
   - Every round adjudicates `(circle, square)` pairs across `attributes`, choosing exactly one of 4 labels: `"circle"`, `"square"`, `"draw"`, `"insufficient signal"` (`rankings_prompt.jinja2`).
   - Maps **1:1** to one 4-option `choice` question per attribute per pair, evaluated in a single forward pass with `state = {"circle": ..., "square": ...}` (randomized order per `circle_first`).
   - Eliminates `_coerce_dict` parsing failures, cuts latency/cost sharply across `n_rounds × matches_per_round × N/2` tournaments (e.g., 5 rounds × 3 matches × 1,000 items = 7,500 pairwise calls), and feeds directly into `fit_bt` / `bt_standard_errors` unchanged.
   - *Subtlety:* For `gemini-decision-1`, choice keys are sorted alphabetically (`circle`, `draw`, `insufficient_signal`, `square`); this is benign, though ties break alphabetically toward `circle`—to stay unbiased, break exact probability ties randomly or as `draw`.
   - *Future research upside:* Calibrated `probabilities["circle"]` vs `probabilities["square"]` could optionally support soft fractional Bradley–Terry updates in a later milestone.

3. **Composite tasks' internal `Classify` stages (Zero-effort cascading wins):**
   - **`paraphrase` ([`tasks/paraphrase.py`](../src/leximeter/tasks/paraphrase.py)):** Generation MUST remain on `LLMClient`, but `_recursive_validate` (lines 245–539) runs `Classify` on every candidate (`instructions_followed` — up to `n_runs × later_round_run_multiplier` candidates per failed row!). Injecting `decision_client` into `Classify` immediately accelerates validation with `state = {"instructions": ..., "original_text": ..., "modified_text": ...}` and a single `noul` question.
   - **`discover` ([`tasks/discover/task.py`](../src/leximeter/tasks/discover/task.py)):** Stage 1 (`Codify` / `Compare`) and Stage 2 (`Bucket`) are generative (`LLMClient`), while Stage 3 (`Classify`, lines 327–533) is often the **highest-volume stage** (classifying the entire corpus on discovered buckets) and switches seamlessly to `DecisionClient`.
   - **`debias` ([`tasks/debias/task.py`](../src/leximeter/tasks/debias/task.py)):** Original & stripped measurement stages (`mode in {"classify", "rank", "rate"}`) and `_measure_remaining_signal` can run on `DecisionClient`, while the removal stage (`codify` / `paraphrase`) stays on `LLMClient` except for its internal `Classify` completion/validation checks.

---

#### Tier 2 — Viable with Important Design & Cost Trade-Offs

4. **`filter` ([`tasks/filter.py`](../src/leximeter/tasks/filter.py)) — Crucial Cost & Latency Comparison**
   - **Current implementation:** Packs **`entities_per_call = 150`** unique entities into one generative LLM prompt (`gpt-5.4-nano`, `filter_prompt.jinja2`) and asks the LLM to echo back matching entities verbatim as JSON.
   - **Naive 1-item-per-call decision strategy vs. Inverted Fan-Out:**

| Strategy (10,000 short entities, ~8 tok/entity, ~60-tok condition, 30% match rate) | HTTP Calls | Total Input Tokens | Output Tokens | Est. API Cost | Est. Wall Time (`n_parallels=50`) | Statistical / Quality Properties |
|---|---|---|---|---|---|---|
| **Current `LLMClient` (`gpt-5.4-nano`, 150/call)** | 67 | ~130k | ~30k | ~$0.03–$0.06 *(model-dependent)* | ~5–15 s | Suffers "lost in the middle" recall drops & verbatim spelling drift; binary hit/miss only (needs `n_runs > 1` for fractional `score`). |
| **Naive 1-item Decision (`clef-flash` @ $0.038/M)** | **10,000** | ~680k *(condition repeated 10,000×)* | **0** | **$0.026** | ~8–15 s *(40 ms × 200 waves; HTTP-bound)* | Zero cross-entity interference; exact calibrated `score` in 1 run; **hits RPM limits hard** (Gemini Tier 1 = 1,500 RPM → 6.7 min; AMD Token Factory = 20 RPM → infeasible!). |
| **Inverted Fan-Out Decision (`state = condition`, $K$ entities as parallel `noul` questions)** | **157** ($K=64$, Clef) or **20** ($K=500$, Gemini/Jev) | **~150k–200k** *(condition billed once per batch!)* | **0** | **$0.006–$0.020** | **~0.5–3 s** | **Best of all worlds:** Condition prefilled once; each entity judged in **strict isolation** (no lost-in-the-middle!); zero verbatim echo errors; exact calibrated `score` ($p \in [0,1]$) in 1 run; RPM-friendly! |

   - **Verdict for `filter`:** Implement **inverted fan-out** (`state = {"condition": cfg.condition, "additional_instructions": ...}`, `questions = {f"e_{i}": Noul(instructions={"entity": ent, "question": "Does `entity` meet `condition`?"})}`)—directly mirroring TypeSafe's official structured-instructions duplicate-matching pattern ([docs.typesafe.ai/primitives/noul](https://docs.typesafe.ai/primitives/noul))! Chunk `K` by provider limit (`min(entities_per_call, client.max_questions_per_call)`). Note that for long prose items (when `filter` is used on paragraphs rather than short entity names), cap `K` by token budget or fall back to 1-item-per-call so `per_request_context` is not exceeded.

5. **`rate` ([`tasks/rate.py`](../src/leximeter/tasks/rate.py)) — Viable with Psychometric Caveats**
   - **Current implementation:** Scores each item **0–100** per natural-language attribute (`ratings_prompt.jinja2`), instructing the LLM to use intermediate gradations (`absent → faint → moderate → abundant → extreme`).
   - **Can Decision models produce continuous 0–100 ratings?**
     - **YES, via `score` expectation rescaling:** Define $L$ ordered descriptive rubric levels ($L \in [5, 10]$; e.g., 5 default levels anchored to Leximeter's own gradation scale: *"0: Completely absent—no direct evidence of {attr}"*, *"1: Faint—minimal or passing direct signal"*, *"2: Moderate—clear, noticeable presence"*, *"3: Abundant—strong, prominent presence throughout"*, *"4: Extreme—overwhelming, defining focus of the text"*). Map the returned continuous expected value $\mathbb{E}[l] \in [0, L-1]$ to `score / (L - 1) * 100.0`.
     - **CRITICAL PITFALL 1 — Never pass bare numbers as `score` criteria:** TypeSafe's primary documentation ([score.md](https://docs.typesafe.ai/primitives/score)) explicitly demonstrates that each level is evaluated independently without seeing its index or neighbors—passing `["0", "1", "2"]` causes severe miscalibration compared to concrete textual descriptions.
     - **CRITICAL PITFALL 2 — Never use `noul × 100` for `rate`:** TypeSafe's [noul.md](https://docs.typesafe.ai/primitives/noul) explicitly proves that a `noul` probability measures *epistemic certainty* of a binary predicate, **not** intensity/degree (an unclear text and a moderately intense text both produce ~0.5).
     - **CRITICAL PITFALL 3 — Distribution & custom scale fidelity:**
       - When a decision model is highly confident in a single level (e.g., $p_2 = 0.99$), expected scores quantize near grid points (`0, 25, 50, 75, 100` for 5 levels; `0, 11.1, 22.2, ..., 100` for 10 levels), yielding coarser granularity than a generative LLM's 0–100 integers unless `L = 7..10` well-differentiated anchors are used (or `gemini-decision-1`'s larger level cap is tested empirically).
       - If the user sets a custom `RateConfig.rating_scale` (e.g., `-5 to +5` or a bespoke domain rubric), auto-synthesized 0–100 anchors will **not** honor free-form `rating_scale` prose unless `RateConfig` adds an explicit `rubric_levels: list[str] | dict[str, list[str]] | None` field (or falls back to `LLMClient` when free-form `rating_scale` is set!).
   - **Verdict for `rate`:** Support `DecisionClient` as an **opt-in** backend (especially valuable when paired with explicit rubric levels or for high-volume corpus rating / `debias` remaining-signal checks), while documenting the anchor-synthesis behavior and falling back to `LLMClient` (or warning) if an unstructured custom `rating_scale` string is provided.

---

#### Tier 3 — Hybrid Only / Non-Fits

6. **`codify` ([`tasks/codify.py`](../src/leximeter/tasks/codify.py)) — CRITICAL SPLIT**
   - **Primary snippet extraction (`_gather_iteration`):** `codify` requires **exact verbatim substring boundaries** (`beginning excerpt` + `ending excerpt`, located via `word_matching.py`) and supports **dynamic mode** (`categories=None`), where the model invents specific category labels (e.g., `"John 3:16"`).
     - **Pure Decision models (`Jev`, `Clef`, `gemini-decision-1`, `d3`, `gpt-6-luna-decisions`) CANNOT perform primary `codify` extraction.** They output zero text tokens and no character offsets. (Jev's `box` primitive is strictly 2D image coordinates.)
   - **Where Decision models DO accelerate `codify` cleanly:**
     1. **Completion loop (`_classify_remaining`, lines 844–916):** When `n_rounds > 1`, `codify` strips extracted snippets and invokes `Classify` to decide which `(row, category)` pairs still contain uncoded evidence. Running this completion check on `DecisionClient` is a **100% lossless drop-in win**.
     2. **Optional sparse-code chunk pre-screening (opt-in optimization):** Before calling the generative `LLMClient` on every `(chunk, category_batch)` pair, a cheap `DecisionClient` `noul` pass over each chunk can skip generative coding for categories with $P(\text{present}) < \tau$ (e.g., $\tau = 0.15$ for high recall). On corpora where most codes appear in <20% of chunks, this cuts generative LLM spend by 60–80% while keeping verbatim snippet precision intact.
   - **Separate future direction — Extraction System One models (`Cloudflare Clairic` / OpenRouter `/api/alpha/extractions`):** Clairic's `type: "list", items: "text"` returns exact `[start, end]` character `spans` into `state`. However, it is currently preview-only, executes one forward pass per text field (multiplying billed input tokens by the number of codes), cannot handle dynamic open-ended category discovery, and has not been validated on multi-sentence qualitative social-science coding rubrics. Treat extraction models as a distinct future `ExtractionClient` research track rather than conflating them with `DecisionClient`.

7. **Generative / Open-Ended Tasks (Non-Fits for Decision Models):**
   - `extract` (open-ended structured values & multi-entity expansion—potential future fit for *extraction* models like Clairic, not decision models), `deidentify`, `paraphrase` (generation), `seed`, `ideate`, `compare`, `bucket` (taxonomy generation), `poll` (persona expansion & open-ended survey text; closed-ended Likert poll questions *could* optionally map to `choice`/`score`), `whatever`.
   - *(Edge-case note: `deduplicate` and `merge` pairwise equivalence verification or `bucket` voting could theoretically use `noul`/`choice` in specialized workflows, but their existing embedding + batched-LLM architecture is already efficient.)*

---

## 5. Candidate Minimal Protocol Abstraction for Leximeter

Following [ADR-0001](../docs/adr/0001-protocol-based-llm-clients.md) (**protocol-based clients, not vendor adapters**) and [ADR-0002](../docs/adr/0002-text-only-scope-and-pluggable-providers.md) (**optional injectable providers with graceful fallback + offline `Dummy*` testability**):

### 5.1 Core Interface (`src/leximeter/client/decision.py`)

Naming note: **`DecisionClient`** (or **`ClassifierClient`**—recommend `DecisionClient` or `ClassifierClient`; `DecisionClient` mirrors the industry protocol and cleanly covers `noul` classification, `choice` ranking/differentiation, and `score` rating without implying fixed-label BERT classifiers):

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

# ── Typed question primitives (text-only per ADR-0002) ─────────────────

@dataclass(frozen=True)
class NoulQuestion:
    instructions: str | dict[str, Any]
    true_criteria: str | None = None
    false_criteria: str | None = None


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str | dict[str, Any]
    criteria: dict[str, str | None]  # {option_id: description}


@dataclass(frozen=True)
class ScoreQuestion:
    instructions: str | dict[str, Any]
    criteria: list[str]  # ordered lowest -> highest (0-indexed)


DecisionQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion

# ── Typed answers (normalized across all providers) ────────────────────

@dataclass(frozen=True)
class NoulAnswer:
    probability: float                 # P(yes) in [0, 1]
    confidence: float | None = None    # |2p - 1| or provider value
    refused: bool = False


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str | None                 # None if refused
    probabilities: dict[str, float]
    confidence: float | None = None
    refused: bool = False


@dataclass(frozen=True)
class ScoreAnswer:
    score: float | None                # E[level] in [0, len(criteria)-1]
    probabilities: dict[int, float]    # int-keyed level probs
    confidence: float | None = None
    refused: bool = False


DecisionAnswer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class DecisionResult:
    answers: dict[str, DecisionAnswer]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class DecisionClient(ABC):
    """Minimal protocol for System One / decision-model providers."""

    max_questions_per_call: int = 64  # safe cross-provider default; override per protocol

    @abstractmethod
    async def decide(
        self,
        state: str | dict[str, Any],
        questions: dict[str, DecisionQuestion],
        *,
        model: str,
        **kwargs: Any,
    ) -> DecisionResult:
        """Evaluate ``questions`` in parallel against ``state``."""
        raise NotImplementedError
```

### 5.2 Protocol Implementations (Only 1 Needed for MVP!)

| Client Class | Protocol Name (`LEXIMETER_DECISION_PROTOCOL`) | Wire Endpoint | Providers Covered | Notes |
|---|---|---|---|---|
| **`SystemOneClient`** *(MVP — implement first!)* | `"systemone"` | `POST {base_url}/v1/systemone` *(or custom path e.g. `/api/alpha/decisions`)* | **TypeSafe Jev**, **OpenRouter (all 44 models incl. Jev, Clef, Gemini, GPT-6 Luna)**, **AMD Radeon Cloud (`d3`, `d3-flash`, `Clef`)**, **self-hosted vLLM (`vllm serve`)**, **SGLang (`Clef`)**, **`vllm-sr`** | Uses plain `httpx.AsyncClient` (already a transitive dep of `openai`). Automatically handles `noul` with/without `criteria` (configurable `send_noul_criteria: bool`), `refusal` answers, optional `TypeSafe-Danger-Zone-Long-Context` header, and `usage.cost`. |
| **`WorkersAIDecisionClient`** *(optional)* | `"cloudflare"` | `POST https://api.cloudflare.com/client/v4/accounts/{id}/ai/run/@cf/cloudflare/{model}` | **Cloudflare Workers AI** direct (`clef`, `clef-flash`, `clef-omni`) | Identical SystemOne request payload; unwraps Cloudflare's `{result: ...}` REST envelope and sets `max_questions_per_call = 64`. |
| **`GeminiDecisionsClient`** *(optional)* | `"gemini_decisions"` | `POST https://generativelanguage.googleapis.com/v1beta/decisions` | **Google AI Studio** direct (`gemini-decision-1`) | Translates `state` → `input` (with `json.dumps` if dict), `true_criteria`/`false_criteria` → `true_description`/`false_description`, `x-goog-api-key` auth, `max_questions_per_call = 1000`. |
| **`OpenAIDecisionsClient`** *(optional)* | `"openai_decisions"` | `POST https://api.openai.com/v1/decisions` | **OpenAI** direct (`gpt-6-luna`) | Translates `questions` dict → list with `name`, `noul` → `predicate`, `max_questions_per_call = 400`. |
| **`DummyDecisionClient`** *(MVP — required per ADR-0002)* | `"dummy"` / test injection | Offline in-memory | Zero-network test suite | Deterministic answers (e.g., `noul=0.5` or configurable mapping, first `choice`, midpoint `score`) matching `DummyClient` / `DummyEmbeddingClient` / `DummySearchTool`. |

### 5.3 Essential Normalization & Engineering Rules

1. **Key Sanitization Layer (Crucial):** User-supplied Leximeter labels/attributes frequently contain spaces, colons, punctuation, or non-ASCII characters (e.g., `"Circle mentions cost > $100"` or Chinese labels). Every `DecisionClient` implementation (or a shared helper in `decision.py`) should map caller keys to safe wire IDs (`q_0`, `q_1`, ...) before sending and map `answers` back to the caller's verbatim keys. This satisfies Cloudflare's `[A-Za-z0-9_.-]{1,100}` and Google's `[A-Za-z0-9_-]{1,64}` rules transparently—remembering that decision models never see question keys anyway (verified across TypeSafe, Google, and Cloudflare docs!).
2. **Option-Key Hygiene for `choice`:** Keep option IDs clean ASCII tokens (e.g., `"circle"`, `"square"`, `"draw"`, `"insufficient_signal"`, `"neither"`) and put human-readable definitions in the `criteria` values.
3. **Graceful Degradation (ADR-0002):** Add `decision_client: DecisionClient | None = None` to `Task.__init__` and `resolve_decision_client()` in `client/resolve.py` (`LEXIMETER_DECISION_PROVIDER` / `LEXIMETER_DECISION_PROTOCOL`, defaulting to `"none"` -> `None`). When `None`, tasks use the existing `LLMClient` path unchanged—100% backward-compatible.
4. **Separate Model Config:** Decision models have different model IDs from generative LLMs (e.g., `model="gpt-5.4-mini"` for generation vs `decision_model="cloudflare/clef-flash"` or `LEXIMETER_DECISION_MODEL`). Composite tasks (`discover`, `debias`, `codify`, `paraphrase`) MUST allow independent `model` (generative) and `decision_model` (decision) so they can pair e.g. a strong generative LLM with a $0.038/M decision model.
5. **Checkpointing (ADR-0003):** Add a `batch_decide(...)` companion in `engine/batch.py` (sharing the same semaphore concurrency, retry/backoff, `Retry-After` handling, and CSV checkpointing by deterministic `Identifier`, storing serialized `DecisionResult` JSON in the `Response` column) so interrupted classification/ranking runs resume identically with zero duplicate calls.
6. **Probability Preservation:** Consider returning raw probabilities/expected scores in disaggregated/raw CSV checkpoints (and optionally exposing `include_probabilities: bool = False` on `ClassifyConfig`)—calibrated continuous probabilities are a major scientific upgrade for social-science researchers running downstream regressions or ROC/threshold tuning.

---

## 6. Explicit Verification Gaps & Unverified Claims

To maintain strict primary-source rigor, the following items could **not** be confirmed from first-party documentation as of 2026-11-25:

1. **Google `gemini-decision-1` on Vertex AI:** Google's official [Decisions API docs](https://aistudio.google.com/docs/decisions-api) explicitly list *"Available on the Gemini Developer API (v1beta) with Gemini Decision 1 only."* No primary Vertex AI endpoint documentation or Vertex model ID was found; treat Vertex AI availability as unconfirmed/not yet GA.
2. **Official latency SLAs for Jev and `gemini-decision-1`:** Neither TypeSafe (`docs.typesafe.ai`) nor Google (`aistudio.google.com/docs/decisions-api`) publishes official latency benchmarks in their developer documentation. Latency numbers cited in this report for Jev (524 ms p50; 92 ms/task) and `gemini-decision-1` (~150–250 ms) come from Cloudflare's benchmark card, Artificial Analysis via `systemonemodels.org`, or third-party articles and should be benchmarked on Leximeter's actual payload sizes.
3. **AMD Token Factory per-token `pts` cost for `d3` / `d3-flash`:** Token Factory is a free development service metered in daily points (`pts`), and per-model point rates are served only via authenticated `GET https://developer.amd.com.cn/radeon/api/v1/models` (requiring an `rc-` key). Static docs do not list the exact `pts / 1M tokens` figures for `d3` or `d3-flash`.
4. **OpenAI Decisions API (`gpt-6-luna`) context window & option limits:** OpenAI's official docs do not publish a separate context limit or max choice count for `/v1/decisions` (though OpenRouter's live catalog lists `1,050,000` context for `openai/gpt-6-luna-decisions`).
5. **Empirical psychometric parity for `Rate` (0–100) and `Codify` via Clairic:** While the API mechanics are verified from primary schemas, whether 5–10-level `score` expectations reproduce GABRIEL/Leximeter 0–100 `Rate` correlations on social-science corpora—or whether Cloudflare `Clairic`'s `list` of `text` spans matches generative `Codify` recall—requires an empirical benchmark on representative Leximeter datasets before making either a default.

---

## 7. Primary Source Index

1. **TypeSafe AI Docs:**
   - Models & pricing: https://docs.typesafe.ai/models
   - System One OpenAPI 3.1.0 spec: https://docs.typesafe.ai/api/system-one
   - Quickstart: https://docs.typesafe.ai/introduction/quickstart
   - Primitives: https://docs.typesafe.ai/primitives/choice, https://docs.typesafe.ai/primitives/noul, https://docs.typesafe.ai/primitives/score, https://docs.typesafe.ai/primitives/box
   - Confidence formulas: https://docs.typesafe.ai/confidence
   - State guide: https://docs.typesafe.ai/concepts/state
2. **Cloudflare Workers AI & Hugging Face:**
   - Clef docs & schemas: https://developers.cloudflare.com/workers-ai/models/clef/ (`schema-input.json`, `schema-output.json`)
   - Clef-Flash docs & schemas: https://developers.cloudflare.com/workers-ai/models/clef-flash/
   - Clef-Omni docs: https://developers.cloudflare.com/workers-ai/models/clef-omni/
   - Clairic docs & schemas: https://developers.cloudflare.com/workers-ai/models/clairic/ (`schema-input.json`, `schema-output.json`)
   - Workers AI pricing: https://developers.cloudflare.com/workers-ai/platform/pricing/
   - Hugging Face model cards: https://huggingface.co/Cloudflare/clef, https://huggingface.co/Cloudflare/clef-flash, https://huggingface.co/Cloudflare/clef-omni
3. **Google Gemini Decisions API:**
   - Model page: https://aistudio.google.com/docs/models/gemini-decision-1
   - Decisions API guide: https://aistudio.google.com/docs/decisions-api
   - REST Reference (`POST /v1beta/decisions`): https://aistudio.google.com/api-reference/decisions/create-decision
   - Pricing: https://ai.google.dev/gemini-api/docs/pricing
   - Rate limits: https://aistudio.google.com/docs/rate-limits
4. **AMD Radeon Cloud / Token Factory & vLLM Semantic Router (`d3`):**
   - Radeon Cloud model docs: https://amd-aim.github.io/radeon-cloud-docs/models/d3/, https://amd-aim.github.io/radeon-cloud-docs/models/d3-flash/, https://amd-aim.github.io/radeon-cloud-docs/models/clef/
   - Token Factory guide & API docs: https://amd-aim.github.io/radeon-cloud-docs/guides/token-factory/, https://amd-aim.github.io/radeon-cloud-docs/api/overview/, https://amd-aim.github.io/radeon-cloud-docs/api/models/, https://amd-aim.github.io/radeon-cloud-docs/api/rate-limits/, https://amd-aim.github.io/radeon-cloud-docs/api/usage/
   - Hugging Face model cards: https://huggingface.co/vllm-sr/d3, https://huggingface.co/vllm-sr/d3-flash
   - vLLM Structured Decisions Models: https://docs.vllm.ai/en/latest/models/structured_decisions_models/
5. **OpenRouter:**
   - Live Decisions catalog: https://openrouter.ai/api/v1/models?output_modalities=decisions
   - OpenAPI specs: https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request, https://openrouter.ai/docs/api/api-reference/systemone/submit-a-system-one-request
   - Community guides: https://openrouter.ai/docs/guides/community/typesafe-sdk, https://openrouter.ai/docs/guides/community/multimodal-decisions
   - Cookbooks & SDK docs: https://openrouter.ai/docs/cookbook/evaluate-and-optimize/jev-classification, https://openrouter.ai/docs/client-sdks/python/sdks/decisions/README.md, https://openrouter.ai/docs/client-sdks/python/sdks/extractions/README.md, https://openrouter.ai/docs/client-sdks/python/sdks/rerank/README.md
6. **Broader Landscape & Benchmarks:**
   - vLLM Pooling Models (`/classify`, `/score`, `/rerank`): https://docs.vllm.ai/en/latest/models/pooling_models.html
   - Cohere Classify (Deprecated): https://docs.cohere.com/reference/classify
   - Hugging Face TEI: https://huggingface.co/docs/text-embeddings-inference/en/quick_tour
   - System One Model Directory & OpenAI Decisions API summary: https://systemonemodels.org/, https://systemonemodels.org/models/openai-decisions-api/, https://systemonemodels.org/models/jev/
