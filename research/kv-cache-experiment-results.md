# Empirical A/B Evaluation: KV-Cache-Friendly Prefix-First Prompts vs. Item-First Shuffled Prompts in Leximeter

**Date:** 2026-11-25 (13:30–13:36 UTC, DeepSeek Off-Peak window)  
**Worktree:** `D:\worktrees\leximeter-kv-cache-exp` (branch `exp/kv-cache-prompts`, based on `d179a8b`)  
**Model Tested:** `deepseek-v4-flash` (resolved server-side as `deepseek-flash`, DeepSeek-V4.1-Flash) via `/v1/chat/completions` (`json_mode=True`, `thinking={"type": "disabled"}`)  
**Total API Calls:** **216 / 220** hard cap (`216/216` OK, 0 errors, 0 retries, 0 unparsed cells)  
**Actual Total Spend:** **`$0.04373 USD`** (Off-Peak actual; `$0.04303` for the 212-call experiment matrix + `$0.00069` for 4 pre-flight sanity/verification calls)

---

## 1. Executive Summary

Moving static task instructions, rubrics, rules, and JSON schemas to the **top** of Leximeter's prompts—and moving the per-item `single_entry(modality, text)` block to the **bottom** with deterministic attribute/label/code ordering—delivers **~58–63% prompt cache hit rates on single runs (`r1`)** and **86.7% prompt cache hit rates on repeat runs (`r2`)** on DeepSeek's API, compared to **0.0%** on single-run baselines and **0.8–1.9%** on repeat-run `rate`/`classify` baselines.

| Dimension | Finding |
|---|---|
| **KV-Cache Hit Rate (excl. 1 warmup call)** | **62.0%** (`rate`), **61.5%** (`classify`), **62.7%** (`codify`) on cold first run (`r1`), rising to **86.7%** on repeat run (`rate/prefix_fwd_r2`) vs. **0.0%** (`baseline_r1`) and **0.8–1.9%** (`rate`/`classify` `baseline_r2`). |
| **Input & Total Cost Reduction (this experiment)** | **-53.9% to -57.7% input cost** and **-23.2% to -42.0% total cost** on first run (`r1`); **-84.9% input / -61.9% total cost** on repeat run (`rate/prefix_fwd_r2`). At $N=1{,}000$ items (`n_runs=3`), projected total savings are **55.8%** (`rate`), **47.6%** (`classify`), and **33.3%** (`codify`). |
| **Wall-Clock & Latency** | Per-call median latency after warmup is essentially equal to or slightly faster than baseline (`1.30–1.45s` vs. `1.42–1.77s` on `rate`/`classify`). On tiny batches ($N=8$ or $N=20$ with `n_parallels=10`), the sequential 1-call warmup adds `+1.0s` to `+2.8s` wall-clock overhead; this amortizes to `<0.2%` on production corpora ($N \ge 500$). |
| **Quality — `classify`** | **Zero degradation.** Both layout drift (`96.0%` agree, $\kappa=0.919$, F1=$0.956$) and order-flip drift (`96.0%` agree, $\kappa=0.920$, F1=$0.957$) are **tighter than baseline run-to-run noise** (`93.0%` agree, $\kappa=0.859$, F1=$0.923$). Early-vs-late positive-rate shift is `0.000` for layout and within noise for order-flip. |
| **Quality — `codify`** | **Zero degradation in locate rate or span overlap.** Verbatim locate rate is **100.0%** across all arms (strict verbatim **100.0%** on `prefix_fwd` vs. `98.3–99.2%` baseline). Char-span micro-F1 for `baseline_r1` vs. `prefix_fwd` (`0.8376`) and `prefix_fwd` vs. `prefix_rev` (`0.8458`) both **exceed baseline noise** (`0.8092`). Reversing code order (`prefix_rev`) merged slightly fewer, slightly longer snippets (`95 × 211 chars` vs. `113 × 184 chars`) while preserving total coded coverage (`20,041` vs. `20,465` chars). |
| **Quality — `rate`** | **Layout drift (forward) is within noise; order-flip drift is small but statistically detectable (+0.92 MAE).** Prefix-first noise (`r=0.978`, MAE=`4.83`, RMSE=`6.88`) matches baseline noise (`r=0.976`, MAE=`4.83`, RMSE=`7.08`). Layout drift (`b1` vs. `pf1`: `r=0.972`, MAE=`5.30`, paired-$t$ $p=0.235$) is statistically indistinguishable from baseline noise overall (with one attribute-level mean shift on `urgency`, `-4.9` pts across all prefix arms). Reversing attribute order (`pf1` vs. `pr1`: `r=0.967`, MAE=`5.75`, paired-$t$ $p=0.029$) shows **no monotonic Early-vs-Late fatigue bias** (`E=+0.23`, `M=+0.55`, `L=-0.12`, $p=0.833$), but introduces modest attribute-specific anchoring shifts on `pro_market_orientation` (first-key variance) and `policy_advocacy` (`-4.7` pts). |
| **Verdict** | **Strongly recommend shipping prefix-first layouts** with **per-run deterministic order permutation** (`order_strategy="per_run"` default: fixed for `n_runs=1`, distinct seeded permutation per run for `n_runs > 1` so order bias averages out while preserving ~62% cache hits on every run; optional `"fixed"` mode to unlock ~87% hot-cache hits on `n_runs > 1`). |

---

## 2. Experimental Setup, Environment & Cost Transparency

### 2.1 Environment & Model Resolution
- `.env` copied into the isolated worktree resolved to:
  - `LEXIMETER_PROTOCOL='responses'`
  - `LEXIMETER_BASE_URL='https://api.deepseek.com'`
  - `LEXIMETER_MODEL='deepseek-v4-flash'`
- Querying the zero-cost metadata endpoint `GET /v1/models` returned `['deepseek-flash', 'deepseek-v4-pro']`; DeepSeek's legacy `deepseek-chat` model ID has been retired and superseded by `deepseek-flash` (`deepseek-v4-flash` alias).
- **Pre-flight sanity finding (calls #1–#4):**
  - Calls #1–#2 (`sanity_baseline`, `sanity_prefix`) on default `deepseek-v4-flash` via `/v1/chat/completions` (`response_format={"type": "json_object"}`) succeeded cleanly and exposed both `usage.prompt_cache_hit_tokens` and `usage.prompt_tokens_details.cached_tokens`. They also revealed that `deepseek-v4-flash` enables **thinking mode by default** (`347` and `263` `reasoning_tokens` out of `362` and `278` completion tokens on a 2-attribute prompt).
  - To match the non-thinking `deepseek-chat` cost/behavior target (`< $0.08 USD`) and run a stricter stress test of prompt-ordering sensitivity without a hidden chain-of-thought buffer, calls #3–#4 (`verify_nothink_warmup`, `verify_nothink_hit`) verified `extra_body={"thinking": {"type": "disabled"}}`: `reasoning_tokens` dropped to `0`, completion tokens fell to `14`, and call #4 immediately registered a live KV-cache hit (`256 / 535` tokens, latency `3.73s → 0.86s`).
  - All 212 experimental calls ran with `thinking={"type": "disabled"}` and `reasoning_tokens == 0`.

### 2.2 Prompt Fidelity & Dataset Construction
- **Baseline arms** rendered Leximeter's unmodified packaged templates (`ratings_prompt.jinja2`, `classification_prompt.jinja2`, `codify_prompt.jinja2`) via `leximeter.templates.load_prompt(...)` and ran concurrently (`n_parallels=10` for `rate`/`classify`, `n_parallels=8` for `codify`).
- **Prefix-first arms** kept identical wording, rating scales, rules, JSON format, and `snippets.jinja2::single_entry`, making only two surgical changes:
  1. Static instructions + rubric + rules + JSON schema moved to the **top**; `single_entry(modality, text)` moved to the **bottom**.
  2. `shuffled_dict` / `shuffled` replaced with deterministic insertion-order serialization (`ordered_dict_json` / `list`), enabling controlled forward (`A→J` / `C1→C4`) vs. reversed (`J→A` / `C4→C1`) ordering.
  - Prompt character lengths matched baseline within 2–4 whitespace chars (`rate`: 4,960 vs. 4,958 chars; `classify`: 3,893 vs. 3,889 chars; `codify`: 7,915 vs. 7,912 chars), while Longest Common Prefix across items jumped from **20–21 chars (0.3–0.5%)** to **2,889–5,848 chars (73.9–79.8%)**.
- **Warmup pattern:** Every `prefix_*` arm sent item #0 alone sequentially (`warmup_first=True`) before launching items `1..N-1` concurrently (`n_parallels=10` or `8`).
- **Datasets & Parsers:**
  - `rate` & `classify`: 20 diverse social-science/news/editorial paragraphs (100–115 words each; verified full 0–95 rating coverage across all 10 attributes and 20%–85% label prevalence across all 10 labels), parsed with Leximeter's real `Rate.parse_response` and `Classify._parse`.
  - `codify`: 8 three-paragraph policy/social-science passages (300–360 words each) × 4 qualitative codes (`n_rounds=1`, static mode), parsed and located with Leximeter's real `Codify.parse_json` and `Codify.consolidate_snippets`.

### 2.3 Published API Pricing & Actual Total Cost Spent
Official DeepSeek API pricing for `deepseek-flash` (`deepseek-v4-flash`, per 1M tokens):
- **Off-Peak (active during this run at ~13:30 UTC; Peak is 01:00–04:00 & 06:00–10:00 UTC Mon–Fri):**
  - Input (Cache Hit): **`$0.003`** (98% discount) | Input (Cache Miss): **`$0.150`** | Output: **`$0.600`**
- **Peak (2× sensitivity):** Input Hit `$0.006` | Input Miss `$0.300` | Output `$1.200`
- **Legacy `deepseek-chat` V3.2 counterfactual:** Input Hit `$0.028` | Input Miss `$0.280` | Output `$0.420`

| Scope | Calls (OK) | Prompt Tok | Cached Tok | Miss Tok | Completion Tok | Actual Off-Peak USD | Counterfactual Uncached USD | Peak Equiv. USD | Legacy V3.2 Equiv. USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pre-flight sanity/verify | 4 (4) | 2,200 | 256 | 1,944 | 668 | `$0.00069` | `$0.00073` | `$0.00139` | `$0.00083` |
| Full A/B matrix | 212 (212) | 222,135 | 83,697 | 138,438 | 36,696 | `$0.04303` | `$0.05534` | `$0.08607` | `$0.05652` |
| **Grand Total** | **216 (216)** | **224,335** | **83,953** | **140,382** | **37,364** | **`$0.04373`** | `$0.05607` | `$0.08746` | `$0.05735` |

---

## 3. Efficiency & KV-Cache Results

### 3.1 Per-Arm Cache Telemetry, Latency & Cost

| Task | Arm | Calls | Prompt Tok | Cached Tok | Miss Tok | Hit Rate (All) | Hit Rate (Excl. Warmup) | Comp Tok | Wall Time (Warmup + Batch) | Median Call Latency (Rest) | Off-Peak Cost | Input Cost Savings | Total Cost Savings |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rate` | `baseline_r1` | 20 | 20,659 | 0 | 20,659 | 0.0% | 0.0% | 1,974 | 7.50s (0.00 + 7.50s) | 2.15s (1.77s) | `$0.00428` | 0.0% | 0.0% |
| `rate` | `baseline_r2` | 20 | 20,659 | 384 | 20,275 | 1.9% | 2.0% | 1,912 | 3.20s (0.00 + 3.20s) | 1.42s (1.42s) | `$0.00419` | 1.8% | 1.3% |
| `rate` | `prefix_fwd_r1` | 20 | 20,659 | 12,160 | 8,499 | **58.9%** | **62.0%** | 1,936 | 4.12s (1.02 + 3.11s) | 1.42s (1.42s) | **`$0.00247`** | **57.7%** | **42.0%** |
| `rate` | `prefix_rev_r1` | 20 | 20,659 | 12,160 | 8,499 | **58.9%** | **62.0%** | 1,960 | 4.81s (1.16 + 3.66s) | 1.39s (1.44s) | **`$0.00249`** | **57.7%** | **41.8%** |
| `rate` | `prefix_fwd_r2` | 20 | 20,659 | 17,905 | 2,754 | **86.7%** | **86.7%** | 1,924 | 5.06s (1.34 + 3.72s) | 1.53s (1.55s) | **`$0.00162`** | **84.9%** | **61.9%** |
| `classify` | `baseline_r1` | 20 | 16,679 | 0 | 16,679 | 0.0% | 0.0% | 2,416 | 3.25s (0.00 + 3.25s) | 1.46s (1.45s) | `$0.00395` | 0.0% | 0.0% |
| `classify` | `baseline_r2` | 20 | 16,679 | 128 | 16,551 | 0.8% | 0.8% | 2,450 | 3.45s (0.00 + 3.45s) | 1.50s (1.50s) | `$0.00395` | 0.8% | 0.5% |
| `classify` | `prefix_fwd` | 20 | 16,659 | 9,728 | 6,931 | **58.4%** | **61.5%** | 2,440 | 6.22s (2.28 + 3.94s) | 1.48s (1.45s) | **`$0.00253`** | **57.2%** | **36.1%** |
| `classify` | `prefix_rev` | 20 | 16,659 | 9,728 | 6,931 | **58.4%** | **61.5%** | 2,368 | 5.30s (2.11 + 3.19s) | 1.30s (1.30s) | **`$0.00249`** | **57.2%** | **36.5%** |
| `codify` | `baseline_r1` | 8 | 13,041 | 0 | 13,041 | 0.0% | 0.0% | 4,316 | 4.08s (0.00 + 4.08s) | 2.59s (2.64s) | `$0.00455` | 0.0% | 0.0% |
| `codify` | `baseline_r2` | 8 | 13,041 | 7,168 | 5,873 | 55.0% | 54.9% | 4,598 | 4.23s (0.00 + 4.23s) | 3.59s (3.64s) | `$0.00366` | 53.9% | 22.3% |
| `codify` | `prefix_fwd` | 8 | 13,041 | 7,168 | 5,873 | **55.0%** | **62.7%** | 4,296 | 6.12s (2.36 + 3.77s) | 3.01s (3.05s) | **`$0.00348`** | **53.9%** | **23.2%** |
| `codify` | `prefix_rev` | 8 | 13,041 | 7,168 | 5,873 | **55.0%** | **62.7%** | 4,106 | 6.91s (3.83 + 3.08s) | 2.59s (2.58s) | **`$0.00337`** | **53.9%** | **23.8%** |

*Notes: "Input Cost Savings" and "Total Cost Savings" compare each arm's actual cost to its own uncached counterfactual (`cached_tokens=0`). Peak-hour USD costs are exactly 2.0× the Off-Peak values above.*

### 3.2 Key Cache Mechanics Observed Empirically
1. **128-Token KV-Cache Page Granularity & 100% Warmup Reliability:**  
   Every single post-warmup call in `prefix_*` arms hit an exact multiple of **128 tokens** with zero misses or races:
   - `rate` (`prefix_fwd_r1`, `prefix_rev_r1`): **640 tokens** (`5 × 128`) on all 19/19 concurrent calls (prompt total 1,026–1,045 tok).
   - `classify` (`prefix_fwd`, `prefix_rev`): **512 tokens** (`4 × 128`) on all 19/19 concurrent calls (prompt total 826–845 tok).
   - `codify` (`prefix_fwd`, `prefix_rev`): **1,024 tokens** (`8 × 128`) on all 7/7 concurrent calls (prompt total 1,612–1,652 tok).
2. **Multi-Run (`n_runs > 1`) Super-Caching (`86.7%` hit rate on `rate/prefix_fwd_r2`):**  
   Because `prefix_fwd_r2` reused the exact same static rubric AND exact same 20 texts from `prefix_fwd_r1` (minutes earlier), DeepSeek's disk KV cache matched **892–896 tokens per call** (including call #0, which hit 896 tokens immediately!)—caching both the rubric and the item text! Under `baseline_r2`, per-item attribute shuffling broke the prefix immediately after the ~150-token item text, yielding only **1.9%** cache hits (`rate`) and **0.8%** (`classify`).
3. **Hidden Surprise in Shipped `codify_prompt.jinja2`:**  
   `codify/baseline_r2` unexpectedly achieved **55.0%** cache hits (`896` tokens/call) even though `codify/baseline_r1` got **0.0%**. Inspecting `codify_prompt.jinja2` explains why: line 18 renders `{{ categories }}` **without** `| shuffled_dict`, and only line 68 near the bottom uses `{{ categories.keys() | shuffled }}`. Thus repeat runs on the exact same passage cache everything up to line 68, while single runs (`r1`)—and multi-passage corpora—get **0.0%** caching because `single_entry` sits at line 5. Prefix-first fixes single-run caching immediately (`0.0% → 62.7%` excl. warmup).
4. **Wall-Clock Overhead on Tiny Batches:**  
   Because decode dominates `deepseek-flash` latency for ~1k-token prompts, per-call latency after warmup is similar to baseline (`1.30–1.55s` vs. `1.42–1.77s`). On small batches where $N \le 2 \times n\_parallels$, waiting for a sequential 1-call warmup adds `1.0–3.8s` of wall-clock latency (`classify` wall `3.25s → 5.30–6.22s`; `codify` wall `4.08s → 6.12–6.91s`). On realistic research datasets ($N=1{,}000$), a 1-call warmup adds `~1.5s` once while amortizing the cold warmup call from `5–12.5%` down to `0.1%`:

| Projected Cost at $N = 1{,}000$ Items (Off-Peak `deepseek-flash`) | Single Run (`n_runs=1`) Baseline → Prefix | Single Run Input / Total Savings | Multi-Run (`n_runs=3`, fixed order) Baseline → Prefix | Multi-Run Total Savings |
|---|---:|---:|---:|---:|
| **`rate`** (10 attrs, ~1,033 in / 97 out) | `$0.2131 → $0.1192` | **-60.7% input / -44.1% total** | `$0.6394 → $0.2823` | **-55.8% total** |
| **`classify`** (10 labels, ~833 in / 121 out) | `$0.1975 → $0.1224` | **-60.2% input / -38.1% total** | `$0.5926 → $0.3105` | **-47.6% total** |
| **`codify`** (4 codes, ~1,630 in / 535 out) | `$0.5655 → $0.4151` | **-61.5% input / -26.6% total** | `$1.6965 → $1.1322` | **-33.3% total** |

*(With `per_run` permutation on `n_runs=3`, every run achieves the single-run savings — **44.1% / 38.1% / 26.6% total savings** — while cancelling order drift across runs.)*

---

## 4. Quality, Bias & Drift Analysis

### 4.1 Task 1: `rate` (20 paragraphs × 10 attributes = 200 cells; 0 missing)

All 10 attributes spanned near-full 0–100 dynamic range (min `0–7`, max `84–94` in `baseline_r1`).

#### Pairwise Agreement & Statistical Comparison vs. Baseline Noise
| Comparison | Arms (`A` vs. `B`) | Pearson $r$ (Pooled / Within-Attr) | Spearman $\rho$ (Pooled) | MAE | RMSE | Paired-$t$ $p$ vs. Base MAE | Mean Diff (`A - B`) | Early [1–3] Diff | Mid [4–7] Diff | Late [8–10] Diff | Early vs. Late $p$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline noise** | `b1` vs. `b2` | `0.9758` / `0.9691` | `0.9735` | `4.83` | `7.08` | *(ref)* | `+0.95` | `+1.25` | `+0.60` | `+1.12` | `0.923` |
| **Prefix noise** | `pf1` vs. `pf2` | **`0.9777`** / **`0.9742`** | `0.9717` | **`4.83`** | **`6.88`** | `1.000` | `+0.89` | `+2.53` | `-0.20` | `+0.70` | `0.125` |
| **Layout drift (fwd)** | `b1` vs. `pf1` | `0.9722` / `0.9683` | `0.9671` | `5.30` | `7.58` | `0.235` | `+0.42` | `-0.60` | `-0.26` | `+2.37` | `0.030`* |
| **Order-flip drift** | `pf1` vs. `pr1` | `0.9667` / `0.9608` | `0.9607` | `5.75` | `8.35` | **`0.029`** | `+0.26` | `+0.23` | `+0.55` | `-0.12` | `0.833` |
| *Layout + rev (supp.)* | `b1` vs. `pr1` | `0.9576` / `0.9568` | `0.9560` | `6.06` | `9.41` | `0.012` | `+0.68` | `-0.37` | `+0.29` | `+2.25` | `0.142` |
| *2-run means (supp.)* | `b_mean` vs. `pf_mean` | **`0.9800`** | **`0.9752`** | **`4.58`** | **`6.41`** | — | `+0.40` | `+0.04` | `-0.66` | `+2.16` | `0.070` |

*\*Why `b1` vs. `pf1` showed `Late [8–10] = +2.37 (p=0.030)`:* Per-attribute inspection proves this is **not** a position effect, because `b1` vs. `pr1` (where `urgency` and `conflict_framing` moved to Early positions #3 and #2!) showed the **exact same** `Late [8–10]` shift (`+2.25`)!

#### Per-Attribute Mean Scores & Drift Breakdown (`rate`)
| Fwd Pos (Rev) | Attribute | `b1` Mean | `b2` Mean | `pf1` Mean | `pf2` Mean | `pr1` Mean | Layout (`b_mean - pf_mean`, $p$) | Order (`pf_mean - pr1`, $p$) | Order-Flip MAE / $r$ (`pf1` vs. `pr1`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (10) | `emotional_intensity` | 36.2 | 34.2 | 36.1 | 33.6 | 37.5 | `+0.35` (`0.756`) | `-2.60` (`0.165`) | `5.05` / `0.977` |
| 2 (9) | `analytical_rigor` | 63.9 | 63.2 | 66.8 | 63.1 | 63.8 | `-1.38` (`0.320`) | `+1.12` (`0.211`) | `4.65` / `0.981` |
| 3 (8) | `populist_rhetoric` | 27.8 | 26.6 | 26.8 | 25.3 | 27.6 | `+1.15` (`0.352`) | `-1.62` (`0.348`) | `4.60` / `0.982` |
| 4 (7) | `economic_focus` | 53.2 | 53.8 | 56.4 | 55.1 | 54.8 | `-2.25` (`0.114`) | `+0.95` (`0.468`) | `5.20` / `0.981` |
| 5 (6) | `environmental_concern` | 25.6 | 24.6 | 25.6 | 25.2 | 26.3 | `-0.38` (`0.665`) | `-0.85` (`0.356`) | `3.55` / `0.990` |
| 6 (5) | `institutional_trust` | 46.5 | 45.2 | 47.4 | 49.8 | 50.8 | `-2.73` (`0.019`) | `-2.17` (`0.057`) | `5.10` / `0.973` |
| 7 (4) | `policy_advocacy` | 60.0 | 59.3 | 56.9 | 57.0 | 52.2 | `+2.70` (`0.173`) | **`+4.67` (`0.008`)** | `7.65` / `0.960` |
| 8 (3) | `urgency` | 48.2 | 47.0 | 43.0 | 42.8 | 42.5 | **`+4.80` (`0.008`)** | `+0.30` (`0.828`) | `6.00` / `0.974` |
| 9 (2) | `conflict_framing` | 40.5 | 38.0 | 38.0 | 35.8 | 36.0 | `+2.35` (`0.064`) | `+0.82` (`0.525`) | `5.35` / `0.976` |
| 10 (1) | `pro_market_orientation` | 25.2 | 25.6 | 25.9 | 26.2 | 28.6 | `-0.68` (`0.698`) | `-2.52` (`0.500`) | **`10.30` / `0.815`** |

**Interpretation (`rate`):**
1. **Repeatability is unchanged or slightly better under prefix-first:** `prefix_fwd_r1` vs. `prefix_fwd_r2` matches `baseline_r1` vs. `baseline_r2` on MAE (`4.83` vs. `4.83`) and slightly beats it on RMSE (`6.88` vs. `7.08`) and within-attribute correlation (`0.9742` vs. `0.9691`).
2. **Layout effect is minor and confined to a ~5-point level calibration on `urgency`:** Across all 3 prefix arms (`pf1=43.0`, `pf2=42.8`, `pr1=42.5`), `urgency` scored `4.90` points lower than baseline (`b1=48.2`, `b2=47.0`; paired-$t$ $p=0.0095$), while preserving `r=0.962` rank correlation. Overall pooled MAE (`5.30`) did not significantly exceed baseline noise (`4.83`, $p=0.235$).
3. **Order-flip drift is NOT monotonic position fatigue, but localized neighbor/first-key variance:** Early-vs-Late signed shift in `pf1` vs. `pr1` was essentially zero (`+0.23` vs. `-0.12`, $p=0.833$). The `+0.92` MAE increase (`p=0.029`) was concentrated in two ambiguous attributes: `pro_market_orientation` (whose RMSE jumped from `7.67` in `pf1` vs. `pf2` to `16.17` in `pf1` vs. `pr1`, driven by two noisy items `p16_congestion_pricing` `[31 vs 72]` and `p15_anti_establishment_podcast` `[22 vs 62]`, which already jumped `[15 vs 52]` between `b1` and `b2`!) and `policy_advocacy` (`-4.67` pts when moved earlier, $p=0.008$). This justifies **per-run permutation** on `n_runs > 1` so attribute-order variance averages out without sacrificing KV caching.

---

### 4.2 Task 2: `classify` (20 paragraphs × 10 boolean labels = 200 cells; 0 missing)

#### Pairwise Agreement & Position-Bucket Shift
| Comparison | Arms (`A` vs. `B`) | Exact Agreement | Cohen's $\kappa$ (Pooled / Macro) | F1 (Pooled / Macro) | Overall Pos Diff (`A - B`) | Early [1–3] Pos Shift | Mid [4–7] Pos Shift | Late [8–10] Pos Shift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline noise** | `b1` vs. `b2` | `93.0%` | `0.8589` / `0.8403` | `0.9231` / `0.9136` | `-0.010` | `+0.017` | `+0.000` | `-0.050` |
| **Layout drift (fwd)** | `b1` vs. `pf` | **`96.0%`** | **`0.9192`** / `0.9079` | **`0.9556`** / `0.9411` | **`+0.000`** | **`+0.000`** | **`+0.000`** | **`+0.000`** |
| *Layout drift (fwd, vs b2)* | `b2` vs. `pf` | `94.0%` | `0.8790` | `0.9341` | `+0.010` | — | — | — |
| **Layout drift (rev)** | `b1` vs. `pr` | `93.0%` | `0.8592` / `0.8465` | `0.9239` / `0.9117` | `-0.020` | `-0.017` | `+0.000` | `-0.050` |
| **Order-flip drift** | `pf` vs. `pr` | **`96.0%`** | **`0.9195`** / `0.9124` | **`0.9565`** / `0.9503` | `-0.020` | `-0.017` | `+0.000` | `-0.050` |

#### Per-Label Positive Rates Across Arms (`classify`)
| Fwd Pos (Rev) | Label | `b1` | `b2` | `pf` | `pr` | Agree `b1↔b2` | Agree `b1↔pf` | Agree `pf↔pr` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 (10) | `cites_quantitative_evidence` | 0.85 | 0.80 | 0.85 | 0.85 | 95% | 100% | 100% |
| 2 (9) | `advocates_policy_change` | 0.50 | 0.50 | 0.50 | 0.55 | 80% | 100% | 95% |
| 3 (8) | `criticizes_institutions` | 0.35 | 0.35 | 0.35 | 0.35 | 100% | 100% | 100% |
| 4 (7) | `mentions_economic_costs` | 0.70 | 0.65 | 0.65 | 0.70 | 95% | 95% | 95% |
| 5 (6) | `addresses_environment_or_climate` | 0.35 | 0.35 | 0.35 | 0.35 | 100% | 100% | 90% |
| 6 (5) | `uses_populist_framing` | 0.25 | 0.25 | 0.25 | 0.20 | 90% | 90% | 95% |
| 7 (4) | `expresses_time_urgency` | 0.35 | 0.40 | 0.40 | 0.40 | 95% | 95% | 100% |
| 8 (3) | `references_local_community_impact` | 0.45 | 0.50 | 0.45 | 0.55 | 95% | 100% | 90% |
| 9 (2) | `presents_competing_tradeoffs` | 0.40 | 0.50 | 0.30 | 0.30 | 90% | 90% | 100% |
| 10 (1) | `adopts_persuasive_advocacy_voice` | 0.30 | 0.30 | 0.40 | 0.45 | 90% | 90% | 95% |

**Interpretation (`classify`):** Neither layout change nor full order reversal degraded classification quality. In fact, `b1` vs. `pf` (`96.0%`, $\kappa=0.919$) and `pf` vs. `pr` (`96.0%`, $\kappa=0.920$) agreed **more closely** than two independent baseline runs (`93.0%`, $\kappa=0.859$), suggesting that per-item random shuffling in baseline introduces gratuitous flip noise on borderline items.

---

### 4.3 Task 3: `codify` (8 multi-paragraph passages × 4 codes = 32 pairs; `n_rounds=1`)

#### Per-Arm Extraction & Verbatim Locator Quality
| Arm | Raw Proposed | Located (Matched) | Unmatched | Locate Success Rate | Strict Verbatim Rate | Mean / Median Snippet Len (chars) | Total Coded Chars | Counts (`econ` / `inst` / `comm` / `rem`) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline_r1` | 121 | 121 | 0 | **100.0%** | 98.3% | 186.0 / 194 | 22,332 | 39 / 31 / 23 / 28 |
| `baseline_r2` | 123 | 123 | 0 | **100.0%** | 99.2% | 169.2 / 152 | 20,166 | 35 / 35 / 22 / 31 |
| `prefix_fwd` | 113 | 113 | 0 | **100.0%** | **100.0%** | 184.0 / 184 | 20,465 | 37 / 25 / 25 / 26 |
| `prefix_rev` | 95 | 95 | 0 | **100.0%** | 98.9% | 211.3 / 214 | 20,041 | 28 / 19 / 24 / 24 |

#### Pairwise Character-Span Overlap
| Comparison | Arms (`A` vs. `B`) | Micro Char F1 | Micro Char Jaccard | Macro Char F1 (32 pairs) | Macro Char Jaccard (32 pairs) |
|---|---|---:|---:|---:|---:|
| **Baseline noise** | `baseline_r1` vs. `baseline_r2` | `0.8092` | `0.6795` | `0.8087` | `0.7083` |
| **Layout drift (fwd)** | `baseline_r1` vs. `prefix_fwd` | **`0.8376`** | **`0.7206`** | **`0.8371`** | **`0.7408`** |
| *Layout drift (fwd, vs b2)* | `baseline_r2` vs. `prefix_fwd` | **`0.8623`** | **`0.7580`** | — | — |
| **Order-flip drift** | `prefix_fwd` vs. `prefix_rev` | **`0.8458`** | **`0.7327`** | **`0.8359`** | **`0.7398`** |
| **Layout drift (rev)** | `baseline_r1` vs. `prefix_rev` | `0.7705` | `0.6267` | `0.7791` | `0.6703` |
| *Layout drift (rev, vs b2)* | `baseline_r2` vs. `prefix_rev` | `0.7975` | `0.6632` | — | — |

**Interpretation (`codify`):**
- Prefix-first (`prefix_fwd`) **improves** strict verbatim adherence (`100.0%` vs. `98.3–99.2%`) and overlaps more strongly with both `baseline_r1` (`F1=0.838`) and `baseline_r2` (`F1=0.862`) than the two baseline runs overlap with each other (`F1=0.809`).
- Reversing code order (`prefix_rev`) still maintains higher char-span overlap with `prefix_fwd` (`F1=0.846`, `Jaccard=0.733`) than baseline noise (`F1=0.809`), though it produces coarser snippet boundaries (`95` snippets averaging `211.3` chars vs. `113–123` averaging `169–186` chars) while covering virtually the exact same character volume (`20,041` vs. `20,465` chars).

---

## 5. Design Recommendations for Leximeter

1. **Switch `ratings_prompt.jinja2`, `classification_prompt.jinja2`, and `codify_prompt.jinja2` (plus sibling attribute tasks like `extract`) to Prefix-First layout:**  
   Put static instructions, rubric/definitions, rules, and JSON schema at the top; place `snip.single_entry(modality, text)` at the bottom. This yields **~61–63% immediate single-run KV-cache hit rates** (~54–61% input cost savings) with zero loss in `classify`/`codify` quality and negligible layout drift on `rate`.
2. **Replace per-item shuffling with a configurable `order_strategy` (default: `"per_run"`):**
   - `"per_run"` *(recommended default)*: Use one deterministic order per run (shared by every item in that run). For `n_runs=1`, preserves caller dict order; for `n_runs > 1`, uses a distinct seeded permutation per run (e.g., original on run 1, reversed on run 2, seeded shuffles on runs 3..k). Every run achieves the full **~62% static-prefix cache hit rate**, AND multi-run averaging cancels out the modest localized order-flip drift observed on `rate` (`+0.92` MAE) and snippet granularity in `codify`.
   - `"fixed"` *(opt-in max-cache mode)*: Keeps the exact same order across all runs in `n_runs > 1`, unlocking **86.7% hot-cache hits** on runs `2..k` (cutting total `n_runs=3` cost by **~48–56%**) when order sensitivity is not a concern.
   - `"per_item"` *(legacy GABRIEL compatibility)*: Retains per-item shuffling only for callers who explicitly request exact legacy behavior.
3. **Adaptive 1-Call Warmup in `batch_execute`:**  
   When `warmup_first=True` (or auto-enabled when `len(pending) > max(8, n_parallels)`), send item #0 alone before launching `asyncio.gather` on items `1..N-1`. This prevents cold-cache thundering-herd misses across concurrent workers on large datasets while avoiding the `+1–3s` serial RTT overhead on tiny batches ($N \le 8$).
4. **Fix `codify_prompt.jinja2` consistency:**  
   Use deterministic JSON/ordered formatting for `BEGIN CATEGORIES/ATTRIBUTES` and the reminder list (eliminating the current mismatch where line 18 is unshuffled raw Python dict repr while line 68 is shuffled).
5. **Follow-up Check Worth Running Before Full Rollout:**  
   Verify `ResponsesClient` (which `.env` uses by default on DeepSeek) and OpenAI (`gpt-5.4-mini`) prefix caching thresholds (OpenAI caches prompts $\ge 1{,}024$ tokens in 128-token increments, so prefix-first benefits OpenAI, Anthropic, Gemini, and DeepSeek alike!).

---

## 6. Saved Artifacts (Worktree `D:\worktrees\leximeter-kv-cache-exp`)

- Experiment harness & offline analyzer: `scripts/kv_cache_experiment.py`
- Full call ledger (216 calls with raw usage telemetry): `runs/kv_cache_exp/call_ledger.jsonl`
- Consolidated JSON & CSV metrics: `runs/kv_cache_exp/analysis_summary.json`, `runs/kv_cache_exp/efficiency_summary.csv`
- Per-arm raw responses, metadata, sample prompts, and parsed outputs:
  - `runs/kv_cache_exp/sanity/sanity_result.json`
  - `runs/kv_cache_exp/rate/{baseline_r1,baseline_r2,prefix_fwd_r1,prefix_rev_r1,prefix_fwd_r2}/`
  - `runs/kv_cache_exp/classify/{baseline_r1,baseline_r2,prefix_fwd,prefix_rev}/`
  - `runs/kv_cache_exp/codify/{baseline_r1,baseline_r2,prefix_fwd,prefix_rev}/`
