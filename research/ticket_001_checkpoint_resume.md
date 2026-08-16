# TICKET-001: 复杂任务断点续接 + 已完成验证（任务级 checkpoint）

- **状态**: DONE（2026-08-16 实施；含用户补充的"leximeter 层面 max retry 包装"需求）
- **优先级**: 中（影响长任务的可用性；当前"重跑可完成但会重做编排"，无数据丢失风险，纯效率与确定性损失）
- **标签**: checkpoint / resume / complex-tasks

## 实施记录（2026-08-16）

按方案 C→B 实施，并补充任务级 retry wrapper（用户要求）。**未做方案 A 的显式 stage 状态机**——分析发现 debias/poll/discover 的子任务（rate/classify/bucket/batch_execute）本身已有检查点，补上 C+B+seed 幂等 后阶段续跑由子任务检查点自动达成。

| 改动 | 文件 |
|---|---|
| 任务级 retry wrapper（重试强制 `reset_files=False` 从检查点续跑） | `src/leximeter/engine/retry.py`（新）|
| `{task}_state.json` manifest + `compute_input_signature` 共用件 | `src/leximeter/tasks/_common.py` |
| seed：**任务级** retry（api `max_retries` → wrapper 整轮重跑）+ 已完成验证 | `seed.py`、`api.py` |
| 已完成验证（零 API 重跑） | `ideate`/`poll`/`debias`（重建 DebiasResult）/`discover`（snapshot 重建）|
| 子任务幂等 | `codify`（coded_passages.csv + signature 门控，list 列 JSON 化落盘）、`compare`（content-hash 自验证短接）|
| api 层 `max_retries` | `api.py`：discover/ideate/poll/debias 任务级（默认 0）；seed 逐 batch（默认 3）|

**关键设计点**：
- **位置式 identifier 的 stale 检查点清除**：codify（`row{idx}_iter{iter}_chunk{ci}`）与 ideate（`idea-{i:05d}`）的 batch 检查点不编码输入内容——signature 失效时必须清除（ideate 删 `ideation_raw.csv`；codify 删 per-iteration CSVs + completion_checks/），否则"改输入 → 全量重跑"的验收不成立。
- **compare 自验证短接**：identifier 是 pair 内容 hash，检查点全 done 即等价于输入未变，无需 signature。
- **codify 落盘格式**：list/dict 列（snippets、coded_passages）改存 JSON 字符串（debias 的 `_collect_snippets` 本就按 JSON 解析），加载时 `json.loads` 还原，避免 CSV 往返丢类型。
- **code-review 修复**（Standards + Spec 双轴）：`engine/retry.py` 不导入 tasks 层（鸭子类型，保持 engine 只依赖 client 契约）；`_load_completed` 守卫收敛为共用 `_common.load_completed_csv`；signature 覆盖修正——debias 签**整个 df**（结果嵌所有输入列）、seed/codify 补 `model`；debias 重载 metadata 补齐 `metadata_path` 键且 rebuild 失败（TypeError/KeyError/ValueError）回退全量重跑；retry wrapper 不重试确定性错误（ValueError/TypeError）。
- **seed 重试设计修正（慢响应窗口复验后）**：初版给 seed 加了逐 batch `complete_with_retry`（leximeter 3 次 × SDK 2 次双层重试）——这正是 worksummary_0816「退出保护决策」拍板回滚的方案，慢响应窗口下失败请求（server 端已计费）被放大：耗时 ×N + token ×N。已字面回滚：seed 恢复直接 `client.complete()`（SDK 600s 超时 + 2 次重试保留），api `max_retries` 改为**任务级**语义（整轮重跑，与复杂链路一致；符合"失败重跑"哲学）。
- 验收标准 1-6 均有测试覆盖（`tests/test_checkpoint.py`，32 个用例全假 provider 零网络；含 discover/poll/codify 的 reset 路径、AC3 签名敏感性、discover 恢复全量一致性断言）。

## 问题描述

四个复杂任务（`discover` / `debias` / `poll` / `ideate`）**中断后重跑只能从主流程开头重新执行**——没有"任务已完成则直接加载结果返回"的验证逻辑，也没有"从上次中断的阶段恢复"的状态机。0816 冒烟期间 poll 两次 420s 超时杀进程后重跑，整个 seed→persona→answers 编排全部重做（好在底层 API 调用部分被跳过）。

**底层有续跑、编排层没有**——这是现状的核心矛盾：

| 层 | 续跑能力 | 说明 |
|---|---|---|
| `engine/batch_execute` | ✅ 有 | 按 Identifier 去重，CSV 检查点续跑（`_checkpoint.load_checkpoint` / `completed_identifiers`） |
| `AttributeTask`（rate/classify/extract/rank 等） | ✅ 有 | `{base}_raw_responses.csv` 检查点，Identifier 不变则跳过 API 调用 |
| `bucket` | ✅ 有（唯一完整的任务级续跑） | `bucket_state.json` 全程三阶段续跑；`terms_signature`（sha1）变化清空 state |
| **`discover` 主流程** | ❌ 无 | run 开头直接执行阶段 1（codify/compare）；`reset_files` 只转发给子任务 |
| **`debias` 主流程** | ❌ 无 | run 开头直接测量/剥离/回归，不检查 `debias_results.csv` 是否已存在 |
| **`poll` 主流程** | ❌ 无 | run 开头直接 seed→persona→answers；seed 阶段每次重跑（除非传 df） |
| **`ideate` 主流程** | ❌ 无 | 每次 batch_execute + 全量解析重做（API 调用可跳过） |

**级联不确定性**（比"重做编排"更严重的问题）：`discover` 重跑时，candidates 来自 codify/compare 的 LLM 输出——若输出略有变化，bucket 的 `terms_signature` 变化 → bucket state 清空 → 重新投票 → 最终 labels 可能与上次不同。**断点续跑被 LLM 非确定性破坏**，即使 `reset_files=False`。

## 目标

1. **已完成验证**：任务已成功完成过（结果文件存在且完整）→ 重跑（`reset_files=False`）直接加载并返回结果，不再触发任何 API 调用。
2. **阶段级断点续接**：中断后重跑（`reset_files=False`）→ 跳过已完成的阶段，从上次中断点继续；已完成阶段的中间产物（LLM 输出）**必须被复用**，不得重跑。
3. **确定性**：恢复路径与首次运行路径在相同输入下产生相同输出（identifier 契约不变：`{sha1_16}_batch{b}_run{r}` 等既有格式勿改，会破坏旧检查点）。

## 方案选项

### A. 任务级 manifest 状态机（推荐，对齐 GABRIEL 契约）

每个复杂任务在 `save_dir` 维护一个 `{task}_state.json`（仿 `bucket_state.json` 模式）：

```json
{
  "task": "debias",
  "stages": {
    "baseline_measurement": {"status": "done", "artifacts": ["debias_brand/original_raw_responses.csv"]},
    "codify_strip": {"status": "done", "artifacts": ["codify/coded_passages.csv"]},
    "stripped_measurement": {"status": "done", "artifacts": ["debias_brand/stripped_raw_responses.csv"]},
    "remaining_signal": {"status": "done", "artifacts": ["debias_brand/remaining_signal_raw_responses.csv"]},
    "regression": {"status": "done", "artifacts": ["debias_results.csv", "debias_metadata.json"]}
  },
  "input_signature": "sha1(df+params)",   // 输入或参数变化 → 全量重跑
  "completed_at": "ISO8601"
}
```

- run 开头：`input_signature` 一致且全部 stages done → 加载最终产物直接返回（零 API 调用）。
- 部分 done：按 stages 顺序恢复，跳过 done 阶段的子任务调用，复用其 artifacts。
- `reset_files=True` → 删除 state 全量重跑。
- 关键：**恢复时子任务必须读自己的中间产物**（如 codify 读 `coded_passages.csv`、rate 系读 raw csv）——这要求给 codify/compare 等补"run 开头读已有产物"逻辑（见 B 的共用件）。

### B. 子任务"产物已存在则加载"补全（A 的前置依赖）

`codify`（run 开头读 `coded_passages.csv` 且 input_signature 匹配则直接返回聚合结果）、`compare`（读 `comparison_responses.csv`）目前**落盘但不读**——补上后，主流程的"跳过阶段"才有意义（子任务自身幂等）。codify 内部 `_gather_iteration` 已有部分检查点逻辑（iteration 0 用 `reset_files`、后续轮 False），可先核实其现有缓存行为再补主流程层。

### C. 最小成本路径（先做验证，不做阶段恢复）

只做"已完成验证"（目标 1）：每个任务 run 开头检查最终产物（`debias_results.csv` / `poll_results.csv` / `ideation.csv` / `discover_results_snapshot.json`）+ `input_signature`，存在即加载返回。阶段恢复（目标 2）后续再补。**适合作为第一步 PR**。

## 验收标准

1. 跑 `discover`（pair 模式）到中断（如 kill 掉 bucket 阶段），重跑 `reset_files=False`：compare/codify 阶段的 API 调用数为 0（可用 CountingClient 断言），bucket 从 state 恢复，最终结果与首次运行一致。
2. 完整跑完 `debias` 后重跑 `reset_files=False`：**零 API 调用**（CountingClient 断言），返回结果与首次一致。
3. 改输入数据或参数后重跑：`input_signature` 变化 → 全量重跑（不误用旧结果）。
4. `reset_files=True`：state 清除，全量重跑。
5. 单元测试零网络零密钥（假 provider），覆盖：已完成验证、阶段恢复、signature 失效、reset 四种路径。
6. identifier/检查点格式契约不变（`{sha1_16}_batch{b}_run{r}`、`bucket_state.json`、`merge_groups_attempt*.json`）。

## 涉及文件

- `src/leximeter/tasks/discover/task.py`（主流程 + 子任务 state）
- `src/leximeter/tasks/debias/task.py`（主流程，阶段最多：baseline→strip→measure→regress）
- `src/leximeter/tasks/poll/task.py`（seed→persona→answers）
- `src/leximeter/tasks/ideate/task.py`（最简单，可做参考实现）
- `src/leximeter/tasks/codify.py`、`compare.py`（补"读已有产物"）
- 共用件候选：`src/leximeter/tasks/_common.py`（已有 `load_json_cache`/`write_json_cache`/`load_run_metadata` 等，可仿照加 `load_task_state`/`save_task_state`）
- 参考实现：`src/leximeter/tasks/bucket.py`（`bucket_state.json` + `terms_signature` 模式）
- 测试：`tests/test_tasks.py`（现有复杂任务测试附近）或新 `tests/test_checkpoint.py`

## 实施顺序建议

1. **C（最小）**：四个任务加"已完成验证 + input_signature"→ 测试（CountingClient 零调用断言）。
2. **B**：codify/compare 读已有产物（幂等化子任务）。
3. **A**：按 stages 恢复（debias 最复杂，先做；ideate 最简单可先做参考实现）。
4. 全量回归 + 真实 provider 小样本复验（注意复杂任务冒烟需后台 + 长超时，见 worksummary_0816）。

## 参考

- GABRIEL 的 checkpoint 契约：`research/gabriel-source-notes.md`（round CSV 数推断、`attributes.json`、`bucket_state.json` 等）
- 0816 冒烟经验：`research/worksummary_0816.md`（poll 两次中断重跑、discover 级联不确定性的观察）
- spec.md「Backlog」无此项（新发现的缺口）
