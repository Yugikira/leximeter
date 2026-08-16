# Leximeter 冒烟小结：TICKET-001 checkpoint 版复验（2026-08-16）

> 本次会话：TICKET-001（复杂任务断点续接 + 已完成验证 + 任务级 max retry）实施后的真实 provider 复验冒烟。**带 `_checkpoint` 后缀，结果目录不覆盖旧的 `runs/smoke_*`**。

## 一、冒烟结果

| 任务 | 样本 | 结果 | 备注 |
|---|---|---|---|
| compare_checkpoint | 4 对 circle/square | ✅ PASS（21 行 attribute） | ~3 分钟（provider 慢窗口）；结果 `runs/smoke_compare_checkpoint` |
| discover_checkpoint | 10 对科技乐观/悲观文本，3 buckets | ✅ PASS（summary 3 labels） | **fresh 1350.4s（~22.5 分钟）/ resume 0.0s**；结果 `runs/smoke_discover_checkpoint` |
| poll_checkpoint | 6 personas + 2 问题 | ⏸ 未跑（用户取消） | 无结果目录（残留已清理） |

## 二、核心验证点：已完成验证（零 API 续接）

discover 第二次运行（`reset_files=False`）命中 `discover_state.json` + snapshot 重建：

```
[discover_checkpoint] first run (fresh): 1350.4s
[Discover] Found completed result in runs\smoke_discover_checkpoint\discover_results_snapshot.json; loading snapshot.
[discover_checkpoint] second run (resume): 0.0s — expected ~0s via completed-run verification
>>> [discover_checkpoint] PASS summary 3 labels; resume took 0.0s (vs 1350.4s fresh)
```

buckets / summary labels / net_pct 首次与恢复一致断言全过——**TICKET-001 验收标准 2 的真实 provider 复验达成**。

## 三、期间发现与修复

1. **seed 双层重试回滚（commit `e7d0e60`）**：初版给 seed 加了逐 batch leximeter 重试，慢响应窗口下叠加 SDK 重试导致耗时/token 成倍放大（正是 0816「退出保护决策」回滚的方案）。已字面回滚：seed 恢复直接 `client.complete()`，api `max_retries` 改为任务级（整轮重跑）。
2. **compare 耗时对照实验**（`bench_compare.py` + `leximeter_baseline` worktree @ 8abc9de）：同一样本交替 4 轮，HEAD 61.6s/59.3s vs BASE 23.2s/**348.2s**——无一致的 HEAD>BASE 模式（相邻两轮方向相反），证明耗时差异来自 **provider 响应时间剧烈波动**（23s~348s，同一请求类型），非 ticket 修改引入。慢响应下 SDK 层重试的重复计费在修改前代码上同样发生。
3. **冒烟脚本更新**（`examples/smoke_tasks.py`）：新增 `compare_checkpoint`/`discover_checkpoint`/`poll_checkpoint` 三个带后缀变体（新结果目录 + per-request `timeout=180` 兜底 + 双跑验证：fresh → resume 零调用一致性）。

## 四、待办

1. **poll_checkpoint**：未跑（用户取消）。后续如需：后台 + 长超时（慢窗口下预计 20-40 分钟），跑完第二次运行应 0.0s 返回。
2. **bench 工具清理决策**：`bench_compare.py`（untracked）+ `leximeter_baseline` worktree（8abc9de）保留用于复测；不需要时 `git worktree remove leximeter_baseline` + 删除脚本。
