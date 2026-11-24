# Leximeter 工作小结（2026-08-16）

> 本次会话：embedding 解耦确认（model 传参检查）+ 13 个任务真实 provider 冒烟（除 rate 外全部）+ 修复 1 个测试隔离缺陷。本文是交接文档。

## 一、做了什么

1. **确认 embedding 调用无 model 传参**：`EmbeddingClient.embed(texts)` 接口与全部调用点（`deduplicate.py:107`、`merge.py:142`）都只传 texts；`model` 仅在构造期绑定（`OpenAIEmbeddingClient`/`LocalBgeClient` 的 `__init__`），解析层 `resolve_embedding_client(model=...)` 支持转发（env `LEXIMETER_EMBEDDING_MODEL`）；任务懒加载 `Task._get_embedding_client()` 无参 → 只走 env 或 `embedding_client=` 注入。这是解耦（commit `4fd43ea`）的设计意图：embedding 与 LLM 的 model 完全隔离。
2. **新建 `examples/smoke_tasks.py`**：真实 provider 冒烟脚本，argv 选任务（零参数=全跑）、单任务独立 try/except、末尾 PASS/FAIL 汇总、小样本控制成本。已含 13 个任务的断言（输出列名、dtype、覆盖率、行数等公开契约）。
3. **13 个任务真实 provider 冒烟全过**（DeepSeek `deepseek-v4-flash`，responses 协议，`.env` 配置）：

   | 批次 | 任务 | 结果 |
   |---|---|---|
   | 1 | classify、extract | ✅ |
   | 2 | compare、paraphrase、codify、bucket | ✅ |
   | 3 | rank（BT 拟合三列输出） | ✅ |
   | 4 | merge、deduplicate（**真实 embedding 路径**） | ✅ |
   | 5 | filter、deidentify、seed、whatever | ✅ |

   - merge：long 表 80 项（≥ `long_list_len=64` 门槛）走通 kmeans2 预筛 + 自动匹配，8/8 左键匹配。
   - deduplicate：500 唯一项（25 基础词 × 20 变体程序化生成），`group_size=100` → 5 簇，500 项归并为 286 个代表。
   - **`embedding_openai.py` 首次真实 API 实测**（SiliconFlow `BAAI/bge-m3`，`LEXIMETER_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1`）——memory 旧条目「未实测」解除。
4. **修复 `tests/test_resolve.py::test_resolve_embedding_openai_falls_back_to_llm_base_url`**：该测试假设环境无 embedding 专用变量，但项目 `.env`（import 时加载）提供 `LEXIMETER_EMBEDDING_BASE_URL`，导致回退断言失败（**既有缺陷**，与本次改动无关）。修复：`monkeypatch.delenv` 显式清除两个 embedding 变量。修复后 **62 个 pytest 全绿**。

## 二、冒烟中发现的问题

1. **seed 曾因网络瞬时断连失败一次**（`openai.APIConnectionError`：DeepSeek peer 中途关闭 chunked 连接），重跑即过。期间尝试过给 seed/deidentify 直接调用加 leximeter 层重试（`complete_with_retry`），**经用户审阅后决定回滚**（见下「退出保护决策」）。
2. **性能观察（非缺陷）**：DeepSeek `json_mode=True`（json_object）响应显著慢于普通模式（13–21s vs 0.8–2.2s 直连）。慢响应 × 双层重试（openai SDK 默认 600s 超时 + 2 次重试 + 当时的 leximeter 3 次重试）叠加时单次调用可达分钟级，seed 冒烟曾两次触发 300s+ 超时；复现确认是偶发慢响应而非挂起。

### 退出保护决策（用户拍板：字面回滚）

- **最终方案**：回滚全部重试相关改动——`engine/batch.py` 恢复私有 `_complete_with_retry`（仅 batch_execute 用）、`seed.py`/`deidentify.py` 恢复直接 `client.complete()`（SDK 默认 600s 超时 + 2 次重试保留）、`client/responses.py`/`chat.py` 恢复 SDK 默认构造。**保留**：seed 生成循环预算（`max_batches = max(2, ceil(num/entities_per_gen)+2)`，超预算退出并打印 `[Seed] Stopped after N batches...`）——这是唯一保留下来的退出保护。
- **决策依据**：事件 1 的根因是网络波动（SDK 层重试本身磨了 600s+ 后才失败）；用户认为低概率波动下"失败重跑"优于重试机制，且 SDK 默认重试保留。事件 2/3 的慢响应问题在此方案下仍可能发生（SDK 600s 超时），但属已接受的取舍。
- **测试**：`tests/test_engine.py` 只保留 `test_seed_generation_loop_has_budget` 1 例 → **63 个 pytest 全绿**（62 + 1 新增）。
2. 冒烟脚本自身 3 次断言错误（非库缺陷，已修正）：classify 有 `predicted_classes` 契约列、pandas 3.x 拒绝 dict 索引器、paraphrase 批准列名是 `text_revised_approved`。

## 三、embedding 传参确认的完整链路

```
任务调用 embed(texts)                  ← 无 model
  └─ embedding_client 由谁给？
       ├─ 显式注入 embedding_client=    ← API 层参数，绕过解析
       └─ Task._get_embedding_client()  ← resolve_embedding_client() 无参，只读 env
            └─ OpenAIEmbeddingClient(model=env 或默认 text-embedding-3-small)
               LocalBgeClient(model=env 或默认 BAAI/bge-m3)
```

注意：embedding 缓存 JSON 按「文本」做 key，换 embedding 模型不删 `save_dir` 会读到旧向量（GABRIEL 同契约）。

## 四、如何验证

```bash
cd leximeter

# 单元测试（零网络零密钥）
uv run pytest                          # 62 passed

# 真实 provider 冒烟（读 .env；复杂链路 discover/ideate/poll/debias 未含）
uv run python examples/smoke_tasks.py            # 全跑 13 个
uv run python examples/smoke_tasks.py rank       # 指定单个
```

## 五、复杂链路冒烟（2026-08-16 下午，全部完成）

> 用户决策：四个任务**逐个独立测试**（不批量）；debias 走**降本备选**（回归数值进单测 + 小样本冒烟）。

| 任务 | 模拟数据策略 | 结果 |
|---|---|---|
| ideate | `n_ideas=5` 零合成数据 | ✅ 5 想法×9 列（title/nutshell/paragraph/…） |
| poll | `num_personas=6` + 2 问题 | ✅ 6 persona×2 答案列（persona 传记质量高） |
| discover | 程序化 circle/square 对立文本各 10 条（科技乐观 vs 悲观）+ bucket 压缩轮数 | ✅ 3 buckets，net_pct [30, 100, 60] **方向性正确**（乐观类 100% actual / 0% inverted） |
| debias | **降本**：① 单测新增 `test_fit_ols_robust_matches_classical_coefficients` + `test_debias_twostep_numerics`（numpy/假 provider 已知真值：δ=-1.0 精确恢复、`y-s+δ·r` 公式反算、robust 系数一致性）→ **65 个 pytest 全绿**；② 冒烟 20 条程序化文本自带 raw 测量列（跳过 baseline），remaining signal 用 revenue 提及（剥离后仍有变异） | ✅ 全链路跑通，δ=0.011 |

**经验**：① persona 模板（750 词+30 字段）单次调用 ~63s，poll 全链路需 8-10 分钟——**复杂任务冒烟要用后台 + 长超时**，前台 420s 必误杀；② debias 若剥离属性与 remaining signal 属性相同，100% 剥离后代理无变异 → 跳过 stage-1 回归 → 无 debiased 列（**真实语义，非 bug**）——数据要设计成"剥离后剩余信号仍有变异"；③ 期间 seed/ideate/poll 各遇一次偶发慢响应，重试即过（与回滚决策一致：失败重跑）。

## 六、待办

1. ~~**复杂链路冒烟（discover / ideate / poll / debias）**~~：**已完成**（见上）。
2. ~~**seed/deidentify 直接调用无重试**~~：**已回滚为直接调用（用户拍板：字面回滚，失败重跑即可）**。已知取舍：SDK 默认 600s 超时 + 2 次重试保留，慢响应时单次调用仍可能磨分钟级；如后续要加固，选项为给 LLM client 设 `timeout` 或调低 SDK `max_retries`（未做）。
3. **【TICKET-001】复杂任务断点续接 + 已完成验证**：`research/ticket_001_checkpoint_resume.md`（OPEN）——discover/debias/poll/ideate 主流程无任务级 checkpoint；底层 batch_execute 有 Identifier 续跑但编排层没有；discover 有级联不确定性。方案 A（manifest 状态机）/ B（codify/compare 幂等化）/ C（最小：仅已完成验证）。
4. **`LocalBgeClient` 未真实加载验证**：本地 bge-m3 模型下载 + CPU 编码路径仅单测覆盖。
5. **真实 Tavily 冒烟**：`TAVILY_API_KEY` 未配，搜索工具只有假 client 单测。
