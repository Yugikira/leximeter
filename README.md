# Leximeter

Leximeter 是一个面向纯文本、与模型 Provider 解耦的 LLM 测量库。它将访谈、演讲、评论、档案等非结构化定性文本转化为可分析的结构化数据。

本项目 fork 自 [openai/GABRIEL](https://github.com/openai/GABRIEL)，完整保留其全部文本测量、清洗与生成任务，并重构了内部实现：

- **模块解耦。** 原项目 6900 行的 `openai_utils.py` 已拆分为协议层 `client/` 与通用编排层 `engine/`，并通过 `Task` 与 `AttributeTask` 基类复用公共流程。
- **Provider 中立。** `LLMClient` 通过 `ResponsesClient` 与 `ChatCompletionsClient` 对接 OpenAI、DeepSeek、MiniMax、Ollama、vLLM 等兼容接口。`DecisionClient` 通过 `SystemOneClient` 实现 `POST /v1/systemone` 协议，支持 TypeSafe Jev、OpenRouter 托管的 44 款决策模型、AMD Radeon Token Factory 与自托管 vLLM 或 SGLang。文本嵌入与联网搜索作为独立的可选 Provider 单独配置。
- **前缀缓存友好。** 全部 14 份提示词模板都将静态指令、量表定义与输出格式放在前缀，把动态文本放在末尾。配合默认的 `order_strategy="per_run"` 与 `batch_execute` 自适应首条预热，DeepSeek 实测缓存命中率达到 61.5% 到 86.7%，总费用下降 23% 到 62%，测量质量与上游 GABRIEL 保持一致。详见 [research/kv-cache-experiment-results.md](research/kv-cache-experiment-results.md)。
- **离线可测试。** 每个 Provider 都支持注入假实现，例如 `DummyClient` 与 `DummyDecisionClient`。全套离线测试无需 API 密钥或网络连接即可运行。

更多文档如下：

- [leximeter_wiki/index.html](leximeter_wiki/index.html) 面向研究者的使用指南，包含示例代码、运行结果、算法讲解与参数说明。
- [GLOSSARY.md](GLOSSARY.md) 领域术语表。
- [docs/adr/](docs/adr/) 架构决策记录。
- [research/spec.md](research/spec.md) 设计规范。
- [research/classifier-models-research.md](research/classifier-models-research.md) 与 [research/kv-cache-experiment-results.md](research/kv-cache-experiment-results.md) 决策模型调研与前缀缓存实验报告。
- [research/gabriel-source-notes.md](research/gabriel-source-notes.md) GABRIEL 源码移植笔记。

## 安装

推荐使用 `uv` 安装依赖，也可以使用 `pip`：

```bash
# 使用 uv 安装核心依赖
uv sync

# 或使用 pip 可编辑安装
pip install -e .

# 按需安装可选依赖
pip install -e ".[embeddings]"   # 本地 BGE 文本嵌入
pip install -e ".[search]"       # Tavily 联网搜索
```

## 配置 Provider

所有生成任务底层都通过 `LLMClient` 调用模型。可以在项目根目录复制 [.env.example](.env.example) 为 `.env` 后按需填写，全部配置项均为可选。

导入 `leximeter` 时会自动调用 `load_dotenv()` 加载 `.env`，且不会覆盖已存在的系统环境变量。系统环境变量的优先级始终高于 `.env`。调用任何任务时，也可以通过 `client=`、`decision_client=`、`embedding_client=` 或 `search_tool=` 参数覆盖默认配置。

```bash
# LLM 协议与端点。protocol 可选 responses 或 chat，默认 responses。
LEXIMETER_PROTOCOL=chat
LEXIMETER_BASE_URL=https://api.deepseek.com
LEXIMETER_API_KEY=sk-...
LEXIMETER_MODEL=deepseek-chat

# 使用 OpenAI Responses API 时，仅配置以下密钥即可。
OPENAI_API_KEY=sk-...

# 文本嵌入 Provider，可选 local、openai 或 none。
# 设为 local 时默认使用本地 BGE 模型，需要安装 embeddings 依赖。
# 设为 none 时禁用嵌入，merge 与 deduplicate 自动回退到纯 LLM 路径。
LEXIMETER_EMBEDDING_PROVIDER=local
LEXIMETER_EMBEDDING_MODEL=BAAI/bge-m3
# 使用 openai 兼容嵌入服务时，可独立配置端点与密钥，例如 SiliconFlow。
# 留空时默认复用 LEXIMETER_BASE_URL、LEXIMETER_API_KEY 或 OPENAI_BASE_URL。
LEXIMETER_EMBEDDING_BASE_URL=
LEXIMETER_EMBEDDING_API_KEY=

# 联网搜索 Provider，可选 tavily 或 none，默认 none。
LEXIMETER_SEARCH_PROVIDER=none
TAVILY_API_KEY=tvly-...

# Decision Provider，可选 systemone、dummy 或 none，默认 none。
# 未启用时自动回退到 LLMClient。支持 TypeSafe Jev、OpenRouter、AMD Token Factory、vLLM 与 SGLang。
LEXIMETER_DECISION_PROTOCOL=none
LEXIMETER_DECISION_BASE_URL=https://openrouter.ai/api
LEXIMETER_DECISION_API_KEY=sk-or-...
LEXIMETER_DECISION_MODEL=cloudflare/clef-flash

# safest_json 解析失败时的兜底修复模型，默认 gpt-5.4-mini。
JSON_LLM_MODEL=gpt-5.4-mini
```

## 快速上手

以下示例按两个自然语言属性为一组菜品打分：

```python
import pandas as pd
import leximeter

toy_data = pd.DataFrame(
    {"entity": ["turkey", "pumpkin pie", "green bean casserole", "cornbread"]}
)
attributes = {
    "savory taste": "How savory the dish is",
    "sweet taste": "Dessert-like sweetness",
}

results = await leximeter.rate(
    toy_data,
    "entity",
    attributes=attributes,
    save_dir="toy_rate",
    modality="entity",
)
results.head()  # 每个属性输出一列 0 到 100 的评分
```

## 功能一览

| 函数 | 类别 | 说明 |
|---|---|---|
| `rate` | 测量 | 按自然语言属性对条目打 0 到 100 分 |
| `rank` | 测量 | 通过多轮两两比较拟合 Bradley-Terry 排序分数与标准误 |
| `classify` | 测量 | 判断条目是否符合指定布尔标签 |
| `extract` | 测量 | 从文本中抽取结构化事实，支持单条文本展开多个实体 |
| `discover` | 测量 | 从单列或成对文本中发现判别特征并归类汇总 |
| `merge` | 清洗 | 结合向量聚类与 LLM 模糊连接两张表 |
| `deduplicate` | 清洗 | 将语义等价的术语映射到同一代表词 |
| `filter` | 清洗 | 按自然语言条件筛选条目 |
| `deidentify` | 清洗 | 用一致的虚构信息替换文本中的个人敏感信息 |
| `seed` | 生成与辅助 | 按指令批量生成互不重复的实体列表 |
| `ideate` | 生成与辅助 | 围绕指定主题批量生成结构化构想报告 |
| `poll` | 生成与辅助 | 生成虚拟群体画像并开展问卷调查 |
| `paraphrase` | 生成与辅助 | 按指令改写文本，支持多轮递归校验 |
| `codify` | 生成与辅助 | 摘录文本中符合编码规则的原文片段，支持多轮补全 |
| `compare` | 生成与辅助 | 对比成对文本并列出差异或共同点 |
| `bucket` | 生成与辅助 | 通过多轮投票将候选术语归并为互斥类别桶 |
| `debias` | 生成与辅助 | 剥离混淆因素并执行两步去偏与稳健回归 |
| `whatever` | 生成与辅助 | 使用统一并发与检查点引擎运行自定义提示词 |
| `load` | 数据加载 | 将文件夹中的文本文件读取为 DataFrame |

## 任务参考

下文中的 `{base}` 表示 `file_name` 参数去掉 `.csv` 后缀后的文件名主干，所有输出文件与检查点都保存在 `save_dir` 目录下。所有任务默认 `reset_files=False`，中断后再次运行会自动从检查点续跑。

### 基础测量与清洗

这组任务直接对输入 DataFrame 的目标列逐条执行测量或变换，并在输出 DataFrame 中追加结果列：

- **`rate`。** 调用签名为 `rate(df, column_name, *, attributes, save_dir, n_runs=1, n_parallels=50, modality="text", ...)`。它为每个属性添加一列 0 到 100 的数值评分，多次运行时取各次 `run` 的均值。原始响应写入 `{base}_raw_responses.csv`，聚合结果写入 `{base}_cleaned.csv`，属性定义写入 `attributes.json`。当 `n_runs` 大于 1 时，按 `(id, run)` 展开的完整记录额外写入 `{base}_full_disaggregated.csv`。
- **`classify`。** 调用签名为 `classify(df, column_name=None, *, labels, save_dir, ...)`。它为每个标签添加一列布尔值，并附带汇总列 `predicted_classes`。当通过 Decision Provider 运行且 `include_probabilities=True` 时，结果额外包含 `{label}_prob` 概率列。清洗后的结果写入 `{base}_cleaned.csv`。
- **`filter`。** 调用签名为 `filter(df, column_name, *, condition, save_dir, ...)`。它按自然语言条件 `condition` 筛选条目，输出布尔列 `keep` 与得分列 `score`，结果写入 `{base}_cleaned.csv`。
- **`deidentify`。** 调用签名为 `deidentify(df, column_name, *, save_dir, grouping_column=None, mapping_column=None, ...)`。它将文本中的个人敏感信息替换为一致的虚构信息，并输出 `deidentified_text` 与 `mapping` 两列，结果写入 `{base}_cleaned.csv`。指定 `grouping_column` 时，同组文本共享同一份替换映射。指定 `mapping_column` 时，任务会沿用传入的既有映射。
- **`compare`。** 调用签名为 `compare(df, circle_column_name, square_column_name, *, save_dir, differentiate=True, ...)`。它对比成对文本的差异或共同点，返回长表格式的 DataFrame，包含 `circle`、`square`、`attribute` 与 `explanation` 四列。

### 合成生成

这组任务不依赖已有评分表，而是按指令批量生成实体、构想或虚拟受访者问卷结果，并在输出目录中保存 `{task}_state.json` 以支持完成态缓存与任务级重试：

- **`seed`。** 调用签名为 `seed(instructions, *, save_dir, num_entities=1000, ...)`。它生成 `num_entities` 个互不重复的实体，返回包含 `entity` 与 `entity_id` 两列的 DataFrame，默认写入 `seed_entities.csv`。
- **`ideate`。** 调用签名为 `ideate(topic, *, save_dir, n_ideas=1000, ...)`。它围绕指定主题生成 `n_ideas` 份结构化构想报告并解析字段，默认写入 `ideation.csv`。
- **`poll`。** 调用签名为 `poll(df=None, column_name=None, *, population_description, questions, save_dir, ...)`。它根据群体描述生成虚拟画像并逐题作答，也可以直接对传入的既有实体列开展调查，默认写入 `poll_results.csv`。

### `extract`

调用签名为 `extract(df, column_name, *, attributes, save_dir, types=None, ...)`。

该任务从每条文本中抽取结构化事实。如果模型在单条文本中识别出多个实体，任务会将每个实体展开为独立一行，输出 `entity_name` 列以及各属性列，并在控制台打印去重提示。多次运行时，同一属性取首个非 `"unknown"` 的有效值。

传入 `types` 参数后，任务会对对应列执行类型转换。例如 `int` 会四舍五入后转为 pandas `Int64`，`datetime` 会交由 `pd.to_datetime` 解析，并打印转换失败的条数。原始聚合结果写入 `{base}_cleaned.csv`，类型转换后的结果写入 `{base}_cleaned_coerced.csv`。

### `rank`

调用签名为 `rank(df, column_name, *, attributes, save_dir, n_rounds=5, matches_per_round=3, ...)`。

该任务通过多轮两两比较评估条目在各属性上的相对顺序，默认运行 5 轮，每轮每个条目参与 3 场对决。每轮结束后，任务会使用截至当前轮次的全部历史胜负记录重新拟合 Bradley-Terry 模型。

每个属性输出三列，其中 `{attr}` 为总体 z 分数，`{attr}_raw` 为居中后的 Bradley-Terry 估计值，`{attr}_se` 为标准误。每轮对决记录保存在 `{base}_round{rnd}.csv` 中，包含 `Batch`、`Pair`、`IdA` 与 `IdB` 列。断点续跑时，任务会根据已生成的轮次文件自动接续未完成的轮次。

### `bucket`

调用签名为 `bucket(df, column_name, *, save_dir, bucket_count=10, repeat_bucketing=5, repeat_voting=25, next_round_frac=0.25, top_k_per_round=1, raw_term_definitions=True, ...)`。

该任务将大量候选术语归并为 `bucket_count` 个互斥的类别桶，返回包含 `bucket` 与 `definition` 两列的 DataFrame，默认写入 `bucket_definitions.csv`。

任务按三个阶段依次执行：

1. 生成阶段重复调用模型产出多组候选类别桶方案，结果写入 `bucket_generation.csv`。
2. 归约阶段通过多轮投票按得票从高到低淘汰候选桶，每轮保留 `max(bucket_count, 当前候选数 * next_round_frac)` 个桶，投票记录写入 `vote_reduce{r}.csv`。
3. 终选阶段按每轮选出 `top_k_per_round` 个桶的方式补足最终名额，记录写入 `vote_final{i}.csv`。

全流程状态保存在 `bucket_state.json` 中，文件内记录了包含术语及其定义在内的 SHA-1 签名。输入术语或定义一旦变化，任务会自动清空旧状态并重新执行。

### `codify`

调用签名为 `codify(df, column_name, *, save_dir, categories=None, additional_instructions="", n_rounds=2, ...)`。

该任务从文本中摘录符合编码规则的原文片段，最终结果统一写入 `coded_passages.csv`。传入 `categories` 时进入静态模式，输出 DataFrame 为每个编码类别单独建一列，单元格内是摘录到的原文片段列表。不传 `categories` 而提供 `additional_instructions` 时进入动态模式，输出单个 `coded_passages` 列。

默认 `n_rounds=2`。当 `n_rounds` 大于 1 时，任务会启动多轮补全循环。每一轮先剥离已摘录的原文片段，再调用 `Classify` 判断剩余文本中是否仍有遗漏的可编码片段。随后任务仅对被标记存在遗漏的类别再次摘录并合并结果。每轮的遗漏检查结果写入 `completion_checks/{base}_completion_iter{i}.csv`，当轮新增摘录写入 `{base}_iter{i}.csv`。

### `paraphrase`

调用签名为 `paraphrase(df, column_name, *, instructions, save_dir, n_rounds=1, n_runs=1, later_round_run_multiplier=5, validation_attribute=None, ...)`。

该任务按 `instructions` 改写每条文本。输出列名为 `revised_column_name`，未指定时默认为 `{column_name}_revised`。当 `n_runs` 大于 1 时，结果按 `_1` 到 `_n` 后缀拆分为多列，每个改写列都附带对应的 `{col}_approved` 布尔列。最终结果写入 `{base}_cleaned.csv`。

当 `n_rounds` 大于 1 时，任务会启动多轮递归校验。每一轮先为尚未通过的条目生成多个改写候选。第 0 轮生成 `n_runs` 个候选，后续轮次生成 `n_runs * later_round_run_multiplier` 个候选，各轮候选结果保存在 `{base}_round{r}.csv` 中。接着任务调用 `Classify` 对比原文与改写稿，默认判断改写是否遵循了 `instructions`，也可以通过 `validation_attribute` 自定义校验标准。通过校验的条目采纳首个合格候选，未通过的条目进入下一轮重新生成。如果达到轮次上限仍未通过，任务会从最后一轮候选中随机选取一条，并将对应的 `_approved` 标记为 `False`。

### `discover`

调用签名为 `discover(df, *, column_name=None, circle_column_name=None, square_column_name=None, save_dir, bucket_count=10, max_retries=0, ...)`。

该任务自动挖掘文本中的判别特征并量化其频率，分为三个阶段。传入单个 `column_name` 时，第一阶段通过动态模式 `codify` 挖掘候选特征。传入 `circle_column_name` 与 `square_column_name` 时，第一阶段通过 `compare` 挖掘两列文本的差异特征。第二阶段调用 `bucket` 将候选特征归并为互斥类别桶。第三阶段调用 `classify` 将这些标签回标到数据上。

任务返回一个字典，包含 `candidates`、`buckets`、`classification`、`bucket_df`、`summary` 以及中间阶段结果 `compare` 或 `codify`。成对模式下，每个标签会自动扩展为 `{lab}_actual` 以及互换 `circle` 和 `square` 后的 `{lab}_inverted` 两个方向，`summary` 表包含 `label`、`actual_true`、`inverted_true`、`total`、`actual_pct`、`inverted_pct` 与 `net_pct` 七列。完整快照写入 `discover_results_snapshot.json`，同时将各张表打包为 `discover_results_export.zip`。`discover_state.json` 会记录输入签名，相同输入再次运行时直接加载快照，不发起任何 Provider 调用。

### `merge`

调用签名为 `merge(df_left, df_right, *, save_dir, on=None, left_on=None, right_on=None, how="left", short_list_len=16, long_list_len=256, use_embeddings=True, max_attempts=4, auto_match_threshold=0.75, ...)`。

该任务借助 LLM 与向量聚类对两张表执行模糊连接，默认按 `how="left"` 将右表合并到左表。

任务最多执行 `max_attempts` 轮匹配，默认 4 轮，每轮将短表侧的分组长度减半，依次为 16、8、4、2。默认 `long_list_len` 为 256，`auto_match_threshold` 为 0.75。启用文本嵌入且长表去重后的键数不少于 `long_list_len` 时，任务先用 `kmeans2` 对未匹配键聚类，再在簇内分段并构造候选窗口。如果某个键与唯一候选的余弦相似度达到 `auto_match_threshold`，任务直接完成自动匹配。未满足嵌入条件时，任务自动走纯 LLM 分批匹配。运行过程会保存 `merge_responses.csv`、每轮分组计划 `merge_groups_attempt{n}.json`、进度表 `merge_progress.csv` 以及嵌入缓存 `short_embeddings.json` 与 `long_embeddings.json`。

### `deduplicate`

调用签名为 `deduplicate(df, column_name, *, save_dir, group_size=500, use_embeddings=True, ...)`。

该任务将语义等价的术语映射到同一代表词，并在输出 DataFrame 中新增 `mapped_{column_name}` 列。默认 `group_size` 为 500。启用文本嵌入且去重后的术语数不少于 `group_size` 时，任务先用 `kmeans2` 将相近术语聚到同一分组再交由模型归并，否则直接按顺序分块。原始响应写入 `deduplicate_responses.csv`，嵌入缓存写入 `deduplicate_embeddings.json`，最终结果写入 `{base}_cleaned.csv`。

### `debias`

调用签名为 `debias(df, column_name, *, save_dir="debias", measurement_attribute, removal_attribute, attributes, signal_dictionary, removal_method="codify", remaining_signal=True, strip_percentages=None, robust_regression=True, random_seed=12345, run_name=None, max_retries=0, ...)`。

该任务衡量并剔除文本中混淆因素对目标属性测量的影响，返回包含 `results`、`metadata` 与 `regression` 三个字段的 `DebiasResult`。

流程包含四个步骤。首先对原始文本测量 `measurement_attribute`。接着按 `removal_method` 移除 `removal_attribute` 对应的混淆因素，可选 `codify` 剥离原文片段或 `paraphrase` 改写文本。选择 `codify` 时，任务依据 `random_seed` 确定性地按 `strip_percentages` 构造多个剥离比例的文本变体。随后任务重新测量各变体，并在 `remaining_signal=True` 时使用 `rate` 测量剥离后的残余信号。最后计算简单差值列 `{attr} (raw - stripped)` 与两步去偏列 `{attr} (debiased)`，计算公式为 `y - s + delta * r`。

回归诊断由 `fit_ols` 执行，默认使用 statsmodels 计算 HC3 稳健标准误，不可用时回退到 HC1 或 numpy。`robust_regression` 仅影响标准误估计，不改变回归系数。输出保存在 `save_dir/{run_name}` 目录下，未指定 `run_name` 时默认为 `save_dir/debias_{attr}`，包含 `debias_results.csv`、`debias_metadata.json` 与 `debias_state.json`。输入 DataFrame 与关键参数不变时，再次运行会直接从磁盘重建 `DebiasResult`，不发起任何 Provider 调用。

### `whatever`

调用签名为 `whatever(prompts=None, *, df=None, column_name=None, identifiers=None, save_dir="whatever", web_search=None, web_search_filters=None, search_context_size="medium", ...)`。

该任务使用统一的并发调度与检查点机制运行任意自定义提示词，既可以直接传入 `prompts` 列表，也可以传入 `df` 与 `column_name`。

传入 `web_search=True` 或提供 `web_search_filters` 时，任务会在调用模型前先通过检索工具拉取网页上下文并拼接到提示词头部。相关参数与 GABRIEL 对齐：

- `web_search` 默认为 `None`，此时根据是否传入 `web_search_filters` 自动推导。
- `web_search_filters` 中的 `allowed_domains` 与 `country` 会映射到 Tavily 的 `include_domains` 与 `country`。
- `search_context_size` 支持 `"low"`、`"medium"` 与 `"high"`，并兼容 `"small"` 与 `"large"` 别名。

启用联网检索需要配置 `LEXIMETER_SEARCH_PROVIDER=tavily` 与 `TAVILY_API_KEY`，或通过 `search_tool=` 注入自定义检索工具。未配置时任务会打印警告，并按原始提示词继续运行。

## 断点续跑与 `reset_files`

Leximeter 通过四层机制保障长任务可安全续跑且不复用过期结果：

- **逐请求检查点。** 所有任务都会按确定性标识符将原始响应写入检查点 CSV，标识符包含文本哈希以及批次、run 与轮次信息。中断后重新执行时，已完成的请求会自动跳过。
- **子阶段检查点自动失效。** 当模型名称、Provider 类型或提示词配置发生变化时，`classify`、`filter`、`rank`、`codify` 等子阶段检查点会自动失效并清理旧产物，防止误用旧配置生成的缓存。
- **任务级完成状态缓存。** `discover`、`debias`、`ideate`、`poll`、`seed`、`codify` 与 `bucket` 会在输出目录维护 `{task}_state.json`，记录输入数据与关键参数的 SHA-1 签名。相同输入再次运行时直接加载磁盘结果，不发起任何 Provider 调用。输入数据或模型等参数发生变化时，任务自动全量重跑。
- **任务级重试。** `discover`、`debias`、`ideate`、`poll` 与 `seed` 接受 `max_retries` 参数，默认值为 0。整个任务抛出异常时，重试包装器会自动将 `reset_files` 设为 `False` 并从检查点续跑，最多重试 `max_retries` 次。参数错误等确定性异常不会触发重试。单次请求的超时与重试由底层 SDK 负责，默认超时 600 秒并重试 2 次，任务层不叠加内层重试。

传入 `reset_files=True` 会清空当前任务的检查点文件与状态文件后重新执行。标识符格式遵循 GABRIEL 契约。如果从早期版本的 Leximeter 升级后遇到检查点不兼容，传入 `reset_files=True` 或删除旧的 `save_dir` 即可。

## Decision Provider

对于有界判断类任务，专用 Decision Provider 可以在单次前向传播中并行评估几十到上百个类型化问题，包括是非判断 `noul`、单选 `choice` 与评分 `score`，直接返回校准概率且不产生输出 token 费用。

`SystemOneClient` 实现了开放协议 `POST /v1/systemone`，自动完成非 ASCII 或含空格标签键的安全转义、遵循 `Retry-After` 的单层重试以及逐题拒答降级。支持的服务包括 TypeSafe Jev、OpenRouter、AMD Radeon Token Factory 以及自托管的 vLLM 和 SGLang。其中 OpenRouter 托管了 44 款决策模型，包括 Cloudflare Clef、Google `gemini-decision-1` 与 OpenAI `gpt-6-luna-decisions`，AMD Radeon Token Factory 提供 `d3` 与 `d3-flash`。

各任务对 Decision Provider 的使用分为两类路径：

- **纯 Decision 路径。** 传入 `decision_client=` 或配置 `LEXIMETER_DECISION_PROTOCOL=systemone` 后，下列任务完全不调用 `LLMClient`。未配置时自动回退到 `LLMClient`。
  - `classify` 在单列模式下走并行 `noul` 判断，传入 `include_probabilities=True` 时额外输出 `{label}_prob` 列，原始概率始终写入检查点。在成对模式 `differentiate=True` 下走三选一 `choice` 判断，将正反命题直接绑定到选项并随机化顺序，在结构上保证互斥，无需解析自然语言极性。
  - `rank` 的两两对决走四选一 `choice` 判断，选项对应 `circle`、`square`、`draw` 与 `insufficient_signal`，同时保留 `circle_first` 顺序随机化以及 Bradley-Terry 极大似然估计与标准误计算。
  - `filter` 采用反向扇出，将筛选条件作为状态上下文，把去重后的候选条目批量提交为并行 `noul` 问题，单轮即可得到校准后的 `score` 与 `keep`。
- **混合路径。** 生成阶段继续使用 `LLMClient` 与 `model`，内部的判断子阶段自动委派给 `DecisionClient` 与独立的 `decision_model`。采用混合路径的任务包括 `discover` 的第三阶段分类、`paraphrase` 多轮模式的递归校验、`codify` 多轮模式的遗漏检查，以及 `debias` 中 `mode="classify"` 或 `mode="rank"` 的测量阶段。

以下是通过 OpenRouter 与 AMD Radeon Token Factory 调用 Decision Provider 的示例：

```python
import os
import leximeter
from leximeter.client import SystemOneClient

# 示例 1：通过 OpenRouter 调用 Cloudflare Clef-Flash、TypeSafe Jev 或 Gemini Decision 1
or_decision = SystemOneClient(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
out = await leximeter.classify(
    df,
    "text",
    labels=labels,
    save_dir="run_cls",
    decision_client=or_decision,
    decision_model="cloudflare/clef-flash",
    include_probabilities=True,
)

# 示例 2：混合使用 DeepSeek 生成改写稿与 AMD Radeon Token Factory 的 d3-flash 校验
rad_decision = SystemOneClient(
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    api_key=os.environ["RADEON_API_KEY"],
    send_noul_criteria=False,  # d3 严格模式要求 noul 问题不附带 criteria
)
revised = await leximeter.paraphrase(
    df,
    "text",
    instructions="...",
    save_dir="run_para",
    model="deepseek-v4-flash",
    n_rounds=2,
    decision_client=rad_decision,
    decision_model="d3-flash",
)
```

## 前缀缓存友好提示词与 `order_strategy`

Leximeter 对提示词布局与请求派发顺序做了三项优化，在保持测量质量的同时提高服务端前缀缓存命中率：

- **前缀固定布局。** 全部 14 份提示词模板都将静态指令、量表定义、输出格式与 `additional_instructions` 放在开头，将动态内容如待测条目、`circle` 与 `square` 文本、虚拟画像以及增量映射放在末尾。成对任务的 `circle_first` 顺序随机化只发生在未缓存后缀中，在消除位置偏差的同时不影响前缀缓存命中。
- **键序策略 `order_strategy`。** `rate`、`classify`、`extract`、`rank`、`codify` 与 `merge` 均支持该参数，包含三种模式：
  - `"per_run"` 为默认策略，按批次、run、条目的顺序派发任务。同一次 run 内的所有条目共享固定键序以提高前缀缓存命中率。不同 run 或不同轮次采用严格互异的确定性置换，抵消局部属性锚定偏差。
  - `"fixed"` 在所有 run 中保持调用方原始键序，多次重复运行时可触发跨轮热缓存，在 DeepSeek 实测中命中率最高达到 86.7%。
  - `"per_item"` 保留上游 GABRIEL 逐条随机打乱键序的行为。
- **自适应首条预热。** `batch_execute` 提供 `warmup_first` 参数，默认为 `None`。当待处理请求数超过 `max(8, n_parallels)` 时，引擎先顺序发送首条请求以填充服务端前缀缓存，再并发派发其余请求。小批次不会产生额外的串行等待。

完整实验数据与质量对照见 [research/kv-cache-experiment-results.md](research/kv-cache-experiment-results.md) 与 [docs/adr/0005-prefix-first-prompts-and-per-run-ordering.md](docs/adr/0005-prefix-first-prompts-and-per-run-ordering.md)。

## 覆盖与扩展 Provider

每次调用任务时都可以通过关键字参数注入 Provider 实例，覆盖 `.env` 中的默认配置：

```python
import leximeter
from leximeter.client import ChatCompletionsClient, ResponsesClient, SystemOneClient

results = await leximeter.rate(
    df,
    "text",
    attributes=attrs,
    save_dir="run_a",
    client=ChatCompletionsClient(api_key="sk-...", base_url="https://api.deepseek.com"),
)

# 也可以在不同调用中混用不同 Provider
a = await leximeter.rate(df, "text", attributes=attrs, save_dir="run_b", client=ResponsesClient())
b = await leximeter.classify(
    df,
    "text",
    labels=labels,
    save_dir="run_c",
    decision_client=SystemOneClient(base_url="https://openrouter.ai/api/v1", api_key="sk-or-..."),
    decision_model="typesafe/jev-1.14",
)
```

如需接入自定义后端，实现以下最小协议即可：

```python
class LLMClient:
    async def complete(self, prompt, *, model, json_mode, **kw) -> str: ...

class DecisionClient:
    async def decide(self, state, questions, *, model, **kw) -> DecisionResult: ...

class EmbeddingClient:
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

离线测试可以使用内置的 `DummyClient`、`DummyDecisionClient`、`DummyEmbeddingClient` 与 `DummySearchTool`。`tests/conftest.py` 中还提供了 `DictClient`、`DictDecisionClient`、`CountingClient`、`CountingDecisionClient` 与 `RecordingClient`。

## 完整示例

[examples/reference.py](examples/reference.py) 包含 14 个章节，与 GABRIEL 教程 notebook 中的文本任务逐一对应，并标注了对应的 notebook cell 编号。示例使用的数据、参数与上游一致，其中包括需要 `TAVILY_API_KEY` 的 `whatever` 联网检索示例。配置好 `.env` 后可直接运行：

```bash
uv run python examples/reference.py
```

## 测试

运行离线测试套件不需要 API 密钥或网络连接，全部用例通过假 Provider 跑通完整链路：

```bash
uv run pytest
```

如需针对 `.env` 中配置的真实 Provider 执行冒烟验证，可以运行以下脚本。脚本逐个独立执行各任务，单个任务失败不会阻塞后续任务：

```bash
uv run python examples/smoke_tasks.py                      # 运行全部 18 个任务
uv run python examples/smoke_tasks.py rank                 # 仅运行指定任务
uv run python examples/smoke_tasks.py discover_checkpoint  # 复验断点续跑，第二次运行零 Provider 调用
```

## 常见问题

- **如何切换模型？** 修改 `.env` 中的 `LEXIMETER_MODEL`，或在调用任务时传入 `model=` 或 `client=`。Leximeter 不限制模型名称，请求协议由 `LEXIMETER_PROTOCOL` 决定。
- **如何使用本地模型？** 启动 Ollama、vLLM 或 LM Studio 等 OpenAI 兼容网关后，设置 `LEXIMETER_PROTOCOL=chat` 并将 `LEXIMETER_BASE_URL` 指向本地服务地址，例如 `http://localhost:11434/v1`。
- **文本嵌入不可用怎么办？** 当 `LEXIMETER_EMBEDDING_PROVIDER=none` 或未安装 `embeddings` 可选依赖时，`merge` 与 `deduplicate` 会自动回退到纯 LLM 路径。在 `merge` 中也可以传入 `use_embeddings=False` 强制关闭嵌入。
- **任务运行中途中断怎么办？** 直接重新执行同一命令即可续跑。已完成的请求会按检查点跳过，完整跑过的 `discover`、`debias`、`ideate`、`poll`、`seed`、`codify` 与 `bucket` 在输入不变时会直接从磁盘加载结果。如需强制重新运行，传入 `reset_files=True`。
- **`.env` 配置没有生效怎么办？** 确认 `.env` 位于项目根目录或其上层目录，并且当前环境中没有设置同名的系统环境变量，因为系统环境变量优先级更高。
- **输出列与上游 GABRIEL 有差异吗？** 输出列名严格对齐 GABRIEL 契约。目前本 fork 与上游仅有两处刻意差异，详见 [research/spec.md](research/spec.md)。一是 `merge` 的嵌入缓存改为 JSON 格式以保证反序列化安全，二是 `view` 界面暂未移植。

## 引用

本项目 fork 自 GABRIEL。在学术研究中使用时，请同时引用上游论文与仓库：

- Asirvatham, H., Mokski, E., and Shleifer, A. (2026). *GPT as a Measurement Tool*. NBER Working Paper No. 34834.
- Asirvatham, H. and Mokski, E. (2026). *GABRIEL*. https://github.com/openai/GABRIEL

## 许可证

本项目沿用上游 [openai/GABRIEL](https://github.com/openai/GABRIEL) 的 Apache License 2.0 协议，详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
