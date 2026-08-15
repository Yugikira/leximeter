# Gensim 调研：能否替代 leximeter 自建的 embedding（词向量）部分

> 调研日期：2026-08-15。结论先行：**不建议替代**——gensim 是词级（word-level）静态向量库，无法免训练产出句子/短语级向量，对 leximeter 的多语言短文本语义预筛场景存在质量与冷启动双重硬伤；其相似度计算层相对 leximeter 现有实现也没有性能优势（瓶颈是纯 Python 循环，用已有依赖 numpy 向量化即可解决）。gensim 仅在「纯英文 + 词级推理 + 有领域语料可训练」的场景下才值得考虑。
> 本文所有事实均以官方文档 / PyPI / GitHub 页面为准，出处见文末「参考来源」。
>
> **更新（2026-08-15）**：报告成文后按用户决策，leximeter 的 deduplicate 嵌入预筛已对齐 GABRIEL 上游契约改为 `kmeans2` + 簇内分段（见 spec.md「嵌入预筛」条目）；`utils/similarity.py`（含 `top_k_similar` 死代码）已删除。本报告 4.2 / 5.3 关于相似度计算层的建议随之过时——性能优化通过对齐上游实现达成，未引入 gensim。

## 一、背景：leximeter 当前的 embedding 实现

### 1.1 Embedding 客户端抽象

`src/leximeter/client/embedding.py` 定义 `EmbeddingClient.embed(texts) -> list[list[float]]`，三个实现：

| 实现 | 文件 | 说明 |
|---|---|---|
| `LocalBgeClient` | `embedding_local.py` | sentence-transformers 跑 `BAAI/bge-m3`（多语言），CPU，懒加载，`normalize_embeddings=True`，`[embeddings]` 可选依赖 |
| `OpenAIEmbeddingClient` | `embedding_openai.py` | OpenAI `text-embedding-3-small` API |
| `DummyEmbeddingClient` | `embedding.py` | 测试桩（全零向量） |

### 1.2 相似度工具

`src/leximeter/utils/similarity.py`（纯 Python，约 60 行）：`cosine()`、`top_k_similar()`（全排序）、`cluster_by_similarity()`（O(n²) 两两 cosine + union-find 贪心连通分量）。

### 1.3 消费方

- **`deduplicate` 任务**（`tasks/deduplicate.py:83-89`）：对所有唯一项 embed → `cluster_by_similarity`（阈值 0.85）聚类 → 每组交给 LLM 去重。
- **`merge` 任务**（`tasks/merge.py`）：short/long 两侧 embed（JSON 检查点缓存 `short_embeddings.json`/`long_embeddings.json`）→ NumPy 向量化 cosine 自动匹配（阈值 0.75）→ `scipy.cluster.vq.kmeans2` 聚类 + 窗口候选 → LLM 匹配。
- **场景本质**：短文本（术语/实体名/键值，通常 1~5 个词）的语义近似匹配与 LLM 预筛，**不是长文档检索**；文本跨语言（中文/英文混用是常态）。

### 1.4 相关事实

- leximeter `pyproject.toml` 已依赖 `numpy>=1.18.0`、`scipy>=1.8.0`（merge 的 kmeans2 用了 scipy），`sentence-transformers` 是可选项；`requires-python = ">=3.10"`。

---

## 二、gensim 现状（版本 / 维护 / 依赖）

| 项 | 事实 | 来源 |
|---|---|---|
| 最新版本 | **4.4.0**，PyPI 发布于 **2025-10-18**（GitHub Release 2025-10-16，changelog 标 2025-10-13） | [PyPI gensim](https://pypi.org/project/gensim/)；[GitHub releases](https://github.com/piskvorky/gensim/releases) |
| Python 要求 | `>=3.9`（4.4.0 加了 Python 3.13 wheel，去掉了 3.8） | [PyPI gensim](https://pypi.org/project/gensim/) |
| 核心依赖 | 官方声明 "This software depends on **NumPy and Scipy**" | [PyPI gensim](https://pypi.org/project/gensim/) |
| 维护状态 | **明确进入稳定维护模式**：README 原文 "Gensim is in stable maintenance mode: we are **not accepting new features**, but bug and documentation fixes are still welcome!" | [GitHub piskvorky/gensim](https://github.com/piskvorky/gensim) |
| 社区规模 | 16.5k stars / 4.4k forks | [GitHub piskvorky/gensim](https://github.com/piskvorky/gensim) |

**发布节奏佐证维护状态**（[GitHub releases 页面](https://github.com/piskvorky/gensim/releases)）：

- 2021：5 个版本（4.0.0/4.0.1/4.1.0/4.1.1/4.1.2，活跃期）
- 2022：2 个（4.2.0、4.3.0）
- 2023：1 个（4.3.2）
- **2024：0 个**
- 2025：1 个（4.4.0，内容以 NumPy 2.0 兼容 + Python 3.13 wheel 等兼容性修复为主）

即 4.3.2（2023-08）到 4.4.0（2025-10）之间有约两年空窗，近年版本基本是「给旧库续命」的兼容性更新，无新功能。

**安装/分发**：`pip install gensim`，核心 Cython 扩展随 wheel 分发（4.4.0 提供 Python 3.9~3.13 wheel）；运行时依赖 numpy/scipy（见上表）。注意 gensim 本体很小（几 MB），但**配套的预训练向量动辄几百 MB~1.6 GB**（3.2 节），"安装成本"主要花在数据下载上（[downloader 文档](https://radimrehurek.com/gensim/downloader.html) 说明数据落在 `~/gensim-data`）。

---

## 三、gensim 能力盘点

### 3.1 模型家族（全部为**词级 / 文档级（需自训）**模型）

| 模型 | 能力 | 出处 |
|---|---|---|
| **Word2Vec** | 词向量，输入是 `List[List[str]]` 的 token 化语料；训练后 `model.wv` 提供词向量查询 | [models/word2vec.html](https://radimrehurek.com/gensim/models/word2vec.html) |
| **FastText** | 在 Word2Vec 基础上增加 **subword（字符 n-gram）**，可对 **OOV 词**计算向量："FastText achieves this by keeping vectors for ngrams: adding the vectors for the ngrams of an entity"；"Even OOV words have vectors in FastText"；唯一失败场景是 `get_vector()` 中"word and all its ngrams not in vocabulary" | [models/fasttext.html](https://radimrehurek.com/gensim/models/fasttext.html) |
| **Doc2Vec** | 段落/文档向量（PV-DM / PV-DBOW），输入 `TaggedDocument`；**必须训练**——"If you don't supply documents, the model is left uninitialized"；新文档向量靠 `infer_vector`（"post-bulk training"） | [models/doc2vec.html](https://radimrehurek.com/gensim/models/doc2vec.html) |
| **KeyedVectors** | 统一的词向量查询层（Word2Vec/Doc2Vec/FastText 的 `wv` 均为其别名）：`most_similar` / `similarity` / `n_similarity` / `cosine_similarities` / `get_mean_vector` / `get_normed_vectors`；支持 mmap 磁盘加载（"Load data from disk instantaneously"）、多进程共享单份内存 | [models/keyedvectors.html](https://radimrehurek.com/gensim/models/keyedvectors.html) |

要点：**gensim 没有任何「加载即用」的句子/短语级编码器**。`Doc2Vec` 是唯一句子级方案，但必须自己准备标注语料训练。

#### 3.1.1 KeyedVectors 常用 API 速览（评估计算层时对照）

| API | 行为 | 复杂度特征 |
|---|---|---|
| `most_similar(positive, negative, topn)` | 精确 cosine，扫描全词表（可 `restrict_vocab` 裁剪范围） | 每查询 O(V)（V=词表大小） |
| `similarity(w1, w2)` | 两个词之间精确 cosine | O(1) |
| `n_similarity(ws1, ws2)` | 两组词各取**均值向量**后做 cosine | 聚合后 O(1) |
| `cosine_similarities(vec1, vectors_all)` | 一查多的批量 cosine（静态方法） | O(V) |
| `get_normed_vectors()` / `fill_norms()` | 预计算 L2 归一化，加速批量相似度 | 一次性 O(V) |
| `AnnoyIndexer(model, num_trees)` | 近似最近邻，`most_similar(..., indexer=indexer)` 挂载 | 建树 O(V·trees)，查询亚线性（近似） |

（出处：[keyedvectors 文档](https://radimrehurek.com/gensim/models/keyedvectors.html)、[similarities/annoy 文档](https://radimrehurek.com/gensim/similarities/annoy.html)；词表量级参考：gensim-data 目录中 google-news 模型体积 1662 MB，其词表为百万级）

#### 3.1.2 gensim 4.0 API 迁移提醒

gensim 4.0（2021-03）是破坏性大版本：词向量全部从 `model.most_similar(...)` 迁到 `model.wv.most_similar(...)`，移除了 `summarization`、Mallet wrapper 等模块（[releases 页面](https://github.com/piskvorky/gensim/releases)）。若未来任何代码/资料引用旧式 API，需按 4.x 迁移；这属于存量知识，与本次评估无关，但提醒"网上大量 gensim 教程是 3.x 时代的"。

### 3.2 预训练模型：`gensim.downloader`（gensim-data 包）

`gensim.downloader` 是官方下载/加载 API（`api.load` / `api.info`，数据来自 RaRe-Technologies/gensim-data 仓库）([downloader 文档](https://radimrehurek.com/gensim/downloader.html)；[gensim-data 目录](https://github.com/RaRe-Technologies/gensim-data)）。

**可用模型清单（共 12 个，全部为 2012~2017 年的语料产物，仓库内容多年未扩充）：**

| 名称 | 大小 | 底座语料 |
|---|---|---|
| `word2vec-google-news-300` | 1662 MB | Google News（约 1000 亿词） |
| `glove-wiki-gigaword-50/100/200/300` | 65/128/252/376 MB | Wikipedia 2014 + Gigaword 5（6B tokens，uncased） |
| `glove-twitter-25/50/100/200` | 104/199/387/758 MB | Twitter（27B tokens） |
| `fasttext-wiki-news-subwords-300` | 958 MB | Wikipedia 2017 等（16B tokens，带 subword） |
| `conceptnet-numberbatch-17-06-300` | 1168 MB | ConceptNet + word2vec + GloVe + OpenSubtitles（**唯一跨语言对齐选项**） |
| `word2vec-ruscorpora-300` | 198 MB | 俄语语料 |

**关键事实：目录中没有中文模型，也没有任何句子级模型。** 除 Numberbatch 外全部是单一语言的词向量。

### 3.3 中文词向量：只能外部获取，且都是词级

| 来源 | 事实 | 出处 |
|---|---|---|
| **腾讯 AI Lab 词向量** | 12,287,936 条词/短语，200 维（另有 100 维）；v0.2.0（2021-12-24）；压缩包约 9 GB（解压 22 GB），另有 2M 条小版本（约 1.5 GB）；官方声明 **"for research purpose only"**（仅限研究用途，商业使用受限） | [Tencent AI Lab 官方页](https://ailab.tencent.com/ailab/nlp/en/embedding.html) |
| **fastText wiki.zh** | `wiki.zh.zip` / `wiki.zh.vec`，300 维，Wikipedia skip-gram，CC BY-SA 3.0 | [fastText 预训练向量页](https://fasttext.cc/docs/en/pretrained-vectors.html) |
| **fastText crawl 向量（157 语言）** | 含中文 `cc.zh.300.bin.gz` / `cc.zh.300.vec.gz`；CBOW + position-weights，300 维，字符 n-gram 长度 5 | [fastText crawl 向量页](https://fasttext.cc/docs/en/crawl-vectors.html) |

注意两点：**(1) 中文词向量全部是「词」级**，使用时必须先用 jieba 等工具对中文分词（gensim 本身不做分词，训练/查询输入都是 token 列表）；**(2) 许可证风险**——质量最好的腾讯词向量仅限研究用途，不适合直接进生产库。

### 3.4 相似度 API 与近似索引（Annoy）

- **精确查询**（[keyedvectors 文档](https://radimrehurek.com/gensim/models/keyedvectors.html)）：`most_similar`（对 vocab 全量扫描做 cosine，支持 `restrict_vocab`/`topn`/`clip_start`/`clip_end` 裁剪）、`similarity(w1,w2)`、`n_similarity(ws1,ws2)`（"cosine similarity between two sets of keys"，实现是两组词向量的均值间 cosine）、静态方法 `cosine_similarities(vector_1, vectors_all)`。
- **近似索引**（[similarities/annoy 文档](https://radimrehurek.com/gensim/similarities/annoy.html)）：`gensim.similarities.annoy.AnnoyIndexer(model=None, num_trees=None)` 集成 Spotify Annoy，接入方式为 `model.most_similar("cat", topn=2, indexer=indexer)`——"allows the use of Annoy for fast (approximate) vector retrieval in most_similar() calls"，需 `pip install annoy`；`save` 只存索引不存模型。
- 另有 `similarities.nmslib` 模块提供同一模式的 NMSLIB 近似索引（[keyedvectors 侧栏](https://radimrehurek.com/gensim/models/keyedvectors.html)）。

### 3.5 回答调研问题 3：「不训练能否直接得到句子/短语级向量？」

**答案是否定**（与预期一致，已有官方出处佐证）：

- gensim 预训练目录里没有句子级模型（3.2 节清单）；
- Doc2Vec 是唯一文档级方案，但[文档明确](https://radimrehurek.com/gensim/models/doc2vec.html)需要自己构造 `TaggedDocument` 语料训练，`infer_vector` 是训练后的推理能力；
- 词向量只能靠「均值/加权」聚合近似句子向量（`n_similarity` / `get_mean_vector`），这是词袋近似，不是句子编码。

---

## 四、对照评估：能否替代 leximeter 的 embedding 部分

### 4.0 总览

| 评估维度 | 替代 LocalBgeClient（语义质量） | 替代 utils/similarity.py（计算层） | 走训练路线 |
|---|---|---|---|
| 能力匹配 | ✗ 词级 vs 句子级错位 | ✗ 小 N 无优势 | ✗ 无训练语料 |
| 多语言/中文 | ✗ 需外部词向量 + 分词 | — | ✗ 中文分词前置 |
| 冷启动 | ✓ 免训练（但质量差） | — | ✗ 新键仍是 OOV |
| 维护风险 | ✗ 维护模式、目录冻结 | ✗ 为 60 行代码引重依赖 | ✗ 调参是研究项目 |
| **结论** | **不替代** | **不替代** | **不替代** |

### 4.1 替代 `LocalBgeClient`（质量维度）：不能

leximeter 的匹配对象是**多语言短文本键值**（"NYC" vs "New York City"、中英混排、拼写变体），而：

1. **模型类型错位**。BGE-M3 是面向句子/文档的编码器：100+ 语言、dense+sparse+multi-vector 三合一、输入最长 8192 token、在 MIRACL/MKQA 多语言与跨语言基准上宣称 SOTA（[FlagEmbedding README](https://github.com/FlagOpen/FlagEmbedding)）。gensim 能免训练拿到的只有**词级静态向量**，对"短语 vs 短语"只能 `n_similarity` 式均值近似——词袋均值丢失词序、无法消歧（一词多义）、对"缩写-全称"（NYC / New York City）这类依赖短语级共现结构的匹配效果不可控。
2. **跨语言不可用**。主流 gensim-data 模型（google-news / glove / fasttext-wiki-news）全是单语（英文为主）；中文向量需外部获取 + 分词预处理（3.3 节）。要做"纽约 ↔ New York"这种跨语言匹配，唯一的对齐向量（Numberbatch）也仅覆盖其词表，且数据冻结于 2017 年。
3. **OOV 与拼写变体**。Word2Vec/GloVe 对词表外词直接失败（KeyError 语义）；FastText 用 subword 兜底（见 3.1），但 leximeter 的输入是"数据库键值"而非天然分词的中英文混合文本——英文缩略词、中文未分词串都会先死在分词/词表环节。

#### 4.1.1 词级平均聚合（`n_similarity` 式）的已知局限清单

gensim 文档明确定义 `n_similarity` 为"two sets of keys"的均值向量 cosine（[keyedvectors 文档](https://radimrehurek.com/gensim/models/keyedvectors.html)）。把这种聚合当"句子相似度"用，存在四类结构性缺陷（前两条有官方文档直接佐证，后两条是词袋模型的公认行为，这里结合 gensim API 机制说明）：

| 缺陷 | 机制 | 对 leximeter 场景的影响 |
|---|---|---|
| **OOV** | 词表外的词查询即失败；FastText 仅靠字符 n-gram 兜底（[fasttext 文档](https://radimrehurek.com/gensim/models/fasttext.html)） | 数据库键里的专名/缩写/新词大量落空 |
| **词序无关** | 均值聚合把 "A of B" 与 "B of A" 视作同一向量 | "Item price" vs "price item" 不可区分 |
| **上下文无关** | 静态向量一词一义 | "bank"（银行/河岸）类多义词在键值场景误匹配 |
| **反义词** | 反义词共现环境相似，向量距离反而近 | "include" vs "exclude" 这类键值对会被高估相似度、误聚类 |

**质量结论：若用 gensim 词向量替换 bge-m3，deduplicate/merge 的预筛召回会显著下降，LLM 兜底负担加重，且多语言场景直接退化。**

### 4.2 替代 `utils/similarity.py` 计算层（性能维度）：没必要

- 现有实现的问题是**纯 Python 的 O(n²) 两两 cosine**（`cluster_by_similarity`）与全排序 `top_k_similar`，瓶颈在解释器循环；N 是「单批次唯一键数」（百~千级），不是百万级。
- KeyedVectors 的 `most_similar` 是**全词表扫描**（O(V)，V 是词表大小——google-news 有 300 万词），对小 N 反而引入无关开销；gensim 不提供 N×N 两两矩阵 API，`cosine_similarities` 也只支持"一查多"。
- Annoy 索引的适用前提是「**静态大库 + 反复查询**」（千万级向量、索引建一次用多次）；leximeter 每个任务批次的向量集合都是**一次性计算、用完即弃**，建树开销纯属倒贴，且官方文档强调 Annoy 结果是近似的。
- **更优解是零依赖**：leximeter 已依赖 numpy/scipy，把 `similarity.py` 的循环改成 `numpy` 矩阵 cosine（或 `sklearn.metrics.pairwise.cosine_similarity`）即可获得比 gensim 更大的提速，还不用引入新依赖。

### 4.3 训练路径：冷启动死局

若走 gensim 训练路线（Word2Vec/FastText/Doc2Vec）：

- leximeter **没有语料**，只有「待匹配的短键值」本身。在这些键值上自训（把键串分词后当句子训）得到的向量只认识训练时见过的键，运行时出现新键 → 又是 OOV；词表只有几百~几千词时，共现信号稀薄，向量质量远低于预训练编码器。
- Doc2Vec 对 1~5 词"文档"退化接近词向量，且 `TaggedDocument` 标注、超参（vector_size/window/min_count/epochs）调优是完整的研究项目，不是换库。
- 结论：**训练路线在 leximeter 场景下是冷启动死局**（无数据可训、训了也覆盖不了运行时新键）。

### 4.4 维护风险

| 维度 | gensim 4.4.0 | sentence-transformers 5.7.0（bge-m3 载体） |
|---|---|---|
| 状态 | **稳定维护模式**，明确"不收新功能"（2025-10 后无新版本） | 活跃，最新版 2026-08-06 发布 |
| 社区 | 16.5k stars | 19k stars，Hugging Face 官方维护（Tom Aarsen） |
| 生态 | 预训练目录 2017 年冻结，无新模型 | HF Hub 上 1.5 万+ 预训练句向量模型，BGE 系列持续更新 |

（出处：[gensim README](https://github.com/piskvorky/gensim)、[gensim releases](https://github.com/piskvorky/gensim/releases)、[sentence-transformers](https://github.com/UKPLab/sentence-transformers)、[PyPI sentence-transformers](https://pypi.org/project/sentence-transformers/)）

#### 4.4.1 活跃度差距的根因：模型世代不同

两者不是"同类库的竞争"，而是**两个时代的产物**：

- gensim 的巅峰在 2013~2018 年（word2vec 经典论文时代），其预训练目录里的模型语料最新也只到 2017 年（Wikipedia 2017、ConceptNet 2017、Google News 更是 2012 年的语料，见 [gensim-data 目录](https://github.com/RaRe-Technologies/gensim-data)）。
- sentence-transformers 生于 2019 年后上下文编码器（BERT 系）时代，其上 bge-m3 的 MIRACL/MKQA 基准、100+ 语言、8192 token 等能力（[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)）在词级模型框架内**不存在对应物**。
- 因此"gensim 活跃度下降"与其说是失修，不如说是其技术路线被时代替代；维护模式声明（[gensim README](https://github.com/piskvorky/gensim)）是对这一现实的承认。

### 4.5 兼容性

gensim 4.4.0 要求 Python >=3.9，与 leximeter 的 `>=3.10` 兼容；但其编译扩展 + numpy/scipy 依赖与 leximeter 现状无冲突——**技术兼容没问题，纯粹是"值不值得"的问题**。

### 4.6 操作性影响：若强行替换，改动面有多大

假设仍要引入 gensim，受影响的不只是"换一个 embed 实现"，而是整条链路：

1. **`EmbeddingClient` 接口形状不匹配**：`embed(texts) -> list[list[float]]` 的契约建立在"句子级、批量、等长语义空间"上；gensim 需要外部封装把短键分词→查词向量→均值聚合，封装后的行为（OOV 返回什么？空串怎么办？）必须自己定义——等于把 `utils/similarity.py` 之外的**新一段自建代码**写进库。
2. **`deduplicate.py` 的聚类阈值（0.85）与 `merge.py` 的匹配阈值（0.75）需要重新标定**：词向量均值的相似度分布与句子编码器完全不同（更稀疏、更集中在低分区间），现有阈值是配合 bge-m3 调出来的；换向量源 = 重新调参 + 重新验证，这是隐性成本。
3. **`merge` 的 JSON 检查点缓存（`short_embeddings.json`/`long_embeddings.json`）**：换向量源后旧缓存作废（维度从 1024 变 300），需要清理或版本化缓存键。
4. **`DummyEmbeddingClient`（3 维全零）与测试**：现有测试假设向量是句子级编码结果；gensim 方案下测试桩语义也要重写。
5. **多语言路径**：中文键需 jieba 分词 + 外部中文词向量（3.3 节），分词错误会传导进匹配——新增一整类 bug 面。

即：即便忽略质量，**替换的工程成本 ≈ 重写 embedding 抽象 + 重标阈值 + 迁移缓存 + 新增分词依赖**，而收益为零（无性能、无新能力）。

---

## 五、结论建议

### 5.1 总体结论：**不替代**

| 候选替代项 | 结论 | 理由摘要 |
|---|---|---|
| gensim 词向量 → 替代 `LocalBgeClient` | **不替代** | 词级 vs 句子级模型错位；无免训练句子向量；无中文、跨语言受限；质量与召回下降；OOV/分词硬伤 |
| gensim KeyedVectors/Annoy → 替代 `utils/similarity.py` | **不替代** | 小 N 场景无性能收益，Annoy 面向静态大库；瓶颈是纯 Python 循环，用已有 numpy 向量化即可解决 |
| gensim 训练路线（Word2Vec/FastText/Doc2Vec） | **不替代** | 无语料冷启动死局；运行时新键仍是 OOV；Doc2Vec 对短键值退化 |
| 整体依赖 | **不引入** | 维护模式、2017 年冻结的预训练生态；leximeter 现方案（sentence-transformers + OpenAI API）维护活跃 |

### 5.2 gensim 在什么场景下仍值得考虑

1. **纯英文**、词级语义任务（词类比、词距离、`doesnt_match` 这类词汇学分析）——`most_similar`/`n_similarity` 是标准工具，google-news-300 质量依然能打；
2. **有领域语料可训**（如已有大规模规范术语表/词表）时，Word2Vec/FastText 训练快、体积小（mmap 秒加载、多进程共享），是离线小体积场景的合理选择；
3. **拼写变体容错**：FastText 的 subword 机制对"词表外拼写变体"天然兜底（3.1 节），可作为 BGE 的**互补**而非替代（例如在 embedding 之前用 fasttext 词向量做拼写归一化预筛）——但中文仍需分词前置；
4. 纯粹想做**词级分析/教学/研究**的独立工具。

### 5.3 顺手建议（与 gensim 无关的等价优化）

若目的是"把自建部分做得更成熟"，优先做这两件低成本的事，而不是换 gensim：

1. `utils/similarity.py` 用 numpy 矩阵化重写（依赖已存在），O(n²) 循环提速 1~2 个数量级；
2. `cluster_by_similarity` 的阈值聚类可考虑用 scipy 的层次聚类（`scipy.cluster.hierarchy`）替代贪心 union-find，行为更可控。

### 5.4 实证验证方案（如后续想亲自确认，而非信任本文推理）

本文结论是架构与文档层面的定性评估，没有跑真实模型对比（google-news 下载 1.6 GB、bge-m3 加载 ~2 GB，超出本调研的轻量范围）。若需要数字证据，建议按下面协议做一次对比冒烟（不改生产代码）：

1. 构造 30~50 组 leximeter 真实场景样例键（含：缩写 vs 全称、"纽约" vs "New York"、拼写变体、同义词、无关干扰项，标注期望的相似对）；
2. 三路打分：bge-m3（现方案）／`fasttext-wiki-news-subwords-300`（gensim-data 中唯一带 subword 的模型，`gensim.downloader.api.load` 一次搞定，958 MB）／腾讯 200 维（中文，需分词，仅研究用途）；
3. 对每组打分算 top-k 命中率与 AUC，对比 `similarity.py` 阈值 0.85/0.75 下的聚类结果；
4. 顺带测 `cluster_by_similarity` 纯 Python vs numpy 矩阵化的耗时，验证 5.3 第 1 条。

预期结果：bge-m3 在跨语言/缩写-全称组明显领先；fasttext 仅在"拼写变体"组可能与 bge 打平；numpy 矩阵化在 N≥500 时快一个数量级。

---

## 六、参考来源

1. gensim PyPI（版本 4.4.0 / 2025-10-18 / Python >=3.9 / "depends on NumPy and Scipy"）：https://pypi.org/project/gensim/
2. gensim GitHub（"stable maintenance mode" 声明 / 16.5k stars）：https://github.com/piskvorky/gensim
3. gensim Releases（发布历史与节奏）：https://github.com/piskvorky/gensim/releases
4. gensim downloader 文档（api.load / api.info / gensim-data）：https://radimrehurek.com/gensim/downloader.html
5. gensim-data 仓库（12 模型 + 8 数据集完整清单、体积与底座语料）：https://github.com/RaRe-Technologies/gensim-data
6. gensim KeyedVectors 文档（most_similar / similarity / n_similarity / cosine_similarities / mmap）：https://radimrehurek.com/gensim/models/keyedvectors.html
7. gensim similarities.annoy 文档（AnnoyIndexer / most_similar(indexer=)）：https://radimrehurek.com/gensim/similarities/annoy.html
8. gensim FastText 文档（subword / OOV 词向量）：https://radimrehurek.com/gensim/models/fasttext.html
9. gensim Doc2Vec 文档（TaggedDocument / infer_vector / 必须训练）：https://radimrehurek.com/gensim/models/doc2vec.html
10. gensim Word2Vec 文档：https://radimrehurek.com/gensim/models/word2vec.html
11. fastText 预训练向量页（wiki.zh.zip / CC BY-SA 3.0）：https://fasttext.cc/docs/en/pretrained-vectors.html
12. fastText crawl 向量页（157 语言含 cc.zh.300，CBOW 300 维）：https://fasttext.cc/docs/en/crawl-vectors.html
13. Tencent AI Lab Embedding 官方页（12,287,936 词条 / 200 维 / research purpose only）：https://ailab.tencent.com/ailab/nlp/en/embedding.html
14. BGE-M3 / FlagEmbedding（100+ 语言 / dense+sparse+multi-vector / 8192 token / MIRACL·MKQA SOTA）：https://github.com/FlagOpen/FlagEmbedding
15. sentence-transformers GitHub（19k stars / 活跃维护）：https://github.com/UKPLab/sentence-transformers
16. sentence-transformers PyPI（5.7.0 / 2026-08-06）：https://pypi.org/project/sentence-transformers/
