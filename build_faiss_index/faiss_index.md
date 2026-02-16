# HVM-SVG 数据预处理工具 (`precompute_hvm_data.py`)

本目录下的 `precompute_hvm_data.py` 是 HVM-SVG 训练流程中的核心数据处理脚本。它负责将原始的 SVG Parquet 数据集转换为训练所需的多模态预计算数据，包括元数据索引、RAG 检索结果、视觉特征缓存、SVG 路径分组以及逐 group 独立渲染的视觉特征。

先看一下原始omnisvg数据（/mnt/data/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_process)
<img width="746" height="816" alt="image" src="https://github.com/user-attachments/assets/5ef18874-90c9-4b83-828f-06e0911294fa" />


## 📋 功能概述

该脚本包含五个处理阶段（Stage），支持独立运行或全流程运行：

1. **Stage 1: Metadata (元数据)**
   - 扫描所有 Parquet 数据文件。
   - 构建全局整数索引（`idx`）与原始样本 ID 的映射。
   - 提取文本描述、关键词等元数据。
   - 得到/mnt/data/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed/id_to_idx.json
   - 例如："00965a7e45b453e83afe51edc9208bb7": 0,
   - /mnt/data/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed/metadata.jsonl
   - {"idx": 0, "id": "00965a7e45b453e83afe51edc9208bb7", "description": "A stack of colorful books with a pink top book and a green bottom book.", "keywords": "books, stack, colorful, pink, green, top, bottom", "detail": "The image depicts a neatly arranged stack of books with vibrant colors. The topmost book is pink, followed by a brown book, another pink book, a green book, an orange book, and finally a blue book at the bottom. The books are stacked vertically, creating a visually appealing and organized appearance.", "token_len": 1171, "parquet_file": "train-small-1000.parquet", "parquet_row": 0}

2. **Stage 2: RAG (检索增强)**
   - 使用 **CLIP text encoder** (`clip-vit-large-patch14`) 对样本描述进行语义编码。
   - 拼接 `description + keywords` 以提高检索质量。
   - 构建 FAISS 向量索引（Inner Product，等效于 L2 归一化后的 cosine similarity）。
   - 为每个样本检索语义最相似的 Top-3 个样本（作为 In-Context Learning 的参考）。

3. **Stage 3: Features (整图视觉特征, for GME)**
   - 使用 Qwen2.5-VL 的完整 Vision Pipeline（ViT encoder + merger）提取 **post-merge** 特征。
   - 直接运行 `visual()` 模块取最终输出，不需要 hook / reverse_indices / 空间 reshape。
   - 输出形状：`[256, 3584]`（16×16 = 256 个 post-merge vision token，3584 维，已对齐 LLM 空间）。
   - 支持多 GPU / 多节点并行分片处理，支持断点续传。

4. **Stage 4: Groups (路径分组)**
   - 解析 SVG 源码，提取所有 Path 指令。
   - 计算每个 Path 的几何复杂度（基于指令类型和覆盖面积）。
   - 根据复杂度将 Path 智能划分为 1~4 个组（Group），用于分层生成。

5. **Stage 5: Group Features (逐 group 视觉特征, for PME)**
   - 对每个样本的每个 group：只渲染该 group 的 SVG paths，viewbox 设为 group 的 bbox（自动 crop + zoom），得到 448×448 的 PIL Image。
   - 通过 Qwen2.5-VL 完整 vision pipeline（encoder + merger）得到 `[256, 3584]`。
   - 优势：
     - 每个 group 的特征纯净，只表示该 group 的视觉外观
     - 解决了 SVG 中先填色后描边导致 bbox 重叠的问题
     - 小 group 被放大到 448×448，获得更精细的特征
     - 已对齐 LLM 空间（3584 维），复用了 Qwen 训练好的 merger
   - 支持多 GPU / 多节点并行分片处理，支持断点续传。

## ⚙️ 环境依赖

请确保已安装以下 Python 库：

```bash
pip install torch numpy pandas pyarrow pillow tqdm transformers faiss-cpu qwen_vl_utils cairosvg
```



## 🚀 使用指南

### 基本用法

```bash
# 运行全部阶段
python precompute_hvm_data.py --stage all

# 只运行某个阶段
python precompute_hvm_data.py --stage metadata
python precompute_hvm_data.py --stage rag
python precompute_hvm_data.py --stage features --gpu_id 0
python precompute_hvm_data.py --stage groups
python precompute_hvm_data.py --stage group_features --gpu_id 0
```

### 使用 `run_precompute.sh` 一键执行（推荐）

```bash
# 全部阶段
bash run_precompute.sh

# 只跑某个阶段
bash run_precompute.sh features
bash run_precompute.sh group_features
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--stage` | str | `all` | 运行阶段：`metadata`, `rag`, `features`, `groups`, `group_features`, 或 `all` |
| `--data_dir` | str | (见源码) | 原始 Parquet 数据目录路径 |
| `--model_path` | str | (见源码) | Qwen2.5-VL 模型权重目录路径 |
| `--clip_model_path` | str | (见源码) | CLIP 模型路径（用于 RAG 文本编码） |
| `--output_dir` | str | (见源码) | 预计算结果输出目录 |
| `--gpu_id` | int | `0` | 指定使用的 GPU ID（用于 Features 和 Group Features 阶段） |
| `--batch_size` | int | `16` | 视觉特征提取的 Batch Size |
| `--num_shards` | int | `1` | 并行处理的总分片数 |
| `--shard_id` | int | `0` | 当前进程处理的分片 ID（0 ~ num_shards-1） |

---

## 📚 详细流程与输出

输出目录为 `hvm_precomputed/`，各阶段产物如下：

### Stage 1: Metadata

- **输入**：`data_dir` 下的 `*.parquet` 文件。
- **输出**：
  - `metadata.jsonl`：包含 `idx`（全局索引）、`id`（原始 ID）、`description`、`keywords`、`detail`、`token_len`、`parquet_file`、`parquet_row`。
  - `id_to_idx.json`：原始字符串 ID 到整数索引的映射字典。

### Stage 2: RAG

- **依赖**：需要先运行 Metadata 阶段。需要 CLIP 模型权重。
- **过程**：
  1. 加载 CLIP text encoder（`clip-vit-large-patch14`，embedding 维度 768）。
  2. 拼接每个样本的 `description + keywords` 作为检索文本。
  3. 批量编码并 L2 归一化，得到 `[N, 768]` 的文本向量。
  4. 构建 FAISS `IndexFlatIP` 索引。
  5. 检索 Top-K+1 近邻，排除自身后取 Top-3。
- **输出**：
  - `rag_results.jsonl`：每行包含 `idx`、`ref_indices`（Top-3 相似样本的 idx）、`ref_scores`。
  - `text_embeddings.npy`：所有样本的文本向量 `[N, 768]`（float32）。
  - `faiss_index.bin`：保存的 FAISS 索引文件。faiss_index.bin 是 FAISS 索引的序列化文件,包含所有样本的 CLIP 文本向量(768 维)和索引结构。它的作用是为未来的推理提供实时检索能力:用户输入新文本 → CLIP 编码 → 在这个索引里搜 Top-3 → 拿到参考样本。


### Stage 3: Features (整图 post-merge 特征)

- **依赖**：Metadata 阶段。需要加载 Qwen2.5-VL Vision Encoder。
- **过程**：
  1. 只加载 Vision Encoder（encoder + merger）到 GPU，释放 LLM 部分以节省显存。
  2. 直接运行 `visual(pixel_values, grid_thw)` 获取完整 pipeline 输出。
  3. 每张图输出 `[256, 3584]` 的张量。
- **输出**：
  - `features/{idx//1000:03d}/{idx:06d}.pt`：PyTorch Tensor 文件。
  - 目录结构：按 `idx // 1000` 分子文件夹存储，避免单目录文件过多。
  - Tensor Shape：`[256, 3584]`（float16, ~1.8MB/个）。

> **为何使用 post-merge 特征？**  
> 旧版使用 pre-merge `[32,32,1280]` 特征需要 hook 截取、reverse_indices 恢复空间顺序、手动 reshape 等复杂操作。
> 新版直接取 `visual()` 输出的 post-merge 特征 `[256, 3584]`，优势：
> - **代码简化**：无需 hook / reverse_indices / 空间 reshape
> - **QFormer 更快**：GME 输入从 3×1024=3072 tokens 降到 3×256=768 tokens（快 4 倍）
> - **存储更省**：每图从 2.6MB 降到 1.8MB（省 30%）
> - **GME 和 PME 统一 d_vision=3584**，config 更简洁

#### ⚡️ 并行提取特征示例（推荐）

特征提取较慢，建议在多张 GPU 上并行运行：

```bash
# 使用 run_precompute.sh（自动 8 GPU 并行）
bash run_precompute.sh features

# 或手动分片
python precompute_hvm_data.py --stage features --gpu_id 0 --num_shards 8 --shard_id 0
python precompute_hvm_data.py --stage features --gpu_id 1 --num_shards 8 --shard_id 1
...
```

### Stage 4: Groups

- **依赖**：Metadata 阶段。
- **过程**：
  1. 解析 SVG `<path>` 标签，提取 `d` 属性和 `fill` 颜色。
  2. 计算每个 path 的 bounding box（曲线使用控制点近似）。
  3. 计算复杂度：`命令加权分数 × 面积权重`。
  4. 根据总复杂度和 path 数量决定分组数（1~4 组）。
  5. 按 SVG path 顺序，基于累积复杂度均匀切分。
- **输出**：
  - `groups.jsonl`：每行包含 `num_paths`、`total_complexity`、`num_groups`，以及 `groups` 列表。每个 Group 包含：
    - `path_indices`：该组包含的 Path 索引
    - `bbox`：SVG viewBox 坐标包围盒（含 10% padding）
    - `bbox_feature`：（legacy）映射到 32×32 pre-merge 网格的坐标，训练时不再使用
    - `complexity`：组复杂度分数

#### 分组策略

| 总复杂度 | Path 数 | 分组数 |
| :--- | :--- | :--- |
| < 30 | ≤ 2 | 1 |
| < 80 | ≤ 5 | 2 |
| < 150 | — | 3 |
| ≥ 150 | — | 4 |

### Stage 5: Group Features (逐 group 独立渲染特征)

- **依赖**：Metadata、Groups 阶段。需要加载 Qwen2.5-VL Vision Encoder。
- **过程**：
  1. 对每个样本，读取其 SVG 和 group 分组信息。
  2. 对每个 group：
     - 提取该 group 包含的 `<path>` 标签（支持 `<path .../>` 和 `<path ...></path>` 两种写法）
     - 设置 viewbox 为 group 的 bbox（自动裁剪 + 缩放）
     - 用 `cairosvg` 渲染为 448×448 PNG
     - 通过 Qwen2.5-VL 完整 vision pipeline 提取 `[256, 3584]` 特征
  3. 每个样本保存一个 list of tensors。
- **输出**：
  - `group_features/{idx//1000:03d}/{idx:06d}.pt`：PyTorch 文件，包含 list of `[256, 3584]` tensors。
  - 每个 tensor 对应一个 group 的独立渲染特征。
  - 1~4 个 tensor/样本（取决于 group 数）。

#### ⚡️ 并行提取 group 特征

```bash
# 使用 run_precompute.sh（自动 8 GPU 并行）
bash run_precompute.sh group_features

# 或手动分片
python precompute_hvm_data.py --stage group_features --gpu_id 0 --num_shards 8 --shard_id 0
...
```

---

## 📂 最终目录结构

执行完所有阶段后，`output_dir` 结构如下：

```text
hvm_precomputed/
├── metadata.jsonl          # [Stage 1] 基础元数据（idx, id, description, parquet定位）
├── id_to_idx.json          # [Stage 1] 原始ID → 整数索引映射
├── text_embeddings.npy     # [Stage 2] CLIP 文本向量 [N, 768] float32
├── faiss_index.bin         # [Stage 2] FAISS 索引（IndexFlatIP）
├── rag_results.jsonl       # [Stage 2] Top-3 检索结果
├── groups.jsonl            # [Stage 4] SVG 路径分组信息
├── features/               # [Stage 3] 整图 post-merge 视觉特征 (for GME)
│   ├── 000/
│   │   ├── 000000.pt       # [256, 3584] float16 (~1.8MB)
│   │   ├── 000001.pt
│   │   └── ...
│   ├── 001/
│   └── ...
└── group_features/         # [Stage 5] 逐 group 独立渲染特征 (for PME)
    ├── 000/
    │   ├── 000000.pt       # list of [256, 3584] float16, 1~4 个 group
    │   ├── 000001.pt
    │   └── ...
    ├── 001/
    └── ...
```

---

## 🛠️ 常见问题 (FAQ)

### Q: 为什么 Features 和 Group Features 都用 post-merge 特征？

统一使用 Qwen2.5-VL merger 输出的 post-merge `[256, 3584]` 特征有多个好处：
1. **复用 Qwen 训练好的 merger**，特征质量有保障
2. **GME 和 PME 统一 d_vision=3584**，架构简洁
3. **GME QFormer 输入从 3072 tokens 降到 768 tokens**，训练快 4 倍
4. **不需要 hook / reverse_indices**，代码简洁可靠
5. **存储省 30%**：1.8MB vs 旧版 2.6MB per image

### Q: 为什么 RAG 检索用 CLIP 而不是 Qwen 的 embed_tokens？

CLIP 经过大规模文本-图像对比学习（contrastive learning），其文本编码器的 embedding 空间天然具有语义聚类特性，适合做最近邻检索。而 Qwen 的 `embed_tokens` 只是 LLM 的输入嵌入层，其设计目标是为下游 Transformer 层提供初始表征，没有经过检索目标（如对比学习）的训练。对其做 mean pooling 后进行余弦相似度检索，语义区分度远不如 CLIP。

### Q: 为什么 RAG 检索要排除自身？

在训练时，RAG 的目的是提供"参考示例"（reference examples）。如果检索结果包含自身，模型学会的是"复制粘贴"而不是"模仿风格"，这会导致过拟合和信息泄露。

### Q: Features 和 Group Features 阶段支持断点续传吗？

**支持。** 如果检测到目标 `.pt` 文件已存在，脚本会自动跳过。可以安全地中断后重新运行。其他阶段（Metadata, RAG, Groups）通常运行较快，默认会覆盖重写。

### Q: Group Features 渲染失败怎么办？

`render_group_to_image` 使用 `cairosvg` 渲染 SVG group。如果某个 group 渲染失败（如 SVG 格式异常），该 group 会被跳过。如果一个样本的所有 group 都渲染失败，该样本会被标记为 skipped，不会产生 `.pt` 文件，训练时该样本会被 dataset 过滤掉。

### Q: CLIP 的 77 token 限制会不会截断描述？

CLIP tokenizer 的最大长度为 77 tokens。描述较长时会被截断，但由于 `description` 放在前面、`keywords` 放在后面，核心语义信息通常不会丢失。对于极长的描述，被截断的主要是末尾的关键词部分，对检索质量影响有限。
