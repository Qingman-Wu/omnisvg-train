# HVM-SVG 数据预处理工具 (`precompute_hvm_data.py`)

本目录下的 `precompute_hvm_data.py` 是 HVM-SVG 训练流程中的核心数据处理脚本。它负责将原始的 SVG Parquet 数据集转换为训练所需的多模态预计算数据，包括元数据索引、RAG 检索结果、视觉特征缓存以及 SVG 路径分组信息。

先看一下原始omnisvg数据（/mnt/data/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_process)
<img width="746" height="816" alt="image" src="https://github.com/user-attachments/assets/5ef18874-90c9-4b83-828f-06e0911294fa" />


## 📋 功能概述

该脚本包含四个处理阶段（Stage），支持独立运行或全流程运行：

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

3. **Stage 3: Features (视觉特征)**
   - 使用 Qwen2.5-VL 的 Vision Encoder 提取图像特征。
   - **关键特性**：提取的是 **Pre-merge** 特征，保留了更细粒度的空间信息。
   - 输出形状：`[16, 16, 4, 1280]`（对应 16×16 的 Grid，每个 Grid 包含 4 个 Patch，维度 1280）。
   - 支持多 GPU / 多节点并行分片处理，支持断点续传。

4. **Stage 4: Groups (路径分组)**
   - 解析 SVG 源码，提取所有 Path 指令。
   - 计算每个 Path 的几何复杂度（基于指令类型和覆盖面积）。
   - 根据复杂度将 Path 智能划分为 1~4 个组（Group），用于分层生成。

## ⚙️ 环境依赖

请确保已安装以下 Python 库：

```bash
pip install torch numpy pandas pyarrow pillow tqdm transformers faiss-cpu qwen_vl_utils
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
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--stage` | str | `all` | 运行阶段：`metadata`, `rag`, `features`, `groups`, 或 `all` |
| `--data_dir` | str | (见源码) | 原始 Parquet 数据目录路径 |
| `--model_path` | str | (见源码) | Qwen2.5-VL 模型权重目录路径 |
| `--clip_model_path` | str | (见源码) | CLIP 模型路径（用于 RAG 文本编码） |
| `--output_dir` | str | (见源码) | 预计算结果输出目录 |
| `--gpu_id` | int | `0` | 指定使用的 GPU ID（仅用于 Features 阶段） |
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


### Stage 3: Features

- **依赖**：Metadata 阶段。需要加载 Qwen2.5-VL Vision Encoder。
- **过程**：
  1. 只加载 Vision Encoder 到 GPU，释放 LLM 部分以节省显存。
  2. 通过 `register_forward_pre_hook` 在 merger 之前截取 pre-merge 特征。
  3. 处理 window attention 的 shuffle：通过 `get_window_index` 获取 reverse indices，恢复空间顺序。
  4. 每张图输出 `[16, 16, 4, 1280]` 的张量。
- **输出**：
  - `features/{idx//1000:03d}/{idx:06d}.pt`：PyTorch Tensor 文件。
  - 目录结构：按 `idx // 1000` 分子文件夹存储，避免单目录文件过多。
  - Tensor Shape：`[16, 16, 4, 1280]`（float16）。

#### ⚡️ 并行提取特征示例（推荐）

特征提取较慢，建议在多张 GPU 上并行运行：

**Terminal 1 (GPU 0):**
```bash
python precompute_hvm_data.py --stage features --gpu_id 0 --num_shards 4 --shard_id 0
```

**Terminal 2 (GPU 1):**
```bash
python precompute_hvm_data.py --stage features --gpu_id 1 --num_shards 4 --shard_id 1
```

**Terminal 3 (GPU 2):**
```bash
python precompute_hvm_data.py --stage features --gpu_id 2 --num_shards 4 --shard_id 2
```

**Terminal 4 (GPU 3):**
```bash
python precompute_hvm_data.py --stage features --gpu_id 3 --num_shards 4 --shard_id 3
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
    - `bbox_feature`：映射到 16×16 Grid 的坐标 `[row_start, row_end, col_start, col_end]`
    - `complexity`：组复杂度分数

#### 分组策略

| 总复杂度 | Path 数 | 分组数 |
| :--- | :--- | :--- |
| < 30 | ≤ 2 | 1 |
| < 80 | ≤ 5 | 2 |
| < 150 | — | 3 |
| ≥ 150 | — | 4 |

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
└── features/               # [Stage 3] Pre-merge 视觉特征
    ├── 000/
    │   ├── 000000.pt       # [16, 16, 4, 1280] float16
    │   ├── 000001.pt
    │   └── ...
    ├── 001/
    └── ...
```

---

## 🛠️ 常见问题 (FAQ)

### Q: 为什么 Features 阶段要用 pre-merge 特征？

Qwen2.5-VL 原生会对视觉 Token 进行 spatial merge（2×2 patches → 1 unit）以减少序列长度。为了在生成 SVG 时保持与空间位置的强对齐，我们需要 merge 前的原始特征。这样可以获得固定的 `16×16` 空间网格，其中每个 grid 位置保留了 4 个原始 patch 的完整信息（`[4, 1280]`），有利于作为空间 Condition 输入到 PME（Patch-level Modulated Encoder）。

### Q: 为什么 RAG 检索用 CLIP 而不是 Qwen 的 embed_tokens？

CLIP 经过大规模文本-图像对比学习（contrastive learning），其文本编码器的 embedding 空间天然具有语义聚类特性，适合做最近邻检索。而 Qwen 的 `embed_tokens` 只是 LLM 的输入嵌入层，其设计目标是为下游 Transformer 层提供初始表征，没有经过检索目标（如对比学习）的训练。对其做 mean pooling 后进行余弦相似度检索，语义区分度远不如 CLIP。

### Q: 为什么 RAG 检索要排除自身？

在训练时，RAG 的目的是提供"参考示例"（reference examples）。如果检索结果包含自身，模型学会的是"复制粘贴"而不是"模仿风格"，这会导致过拟合和信息泄露。

### Q: Features 阶段支持断点续传吗？

**支持。** 如果检测到目标 `.pt` 文件已存在，脚本会自动跳过。可以安全地中断后重新运行。其他阶段（Metadata, RAG, Groups）通常运行较快，默认会覆盖重写。

### Q: window_index 和 reverse_indices 是什么？

Qwen2.5-VL 的 Vision Encoder 使用 window attention，会对 patch 顺序进行 shuffle。`get_window_index` 返回 shuffle 后的索引，`torch.argsort` 得到逆映射（reverse indices），用于将 pre-merge 特征恢复到正确的空间位置。这是保证 `[16, 16, 4, 1280]` 中空间对齐的关键步骤。

### Q: CLIP 的 77 token 限制会不会截断描述？

CLIP tokenizer 的最大长度为 77 tokens。描述较长时会被截断，但由于 `description` 放在前面、`keywords` 放在后面，核心语义信息通常不会丢失。对于极长的描述，被截断的主要是末尾的关键词部分，对检索质量影响有限。


```
