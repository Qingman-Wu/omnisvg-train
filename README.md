# HVM-SVG 训练与实验手册


## 1. 项目目标与核心思路

HVM-SVG 用于缓解 OmniSVG 在复杂样本上的长序列退化（重复 path 命令、循环输出、被 max length 截断）。

核心做法：

1. 对每个训练样本离线检索 Top-3 参考样本（RAG）
2. 参考图像预计算视觉特征（不在训练时跑 vision encoder）
3. 用两类记忆编码器压缩视觉信息
   - GME（Gist Memory Encoder）：全局记忆，输入 3 张参考图的整图 post-merge 特征
   - PME（Part Memory Encoder）：局部零件记忆，输入 Top-1 参考图的逐 group 独立渲染特征
4. 在 OmniSVG decoder 的若干层后插入 PIM
5. 通过 Adaptive Gate 把视觉修正量注入 hidden states

---

## 2. 代码结构（HVM 相关）

核心文件：

- `build_faiss_index/precompute_hvm_data.py`
  - 离线预计算：metadata / RAG / feature / path groups / group features
- `build_faiss_index/run_precompute.sh`
  - 一键运行全部 5 个 stage（支持多 GPU 并行提取特征）
- `hvm_dataset.py`
  - 训练数据集与 collate，读取预计算结果
- `hvm_modules.py`
  - HVMConfig、QFormer、GME、PME、PIM、AdaptiveGate
- `hvm_decoder.py`
  - 在 OmniSVG base decoder 上通过 hook 注入 PIM
- `train_hvm.py`
  - 训练主脚本（Accelerate + DeepSpeed）
- `run_train_hvm.sh`
  - 一键启动脚本（参数与恢复逻辑）

---

## 3. 架构总览（与当前实现一致）

### 3.1 输入

每个 batch 训练侧输入主要包括：

- 文本+SVG 序列：`input_ids`, `attention_mask`, `labels`
- 参考视觉特征：
  - `ref_features`: `[B, 3, 256, 3584]`（Top-3 参考图的整图 post-merge 特征, for GME）
  - `group_features_list`: `List[List[Tensor]]`，每个 Tensor 为 `[256, 3584]`（Top-1 参考图逐 group 独立渲染特征, for PME）
- 参考文本：
  - `ref_text_ids`: `[B, Nt]`
  - `ref_text_mask`: `[B, Nt]`

### 3.2 视觉记忆构建

```
┌─── GME (Global) ───┐     ┌─── PME (Local) ────┐
│                     │     │                     │
│ 3张参考图整图特征     │     │ Top-1参考逐group     │
│ [B, 3, 256, 3584]  │     │ 独立渲染后特征        │
│ (post-merge)       │     │ list of [256, 3584] │
│       │            │     │ (post-merge)        │
│       ▼            │     │       │             │
│ QFormer (6 层)     │     │ QFormer (4 层)      │
│ 768 tokens → 32 Q  │     │ 256 tok → 4 Q/group │
│       │            │     │       │             │
│       ▼            │     │       ▼             │
│ gist_feats         │     │ part_feats          │
│ [B, 32, 3584]      │     │ [B, 16, 3584]       │
└─────────────────────┘     └─────────────────────┘
         │                           │
         └─────────┬─────────────────┘
                   ▼
         ┌─── PIM × 7 ──────────────────┐
         │ Step 1: Part × Gist cross-attn│
         │ Step 2: Part self-attn        │
         │ Step 3: Visual × Text cross   │
         │ Step 4: Hidden × Aligned cross│
         │ Step 5: AdaptiveGate          │
         └───────────────────────────────┘
                   ↓
         注入 decoder hidden states
```

1. **GME**
   - 输入：3 张参考图 post-merge features `[B, 3, 256, 3584]`
   - 展平：`[B, 768, 3584]`（3×256 tokens）
   - QFormer 32 queries × 6 layers
   - 输出：`gist_feats [B, 32, 3584]`
2. **PME**
   - 输入：Top-1 参考图逐 group 独立渲染特征，每组 `[256, 3584]`
   - 每组 QFormer 4 queries × 4 layers
   - 最多 4 组，输出：
     - `part_feats [B, 16, 3584]`（padded）
     - `part_mask [B, 16]`

### 3.3 PIM 注入流程

每个 PIM 执行：

1. Part × Gist cross-attn
2. Part self-attn
3. Visual × Text cross-attn（会使用 `text_mask`）
4. Hidden × Aligned cross-attn 得到 `delta`
5. AdaptiveGate：`hidden + gate * delta`

注意：PIM 是通过 `hvm_decoder.py` 的 decoder layer hook 注入，不改 base model 代码。

### 3.4 冻结策略

- OmniSVG base model 全冻结
- 仅训练：
  - `gme`（~106M params）
  - `pme`（~72M params）
  - `pims` × 7（~251M params）
  - 共 ~429M params，占 base model 4.7%

---

## 4. 离线预计算数据

脚本：`build_faiss_index/precompute_hvm_data.py`

### 4.1 五个 stage

1. `metadata`
   - 输出：`metadata.jsonl`, `id_to_idx.json`
2. `rag`
   - 输出：`rag_results.jsonl`, `text_embeddings.npy`, `faiss_index.bin`
3. `features`（整图特征, for GME）
   - 输出：`features/{idx//1000}/{idx}.pt`，每个 `[256, 3584]` float16
4. `groups`
   - 输出：`groups.jsonl`
5. `group_features`（逐 group 渲染特征, for PME）
   - 输出：`group_features/{idx//1000}/{idx}.pt`，每个 list of `[256, 3584]` float16

### 4.2 推荐执行方式

一键运行（推荐，自动 8 GPU 并行提取特征和 group 特征）：

```bash
cd build_faiss_index
bash run_precompute.sh
```



### 4.3 预计算目录检查

至少应包含：

- `metadata.jsonl`
- `rag_results.jsonl`
- `groups.jsonl`
- `features/`（每个样本一个 `.pt`）
- `group_features/`（每个有 groups 信息的样本一个 `.pt`）

并且三份 jsonl 记录数应一致。

---

## 5. 训练启动

主入口：`run_train_hvm.sh`

### 5.1 常用命令

默认 8 卡：

```bash
bash run_train_hvm.sh
```

单卡调试：

```bash
bash run_train_hvm.sh --num_gpus 1 --batch_size 1
```

恢复完整训练状态（含 optimizer/scheduler/step）：

```bash
bash run_train_hvm.sh --resume /path/to/checkpoint-step-XXXX
```

只加载 HVM 权重初始化：

```bash
bash run_train_hvm.sh --hvm_ckpt /path/to/hvm_step_XXXX.pt
```

### 5.2 恢复逻辑说明

- `--resume` 对应 `train_hvm.py --resume_from`
  - 恢复 model + optimizer + scheduler + global_step
- `--hvm_ckpt` 对应 `train_hvm.py --hvm_checkpoint`
  - 只加载 HVM 模块参数，不恢复训练状态
- 两者互斥，脚本和 `train_hvm.py` 都会检查

---

## 6. 关键训练参数与建议

当前默认（脚本）：

- `batch_size=2`（每卡）
- `grad_accum=4`
- `num_gpus=8`
- 有效 batch size：`2*4*8=64`

### 6.1 重点：warmup 与总步数

`train_hvm.py` 默认 `warmup_steps = total_steps * 10%`。  
若 `epochs` 设很大（如 30000），warmup 会非常长，前期学习率非常小，看起来像"不收敛"。

建议实验期显式传入 `--warmup`，例如 500~2000。


### 6.2 Gate 观察

训练日志会打印：

- `Step xxxx | Gates: [...]`

含义：`tanh(base_alpha)`，初始接近 0 是正常的。  
如果长期几乎不变，可检查 lr / weight_decay 配置。

---

## 7. 训练输出文件

`output_dir`（默认 `outputs_hvm`）中常见内容：

- `hvm_config.json`：运行参数快照
- `hvm_model_config.json`：HVM 配置（含 PIM 层索引）
- `checkpoint-step-XXXX/`
  - accelerate 完整状态
  - `training_metadata.json`
  - `lr_scheduler.pt`
- `hvm_step_XXXX.pt`
  - HVM-only 轻量权重
- `swanlog/`
  - SwanLab 本地日志

---


## 9. 张量维度速查表

### 预计算特征

- 整图特征（for GME）：`[256, 3584]` per image, float16
- Group 特征（for PME）：`list of [256, 3584]` per sample, 1~4 个 group, float16

### 训练 batch

- `ref_features`: `[B, 3, 256, 3584]`（3 张参考图整图特征, for GME）
- `group_features_list`: `List[List[Tensor]]`（Top-1 参考逐 group 特征, for PME）
- `gist_feats`: `[B, 32, 3584]`（GME 输出）
- `part_feats`: `[B, 16, 3584]`（PME 输出, padded）
- `part_mask`: `[B, 16]`（bool）
- `ref_text_ids`: `[B, Nt]`
- `ref_text_mask`: `[B, Nt]`
- `text_feats`: `[B, Nt, 3584]`
- `hidden_state`: `[B, L, 3584]`
- `delta`: `[B, L, 3584]`

### HVMConfig 关键参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `d_model` | 3584 | LLM hidden dim (Qwen2.5-7B) |
| `d_vision` | 3584 | Vision post-merge dim (GME & PME 统一) |
| `d_qformer` | 1024 | QFormer 内部维度 |
| `d_pim_inner` | 512 | PIM attention bottleneck 维度 |
| `gme_num_queries` | 32 | GME queries |
| `gme_num_layers` | 6 | GME QFormer 层数 |
| `pme_queries_per_group` | 4 | PME 每 group queries |
| `pme_num_layers` | 4 | PME QFormer 层数 |
| `pme_max_groups` | 4 | 最大 group 数 |
| `pim_layer_interval` | 4 | 每隔 N 层插入 PIM |

---