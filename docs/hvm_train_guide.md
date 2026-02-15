# HVM-SVG 训练手册

本手册聚焦「怎么稳定跑起来 + 怎么排查问题」。

---

## 1. 训练目标与默认流程

HVM-SVG 在 OmniSVG 基础上：

- 冻结 base model
- 只训练 HVM 模块（GME / PME / PIM / Gate）
- 使用离线预计算 RAG 与视觉特征

默认训练入口：

- `run_train_hvm.sh`
- 实际 Python 入口：`train_hvm.py`

---

## 2. 先决条件

建议先确认：

1. 环境可用
   - Python 环境：`/mnt/data/wuqingman/miniconda3/envs/omnisvg`
   - `accelerate` 与 deepspeed 配置文件可用
2. 预计算数据完整
   - `metadata.jsonl`
   - `rag_results.jsonl`
   - `groups.jsonl`
   - `features/`
3. 三个 jsonl 条数一致

---

## 3. 预计算数据（训练前）

脚本：`build_faiss_index/precompute_hvm_data.py`

四阶段：

1. `metadata`
2. `rag`
3. `features`
4. `groups`

示例：

```bash
python build_faiss_index/precompute_hvm_data.py --stage metadata
python build_faiss_index/precompute_hvm_data.py --stage rag
python build_faiss_index/precompute_hvm_data.py --stage features --gpu_id 0
python build_faiss_index/precompute_hvm_data.py --stage groups
```

---

## 4. 启动训练

### 4.1 常用命令

8 卡默认：

```bash
bash run_train_hvm.sh
```

单卡调试：

```bash
bash run_train_hvm.sh --num_gpus 1 --batch_size 1
```

短跑验证（推荐）：

```bash
bash run_train_hvm.sh --epochs 5 --save_every 50 --warmup 20
```

---

## 5. 恢复与初始化

### 5.1 恢复完整训练状态

```bash
bash run_train_hvm.sh --resume /path/to/checkpoint-step-XXXX
```

对应 `train_hvm.py --resume_from`，恢复：

- model/deepspeed state
- optimizer
- scheduler
- global step

### 5.2 仅加载 HVM 权重初始化

```bash
bash run_train_hvm.sh --hvm_ckpt /path/to/hvm_step_XXXX.pt
```

对应 `train_hvm.py --hvm_checkpoint`，只加载 HVM 参数，不恢复优化器/步数。

说明：`--resume` 和 `--hvm_ckpt` 互斥。

---

## 6. 关键参数建议

### 6.1 先看有效 batch size

有效 batch size = `batch_size * grad_accum * num_gpus`  
默认是 `2 * 4 * 8 = 64`。

### 6.2 warmup 建议显式指定

`train_hvm.py` 默认 warmup = `total_steps * 10%`。  
当 `epochs` 很大（如 30000），warmup 会过长，前期 loss 看起来不收敛。

建议实验阶段固定：

- `--warmup 500`（或 1000/2000）

### 6.3 学习率建议

起步可试：

- `1e-4`（默认）
- 若抖动大可试 `5e-5`

---

## 7. 日志怎么看

你会看到：

- `loss`: 当前 step loss
- `avg`: 近一段滑窗平均
- `lr`: 当前学习率
- `Step XXXX | Gates: [...]`: 每个 PIM 的 `tanh(alpha)`

Gate 初期接近 0 正常；如果长期非常接近 0，可检查 lr/warmup/weight decay。

---

## 8. 输出目录结构

默认输出：`outputs_hvm/`

常见文件：

- `hvm_config.json`
- `hvm_model_config.json`
- `checkpoint-step-XXXX/`
  - `training_metadata.json`
  - `lr_scheduler.pt`
- `hvm_step_XXXX.pt`
- `swanlog/`

---

## 9. 常见问题排查

### 9.1 loss 不收敛

优先检查：

1. warmup 是否过长
2. 当前 lr 是否仍很低
3. 数据量是否偏小（step 抖动会明显）
4. gate 是否长期接近 0

### 9.2 训练可跑但显存异常增长

检查是否有临时文件/数据重试问题；长训时建议定期观察 `/tmp` 与 worker 行为。

### 9.3 恢复后进度不对

确认你用的是 `--resume` 而不是 `--hvm_ckpt`。

---

## 10. 推荐实验流程

1. 单卡 smoke test（2~5 epoch）
2. 多卡短跑（100~500 epoch）
3. 固定一组配置跑 ablation
4. 再拉长训练

建议每次只改一个变量（warmup / lr / group策略 / PIM层间隔）。

