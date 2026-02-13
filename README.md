# Tokenizer Debug 功能

本次提交新增了 **训练样本提取和 tokenizer 验证** 功能，用于排查训练时 tokenizer 可能存在的 bug。

## 背景

训练代码的 tokenizer 配置可能存在问题（如 `tokenization.yaml` 中的 token ID 配置错误），导致训练时的 token 序列与推理时不一致。为了验证这一点，我们需要：

1. 从训练过程中提取真实的 `input_ids` 和原始 GT SVG
2. 用推理端的 tokenizer（已验证正确）decode `input_ids`
3. 对比 decoded SVG 与原始 GT SVG

## 新增文件

### 训练端 (`omnisvg-train/`)

| 文件 | 说明 |
|------|------|
| `run_with_debug.sh` | Debug 模式启动脚本，保存训练样本 |
| `train_samples_debug/` | 保存的训练样本目录（运行时生成） |

### 推理端 (`omnisvg-inference/`)

| 文件 | 说明 |
|------|------|
| `verify_train_samples.py` | 验证脚本，decode 训练样本并保存 SVG |
| `analyze_tokenizer_diff.py` | 分析 token 序列，识别特殊 token、命令等 |
| `verification_output/` | 验证结果输出目录 |

### 根目录 (`/mnt/data/wuqingman/`)

| 文件 | 说明 |
|------|------|
| `verify_train_samples.sh` | 一键验证脚本 |

## 代码修改

### 1. `utils/dataset.py`

- `__getitem__` 返回值从 3 个增加到 4 个
- 新增返回 `original_svg`（数据集中的原始 SVG 字符串）

```python
# 修改前
def __getitem__(self, index) -> Tuple[str, Image.Image, List[int]]:
    return text, image, tokens.tolist()

# 修改后
def __getitem__(self, index) -> Tuple[str, Image.Image, List[int], str]:
    return text, image, tokens.tolist(), svg_code
```

### 2. `train.py`

- `collate_fn` 接收 4 个值，返回 `original_svgs`
- 训练循环接收 `original_svgs`
- 新增 debug 模式：当环境变量 `SAVE_TRAIN_SAMPLES=true` 时，保存训练样本

```python
# 环境变量控制
SAVE_TRAIN_SAMPLES=true   # 启用保存
MAX_TRAIN_SAMPLES=10      # 保存数量
TRAIN_SAMPLES_DIR=./train_samples_debug  # 保存目录
```

### 3. `run.sh`

- 新增 debug 配置区块
- 支持通过环境变量覆盖默认值

## 使用方法

### 步骤 1: 保存训练样本

```bash
cd /mnt/data/wuqingman/omnisvg-train

# 保存 20 个训练样本
bash run_with_debug.sh 20

# 看到 "[DEBUG] Finished saving 20 samples" 后按 Ctrl+C
```

### 步骤 2: 验证样本

```bash
cd /mnt/data/wuqingman
bash verify_train_samples.sh
```

### 步骤 3: 查看结果

```bash
# 查看汇总
cat omnisvg-inference/verification_output/verification_summary.json

# 对比 SVG（在浏览器中）
firefox omnisvg-inference/verification_output/sample_0000_gt.svg \
        omnisvg-inference/verification_output/sample_0000_from_input_ids.svg
```

## 输出文件说明

每个样本生成以下文件：

| 文件 | 说明 |
|------|------|
| `sample_XXXX_gt.svg` | 原始 GT SVG（直接从数据集获取） |
| `sample_XXXX_from_input_ids.svg` | 从 input_ids decode 的 SVG |
| `sample_XXXX_tokens.py` | Python 格式的 token 列表（可复制到 inference.py 测试） |
| `sample_XXXX_info.json` | 详细对比信息 |

## 结果判断

### ✅ 正常

- `sample_XXXX_gt.svg` 和 `sample_XXXX_from_input_ids.svg` **显示一致**
- `info.json` 中 `tokens_match: true`

### ❌ 异常

- 两个 SVG 显示不同 → 训练 tokenizer 有 bug
- decode 失败 → token ID 配置错误
- `tokens_match: false` → input_ids 构建逻辑有问题

## 注意事项

1. **Debug 模式会影响训练性能**，仅用于调试
2. 样本保存后可以**停止训练**（Ctrl+C），无需完成整个训练
3. 验证脚本使用**推理端的 tokenizer**，确保 decode 逻辑正确
4. 正常训练时请使用 `bash run.sh`（不保存样本）

## 恢复正常训练

Debug 模式不会修改 `run.sh` 的默认配置。直接运行 `bash run.sh` 即可正常训练。

如果需要清理 debug 样本：
```bash
rm -rf /mnt/data/wuqingman/omnisvg-train/train_samples_debug
```
