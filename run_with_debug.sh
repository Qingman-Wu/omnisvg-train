#!/bin/bash
# 启用 debug 模式运行训练，保存训练样本用于 tokenizer 验证
# 用法: bash run_with_debug.sh [num_samples]

set -e

NUM_SAMPLES=${1:-10}

echo "=============================================="
echo "运行训练并保存样本（Debug 模式）"
echo "=============================================="
echo "将保存 ${NUM_SAMPLES} 个训练样本"
echo ""

# 设置环境变量，run.sh 会读取这些变量
export SAVE_TRAIN_SAMPLES="true"
export MAX_TRAIN_SAMPLES="${NUM_SAMPLES}"
export TRAIN_SAMPLES_DIR="./train_samples_debug"

# 直接调用 run.sh
bash run.sh

echo ""
echo "=============================================="
echo "训练完成或已停止"
echo "=============================================="

# 检查是否成功保存了样本
if [ -d "${TRAIN_SAMPLES_DIR}" ]; then
    NUM_SAVED=$(ls -1 ${TRAIN_SAMPLES_DIR}/sample_*.json 2>/dev/null | wc -l)
    if [ $NUM_SAVED -gt 0 ]; then
        echo "成功保存了 ${NUM_SAVED} 个样本到 ${TRAIN_SAMPLES_DIR}"
    else
        echo "警告: 没有找到保存的样本文件"
    fi
else
    echo "警告: 样本目录不存在: ${TRAIN_SAMPLES_DIR}"
fi
echo ""
