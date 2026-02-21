#!/bin/bash
# =============================================================================
# HVM-SVG 训练启动脚本
# =============================================================================
#
# 使用方法:
#   bash run_train_hvm.sh --num_gpus 1       # 单卡调试
#   bash run_train_hvm.sh --num_gpus 6       # 8 卡训练
#   CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 bash run_train_hvm.sh --num_gpus 7
#   bash run_train_hvm.sh --resume /mnt/data/wuqingman/omnisvg-train/outputs_hvm/checkpoint-step-5000  # 恢复完整训练状态
#   bash run_train_hvm.sh --hvm_ckpt /mnt/data/wuqingman/omnisvg-train/outputs_hvm/hvm_step_5000.pt     # 仅加载 HVM 权重初始化
#
# =============================================================================

set -e

# ===================== 环境配置 =====================
export CUDA_HOME="/mnt/data/wuqingman/miniconda3/envs/omnisvg"
export TOKENIZERS_PARALLELISM=false
PYTHON="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/python"
ACCELERATE="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/accelerate"

# ===================== 训练参数 =====================

# -- GPU --
NUM_GPUS=8                          # GPU 数量

# -- 数据 --
DATA_DIR="/mnt/data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test"
HVM_DIR="/mnt/data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed"

# -- 模型 --
MODEL_SIZE="8B"
OMNISVG_CHECKPOINT=""               # 留空使用默认路径

# -- HVM 架构 --
D_QFORMER=1024                      # QFormer 内部维度
D_PIM_INNER=512                     # PIM attention bottleneck 维度
PIM_LAYER_INTERVAL=4                # 每隔 N 层插入 PIM

# -- 训练超参 --
BATCH_SIZE=4                        # 每卡 batch size
GRAD_ACCUM=4                        # 梯度累积步数
EPOCHS=30000
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
WARMUP_STEPS=""  # 为空时自动使用 10% of total steps
SEED=42
MIXED_PRECISION="bf16"              # 混合精度: bf16 / fp16 / no
ACCELERATE_CONFIG="./configs/ds_zero2_hvm.yaml"  # DeepSpeed ZeRO-2 (float32 optimizer states)

# -- DataLoader --
NUM_WORKERS=4

# -- 日志与保存 --
OUTPUT_DIR="/mnt/data2/wuqingman/omnisvg-train/outputs_hvm"
LOG_EVERY=10
SAVE_EVERY=1000
SWANLAB_MODE="cloud"                # cloud / local / disabled
SWANLAB_RUN_NAME=""                 # 留空自动生成

# -- 恢复训练 --
RESUME_FROM=""                      # 完整训练 checkpoint 目录 (含 optimizer/scheduler/step)
HVM_CHECKPOINT=""                   # HVM-only 权重 .pt (仅加载模型权重, 不恢复训练状态)

# ===================== 解析命令行覆盖 =====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)       NUM_GPUS="$2";           shift 2 ;;
        --batch_size)     BATCH_SIZE="$2";          shift 2 ;;
        --grad_accum)     GRAD_ACCUM="$2";          shift 2 ;;
        --epochs)         EPOCHS="$2";              shift 2 ;;
        --lr)             LEARNING_RATE="$2";       shift 2 ;;
        --warmup)         WARMUP_STEPS="$2";        shift 2 ;;
        --resume)         RESUME_FROM="$2";         shift 2 ;;
        --hvm_ckpt)       HVM_CHECKPOINT="$2";      shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";          shift 2 ;;
        --data_dir)       DATA_DIR="$2";            shift 2 ;;
        --hvm_dir)        HVM_DIR="$2";             shift 2 ;;
        --swanlab_mode)   SWANLAB_MODE="$2";        shift 2 ;;
        --run_name)       SWANLAB_RUN_NAME="$2";    shift 2 ;;
        --save_every)     SAVE_EVERY="$2";          shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 参数互斥检查
if [ -n "$RESUME_FROM" ] && [ -n "$HVM_CHECKPOINT" ]; then
    echo "Error: --resume and --hvm_ckpt are mutually exclusive."
    echo "  --resume:   restore full training state (optimizer/scheduler/step)"
    echo "  --hvm_ckpt: load HVM-only weights for initialization"
    exit 1
fi

# ===================== 自动计算 =====================
EFFECTIVE_BS=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

# 自动生成 run name
if [ -z "$SWANLAB_RUN_NAME" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SWANLAB_RUN_NAME="hvm-${MODEL_SIZE}-bs${EFFECTIVE_BS}-lr${LEARNING_RATE}-${TIMESTAMP}"
fi

# ===================== 打印配置 =====================
echo "============================================================"
echo "HVM-SVG Training Configuration"
echo "============================================================"
echo "  GPUs:              ${NUM_GPUS}"
echo "  Batch size/GPU:    ${BATCH_SIZE}"
echo "  Gradient accum:    ${GRAD_ACCUM}"
echo "  Effective BS:      ${EFFECTIVE_BS}"
echo "  Epochs:            ${EPOCHS}"
echo "  Learning rate:     ${LEARNING_RATE}"
echo "  Mixed precision:   ${MIXED_PRECISION}"
echo "  Warmup steps:      ${WARMUP_STEPS}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "  SwanLab mode:      ${SWANLAB_MODE}"
echo "  Run name:          ${SWANLAB_RUN_NAME}"
if [ -n "$RESUME_FROM" ]; then
    echo "  Resume from:       ${RESUME_FROM}"
elif [ -n "$HVM_CHECKPOINT" ]; then
    echo "  Init HVM weights:  ${HVM_CHECKPOINT}"
fi
echo "============================================================"

# ===================== 构建训练命令 =====================
TRAIN_ARGS=(
    --model_size "$MODEL_SIZE"
    --data_dir "$DATA_DIR"
    --hvm_dir "$HVM_DIR"
    --d_qformer "$D_QFORMER"
    --d_pim_inner "$D_PIM_INNER"
    --pim_layer_interval "$PIM_LAYER_INTERVAL"
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --epochs "$EPOCHS"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --max_grad_norm "$MAX_GRAD_NORM"
    --seed "$SEED"
    --output_dir "$OUTPUT_DIR"
    --log_every "$LOG_EVERY"
    --save_every "$SAVE_EVERY"
    --num_workers "$NUM_WORKERS"
    --swanlab_mode "$SWANLAB_MODE"
    --swanlab_run_name "$SWANLAB_RUN_NAME"
)

# Warmup steps (为空时由 train_hvm.py 自动计算为 10% of total)
if [ -n "$WARMUP_STEPS" ]; then
    TRAIN_ARGS+=(--warmup_steps "$WARMUP_STEPS")
fi

# OmniSVG checkpoint
if [ -n "$OMNISVG_CHECKPOINT" ]; then
    TRAIN_ARGS+=(--omnisvg_checkpoint "$OMNISVG_CHECKPOINT")
fi

# Resume: 完整训练恢复 (--resume_from) 和 HVM 权重初始化 (--hvm_checkpoint) 互斥
if [ -n "$RESUME_FROM" ]; then
    TRAIN_ARGS+=(--resume_from "$RESUME_FROM")
elif [ -n "$HVM_CHECKPOINT" ]; then
    TRAIN_ARGS+=(--hvm_checkpoint "$HVM_CHECKPOINT")
fi

# ===================== 启动训练 =====================
# 使用 DeepSpeed ZeRO-2 训练:
#   - 模型参数保持 bf16 (省显存)
#   - DeepSpeed 自动维护 float32 optimizer states (训练精度)
#   - Optimizer states 分片到多卡 (进一步省显存)
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Starting single-GPU training (mixed precision: ${MIXED_PRECISION})..."
    $ACCELERATE launch \
        --num_processes 1 \
        --mixed_precision "$MIXED_PRECISION" \
        train_hvm.py "${TRAIN_ARGS[@]}"
else
    echo "Starting ${NUM_GPUS}-GPU training (DeepSpeed ZeRO-2, ${MIXED_PRECISION})..."
    $ACCELERATE launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_processes "$NUM_GPUS" \
        train_hvm.py "${TRAIN_ARGS[@]}"
fi
