#!/bin/bash
# =============================================================================
# HVM-SVG 训练启动脚本
# =============================================================================
#
# 使用方法:
#   bash run_train_hvm.sh --num_gpus 1       # 单卡调试
#   bash run_train_hvm.sh --num_gpus 6       # 8 卡训练
#   CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 bash run_train_hvm.sh --num_gpus 7
#   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash run_train_hvm.sh --num_gpus 6 --warmup 200
#   CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 bash run_train_hvm.sh --num_gpus 7 --resume /mnt/data2/wuqingman/omnisvg-train/outputs_hvm/checkpoint-step-4000  # 恢复完整训练状态
#   bash run_train_hvm.sh --hvm_ckpt /mnt/data/wuqingman/omnisvg-train/outputs_hvm/hvm_step_5000.pt     # 仅加载 HVM 权重初始化

#   # Stage1 (GME-only + last-layer single PIM + fixed scale)
#   # CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash run_train_hvm.sh --num_gpus 6 --memory_mode gme --inject_mode fixed --inject_scale 0.03 --pim_layer_indices -1 --output_dir ./outputs_s1_fixed003
#   # CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash run_train_hvm.sh --num_gpus 6 --memory_mode gme --inject_mode fixed --inject_scale 0.1  --pim_layer_indices -1 --output_dir ./outputs_s1_fixed010


#   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash run_train_hvm.sh --num_gpus 6 --disable_hvm --epochs 3000 --output_dir ./outputs_baseline --run_name "baseline-no-hvm" --save_every 999999
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash run_train_hvm.sh --num_gpus 6 --memory_mode gme --inject_mode fixed --inject_scale 0.03 --pim_layer_indices -1 --run_name s1_gme_last1_fixed003 --output_dir /mnt/data2/wuqingman/omnisvg-train/outputs_s1_fixed0.03
#  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash run_train_hvm.sh --num_gpus 6 --memory_mode gme --inject_mode fixed --inject_scale 0.03 --pim_layer_indices -1 --shuffle_rag --run_name s1_gme_last1_fixed003_shuffle --output_dir ./outputs_s1_fixed0.03_shuffle

#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash run_train_hvm.sh --num_gpus 6 --memory_mode gme --inject_mode fixed --inject_scale 0.03 --pim_layer_indices -1 --shuffle_rag --run_name s1_gme_last1_fixed003_allshuffle --output_dir ./outputs_s1_fixed0.03_allshuffle




set -e
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ===================== 环境配置 =====================
export CUDA_HOME="/mnt/data/wuqingman/miniconda3/envs/omnisvg"
# export CUDA_HOME="/usr/local/cuda-12.1"
# export PATH=$CUDA_HOME/bin:$PATH
export TOKENIZERS_PARALLELISM=false
PYTHON="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/python"
ACCELERATE="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/accelerate"

# ===================== 训练参数 =====================

# -- GPU --
NUM_GPUS=6

# -- 数据 --
DATA_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test"
HVM_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_1w"

# -- 模型 --
MODEL_SIZE="8B"
OMNISVG_CHECKPOINT=""               # 留空使用默认路径

# -- HVM 架构 --
D_QFORMER=1024                      # QFormer 内部维度
D_PIM_INNER=512                     # PIM attention bottleneck 维度
PIM_LAYER_INTERVAL=4                # 每隔 N 层插入 PIM
GATE_ALPHA_INIT=0.0                # AdaptiveGate 冷启动初值 (tanh后约等于本值)
GME_NUM_QUERIES=32                  # GME QFormer query 数量
MEMORY_MODE="full"                  # full / gme
INJECT_MODE="adaptive"              # adaptive / fixed
INJECT_SCALE=0.1                    # fixed 注入强度
PIM_LAYER_INDICES=""                # 逗号分隔层索引，支持 -1 表示最后一层

# -- 训练超参 --
BATCH_SIZE=4                        # 每卡 batch size
GRAD_ACCUM=4                        # 梯度累积步数
EPOCHS=30000
LEARNING_RATE=5e-4
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
WARMUP_STEPS=100
SEED=42
MIXED_PRECISION="bf16"              # 混合精度: bf16 / fp16 / no
ACCELERATE_CONFIG="./configs/ds_zero2_hvm.yaml"  # DeepSpeed ZeRO-2 (float32 optimizer states)

# -- DataLoader --
NUM_WORKERS=4

# -- 日志与保存 --
OUTPUT_DIR="/mnt/data2/wuqingman/omnisvg-train/outputs_hvm_2026_02_28_20_04"
LOG_EVERY=10
SAVE_EVERY=2000
SWANLAB_MODE="cloud"                # cloud / local / disabled
SWANLAB_RUN_NAME=""                 # 留空自动生成

# -- 恢复训练 --
RESUME_FROM=""
# HVM_CHECKPOINT="/mnt/data2/wuqingman/omnisvg-train/outputs_hvm_2026_02_23_00_18/hvm_step_5000.pt"
HVM_CHECKPOINT=""

# -- 验证集 --
VAL_DATA_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_val"
VAL_HVM_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_val"
EVAL_EVERY=500                      # 每 N 个 optimizer step 评估一次 val loss

# -- Ablation --
DISABLE_HVM=false                   # true: baseline 模式，不注入 HVM (只跑冻结 OmniSVG)
SHUFFLE_RAG=false                   # true: 打乱 batch 内 ref_features 对应关系 (验证 RAG 信息 vs 参数效应)
DELTA_LN=false                      # true: delta 上加 LayerNorm 稳定 scale

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
        --gate_alpha_init) GATE_ALPHA_INIT="$2";    shift 2 ;;
        --gme_num_queries) GME_NUM_QUERIES="$2";   shift 2 ;;
        --memory_mode)    MEMORY_MODE="$2";         shift 2 ;;
        --inject_mode)    INJECT_MODE="$2";         shift 2 ;;
        --inject_scale)   INJECT_SCALE="$2";        shift 2 ;;
        --pim_layer_indices|--pim_layers) PIM_LAYER_INDICES="$2"; shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";          shift 2 ;;
        --data_dir)       DATA_DIR="$2";            shift 2 ;;
        --hvm_dir)        HVM_DIR="$2";             shift 2 ;;
        --swanlab_mode)   SWANLAB_MODE="$2";        shift 2 ;;
        --run_name)       SWANLAB_RUN_NAME="$2";    shift 2 ;;
        --save_every)     SAVE_EVERY="$2";          shift 2 ;;
        --val_data_dir)   VAL_DATA_DIR="$2";        shift 2 ;;
        --val_hvm_dir)    VAL_HVM_DIR="$2";         shift 2 ;;
        --eval_every)     EVAL_EVERY="$2";          shift 2 ;;
        --no_val)         VAL_DATA_DIR=""; VAL_HVM_DIR=""; shift 1 ;;
        --disable_hvm)    DISABLE_HVM=true;         shift 1 ;;
        --shuffle_rag)    SHUFFLE_RAG=true;         shift 1 ;;
        --delta_ln)       DELTA_LN=true;            shift 1 ;;
        --dra_d_inner)    DRA_D_INNER="$2";         shift 2 ;;
        --dra_n_heads)    DRA_N_HEADS="$2";         shift 2 ;;
        --cdm_num_queries) CDM_NUM_QUERIES="$2";    shift 2 ;;
        --cdm_num_layers) CDM_NUM_LAYERS="$2";      shift 2 ;;
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
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "HVM-SVG Training Configuration"
echo "============================================================"
echo "  GPUs:              ${NUM_GPUS}"
echo "  Batch size/GPU:    ${BATCH_SIZE}"
echo "  Gradient accum:    ${GRAD_ACCUM}"
echo "  Effective BS:      ${EFFECTIVE_BS}"
echo "  Epochs:            ${EPOCHS}"
echo "  Learning rate:     ${LEARNING_RATE}"
echo "  Gate alpha init:   ${GATE_ALPHA_INIT}"
echo "  GME num queries:   ${GME_NUM_QUERIES}"
echo "  Memory mode:       ${MEMORY_MODE}"
echo "  Inject mode:       ${INJECT_MODE}"
echo "  Inject scale:      ${INJECT_SCALE}"
if [ -n "$PIM_LAYER_INDICES" ]; then
    echo "  PIM layers:        ${PIM_LAYER_INDICES}"
else
    echo "  PIM interval:      every ${PIM_LAYER_INTERVAL} layers"
fi
echo "  Mixed precision:   ${MIXED_PRECISION}"
echo "  Warmup steps:      ${WARMUP_STEPS}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "  SwanLab mode:      ${SWANLAB_MODE}"
echo "  Run name:          ${SWANLAB_RUN_NAME}"
if [ -n "$VAL_DATA_DIR" ] && [ -n "$VAL_HVM_DIR" ]; then
    echo "  Val data dir:      ${VAL_DATA_DIR}"
    echo "  Val HVM dir:       ${VAL_HVM_DIR}"
    echo "  Eval every:        ${EVAL_EVERY} steps"
else
    echo "  Validation:        DISABLED"
fi
if [ "$DISABLE_HVM" = true ]; then
    echo "  *** BASELINE MODE: HVM DISABLED ***"
fi
if [ "$SHUFFLE_RAG" = true ]; then
    echo "  *** SHUFFLE RAG ABLATION: ref correspondence broken ***"
fi
if [ "$DELTA_LN" = true ]; then
    echo "  *** DELTA LAYERNORM: enabled ***"
fi
if [ "$MEMORY_MODE" = "gme_dra" ]; then
    echo "  DRA d_inner:       ${DRA_D_INNER:-128}"
    echo "  DRA n_heads:       ${DRA_N_HEADS:-4}"
fi
if [ "$MEMORY_MODE" = "gme_cdm" ]; then
    echo "  CDM queries:       ${CDM_NUM_QUERIES:-16}"
    echo "  CDM layers:        ${CDM_NUM_LAYERS:-6}"
fi
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
    --memory_mode "$MEMORY_MODE"
    --inject_mode "$INJECT_MODE"
    --inject_scale "$INJECT_SCALE"
    --gate_alpha_init "$GATE_ALPHA_INIT"
    --gme_num_queries "$GME_NUM_QUERIES"
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

if [ -n "$PIM_LAYER_INDICES" ]; then
    TRAIN_ARGS+=(--pim_layer_indices "$PIM_LAYER_INDICES")
fi

# Val dataset
if [ -n "$VAL_DATA_DIR" ] && [ -n "$VAL_HVM_DIR" ]; then
    TRAIN_ARGS+=(--val_data_dir "$VAL_DATA_DIR" --val_hvm_dir "$VAL_HVM_DIR" --eval_every "$EVAL_EVERY")
fi

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

# Ablation: baseline 模式
if [ "$DISABLE_HVM" = true ]; then
    TRAIN_ARGS+=(--disable_hvm)
fi

# Ablation: shuffle RAG
if [ "$SHUFFLE_RAG" = true ]; then
    TRAIN_ARGS+=(--shuffle_rag)
fi

# Ablation: delta LayerNorm
if [ "$DELTA_LN" = true ]; then
    TRAIN_ARGS+=(--delta_ln)
fi

# DRA 参数
if [ -n "$DRA_D_INNER" ]; then
    TRAIN_ARGS+=(--dra_d_inner "$DRA_D_INNER")
fi
if [ -n "$DRA_N_HEADS" ]; then
    TRAIN_ARGS+=(--dra_n_heads "$DRA_N_HEADS")
fi

# CDM 参数
if [ -n "$CDM_NUM_QUERIES" ]; then
    TRAIN_ARGS+=(--cdm_num_queries "$CDM_NUM_QUERIES")
fi
if [ -n "$CDM_NUM_LAYERS" ]; then
    TRAIN_ARGS+=(--cdm_num_layers "$CDM_NUM_LAYERS")
fi

# ===================== 保存启动快照 =====================
cp "$SCRIPT_PATH" "${OUTPUT_DIR}/run_train_hvm.snapshot.sh"

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
