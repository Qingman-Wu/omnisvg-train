#!/bin/bash
# =============================================================================
# HVM-SVG 训练启动脚本 (A100_1_2 / EDR top1)
# =============================================================================
#
# 使用方法:
#   bash run_train_hvm_a100_1_2.sh
#   CUDA_VISIBLE_DEVICES=3,4,5 bash run_train_hvm_a100_1_2.sh --num_gpus 3
#
# 默认实验:
#   GME + CDM + EDR(E1) + last4 + adaptive
#   稀疏路由变体: top_k=1 (每个 token 只保留 1 个 detail slot)

set -e
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ===================== 环境配置 =====================
export CUDA_HOME="/mnt/data/wuqingman/miniconda3/envs/omnisvg"
export TOKENIZERS_PARALLELISM=false
PYTHON="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/python"
ACCELERATE="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/accelerate"

# ===================== 训练参数 =====================

# -- GPU --
NUM_GPUS=3

# -- 数据 --
DATA_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test"
HVM_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_1w"

# -- 模型 --
MODEL_SIZE="8B"
OMNISVG_CHECKPOINT=""

# -- HVM 架构 --
D_QFORMER=1024
D_PIM_INNER=512
PIM_LAYER_INTERVAL=4
GATE_ALPHA_INIT=0.05
GME_NUM_QUERIES=32
CDM_NUM_QUERIES=16
CDM_NUM_LAYERS=6
EDR_D_ROUTER=256
EDR_TOP_K=1
EDR_DISABLE_CONF=false
MEMORY_MODE="gme_cdm_edr"
INJECT_MODE="adaptive"
INJECT_SCALE=0.1
PIM_LAYER_INDICES="24,25,26,27"

# -- 训练超参 --
# 3 卡保持与 6 卡主实验接近的有效 batch: 4 x 8 x 3 = 96
BATCH_SIZE=4
GRAD_ACCUM=8
EPOCHS=30000
LEARNING_RATE=5e-4
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
WARMUP_STEPS=100
SEED=42
MIXED_PRECISION="bf16"
ACCELERATE_CONFIG="./configs/ds_zero2_hvm.yaml"

# -- DataLoader --
NUM_WORKERS=4

# -- 日志与保存 --
OUTPUT_DIR="/mnt/data2/wuqingman/omnisvg-train/outputs_s4_gme_cdm_edr_e1_topk1_last4"
LOG_EVERY=10
SAVE_EVERY=2000
SWANLAB_MODE="cloud"
SWANLAB_RUN_NAME="s4_gme_cdm_edr_e1_topk1_last4"

# -- 恢复训练 --
RESUME_FROM=""
HVM_CHECKPOINT=""

# -- 验证集 --
VAL_DATA_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_val"
VAL_HVM_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_val"
EVAL_EVERY=500

# -- Ablation --
DISABLE_HVM=false
SHUFFLE_RAG=false
DELTA_LN=false

# ===================== 解析命令行覆盖 =====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)       NUM_GPUS="$2";             shift 2 ;;
        --batch_size)     BATCH_SIZE="$2";           shift 2 ;;
        --grad_accum)     GRAD_ACCUM="$2";           shift 2 ;;
        --epochs)         EPOCHS="$2";               shift 2 ;;
        --lr)             LEARNING_RATE="$2";        shift 2 ;;
        --warmup)         WARMUP_STEPS="$2";         shift 2 ;;
        --resume)         RESUME_FROM="$2";          shift 2 ;;
        --hvm_ckpt)       HVM_CHECKPOINT="$2";       shift 2 ;;
        --gate_alpha_init) GATE_ALPHA_INIT="$2";     shift 2 ;;
        --gme_num_queries) GME_NUM_QUERIES="$2";     shift 2 ;;
        --cdm_num_queries) CDM_NUM_QUERIES="$2";     shift 2 ;;
        --cdm_num_layers) CDM_NUM_LAYERS="$2";       shift 2 ;;
        --edr_d_router)   EDR_D_ROUTER="$2";         shift 2 ;;
        --edr_top_k)      EDR_TOP_K="$2";            shift 2 ;;
        --edr_disable_conf) EDR_DISABLE_CONF=true;   shift 1 ;;
        --memory_mode)    MEMORY_MODE="$2";          shift 2 ;;
        --inject_mode)    INJECT_MODE="$2";          shift 2 ;;
        --inject_scale)   INJECT_SCALE="$2";         shift 2 ;;
        --pim_layer_indices|--pim_layers) PIM_LAYER_INDICES="$2"; shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";           shift 2 ;;
        --data_dir)       DATA_DIR="$2";             shift 2 ;;
        --hvm_dir)        HVM_DIR="$2";              shift 2 ;;
        --swanlab_mode)   SWANLAB_MODE="$2";         shift 2 ;;
        --run_name)       SWANLAB_RUN_NAME="$2";     shift 2 ;;
        --save_every)     SAVE_EVERY="$2";           shift 2 ;;
        --val_data_dir)   VAL_DATA_DIR="$2";         shift 2 ;;
        --val_hvm_dir)    VAL_HVM_DIR="$2";          shift 2 ;;
        --eval_every)     EVAL_EVERY="$2";           shift 2 ;;
        --no_val)         VAL_DATA_DIR=""; VAL_HVM_DIR=""; shift 1 ;;
        --disable_hvm)    DISABLE_HVM=true;          shift 1 ;;
        --shuffle_rag)    SHUFFLE_RAG=true;          shift 1 ;;
        --delta_ln)       DELTA_LN=true;             shift 1 ;;
        --dra_d_inner)    DRA_D_INNER="$2";          shift 2 ;;
        --dra_n_heads)    DRA_N_HEADS="$2";          shift 2 ;;
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

if [ -z "$SWANLAB_RUN_NAME" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SWANLAB_RUN_NAME="hvm-${MODEL_SIZE}-bs${EFFECTIVE_BS}-lr${LEARNING_RATE}-${TIMESTAMP}"
fi

# ===================== 生成 DeepSpeed 配置 =====================
ACCELERATE_CONFIG_RUNTIME="${OUTPUT_DIR}/ds_zero2_hvm_runtime.yaml"
mkdir -p "${OUTPUT_DIR}"
sed "s/gradient_accumulation_steps:.*/gradient_accumulation_steps: ${GRAD_ACCUM}/" \
    ./configs/ds_zero2_hvm.yaml > "${ACCELERATE_CONFIG_RUNTIME}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG_RUNTIME}"

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
if [ "$MEMORY_MODE" = "gme_cdm" ] || [ "$MEMORY_MODE" = "gme_cdm_edr" ]; then
    echo "  CDM queries:       ${CDM_NUM_QUERIES}"
    echo "  CDM layers:        ${CDM_NUM_LAYERS}"
fi
if [ "$MEMORY_MODE" = "gme_cdm_edr" ]; then
    echo "  EDR d_router:      ${EDR_D_ROUTER}"
    echo "  EDR top-k:         ${EDR_TOP_K}"
    echo "  EDR disable conf:  ${EDR_DISABLE_CONF}"
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
    --cdm_num_queries "$CDM_NUM_QUERIES"
    --cdm_num_layers "$CDM_NUM_LAYERS"
    --edr_d_router "$EDR_D_ROUTER"
    --edr_top_k "$EDR_TOP_K"
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

if [ -n "$VAL_DATA_DIR" ] && [ -n "$VAL_HVM_DIR" ]; then
    TRAIN_ARGS+=(--val_data_dir "$VAL_DATA_DIR" --val_hvm_dir "$VAL_HVM_DIR" --eval_every "$EVAL_EVERY")
fi

if [ -n "$WARMUP_STEPS" ]; then
    TRAIN_ARGS+=(--warmup_steps "$WARMUP_STEPS")
fi

if [ -n "$OMNISVG_CHECKPOINT" ]; then
    TRAIN_ARGS+=(--omnisvg_checkpoint "$OMNISVG_CHECKPOINT")
fi

if [ -n "$RESUME_FROM" ]; then
    TRAIN_ARGS+=(--resume_from "$RESUME_FROM")
elif [ -n "$HVM_CHECKPOINT" ]; then
    TRAIN_ARGS+=(--hvm_checkpoint "$HVM_CHECKPOINT")
fi

if [ "$DISABLE_HVM" = true ]; then
    TRAIN_ARGS+=(--disable_hvm)
fi

if [ "$SHUFFLE_RAG" = true ]; then
    TRAIN_ARGS+=(--shuffle_rag)
fi

if [ "$DELTA_LN" = true ]; then
    TRAIN_ARGS+=(--delta_ln)
fi

if [ -n "$DRA_D_INNER" ]; then
    TRAIN_ARGS+=(--dra_d_inner "$DRA_D_INNER")
fi
if [ -n "$DRA_N_HEADS" ]; then
    TRAIN_ARGS+=(--dra_n_heads "$DRA_N_HEADS")
fi

if [ "$EDR_DISABLE_CONF" = true ]; then
    TRAIN_ARGS+=(--edr_disable_conf)
fi

# ===================== 保存启动快照 =====================
cp "$SCRIPT_PATH" "${OUTPUT_DIR}/run_train_hvm.snapshot.sh"

# ===================== 启动训练 =====================
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
