#!/bin/bash
# =============================================================================
# HVM-SVG 训练脚本 (A100_3 / EDR only-detail)
# =============================================================================
#
# 用法:
#   CUDA_VISIBLE_DEVICES=4,5,7 bash run_train_hvm_a100_3.sh
#   CUDA_VISIBLE_DEVICES=4,5,7 bash run_train_hvm_a100_3.sh --num_gpus 3
#
# 当前实验:
#   s4_gme_cdm_edr_e1_onlydetail_last4
#   GME + CDM + EDR(E1), 关闭 gist injection，保留 detail router 全部逻辑:
#       inject = gate_detail * conf * routed_detail

set -e
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ===================== 环境配置 =====================
export CUDA_HOME="/usr/local/cuda-12.1"
export TOKENIZERS_PARALLELISM=false
PYTHON="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/python"
ACCELERATE="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/accelerate"

# ===================== 实验配置 =====================

NUM_GPUS=3

# -- 数据 (共享盘) --
DATA_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test"
HVM_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_1w"
VAL_DATA_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_val"
VAL_HVM_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_val"

# -- 模型 --
MODEL_SIZE="8B"
OMNISVG_CHECKPOINT="/mnt/a100_1_data2/wuqingman/models/OmniSVG/OmniSVG1.1_8B"

# -- HVM 架构 --
D_QFORMER=1024
D_PIM_INNER=512
MEMORY_MODE="gme_cdm_edr"
INJECT_MODE="adaptive"
INJECT_SCALE=0.1
PIM_LAYER_INDICES="24,25,26,27"
GATE_ALPHA_INIT=0.05
GME_NUM_QUERIES=32
DELTA_LN=false
DRA_D_INNER=128
DRA_N_HEADS=4
CDM_NUM_QUERIES=16
CDM_NUM_LAYERS=6
EDR_D_ROUTER=256
EDR_TOP_K=2
EDR_DISABLE_CONF=false
EDR_DISABLE_GIST=true

# -- 训练超参 --
# 3 卡: bs4 × grad_accum8 × 3gpu = 96
BATCH_SIZE=4
GRAD_ACCUM=8
EPOCHS=30000
LEARNING_RATE=5e-4
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
WARMUP_STEPS=100
SEED=42
MIXED_PRECISION="bf16"

# -- DataLoader --
NUM_WORKERS=4

# -- 日志与保存 --
OUTPUT_DIR="/mnt/a100_1_data3/wuqingman/omnisvg-train/outputs_s4_gme_cdm_edr_e1_onlydetail_last4"
LOG_EVERY=10
SAVE_EVERY=2000
EVAL_EVERY=500
SWANLAB_MODE="cloud"
SWANLAB_RUN_NAME="s4_gme_cdm_edr_e1_onlydetail_last4"

# -- 恢复训练 --
RESUME_FROM=""
HVM_CHECKPOINT=""

# -- Ablation --
DISABLE_HVM=false
SHUFFLE_RAG=false

# ===================== 解析命令行覆盖 =====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)        NUM_GPUS="$2";              shift 2 ;;
        --batch_size)      BATCH_SIZE="$2";            shift 2 ;;
        --grad_accum)      GRAD_ACCUM="$2";            shift 2 ;;
        --epochs)          EPOCHS="$2";                shift 2 ;;
        --lr)              LEARNING_RATE="$2";         shift 2 ;;
        --warmup)          WARMUP_STEPS="$2";          shift 2 ;;
        --resume)          RESUME_FROM="$2";           shift 2 ;;
        --hvm_ckpt)        HVM_CHECKPOINT="$2";        shift 2 ;;
        --gate_alpha_init) GATE_ALPHA_INIT="$2";       shift 2 ;;
        --gme_num_queries) GME_NUM_QUERIES="$2";       shift 2 ;;
        --memory_mode)     MEMORY_MODE="$2";           shift 2 ;;
        --inject_mode)     INJECT_MODE="$2";           shift 2 ;;
        --inject_scale)    INJECT_SCALE="$2";          shift 2 ;;
        --pim_layer_indices|--pim_layers) PIM_LAYER_INDICES="$2"; shift 2 ;;
        --output_dir)      OUTPUT_DIR="$2";            shift 2 ;;
        --run_name)        SWANLAB_RUN_NAME="$2";      shift 2 ;;
        --save_every)      SAVE_EVERY="$2";            shift 2 ;;
        --eval_every)      EVAL_EVERY="$2";            shift 2 ;;
        --no_val)          VAL_DATA_DIR=""; VAL_HVM_DIR=""; shift 1 ;;
        --disable_hvm)     DISABLE_HVM=true;           shift 1 ;;
        --shuffle_rag)     SHUFFLE_RAG=true;           shift 1 ;;
        --delta_ln)        DELTA_LN=true;              shift 1 ;;
        --no_delta_ln)     DELTA_LN=false;             shift 1 ;;
        --dra_d_inner)     DRA_D_INNER="$2";           shift 2 ;;
        --dra_n_heads)     DRA_N_HEADS="$2";           shift 2 ;;
        --cdm_num_queries) CDM_NUM_QUERIES="$2";       shift 2 ;;
        --cdm_num_layers)  CDM_NUM_LAYERS="$2";        shift 2 ;;
        --edr_d_router)    EDR_D_ROUTER="$2";          shift 2 ;;
        --edr_top_k)       EDR_TOP_K="$2";             shift 2 ;;
        --edr_disable_conf) EDR_DISABLE_CONF=true;     shift 1 ;;
        --no_edr_disable_conf) EDR_DISABLE_CONF=false; shift 1 ;;
        --edr_disable_gist) EDR_DISABLE_GIST=true;     shift 1 ;;
        --no_edr_disable_gist) EDR_DISABLE_GIST=false; shift 1 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 参数互斥检查
if [ -n "$RESUME_FROM" ] && [ -n "$HVM_CHECKPOINT" ]; then
    echo "Error: --resume and --hvm_ckpt are mutually exclusive."
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
echo "HVM-SVG Training (A100_3)"
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
echo "  PIM layers:        ${PIM_LAYER_INDICES}"
echo "  Delta LayerNorm:   ${DELTA_LN}"
if [ "$MEMORY_MODE" = "gme_dra" ]; then
    echo "  DRA d_inner:       ${DRA_D_INNER}"
    echo "  DRA n_heads:       ${DRA_N_HEADS}"
fi
if [ "$MEMORY_MODE" = "gme_cdm" ] || [ "$MEMORY_MODE" = "gme_cdm_edr" ]; then
    echo "  CDM queries:       ${CDM_NUM_QUERIES}"
    echo "  CDM layers:        ${CDM_NUM_LAYERS}"
fi
if [ "$MEMORY_MODE" = "gme_cdm_edr" ]; then
    echo "  EDR d_router:      ${EDR_D_ROUTER}"
    echo "  EDR top-k:         ${EDR_TOP_K}"
    echo "  EDR disable conf:  ${EDR_DISABLE_CONF}"
    echo "  EDR disable gist:  ${EDR_DISABLE_GIST}"
fi
echo "  Mixed precision:   ${MIXED_PRECISION}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "  Run name:          ${SWANLAB_RUN_NAME}"
if [ -n "$VAL_DATA_DIR" ] && [ -n "$VAL_HVM_DIR" ]; then
    echo "  Eval every:        ${EVAL_EVERY} steps"
else
    echo "  Validation:        DISABLED"
fi
if [ "$DISABLE_HVM" = true ]; then
    echo "  *** BASELINE MODE ***"
fi
if [ "$SHUFFLE_RAG" = true ]; then
    echo "  *** SHUFFLE RAG ***"
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

if [ "$MEMORY_MODE" = "gme_dra" ]; then
    TRAIN_ARGS+=(--dra_d_inner "$DRA_D_INNER" --dra_n_heads "$DRA_N_HEADS")
fi

if [ "$MEMORY_MODE" = "gme_cdm" ] || [ "$MEMORY_MODE" = "gme_cdm_edr" ]; then
    TRAIN_ARGS+=(--cdm_num_queries "$CDM_NUM_QUERIES" --cdm_num_layers "$CDM_NUM_LAYERS")
fi

if [ "$MEMORY_MODE" = "gme_cdm_edr" ]; then
    TRAIN_ARGS+=(--edr_d_router "$EDR_D_ROUTER" --edr_top_k "$EDR_TOP_K")
    if [ "$EDR_DISABLE_CONF" = true ]; then
        TRAIN_ARGS+=(--edr_disable_conf)
    fi
    if [ "$EDR_DISABLE_GIST" = true ]; then
        TRAIN_ARGS+=(--edr_disable_gist)
    fi
fi

# ===================== 保存启动快照 =====================
cp "$SCRIPT_PATH" "${OUTPUT_DIR}/run_train_snapshot.sh"

# ===================== 启动训练 =====================
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Starting single-GPU training (${MIXED_PRECISION})..."
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
#!/bin/bash
# =============================================================================
# HVM-SVG 训练脚本 (A100_3 / EDR 无置信度抑制)
# =============================================================================
#
# 用法:
#   CUDA_VISIBLE_DEVICES=4,5,7 bash run_train_hvm_a100_3.sh
#   CUDA_VISIBLE_DEVICES=4,5,7 bash run_train_hvm_a100_3.sh --num_gpus 3
#
# 当前实验:
#   s4_gme_cdm_edr_e1_noconf_last4
#   GME + CDM + EDR(E1), 但关闭 conf 抑制:
#       inject_detail = gate_detail * 1 * delta_detail

set -e
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ===================== 环境配置 =====================
export CUDA_HOME="/usr/local/cuda-12.1"
export TOKENIZERS_PARALLELISM=false
PYTHON="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/python"
ACCELERATE="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/accelerate"

# ===================== 实验配置 =====================

NUM_GPUS=3

# -- 数据 (共享盘) --
DATA_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test"
HVM_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_1w"
VAL_DATA_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_val"
VAL_HVM_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_val"

# -- 模型 --
MODEL_SIZE="8B"
OMNISVG_CHECKPOINT="/mnt/a100_1_data2/wuqingman/models/OmniSVG/OmniSVG1.1_8B"

# -- HVM 架构 --
D_QFORMER=1024
D_PIM_INNER=512
MEMORY_MODE="gme_cdm_edr"
INJECT_MODE="adaptive"
INJECT_SCALE=0.1
PIM_LAYER_INDICES="24,25,26,27"
GATE_ALPHA_INIT=0.05
GME_NUM_QUERIES=32
DELTA_LN=false
DRA_D_INNER=128
DRA_N_HEADS=4
CDM_NUM_QUERIES=16
CDM_NUM_LAYERS=6
EDR_D_ROUTER=256
EDR_TOP_K=2
EDR_DISABLE_CONF=true

# -- 训练超参 --
# 3 卡: bs4 × grad_accum8 × 3gpu = 96
BATCH_SIZE=4
GRAD_ACCUM=8
EPOCHS=30000
LEARNING_RATE=5e-4
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
WARMUP_STEPS=100
SEED=42
MIXED_PRECISION="bf16"

# -- DataLoader --
NUM_WORKERS=4

# -- 日志与保存 --
OUTPUT_DIR="/mnt/a100_1_data3/wuqingman/omnisvg-train/outputs_s4_gme_cdm_edr_e1_noconf_last4"
LOG_EVERY=10
SAVE_EVERY=2000
EVAL_EVERY=500
SWANLAB_MODE="cloud"
SWANLAB_RUN_NAME="s4_gme_cdm_edr_e1_noconf_last4"

# -- 恢复训练 --
RESUME_FROM=""
HVM_CHECKPOINT=""

# -- Ablation --
DISABLE_HVM=false
SHUFFLE_RAG=false

# ===================== 解析命令行覆盖 =====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)        NUM_GPUS="$2";            shift 2 ;;
        --batch_size)      BATCH_SIZE="$2";          shift 2 ;;
        --grad_accum)      GRAD_ACCUM="$2";          shift 2 ;;
        --epochs)          EPOCHS="$2";              shift 2 ;;
        --lr)              LEARNING_RATE="$2";       shift 2 ;;
        --warmup)          WARMUP_STEPS="$2";        shift 2 ;;
        --resume)          RESUME_FROM="$2";         shift 2 ;;
        --hvm_ckpt)        HVM_CHECKPOINT="$2";      shift 2 ;;
        --gate_alpha_init) GATE_ALPHA_INIT="$2";     shift 2 ;;
        --gme_num_queries) GME_NUM_QUERIES="$2";     shift 2 ;;
        --memory_mode)     MEMORY_MODE="$2";         shift 2 ;;
        --inject_mode)     INJECT_MODE="$2";         shift 2 ;;
        --inject_scale)    INJECT_SCALE="$2";        shift 2 ;;
        --pim_layer_indices|--pim_layers) PIM_LAYER_INDICES="$2"; shift 2 ;;
        --output_dir)      OUTPUT_DIR="$2";          shift 2 ;;
        --run_name)        SWANLAB_RUN_NAME="$2";    shift 2 ;;
        --save_every)      SAVE_EVERY="$2";          shift 2 ;;
        --eval_every)      EVAL_EVERY="$2";          shift 2 ;;
        --no_val)          VAL_DATA_DIR=""; VAL_HVM_DIR=""; shift 1 ;;
        --disable_hvm)     DISABLE_HVM=true;         shift 1 ;;
        --shuffle_rag)     SHUFFLE_RAG=true;         shift 1 ;;
        --delta_ln)        DELTA_LN=true;            shift 1 ;;
        --no_delta_ln)     DELTA_LN=false;           shift 1 ;;
        --dra_d_inner)     DRA_D_INNER="$2";         shift 2 ;;
        --dra_n_heads)     DRA_N_HEADS="$2";         shift 2 ;;
        --cdm_num_queries) CDM_NUM_QUERIES="$2";     shift 2 ;;
        --cdm_num_layers)  CDM_NUM_LAYERS="$2";      shift 2 ;;
        --edr_d_router)    EDR_D_ROUTER="$2";        shift 2 ;;
        --edr_top_k)       EDR_TOP_K="$2";           shift 2 ;;
        --edr_disable_conf) EDR_DISABLE_CONF=true;   shift 1 ;;
        --no_edr_disable_conf) EDR_DISABLE_CONF=false; shift 1 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 参数互斥检查
if [ -n "$RESUME_FROM" ] && [ -n "$HVM_CHECKPOINT" ]; then
    echo "Error: --resume and --hvm_ckpt are mutually exclusive."
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
echo "HVM-SVG Training (A100_3)"
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
echo "  PIM layers:        ${PIM_LAYER_INDICES}"
echo "  Delta LayerNorm:   ${DELTA_LN}"
if [ "$MEMORY_MODE" = "gme_dra" ]; then
    echo "  DRA d_inner:       ${DRA_D_INNER}"
    echo "  DRA n_heads:       ${DRA_N_HEADS}"
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
echo "  Mixed precision:   ${MIXED_PRECISION}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "  Run name:          ${SWANLAB_RUN_NAME}"
if [ -n "$VAL_DATA_DIR" ] && [ -n "$VAL_HVM_DIR" ]; then
    echo "  Eval every:        ${EVAL_EVERY} steps"
else
    echo "  Validation:        DISABLED"
fi
if [ "$DISABLE_HVM" = true ]; then
    echo "  *** BASELINE MODE ***"
fi
if [ "$SHUFFLE_RAG" = true ]; then
    echo "  *** SHUFFLE RAG ***"
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

if [ "$MEMORY_MODE" = "gme_dra" ]; then
    TRAIN_ARGS+=(--dra_d_inner "$DRA_D_INNER" --dra_n_heads "$DRA_N_HEADS")
fi

if [ "$MEMORY_MODE" = "gme_cdm" ] || [ "$MEMORY_MODE" = "gme_cdm_edr" ]; then
    TRAIN_ARGS+=(--cdm_num_queries "$CDM_NUM_QUERIES" --cdm_num_layers "$CDM_NUM_LAYERS")
fi

if [ "$MEMORY_MODE" = "gme_cdm_edr" ]; then
    TRAIN_ARGS+=(--edr_d_router "$EDR_D_ROUTER" --edr_top_k "$EDR_TOP_K")
    if [ "$EDR_DISABLE_CONF" = true ]; then
        TRAIN_ARGS+=(--edr_disable_conf)
    fi
fi

# ===================== 保存启动快照 =====================
cp "$SCRIPT_PATH" "${OUTPUT_DIR}/run_train_snapshot.sh"

# ===================== 启动训练 =====================
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Starting single-GPU training (${MIXED_PRECISION})..."
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
#!/bin/bash
# =============================================================================
# HVM-SVG 训练脚本 (a100_2 单卡/少卡版)
# =============================================================================
#
# 用法:
#   CUDA_VISIBLE_DEVICES=0 bash run_train_a100_2.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash run_train_a100_2.sh --num_gpus 2
#
# 当前实验:
#   s3_cdm_last4 (GME + Complementary Detail Memory)
#   对比: s2_gme_last4_adaptive_gate (GME-only adaptive)
#   核心假设: 16 个 learnable queries 先看 gist 再从 ref 中提取互补细节,
#             能否捕获 GME 32 gist tokens 遗漏的信息？

set -e
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ===================== 环境配置 =====================
export CUDA_HOME="/usr/local/cuda-12.1"
export TOKENIZERS_PARALLELISM=false
PYTHON="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/python"
ACCELERATE="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/accelerate"

# ===================== 实验配置 =====================

NUM_GPUS=3

# -- 数据 (共享盘) --
DATA_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test"
HVM_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_1w"
VAL_DATA_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_val"
VAL_HVM_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_val"

# -- 模型 --
MODEL_SIZE="8B"
OMNISVG_CHECKPOINT="/mnt/a100_1_data2/wuqingman/models/OmniSVG/OmniSVG1.1_8B"

# -- HVM 架构 --
D_QFORMER=1024
D_PIM_INNER=512
MEMORY_MODE="gme_cdm"
INJECT_MODE="adaptive"
INJECT_SCALE=0.1
PIM_LAYER_INDICES="24,25,26,27"
GATE_ALPHA_INIT=0.05
GME_NUM_QUERIES=32
DELTA_LN=false
DRA_D_INNER=128
DRA_N_HEADS=4
CDM_NUM_QUERIES=16
CDM_NUM_LAYERS=6

# -- 训练超参 --
# 3卡: bs4 × grad_accum8 × 3gpu = 96 (与 a100_1 的 bs4 × 4 × 6 = 96 一致)
BATCH_SIZE=4
GRAD_ACCUM=8
EPOCHS=30000
LEARNING_RATE=5e-4
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
WARMUP_STEPS=100
SEED=42
MIXED_PRECISION="bf16"

# -- DataLoader --
NUM_WORKERS=4

# -- 日志与保存 --
OUTPUT_DIR="/mnt/a100_1_data3/wuqingman/omnisvg-train/outputs_s3_cdm_last4"
LOG_EVERY=10
SAVE_EVERY=2000
EVAL_EVERY=500
SWANLAB_MODE="cloud"
SWANLAB_RUN_NAME="s3_cdm_last4"

# -- 恢复训练 --
RESUME_FROM=""
HVM_CHECKPOINT=""

# -- Ablation --
DISABLE_HVM=false
SHUFFLE_RAG=false

# ===================== 解析命令行覆盖 =====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)        NUM_GPUS="$2";           shift 2 ;;
        --batch_size)      BATCH_SIZE="$2";          shift 2 ;;
        --grad_accum)      GRAD_ACCUM="$2";          shift 2 ;;
        --epochs)          EPOCHS="$2";              shift 2 ;;
        --lr)              LEARNING_RATE="$2";       shift 2 ;;
        --warmup)          WARMUP_STEPS="$2";        shift 2 ;;
        --resume)          RESUME_FROM="$2";         shift 2 ;;
        --hvm_ckpt)        HVM_CHECKPOINT="$2";      shift 2 ;;
        --gate_alpha_init) GATE_ALPHA_INIT="$2";     shift 2 ;;
        --gme_num_queries) GME_NUM_QUERIES="$2";    shift 2 ;;
        --memory_mode)     MEMORY_MODE="$2";         shift 2 ;;
        --inject_mode)     INJECT_MODE="$2";         shift 2 ;;
        --inject_scale)    INJECT_SCALE="$2";        shift 2 ;;
        --pim_layer_indices|--pim_layers) PIM_LAYER_INDICES="$2"; shift 2 ;;
        --output_dir)      OUTPUT_DIR="$2";          shift 2 ;;
        --run_name)        SWANLAB_RUN_NAME="$2";    shift 2 ;;
        --save_every)      SAVE_EVERY="$2";          shift 2 ;;
        --eval_every)      EVAL_EVERY="$2";          shift 2 ;;
        --no_val)          VAL_DATA_DIR=""; VAL_HVM_DIR=""; shift 1 ;;
        --disable_hvm)     DISABLE_HVM=true;         shift 1 ;;
        --shuffle_rag)     SHUFFLE_RAG=true;         shift 1 ;;
        --delta_ln)        DELTA_LN=true;            shift 1 ;;
        --no_delta_ln)     DELTA_LN=false;           shift 1 ;;
        --dra_d_inner)     DRA_D_INNER="$2";         shift 2 ;;
        --dra_n_heads)     DRA_N_HEADS="$2";         shift 2 ;;
        --cdm_num_queries) CDM_NUM_QUERIES="$2";     shift 2 ;;
        --cdm_num_layers)  CDM_NUM_LAYERS="$2";      shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 参数互斥检查
if [ -n "$RESUME_FROM" ] && [ -n "$HVM_CHECKPOINT" ]; then
    echo "Error: --resume and --hvm_ckpt are mutually exclusive."
    exit 1
fi

# ===================== 自动计算 =====================
EFFECTIVE_BS=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

if [ -z "$SWANLAB_RUN_NAME" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SWANLAB_RUN_NAME="hvm-${MODEL_SIZE}-bs${EFFECTIVE_BS}-lr${LEARNING_RATE}-${TIMESTAMP}"
fi

# ===================== 生成 DeepSpeed 配置 =====================
# 动态写入 gradient_accumulation_steps，避免与 YAML 硬编码值不一致
ACCELERATE_CONFIG_RUNTIME="${OUTPUT_DIR}/ds_zero2_hvm_runtime.yaml"
mkdir -p "${OUTPUT_DIR}"
sed "s/gradient_accumulation_steps:.*/gradient_accumulation_steps: ${GRAD_ACCUM}/" \
    ./configs/ds_zero2_hvm.yaml > "${ACCELERATE_CONFIG_RUNTIME}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG_RUNTIME}"

# ===================== 打印配置 =====================

echo "============================================================"
echo "HVM-SVG Training (a100_2)"
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
echo "  PIM layers:        ${PIM_LAYER_INDICES}"
echo "  Delta LayerNorm:   ${DELTA_LN}"
if [ "$MEMORY_MODE" = "gme_dra" ]; then
    echo "  DRA d_inner:       ${DRA_D_INNER}"
    echo "  DRA n_heads:       ${DRA_N_HEADS}"
fi
if [ "$MEMORY_MODE" = "gme_cdm" ]; then
    echo "  CDM queries:       ${CDM_NUM_QUERIES}"
    echo "  CDM layers:        ${CDM_NUM_LAYERS}"
fi
echo "  Mixed precision:   ${MIXED_PRECISION}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "  Run name:          ${SWANLAB_RUN_NAME}"
if [ -n "$VAL_DATA_DIR" ] && [ -n "$VAL_HVM_DIR" ]; then
    echo "  Eval every:        ${EVAL_EVERY} steps"
else
    echo "  Validation:        DISABLED"
fi
if [ "$DISABLE_HVM" = true ]; then
    echo "  *** BASELINE MODE ***"
fi
if [ "$SHUFFLE_RAG" = true ]; then
    echo "  *** SHUFFLE RAG ***"
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

if [ "$MEMORY_MODE" = "gme_dra" ]; then
    TRAIN_ARGS+=(--dra_d_inner "$DRA_D_INNER" --dra_n_heads "$DRA_N_HEADS")
fi

if [ "$MEMORY_MODE" = "gme_cdm" ]; then
    TRAIN_ARGS+=(--cdm_num_queries "$CDM_NUM_QUERIES" --cdm_num_layers "$CDM_NUM_LAYERS")
fi

# ===================== 保存启动快照 =====================
cp "$SCRIPT_PATH" "${OUTPUT_DIR}/run_train_snapshot.sh"

# ===================== 启动训练 =====================
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Starting single-GPU training (${MIXED_PRECISION})..."
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
