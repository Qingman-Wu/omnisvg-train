#!/bin/bash
# =============================================================================
# HVM-SVG 训练启动脚本 (A100_2 / gme_cdm 消融: CDM only, no EDR)
# =============================================================================
#
# 使用方法:
#   bash run_train_hvm_a100_2.sh --num_gpus 1
#   CUDA_VISIBLE_DEVICES=0,1,2 bash run_train_hvm_a100_2.sh --num_gpus 3
#
# 默认实验:
#   Top3 refs × 4 groups/ref = 12 groups
#   group-wise CDM(part-tag, no-gist) + NO EDR + last4 + adaptive
#   每个 group 经过共享 CDM 后仅输出 1 个 slot:
#       12 groups × 1 slot/group = 12 detail slots
#   group_id 采用全局编号:
#       ref0 -> 0,1,2,3; ref1 -> 4,5,6,7; ref2 -> 8,9,10,11
#   用于“弱化 router / sparse routing”消融

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

# ===================== 环境配置 =====================
ENV_PREFIX="/mnt/data/wuqingman/miniconda3/envs/omnisvg"
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
PYTHON="${ENV_PREFIX}/bin/python"
ACCELERATE_LAUNCH=("$PYTHON" -m accelerate.commands.launch)
TRAIN_SCRIPT="${SCRIPT_DIR}/train_hvm.py"
ACCELERATE_CONFIG_TEMPLATE="${SCRIPT_DIR}/configs/ds_zero2_hvm.yaml"

if [ ! -x "$PYTHON" ]; then
    echo "Error: Python not found or not executable: $PYTHON"
    exit 1
fi

NVCC_PATH="$(command -v nvcc || true)"
if [ -z "$NVCC_PATH" ]; then
    echo "Error: nvcc not found in PATH."
    exit 1
fi

NVCC_REAL_PATH="$(readlink -f "$NVCC_PATH")"
if [ "$NVCC_REAL_PATH" = "$NVCC_PATH" ] && [ -f "$NVCC_PATH" ]; then
    NVCC_WRAPPED_PATH="$(sed -n 's|^exec \(.*\/bin/nvcc\) .*|\1|p' "$NVCC_PATH")"
    if [ -n "$NVCC_WRAPPED_PATH" ]; then
        NVCC_REAL_PATH="$NVCC_WRAPPED_PATH"
    fi
fi

if [ ! -x "$NVCC_REAL_PATH" ]; then
    echo "Error: resolved nvcc is not executable: $NVCC_REAL_PATH"
    exit 1
fi

export CUDA_HOME="$(dirname "$(dirname "$NVCC_REAL_PATH")")"
export PATH="${CUDA_HOME}/bin:${PATH}"

# ===================== 训练参数 =====================

# -- GPU --
NUM_GPUS=4

# -- 数据 --
DATA_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_retrieval_corpus"
HVM_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_1w_nozoom_top3part"

# -- 模型 --
MODEL_SIZE="8B"
OMNISVG_CHECKPOINT="/mnt/a100_1_data2/wuqingman/models/OmniSVG/OmniSVG1.1_8B"
BASE_MODEL="/mnt/a100_1_data/wuqingman/models/Qwen/Qwen2.5-VL-7B-Instruct"


# -- HVM 架构 --
D_QFORMER=1024
D_PIM_INNER=512
PIM_LAYER_INTERVAL=4
GATE_ALPHA_INIT=0.05
GME_NUM_QUERIES=32
PART_NUM_REFS=3
PME_MAX_GROUPS=12
CDM_NUM_QUERIES=12
CDM_NUM_LAYERS=6
CDM_LAYOUT="groupwise"
CDM_GROUP_QUERIES_PER_GROUP=1
CDM_DETAIL_SOURCE="part"

CDM_DISABLE_GIST=true
CDM_DISABLE_TAG_META=false
CDM_DISABLE_GROUP_ID=false
EDR_D_ROUTER=256
EDR_TOP_K=1
EDR_DISABLE_CONF=false
EDR_RANDOM_REPLACE_TOP1=false
MEMORY_MODE="gme_cdm"
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
OUTPUT_DIR="/mnt/a100_1_data3/wuqingman/omnisvg-train/outputs_s8_gmecdm_noedr_top3part_12slot_nogist_parttag_nozoom_last4"
LOG_EVERY=10
SAVE_EVERY=2000
SWANLAB_MODE="cloud"
SWANLAB_RUN_NAME="s8_gmecdm_noedr_top3part_12slot_nogist_parttag_nozoom_last4"

# -- 恢复训练 --
RESUME_FROM=""
HVM_CHECKPOINT=""

# -- 验证集 --
VAL_DATA_DIR="/mnt/a100_1_data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_val"
VAL_HVM_DIR="/mnt/a100_1_data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_val_nozoom_top3part"
EVAL_EVERY=500

# -- Ablation --
DISABLE_HVM=false
SHUFFLE_RAG=false
SHUFFLE_GME=false
SHUFFLE_CDM=false
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
        --part_num_refs)  PART_NUM_REFS="$2";        shift 2 ;;
        --pme_max_groups) PME_MAX_GROUPS="$2";       shift 2 ;;
        --cdm_num_queries) CDM_NUM_QUERIES="$2";     shift 2 ;;
        --cdm_num_layers) CDM_NUM_LAYERS="$2";       shift 2 ;;
        --cdm_layout)     CDM_LAYOUT="$2";           shift 2 ;;
        --cdm_group_queries_per_group) CDM_GROUP_QUERIES_PER_GROUP="$2"; shift 2 ;;
        --cdm_detail_source) CDM_DETAIL_SOURCE="$2"; shift 2 ;;
        --cdm_disable_gist) CDM_DISABLE_GIST=true;   shift 1 ;;
        --no_cdm_disable_gist) CDM_DISABLE_GIST=false; shift 1 ;;
        --cdm_disable_tag_meta) CDM_DISABLE_TAG_META=true; shift 1 ;;
        --no_cdm_disable_tag_meta) CDM_DISABLE_TAG_META=false; shift 1 ;;
        --cdm_disable_group_id) CDM_DISABLE_GROUP_ID=true; shift 1 ;;
        --no_cdm_disable_group_id) CDM_DISABLE_GROUP_ID=false; shift 1 ;;
        --edr_d_router)   EDR_D_ROUTER="$2";         shift 2 ;;
        --edr_top_k)      EDR_TOP_K="$2";            shift 2 ;;
        --edr_disable_conf) EDR_DISABLE_CONF=true;   shift 1 ;;
        --edr_random_replace_top1) EDR_RANDOM_REPLACE_TOP1=true; shift 1 ;;
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
        --shuffle_gme)    SHUFFLE_GME=true;          shift 1 ;;
        --shuffle_cdm)    SHUFFLE_CDM=true;          shift 1 ;;
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
    "${ACCELERATE_CONFIG_TEMPLATE}" > "${ACCELERATE_CONFIG_RUNTIME}"
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
echo "  Part refs used:    ${PART_NUM_REFS}"
echo "  PME max groups:    ${PME_MAX_GROUPS}"
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
if [ "$SHUFFLE_GME" = true ]; then
    echo "  *** SHUFFLE GME ABLATION: GME gets random refs, CDM correct ***"
fi
if [ "$SHUFFLE_CDM" = true ]; then
    echo "  *** SHUFFLE CDM ABLATION: CDM gets random parts, GME correct ***"
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
    echo "  CDM layout:        ${CDM_LAYOUT}"
    echo "  CDM slots/group:   ${CDM_GROUP_QUERIES_PER_GROUP}"
    echo "  CDM detail source: ${CDM_DETAIL_SOURCE}"
    echo "  CDM disable gist:  ${CDM_DISABLE_GIST}"
    echo "  CDM disable tag:   ${CDM_DISABLE_TAG_META}"
    echo "  CDM disable gid:   ${CDM_DISABLE_GROUP_ID}"
fi
if [ "$MEMORY_MODE" = "gme_cdm_edr" ]; then
    echo "  EDR d_router:      ${EDR_D_ROUTER}"
    echo "  EDR top-k:         ${EDR_TOP_K}"
    echo "  EDR disable conf:  ${EDR_DISABLE_CONF}"
    echo "  EDR random top1:   ${EDR_RANDOM_REPLACE_TOP1}"
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
    --base_model "$BASE_MODEL"
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
    --part_num_refs "$PART_NUM_REFS"
    --pme_max_groups "$PME_MAX_GROUPS"
    --cdm_num_queries "$CDM_NUM_QUERIES"
    --cdm_num_layers "$CDM_NUM_LAYERS"
    --cdm_layout "$CDM_LAYOUT"
    --cdm_group_queries_per_group "$CDM_GROUP_QUERIES_PER_GROUP"
    --cdm_detail_source "$CDM_DETAIL_SOURCE"
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

if [ "$SHUFFLE_GME" = true ]; then
    TRAIN_ARGS+=(--shuffle_gme)
fi

if [ "$SHUFFLE_CDM" = true ]; then
    TRAIN_ARGS+=(--shuffle_cdm)
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

if [ "$EDR_RANDOM_REPLACE_TOP1" = true ]; then
    TRAIN_ARGS+=(--edr_random_replace_top1)
fi

if [ "$CDM_DISABLE_GIST" = true ]; then
    TRAIN_ARGS+=(--cdm_disable_gist)
fi

if [ "$CDM_DISABLE_TAG_META" = true ]; then
    TRAIN_ARGS+=(--cdm_disable_tag_meta)
fi

if [ "$CDM_DISABLE_GROUP_ID" = true ]; then
    TRAIN_ARGS+=(--cdm_disable_group_id)
fi

# ===================== 保存启动快照 =====================
cp "$SCRIPT_PATH" "${OUTPUT_DIR}/run_train_hvm.snapshot.sh"

# ===================== 参数验证 =====================
echo ""
echo "[VERIFY] Key args passed to train_hvm.py:"
for i in "${!TRAIN_ARGS[@]}"; do
    if [ "${TRAIN_ARGS[$i]}" = "--memory_mode" ] || [ "${TRAIN_ARGS[$i]}" = "--output_dir" ]; then
        echo "  ${TRAIN_ARGS[$i]} = ${TRAIN_ARGS[$((i+1))]}"
    fi
done
echo ""
# ===================== 启动训练 =====================
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Starting single-GPU training (mixed precision: ${MIXED_PRECISION})..."
    "${ACCELERATE_LAUNCH[@]}" \
        --num_processes 1 \
        --mixed_precision "$MIXED_PRECISION" \
        "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
else
    echo "Starting ${NUM_GPUS}-GPU training (DeepSpeed ZeRO-2, ${MIXED_PRECISION})..."
    "${ACCELERATE_LAUNCH[@]}" \
        --config_file "$ACCELERATE_CONFIG" \
        --num_processes "$NUM_GPUS" \
        "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
fi
