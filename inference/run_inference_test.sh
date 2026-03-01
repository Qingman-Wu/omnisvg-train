#!/bin/bash
# =============================================================================
# HVM-SVG 测试集推理启动脚本
# =============================================================================
#
# 使用方法:
#   # 测试所有1000个样本 (使用5张GPU)
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash run_inference_test.sh --step 8000 --num_gpus 5
#
#   # 测试前100个样本 (快速验证)
#   CUDA_VISIBLE_DEVICES=0,1 bash run_inference_test.sh --step 8000 --num_samples 100
#
#   # 测试不同checkpoint
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash run_inference_test.sh --step 6000
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash run_inference_test.sh --step 4000
#
#   # 自定义输出目录
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash run_inference_test.sh --step 8000 --output_dir ./my_results

set -e

# ===================== 环境配置 =====================
export TOKENIZERS_PARALLELISM=false
PYTHON="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/python"

# ===================== 默认参数 =====================

# 训练checkpoint
CHECKPOINT_STEP=8000
CHECKPOINT_DIR="/mnt/data2/wuqingman/omnisvg-train/outputs_s1_fixed0.03"

# 测试数据集 (holdout)
DATA_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test_holdout"
HVM_DIR="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_test"

# GPU
NUM_GPUS=""  # 留空则使用所有可见GPU

# 推理参数
NUM_SAMPLES=1000  # 测试样本数量 (0-999)
NUM_CANDIDATES=5  # 每个样本生成几个候选
TEMPERATURE=0.5
TOP_P=0.90
TOP_K=50

# 输出
OUTPUT_DIR=""  # 留空则自动生成
SAVE_PNG=true
SAVE_GT=true
SAVE_REFS=true
RESUME=true

# ===================== 解析命令行参数 =====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --num_candidates)
            NUM_CANDIDATES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --hvm_dir)
            HVM_DIR="$2"
            shift 2
            ;;
        --no_png)
            SAVE_PNG=false
            shift 1
            ;;
        --no_gt)
            SAVE_GT=false
            shift 1
            ;;
        --no_refs)
            SAVE_REFS=false
            shift 1
            ;;
        --no_resume)
            RESUME=false
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_inference_test.sh [options]"
            echo "Options:"
            echo "  --step STEP              Checkpoint step (default: 8000)"
            echo "  --checkpoint_dir DIR     Checkpoint directory (default: outputs_s1_fixed0.03)"
            echo "  --num_gpus N             Number of GPUs to use (default: all visible)"
            echo "  --num_samples N          Number of samples to test (default: 1000)"
            echo "  --num_candidates N       Number of candidates per sample (default: 5)"
            echo "  --output_dir DIR         Output directory (default: auto-generated)"
            echo "  --no_png                 Don't save PNG renders"
            echo "  --no_gt                  Don't save ground truth"
            echo "  --no_refs                Don't save references"
            echo "  --no_resume              Don't skip existing results"
            exit 1
            ;;
    esac
done

# ===================== 构建路径 =====================
HVM_CHECKPOINT="${CHECKPOINT_DIR}/hvm_step_${CHECKPOINT_STEP}.pt"

# 检查checkpoint是否存在
if [ ! -f "$HVM_CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $HVM_CHECKPOINT"
    echo "Available checkpoints in ${CHECKPOINT_DIR}:"
    ls -lh "${CHECKPOINT_DIR}"/hvm_step_*.pt 2>/dev/null || echo "  (none found)"
    exit 1
fi

# 自动生成输出目录
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/mnt/data2/wuqingman/omnisvg-train/inference_results/s1_fixed0.03_step${CHECKPOINT_STEP}_hvm_n${NUM_SAMPLES}"
fi

# 生成样本索引序列
END_IDX=$((NUM_SAMPLES - 1))
SAMPLE_INDICES=$(seq 0 $END_IDX)

# ===================== 打印配置 =====================
echo "============================================================"
echo "HVM-SVG Test Holdout Inference"
echo "============================================================"
echo "  Checkpoint       : ${HVM_CHECKPOINT}"
echo "  Data dir         : ${DATA_DIR}"
echo "  HVM dir          : ${HVM_DIR}"
echo "  Num samples      : ${NUM_SAMPLES} (indices 0-${END_IDX})"
echo "  Num candidates   : ${NUM_CANDIDATES}"
echo "  Output dir       : ${OUTPUT_DIR}"
if [ -n "$NUM_GPUS" ]; then
    echo "  Num GPUs         : ${NUM_GPUS}"
else
    echo "  Num GPUs         : all visible"
fi
echo "  Mode             : HVM"
echo "  Save PNG         : ${SAVE_PNG}"
echo "  Save GT          : ${SAVE_GT}"
echo "  Save refs        : ${SAVE_REFS}"
echo "  Resume           : ${RESUME}"
echo "============================================================"

# 确认继续
read -p "Press Enter to start inference, or Ctrl+C to cancel..."

# ===================== 构建推理命令 =====================
INFERENCE_ARGS=(
    --hvm_checkpoint "$HVM_CHECKPOINT"
    --data_dir "$DATA_DIR"
    --hvm_dir "$HVM_DIR"
    --sample_indices $SAMPLE_INDICES
    --output_dir "$OUTPUT_DIR"
    --num_candidates "$NUM_CANDIDATES"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_k "$TOP_K"
)

if [ -n "$NUM_GPUS" ]; then
    INFERENCE_ARGS+=(--num_gpus "$NUM_GPUS")
fi

if [ "$SAVE_PNG" = true ]; then
    INFERENCE_ARGS+=(--save_png)
fi

if [ "$SAVE_GT" = true ]; then
    INFERENCE_ARGS+=(--save_gt)
fi

if [ "$SAVE_REFS" = true ]; then
    INFERENCE_ARGS+=(--save_refs)
fi

if [ "$RESUME" = true ]; then
    INFERENCE_ARGS+=(--resume)
fi

# ===================== 启动推理 =====================
echo ""
echo "Starting inference..."
echo "Command: $PYTHON inference/inference_hvm_s1_test.py ${INFERENCE_ARGS[@]}"
echo ""

$PYTHON inference/inference_hvm_s1_test.py "${INFERENCE_ARGS[@]}"

# ===================== 完成 =====================
echo ""
echo "============================================================"
echo "Inference completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
echo ""
echo "Summary:"
echo "  Total samples    : ${NUM_SAMPLES}"
echo "  Candidates/sample: ${NUM_CANDIDATES}"
echo "  Generated files  : HVM SVGs"
if [ "$SAVE_PNG" = true ]; then
    echo "  PNG renders      : Yes"
fi
if [ "$SAVE_GT" = true ]; then
    echo "  Ground truth     : Yes"
fi
if [ "$SAVE_REFS" = true ]; then
    echo "  References       : Yes"
fi
echo ""
echo "Next steps:"
echo "  1. 查看生成结果: ls -lh ${OUTPUT_DIR}/"
echo "  2. 运行评估指标: python evaluation/evaluate_results.py --results_dir ${OUTPUT_DIR}"
echo ""
