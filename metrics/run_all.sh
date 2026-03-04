#!/bin/bash
# ============================================================================
# SVG Generation Metrics - 统一运行脚本
# ============================================================================
#
# 用法:
#   # tag=base
#   CUDA_VISIBLE_DEVICES=4 RESUME=1 bash metrics/run_all.sh inference_results/omnisvg_baseline base
#   # tag=hvm

#   CUDA_VISIBLE_DEVICES=4 RESUME=1 bash metrics/run_all.sh inference_results/0302_1416_s1_fixed0.03_last4_step4000 hvm
#
#   # 指定 GPU
#   CUDA_VISIBLE_DEVICES=1 bash metrics/run_all.sh inference_results/exp1
#
#   # 调试: 只跑前 20 个样本
#   MAX_SAMPLES=20 bash metrics/run_all.sh inference_results/exp1
#
# 输出保存到: metrics/metrics_results/<exp_name>/
#   ├── ssim.csv + ssim.json
#   ├── mse.csv + mse.json
#   ├── clip_i.csv + clip_i.json
#   ├── clip_t.csv + clip_t.json
#   ├── dino_i.csv + dino_i.json
#   ├── aesthetic.csv + aesthetic.json
#   └── hps.csv + hps.json
# ============================================================================

set -e

RESULT_DIR="${1:?用法: bash metrics/run_all.sh <result_dir> [tag]}"
TAG="${2:-hvm}"
DEVICE="${DEVICE:-cuda:0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RESUME="${RESUME:-}"
OUTPUT_DIR="/mnt/a100_1_data2/wuqingman/omnisvg-train/metrics/metrics_results"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 构建公共参数
COMMON_ARGS="--result_dir $RESULT_DIR --tag $TAG --output_dir $OUTPUT_DIR --device $DEVICE"
if [ -n "$MAX_SAMPLES" ]; then
    COMMON_ARGS="$COMMON_ARGS --max_samples $MAX_SAMPLES"
fi
if [ -n "$RESUME" ]; then
    COMMON_ARGS="$COMMON_ARGS --resume"
fi

EXP_NAME="$(basename $RESULT_DIR)"
echo "============================================================"
echo "  SVG Metrics Pipeline"
echo "============================================================"
echo "  Result dir:  $RESULT_DIR"
echo "  Tag:         $TAG"
echo "  Device:      $DEVICE"
echo "  Exp name:    $EXP_NAME"
echo "  Resume:      ${RESUME:-no}"
echo "  Output:      $OUTPUT_DIR/$EXP_NAME/"
echo "============================================================"

echo ""
echo "[1/7] Computing SSIM..."
python "$SCRIPT_DIR/compute_ssim.py" $COMMON_ARGS

echo ""
echo "[2/7] Computing MSE..."
python "$SCRIPT_DIR/compute_mse.py" $COMMON_ARGS

echo ""
echo "[3/7] Computing CLIP-I..."
python "$SCRIPT_DIR/compute_clip_i.py" $COMMON_ARGS

echo ""
echo "[4/7] Computing CLIP-T..."
python "$SCRIPT_DIR/compute_clip_t.py" $COMMON_ARGS

echo ""
echo "[5/7] Computing DINO-I..."
python "$SCRIPT_DIR/compute_dino_i.py" $COMMON_ARGS

echo ""
echo "[6/7] Computing Aesthetic..."
python "$SCRIPT_DIR/compute_aesthetic.py" $COMMON_ARGS

echo ""
echo "[7/7] Computing HPS..."
python "$SCRIPT_DIR/compute_hps.py" $COMMON_ARGS

echo ""
echo "============================================================"
echo "  All metrics done!"
echo "  Results: $OUTPUT_DIR/$EXP_NAME/"
echo "============================================================"

# 汇总打印所有 JSON
echo ""
echo "=== Summary ==="
for f in "$OUTPUT_DIR/$EXP_NAME"/*.json; do
    if [ -f "$f" ]; then
        METRIC=$(basename "$f" .json)
        echo "--- $METRIC ---"
        python -c "
import json, sys
with open('$f') as fh:
    d = json.load(fh)
for k in ['min_mean','max_mean','avg_mean','trimmed_mean']:
    if k in d:
        print(f'  {k:<16} {d[k]:.4f}')
"
    fi
done
echo "==============="
