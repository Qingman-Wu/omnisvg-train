#!/bin/bash
{
# ============================================================================
# Text2SVG Benchmark Metrics (no GT required)
# ============================================================================
#
# Computes: CLIP-T, Aesthetic, HPS, FID (with external reference images)
# Does NOT compute: SSIM, MSE, CLIP-I, DINO-I (these require GT)
#
# Auto-splits into 3 runs:
#   1) icon        (sample 0 ~ 149)
#   2) illustration (sample 150 ~ 299)
#   3) all          (sample 0 ~ 299)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash metrics/run_mmsvg_benchmark.sh \
#     /mnt/a100_1_data2/wuqingman/OmniSVG/OmniSVG/inference_results/baseline_omnisvg_mmsvgbench \
#     base
#
#   CUDA_VISIBLE_DEVICES=6 bash metrics/run_mmsvg_benchmark.sh /mnt/a100_1_data2/wuqingman/SVGen_test/baseline_unisvg_mmsvgbench_transformers 
#
#   # Custom ref dir
#   CUDA_VISIBLE_DEVICES=6 bash metrics/run_mmsvg_benchmark.sh \
#     inference_results/mmsvgbench_text2svg_hvm \
#     hvm \
#     /mnt/a100_1_data2/wuqingman/SVGen_test/baseline_unisvg_mmsvgbench_transformers
# 
# ============================================================================

set -e

RESULT_DIR="${1:?Usage: bash metrics/run_mmsvg_benchmark.sh <result_dir> <tag> [ref_dir]}"
TAG="${2:-}"
REF_DIR="${3:-/mnt/a100_1_data2/wuqingman/omnisvg-train/inference_results/s1_fixed0.03_step4000_test}"
REF_PATTERN="${REF_PATTERN:-*_gt.png}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="/mnt/a100_1_data2/wuqingman/omnisvg-train/metrics/metrics_results"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_EXP="$(basename $RESULT_DIR)"

# ---------- helper: print summary for a given output subdir ----------
print_summary() {
    local dir="$1"
    echo ""
    echo "=== Summary ($2) ==="
    for f in "$dir"/*.json; do
        if [ -f "$f" ]; then
            METRIC=$(basename "$f" .json)
            echo "--- $METRIC ---"
            python -c "
import json
with open('$f') as fh:
    d = json.load(fh)
for k in ['min_mean','max_mean','avg_mean','trimmed_mean']:
    if k in d:
        print(f'  {k:<25} {d[k]:.4f}')
for k in sorted(d.keys()):
    if k.startswith('fid_c') or k == 'fid_all':
        print(f'  {k:<25} {d[k]:.4f}')
"
        fi
    done
    echo "===================="
}

# ---------- helper: build common args ----------
build_args() {
    local exp_name="$1"
    local extra="$2"
    local a="--result_dir $RESULT_DIR --output_dir $OUTPUT_DIR --device $DEVICE --exp_name $exp_name"
    if [ -n "$TAG" ]; then
        a="$a --tag $TAG"
    fi
    if [ -n "$REF_DIR" ]; then
        a="$a --ref_dir $REF_DIR"
        if [ -n "$REF_PATTERN" ]; then
            a="$a --ref_pattern $REF_PATTERN"
        fi
    fi
    if [ -n "$MAX_SAMPLES" ]; then
        a="$a --max_samples $MAX_SAMPLES"
    fi
    echo "$a $extra"
}

echo "============================================================"
echo "  MMSVGBench Metrics (icon + illustration + all)"
echo "============================================================"
echo "  Result dir:  $RESULT_DIR"
echo "  Tag:         $TAG"
echo "  Ref dir:     ${REF_DIR:-N/A}"
echo "  Ref pattern: $REF_PATTERN"
echo "  Device:      $DEVICE"
echo "  Output:      $OUTPUT_DIR/${BASE_EXP}_*/"
echo "============================================================"

# ==================== 1. Icon (0 ~ 149) ====================
SPLIT="icon"
EXP="${BASE_EXP}_${SPLIT}"
echo ""
echo "############################################################"
echo "  [1/3] $SPLIT  (sample 0 ~ 149)"
echo "############################################################"
ARGS=$(build_args "$EXP" "--start_idx 0 --end_idx 150")
python "$SCRIPT_DIR/compute_text2svg_metrics.py" $ARGS
print_summary "$OUTPUT_DIR/$EXP" "$SPLIT"

# ==================== 2. Illustration (150 ~ 299) ====================
SPLIT="illustration"
EXP="${BASE_EXP}_${SPLIT}"
echo ""
echo "############################################################"
echo "  [2/3] $SPLIT  (sample 150 ~ 299)"
echo "############################################################"
ARGS=$(build_args "$EXP" "--start_idx 150 --end_idx 300")
python "$SCRIPT_DIR/compute_text2svg_metrics.py" $ARGS
print_summary "$OUTPUT_DIR/$EXP" "$SPLIT"

# ==================== 3. All (0 ~ 299) ====================
SPLIT="all"
EXP="${BASE_EXP}_${SPLIT}"
echo ""
echo "############################################################"
echo "  [3/3] $SPLIT  (sample 0 ~ 299)"
echo "############################################################"
ARGS=$(build_args "$EXP" "")
python "$SCRIPT_DIR/compute_text2svg_metrics.py" $ARGS
print_summary "$OUTPUT_DIR/$EXP" "$SPLIT"

echo ""
echo "============================================================"
echo "  All 3 splits done!"
echo "  Results:"
echo "    $OUTPUT_DIR/${BASE_EXP}_icon/"
echo "    $OUTPUT_DIR/${BASE_EXP}_illustration/"
echo "    $OUTPUT_DIR/${BASE_EXP}_all/"
echo "============================================================"

exit
}
