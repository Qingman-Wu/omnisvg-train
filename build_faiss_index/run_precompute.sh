#!/bin/bash
#
# HVM-SVG 数据预计算启动脚本
# ================================
# Stage 1 (metadata) + Stage 2 (rag) + Stage 4 (groups): 单进程，CPU
# Stage 3 (features): 8 GPU 并行
#
# Usage:
#   bash run_precompute.sh          # 运行全部阶段
#   bash run_precompute.sh features # 只运行 features 阶段（8 GPU 并行）
#   bash run_precompute.sh rag      # 只运行 rag 阶段
#

PYTHON="/mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/python"
SCRIPT="/mnt/data/wuqingman/omnisvg-train/precompute_hvm_data.py"
export CUDA_HOME="/mnt/data/wuqingman/miniconda3/envs/omnisvg"

NUM_GPUS=8
BATCH_SIZE=16

STAGE=${1:-all}

run_metadata() {
    echo "=========================================="
    echo "Running Stage 1: Metadata"
    echo "=========================================="
    $PYTHON $SCRIPT --stage metadata
}

run_rag() {
    echo "=========================================="
    echo "Running Stage 2: RAG Retrieval"
    echo "=========================================="
    $PYTHON $SCRIPT --stage rag
}

run_features() {
    echo "=========================================="
    echo "Running Stage 3: Feature Extraction (${NUM_GPUS} GPUs)"
    echo "=========================================="

    # 启动 N 个并行进程，每个用不同 GPU
    pids=()
    for shard_id in $(seq 0 $((NUM_GPUS - 1))); do
        echo "Starting shard ${shard_id} on GPU ${shard_id}..."
        CUDA_VISIBLE_DEVICES=$shard_id $PYTHON $SCRIPT \
            --stage features \
            --gpu_id 0 \
            --batch_size $BATCH_SIZE \
            --num_shards $NUM_GPUS \
            --shard_id $shard_id \
            > /tmp/hvm_features_shard_${shard_id}.log 2>&1 &
        pids+=($!)
    done

    echo "All ${NUM_GPUS} shards started. PIDs: ${pids[*]}"
    echo "Logs: /tmp/hvm_features_shard_*.log"
    echo ""
    echo "Waiting for all shards to complete..."

    # 等待所有进程并检查返回码
    all_ok=true
    for i in "${!pids[@]}"; do
        wait ${pids[$i]}
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "ERROR: Shard $i (PID ${pids[$i]}) failed with exit code $exit_code"
            echo "Check log: /tmp/hvm_features_shard_${i}.log"
            all_ok=false
        else
            echo "Shard $i (PID ${pids[$i]}) completed successfully"
        fi
    done

    if $all_ok; then
        echo "All feature extraction shards completed successfully!"
    else
        echo "Some shards failed. Check logs above."
        return 1
    fi
}

run_groups() {
    echo "=========================================="
    echo "Running Stage 4: SVG Path Grouping"
    echo "=========================================="
    $PYTHON $SCRIPT --stage groups
}

# 根据参数决定运行哪些阶段
case $STAGE in
    all)
        run_metadata
        run_rag
        run_features
        run_groups
        ;;
    metadata)
        run_metadata
        ;;
    rag)
        run_rag
        ;;
    features)
        run_features
        ;;
    groups)
        run_groups
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: bash run_precompute.sh [all|metadata|rag|features|groups]"
        exit 1
        ;;
esac

echo ""
echo "Done!"
