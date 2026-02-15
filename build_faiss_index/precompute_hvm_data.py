#!/usr/bin/env python
"""
HVM-SVG Data Preprocessing Script
==================================
为 HVM-SVG 训练预计算所需数据：
  Stage 1 (metadata): 加载所有 parquet，提取元数据，分配全局整数索引
  Stage 2 (rag):      用 CLIP text encoder + FAISS 做 Top-3 语义文本检索
  Stage 3 (features): 提取 Qwen2.5-VL pre-merge vision features [16,16,4,1280]
  Stage 4 (groups):   解析 SVG，计算 path 复杂度和分组信息

Usage:
    # 运行全部阶段
    python precompute_hvm_data.py --stage all

    # 只运行某个阶段
    python precompute_hvm_data.py --stage metadata
    python precompute_hvm_data.py --stage rag
    python precompute_hvm_data.py --stage features --gpu_id 0
    python precompute_hvm_data.py --stage groups

    # 多GPU并行提取特征（在不同终端分别运行）
    python precompute_hvm_data.py --stage features --gpu_id 0 --num_shards 8 --shard_id 0
    python precompute_hvm_data.py --stage features --gpu_id 1 --num_shards 8 --shard_id 1
    ...
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import io
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_DATA_DIR = "/mnt/data/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test2"
DEFAULT_MODEL_PATH = "/mnt/data/wuqingman/models/Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_CLIP_MODEL_PATH = "/mnt/data/wuqingman/models/openai/clip-vit-large-patch14"
DEFAULT_OUTPUT_DIR = "/mnt/data/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed"

VIEWBOX_SIZE = 200      # SVG viewBox 尺寸
IMAGE_SIZE = 448        # 图像尺寸
VISION_HIDDEN_DIM = 1280  # Vision encoder hidden dim (pre-merge)
LLM_HIDDEN_DIM = 3584    # LLM hidden dim (post-merge)
SPATIAL_MERGE_SIZE = 2
FEATURE_GRID_H = 16      # 448 / 14 / 2 = 16 (post-merge grid)
FEATURE_GRID_W = 16
PATCH_GRID_H = 32        # 448 / 14 = 32 (pre-merge grid)
PATCH_GRID_W = 32
PATCHES_PER_UNIT = SPATIAL_MERGE_SIZE ** 2  # 4

RAG_TOP_K = 3             # 检索 Top-K 参考
MAX_GROUPS = 4             # 最多分组数
QUERIES_PER_GROUP = 4      # 每组 query 数（PME）


# ============================================================================
# Stage 1: Load Metadata
# ============================================================================

def stage_metadata(data_dir: str, output_dir: str):
    """
    加载所有 parquet 文件，提取元数据并分配全局整数索引。
    输出: metadata.jsonl, id_to_idx.json
    """
    import pyarrow.parquet as pq

    print("=" * 60)
    print("Stage 1: Extracting Metadata")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    id_map_path = os.path.join(output_dir, "id_to_idx.json")

    # 找到所有 parquet 文件
    parquet_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith(".parquet")
    ])
    print(f"Found {len(parquet_files)} parquet files")

    id_to_idx = {}
    global_idx = 0

    with open(metadata_path, "w", encoding="utf-8") as fout:
        for pf in tqdm(parquet_files, desc="Loading parquets"):
            table = pq.read_table(pf)
            n = table.num_rows

            for i in range(n):
                sample_id = table.column("id")[i].as_py()
                description = table.column("description")[i].as_py()
                keywords = table.column("keywords")[i].as_py()
                detail = table.column("detail")[i].as_py()
                token_len = table.column("token_len")[i].as_py()

                record = {
                    "idx": global_idx,
                    "id": sample_id,
                    "description": description,
                    "keywords": keywords or "",
                    "detail": detail or "",
                    "token_len": token_len,
                    "parquet_file": os.path.basename(pf),
                    "parquet_row": i,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                id_to_idx[sample_id] = global_idx
                global_idx += 1

    # 保存 id -> idx 映射
    with open(id_map_path, "w") as f:
        json.dump(id_to_idx, f)

    print(f"Total samples: {global_idx}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved id mapping to: {id_map_path}")
    return global_idx


# ============================================================================
# Stage 2: RAG Retrieval
# ============================================================================

def stage_rag(output_dir: str, clip_model_path: str):
    """
    用 CLIP text encoder 编码 descriptions，构建 FAISS 索引，检索 Top-3 相似样本。
    CLIP 经过文本-图像对比学习，语义检索质量远优于 Qwen 的 raw embed_tokens。
    输出: rag_results.jsonl, text_embeddings.npy, faiss_index.bin
    """
    import faiss
    from transformers import CLIPModel, CLIPTokenizer

    print("=" * 60)
    print("Stage 2: RAG Retrieval (CLIP)")
    print("=" * 60)

    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    rag_path = os.path.join(output_dir, "rag_results.jsonl")
    embeddings_path = os.path.join(output_dir, "text_embeddings.npy")
    index_path = os.path.join(output_dir, "faiss_index.bin")

    # 1. 加载 metadata
    print("Loading metadata...")
    records = []
    with open(metadata_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    N = len(records)
    print(f"Total samples: {N}")

    # 拼接 description + keywords 提高检索质量
    texts = []
    for r in records:
        desc = r["description"]
        kw = r.get("keywords", "")
        texts.append(f"{desc}. {kw}" if kw else desc)

    # 2. 加载 CLIP text encoder
    print(f"Loading CLIP text encoder from {clip_model_path}...")
    clip_model = CLIPModel.from_pretrained(clip_model_path)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = clip_model.to(device).eval()
    embed_dim = clip_model.config.projection_dim  # 768
    print(f"CLIP text embedding dim: {embed_dim}")

    # 3. 编码所有 descriptions
    print("Encoding descriptions with CLIP...")
    BATCH_SIZE = 256
    all_embeddings = np.zeros((N, embed_dim), dtype=np.float32)

    for start in tqdm(range(0, N, BATCH_SIZE), desc="Encoding"):
        end = min(start + BATCH_SIZE, N)
        batch_texts = texts[start:end]

        inputs = clip_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
            # L2 normalize (cosine similarity = inner product after normalization)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        all_embeddings[start:end] = text_features.cpu().numpy()

    # 释放 CLIP 模型
    del clip_model
    torch.cuda.empty_cache()

    # 保存 embeddings
    np.save(embeddings_path, all_embeddings)
    print(f"Saved embeddings to: {embeddings_path} (shape: {all_embeddings.shape})")

    # 4. 构建 FAISS 索引
    print("Building FAISS index (IndexFlatIP)...")
    index = faiss.IndexFlatIP(embed_dim)
    index.add(all_embeddings)
    print(f"FAISS index size: {index.ntotal}")

    # 5. 检索 Top-K+1（包含自身，后续排除）
    print("Searching Top-K nearest neighbors...")
    K = RAG_TOP_K + 1  # 多检索 1 个，排除自身

    # 分批检索（避免 OOM）
    SEARCH_BATCH = 10000
    all_indices = np.zeros((N, K), dtype=np.int64)
    all_scores = np.zeros((N, K), dtype=np.float32)

    for start in tqdm(range(0, N, SEARCH_BATCH), desc="FAISS search"):
        end = min(start + SEARCH_BATCH, N)
        query = all_embeddings[start:end]
        scores, indices = index.search(query, K)
        all_indices[start:end] = indices
        all_scores[start:end] = scores

    # 6. 生成 RAG 结果（排除自身）
    print("Generating RAG results...")
    with open(rag_path, "w") as fout:
        for i in tqdm(range(N), desc="Writing RAG results"):
            ref_indices = []
            ref_scores = []
            for j in range(K):
                candidate_idx = int(all_indices[i, j])
                if candidate_idx != i:  # 排除自身
                    ref_indices.append(candidate_idx)
                    ref_scores.append(float(all_scores[i, j]))
                if len(ref_indices) == RAG_TOP_K:
                    break

            # 补齐（极端情况：所有 K 个结果都是自身）
            while len(ref_indices) < RAG_TOP_K:
                for j in range(K):
                    candidate = int(all_indices[i, j])
                    if candidate not in ref_indices and candidate != i:
                        ref_indices.append(candidate)
                        ref_scores.append(float(all_scores[i, j]))
                        break
                else:
                    rand_idx = np.random.randint(0, N)
                    while rand_idx == i or rand_idx in ref_indices:
                        rand_idx = np.random.randint(0, N)
                    ref_indices.append(rand_idx)
                    ref_scores.append(0.0)

            record = {
                "idx": i,
                "ref_indices": ref_indices,
                "ref_scores": ref_scores,
            }
            fout.write(json.dumps(record) + "\n")

    # 保存 FAISS 索引
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to: {index_path}")
    print(f"Saved RAG results to: {rag_path}")


# ============================================================================
# Stage 3: Extract Vision Features
# ============================================================================

def stage_features(
    data_dir: str,
    output_dir: str,
    model_path: str,
    gpu_id: int = 0,
    batch_size: int = 16,
    num_shards: int = 1,
    shard_id: int = 0,
):
    """
    提取 Qwen2.5-VL pre-merge vision features。
    每张图 → [16, 16, 4, 1280] (16x16 merger units, 每个 unit 4 个 patch, 1280 维)
    输出: features/{idx//1000}/{idx}.pt
    """
    import pyarrow.parquet as pq

    print("=" * 60)
    print(f"Stage 3: Extract Vision Features (shard {shard_id}/{num_shards}, GPU {gpu_id})")
    print("=" * 60)

    device = torch.device(f"cuda:{gpu_id}")
    features_dir = os.path.join(output_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    # 1. 加载 metadata 确定总数和分片范围
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    records = []
    with open(metadata_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    N = len(records)

    # 计算分片范围
    shard_size = math.ceil(N / num_shards)
    start_idx = shard_id * shard_size
    end_idx = min(start_idx + shard_size, N)
    print(f"Processing indices [{start_idx}, {end_idx}) ({end_idx - start_idx} samples)")

    # 2. 加载 vision encoder
    print("Loading vision encoder...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    visual = model.visual.to(device).eval()

    # 释放 LLM 部分
    del model.model
    del model.lm_head
    del model
    torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(model_path)
    print(f"Vision encoder loaded on GPU {gpu_id}, memory: {torch.cuda.memory_allocated(device) / 1e6:.0f} MB")

    # 3. 建立 parquet 文件的随机访问索引
    #    预加载所有 parquet 的行 → (parquet_file, row_in_file) 映射
    print("Building parquet index...")
    parquet_tables = {}
    idx_to_location = {}

    for rec in records[start_idx:end_idx]:
        pf = rec["parquet_file"]
        if pf not in parquet_tables:
            full_path = os.path.join(data_dir, pf)
            parquet_tables[pf] = pq.read_table(full_path)
        idx_to_location[rec["idx"]] = (pf, rec["parquet_row"])

    print(f"Loaded {len(parquet_tables)} parquet files into memory")

    # 4. 提取特征
    def get_image(idx: int) -> Image.Image:
        pf, row = idx_to_location[idx]
        table = parquet_tables[pf]
        img_data = table.column("image")[row].as_py()
        img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        return img

    def extract_pre_merge_batch(images: List[Image.Image]) -> List[torch.Tensor]:
        """
        批量提取 pre-merge features。
        返回: List of [16, 16, 4, 1280] tensors (float16, CPU)
        """
        from qwen_vl_utils import process_vision_info

        # 构造 processor 输入
        all_pixel_values = []
        all_grid_thws = []

        for img in images:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "x"},
            ]}]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text_input], images=image_inputs,
                return_tensors="pt"
            )
            all_pixel_values.append(inputs["pixel_values"])
            all_grid_thws.append(inputs["image_grid_thw"])

        # 每张图的 pixel_values shape: [1024, 1176]
        # 每张图的 grid_thw: [1, 3] = [[1, 32, 32]]
        # 为了批处理，需要 concat 并正确处理 grid_thw

        pixel_values = torch.cat(all_pixel_values, dim=0).to(device, dtype=torch.float16)
        grid_thw = torch.cat(all_grid_thws, dim=0).to(device)
        # pixel_values: [B*1024, 1176], grid_thw: [B, 3]

        # Hook 捕获 pre-merge features
        pre_merge_data = {}

        def pre_hook(module, args):
            pre_merge_data["x"] = args[0].detach().clone()

        handle = visual.merger.register_forward_pre_hook(pre_hook)

        with torch.no_grad():
            visual(pixel_values, grid_thw=grid_thw)

        handle.remove()

        pre_merge = pre_merge_data["x"]  # [B*1024, 1280]

        # 获取 window_index 和 reverse_indices
        window_index, _ = visual.get_window_index(grid_thw)
        reverse_indices = torch.argsort(window_index)

        # 按图像拆分
        B = len(images)
        tokens_per_image = PATCH_GRID_H * PATCH_GRID_W  # 1024
        units_per_image = FEATURE_GRID_H * FEATURE_GRID_W  # 256

        results = []
        for b in range(B):
            # 提取当前图像的 pre-merge tokens
            img_pre = pre_merge[b * tokens_per_image: (b + 1) * tokens_per_image]  # [1024, 1280]

            # 获取当前图像的 reverse_indices
            img_rev = reverse_indices[b * units_per_image: (b + 1) * units_per_image]  # [256]
            img_rev = img_rev - b * units_per_image  # 调整为相对索引

            # Reshape 为 merger units
            smu = PATCHES_PER_UNIT  # 4
            img_units = img_pre.reshape(-1, smu, VISION_HIDDEN_DIM)  # [256, 4, 1280]

            # 恢复空间顺序
            img_units_spatial = img_units[img_rev, :, :]  # [256, 4, 1280]

            # Reshape 为 [16, 16, 4, 1280]
            feature_grid = img_units_spatial.reshape(
                FEATURE_GRID_H, FEATURE_GRID_W, PATCHES_PER_UNIT, VISION_HIDDEN_DIM
            )  # [16, 16, 4, 1280]

            results.append(feature_grid.cpu().half())

        return results

    # 主循环：批量处理
    processed = 0
    skipped = 0
    errors = 0
    indices_to_process = list(range(start_idx, end_idx))

    pbar = tqdm(total=len(indices_to_process), desc="Extracting features")

    for batch_start in range(0, len(indices_to_process), batch_size):
        batch_indices = indices_to_process[batch_start: batch_start + batch_size]

        # 检查是否已存在（断点续传）
        indices_need = []
        for idx in batch_indices:
            subdir = os.path.join(features_dir, f"{idx // 1000:03d}")
            feat_path = os.path.join(subdir, f"{idx:06d}.pt")
            if os.path.exists(feat_path):
                skipped += 1
                pbar.update(1)
            else:
                indices_need.append(idx)

        if not indices_need:
            continue

        # 加载图像
        try:
            images = [get_image(idx) for idx in indices_need]
        except Exception as e:
            print(f"\nError loading images for batch starting at {batch_indices[0]}: {e}")
            errors += len(indices_need)
            pbar.update(len(indices_need))
            continue

        # 提取特征
        try:
            features = extract_pre_merge_batch(images)
        except Exception as e:
            print(f"\nError extracting features for batch starting at {batch_indices[0]}: {e}")
            import traceback
            traceback.print_exc()
            errors += len(indices_need)
            pbar.update(len(indices_need))
            continue

        # 保存
        for idx, feat in zip(indices_need, features):
            subdir = os.path.join(features_dir, f"{idx // 1000:03d}")
            os.makedirs(subdir, exist_ok=True)
            feat_path = os.path.join(subdir, f"{idx:06d}.pt")
            torch.save(feat, feat_path)
            processed += 1
            pbar.update(1)

    pbar.close()
    print(f"\nDone! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")


# ============================================================================
# Stage 4: SVG Path Grouping
# ============================================================================

def parse_svg_paths(svg_string: str) -> List[Dict]:
    """
    解析 SVG 字符串，提取所有 path 的信息。
    返回: [{d: str, fill: str, commands: List, bbox: (x0,y0,x1,y1), complexity: float}, ...]
    """
    paths = []

    # 匹配所有 <path ... /> 标签
    path_pattern = re.compile(r'<path\s+([^>]*)/?>', re.DOTALL)

    for match in path_pattern.finditer(svg_string):
        attrs_str = match.group(1)

        # 提取 d 属性
        d_match = re.search(r'd="([^"]*)"', attrs_str)
        if not d_match:
            continue
        d_str = d_match.group(1).strip()
        if not d_str:
            continue

        # 提取 fill 颜色
        fill_match = re.search(r'fill="([^"]*)"', attrs_str)
        fill = fill_match.group(1) if fill_match else "#000000"

        # 解析命令和坐标
        commands = parse_path_commands(d_str)
        if not commands:
            continue

        # 计算 bounding box
        bbox = compute_path_bbox(commands)

        # 计算复杂度
        complexity = compute_path_complexity(commands, bbox)

        paths.append({
            "d": d_str,
            "fill": fill,
            "commands": commands,
            "bbox": bbox,
            "complexity": complexity,
        })

    return paths


def parse_path_commands(d_str: str) -> List[Dict]:
    """
    解析 SVG path 的 d 属性，提取命令和坐标。
    只处理绝对命令（OmniSVG 的 SVG 都是绝对坐标）。
    """
    commands = []

    # 分割命令：每个大写字母开始一个新命令
    # 匹配: 命令字母 + 后续数字/空格/逗号/负号
    tokens = re.findall(r'([MLCQAZmlcqaz])([\s\d.,eE+-]*)', d_str)

    for cmd_char, coords_str in tokens:
        cmd_type = cmd_char.upper()

        # 提取数字
        numbers = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', coords_str)
        coords = [float(n) for n in numbers]

        if cmd_type == 'Z':
            commands.append({"type": "Z", "coords": []})
        elif cmd_type == 'M':
            # MoveTo: x, y
            for i in range(0, len(coords) - 1, 2):
                commands.append({"type": "M", "coords": coords[i:i + 2]})
        elif cmd_type == 'L':
            # LineTo: x, y
            for i in range(0, len(coords) - 1, 2):
                commands.append({"type": "L", "coords": coords[i:i + 2]})
        elif cmd_type == 'C':
            # CurveTo (cubic bezier): x1,y1, x2,y2, x,y
            for i in range(0, len(coords) - 5, 6):
                commands.append({"type": "C", "coords": coords[i:i + 6]})
        elif cmd_type == 'Q':
            # QuadTo: x1,y1, x,y
            for i in range(0, len(coords) - 3, 4):
                commands.append({"type": "Q", "coords": coords[i:i + 4]})
        elif cmd_type == 'A':
            # Arc: rx,ry, rotation, large_arc, sweep, x,y
            for i in range(0, len(coords) - 6, 7):
                commands.append({"type": "A", "coords": coords[i:i + 7]})

    return commands


def compute_path_bbox(commands: List[Dict]) -> Tuple[float, float, float, float]:
    """
    从 path 命令中计算 bounding box。
    对于曲线，使用控制点的 bbox 作为近似（会略大于真实 bbox，但足够用）。
    返回: (x_min, y_min, x_max, y_max)
    """
    all_x = []
    all_y = []

    for cmd in commands:
        coords = cmd["coords"]
        if cmd["type"] == "Z":
            continue
        elif cmd["type"] in ("M", "L"):
            if len(coords) >= 2:
                all_x.append(coords[0])
                all_y.append(coords[1])
        elif cmd["type"] == "C":
            # 3 个点：控制点1, 控制点2, 终点
            for i in range(0, len(coords) - 1, 2):
                all_x.append(coords[i])
                all_y.append(coords[i + 1])
        elif cmd["type"] == "Q":
            # 2 个点：控制点, 终点
            for i in range(0, len(coords) - 1, 2):
                all_x.append(coords[i])
                all_y.append(coords[i + 1])
        elif cmd["type"] == "A":
            # 终点
            if len(coords) >= 7:
                all_x.append(coords[5])
                all_y.append(coords[6])

    if not all_x or not all_y:
        return (0, 0, VIEWBOX_SIZE, VIEWBOX_SIZE)

    return (min(all_x), min(all_y), max(all_x), max(all_y))


def compute_path_complexity(commands: List[Dict], bbox: Tuple[float, float, float, float]) -> float:
    """
    计算 path 的复杂度分数。
    复杂度 = 命令加权分数 × 面积权重
    """
    # 命令加权
    cmd_weights = {"M": 0.5, "L": 1.0, "Q": 2.0, "C": 3.0, "A": 3.0, "Z": 0.2}
    cmd_score = sum(cmd_weights.get(c["type"], 1.0) for c in commands)

    # 面积权重：bbox 面积 / 总面积
    total_area = VIEWBOX_SIZE * VIEWBOX_SIZE
    bbox_w = max(bbox[2] - bbox[0], 1.0)
    bbox_h = max(bbox[3] - bbox[1], 1.0)
    area_ratio = (bbox_w * bbox_h) / total_area
    area_weight = max(area_ratio, 0.05)  # 最少 5%

    return cmd_score * (0.5 + 0.5 * area_weight)


def decide_num_groups(total_complexity: float, num_paths: int) -> int:
    """
    根据总复杂度和 path 数量决定分组数。
    """
    if total_complexity < 30 or num_paths <= 2:
        return 1
    elif total_complexity < 80 or num_paths <= 5:
        return 2
    elif total_complexity < 150:
        return 3
    else:
        return 4


def group_paths_sequential(
    paths: List[Dict],
    num_groups: int,
) -> List[Dict]:
    """
    按 SVG path 顺序，根据累积复杂度均匀切分。
    每次 split 后动态重算 target，保证尽量均匀。
    返回: [{path_indices: [int], bbox: (x0,y0,x1,y1), complexity: float}, ...]
    """
    if num_groups <= 0 or num_groups == 1 or len(paths) <= 1:
        # 单组或无法分割
        total_c = sum(p["complexity"] for p in paths)
        return [_make_group(paths, list(range(len(paths))), total_c)]

    # 限制组数不超过 path 数
    num_groups = min(num_groups, len(paths))

    complexities = [p["complexity"] for p in paths]
    remaining_complexity = sum(complexities)

    groups = []
    current_indices = []
    current_sum = 0.0

    for i, (path, c) in enumerate(zip(paths, complexities)):
        current_indices.append(i)
        current_sum += c

        groups_still_needed = num_groups - len(groups)  # 包含当前正在构建的这一组
        remaining_paths = len(paths) - i - 1

        # 动态计算当前组的 target（剩余复杂度 / 剩余组数）
        target = remaining_complexity / groups_still_needed if groups_still_needed > 0 else remaining_complexity

        # 强制分割条件：剩余 path 数 == 剩余待建组数（不含当前组）
        # 即每个剩余 path 必须独占一组
        force_split = (remaining_paths > 0 and remaining_paths == groups_still_needed - 1)

        # 正常分割条件
        normal_split = (
            current_sum >= target
            and groups_still_needed > 1  # 还需要至少 2 组（当前 + 后续）
            and remaining_paths >= 1     # 后面还有 path
        )

        if force_split or normal_split:
            groups.append(_make_group(paths, current_indices, current_sum))
            remaining_complexity -= current_sum
            current_indices = []
            current_sum = 0.0

    # 最后一组
    if current_indices:
        groups.append(_make_group(paths, current_indices, current_sum))

    return groups


def _make_group(paths: List[Dict], indices: List[int], complexity: float) -> Dict:
    """构造一个分组记录。"""
    # 合并 bounding box
    all_x0, all_y0, all_x1, all_y1 = [], [], [], []
    for i in indices:
        bbox = paths[i]["bbox"]
        all_x0.append(bbox[0])
        all_y0.append(bbox[1])
        all_x1.append(bbox[2])
        all_y1.append(bbox[3])

    merged_bbox = (min(all_x0), min(all_y0), max(all_x1), max(all_y1))

    # 加 padding (10%)
    bw = merged_bbox[2] - merged_bbox[0]
    bh = merged_bbox[3] - merged_bbox[1]
    pad = max(bw, bh) * 0.1
    padded_bbox = (
        max(merged_bbox[0] - pad, 0),
        max(merged_bbox[1] - pad, 0),
        min(merged_bbox[2] + pad, VIEWBOX_SIZE),
        min(merged_bbox[3] + pad, VIEWBOX_SIZE),
    )

    return {
        "path_indices": indices,
        "bbox": padded_bbox,
        "complexity": complexity,
    }


def bbox_to_feature_coords(bbox: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    """
    将 SVG viewbox 坐标的 bbox 映射到 feature map 的网格坐标 (16×16)。
    返回: (row_start, row_end, col_start, col_end)，用于 feature_map[r1:r2, c1:c2]
    """
    scale = FEATURE_GRID_H / VIEWBOX_SIZE  # 16 / 200 = 0.08

    col_start = int(bbox[0] * scale)
    row_start = int(bbox[1] * scale)
    col_end = max(int(math.ceil(bbox[2] * scale)), col_start + 1)
    row_end = max(int(math.ceil(bbox[3] * scale)), row_start + 1)

    # Clamp
    col_start = max(0, min(col_start, FEATURE_GRID_W - 1))
    row_start = max(0, min(row_start, FEATURE_GRID_H - 1))
    col_end = max(1, min(col_end, FEATURE_GRID_W))
    row_end = max(1, min(row_end, FEATURE_GRID_H))

    return (row_start, row_end, col_start, col_end)


def stage_groups(data_dir: str, output_dir: str):
    """
    解析所有样本的 SVG，计算 path 分组信息。
    输出: groups.jsonl
    """
    import pyarrow.parquet as pq

    print("=" * 60)
    print("Stage 4: SVG Path Grouping")
    print("=" * 60)

    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    groups_path = os.path.join(output_dir, "groups.jsonl")

    # 加载 metadata
    records = []
    with open(metadata_path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    N = len(records)

    # 预加载 parquet tables
    print("Loading parquet files...")
    parquet_tables = {}
    parquet_files = sorted([
        f for f in os.listdir(data_dir) if f.endswith(".parquet")
    ])
    for pf in tqdm(parquet_files, desc="Loading parquets"):
        parquet_tables[pf] = pq.read_table(os.path.join(data_dir, pf))

    # 处理每个样本
    stats = {"1_group": 0, "2_groups": 0, "3_groups": 0, "4_groups": 0, "errors": 0}

    with open(groups_path, "w") as fout:
        for rec in tqdm(records, desc="Parsing SVGs"):
            idx = rec["idx"]
            pf = rec["parquet_file"]
            row = rec["parquet_row"]

            try:
                svg_str = parquet_tables[pf].column("svg")[row].as_py()
                paths = parse_svg_paths(svg_str)

                if not paths:
                    # 无法解析的 SVG，创建默认单组
                    group_record = {
                        "idx": idx,
                        "num_paths": 0,
                        "total_complexity": 0,
                        "num_groups": 1,
                        "groups": [{
                            "path_indices": [],
                            "bbox": (0, 0, VIEWBOX_SIZE, VIEWBOX_SIZE),
                            "bbox_feature": (0, FEATURE_GRID_H, 0, FEATURE_GRID_W),
                            "complexity": 0,
                        }],
                    }
                else:
                    total_complexity = sum(p["complexity"] for p in paths)
                    num_groups = decide_num_groups(total_complexity, len(paths))
                    groups = group_paths_sequential(paths, num_groups)

                    # 添加 feature map 坐标
                    group_dicts = []
                    for g in groups:
                        feat_coords = bbox_to_feature_coords(g["bbox"])
                        group_dicts.append({
                            "path_indices": g["path_indices"],
                            "bbox": list(g["bbox"]),
                            "bbox_feature": list(feat_coords),
                            "complexity": round(g["complexity"], 2),
                        })

                    group_record = {
                        "idx": idx,
                        "num_paths": len(paths),
                        "total_complexity": round(total_complexity, 2),
                        "num_groups": num_groups,
                        "groups": group_dicts,
                    }

                    stats[f"{num_groups}_group{'s' if num_groups > 1 else ''}"] += 1

                fout.write(json.dumps(group_record) + "\n")

            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 10:
                    print(f"\nError processing idx={idx}: {e}")
                # 写入默认值
                group_record = {
                    "idx": idx,
                    "num_paths": 0,
                    "total_complexity": 0,
                    "num_groups": 1,
                    "groups": [{
                        "path_indices": [],
                        "bbox": [0, 0, VIEWBOX_SIZE, VIEWBOX_SIZE],
                        "bbox_feature": [0, FEATURE_GRID_H, 0, FEATURE_GRID_W],
                        "complexity": 0,
                    }],
                }
                fout.write(json.dumps(group_record) + "\n")

    print(f"\nGrouping statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Saved groups to: {groups_path}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="HVM-SVG Data Preprocessing")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "metadata", "rag", "features", "groups"],
                        help="Which stage to run")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Path to parquet data directory")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to Qwen2.5-VL-7B model")
    parser.add_argument("--clip_model_path", type=str, default=DEFAULT_CLIP_MODEL_PATH,
                        help="Path to CLIP model for text embedding (RAG retrieval)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for precomputed data")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID for feature extraction")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for feature extraction")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Number of shards for parallel feature extraction")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard ID (0-indexed)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"HVM-SVG Data Preprocessing")
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Model path:      {args.model_path}")
    print(f"  CLIP model path: {args.clip_model_path}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Stage:           {args.stage}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    stages = {
        "metadata": lambda: stage_metadata(args.data_dir, args.output_dir),
        "rag": lambda: stage_rag(args.output_dir, args.clip_model_path),
        "features": lambda: stage_features(
            args.data_dir, args.output_dir, args.model_path,
            gpu_id=args.gpu_id,
            batch_size=args.batch_size,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
        ),
        "groups": lambda: stage_groups(args.data_dir, args.output_dir),
    }

    if args.stage == "all":
        for stage_name in ["metadata", "rag", "features", "groups"]:
            print(f"\n{'='*60}")
            print(f"Running stage: {stage_name}")
            print(f"{'='*60}\n")
            stages[stage_name]()
    else:
        stages[args.stage]()

    print("\nAll done!")


if __name__ == "__main__":
    main()
