#!/usr/bin/env python
"""
为 val / test 数据集构建 HVM 预计算数据。

复用已有的全库 faiss_index.bin + metadata.jsonl + text_embeddings.npy 做检索，
然后提取 val/test 样本自身及其 ref 的 features / groups / group_features。

输出目录结构（以 val 为例）:
  hvm_val/
    metadata.jsonl          # val 样本的 metadata（idx 从 0 开始）
    rag_results_train.jsonl # val 样本的 RAG 检索结果（ref_indices 指向全库 idx）
    groups_train_ref.jsonl  # Top-1 ref 的 SVG 分组信息
    features/               # val 样本 + ref 的整图 features
    group_features/         # Top-1 ref 的 group features

用法:
    # 1. RAG 检索 + metadata（CPU，快速）
    python build_faiss_index/build_val_test_hvm.py --split val --stage rag

    # 2. 提取 features（GPU，多卡并行）
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i python build_faiss_index/build_val_test_hvm.py \
            --split val --stage features --shard_id $i --num_shards 8 --batch_size 16 &
    done; wait

    # 3. 生成 groups（CPU）
    python build_faiss_index/build_val_test_hvm.py --split val --stage groups

    # 4. 提取 group_features（GPU，多卡并行）
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i python build_faiss_index/build_val_test_hvm.py \
            --split val --stage group_features --shard_id $i --num_shards 8 --batch_size 8 &
    done; wait

    # 或者一步到位（单卡，适合 500 条小数据）:
    CUDA_VISIBLE_DEVICES=0 python build_faiss_index/build_val_test_hvm.py --split val --stage all
"""

import argparse
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ============================================================================
# Paths
# ============================================================================

BASE_DIR = "/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration"
CORPUS_HVM_DIR = os.path.join(BASE_DIR, "hvm_precomputed_1w")       # 全库元数据 (metadata, faiss_index, text_embeddings)
TRAIN_HVM_DIR = os.path.join(BASE_DIR, "hvm_precomputed_1w_nozoom") # train 处理结果 (features, groups, group_features)
CORPUS_DATA_DIR = os.path.join(BASE_DIR, "data_retrieval_corpus")
MODEL_PATH = "/mnt/data/wuqingman/models/Qwen/Qwen2.5-VL-7B-Instruct"
CLIP_MODEL_PATH = "/mnt/data/wuqingman/models/openai/clip-vit-large-patch14"

SPLIT_CONFIG = {
    "val": {
        "parquet_dir": os.path.join(BASE_DIR, "data_val"),
        "hvm_dir": os.path.join(BASE_DIR, "hvm_val_nozoom"),
    },
    "test": {
        "parquet_dir": os.path.join(BASE_DIR, "data_test_holdout"),
        "hvm_dir": os.path.join(BASE_DIR, "hvm_test_nozoom"),
    },
}

VIEWBOX_SIZE = 200
IMAGE_SIZE = 448
TOKENS_PER_IMAGE = 256
RAG_TOP_K = 3


# ============================================================================
# SVG parsing (same as extract_train_ref_data.py)
# ============================================================================

def parse_path_commands(d_str):
    commands = []
    tokens = re.findall(r'([MLCQAZmlcqaz])([\s\d.,eE+-]*)', d_str)
    for cmd_char, coords_str in tokens:
        cmd_type = cmd_char.upper()
        numbers = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', coords_str)
        coords = [float(n) for n in numbers]
        if cmd_type == 'Z':
            commands.append({"type": "Z", "coords": []})
        elif cmd_type == 'M':
            for i in range(0, len(coords) - 1, 2):
                commands.append({"type": "M", "coords": coords[i:i+2]})
        elif cmd_type == 'L':
            for i in range(0, len(coords) - 1, 2):
                commands.append({"type": "L", "coords": coords[i:i+2]})
        elif cmd_type == 'C':
            for i in range(0, len(coords) - 5, 6):
                commands.append({"type": "C", "coords": coords[i:i+6]})
        elif cmd_type == 'Q':
            for i in range(0, len(coords) - 3, 4):
                commands.append({"type": "Q", "coords": coords[i:i+4]})
        elif cmd_type == 'A':
            for i in range(0, len(coords) - 6, 7):
                commands.append({"type": "A", "coords": coords[i:i+7]})
    return commands


def compute_path_bbox(commands):
    all_x, all_y = [], []
    for cmd in commands:
        coords = cmd["coords"]
        if cmd["type"] == "Z":
            continue
        elif cmd["type"] in ("M", "L"):
            if len(coords) >= 2:
                all_x.append(coords[0]); all_y.append(coords[1])
        elif cmd["type"] == "C":
            for i in range(0, len(coords) - 1, 2):
                all_x.append(coords[i]); all_y.append(coords[i+1])
        elif cmd["type"] == "Q":
            for i in range(0, len(coords) - 1, 2):
                all_x.append(coords[i]); all_y.append(coords[i+1])
        elif cmd["type"] == "A":
            if len(coords) >= 7:
                all_x.append(coords[5]); all_y.append(coords[6])
    if not all_x or not all_y:
        return (0, 0, VIEWBOX_SIZE, VIEWBOX_SIZE)
    return (min(all_x), min(all_y), max(all_x), max(all_y))


def compute_path_complexity(commands, bbox):
    cmd_weights = {"M": 0.5, "L": 1.0, "Q": 2.0, "C": 3.0, "A": 3.0, "Z": 0.2}
    cmd_score = sum(cmd_weights.get(c["type"], 1.0) for c in commands)
    total_area = VIEWBOX_SIZE * VIEWBOX_SIZE
    bbox_w = max(bbox[2] - bbox[0], 1.0)
    bbox_h = max(bbox[3] - bbox[1], 1.0)
    area_ratio = (bbox_w * bbox_h) / total_area
    area_weight = max(area_ratio, 0.05)
    return cmd_score * (0.5 + 0.5 * area_weight)


def parse_svg_paths(svg_string):
    paths = []
    path_pattern = re.compile(r'<path\s+([^>]*)/?>', re.DOTALL)
    for match in path_pattern.finditer(svg_string):
        attrs_str = match.group(1)
        d_match = re.search(r'd="([^"]*)"', attrs_str)
        if not d_match:
            continue
        d_str = d_match.group(1).strip()
        if not d_str:
            continue
        fill_match = re.search(r'fill="([^"]*)"', attrs_str)
        fill = fill_match.group(1) if fill_match else "#000000"
        commands = parse_path_commands(d_str)
        if not commands:
            continue
        bbox = compute_path_bbox(commands)
        complexity = compute_path_complexity(commands, bbox)
        paths.append({"d": d_str, "fill": fill, "commands": commands, "bbox": bbox, "complexity": complexity})
    return paths


def decide_num_groups(total_complexity, num_paths):
    """固定分为 4 组。"""
    return 4


def _make_group(paths, indices, complexity):
    all_x0, all_y0, all_x1, all_y1 = [], [], [], []
    for i in indices:
        bbox = paths[i]["bbox"]
        all_x0.append(bbox[0]); all_y0.append(bbox[1])
        all_x1.append(bbox[2]); all_y1.append(bbox[3])
    merged_bbox = (min(all_x0), min(all_y0), max(all_x1), max(all_y1))
    bw = merged_bbox[2] - merged_bbox[0]
    bh = merged_bbox[3] - merged_bbox[1]
    pad = max(bw, bh) * 0.1
    padded_bbox = (
        max(merged_bbox[0] - pad, 0), max(merged_bbox[1] - pad, 0),
        min(merged_bbox[2] + pad, VIEWBOX_SIZE), min(merged_bbox[3] + pad, VIEWBOX_SIZE),
    )
    return {"path_indices": indices, "bbox": padded_bbox, "complexity": complexity}


def group_paths_sequential(paths, num_groups):
    if num_groups <= 0 or num_groups == 1 or len(paths) <= 1:
        total_c = sum(p["complexity"] for p in paths)
        return [_make_group(paths, list(range(len(paths))), total_c)]
    num_groups = min(num_groups, len(paths))
    complexities = [p["complexity"] for p in paths]
    remaining_complexity = sum(complexities)
    groups = []
    current_indices = []
    current_sum = 0.0
    for i, (path, c) in enumerate(zip(paths, complexities)):
        current_indices.append(i)
        current_sum += c
        groups_still_needed = num_groups - len(groups)
        remaining_paths = len(paths) - i - 1
        target = remaining_complexity / groups_still_needed if groups_still_needed > 0 else remaining_complexity
        force_split = (remaining_paths > 0 and remaining_paths == groups_still_needed - 1)
        normal_split = (current_sum >= target and groups_still_needed > 1 and remaining_paths >= 1)
        if force_split or normal_split:
            groups.append(_make_group(paths, current_indices, current_sum))
            remaining_complexity -= current_sum
            current_indices = []
            current_sum = 0.0
    if current_indices:
        groups.append(_make_group(paths, current_indices, current_sum))
    return groups


def render_group_to_image(svg_string, group_path_indices, image_size=IMAGE_SIZE):
    """使用原始 SVG 尺寸渲染 group paths，不 crop/zoom。"""
    import cairosvg
    path_pattern = re.compile(r'<path\s[^>]*?(?:/>|>\s*</path>)', re.DOTALL)
    all_path_tags = path_pattern.findall(svg_string)
    if not all_path_tags:
        return None
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {VIEWBOX_SIZE} {VIEWBOX_SIZE}" '
        f'width="{image_size}" height="{image_size}">',
        f'<rect x="0" y="0" width="{VIEWBOX_SIZE}" height="{VIEWBOX_SIZE}" fill="white"/>',
    ]
    for pi in group_path_indices:
        if pi < len(all_path_tags):
            svg_lines.append(all_path_tags[pi])
    svg_lines.append('</svg>')
    svg_content = '\n'.join(svg_lines)
    try:
        png_bytes = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'),
                                      output_width=image_size, output_height=image_size)
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception:
        return None


# ============================================================================
# Helpers
# ============================================================================

def load_full_metadata():
    """加载全库 metadata（25w）"""
    meta_path = os.path.join(CORPUS_HVM_DIR, "metadata.jsonl")
    meta = {}
    with open(meta_path) as f:
        for line in f:
            r = json.loads(line)
            meta[r["idx"]] = r
    return meta


def load_parquets_for_meta(meta_map, data_dir):
    """根据 metadata 加载需要的 parquet 文件。"""
    import pyarrow.parquet as pq
    needed = set(m["parquet_file"] for m in meta_map.values())
    tables = {}
    for pf in sorted(needed):
        # 优先从 corpus 目录加载，如果不存在则尝试 data_test 目录
        full_path = os.path.join(data_dir, pf)
        if os.path.exists(full_path):
            tables[pf] = pq.read_table(full_path)
        else:
            alt_path = os.path.join(BASE_DIR, "data_test", pf)
            if os.path.exists(alt_path):
                tables[pf] = pq.read_table(alt_path)
            else:
                print(f"  WARNING: {pf} not found in {data_dir} or data_test, skipping")
    return tables


# ============================================================================
# Stage: RAG (用已有 faiss_index 检索)
# ============================================================================

def run_rag(split_name: str, hvm_out_dir: str, parquet_dir: str):
    """
    为 val/test 构建 metadata + RAG 检索结果。

    val/test 样本的 idx 从 0 开始编号（独立于训练集）。
    RAG 检索结果中的 ref_indices 指向全库 idx（与 hvm_precomputed_1w/metadata.jsonl 一致）。
    """
    import faiss
    import pyarrow.parquet as pq
    from transformers import CLIPModel, CLIPTokenizer

    print("=" * 60)
    print(f"Stage: RAG for {split_name}")
    print("=" * 60)

    os.makedirs(hvm_out_dir, exist_ok=True)

    # 1. 加载 val/test parquet，生成 metadata
    parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith(".parquet")])
    print(f"  Parquet files: {parquet_files}")

    records = []
    global_idx = 0
    for pf in parquet_files:
        table = pq.read_table(os.path.join(parquet_dir, pf))
        for i in range(table.num_rows):
            rec = {
                "idx": global_idx,
                "id": table.column("id")[i].as_py(),
                "description": table.column("description")[i].as_py(),
                "keywords": table.column("keywords")[i].as_py() or "",
                "detail": table.column("detail")[i].as_py() or "",
                "token_len": table.column("token_len")[i].as_py(),
                "parquet_file": pf,
                "parquet_row": i,
            }
            records.append(rec)
            global_idx += 1

    print(f"  {split_name} samples: {len(records)}")

    # 保存 metadata
    meta_path = os.path.join(hvm_out_dir, "metadata.jsonl")
    with open(meta_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved metadata: {meta_path}")

    # 2. 加载已有的 faiss_index（全库 25w）
    index_path = os.path.join(CORPUS_HVM_DIR, "faiss_index.bin")
    print(f"  Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    print(f"  FAISS index size: {index.ntotal}")

    # 3. 用 CLIP 编码 val/test 的 descriptions
    print(f"  Loading CLIP model from {CLIP_MODEL_PATH}...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH)
    clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = clip_model.to(device).eval()
    embed_dim = clip_model.config.projection_dim

    texts = []
    for r in records:
        desc = r["description"]
        kw = r.get("keywords", "")
        texts.append(f"{desc}. {kw}" if kw else desc)

    N = len(texts)
    embeddings = np.zeros((N, embed_dim), dtype=np.float32)
    BATCH_SIZE = 256
    for start in tqdm(range(0, N, BATCH_SIZE), desc="Encoding CLIP"):
        end = min(start + BATCH_SIZE, N)
        inputs = clip_tokenizer(
            texts[start:end], padding=True, truncation=True,
            max_length=77, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            feats = clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings[start:end] = feats.cpu().numpy()

    del clip_model
    torch.cuda.empty_cache()

    # 4. 检索 Top-K（从全库中检索，不排除自身，因为 val/test 不在全库中）
    K = RAG_TOP_K
    print(f"  Searching Top-{K} from full corpus...")
    scores, indices = index.search(embeddings, K)

    # 5. 保存 RAG 结果
    rag_path = os.path.join(hvm_out_dir, "rag_results_train.jsonl")
    with open(rag_path, "w") as f:
        for i in range(N):
            ref_indices = [int(indices[i, j]) for j in range(K)]
            ref_scores = [float(scores[i, j]) for j in range(K)]
            rec = {
                "idx": i,
                "ref_indices": ref_indices,
                "ref_scores": ref_scores,
            }
            f.write(json.dumps(rec) + "\n")

    print(f"  Saved RAG results: {rag_path}")
    print(f"  Done! {N} samples, each with {K} refs")


# ============================================================================
# Stage: Features
# ============================================================================

def run_features(split_name: str, hvm_out_dir: str, parquet_dir: str,
                 shard_id: int, num_shards: int, batch_size: int):
    """
    提取 val/test 样本自身 + 其 ref 的整图 features [256, 3584]。
    对于 ref，优先从 hvm_precomputed_1w/features/ 复制（避免重复计算）。
    """
    print("=" * 60)
    print(f"Stage: Features for {split_name} (shard {shard_id}/{num_shards})")
    print("=" * 60)

    features_dir = os.path.join(hvm_out_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    train_features_dir = os.path.join(TRAIN_HVM_DIR, "features")

    # 加载 RAG 结果，收集所有需要 features 的 idx
    rag_path = os.path.join(hvm_out_dir, "rag_results_train.jsonl")
    all_ref_indices = set()
    val_indices = set()
    with open(rag_path) as f:
        for line in f:
            r = json.loads(line)
            val_indices.add(r["idx"])
            for ri in r["ref_indices"]:
                all_ref_indices.add(ri)

    # ref_indices 指向全库 idx，需要从全库 metadata 找到对应的 parquet
    # val 样本自身的 idx 是 local idx（0-based），需要从 val metadata 找到 parquet
    print(f"  Val/test samples: {len(val_indices)}")
    print(f"  Unique ref indices: {len(all_ref_indices)}")

    # ---- 处理 ref features（从全库）----
    # 先检查哪些 ref 已经在 train features 中
    refs_to_compute = []
    refs_copied = 0
    for ref_idx in sorted(all_ref_indices):
        out_subdir = os.path.join(features_dir, f"ref_{ref_idx // 1000:03d}")
        out_path = os.path.join(out_subdir, f"ref_{ref_idx:06d}.pt")
        if os.path.exists(out_path):
            continue

        # 尝试从训练集 features 复制
        train_path = os.path.join(train_features_dir, f"{ref_idx // 1000:03d}", f"{ref_idx:06d}.pt")
        if os.path.exists(train_path):
            os.makedirs(out_subdir, exist_ok=True)
            import shutil
            shutil.copy2(train_path, out_path)
            refs_copied += 1
        else:
            refs_to_compute.append(ref_idx)

    print(f"  Ref features copied from train: {refs_copied}")
    print(f"  Ref features to compute: {len(refs_to_compute)}")

    # ---- 处理 val/test 样本自身的 features ----
    val_meta_path = os.path.join(hvm_out_dir, "metadata.jsonl")
    val_records = []
    with open(val_meta_path) as f:
        for line in f:
            val_records.append(json.loads(line))

    val_to_compute = []
    for rec in val_records:
        idx = rec["idx"]
        out_subdir = os.path.join(features_dir, f"val_{idx // 1000:03d}")
        out_path = os.path.join(out_subdir, f"val_{idx:06d}.pt")
        if not os.path.exists(out_path):
            val_to_compute.append(rec)

    print(f"  Val/test self features to compute: {len(val_to_compute)}")

    # 合并所有需要计算的任务
    all_tasks = []
    # val/test 自身
    for rec in val_to_compute:
        all_tasks.append(("val", rec["idx"], rec["parquet_file"], rec["parquet_row"], parquet_dir))
    # ref（需要从全库 metadata 查找）
    if refs_to_compute:
        full_meta = load_full_metadata()
        for ref_idx in refs_to_compute:
            if ref_idx in full_meta:
                m = full_meta[ref_idx]
                all_tasks.append(("ref", ref_idx, m["parquet_file"], m["parquet_row"], CORPUS_DATA_DIR))

    if not all_tasks:
        print("  All features already exist!")
        return

    # 分片
    shard_size = math.ceil(len(all_tasks) / num_shards)
    start = shard_id * shard_size
    end = min(start + shard_size, len(all_tasks))
    my_tasks = all_tasks[start:end]
    print(f"  Shard {shard_id}: {len(my_tasks)} tasks")

    if not my_tasks:
        return

    # 加载 parquet tables
    import pyarrow.parquet as pq
    parquet_tables = {}
    for task_type, idx, pf, row, data_dir in my_tasks:
        if (data_dir, pf) not in parquet_tables:
            full_path = os.path.join(data_dir, pf)
            if os.path.exists(full_path):
                parquet_tables[(data_dir, pf)] = pq.read_table(full_path)
            else:
                # 尝试 data_test 目录
                alt_path = os.path.join(BASE_DIR, "data_test", pf)
                if os.path.exists(alt_path):
                    parquet_tables[(data_dir, pf)] = pq.read_table(alt_path)

    # 加载 vision encoder
    device = torch.device("cuda:0")
    print("  Loading Qwen2.5-VL vision encoder...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")
    visual = model.visual.to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    del model.model; del model.lm_head; del model
    torch.cuda.empty_cache()

    def extract_batch(images):
        from qwen_vl_utils import process_vision_info
        all_pv, all_gt = [], []
        for img in images:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img}, {"type": "text", "text": "x"}]}]
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text_input], images=image_inputs, return_tensors="pt")
            all_pv.append(inputs["pixel_values"])
            all_gt.append(inputs["image_grid_thw"])
        pixel_values = torch.cat(all_pv, dim=0).to(device, dtype=torch.float16)
        grid_thw = torch.cat(all_gt, dim=0).to(device)
        with torch.no_grad():
            post_merge = visual(pixel_values, grid_thw=grid_thw)
        return [post_merge[b*TOKENS_PER_IMAGE:(b+1)*TOKENS_PER_IMAGE].cpu().half()
                for b in range(len(images))]

    processed, errors = 0, 0
    pbar = tqdm(total=len(my_tasks), desc=f"Features shard {shard_id}")

    for bs in range(0, len(my_tasks), batch_size):
        batch = my_tasks[bs:bs+batch_size]
        try:
            images = []
            for task_type, idx, pf, row, data_dir in batch:
                table = parquet_tables.get((data_dir, pf))
                if table is None:
                    raise ValueError(f"Parquet not loaded: {data_dir}/{pf}")
                img_data = table.column("image")[row].as_py()
                images.append(Image.open(io.BytesIO(img_data["bytes"])).convert("RGB"))

            feats = extract_batch(images)

            for (task_type, idx, pf, row, data_dir), feat in zip(batch, feats):
                if task_type == "val":
                    out_subdir = os.path.join(features_dir, f"val_{idx // 1000:03d}")
                    out_path = os.path.join(out_subdir, f"val_{idx:06d}.pt")
                else:
                    out_subdir = os.path.join(features_dir, f"ref_{idx // 1000:03d}")
                    out_path = os.path.join(out_subdir, f"ref_{idx:06d}.pt")
                os.makedirs(out_subdir, exist_ok=True)
                torch.save(feat, out_path)
                processed += 1
        except Exception as e:
            errors += len(batch)
            if errors <= 10:
                print(f"\n  Error: {e}")
                import traceback; traceback.print_exc()
        pbar.update(len(batch))

    pbar.close()
    print(f"\nDone! Processed: {processed}, Errors: {errors}")


# ============================================================================
# Stage: Groups
# ============================================================================

def run_groups(split_name: str, hvm_out_dir: str):
    """为 Top-1 ref 生成 SVG 分组信息。优先从训练集复制。"""
    print("=" * 60)
    print(f"Stage: Groups for {split_name}")
    print("=" * 60)

    # 加载 RAG 结果
    rag_path = os.path.join(hvm_out_dir, "rag_results_train.jsonl")
    top1_refs = set()
    with open(rag_path) as f:
        for line in f:
            r = json.loads(line)
            top1_refs.add(r["ref_indices"][0])

    print(f"  Unique Top-1 refs: {len(top1_refs)}")

    # 尝试从训练集 groups 复制
    train_groups_path = os.path.join(TRAIN_HVM_DIR, "groups_train_ref.jsonl")
    train_groups = {}
    if os.path.exists(train_groups_path):
        with open(train_groups_path) as f:
            for line in f:
                g = json.loads(line)
                train_groups[g["idx"]] = g

    copied = 0
    to_compute = []
    for ref_idx in sorted(top1_refs):
        if ref_idx in train_groups:
            copied += 1
        else:
            to_compute.append(ref_idx)

    print(f"  Groups from train: {copied}")
    print(f"  Groups to compute: {len(to_compute)}")

    # 写出 groups 文件
    groups_path = os.path.join(hvm_out_dir, "groups_train_ref.jsonl")

    if to_compute:
        full_meta = load_full_metadata()
        needed_meta = {idx: full_meta[idx] for idx in to_compute if idx in full_meta}
        tables = load_parquets_for_meta(needed_meta, CORPUS_DATA_DIR)

        for idx in tqdm(to_compute, desc="Computing groups"):
            if idx not in needed_meta:
                train_groups[idx] = {
                    "idx": idx, "num_paths": 0, "total_complexity": 0,
                    "num_groups": 1, "groups": [{"path_indices": [],
                    "bbox": [0, 0, VIEWBOX_SIZE, VIEWBOX_SIZE], "complexity": 0}]
                }
                continue
            m = needed_meta[idx]
            try:
                svg_str = tables[m["parquet_file"]].column("svg")[m["parquet_row"]].as_py()
                paths = parse_svg_paths(svg_str)
                if not paths:
                    rec = {"idx": idx, "num_paths": 0, "total_complexity": 0,
                           "num_groups": 1, "groups": [{"path_indices": [],
                           "bbox": [0, 0, VIEWBOX_SIZE, VIEWBOX_SIZE], "complexity": 0}]}
                else:
                    total_c = sum(p["complexity"] for p in paths)
                    ng = decide_num_groups(total_c, len(paths))
                    gs = group_paths_sequential(paths, ng)
                    rec = {"idx": idx, "num_paths": len(paths),
                           "total_complexity": round(total_c, 2), "num_groups": ng,
                           "groups": [{"path_indices": g["path_indices"],
                                       "bbox": list(g["bbox"]),
                                       "complexity": round(g["complexity"], 2)} for g in gs]}
                train_groups[idx] = rec
            except Exception as e:
                print(f"  Error idx={idx}: {e}")
                train_groups[idx] = {
                    "idx": idx, "num_paths": 0, "total_complexity": 0,
                    "num_groups": 1, "groups": [{"path_indices": [],
                    "bbox": [0, 0, VIEWBOX_SIZE, VIEWBOX_SIZE], "complexity": 0}]
                }

    with open(groups_path, "w") as f:
        for ref_idx in sorted(top1_refs):
            if ref_idx in train_groups:
                f.write(json.dumps(train_groups[ref_idx]) + "\n")

    print(f"  Saved: {groups_path}")


# ============================================================================
# Stage: Group Features
# ============================================================================

def run_group_features(split_name: str, hvm_out_dir: str,
                       shard_id: int, num_shards: int, batch_size: int):
    """为 Top-1 ref 提取 group-level features。优先从训练集复制。"""
    print("=" * 60)
    print(f"Stage: Group Features for {split_name} (shard {shard_id}/{num_shards})")
    print("=" * 60)

    gf_dir = os.path.join(hvm_out_dir, "group_features")
    os.makedirs(gf_dir, exist_ok=True)
    train_gf_dir = os.path.join(TRAIN_HVM_DIR, "group_features")

    # 加载 RAG 结果
    rag_path = os.path.join(hvm_out_dir, "rag_results_train.jsonl")
    top1_refs = set()
    with open(rag_path) as f:
        for line in f:
            r = json.loads(line)
            top1_refs.add(r["ref_indices"][0])

    top1_refs = sorted(top1_refs)
    print(f"  Unique Top-1 refs: {len(top1_refs)}")

    # 先复制已有的
    to_compute = []
    copied = 0
    for ref_idx in top1_refs:
        out_subdir = os.path.join(gf_dir, f"{ref_idx // 1000:03d}")
        out_path = os.path.join(out_subdir, f"{ref_idx:06d}.pt")
        if os.path.exists(out_path):
            continue
        train_path = os.path.join(train_gf_dir, f"{ref_idx // 1000:03d}", f"{ref_idx:06d}.pt")
        if os.path.exists(train_path):
            os.makedirs(out_subdir, exist_ok=True)
            import shutil
            shutil.copy2(train_path, out_path)
            copied += 1
        else:
            to_compute.append(ref_idx)

    print(f"  Copied from train: {copied}")
    print(f"  To compute: {len(to_compute)}")

    if not to_compute:
        print("  All group features exist!")
        return

    # 分片
    shard_size = math.ceil(len(to_compute) / num_shards)
    start = shard_id * shard_size
    end = min(start + shard_size, len(to_compute))
    my_indices = to_compute[start:end]
    print(f"  Shard {shard_id}: {len(my_indices)} indices")

    if not my_indices:
        return

    # 加载 groups
    groups_path = os.path.join(hvm_out_dir, "groups_train_ref.jsonl")
    groups_data = {}
    with open(groups_path) as f:
        for line in f:
            g = json.loads(line)
            groups_data[g["idx"]] = g

    # 加载 metadata
    full_meta = load_full_metadata()
    needed_meta = {idx: full_meta[idx] for idx in my_indices if idx in full_meta}
    tables = load_parquets_for_meta(needed_meta, CORPUS_DATA_DIR)

    # 加载 vision encoder
    device = torch.device("cuda:0")
    print("  Loading Qwen2.5-VL vision encoder...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")
    visual = model.visual.to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    del model.model; del model.lm_head; del model
    torch.cuda.empty_cache()

    def extract_batch(images):
        from qwen_vl_utils import process_vision_info
        all_pv, all_gt = [], []
        for img in images:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img}, {"type": "text", "text": "x"}]}]
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text_input], images=image_inputs, return_tensors="pt")
            all_pv.append(inputs["pixel_values"])
            all_gt.append(inputs["image_grid_thw"])
        pixel_values = torch.cat(all_pv, dim=0).to(device, dtype=torch.float16)
        grid_thw = torch.cat(all_gt, dim=0).to(device)
        with torch.no_grad():
            post_merge = visual(pixel_values, grid_thw=grid_thw)
        return [post_merge[b*TOKENS_PER_IMAGE:(b+1)*TOKENS_PER_IMAGE].cpu().half()
                for b in range(len(images))]

    processed, errors = 0, 0
    pbar = tqdm(total=len(my_indices), desc=f"GroupFeats shard {shard_id}")

    for idx in my_indices:
        if idx not in needed_meta or idx not in groups_data:
            pbar.update(1)
            continue
        m = needed_meta[idx]
        ginfo = groups_data[idx]
        try:
            svg_str = tables[m["parquet_file"]].column("svg")[m["parquet_row"]].as_py()
            if not svg_str or not ginfo.get("groups"):
                pbar.update(1); continue
            group_images = []
            for g in ginfo["groups"]:
                if not g["path_indices"]:
                    continue
                img = render_group_to_image(svg_str, g["path_indices"])
                if img is not None:
                    group_images.append(img)
            if not group_images:
                pbar.update(1); continue
            group_feats = extract_batch(group_images)
            out_subdir = os.path.join(gf_dir, f"{idx // 1000:03d}")
            os.makedirs(out_subdir, exist_ok=True)
            torch.save(group_feats, os.path.join(out_subdir, f"{idx:06d}.pt"))
            processed += 1
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"\n  Error idx={idx}: {e}")
        pbar.update(1)

    pbar.close()
    print(f"\nDone! Processed: {processed}, Errors: {errors}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["val", "test"])
    parser.add_argument("--stage", type=str, required=True,
                        choices=["rag", "features", "groups", "group_features", "all"])
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    cfg = SPLIT_CONFIG[args.split]
    hvm_dir = cfg["hvm_dir"]
    parquet_dir = cfg["parquet_dir"]

    print(f"Building HVM data for: {args.split}")
    print(f"  Parquet dir: {parquet_dir}")
    print(f"  HVM output:  {hvm_dir}")
    print()

    if args.stage == "all":
        run_rag(args.split, hvm_dir, parquet_dir)
        run_features(args.split, hvm_dir, parquet_dir, 0, 1, args.batch_size)
        run_groups(args.split, hvm_dir)
        run_group_features(args.split, hvm_dir, 0, 1, args.batch_size)
    elif args.stage == "rag":
        run_rag(args.split, hvm_dir, parquet_dir)
    elif args.stage == "features":
        run_features(args.split, hvm_dir, parquet_dir, args.shard_id, args.num_shards, args.batch_size)
    elif args.stage == "groups":
        run_groups(args.split, hvm_dir)
    elif args.stage == "group_features":
        run_group_features(args.split, hvm_dir, args.shard_id, args.num_shards, args.batch_size)


if __name__ == "__main__":
    main()
