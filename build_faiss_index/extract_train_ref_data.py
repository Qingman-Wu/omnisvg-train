#!/usr/bin/env python
"""
训练 1w 样本时，每个样本需要查看它的 Top-3 参考图。这些参考图大部分来自 250k 全库（不在原来的 1w 里），所以需要提前把这些参考图的视觉特征算好存到磁盘，训练时直接读取。
需要算的东西有两种：
Global features（整图特征）→ 给 GME 用，3 个 ref 都需要
Group features（分组特征）→ 给 groupwise CDM / EDR 用，Top-3 ref 都需要

只处理 rag_results_train.jsonl 中出现的 ref idx，而非 25 万全量。
数据来源：
  - 训练集 parquet: data_test (idx 0-9999)
  - 全库 parquet:   data_retrieval_corpus (idx 10000+)

输出目录: hvm_precomputed_1w_nozoom_top3part/ 下的
  - metadata.jsonl
  - rag_results_train.jsonl
  - features/
  - groups_train_ref.jsonl
  - group_features/

阶段：
    --stage groups          CPU，单进程，解析 SVG 生成 groups.jsonl
    --stage features        GPU，多卡并行，提取整图 [256,3584] (for GME)
    --stage group_features  GPU，多卡并行，逐 group 渲染+提取 [256,3584] (for PME)
    --stage all             依次运行 groups → features → group_features

依次运行下列四个命令：
python -u build_faiss_index/extract_train_ref_data.py --stage groups

rm -rf /mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_full/features && \
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python -u build_faiss_index/extract_train_ref_data.py \
        --stage features --shard_id $i --num_shards 8 --batch_size 16 &
done
wait
echo "Features done!"


for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i /mnt/data/wuqingman/miniconda3/envs/omnisvg/bin/python -u \
        build_faiss_index/extract_train_ref_data.py \
        --stage group_features --shard_id $i --num_shards 8 --batch_size 8 &
done
wait
echo "Group features done!"

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

INPUT_HVM_DIR = "/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_1w"
SOURCE_HVM_DIR = "/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_1w_nozoom"
OUTPUT_HVM_DIR = "/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_1w_nozoom_top3part"
FULL_DATA_DIR = "/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_retrieval_corpus"
TRAIN_DATA_DIR = "/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test"
MODEL_PATH = "/mnt/data/wuqingman/models/Qwen/Qwen2.5-VL-7B-Instruct"

VIEWBOX_SIZE = 200
IMAGE_SIZE = 448
TOKENS_PER_IMAGE = 256  # 16x16 post-merge


# ============================================================================
# Helpers: load RAG + metadata
# ============================================================================

def load_train_ref_indices():
    """加载训练 1w 样本需要的所有 ref idx + 训练样本自身 idx。

    返回:
        all_feature_indices: 需要提取 features 的全部 idx（ref + 训练样本自身）
        topk_refs: Top-3 中所有 ref idx（用于 groups / group_features）
    """
    rag_path = os.path.join(INPUT_HVM_DIR, "rag_results_train.jsonl")
    all_refs = set()
    train_indices = set()
    with open(rag_path) as f:
        for line in f:
            r = json.loads(line)
            train_indices.add(r["idx"])
            refs = r["ref_indices"]
            for ref in refs:
                all_refs.add(ref)
    all_feature_indices = sorted(all_refs | train_indices)
    return all_feature_indices, sorted(all_refs)


def load_metadata_for_indices(indices_set):
    """只加载需要的 idx 的 metadata。"""
    meta_path = os.path.join(INPUT_HVM_DIR, "metadata.jsonl")
    meta = {}
    with open(meta_path) as f:
        for line in f:
            r = json.loads(line)
            if r["idx"] in indices_set:
                meta[r["idx"]] = r
    return meta


def load_parquets_for_meta(meta_map):
    """根据 metadata 加载需要的 parquet 文件。"""
    import pyarrow.parquet as pq
    needed = set(m["parquet_file"] for m in meta_map.values())
    tables = {}
    for pf in sorted(needed):
        full_path = os.path.join(FULL_DATA_DIR, pf)
        if os.path.exists(full_path):
            tables[pf] = pq.read_table(full_path)
        else:
            alt_path = os.path.join(TRAIN_DATA_DIR, pf)
            if os.path.exists(alt_path):
                tables[pf] = pq.read_table(alt_path)
            else:
                print(f"  WARNING: {full_path} / {alt_path} not found, skipping")
    return tables


def ensure_output_scaffold():
    os.makedirs(OUTPUT_HVM_DIR, exist_ok=True)
    for filename in ("metadata.jsonl", "rag_results_train.jsonl"):
        dst_path = os.path.join(OUTPUT_HVM_DIR, filename)
        if os.path.exists(dst_path):
            continue
        src_candidates = [
            os.path.join(SOURCE_HVM_DIR, filename),
            os.path.join(INPUT_HVM_DIR, filename),
        ]
        for src_path in src_candidates:
            if os.path.exists(src_path):
                import shutil
                shutil.copy2(src_path, dst_path)
                print(f"  Copied scaffold: {src_path} -> {dst_path}")
                break
        else:
            raise FileNotFoundError(f"Missing scaffold file: {filename}")


def link_or_copy_file(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return
    try:
        os.link(src_path, dst_path)
    except OSError:
        import shutil
        shutil.copy2(src_path, dst_path)


# ============================================================================
# SVG parsing (from precompute_hvm_data.py)
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
# Stage: groups (CPU only, single process)
# ============================================================================

def run_groups(all_ref_indices, topk_ref_indices):
    """为 Top-3 中所有 ref 生成 groups，并尽量复用旧 nozoom 结果。"""
    print("=" * 60)
    print(f"Stage: Groups — 为 {len(topk_ref_indices)} 个 Top-3 refs 解析 SVG 分组")
    print("=" * 60)

    idx_set = set(topk_ref_indices)
    meta = load_metadata_for_indices(idx_set)
    print(f"  Loaded metadata for {len(meta)} indices")

    tables = load_parquets_for_meta(meta)
    print(f"  Loaded {len(tables)} parquet files")

    groups_path = os.path.join(OUTPUT_HVM_DIR, "groups_train_ref.jsonl")
    stats = {"1_group": 0, "2_groups": 0, "3_groups": 0, "4_groups": 0, "errors": 0}
    source_groups = {}
    source_groups_path = os.path.join(SOURCE_HVM_DIR, "groups_train_ref.jsonl")
    if os.path.exists(source_groups_path):
        with open(source_groups_path) as f:
            for line in f:
                rec = json.loads(line)
                source_groups[rec["idx"]] = rec
    print(f"  Existing groups from source nozoom: {len(source_groups)}")

    with open(groups_path, "w") as fout:
        for idx in tqdm(topk_ref_indices, desc="Parsing SVGs"):
            if idx in source_groups:
                fout.write(json.dumps(source_groups[idx]) + "\n")
                continue
            if idx not in meta:
                continue
            m = meta[idx]
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
                    stats[f"{ng}_group{'s' if ng > 1 else ''}"] += 1
                fout.write(json.dumps(rec) + "\n")
            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 10:
                    print(f"\n  Error idx={idx}: {e}")
                rec = {"idx": idx, "num_paths": 0, "total_complexity": 0,
                       "num_groups": 1, "groups": [{"path_indices": [],
                       "bbox": [0, 0, VIEWBOX_SIZE, VIEWBOX_SIZE], "complexity": 0}]}
                fout.write(json.dumps(rec) + "\n")

    print(f"\nStats: {stats}")
    print(f"Saved to: {groups_path}")


# ============================================================================
# Stage: features (GPU, multi-shard)
# ============================================================================

def run_features(all_ref_indices, shard_id, num_shards, batch_size):
    """提取整图 [256, 3584] features (for GME)，只处理需要的 ref。"""
    shard_size = math.ceil(len(all_ref_indices) / num_shards)
    start = shard_id * shard_size
    end = min(start + shard_size, len(all_ref_indices))
    my_indices = all_ref_indices[start:end]

    print("=" * 60)
    print(f"Stage: Features — shard {shard_id}/{num_shards}, {len(my_indices)} indices")
    print("=" * 60)

    features_dir = os.path.join(OUTPUT_HVM_DIR, "features")
    os.makedirs(features_dir, exist_ok=True)
    source_features_dir = os.path.join(SOURCE_HVM_DIR, "features")

    # Skip already done
    todo = []
    copied = 0
    for idx in my_indices:
        subdir = os.path.join(features_dir, f"{idx // 1000:03d}")
        out_path = os.path.join(subdir, f"{idx:06d}.pt")
        if os.path.exists(out_path):
            continue
        src_path = os.path.join(source_features_dir, f"{idx // 1000:03d}", f"{idx:06d}.pt")
        if os.path.exists(src_path):
            link_or_copy_file(src_path, out_path)
            copied += 1
        else:
            todo.append(idx)
    print(f"  Already done/copied: {len(my_indices) - len(todo)} (copied={copied}), todo: {len(todo)}")
    if not todo:
        print("  Nothing to do!")
        return

    idx_set = set(todo)
    meta = load_metadata_for_indices(idx_set)
    tables = load_parquets_for_meta(meta)
    print(f"  Loaded {len(meta)} metadata, {len(tables)} parquets")

    # Load vision encoder
    device = torch.device("cuda:0")
    print("  Loading Qwen2.5-VL vision encoder...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")
    visual = model.visual.to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    del model.model; del model.lm_head; del model
    torch.cuda.empty_cache()
    print(f"  Vision encoder on GPU, mem: {torch.cuda.memory_allocated(device)/1e6:.0f} MB")

    def get_image(idx):
        m = meta[idx]
        img_data = tables[m["parquet_file"]].column("image")[m["parquet_row"]].as_py()
        return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")

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
    pbar = tqdm(total=len(todo), desc=f"Features shard {shard_id}")
    for bs in range(0, len(todo), batch_size):
        batch = todo[bs:bs+batch_size]
        try:
            images = [get_image(idx) for idx in batch]
            feats = extract_batch(images)
            for idx, feat in zip(batch, feats):
                subdir = os.path.join(features_dir, f"{idx // 1000:03d}")
                os.makedirs(subdir, exist_ok=True)
                torch.save(feat, os.path.join(subdir, f"{idx:06d}.pt"))
                processed += 1
        except Exception as e:
            errors += len(batch)
            if errors <= 10:
                print(f"\n  Error batch idx={batch[0]}: {e}")
                import traceback; traceback.print_exc()
        pbar.update(len(batch))
    pbar.close()
    print(f"\nDone! Processed: {processed}, Errors: {errors}")


# ============================================================================
# Stage: group_features (GPU, multi-shard)
# ============================================================================

def run_group_features(topk_ref_indices, shard_id, num_shards, batch_size):
    """为 Top-3 refs 提取 group-level features，并尽量复用旧 nozoom 结果。"""
    shard_size = math.ceil(len(topk_ref_indices) / num_shards)
    start = shard_id * shard_size
    end = min(start + shard_size, len(topk_ref_indices))
    my_indices = topk_ref_indices[start:end]

    print("=" * 60)
    print(f"Stage: Group Features — shard {shard_id}/{num_shards}, {len(my_indices)} top3 refs")
    print("=" * 60)

    gf_dir = os.path.join(OUTPUT_HVM_DIR, "group_features")
    os.makedirs(gf_dir, exist_ok=True)
    source_gf_dir = os.path.join(SOURCE_HVM_DIR, "group_features")

    # Skip done
    todo = []
    copied = 0
    for idx in my_indices:
        subdir = os.path.join(gf_dir, f"{idx // 1000:03d}")
        out_path = os.path.join(subdir, f"{idx:06d}.pt")
        if os.path.exists(out_path):
            continue
        src_path = os.path.join(source_gf_dir, f"{idx // 1000:03d}", f"{idx:06d}.pt")
        if os.path.exists(src_path):
            link_or_copy_file(src_path, out_path)
            copied += 1
        else:
            todo.append(idx)
    print(f"  Already done/copied: {len(my_indices) - len(todo)} (copied={copied}), todo: {len(todo)}")
    if not todo:
        print("  Nothing to do!")
        return

    # Load groups
    groups_path = os.path.join(OUTPUT_HVM_DIR, "groups_train_ref.jsonl")
    groups_data = {}
    with open(groups_path) as f:
        for line in f:
            g = json.loads(line)
            groups_data[g["idx"]] = g

    idx_set = set(todo)
    meta = load_metadata_for_indices(idx_set)
    tables = load_parquets_for_meta(meta)
    print(f"  Loaded {len(meta)} metadata, {len(tables)} parquets, {len(groups_data)} groups")

    # Load vision encoder
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
    pbar = tqdm(total=len(todo), desc=f"GroupFeats shard {shard_id}")

    for idx in todo:
        if idx not in meta or idx not in groups_data:
            pbar.update(1)
            continue
        m = meta[idx]
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

            subdir = os.path.join(gf_dir, f"{idx // 1000:03d}")
            os.makedirs(subdir, exist_ok=True)
            torch.save(group_feats, os.path.join(subdir, f"{idx:06d}.pt"))
            processed += 1

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"\n  Error idx={idx}: {e}")
                import traceback; traceback.print_exc()

        pbar.update(1)

    pbar.close()
    print(f"\nDone! Processed: {processed}, Errors: {errors}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True,
                        choices=["groups", "features", "group_features", "all"])
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    ensure_output_scaffold()
    print("Loading train ref indices...")
    all_feature_indices, topk_refs = load_train_ref_indices()
    print(f"  All feature indices (refs + train): {len(all_feature_indices)}")
    print(f"  Top-3 unique refs (for part bank):  {len(topk_refs)}")
    print()

    if args.stage == "all":
        run_groups(all_feature_indices, topk_refs)
        run_features(all_feature_indices, args.shard_id, args.num_shards, args.batch_size)
        run_group_features(topk_refs, args.shard_id, args.num_shards, args.batch_size)
    elif args.stage == "groups":
        run_groups(all_feature_indices, topk_refs)
    elif args.stage == "features":
        run_features(all_feature_indices, args.shard_id, args.num_shards, args.batch_size)
    elif args.stage == "group_features":
        run_group_features(topk_refs, args.shard_id, args.num_shards, args.batch_size)


if __name__ == "__main__":
    main()
