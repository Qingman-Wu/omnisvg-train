#!/usr/bin/env python3
"""
SVG Generation Evaluation Script
=================================
计算生成 SVG 与 Ground Truth 之间的质量指标。
使用本地已有的预训练模型，无需联网下载。

支持指标:
  - SSIM      (Structural Similarity Index, ↑ better)
  - MSE       (Mean Squared Error, ↓ better)
  - CLIP-I    (CLIP image-image cosine similarity vs GT, ↑ better)
  - CLIP-T    (CLIP text-image cosine similarity, ↑ better)
  - DINO-I    (DINOv2 image-image cosine similarity vs GT, ↑ better)
  - CLIP-FID  (FID computed with CLIP features, ↓ better)
  - DINO-FID  (FID computed with DINOv2 features, ↓ better)

多候选策略:
  - first:   只取 c0
  - best:    oracle best (按 DINO-I 选最好的候选)
  - average: 所有候选平均

输出:
  - 终端汇总表格
  - metrics_summary.json  (全局指标)
  - metrics_per_sample.csv (每个样本的详细指标)

用法:
  # 单实验评估
  conda run -n omnisvg python inference/evaluate.py \
      --result_dir inference_results/s2_gme_last4_adaptive_gate_step4000_test \
      --tag hvm \
      --device cuda:0

  # 多实验对比
  conda run -n omnisvg python inference/evaluate.py \
      --result_dir inference_results/exp1 inference_results/exp2 \
      --tag hvm base \
      --device cuda:0

  # 调试：只跑前 10 个样本
  conda run -n omnisvg python inference/evaluate.py \
      --result_dir inference_results/exp1 \
      --tag hvm --max_samples 10

本地模型路径 (通过 CLI 参数可覆盖):
  --clip_model  /mnt/a100_1_data/wuqingman/models/openai/clip-vit-base-patch32
  --dino_model  /mnt/a100_1_data/wuqingman/models/facebook/dinov2-large
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Workaround: transformers >= 4.51 blocks torch.load on torch < 2.6
try:
    import transformers.modeling_utils
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

EVAL_IMAGE_SIZE = 224

DEFAULT_CLIP_PATH = "/mnt/a100_1_data/wuqingman/models/openai/clip-vit-base-patch32"
DEFAULT_DINO_PATH = "/mnt/a100_1_data/wuqingman/models/facebook/dinov2-large"


# ============================================================================
# Model singletons (lazy-loaded)
# ============================================================================

_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_DINO_MODEL = None
_DINO_PROCESSOR = None


def get_clip(device: str, model_path: str = DEFAULT_CLIP_PATH):
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP_MODEL = CLIPModel.from_pretrained(model_path).to(device).eval()
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_path)
        print(f"[eval] CLIP loaded from {model_path}")
    return _CLIP_MODEL, _CLIP_PROCESSOR


def get_dino(device: str, model_path: str = DEFAULT_DINO_PATH):
    global _DINO_MODEL, _DINO_PROCESSOR
    if _DINO_MODEL is None:
        from transformers import AutoModel, AutoImageProcessor
        _DINO_MODEL = AutoModel.from_pretrained(model_path).to(device).eval()
        _DINO_PROCESSOR = AutoImageProcessor.from_pretrained(model_path)
        print(f"[eval] DINOv2 loaded from {model_path}")
    return _DINO_MODEL, _DINO_PROCESSOR


# ============================================================================
# Image loading
# ============================================================================

def load_image_as_array(path: str, size: int = EVAL_IMAGE_SIZE) -> Optional[np.ndarray]:
    """Load image as [H, W, 3] float32 array in [0, 1]."""
    try:
        img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        return None


def load_image_pil(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ============================================================================
# Per-image metric functions
# ============================================================================

def compute_ssim(gen: np.ndarray, gt: np.ndarray) -> float:
    """SSIM between two [H,W,3] arrays in [0,1]. Higher = more similar."""
    from skimage.metrics import structural_similarity
    return structural_similarity(gen, gt, channel_axis=2, data_range=1.0)


def compute_mse(gen: np.ndarray, gt: np.ndarray) -> float:
    """MSE between two [H,W,3] arrays. Lower = more similar."""
    return float(np.mean((gen - gt) ** 2))


@torch.no_grad()
def compute_clip_image_similarity(
    gen_pil: Image.Image, gt_pil: Image.Image, device: str, model_path: str,
) -> float:
    """CLIP cosine similarity between two images."""
    model, processor = get_clip(device, model_path)
    inputs = processor(images=[gen_pil, gt_pil], return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    feats = F.normalize(feats, dim=-1)
    return (feats[0] @ feats[1]).item()


@torch.no_grad()
def compute_clip_text_similarity(
    gen_pil: Image.Image, text: str, device: str, model_path: str,
) -> float:
    """CLIP cosine similarity between image and text."""
    model, processor = get_clip(device, model_path)
    inputs = processor(
        text=[text], images=[gen_pil], return_tensors="pt",
        padding=True, truncation=True, max_length=77,
    ).to(device)
    img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
    txt_feat = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    img_feat = F.normalize(img_feat, dim=-1)
    txt_feat = F.normalize(txt_feat, dim=-1)
    return (img_feat[0] @ txt_feat[0]).item()


@torch.no_grad()
def compute_dino_image_similarity(
    gen_pil: Image.Image, gt_pil: Image.Image, device: str, model_path: str,
) -> float:
    """DINOv2 cosine similarity between two images (CLS token)."""
    model, processor = get_dino(device, model_path)
    inputs = processor(images=[gen_pil, gt_pil], return_tensors="pt").to(device)
    outputs = model(**inputs)
    feats = outputs.last_hidden_state[:, 0]  # CLS token
    feats = F.normalize(feats, dim=-1)
    return (feats[0] @ feats[1]).item()


# ============================================================================
# Batch feature extraction for FID
# ============================================================================

@torch.no_grad()
def extract_clip_features(images: List[Image.Image], device: str, model_path: str, batch_size: int = 64) -> np.ndarray:
    model, processor = get_clip(device, model_path)
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)
        all_feats.append(feats.cpu().float().numpy())
    return np.concatenate(all_feats, axis=0)


@torch.no_grad()
def extract_dino_features(images: List[Image.Image], device: str, model_path: str, batch_size: int = 32) -> np.ndarray:
    model, processor = get_dino(device, model_path)
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        outputs = model(**inputs)
        feats = outputs.last_hidden_state[:, 0]
        all_feats.append(feats.cpu().float().numpy())
    return np.concatenate(all_feats, axis=0)


def compute_fid(feats_gen: np.ndarray, feats_gt: np.ndarray) -> float:
    """FID from two sets of features."""
    from scipy.linalg import sqrtm

    mu_gen = feats_gen.mean(axis=0)
    mu_gt = feats_gt.mean(axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    sigma_gt = np.cov(feats_gt, rowvar=False)

    diff = mu_gen - mu_gt
    covmean, _ = sqrtm(sigma_gen @ sigma_gt, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_gen + sigma_gt - 2 * covmean))


# ============================================================================
# Sample discovery
# ============================================================================

def discover_samples(result_dir: Path, tag: str) -> Dict[int, Dict]:
    """
    Scan result_dir for GT PNGs, candidate PNGs (by tag), and text files.
    Returns: {sample_idx: {"gt_png", "candidates": [paths], "text_file"}}
    """
    samples = {}

    gt_pat = re.compile(r"sample_(\d+)_gt\.png$")
    cand_pat = re.compile(rf"sample_(\d+)_{re.escape(tag)}(?:_c(\d+))?\.png$")
    text_pat = re.compile(r"sample_(\d+)\.txt$")

    for f in sorted(result_dir.iterdir()):
        name = f.name

        m = gt_pat.match(name)
        if m:
            idx = int(m.group(1))
            samples.setdefault(idx, {"gt_png": None, "candidates": [], "text_file": None})
            samples[idx]["gt_png"] = str(f)
            continue

        m = cand_pat.match(name)
        if m:
            idx = int(m.group(1))
            ci = int(m.group(2)) if m.group(2) is not None else 0
            samples.setdefault(idx, {"gt_png": None, "candidates": [], "text_file": None})
            samples[idx]["candidates"].append((ci, str(f)))
            continue

        m = text_pat.match(name)
        if m:
            idx = int(m.group(1))
            samples.setdefault(idx, {"gt_png": None, "candidates": [], "text_file": None})
            samples[idx]["text_file"] = str(f)

    for idx in samples:
        samples[idx]["candidates"].sort(key=lambda x: x[0])
        samples[idx]["candidates"] = [p for _, p in samples[idx]["candidates"]]

    return samples


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate_single_dir(
    result_dir: str,
    tag: str,
    device: str,
    clip_model: str,
    dino_model: str,
    skip_fid: bool = False,
    skip_clip: bool = False,
    skip_dino: bool = False,
    max_samples: Optional[int] = None,
) -> Dict:
    result_path = Path(result_dir)
    if not result_path.exists():
        print(f"[ERROR] Not found: {result_dir}")
        return {}

    print(f"\n{'='*70}")
    print(f"Evaluating: {result_dir}")
    print(f"  Tag: {tag}")
    print(f"{'='*70}")

    samples = discover_samples(result_path, tag)
    if not samples:
        print("[ERROR] No samples found!")
        return {}

    valid_indices = sorted([
        idx for idx, info in samples.items()
        if info["gt_png"] is not None and len(info["candidates"]) > 0
    ])
    if max_samples is not None:
        valid_indices = valid_indices[:max_samples]

    total_samples = len(samples)
    total_with_gt = sum(1 for info in samples.values() if info["gt_png"] is not None)
    total_with_cand = sum(1 for info in samples.values() if len(info["candidates"]) > 0)
    valid_count = len(valid_indices)

    print(f"  Total discovered:  {total_samples}")
    print(f"  With GT:           {total_with_gt}")
    print(f"  With candidates:   {total_with_cand}")
    print(f"  Valid (GT+cand):   {valid_count}")
    print(f"  Success rate:      {total_with_cand / max(total_samples, 1) * 100:.1f}%")

    from tqdm import tqdm

    per_sample = []
    fid_gen_images = []
    fid_gt_images = []

    for idx in tqdm(valid_indices, desc="Computing metrics"):
        info = samples[idx]
        gt_arr = load_image_as_array(info["gt_png"])
        gt_pil = load_image_pil(info["gt_png"])
        if gt_arr is None or gt_pil is None:
            continue

        text = ""
        if info["text_file"] and os.path.exists(info["text_file"]):
            text = Path(info["text_file"]).read_text(encoding="utf-8").strip()

        cand_metrics = []
        cand_pils = []
        for cand_path in info["candidates"]:
            gen_arr = load_image_as_array(cand_path)
            gen_pil = load_image_pil(cand_path)
            if gen_arr is None or gen_pil is None:
                continue

            m = {
                "ssim": compute_ssim(gen_arr, gt_arr),
                "mse": compute_mse(gen_arr, gt_arr),
            }
            if not skip_clip:
                m["clip_i"] = compute_clip_image_similarity(gen_pil, gt_pil, device, clip_model)
                if text:
                    m["clip_t"] = compute_clip_text_similarity(gen_pil, text, device, clip_model)
            if not skip_dino:
                m["dino_i"] = compute_dino_image_similarity(gen_pil, gt_pil, device, dino_model)

            cand_metrics.append(m)
            cand_pils.append(gen_pil)

        if not cand_metrics:
            continue

        num_cands = len(cand_metrics)
        metric_keys = list(cand_metrics[0].keys())

        # Oracle best: by DINO-I (if available), else SSIM
        rank_key = "dino_i" if "dino_i" in cand_metrics[0] else "ssim"
        best_idx = max(range(num_cands), key=lambda i: cand_metrics[i][rank_key])

        first_m = cand_metrics[0]
        best_m = cand_metrics[best_idx]
        avg_m = {k: np.mean([cm[k] for cm in cand_metrics if k in cm]) for k in metric_keys}

        row = {"sample_idx": idx, "text": text[:100], "num_candidates": num_cands, "best_idx": best_idx}
        for k in metric_keys:
            row[f"first_{k}"] = first_m.get(k, float("nan"))
            row[f"best_{k}"] = best_m.get(k, float("nan"))
            row[f"avg_{k}"] = avg_m.get(k, float("nan"))

        per_sample.append(row)

        if not skip_fid:
            fid_gen_images.append(cand_pils[best_idx])
            fid_gt_images.append(gt_pil)

    if not per_sample:
        print("[ERROR] No valid samples!")
        return {}

    # ---- Aggregate ----
    metric_keys = [k.replace("first_", "") for k in per_sample[0] if k.startswith("first_")]
    strategies = ["first", "best", "avg"]

    summary = {
        "result_dir": str(result_dir),
        "tag": tag,
        "total_samples": total_samples,
        "valid_samples": len(per_sample),
        "success_rate": round(total_with_cand / max(total_samples, 1) * 100, 1),
        "avg_num_candidates": round(float(np.mean([r["num_candidates"] for r in per_sample])), 2),
    }

    for s in strategies:
        for k in metric_keys:
            col = f"{s}_{k}"
            vals = [r[col] for r in per_sample if not np.isnan(r.get(col, float("nan")))]
            if vals:
                summary[f"{col}_mean"] = round(float(np.mean(vals)), 4)
                summary[f"{col}_std"] = round(float(np.std(vals)), 4)

    # ---- FID ----
    if not skip_fid and len(fid_gen_images) >= 2:
        print(f"\nComputing FID ({len(fid_gen_images)} images)...")
        if not skip_clip:
            try:
                feats_gen = extract_clip_features(fid_gen_images, device, clip_model)
                feats_gt = extract_clip_features(fid_gt_images, device, clip_model)
                summary["clip_fid"] = round(compute_fid(feats_gen, feats_gt), 2)
                print(f"  CLIP-FID: {summary['clip_fid']:.2f}")
            except Exception as e:
                print(f"  CLIP-FID failed: {e}")
        if not skip_dino:
            try:
                feats_gen = extract_dino_features(fid_gen_images, device, dino_model)
                feats_gt = extract_dino_features(fid_gt_images, device, dino_model)
                summary["dino_fid"] = round(compute_fid(feats_gen, feats_gt), 2)
                print(f"  DINO-FID: {summary['dino_fid']:.2f}")
            except Exception as e:
                print(f"  DINO-FID failed: {e}")

    # ---- Print table ----
    # Metric display config: (name, direction_arrow)
    metric_display = {
        "ssim": "SSIM↑", "mse": "MSE↓",
        "clip_i": "CLIP-I↑", "clip_t": "CLIP-T↑", "dino_i": "DINO-I↑",
    }

    print(f"\n{'='*70}")
    print(f"Results: {Path(result_dir).name}")
    print(f"  Samples: {summary['valid_samples']}/{summary['total_samples']}  "
          f"Success: {summary['success_rate']}%  "
          f"Avg cands: {summary['avg_num_candidates']}")
    print(f"{'='*70}")

    header = f"{'Metric':<12}"
    for s in strategies:
        header += f"  {s:>16}"
    print(header)
    print("-" * (12 + 18 * len(strategies)))

    for k in metric_keys:
        display = metric_display.get(k, k)
        row_str = f"{display:<12}"
        for s in strategies:
            mean_key = f"{s}_{k}_mean"
            std_key = f"{s}_{k}_std"
            if mean_key in summary:
                row_str += f"  {summary[mean_key]:>8.4f}±{summary[std_key]:<6.4f}"
            else:
                row_str += f"  {'N/A':>16}"
        print(row_str)

    # FID rows
    for fid_name, fid_display in [("clip_fid", "CLIP-FID↓"), ("dino_fid", "DINO-FID↓")]:
        if fid_name in summary:
            print(f"{fid_display:<12}  {summary[fid_name]:>16.2f}")

    print("=" * (12 + 18 * len(strategies)))

    # ---- Save ----
    out_json = result_path / "metrics_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_json}")

    try:
        import pandas as pd
        df = pd.DataFrame(per_sample)
        out_csv = result_path / "metrics_per_sample.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
    except ImportError:
        pass

    return summary


# ============================================================================
# Multi-experiment comparison
# ============================================================================

def print_comparison(summaries: List[Dict]):
    if len(summaries) <= 1:
        return

    print(f"\n{'='*100}")
    print("COMPARISON (best oracle)")
    print(f"{'='*100}")

    metrics = ["ssim", "mse", "clip_i", "clip_t", "dino_i"]
    labels = ["SSIM↑", "MSE↓", "CLIP-I↑", "CLIP-T↑", "DINO-I↑", "CLIP-FID↓", "DINO-FID↓"]

    header = f"{'Experiment':<45}"
    for l in labels:
        header += f" {l:>10}"
    print(header)
    print("-" * len(header))

    for s in summaries:
        name = Path(s.get("result_dir", "?")).name[:43]
        row = f"{name:<45}"
        for m in metrics:
            key = f"best_{m}_mean"
            row += f" {s[key]:>10.4f}" if key in s else f" {'N/A':>10}"
        for fid in ["clip_fid", "dino_fid"]:
            row += f" {s[fid]:>10.2f}" if fid in s else f" {'N/A':>10}"
        print(row)

    print("=" * 100)


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="SVG Generation Evaluation")
    p.add_argument("--result_dir", type=str, nargs="+", required=True,
                   help="推理结果目录 (可多个，空格分隔)")
    p.add_argument("--tag", type=str, nargs="+", default=["hvm"],
                   help="候选文件中的 tag (如 hvm / base)，需与 result_dir 数量匹配")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--clip_model", type=str, default=DEFAULT_CLIP_PATH,
                   help="CLIP 模型本地路径")
    p.add_argument("--dino_model", type=str, default=DEFAULT_DINO_PATH,
                   help="DINOv2 模型本地路径")
    p.add_argument("--skip_fid", action="store_true")
    p.add_argument("--skip_clip", action="store_true")
    p.add_argument("--skip_dino", action="store_true")
    p.add_argument("--max_samples", type=int, default=None, help="调试：最多评估几个样本")
    args = p.parse_args()

    tags = args.tag
    if len(tags) == 1 and len(args.result_dir) > 1:
        tags = tags * len(args.result_dir)
    assert len(tags) == len(args.result_dir), \
        f"--tag count ({len(tags)}) != --result_dir count ({len(args.result_dir)})"

    summaries = []
    for rdir, t in zip(args.result_dir, tags):
        s = evaluate_single_dir(
            rdir, t, args.device,
            clip_model=args.clip_model,
            dino_model=args.dino_model,
            skip_fid=args.skip_fid,
            skip_clip=args.skip_clip,
            skip_dino=args.skip_dino,
            max_samples=args.max_samples,
        )
        if s:
            summaries.append(s)

    if len(summaries) > 1:
        print_comparison(summaries)


if __name__ == "__main__":
    main()
