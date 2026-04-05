#!/usr/bin/env python3
"""
Text2SVG Benchmark Metrics (no GT required)
============================================
Computes metrics for text-to-SVG generation benchmarks where no ground-truth
images exist.

Supported metrics:
  - CLIP-T:     CLIP text-image cosine similarity (higher better)
  - Aesthetic:  LAION aesthetic score (higher better)
  - HPS:        Human Preference Score v2 (higher better)
  - FID:        Frechet Inception Distance against a reference image set (lower better)

Usage:
  python metrics/compute_text2svg_metrics.py \
    --result_dir /path/to/inference_results \
    --tag base \
    --ref_dir /path/to/reference_images \
    --output_dir metrics/metrics_results
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from tqdm import tqdm

EVAL_IMAGE_SIZE = 224
DEFAULT_CLIP_PATH = "/mnt/a100_1_data/wuqingman/models/openai/clip-vit-base-patch32"
DEFAULT_CLIP_L14_PATH = "/mnt/a100_1_data/wuqingman/models/openai/clip-vit-large-patch14"
DEFAULT_MLP_PATH = os.path.expanduser("~/.cache/aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth")
DEFAULT_HPS_CKPT = "/mnt/a100_1_data2/wuqingman/models/xswu/HPSv2/HPS_v2.pt"
DEFAULT_OUTPUT_DIR = "/mnt/a100_1_data2/wuqingman/omnisvg-train/metrics/metrics_results"


# ============================================================================
# Image loading
# ============================================================================

def load_image_rgb_white_bg(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(bg, rgba).convert("RGB")
    except Exception:
        return None


# ============================================================================
# Sample discovery (no GT required)
# ============================================================================

def discover_text2svg_samples(result_dir: str, tag: str) -> Dict[int, Dict]:
    result_path = Path(result_dir)
    samples = {}

    if tag:
        cand_pat = re.compile(rf"sample_(\d+)_{re.escape(tag)}(?:_c(\d+))?\.png$")
    else:
        cand_pat = re.compile(r"sample_(\d+)(?:_c(\d+))?\.png$")
    text_pat = re.compile(r"sample_(\d+)\.txt$")

    for f in sorted(result_path.iterdir()):
        name = f.name

        m = cand_pat.match(name)
        if m:
            idx = int(m.group(1))
            ci = int(m.group(2)) if m.group(2) is not None else 0
            samples.setdefault(idx, {"candidates": [], "text_file": None})
            samples[idx]["candidates"].append((ci, str(f)))
            continue

        m = text_pat.match(name)
        if m:
            idx = int(m.group(1))
            samples.setdefault(idx, {"candidates": [], "text_file": None})
            samples[idx]["text_file"] = str(f)

    for idx in samples:
        samples[idx]["candidates"].sort(key=lambda x: x[0])
        samples[idx]["candidates"] = [p for _, p in samples[idx]["candidates"]]

    return samples


def collect_ref_images(ref_dir: str, max_images: Optional[int] = None,
                       pattern: Optional[str] = None) -> List[str]:
    ref_path = Path(ref_dir)
    if pattern:
        paths = sorted([str(f) for f in ref_path.glob(pattern)])
    else:
        paths = sorted([
            str(f) for f in ref_path.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        ])
    if max_images is not None:
        paths = paths[:max_images]
    return paths


# ============================================================================
# Per-sample aggregation
# ============================================================================

def aggregate_candidates(values: List[float]) -> Dict[str, float]:
    result = {}
    for i, v in enumerate(values):
        result[f"c{i}"] = v
    arr = np.array(values)
    result["min"] = float(arr.min())
    result["max"] = float(arr.max())
    result["avg"] = float(arr.mean())
    if len(arr) >= 3:
        trimmed = np.sort(arr)[1:-1]
        result["trimmed"] = float(trimmed.mean())
    else:
        result["trimmed"] = float(arr.mean())
    return result


def compute_dataset_summary(per_sample: List[Dict], metric_name: str) -> Dict:
    strategies = ["min", "max", "avg", "trimmed"]
    total = len(per_sample)
    failed = sum(1 for r in per_sample if r.get("failed", False))
    success = total - failed

    summary = {
        "metric": metric_name,
        "total_samples": total,
        "success_samples": success,
        "failed_samples": failed,
        "success_rate": round(success / total * 100, 1) if total > 0 else 0.0,
    }

    for s in strategies:
        vals = [r[s] for r in per_sample if s in r]
        if vals:
            arr = np.array(vals)
            summary[f"{s}_mean"] = round(float(arr.mean()), 6)
            summary[f"{s}_std"] = round(float(arr.std()), 6)

    return summary


def save_metric(per_sample, summary, metric_name, output_dir, exp_name):
    import pandas as pd
    out_path = Path(output_dir) / exp_name
    out_path.mkdir(parents=True, exist_ok=True)
    if per_sample:
        df = pd.DataFrame(per_sample)
        csv_path = out_path / f"{metric_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
    json_path = out_path / f"{metric_name}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")


def print_metric_summary(summary, metric_name, higher_better=True):
    arrow = "\u2191" if higher_better else "\u2193"
    W = 70
    print(f"\n{'=' * W}")
    print(f"  {metric_name.upper()} {arrow}   "
          f"success={summary.get('success_samples', '?')}/{summary.get('total_samples', '?')}")
    print(f"{'=' * W}")
    for key in ["max", "min", "avg", "trimmed"]:
        if f"{key}_mean" in summary:
            print(f"  {key + '_mean':<20} {summary[f'{key}_mean']:>10.4f}")
    print(f"{'=' * W}")


# ============================================================================
# CLIP-T
# ============================================================================

_CLIP_MODEL = None
_CLIP_PROC = None

def _get_clip(device, model_path):
    global _CLIP_MODEL, _CLIP_PROC
    if _CLIP_MODEL is None:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP_MODEL = CLIPModel.from_pretrained(model_path).to(device).eval()
        _CLIP_PROC = CLIPProcessor.from_pretrained(model_path)
    return _CLIP_MODEL, _CLIP_PROC


@torch.no_grad()
def compute_clip_t(gen_pil, text, device, model_path):
    model, proc = _get_clip(device, model_path)
    inputs = proc(text=[text], images=[gen_pil], return_tensors="pt",
                  padding=True, truncation=True, max_length=77).to(device)
    img_f = F.normalize(model.get_image_features(pixel_values=inputs["pixel_values"]), dim=-1)
    txt_f = F.normalize(model.get_text_features(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]), dim=-1)
    return (img_f[0] @ txt_f[0]).item()


# ============================================================================
# Aesthetic
# ============================================================================

class AestheticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16), nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


_CLIP_L14 = None
_CLIP_L14_PROC = None
_AES_MLP = None

def _get_aesthetic(device, clip_path, mlp_path):
    global _CLIP_L14, _CLIP_L14_PROC, _AES_MLP
    if _CLIP_L14 is None:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP_L14 = CLIPModel.from_pretrained(clip_path).to(device).eval()
        _CLIP_L14_PROC = CLIPProcessor.from_pretrained(clip_path)
        _AES_MLP = AestheticMLP()
        _AES_MLP.load_state_dict(torch.load(mlp_path, map_location="cpu", weights_only=True))
        _AES_MLP.to(device).eval()
    return _CLIP_L14, _CLIP_L14_PROC, _AES_MLP


@torch.no_grad()
def compute_aesthetic(gen_pil, device, clip_path, mlp_path):
    clip_model, proc, mlp = _get_aesthetic(device, clip_path, mlp_path)
    inputs = proc(images=[gen_pil], return_tensors="pt").to(device)
    feat = F.normalize(clip_model.get_image_features(**inputs), dim=-1)
    return mlp(feat).item()


# ============================================================================
# HPS v2
# ============================================================================

_HPS_MODEL = None
_HPS_PREP = None
_HPS_TOK = None

def _get_hps(device, ckpt_path):
    global _HPS_MODEL, _HPS_PREP, _HPS_TOK
    if _HPS_MODEL is None:
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        model, _, preprocess_val = create_model_and_transforms(
            'ViT-H-14', '', precision='amp', device=device, jit=False,
            force_quick_gelu=False, force_custom_text=False,
            force_patch_dropout=False, force_image_size=None,
            pretrained_image=False, image_mean=None, image_std=None,
            light_augmentation=True, aug_cfg={}, output_dict=True,
            with_score_predictor=False, with_region_predictor=False,
        )
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        _HPS_MODEL = model.to(device).eval()
        _HPS_PREP = preprocess_val
        _HPS_TOK = get_tokenizer('ViT-H-14')
    return _HPS_MODEL, _HPS_PREP, _HPS_TOK


@torch.no_grad()
def compute_hps(gen_pil, text, device, ckpt_path):
    model, preprocess, tokenizer = _get_hps(device, ckpt_path)
    image = preprocess(gen_pil).unsqueeze(0).to(device=device)
    text_tokens = tokenizer([text]).to(device=device)
    with torch.cuda.amp.autocast():
        outputs = model(image, text_tokens)
        score = (outputs["image_features"] @ outputs["text_features"].T).diagonal().cpu().item()
    return score


# ============================================================================
# FID (with external reference dir)
# ============================================================================

def load_image_tensor(path: str, size: int = EVAL_IMAGE_SIZE):
    try:
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(bg, rgba).convert("RGB")
            rgb = rgb.resize((size, size), Image.LANCZOS)
            arr = np.array(rgb, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)
    except Exception:
        return None


@torch.no_grad()
def extract_features(image_paths, model, device, batch_size=64):
    features = []
    valid_count = 0
    batch = []
    for path in tqdm(image_paths, desc="  extracting features", leave=False):
        tensor = load_image_tensor(path)
        if tensor is None:
            continue
        batch.append(tensor)
        valid_count += 1
        if len(batch) >= batch_size:
            feat = model(torch.stack(batch).to(device))[0].squeeze(-1).squeeze(-1)
            features.append(feat.cpu().numpy())
            batch = []
    if batch:
        feat = model(torch.stack(batch).to(device))[0].squeeze(-1).squeeze(-1)
        features.append(feat.cpu().numpy())
    if not features:
        return None, 0
    return np.concatenate(features, axis=0), valid_count


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def run_fid(samples, ref_paths, device, batch_size, output_dir, exp_name):
    from inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()
    print(f"[FID] InceptionV3 loaded on {device}")

    print(f"[FID] Extracting reference features ({len(ref_paths)} images)...")
    ref_feats, ref_count = extract_features(ref_paths, model, device, batch_size)
    if ref_feats is None or ref_count < 2:
        print("[FID] Error: not enough valid reference images")
        return
    mu_ref, sigma_ref = np.mean(ref_feats, axis=0), np.cov(ref_feats, rowvar=False)
    print(f"  Reference: {ref_count} images")

    cand_paths_by_ci = defaultdict(list)
    all_cand_paths = []
    for idx in sorted(samples.keys()):
        for ci, path in enumerate(samples[idx]["candidates"]):
            cand_paths_by_ci[ci].append(path)
            all_cand_paths.append(path)

    results = {}
    for ci in sorted(cand_paths_by_ci.keys()):
        paths = cand_paths_by_ci[ci]
        print(f"[FID] Extracting c{ci} features ({len(paths)} images)...")
        feats, cnt = extract_features(paths, model, device, batch_size)
        if feats is None or cnt < 2:
            continue
        mu_c, sigma_c = np.mean(feats, axis=0), np.cov(feats, rowvar=False)
        fid_val = calculate_fid(mu_ref, sigma_ref, mu_c, sigma_c)
        results[f"fid_c{ci}"] = round(fid_val, 6)
        print(f"  c{ci}: FID = {fid_val:.4f} ({cnt} images)")

    if len(all_cand_paths) >= 2:
        print(f"[FID] Extracting all candidates ({len(all_cand_paths)} images)...")
        feats, cnt = extract_features(all_cand_paths, model, device, batch_size)
        if feats is not None and cnt >= 2:
            mu_a, sigma_a = np.mean(feats, axis=0), np.cov(feats, rowvar=False)
            fid_all = calculate_fid(mu_ref, sigma_ref, mu_a, sigma_a)
            results["fid_all"] = round(fid_all, 6)
            print(f"  all: FID = {fid_all:.4f} ({cnt} images)")

    summary = {"metric": "fid", "higher_better": False,
               "num_ref_images": ref_count, **results}

    per_cand_fids = [v for k, v in results.items() if k.startswith("fid_c")]
    if per_cand_fids:
        arr = np.array(per_cand_fids)
        summary["min_mean"] = round(float(arr.min()), 6)
        summary["max_mean"] = round(float(arr.max()), 6)
        summary["avg_mean"] = round(float(arr.mean()), 6)
        if len(arr) >= 3:
            trimmed = np.sort(arr)[1:-1]
            summary["trimmed_mean"] = round(float(trimmed.mean()), 6)
        else:
            summary["trimmed_mean"] = summary["avg_mean"]

    W = 70
    print(f"\n{'=' * W}")
    print(f"  FID \u2193  (ref={ref_count} images)")
    print(f"{'=' * W}")
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k:<25} {v:>10.4f}")
    if per_cand_fids:
        print(f"  ---")
        print(f"  {'min_mean (best)':<25} {summary['min_mean']:>10.4f}")
        print(f"  {'max_mean (worst)':<25} {summary['max_mean']:>10.4f}")
        print(f"  {'avg_mean':<25} {summary['avg_mean']:>10.4f}")
        print(f"  {'trimmed_mean':<25} {summary['trimmed_mean']:>10.4f}")
    print(f"{'=' * W}")

    save_metric([], summary, "fid", output_dir, exp_name)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Text2SVG Benchmark Metrics (no GT)")
    p.add_argument("--result_dir", type=str, required=True,
                   help="Inference results directory")
    p.add_argument("--tag", type=str, default="",
                   help="Candidate filename tag (base / hvm / 留空表示无 tag，如 sample_0000_c0.png)")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--start_idx", type=int, default=None,
                   help="Only evaluate samples with index >= start_idx")
    p.add_argument("--end_idx", type=int, default=None,
                   help="Only evaluate samples with index < end_idx")

    p.add_argument("--ref_dir", type=str, default=None,
                   help="Reference image directory for FID (skip FID if not provided)")
    p.add_argument("--ref_pattern", type=str, default=None,
                   help="Glob pattern to filter ref images, e.g. '*_gt.png'")
    p.add_argument("--max_ref_images", type=int, default=None,
                   help="Max number of reference images for FID")
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--clip_model", type=str, default=DEFAULT_CLIP_PATH)
    p.add_argument("--clip_l14_model", type=str, default=DEFAULT_CLIP_L14_PATH)
    p.add_argument("--aesthetic_mlp", type=str, default=DEFAULT_MLP_PATH)
    p.add_argument("--hps_ckpt", type=str, default=DEFAULT_HPS_CKPT)

    p.add_argument("--skip_clip_t", action="store_true")
    p.add_argument("--skip_aesthetic", action="store_true")
    p.add_argument("--skip_hps", action="store_true")
    p.add_argument("--skip_fid", action="store_true")

    args = p.parse_args()
    exp_name = args.exp_name or Path(args.result_dir).name
    device = args.device

    samples = discover_text2svg_samples(args.result_dir, args.tag)
    all_indices = sorted(samples.keys())
    if args.start_idx is not None:
        all_indices = [i for i in all_indices if i >= args.start_idx]
    if args.end_idx is not None:
        all_indices = [i for i in all_indices if i < args.end_idx]
    if args.max_samples:
        all_indices = all_indices[:args.max_samples]

    if args.start_idx is not None and args.end_idx is not None:
        full_range = list(range(args.start_idx, args.end_idx))
        empty_entry = {"text_file": None, "candidates": []}
        for idx in full_range:
            if idx not in samples:
                samples[idx] = empty_entry
        all_indices = full_range

    print("=" * 70)
    print("  Text2SVG Benchmark Metrics")
    print("=" * 70)
    idx_range = ""
    if args.start_idx is not None or args.end_idx is not None:
        idx_range = f" (idx {args.start_idx or 0} ~ {args.end_idx or '∞'})"
    has_cands = sum(1 for i in all_indices if len(samples[i]["candidates"]) > 0)
    print(f"  Result dir:    {args.result_dir}")
    print(f"  Tag:           {args.tag}")
    print(f"  Samples:       {len(all_indices)}{idx_range} ({has_cands} have candidates, {len(all_indices) - has_cands} failed)")
    print(f"  Ref dir (FID): {args.ref_dir or 'N/A'}")
    print(f"  Output:        {args.output_dir}/{exp_name}/")
    print("=" * 70)

    # ---- CLIP-T ----
    if not args.skip_clip_t:
        print(f"\n[1/4] Computing CLIP-T...")
        _get_clip(device, args.clip_model)
        per_sample = []
        for idx in tqdm(all_indices, desc="CLIP-T"):
            info = samples[idx]
            text = ""
            if info["text_file"] and os.path.exists(info["text_file"]):
                text = Path(info["text_file"]).read_text(encoding="utf-8").strip()
            values = []
            if text:
                for cand_path in info["candidates"]:
                    gen_pil = load_image_rgb_white_bg(cand_path)
                    if gen_pil is None:
                        continue
                    values.append(compute_clip_t(gen_pil, text, device, args.clip_model))
            if values:
                per_sample.append({"sample_idx": idx, "failed": False,
                                   **aggregate_candidates(values)})
            else:
                per_sample.append({"sample_idx": idx, "failed": True,
                                   "min": 0.0, "max": 0.0, "avg": 0.0, "trimmed": 0.0})
        summary = compute_dataset_summary(per_sample, "clip_t")
        print_metric_summary(summary, "clip_t", higher_better=True)
        save_metric(per_sample, summary, "clip_t", args.output_dir, exp_name)

    # ---- Aesthetic ----
    if not args.skip_aesthetic:
        print(f"\n[2/4] Computing Aesthetic...")
        _get_aesthetic(device, args.clip_l14_model, args.aesthetic_mlp)
        per_sample = []
        for idx in tqdm(all_indices, desc="Aesthetic"):
            info = samples[idx]
            values = []
            for cand_path in info["candidates"]:
                gen_pil = load_image_rgb_white_bg(cand_path)
                if gen_pil is None:
                    continue
                values.append(compute_aesthetic(gen_pil, device,
                                               args.clip_l14_model, args.aesthetic_mlp))
            if values:
                per_sample.append({"sample_idx": idx, "failed": False,
                                   **aggregate_candidates(values)})
            else:
                per_sample.append({"sample_idx": idx, "failed": True,
                                   "min": 0.0, "max": 0.0, "avg": 0.0, "trimmed": 0.0})
        summary = compute_dataset_summary(per_sample, "aesthetic")
        print_metric_summary(summary, "aesthetic", higher_better=True)
        save_metric(per_sample, summary, "aesthetic", args.output_dir, exp_name)

    # ---- HPS ----
    if not args.skip_hps:
        print(f"\n[3/4] Computing HPS...")
        _get_hps(device, args.hps_ckpt)
        per_sample = []
        for idx in tqdm(all_indices, desc="HPS"):
            info = samples[idx]
            text = ""
            if info["text_file"] and os.path.exists(info["text_file"]):
                text = Path(info["text_file"]).read_text(encoding="utf-8").strip()
            values = []
            if text:
                for cand_path in info["candidates"]:
                    gen_pil = load_image_rgb_white_bg(cand_path)
                    if gen_pil is None:
                        continue
                    values.append(compute_hps(gen_pil, text, device, args.hps_ckpt))
            if values:
                per_sample.append({"sample_idx": idx, "failed": False,
                                   **aggregate_candidates(values)})
            else:
                per_sample.append({"sample_idx": idx, "failed": True,
                                   "min": 0.0, "max": 0.0, "avg": 0.0, "trimmed": 0.0})
        summary = compute_dataset_summary(per_sample, "hps")
        print_metric_summary(summary, "hps", higher_better=True)
        save_metric(per_sample, summary, "hps", args.output_dir, exp_name)

    # ---- FID ----
    if not args.skip_fid and args.ref_dir:
        print(f"\n[4/4] Computing FID...")
        ref_paths = collect_ref_images(args.ref_dir, args.max_ref_images,
                                       pattern=args.ref_pattern)
        print(f"[FID] Reference images: {len(ref_paths)}")
        if len(ref_paths) >= 2:
            fid_samples = {idx: samples[idx] for idx in all_indices}
            run_fid(fid_samples, ref_paths, device, args.batch_size,
                    args.output_dir, exp_name)
        else:
            print("[FID] Skipped: need at least 2 reference images")
    elif not args.skip_fid:
        print("\n[4/4] FID skipped (no --ref_dir provided)")

    print(f"\n{'=' * 70}")
    print(f"  All done! Results: {args.output_dir}/{exp_name}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
