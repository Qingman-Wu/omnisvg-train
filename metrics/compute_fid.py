#!/usr/bin/env python3
"""
计算 FID (Fréchet Inception Distance, ↓ better)

FID 是数据集级别的指标，不是 per-sample 的。
对于多 candidate 场景，分别计算 FID(GT, c0), FID(GT, c1), ... 以及 FID(GT, all_candidates)。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from tqdm import tqdm

from common import (
    add_common_args, discover_samples, get_valid_indices,
    EVAL_IMAGE_SIZE,
)
from inception import InceptionV3

METRIC_NAME = "fid"
HIGHER_BETTER = False

_MODEL = None


def get_model(device):
    global _MODEL
    if _MODEL is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        _MODEL = InceptionV3([block_idx]).to(device).eval()
        print(f"[FID] InceptionV3 loaded on {device}")
    return _MODEL


def load_image_tensor(path: str, size: int = EVAL_IMAGE_SIZE):
    """Load image as [C, H, W] float32 tensor in [0, 1] with white background."""
    try:
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(bg, rgba).convert("RGB")
            rgb = rgb.resize((size, size), Image.LANCZOS)
            arr = np.array(rgb, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]
    except Exception:
        return None


@torch.no_grad()
def extract_features(image_paths, model, device, batch_size=64):
    """Extract 2048-d InceptionV3 features for a list of image paths."""
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
            inp = torch.stack(batch).to(device)
            feat = model(inp)[0].squeeze(-1).squeeze(-1)  # [B, 2048]
            features.append(feat.cpu().numpy())
            batch = []

    if batch:
        inp = torch.stack(batch).to(device)
        feat = model(inp)[0].squeeze(-1).squeeze(-1)
        features.append(feat.cpu().numpy())

    if not features:
        return None, 0
    return np.concatenate(features, axis=0), valid_count


def compute_statistics(features):
    """Compute mean and covariance of features."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute the Fréchet Distance between two multivariate Gaussians.
    Stable implementation by Dougal J. Sutherland.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f"  [FID] Warning: singular product, adding {eps} to diagonal")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    return float(
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    )


def main():
    p = argparse.ArgumentParser(description="Compute FID (Fréchet Inception Distance)")
    add_common_args(p)
    p.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for InceptionV3 feature extraction")
    args = p.parse_args()

    exp_name = args.exp_name or Path(args.result_dir).name
    samples = discover_samples(args.result_dir, args.tag)
    valid_indices = get_valid_indices(samples, args.max_samples)

    print(f"[FID] total valid samples: {len(valid_indices)}")

    # Collect GT paths and per-candidate paths
    gt_paths = []
    cand_paths_by_idx = defaultdict(list)  # candidate_index → [path, ...]
    all_cand_paths = []

    for idx in valid_indices:
        info = samples[idx]
        gt_paths.append(info["gt_png"])
        for ci, cand_path in enumerate(info["candidates"]):
            cand_paths_by_idx[ci].append(cand_path)
            all_cand_paths.append(cand_path)

    num_candidate_indices = len(cand_paths_by_idx)
    print(f"[FID] GT images: {len(gt_paths)}, candidate indices: {num_candidate_indices}, "
          f"total candidates: {len(all_cand_paths)}")

    if len(gt_paths) < 2:
        print("[FID] Error: need at least 2 valid samples to compute FID")
        return

    model = get_model(args.device)

    # Extract GT features (only once)
    print("[FID] Extracting GT features...")
    gt_features, gt_count = extract_features(gt_paths, model, args.device, args.batch_size)
    if gt_features is None:
        print("[FID] Error: no valid GT images")
        return
    mu_gt, sigma_gt = compute_statistics(gt_features)
    print(f"  GT: {gt_count} images, feature shape {gt_features.shape}")

    results = {}

    # Per-candidate-index FID
    for ci in sorted(cand_paths_by_idx.keys()):
        paths = cand_paths_by_idx[ci]
        print(f"[FID] Extracting features for candidate c{ci} ({len(paths)} images)...")
        cand_features, cand_count = extract_features(paths, model, args.device, args.batch_size)
        if cand_features is None or cand_count < 2:
            print(f"  c{ci}: skipped (too few valid images)")
            continue
        mu_c, sigma_c = compute_statistics(cand_features)
        fid_val = calculate_fid(mu_gt, sigma_gt, mu_c, sigma_c)
        results[f"fid_c{ci}"] = round(fid_val, 6)
        print(f"  c{ci}: FID = {fid_val:.4f} ({cand_count} images)")

    # All candidates combined FID
    if len(all_cand_paths) >= 2:
        print(f"[FID] Extracting features for all candidates ({len(all_cand_paths)} images)...")
        all_features, all_count = extract_features(all_cand_paths, model, args.device, args.batch_size)
        if all_features is not None and all_count >= 2:
            mu_all, sigma_all = compute_statistics(all_features)
            fid_all = calculate_fid(mu_gt, sigma_gt, mu_all, sigma_all)
            results["fid_all"] = round(fid_all, 6)
            print(f"  all: FID = {fid_all:.4f} ({all_count} images)")

    # Summary
    summary = {
        "metric": METRIC_NAME,
        "higher_better": HIGHER_BETTER,
        "num_gt_images": gt_count,
        "num_candidate_indices": num_candidate_indices,
        "total_candidate_images": len(all_cand_paths),
        **results,
    }

    # Aggregate across candidate indices (same keys as other metrics)
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

    # Print
    W = 75
    print(f"\n{'=' * W}")
    print(f"  FID ↓   (lower is better)")
    print(f"{'=' * W}")
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k:<25} {v:>10.4f}")
    if per_cand_fids:
        print(f"  {'---'}")
        print(f"  {'min_mean (best)':<25} {summary['min_mean']:>10.4f}")
        print(f"  {'max_mean (worst)':<25} {summary['max_mean']:>10.4f}")
        print(f"  {'avg_mean':<25} {summary['avg_mean']:>10.4f}")
        print(f"  {'trimmed_mean':<25} {summary['trimmed_mean']:>10.4f}")
    print(f"{'=' * W}")

    # Save
    out_path = Path(args.output_dir) / exp_name
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / f"{METRIC_NAME}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
