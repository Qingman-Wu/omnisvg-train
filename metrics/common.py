"""
Metrics Common Utilities
========================
共享的图片加载、样本发现、聚合统计、结果保存工具。
所有 compute_*.py 都依赖此模块。
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
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
DEFAULT_OUTPUT_DIR = "/mnt/a100_1_data2/wuqingman/omnisvg-train/metrics/metrics_results"


# ============================================================================
# Image loading
# ============================================================================

def _load_image_rgb_white_bg(path: str) -> Optional[Image.Image]:
    """Load an image and composite transparency onto a white background."""
    try:
        with Image.open(path) as img:
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
    except Exception:
        return None


def load_image_as_array(path: str, size: int = EVAL_IMAGE_SIZE) -> Optional[np.ndarray]:
    """Load image as [H, W, 3] float32 in [0, 1]."""
    try:
        img = _load_image_rgb_white_bg(path)
        if img is None:
            return None
        img = img.resize((size, size), Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        return None


def load_image_pil(path: str) -> Optional[Image.Image]:
    return _load_image_rgb_white_bg(path)


# ============================================================================
# Sample discovery
# ============================================================================

def discover_samples(result_dir: str, tag: str) -> Dict[int, Dict]:
    """
    扫描 result_dir，返回每个样本的 GT/候选/文本路径。

    Returns:
        {sample_idx: {
            "gt_png": str or None,
            "candidates": [path_c0, path_c1, ...],
            "text_file": str or None,
        }}
    """
    result_path = Path(result_dir)
    samples = {}

    gt_pat = re.compile(r"sample_(\d+)_gt\.png$")
    cand_pat = re.compile(rf"sample_(\d+)_{re.escape(tag)}(?:_c(\d+))?\.png$")
    text_pat = re.compile(r"sample_(\d+)\.txt$")

    for f in sorted(result_path.iterdir()):
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


def get_valid_indices(samples: Dict[int, Dict], max_samples: Optional[int] = None) -> List[int]:
    """筛选同时有 GT 和至少 1 个候选的样本索引。"""
    valid = sorted([
        idx for idx, info in samples.items()
        if info["gt_png"] is not None and len(info["candidates"]) > 0
    ])
    if max_samples is not None:
        valid = valid[:max_samples]
    return valid


def get_all_gt_indices(samples: Dict[int, Dict], max_samples: Optional[int] = None) -> List[int]:
    """返回所有有 GT 的样本索引（包括生成失败的）。"""
    all_gt = sorted([idx for idx, info in samples.items() if info["gt_png"] is not None])
    if max_samples is not None:
        all_gt = all_gt[:max_samples]
    return all_gt


def make_failed_row(idx: int, num_candidates_expected: int = 0) -> Dict:
    """为生成失败的样本创建全 0 行，参与数据集整体统计。"""
    row = {"sample_idx": idx, "failed": True}
    row["min"] = 0.0
    row["max"] = 0.0
    row["avg"] = 0.0
    row["trimmed"] = 0.0
    for i in range(num_candidates_expected):
        row[f"c{i}"] = 0.0
    return row


# ============================================================================
# Per-sample aggregation (across candidates)
# ============================================================================

def aggregate_candidates(values: List[float]) -> Dict[str, float]:
    """
    对一个样本的多个 candidate 指标值做聚合。

    Returns:
        {"min": ..., "max": ..., "avg": ..., "trimmed": ...,
         "c0": ..., "c1": ..., ...}
    """
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


# ============================================================================
# Dataset-level aggregation
# ============================================================================

def compute_dataset_summary(
    per_sample: List[Dict],
    metric_name: str,
) -> Dict:
    """
    汇总整个数据集的指标。

    失败样本 (failed=True) 得分为 0，参与统计。
    对于每种聚合策略 (min/max/avg/trimmed)，收集每个样本在该策略下的值，
    再对整个数据集取 mean 和 std。

    Args:
        per_sample: list of dicts，包含有效样本行和失败样本行（全 0）
        metric_name: 指标名 (用于输出)

    Returns:
        summary dict
    """
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
            summary[f"{s}_mean"]   = round(float(arr.mean()), 6)
            summary[f"{s}_std"]    = round(float(arr.std()),  6)
            summary[f"{s}_best"]   = round(float(arr.max()),  6)
            summary[f"{s}_worst"]  = round(float(arr.min()),  6)

    return summary


# ============================================================================
# Result saving & printing
# ============================================================================

def save_results(
    per_sample: List[Dict],
    summary: Dict,
    metric_name: str,
    output_dir: str,
    exp_name: str,
):
    """保存 CSV (per-sample) 和 JSON (summary) 到 output_dir/exp_name/。"""
    out_path = Path(output_dir) / exp_name
    out_path.mkdir(parents=True, exist_ok=True)

    # CSV
    import pandas as pd
    df = pd.DataFrame(per_sample)
    csv_path = out_path / f"{metric_name}.csv"
    df.to_csv(csv_path, index=False)

    # JSON
    json_path = out_path / f"{metric_name}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {csv_path}")
    print(f"  Saved: {json_path}")


def print_summary(summary: Dict, metric_name: str, higher_better: bool = True):
    """打印单指标的数据集汇总。"""
    arrow = "↑" if higher_better else "↓"
    total = summary.get("total_samples", summary.get("num_samples", "?"))
    success = summary.get("success_samples", total)
    failed = summary.get("failed_samples", 0)
    rate = summary.get("success_rate", 100.0)

    W = 75
    print(f"\n{'='*W}")
    print(f"  {metric_name.upper()} {arrow}   success={success}/{total} ({rate}%)   failed={failed} (scored 0)")
    print(f"{'='*W}")

    # strategy → (label, description)
    strategy_info = {
        "max":     ("best-of-N",  "per sample: pick best  candidate, then aggregate over dataset"),
        "min":     ("worst-of-N", "per sample: pick worst candidate, then aggregate over dataset"),
        "avg":     ("avg-of-N",   "per sample: mean of all candidates, then aggregate over dataset"),
        "trimmed": ("trimmed-N",  "per sample: drop min+max, mean rest, then aggregate over dataset"),
    }

    # header
    print(f"  {'Strategy':<13}  {'mean':>8}  {'best':>8}  {'worst':>8}  {'std':>8}")
    print(f"  {'-'*(W-4)}")

    for key, (label, desc) in strategy_info.items():
        if f"{key}_mean" not in summary:
            continue
        mean  = summary[f"{key}_mean"]
        best  = summary[f"{key}_best"]
        worst = summary[f"{key}_worst"]
        std   = summary[f"{key}_std"]
        print(f"  {label:<13}  {mean:>8.4f}  {best:>8.4f}  {worst:>8.4f}  {std:>8.4f}")

    print(f"{'='*W}")


# ============================================================================
# Common argparse
# ============================================================================

def load_existing_results(output_dir: str, exp_name: str, metric_name: str):
    """
    加载已有的 per-sample CSV 结果，用于 resume 断点续算。

    Returns:
        (per_sample list, done_indices set)
    """
    import pandas as pd
    csv_path = Path(output_dir) / exp_name / f"{metric_name}.csv"
    if not csv_path.exists():
        return [], set()
    try:
        df = pd.read_csv(csv_path)
        per_sample = df.to_dict("records")
        done_indices = {int(r["sample_idx"]) for r in per_sample}
        print(f"  [resume] Loaded {len(done_indices)} existing results from {csv_path}")
        return per_sample, done_indices
    except Exception as e:
        print(f"  [resume] Warning: could not load {csv_path}: {e}")
        return [], set()


def add_common_args(parser):
    """添加所有 compute_*.py 共享的 CLI 参数。"""
    parser.add_argument("--result_dir", type=str, required=True,
                        help="推理结果目录")
    parser.add_argument("--tag", type=str, default="hvm",
                        help="候选文件名中的 tag (如 hvm / base)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="指标结果输出根目录")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="实验名 (默认取 result_dir 的文件夹名)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最多评估几个样本 (调试用)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="GPU 设备")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="断点续算：跳过已计算的样本，追加新结果")
    return parser
