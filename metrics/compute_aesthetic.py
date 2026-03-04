#!/usr/bin/env python3
"""计算 Aesthetic Score (LAION improved-aesthetic-predictor, ↑ better)

使用 CLIP ViT-L/14 特征 + 训练好的 MLP 预测美学评分 (1-10 scale)。
注意: 这是纯生成质量指标，不需要 GT 图片。
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from common import (
    add_common_args, discover_samples, get_valid_indices, get_all_gt_indices,
    load_image_pil, aggregate_candidates, make_failed_row,
    compute_dataset_summary, save_results, print_summary, load_existing_results,
)

METRIC_NAME = "aesthetic"
HIGHER_BETTER = True

DEFAULT_CLIP_L14_PATH = "/mnt/a100_1_data/wuqingman/models/openai/clip-vit-large-patch14"
DEFAULT_MLP_PATH = os.path.expanduser("~/.cache/aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth")

_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_AESTHETIC_MLP = None


class AestheticMLP(nn.Module):
    """Aesthetic predictor MLP (input: 768-d CLIP ViT-L/14 embedding)."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def get_models(device, clip_path, mlp_path):
    global _CLIP_MODEL, _CLIP_PROCESSOR, _AESTHETIC_MLP
    if _CLIP_MODEL is None:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP_MODEL = CLIPModel.from_pretrained(clip_path).to(device).eval()
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(clip_path)
        print(f"[Aesthetic] CLIP ViT-L/14 loaded from {clip_path}")

        _AESTHETIC_MLP = AestheticMLP()
        state = torch.load(mlp_path, map_location="cpu", weights_only=True)
        _AESTHETIC_MLP.load_state_dict(state)
        _AESTHETIC_MLP.to(device).eval()
        print(f"[Aesthetic] MLP loaded from {mlp_path}")

    return _CLIP_MODEL, _CLIP_PROCESSOR, _AESTHETIC_MLP


@torch.no_grad()
def compute_aesthetic(gen_pil, device, clip_path, mlp_path):
    clip_model, processor, mlp = get_models(device, clip_path, mlp_path)
    inputs = processor(images=[gen_pil], return_tensors="pt").to(device)
    feat = clip_model.get_image_features(**inputs)
    feat = F.normalize(feat, dim=-1)
    score = mlp(feat).item()
    return score


def main():
    p = argparse.ArgumentParser(description=f"Compute {METRIC_NAME}")
    add_common_args(p)
    p.add_argument("--clip_l14_model", type=str, default=DEFAULT_CLIP_L14_PATH)
    p.add_argument("--aesthetic_mlp", type=str, default=DEFAULT_MLP_PATH)
    args = p.parse_args()

    exp_name = args.exp_name or Path(args.result_dir).name
    samples = discover_samples(args.result_dir, args.tag)
    all_gt_indices = get_all_gt_indices(samples, args.max_samples)
    valid_indices = get_valid_indices(samples, args.max_samples)
    failed_indices = set(all_gt_indices) - set(valid_indices)

    existing_per_sample, done_indices = load_existing_results(args.output_dir, exp_name, METRIC_NAME) if args.resume else ([], set())
    todo_valid = [i for i in valid_indices if i not in done_indices]
    todo_failed = failed_indices - done_indices
    print(f"[Aesthetic] total={len(all_gt_indices)}, valid={len(valid_indices)}, failed={len(failed_indices)}"
          + (f", skip={len(done_indices)}, todo={len(todo_valid)}" if args.resume else ""))

    per_sample = list(existing_per_sample)
    for idx in tqdm(todo_valid, desc="Aesthetic"):
        info = samples[idx]

        values = []
        for cand_path in info["candidates"]:
            gen_pil = load_image_pil(cand_path)
            if gen_pil is None:
                values.append(float("nan"))
                continue
            values.append(compute_aesthetic(gen_pil, args.device, args.clip_l14_model, args.aesthetic_mlp))

        if not values:
            failed_indices.add(idx)
            continue

        row = {"sample_idx": idx, "failed": False, **aggregate_candidates(values)}
        per_sample.append(row)

    for idx in sorted(todo_failed):
        per_sample.append(make_failed_row(idx))

    summary = compute_dataset_summary(per_sample, METRIC_NAME)
    print_summary(summary, METRIC_NAME, higher_better=HIGHER_BETTER)
    save_results(per_sample, summary, METRIC_NAME, args.output_dir, exp_name)


if __name__ == "__main__":
    main()
