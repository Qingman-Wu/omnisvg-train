#!/usr/bin/env python3
"""计算 DINO-I (DINOv2 image-image cosine similarity vs GT, ↑ better)"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import (
    add_common_args, discover_samples, get_valid_indices, get_all_gt_indices,
    load_image_pil, aggregate_candidates, make_failed_row,
    compute_dataset_summary, save_results, print_summary, load_existing_results,
    DEFAULT_DINO_PATH,
)

METRIC_NAME = "dino_i"
HIGHER_BETTER = True

_MODEL = None
_PROCESSOR = None


def get_model(device, model_path):
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from transformers import AutoModel, AutoImageProcessor
        _MODEL = AutoModel.from_pretrained(model_path).to(device).eval()
        _PROCESSOR = AutoImageProcessor.from_pretrained(model_path)
        print(f"[DINO-I] Model loaded from {model_path}")
    return _MODEL, _PROCESSOR


@torch.no_grad()
def compute_dino_i(gen_pil, gt_pil, device, model_path):
    model, processor = get_model(device, model_path)
    inputs = processor(images=[gen_pil, gt_pil], return_tensors="pt").to(device)
    outputs = model(**inputs)
    feats = outputs.last_hidden_state[:, 0]  # CLS token
    feats = F.normalize(feats, dim=-1)
    return (feats[0] @ feats[1]).item()


def main():
    p = argparse.ArgumentParser(description=f"Compute {METRIC_NAME.upper()}")
    add_common_args(p)
    p.add_argument("--dino_model", type=str, default=DEFAULT_DINO_PATH)
    args = p.parse_args()

    exp_name = args.exp_name or Path(args.result_dir).name
    samples = discover_samples(args.result_dir, args.tag)
    all_gt_indices = get_all_gt_indices(samples, args.max_samples)
    valid_indices = get_valid_indices(samples, args.max_samples)
    failed_indices = set(all_gt_indices) - set(valid_indices)

    existing_per_sample, done_indices = load_existing_results(args.output_dir, exp_name, METRIC_NAME) if args.resume else ([], set())
    todo_valid = [i for i in valid_indices if i not in done_indices]
    todo_failed = failed_indices - done_indices
    print(f"[DINO-I] total={len(all_gt_indices)}, valid={len(valid_indices)}, failed={len(failed_indices)}"
          + (f", skip={len(done_indices)}, todo={len(todo_valid)}" if args.resume else ""))

    per_sample = list(existing_per_sample)
    for idx in tqdm(todo_valid, desc="DINO-I"):
        info = samples[idx]
        gt_pil = load_image_pil(info["gt_png"])
        if gt_pil is None:
            failed_indices.add(idx)
            continue

        values = []
        for cand_path in info["candidates"]:
            gen_pil = load_image_pil(cand_path)
            if gen_pil is None:
                values.append(float("nan"))
                continue
            values.append(compute_dino_i(gen_pil, gt_pil, args.device, args.dino_model))

        if not values:
            failed_indices.add(idx)
            continue

        row = {"sample_idx": idx, "failed": False, **aggregate_candidates(values)}
        per_sample.append(row)

    # 失败样本得分补 0
    for idx in sorted(todo_failed):
        per_sample.append(make_failed_row(idx))

    summary = compute_dataset_summary(per_sample, METRIC_NAME)
    print_summary(summary, METRIC_NAME, higher_better=HIGHER_BETTER)
    save_results(per_sample, summary, METRIC_NAME, args.output_dir, exp_name)


if __name__ == "__main__":
    main()
