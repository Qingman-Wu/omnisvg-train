#!/usr/bin/env python3
"""计算 CLIP-T (CLIP text-image cosine similarity, ↑ better)"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import (
    add_common_args, discover_samples, get_valid_indices, get_all_gt_indices,
    load_image_pil, aggregate_candidates, make_failed_row,
    compute_dataset_summary, save_results, print_summary, load_existing_results,
    DEFAULT_CLIP_PATH,
)

METRIC_NAME = "clip_t"
HIGHER_BETTER = True

_MODEL = None
_PROCESSOR = None


def get_model(device, model_path):
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from transformers import CLIPModel, CLIPProcessor
        _MODEL = CLIPModel.from_pretrained(model_path).to(device).eval()
        _PROCESSOR = CLIPProcessor.from_pretrained(model_path)
        print(f"[CLIP-T] Model loaded from {model_path}")
    return _MODEL, _PROCESSOR


@torch.no_grad()
def compute_clip_t(gen_pil, text, device, model_path):
    model, processor = get_model(device, model_path)
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


def main():
    p = argparse.ArgumentParser(description=f"Compute {METRIC_NAME.upper()}")
    add_common_args(p)
    p.add_argument("--clip_model", type=str, default=DEFAULT_CLIP_PATH)
    args = p.parse_args()

    exp_name = args.exp_name or Path(args.result_dir).name
    samples = discover_samples(args.result_dir, args.tag)
    all_gt_indices = get_all_gt_indices(samples, args.max_samples)
    valid_indices = get_valid_indices(samples, args.max_samples)
    failed_indices = set(all_gt_indices) - set(valid_indices)

    existing_per_sample, done_indices = load_existing_results(args.output_dir, exp_name, METRIC_NAME) if args.resume else ([], set())
    todo_valid = [i for i in valid_indices if i not in done_indices]
    todo_failed = failed_indices - done_indices
    print(f"[CLIP-T] total={len(all_gt_indices)}, valid={len(valid_indices)}, failed={len(failed_indices)}"
          + (f", skip={len(done_indices)}, todo={len(todo_valid)}" if args.resume else ""))

    per_sample = list(existing_per_sample)
    skipped_no_text = 0
    for idx in tqdm(todo_valid, desc="CLIP-T"):
        info = samples[idx]

        text = ""
        if info["text_file"] and os.path.exists(info["text_file"]):
            text = Path(info["text_file"]).read_text(encoding="utf-8").strip()
        if not text:
            skipped_no_text += 1
            continue

        values = []
        for cand_path in info["candidates"]:
            gen_pil = load_image_pil(cand_path)
            if gen_pil is None:
                values.append(float("nan"))
                continue
            values.append(compute_clip_t(gen_pil, text, args.device, args.clip_model))

        if not values:
            failed_indices.add(idx)
            continue

        row = {"sample_idx": idx, "failed": False, **aggregate_candidates(values)}
        per_sample.append(row)

    if skipped_no_text:
        print(f"[CLIP-T] Skipped {skipped_no_text} samples without text (not counted as failed)")

    # 失败样本得分补 0
    for idx in sorted(todo_failed):
        per_sample.append(make_failed_row(idx))

    summary = compute_dataset_summary(per_sample, METRIC_NAME)
    print_summary(summary, METRIC_NAME, higher_better=HIGHER_BETTER)
    save_results(per_sample, summary, METRIC_NAME, args.output_dir, exp_name)


if __name__ == "__main__":
    main()
