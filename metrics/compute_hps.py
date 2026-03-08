#!/usr/bin/env python3
"""计算 HPS v2 (Human Preference Score, ↑ better)

使用 ViT-H-14 backbone + HPS checkpoint 评估图文对齐的人类偏好。
需要 text prompt。
"""

import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from common import (
    add_common_args, discover_samples, get_valid_indices, get_all_gt_indices,
    load_image_pil, aggregate_candidates, make_failed_row,
    compute_dataset_summary, save_results, print_summary, load_existing_results,
)

METRIC_NAME = "hps"
HIGHER_BETTER = True

DEFAULT_HPS_CKPT = "/mnt/a100_1_data2/wuqingman/models/xswu/HPSv2/HPS_v2.pt"

_HPS_MODEL = None
_HPS_PREPROCESS = None
_HPS_TOKENIZER = None


def get_model(device, ckpt_path):
    global _HPS_MODEL, _HPS_PREPROCESS, _HPS_TOKENIZER
    if _HPS_MODEL is not None:
        return _HPS_MODEL, _HPS_PREPROCESS, _HPS_TOKENIZER

    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    model, _, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        '',
        precision='amp',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device).eval()
    tokenizer = get_tokenizer('ViT-H-14')

    _HPS_MODEL = model
    _HPS_PREPROCESS = preprocess_val
    _HPS_TOKENIZER = tokenizer
    print(f"[HPS] Model loaded from {ckpt_path}")
    return model, preprocess_val, tokenizer


@torch.no_grad()
def compute_hps(gen_pil, text, device, ckpt_path):
    model, preprocess, tokenizer = get_model(device, ckpt_path)

    image = preprocess(gen_pil).unsqueeze(0).to(device=device, non_blocking=True)
    text_tokens = tokenizer([text]).to(device=device, non_blocking=True)

    with torch.cuda.amp.autocast():
        outputs = model(image, text_tokens)
        img_feat = outputs["image_features"]
        txt_feat = outputs["text_features"]
        score = (img_feat @ txt_feat.T).diagonal().cpu().item()

    return score


def main():
    p = argparse.ArgumentParser(description=f"Compute {METRIC_NAME.upper()}")
    add_common_args(p)
    p.add_argument("--hps_ckpt", type=str, default=DEFAULT_HPS_CKPT)
    args = p.parse_args()

    exp_name = args.exp_name or Path(args.result_dir).name
    samples = discover_samples(args.result_dir, args.tag)
    all_gt_indices = get_all_gt_indices(samples, args.max_samples)
    valid_indices = get_valid_indices(samples, args.max_samples)
    failed_indices = set(all_gt_indices) - set(valid_indices)

    existing_per_sample, done_indices = load_existing_results(args.output_dir, exp_name, METRIC_NAME) if args.resume else ([], set())
    todo_valid = [i for i in valid_indices if i not in done_indices]
    todo_failed = failed_indices - done_indices
    print(f"[HPS] total={len(all_gt_indices)}, valid={len(valid_indices)}, failed={len(failed_indices)}"
          + (f", skip={len(done_indices)}, todo={len(todo_valid)}" if args.resume else ""))

    per_sample = list(existing_per_sample)
    skipped_no_text = 0
    for idx in tqdm(todo_valid, desc="HPS"):
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
            values.append(compute_hps(gen_pil, text, args.device, args.hps_ckpt))

        if not values:
            failed_indices.add(idx)
            continue

        row = {"sample_idx": idx, "failed": False, **aggregate_candidates(values)}
        per_sample.append(row)

    if skipped_no_text:
        print(f"[HPS] Skipped {skipped_no_text} samples without text (not counted as failed)")

    for idx in sorted(todo_failed):
        per_sample.append(make_failed_row(idx))

    summary = compute_dataset_summary(per_sample, METRIC_NAME)
    print_summary(summary, METRIC_NAME, higher_better=HIGHER_BETTER)
    save_results(per_sample, summary, METRIC_NAME, args.output_dir, exp_name)


if __name__ == "__main__":
    main()
