#!/usr/bin/env python3
"""计算 MSE (Mean Squared Error, ↓ better)"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (
    add_common_args, discover_samples, get_valid_indices, get_all_gt_indices,
    load_image_as_array, aggregate_candidates, make_failed_row,
    compute_dataset_summary, save_results, print_summary, load_existing_results,
)

METRIC_NAME = "mse"
HIGHER_BETTER = False


def compute_mse(gen, gt):
    return float(np.mean((gen - gt) ** 2))


def main():
    p = argparse.ArgumentParser(description=f"Compute {METRIC_NAME.upper()}")
    add_common_args(p)
    args = p.parse_args()

    exp_name = args.exp_name or Path(args.result_dir).name
    samples = discover_samples(args.result_dir, args.tag)
    all_gt_indices = get_all_gt_indices(samples, args.max_samples)
    valid_indices = get_valid_indices(samples, args.max_samples)
    failed_indices = set(all_gt_indices) - set(valid_indices)

    existing_per_sample, done_indices = load_existing_results(args.output_dir, exp_name, METRIC_NAME) if args.resume else ([], set())
    todo_valid = [i for i in valid_indices if i not in done_indices]
    todo_failed = failed_indices - done_indices
    print(f"[{METRIC_NAME.upper()}] total={len(all_gt_indices)}, valid={len(valid_indices)}, failed={len(failed_indices)}"
          + (f", skip={len(done_indices)}, todo={len(todo_valid)}" if args.resume else ""))

    per_sample = list(existing_per_sample)
    for idx in tqdm(todo_valid, desc=METRIC_NAME.upper()):
        info = samples[idx]
        gt_arr = load_image_as_array(info["gt_png"])
        if gt_arr is None:
            failed_indices.add(idx)
            continue

        values = []
        for cand_path in info["candidates"]:
            gen_arr = load_image_as_array(cand_path)
            if gen_arr is None:
                values.append(float("nan"))
                continue
            values.append(compute_mse(gen_arr, gt_arr))

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
