#!/usr/bin/env python3
"""
OmniSVG Baseline Compare Script
================================
Uses the same multi-candidate + validation approach as the original OmniSVG
inference script, but runs on HVM test dataset samples for fair comparison.

Usage:
    CUDA_VISIBLE_DEVICES=4 python inference/inference_omnisvg_compare.py \
        --sample_indices $(seq 0 99) \
        --output_dir inference/output_omnisvg_pure \
        --save_png --save_gt \
        --num_candidates 5 \
        --resume
"""

import argparse
import gc
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, AutoProcessor

from decoder import SketchDecoder
from hvm_dataset import HVMDataset
from utils.config import OmniSVGConfig, TokenizationConfig, TrainConfig, MODEL_DEFAULTS
from train import load_checkpoint_state_dict, find_checkpoint_file
from tokenizer import SVGTokenizer

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

SYSTEM_PROMPT = (
    "You are an expert SVG code generator. "
    "Generate precise, valid SVG path commands that accurately represent "
    "the described scene or object."
)

SVG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
TARGET_IMAGE_SIZE = 448
RENDER_SIZE = 512
EXTRA_CANDIDATES_BUFFER = 4
MIN_SVG_LENGTH = 20
EMPTY_THRESHOLD = 250


def render_svg_to_image(svg_str, size=RENDER_SIZE):
    try:
        import cairosvg
        from PIL import Image

        png_data = cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            output_width=size, output_height=size,
        )
        img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    except Exception as e:
        print(f"  Render error: {e}")
        return None


def is_valid_candidate(svg_str, img):
    if not svg_str or len(svg_str) < MIN_SVG_LENGTH:
        return False, "too_short"
    if "<svg" not in svg_str:
        return False, "no_svg_tag"
    if img is None:
        return False, "render_failed"
    img_array = np.array(img)
    if img_array.mean() > EMPTY_THRESHOLD:
        return False, "empty_image"
    return True, "ok"


def load_gt_svg(dataset, dataset_index):
    try:
        idx = dataset.valid_indices[dataset_index]
        meta = dataset.idx_to_meta[idx]
        table = dataset._get_parquet_table(meta["parquet_file"])
        row = meta["parquet_row"]
        return table.column("svg")[row].as_py()
    except Exception as e:
        print(f"  Warning: failed to load GT SVG: {e}")
        return None


def parse_args():
    p = argparse.ArgumentParser(description="OmniSVG Pure Baseline Compare")
    p.add_argument("--model_size", type=str, default="8B", choices=["4B", "8B"])
    p.add_argument("--config_dir", type=str, default=None)
    p.add_argument("--omnisvg_checkpoint", type=str, default=None)

    p.add_argument("--data_dir", type=str,
                    default="/mnt/a100_1_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test")
    p.add_argument("--hvm_dir", type=str,
                    default="/mnt/a100_1_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed")
    p.add_argument("--sample_indices", type=int, nargs="+", default=[0])

    p.add_argument("--max_new_tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=0.90)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--num_candidates", type=int, default=5,
                    help="Num valid candidates to collect (generates more with buffer)")

    p.add_argument("--output_dir", type=str, default="./output_omnisvg_pure")
    p.add_argument("--save_png", action="store_true", default=False)
    p.add_argument("--save_gt", action="store_true", default=False)
    p.add_argument("--save_all_candidates", action="store_true", default=False)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    if args.config_dir is None:
        args.config_dir = os.path.join(PROJECT_ROOT, "configs")

    print("=" * 70)
    print("OmniSVG Pure Baseline Compare")
    print("=" * 70)
    print(f"  Model size     : {args.model_size}")
    print(f"  Num candidates : {args.num_candidates} (+{EXTRA_CANDIDATES_BUFFER} buffer)")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Max new tokens : {args.max_new_tokens}")
    print(f"  Device         : {args.device}")
    print("=" * 70)

    config = OmniSVGConfig(config_dir=args.config_dir, model_size=args.model_size)
    token_config = config.tokenization
    defaults = MODEL_DEFAULTS[args.model_size]
    base_model_path = token_config.base_model or defaults["base_model"]

    print(f"\n[1/3] Loading tokenizer from {base_model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, padding_side="left",
        trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(
        base_model_path, padding_side="left",
        trust_remote_code=True, local_files_only=True)
    processor.tokenizer.padding_side = "left"

    print(f"[2/3] Loading OmniSVG {args.model_size} model ...")
    model = SketchDecoder(
        pix_len=6000,
        text_len=800,
        model_path=base_model_path,
        vocab_size=token_config.extended_vocab_size,
        bos_token_id=token_config.bos_token_id,
        eos_token_id=token_config.eos_token_id,
        pad_token_id=token_config.pad_token_id,
    )

    ckpt_path = args.omnisvg_checkpoint or token_config.checkpoint or defaults["checkpoint"]
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"  Loading OmniSVG checkpoint from {ckpt_path}")
        if os.path.isdir(ckpt_path):
            ckpt_file = find_checkpoint_file(ckpt_path)
            state_dict = load_checkpoint_state_dict(ckpt_file) if ckpt_file else None
        else:
            state_dict = load_checkpoint_state_dict(ckpt_path)
        if state_dict:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print(f"  WARNING: checkpoint not found at {ckpt_path}!")

    model = model.to(args.device).eval()
    if hasattr(model.transformer, "gradient_checkpointing_disable"):
        model.transformer.gradient_checkpointing_disable()
    print(f"  Model loaded on {args.device}")

    print(f"[3/3] Loading SVG tokenizer ...")
    svg_tokenizer = SVGTokenizer(SVG_CONFIG_PATH, model_size=args.model_size)

    print("\nLoading dataset ...")
    dataset = HVMDataset(
        data_dir=args.data_dir,
        hvm_dir=args.hvm_dir,
        token_config=token_config,
        train_config=TrainConfig(model_size=args.model_size),
        max_len=6000,
    )
    print(f"  Dataset size: {len(dataset)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_ok = 0
    total_fail = 0
    total_skipped = 0
    BLACK_COLOR_TOKEN = svg_tokenizer.COLOR_TOKEN_START_RAW + 2

    for idx in args.sample_indices:
        if idx >= len(dataset):
            print(f"\n[Skip] sample {idx} out of range")
            continue

        if args.resume:
            if (output_dir / f"sample_{idx:04d}_omnisvg.svg").exists():
                total_skipped += 1
                continue

        sample = dataset[idx]
        text = sample["text"]
        print(f"\n{'='*70}")
        print(f"Sample {idx}: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"{'='*70}")

        t0 = time.time()

        if args.save_gt:
            gt_svg = load_gt_svg(dataset, idx)
            if gt_svg:
                svg_str = gt_svg
                if "width=" not in svg_str:
                    svg_str = svg_str.replace(
                        "<svg",
                        f'<svg width="{TARGET_IMAGE_SIZE}" height="{TARGET_IMAGE_SIZE}"', 1)
                (output_dir / f"sample_{idx:04d}_gt.svg").write_text(
                    svg_str, encoding="utf-8")
                if args.save_png:
                    img = render_svg_to_image(svg_str)
                    if img is not None:
                        img.save(str(output_dir / f"sample_{idx:04d}_gt.png"))
                print(f"  -> sample_{idx:04d}_gt.svg")

        instruction = (
            f"Generate an SVG illustration for: {text}\n\n"
            "Requirements:\n"
            "- Create complete SVG path commands\n"
            "- Include proper coordinates and colors\n"
            "- Maintain visual clarity and composition"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": instruction}]},
        ]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_input], padding=True, truncation=True, return_tensors="pt")

        bos = torch.full((1, 1), token_config.bos_token_id, dtype=torch.long)
        input_ids = torch.cat([inputs["input_ids"], bos], dim=1).to(args.device)
        attention_mask = torch.cat(
            [inputs["attention_mask"], torch.ones(1, 1, dtype=torch.long)], dim=1
        ).to(args.device)
        print(f"  Input tokens: {input_ids.shape[1]}")

        actual_samples = args.num_candidates + EXTRA_CANDIDATES_BUFFER
        gen_cfg = dict(
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=actual_samples,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=int(args.top_k),
            repetition_penalty=args.repetition_penalty,
            eos_token_id=token_config.eos_token_id,
            pad_token_id=token_config.pad_token_id,
            bos_token_id=token_config.bos_token_id,
            use_cache=True,
        )

        valid_candidates = []
        with torch.no_grad():
            results = model.transformer.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_cfg,
            )

            input_len = input_ids.shape[1]
            generated_ids_batch = results[:, input_len:]

            for i in range(generated_ids_batch.shape[0]):
                try:
                    current_ids = generated_ids_batch[i:i+1].cpu()
                    fake_wrapper = torch.cat([
                        torch.full((1, 1), token_config.bos_token_id),
                        current_ids,
                        torch.full((1, 1), token_config.eos_token_id),
                    ], dim=1)

                    generated_xy = svg_tokenizer.process_generated_tokens(fake_wrapper)
                    if len(generated_xy) == 0:
                        print(f"    Candidate {i}: invalid (empty_tokens)")
                        continue

                    svg_tensors, color_tensors = svg_tokenizer.raster_svg(generated_xy)
                    if not svg_tensors or not svg_tensors[0]:
                        print(f"    Candidate {i}: invalid (raster_failed)")
                        continue

                    num_paths = len(svg_tensors[0])
                    while len(color_tensors) < num_paths:
                        color_tensors.append(BLACK_COLOR_TOKEN)

                    svg_obj = svg_tokenizer.apply_colors_to_svg(
                        svg_tensors[0], color_tensors)
                    svg_str = svg_obj.to_str()

                    if "width=" not in svg_str:
                        svg_str = svg_str.replace(
                            "<svg",
                            f'<svg width="{TARGET_IMAGE_SIZE}" height="{TARGET_IMAGE_SIZE}"',
                            1)

                    img = render_svg_to_image(svg_str)
                    is_valid, reason = is_valid_candidate(svg_str, img)

                    if is_valid:
                        valid_candidates.append({
                            "svg_str": svg_str,
                            "num_paths": num_paths,
                            "img": img,
                        })
                        print(f"    Candidate {i}: VALID ({num_paths} paths)")
                        if len(valid_candidates) >= args.num_candidates:
                            break
                    else:
                        print(f"    Candidate {i}: invalid ({reason}, {num_paths} paths)")

                except Exception as e:
                    print(f"    Candidate {i}: error ({e})")
                    continue

        gen_elapsed = time.time() - t0

        if valid_candidates:
            total_ok += 1
            print(f"  OK: {len(valid_candidates)}/{actual_samples} valid in {gen_elapsed:.1f}s")

            best = valid_candidates[0]
            base_name = f"sample_{idx:04d}_omnisvg"
            (output_dir / f"{base_name}.svg").write_text(
                best["svg_str"], encoding="utf-8")
            print(f"  -> {base_name}.svg ({best['num_paths']} paths)")
            if args.save_png and best["img"] is not None:
                best["img"].save(str(output_dir / f"{base_name}.png"))

            if args.save_all_candidates:
                for ci, cand in enumerate(valid_candidates):
                    cname = f"sample_{idx:04d}_omnisvg_c{ci}"
                    (output_dir / f"{cname}.svg").write_text(
                        cand["svg_str"], encoding="utf-8")
                    if args.save_png and cand["img"] is not None:
                        cand["img"].save(str(output_dir / f"{cname}.png"))
        else:
            total_fail += 1
            print(f"  FAILED: 0/{actual_samples} valid ({gen_elapsed:.1f}s)")

        (output_dir / f"sample_{idx:04d}.txt").write_text(text, encoding="utf-8")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("OmniSVG Pure Baseline Complete!")
    if total_skipped > 0:
        print(f"  Skipped: {total_skipped} (already done)")
    print(f"  Success: {total_ok}/{len(args.sample_indices)}")
    print(f"  Failed : {total_fail}/{len(args.sample_indices)}")
    print(f"  Output : {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
