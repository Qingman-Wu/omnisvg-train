#!/usr/bin/env python3
"""
OmniSVG vLLM Inference for Test Holdout Dataset
=================================================
使用 vLLM 加速推理 OmniSVG 基模（无 HVM），适用于 test_holdout 数据集。

数据集: MMSVG-Illustration/data_test_holdout/data_test_holdout.parquet
  - 1000 个样本，每个包含 id, svg(GT), description, keywords, detail, image, token_len

用法示例:

  CUDA_VISIBLE_DEVICES=4,5 python inference/inference_vllm_omnisvg.py \
      --model_path /mnt/a100_1_data2/wuqingman/models/OmniSVG/omnisvg_8B_vllm/omnisvg_8B_vllm \
      --data_path /mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test_holdout/data_test_holdout.parquet \
      --output_dir ./inference_results/vllm_omnisvg_baseline \
      --num_gpus 2 \
      --batch_size 16 \
      --num_candidates 5 \
      --save_png \
      --save_gt \
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor
from tokenizer import SVGTokenizer

os.environ['TRITON_CACHE_DIR'] = '/tmp/.triton_cache'
os.environ['VLLM_USAGE_STATS_SERVER'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ---------------------------------------------------------------------------
SVG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
TARGET_IMAGE_SIZE = 448
RENDER_SIZE = 512
MIN_SVG_LENGTH = 20
EMPTY_THRESHOLD = 250
EXTRA_CANDIDATES_BUFFER = 4

BOS_TOKEN_ID = 196998
EOS_TOKEN_ID = 196999
PAD_TOKEN_ID = 151643

SYSTEM_PROMPT = (
    "You are an expert SVG code generator. "
    "Generate precise, valid SVG path commands that accurately represent "
    "the described scene or object."
)


# ============================================================================
# Data loading
# ============================================================================

def load_test_dataset(parquet_path: str) -> List[Dict[str, Any]]:
    """从 parquet 加载 test_holdout 数据集"""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    samples = []
    for i in range(len(table)):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        row["parquet_row"] = i
        samples.append(row)
    print(f"Loaded {len(samples)} samples from {os.path.basename(parquet_path)}")
    return samples


# ============================================================================
# Prompt building
# ============================================================================

def build_prompt(description: str, tokenizer) -> str:
    instruction = (
        f"Generate an SVG illustration for: {description}\n\n"
        "Requirements:\n"
        "- Create complete SVG path commands\n"
        "- Include proper coordinates and colors\n"
        "- Maintain visual clarity and composition"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": instruction}]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ============================================================================
# SVG decoding
# ============================================================================

def decode_tokens_to_svg(token_ids: List[int], svg_tokenizer: SVGTokenizer) -> Optional[Dict]:
    """将 vLLM 生成的 token IDs 解码为 SVG 字符串"""
    try:
        token_tensor = torch.tensor([token_ids], dtype=torch.long)
        fake_wrapper = torch.cat([
            torch.full((1, 1), BOS_TOKEN_ID),
            token_tensor,
            torch.full((1, 1), EOS_TOKEN_ID),
        ], dim=1)

        generated_xy = svg_tokenizer.process_generated_tokens(fake_wrapper)
        if len(generated_xy) == 0:
            return None

        svg_tensors, color_tensors = svg_tokenizer.raster_svg(generated_xy)
        if not svg_tensors or not svg_tensors[0]:
            return None

        num_paths = len(svg_tensors[0])
        BLACK_COLOR_TOKEN = svg_tokenizer.COLOR_TOKEN_START_RAW + 2
        while len(color_tensors) < num_paths:
            color_tensors.append(BLACK_COLOR_TOKEN)

        svg_obj = svg_tokenizer.apply_colors_to_svg(svg_tensors[0], color_tensors)
        svg_str = svg_obj.to_str()

        if "width=" not in svg_str:
            svg_str = svg_str.replace(
                "<svg",
                f'<svg width="{TARGET_IMAGE_SIZE}" height="{TARGET_IMAGE_SIZE}"',
                1,
            )

        return {
            "svg_str": svg_str,
            "num_paths": num_paths,
            "tokens": token_ids,
        }
    except Exception as e:
        return None


def render_svg_to_image(svg_str: str, size: int = RENDER_SIZE):
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
    except Exception:
        return None


def validate_candidate(svg_str: str) -> bool:
    if not svg_str or len(svg_str) < MIN_SVG_LENGTH:
        return False
    if "<svg" not in svg_str:
        return False
    img = render_svg_to_image(svg_str)
    if img is None:
        return False
    img_array = np.array(img)
    if img_array.mean() > EMPTY_THRESHOLD:
        return False
    return True


# ============================================================================
# GT saving
# ============================================================================

def save_gt_svg(output_dir: Path, idx: int, gt_svg: str, save_png: bool):
    base = f"sample_{idx:04d}_gt"
    svg_str = gt_svg
    if "width=" not in svg_str:
        svg_str = svg_str.replace(
            "<svg",
            f'<svg width="{TARGET_IMAGE_SIZE}" height="{TARGET_IMAGE_SIZE}"',
            1,
        )
    (output_dir / f"{base}.svg").write_text(svg_str, encoding="utf-8")
    if save_png:
        img = render_svg_to_image(svg_str)
        if img is not None:
            img.save(str(output_dir / f"{base}.png"))


# ============================================================================
# Main logic
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="OmniSVG vLLM Inference for Test Holdout")

    p.add_argument("--model_path", type=str, required=True,
                   help="已转换的 vLLM 模型路径")
    p.add_argument("--model_size", type=str, default="8B", choices=["4B", "8B"])
    p.add_argument("--data_path", type=str,
                   default="/mnt/a100_4_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test_holdout/data_test_holdout.parquet",
                   help="测试数据集 parquet 路径")
    p.add_argument("--sample_indices", type=int, nargs="+", default=None,
                   help="指定推理的样本索引，默认全部")

    # vLLM
    p.add_argument("--num_gpus", type=int, default=1,
                   help="vLLM tensor parallel GPU 数量")
    p.add_argument("--batch_size", type=int, default=32,
                   help="vLLM 批处理大小")
    p.add_argument("--max_model_len", type=int, default=4096,
                   help="vLLM 最大序列长度")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                   help="vLLM GPU 显存利用率")

    # generation
    p.add_argument("--max_new_tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=0.90)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--num_candidates", type=int, default=5,
                   help="每个样本生成几个候选 SVG")
    p.add_argument("--no_validate", action="store_true", default=False,
                   help="跳过 SVG 验证")

    # output
    p.add_argument("--output_dir", type=str, default="./inference_results/vllm_base_test")
    p.add_argument("--save_png", action="store_true", default=False)
    p.add_argument("--save_gt", action="store_true", default=False,
                   help="保存 Ground Truth SVG")
    p.add_argument("--save_tokens", action="store_true", default=False)
    p.add_argument("--resume", action="store_true", default=False,
                   help="跳过已生成的样本")

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("OmniSVG vLLM Inference  --  Test Holdout Dataset")
    print("=" * 70)
    print(f"  Model path       : {args.model_path}")
    print(f"  Model size       : {args.model_size}")
    print(f"  Data path        : {args.data_path}")
    print(f"  Num GPUs (TP)    : {args.num_gpus}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Max new tokens   : {args.max_new_tokens}")
    print(f"  Num candidates   : {args.num_candidates}")
    print(f"  Output dir       : {args.output_dir}")
    print(f"  Resume           : {args.resume}")
    print("=" * 70)

    # --- Load dataset ---
    samples = load_test_dataset(args.data_path)

    # --- Filter sample indices ---
    if args.sample_indices is not None:
        indices = [i for i in args.sample_indices if 0 <= i < len(samples)]
    else:
        indices = list(range(len(samples)))

    # --- Resume: skip existing ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        remaining = []
        for idx in indices:
            svg_file = output_dir / f"sample_{idx:04d}_base.svg"
            if not svg_file.exists():
                remaining.append(idx)
        skipped = len(indices) - len(remaining)
        if skipped > 0:
            print(f"  Resume: skipping {skipped} already-completed samples")
        indices = remaining

    if not indices:
        print("All samples already processed!")
        return

    print(f"  Samples to process: {len(indices)}")

    # --- Save GT first (before loading vLLM to save memory) ---
    if args.save_gt:
        print("\nSaving Ground Truth SVGs...")
        gt_count = 0
        for idx in indices:
            gt_svg = samples[idx].get("svg")
            if gt_svg:
                save_gt_svg(output_dir, idx, gt_svg, args.save_png)
                gt_count += 1
        print(f"  Saved {gt_count} GT SVGs")

    # --- Load tokenizer ---
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, padding_side="left", trust_remote_code=True
    )

    # --- Build all prompts ---
    print("[2/3] Building prompts...")
    all_prompts = {}
    for idx in indices:
        description = samples[idx]["description"]
        all_prompts[idx] = build_prompt(description, tokenizer)

    # --- Load vLLM model ---
    print("[3/3] Loading vLLM model...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.num_gpus,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # --- SVG tokenizer ---
    svg_tokenizer = SVGTokenizer(SVG_CONFIG_PATH, model_size=args.model_size)

    # --- Sampling params ---
    actual_n = args.num_candidates + EXTRA_CANDIDATES_BUFFER
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        n=actual_n,
        stop_token_ids=[EOS_TOKEN_ID],
    )

    # --- Batch inference ---
    total_ok = 0
    total_fail = 0
    t_start = time.time()

    for batch_start in range(0, len(indices), args.batch_size):
        batch_indices = indices[batch_start: batch_start + args.batch_size]
        batch_prompts = [all_prompts[idx] for idx in batch_indices]

        print(f"\nBatch {batch_start // args.batch_size + 1}: "
              f"samples {batch_indices[0]}~{batch_indices[-1]} "
              f"({len(batch_indices)} prompts, n={actual_n})")

        t_batch = time.time()
        outputs = llm.generate(batch_prompts, sampling_params)
        gen_elapsed = time.time() - t_batch
        print(f"  vLLM generate: {gen_elapsed:.1f}s "
              f"({len(batch_indices) / gen_elapsed:.2f} samples/s)")

        # --- Decode & save ---
        t_decode = time.time()
        for out_idx, (idx, output) in enumerate(zip(batch_indices, outputs)):
            sample = samples[idx]
            description = sample["description"]

            candidates = []
            for comp in output.outputs:
                cand = decode_tokens_to_svg(list(comp.token_ids), svg_tokenizer)
                if cand is None:
                    continue
                if not args.no_validate and not validate_candidate(cand["svg_str"]):
                    continue
                candidates.append(cand)
                if len(candidates) >= args.num_candidates:
                    break

            if candidates:
                total_ok += 1
                for ci, cand in enumerate(candidates):
                    suffix = f"_c{ci}" if len(candidates) > 1 else ""
                    base = f"sample_{idx:04d}_base{suffix}"

                    svg_path = output_dir / f"{base}.svg"
                    svg_path.write_text(cand["svg_str"], encoding="utf-8")

                    if args.save_png:
                        img = render_svg_to_image(cand["svg_str"])
                        if img is not None:
                            img.save(str(output_dir / f"{base}.png"))

                    if args.save_tokens:
                        tok_path = output_dir / f"{base}_tokens.json"
                        tok_path.write_text(json.dumps({
                            "description": description,
                            "tokens": cand["tokens"],
                            "num_tokens": len(cand["tokens"]),
                        }, indent=2), encoding="utf-8")

                print(f"  [{idx:4d}] OK: {len(candidates)} candidates, "
                      f"{candidates[0]['num_paths']} paths | "
                      f"{description[:60]}...")
            else:
                total_fail += 1
                print(f"  [{idx:4d}] FAIL | {description[:60]}...")

            # save prompt text
            (output_dir / f"sample_{idx:04d}.txt").write_text(
                description, encoding="utf-8"
            )

        decode_elapsed = time.time() - t_decode
        print(f"  Decode + save: {decode_elapsed:.1f}s")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"All done!")
    print(f"  Success : {total_ok}/{len(indices)}")
    print(f"  Failed  : {total_fail}/{len(indices)}")
    print(f"  Time    : {total_elapsed:.1f}s ({len(indices) / total_elapsed:.2f} samples/s)")
    print(f"  Output  : {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
