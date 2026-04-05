#!/usr/bin/env python3
"""
Run Qwen/Qwen2.5-VL baseline models on MMSVGBench text2svg with vLLM.

Expected output layout:
  sample_0000.txt
  sample_0000_meta.json
  sample_0000_c0.svg
  sample_0000_c0.png
  ...
  sample_0000_c4.svg
  sample_0000_c4.png

This script is intended for raw text->SVG baseline comparison (no OmniSVG/HVM).
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    import pandas as pd
except ImportError:
    pd = None

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/.triton_cache")
os.environ.setdefault("VLLM_USAGE_STATS_SERVER", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL_PATH = "/mnt/data/wuqingman/models/Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_INPUT_PATH = "/mnt/data/wuqingman/datasets/OmniSVG/MMSVGBench/data/text2svg-00000-of-00001.parquet"
DEFAULT_OUTPUT_DIR = "/mnt/data3/wuqingman/omnisvg-train/baseline_qwen25vl7b_mmsvgbench_vllm"
DEFAULT_RENDER_SIZE = 512
MIN_SVG_LENGTH = 20
EMPTY_THRESHOLD = 250
DEFAULT_VIEWBOX = 200

DEFAULT_SYSTEM_PROMPT = """You are a benchmark-grade SVG author for text-to-SVG generation. Convert each natural-language description into one complete, self-contained SVG illustration. Return exactly one final answer: a single ```svg``` code block containing the full SVG, and nothing else.
Hard requirements:
1. The SVG must be valid XML and must start with `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">` or an equivalent full `<svg ...>` root using the same namespace and a 200x200 viewBox.
2. Generate a complete composition, not fragments. The main subject should be large, centered, and clearly recognizable, typically occupying about 60% to 85% of the canvas.
3. Use only reliable SVG elements and attributes: `svg`, `g`, `path`, `rect`, `circle`, `ellipse`, `polygon`, `polyline`, `line`, `fill`, `stroke`, `stroke-width`, `opacity`, and `transform`.
4. Do not use `text`, CSS, JavaScript, animation, filters, masks, clipPath, pattern, foreignObject, raster images, or external resources.
5. Prefer clean geometry, closed shapes, solid fills, simple strokes, and readable layering. Use concise paths and avoid tiny scattered marks, decorative noise, or nearly invisible details.
6. Faithfully reflect the described subject, pose, color, and spatial layout. If the description is complex, simplify it into the most recognizable visual composition while preserving the core semantics.
7. Ensure the result is renderable, non-empty, and not a blank or near-blank canvas.
8. Do not output explanations, comments, reasoning, or Markdown outside the single ```svg``` block."""

THINK_SVG_SYSTEM_PROMPT = """You are a benchmark-grade SVG author for text-to-SVG generation. First think briefly about composition, geometry, color, and layering. Then output the final result in exactly this format:
<think>
brief plan
</think>
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">...</svg>
```
Hard requirements:
1. The final SVG must be valid, self-contained, renderable, and non-empty.
2. The main subject should be large, centered, and visually clear.
3. Use only reliable SVG elements and attributes: `svg`, `g`, `path`, `rect`, `circle`, `ellipse`, `polygon`, `polyline`, `line`, `fill`, `stroke`, `stroke-width`, `opacity`, and `transform`.
4. Do not use `text`, CSS, JavaScript, animation, filters, masks, clipPath, pattern, foreignObject, raster images, or external resources.
5. Keep the drawing concise and structurally clean; if the scene is complex, simplify it into a strong recognizable composition.
6. After the closing code fence, output nothing."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen/Qwen2.5-VL baseline on MMSVGBench text2svg with vLLM."
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="chat",
        choices=["auto", "chat", "plain", "llama"],
        help="Prompt format. Qwen/Qwen-VL baselines usually use chat.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="code_only",
        choices=["code_only", "think_svg"],
        help="Use direct SVG output or <think> + ```svg``` format.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional inline override for the default system prompt.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=None,
        help="Optional text file containing a custom system prompt.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size. Match CUDA_VISIBLE_DEVICES count.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=3000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=None,
        help="How many candidates to sample in parallel per request. Defaults to --num-candidates.",
    )
    parser.add_argument("--extra-candidates-buffer", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--save-png",
        action="store_true",
        default=False,
        help="Render each saved SVG candidate to PNG as well.",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit sample indices to run.",
    )
    parser.add_argument(
        "--save-raw-output",
        action="store_true",
        default=False,
        help="Also save raw generations for debugging.",
    )
    parser.add_argument(
        "--save-meta",
        action="store_true",
        default=True,
        help="Save a per-sample metadata json file.",
    )
    parser.add_argument(
        "--no-save-meta",
        dest="save_meta",
        action="store_false",
        help="Disable per-sample metadata json saving.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip SVG rendering validation and save the first N extractable candidates.",
    )
    return parser.parse_args()


def load_system_prompt(args: argparse.Namespace) -> str:
    if args.system_prompt is not None:
        return args.system_prompt.strip()
    if args.system_prompt_file:
        return Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt_template == "think_svg":
        return THINK_SVG_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT


def resolve_prompt_style(model_path: str, prompt_style: str) -> str:
    if prompt_style != "auto":
        return prompt_style

    lowered = model_path.lower()
    if "starcoder" in lowered:
        return "plain"
    if "llama" in lowered:
        return "llama"
    return "chat"


def build_user_prompt(user_prompt: str, prompt_template: str) -> str:
    if prompt_template == "think_svg":
        return f"""Create a single self-contained SVG illustration for the description below.
Use a 200x200 canvas. First think briefly, then output one full SVG in a single ```svg``` block.

Description: {user_prompt}"""
    return f"""Create a single self-contained SVG illustration for the description below.
Use a 200x200 canvas and return exactly one complete SVG in a single ```svg``` block.

Description: {user_prompt}"""


def build_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    prompt_style: str,
    prompt_template: str,
    system_prompt: str,
) -> str:
    user_content = build_user_prompt(user_prompt, prompt_template)

    if prompt_style == "plain":
        return (
            f"{system_prompt}\n\n"
            "User description:\n"
            f"{user_content}\n\n"
            "Assistant response:\n"
        )

    if prompt_style == "llama":
        return (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_text2svg_samples(input_path: str) -> List[Dict[str, Any]]:
    if pd is None:
        raise ImportError("pandas is required to read benchmark parquet files.")

    df = pd.read_parquet(input_path)
    if "task_type" in df.columns:
        df = df[df["task_type"] == "text2svg"].reset_index(drop=True)

    text_column = None
    for candidate in ("text", "description", "prompt"):
        if candidate in df.columns:
            text_column = candidate
            break
    if text_column is None:
        raise ValueError("Input parquet must contain one of: text / description / prompt")

    samples: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        prompt = row.get(text_column)
        if prompt is None:
            continue
        prompt = str(prompt).strip()
        if not prompt:
            continue

        sample = {
            "index": len(samples),
            "text": prompt,
            "id": row.get("id"),
            "task_type": row.get("task_type"),
            "type": row.get("type"),
            "url": row.get("url"),
        }
        samples.append(sample)

    return samples


def sample_base_name(sample_index: int) -> str:
    return f"sample_{sample_index:04d}"


def strip_reasoning(raw_output: str) -> str:
    return re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL | re.IGNORECASE).strip()


def extract_svg_code(raw_output: str) -> str:
    text = strip_reasoning(raw_output)

    fence_patterns = [
        r"```svg\s*(.*?)```",
        r"```xml\s*(.*?)```",
        r"```html\s*(.*?)```",
        r"```\s*(.*?)```",
    ]
    for pattern in fence_patterns:
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

    svg_match = re.search(r"(<svg[\s\S]*?</svg>)", text, flags=re.DOTALL | re.IGNORECASE)
    if svg_match:
        return svg_match.group(1).strip()

    return text.strip()


def render_svg_to_image(svg_str: str, size: int = DEFAULT_RENDER_SIZE) -> Optional[Image.Image]:
    try:
        import cairosvg

        png_data = cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            output_width=size,
            output_height=size,
        )
        image_rgba = Image.open(io.BytesIO(png_data)).convert("RGBA")
        background = Image.new("RGB", image_rgba.size, (255, 255, 255))
        background.paste(image_rgba, mask=image_rgba.split()[3])
        return background
    except Exception:
        return None


def wrap_fragment_if_needed(svg_str: str) -> str:
    svg_str = svg_str.strip()
    if not svg_str:
        return svg_str
    if "<svg" in svg_str.lower():
        return svg_str
    if any(tag in svg_str.lower() for tag in ("<path", "<g", "<rect", "<circle", "<ellipse", "<polygon", "<line")):
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {DEFAULT_VIEWBOX} {DEFAULT_VIEWBOX}">{svg_str}</svg>'
        )
    return svg_str


def collect_candidates(request_output, target_count: int, validate: bool) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    sequence_outputs = sorted(request_output.outputs, key=lambda x: x.index)
    for output in sequence_outputs:
        raw_text = output.text.strip()
        svg_text = wrap_fragment_if_needed(extract_svg_code(raw_text))
        if not svg_text or len(svg_text) < MIN_SVG_LENGTH:
            continue
        if "<svg" not in svg_text.lower():
            continue

        png_image = render_svg_to_image(svg_text)
        if validate:
            if png_image is None:
                continue
            if np.array(png_image).mean() > EMPTY_THRESHOLD:
                continue

        candidates.append(
            {
                "svg": svg_text,
                "raw_output": raw_text,
                "png_image": png_image,
            }
        )
        if len(candidates) >= target_count:
            break

    return candidates[:target_count]


def is_sample_complete(sample: Dict[str, Any], output_dir: Path, args: argparse.Namespace) -> bool:
    base = sample_base_name(sample["index"])
    txt_path = output_dir / f"{base}.txt"
    candidate_paths = [output_dir / f"{base}_c{i}.svg" for i in range(args.num_candidates)]
    if not txt_path.exists() or not all(path.exists() for path in candidate_paths):
        return False

    if args.save_png:
        png_paths = [output_dir / f"{base}_c{i}.png" for i in range(args.num_candidates)]
        if not all(path.exists() for path in png_paths):
            return False

    return True


def save_prompt_text(sample: Dict[str, Any], output_dir: Path) -> None:
    base = sample_base_name(sample["index"])
    (output_dir / f"{base}.txt").write_text(str(sample["text"]), encoding="utf-8")


def save_failed_raw_outputs(sample: Dict[str, Any], request_output, output_dir: Path) -> None:
    base = sample_base_name(sample["index"])
    sequence_outputs = sorted(request_output.outputs, key=lambda x: x.index)
    for output in sequence_outputs:
        raw_text = output.text.strip()
        raw_path = output_dir / f"{base}_failed_raw_{output.index}.txt"
        raw_path.write_text(raw_text, encoding="utf-8")


def save_sample_meta(
    sample: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    if not args.save_meta:
        return

    base = sample_base_name(sample["index"])
    meta = {
        "index": sample["index"],
        "id": sample.get("id"),
        "text": sample.get("text"),
        "task_type": sample.get("task_type"),
        "type": sample.get("type"),
        "url": sample.get("url"),
        "num_saved_candidates": len(candidates),
        "candidate_svg_files": [f"{base}_c{i}.svg" for i in range(len(candidates))],
        "candidate_png_files": [f"{base}_c{i}.png" for i in range(len(candidates))] if args.save_png else [],
    }
    (output_dir / f"{base}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_sample_outputs(
    sample: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    base = sample_base_name(sample["index"])
    save_prompt_text(sample, output_dir)

    for candidate_index, candidate in enumerate(candidates):
        svg_path = output_dir / f"{base}_c{candidate_index}.svg"
        svg_path.write_text(candidate["svg"], encoding="utf-8")

        if args.save_png:
            png_image = candidate.get("png_image")
            if png_image is None:
                png_image = render_svg_to_image(candidate["svg"])
            if png_image is not None:
                png_image.save(output_dir / f"{base}_c{candidate_index}.png")

        if args.save_raw_output:
            (output_dir / f"{base}_c{candidate_index}_raw.txt").write_text(
                candidate["raw_output"],
                encoding="utf-8",
            )

    save_sample_meta(sample, candidates, output_dir, args)


def resolve_candidate_batch_size(args: argparse.Namespace) -> int:
    if args.candidate_batch_size is None:
        return args.num_candidates
    if args.candidate_batch_size <= 0:
        raise ValueError("--candidate-batch-size must be >= 1.")
    return min(args.candidate_batch_size, args.num_candidates)


def build_sampling_params(args: argparse.Namespace, candidate_batch_size: Optional[int] = None) -> SamplingParams:
    effective_candidate_batch = (
        resolve_candidate_batch_size(args) if candidate_batch_size is None else candidate_batch_size
    )
    return SamplingParams(
        n=effective_candidate_batch + args.extra_candidates_buffer,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )


def resolve_tensor_parallel_size(requested_size: int) -> int:
    visible_gpus = torch.cuda.device_count()
    if visible_gpus == 0:
        raise RuntimeError("No CUDA GPU is visible to the current process.")
    if requested_size > visible_gpus:
        raise ValueError(
            f"Requested tensor_parallel_size={requested_size}, but only {visible_gpus} GPU(s) are visible. "
            "Please adjust --tensor-parallel-size or CUDA_VISIBLE_DEVICES."
        )
    return requested_size


def main() -> None:
    args = parse_args()
    system_prompt = load_system_prompt(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Qwen/Qwen2.5-VL vLLM benchmark inference")
    print("=" * 80)
    print(f"Model path            : {args.model_path}")
    print(f"Input parquet         : {args.input_path}")
    print(f"Output dir            : {args.output_dir}")
    print(f"Prompt style          : {args.prompt_style}")
    print(f"Prompt template       : {args.prompt_template}")
    print(f"Tensor parallel size  : {args.tensor_parallel_size}")
    print(f"Batch size            : {args.batch_size}")
    print(f"Num candidates        : {args.num_candidates}")
    print(f"Candidate batch size  : {resolve_candidate_batch_size(args)}")
    print(f"Extra buffer          : {args.extra_candidates_buffer}")
    print(f"Max new tokens        : {args.max_new_tokens}")
    print(f"Temperature           : {args.temperature}")
    print(f"Top-p                 : {args.top_p}")
    print(f"Top-k                 : {args.top_k}")
    print(f"Repetition penalty    : {args.repetition_penalty}")
    print(f"Resume                : {args.resume}")
    print(f"Save PNG              : {args.save_png}")
    print(f"Validate              : {not args.no_validate}")
    print("=" * 80)

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not os.path.isfile(args.input_path):
        raise FileNotFoundError(f"Input parquet not found: {args.input_path}")

    samples = load_text2svg_samples(args.input_path)
    print(f"Loaded {len(samples)} text2svg samples from benchmark.")

    if args.sample_indices is not None:
        requested = set(args.sample_indices)
        samples = [sample for sample in samples if sample["index"] in requested]
        print(f"Filtered to {len(samples)} requested samples.")

    if args.resume:
        remaining = [sample for sample in samples if not is_sample_complete(sample, output_dir, args)]
        skipped = len(samples) - len(remaining)
        print(f"Resume mode: skipping {skipped} completed samples, running {len(remaining)} samples.")
        samples = remaining

    if not samples:
        print("Nothing to do.")
        return

    tp_size = resolve_tensor_parallel_size(args.tensor_parallel_size)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Preparing prompts...")
    prompt_style = resolve_prompt_style(args.model_path, args.prompt_style)
    print(f"Resolved prompt style : {prompt_style}")
    prompts = [
        build_prompt(
            tokenizer=tokenizer,
            user_prompt=str(sample["text"]),
            prompt_style=prompt_style,
            prompt_template=args.prompt_template,
            system_prompt=system_prompt,
        )
        for sample in samples
    ]

    candidate_batch_size = resolve_candidate_batch_size(args)
    max_num_seqs = max(args.batch_size, args.batch_size * (candidate_batch_size + args.extra_candidates_buffer))
    print("Loading vLLM model...")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tokenizer_mode="auto",
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # This benchmark is text-only, so do not reserve MM slots.
        limit_mm_per_prompt={"image": 0, "video": 0},
    )

    sampling_params = build_sampling_params(args, candidate_batch_size=candidate_batch_size)

    total_saved = 0
    total_partial = 0
    total_failed = 0

    for start in tqdm(range(0, len(samples), args.batch_size), desc="Batches"):
        batch_samples = samples[start : start + args.batch_size]
        batch_prompts = prompts[start : start + args.batch_size]
        collected_candidates: List[List[Dict[str, Any]]] = [[] for _ in batch_samples]
        last_outputs = [None for _ in batch_samples]
        max_rounds = max(1, math.ceil(args.num_candidates / candidate_batch_size) + 2)

        for _ in range(max_rounds):
            pending_indices = [
                idx for idx, candidates in enumerate(collected_candidates) if len(candidates) < args.num_candidates
            ]
            if not pending_indices:
                break

            pending_prompts = [batch_prompts[idx] for idx in pending_indices]
            outputs = llm.generate(pending_prompts, sampling_params)

            for local_idx, request_output in zip(pending_indices, outputs):
                last_outputs[local_idx] = request_output
                need = args.num_candidates - len(collected_candidates[local_idx])
                if need <= 0:
                    continue
                new_candidates = collect_candidates(
                    request_output=request_output,
                    target_count=need,
                    validate=not args.no_validate,
                )
                collected_candidates[local_idx].extend(new_candidates)

        for sample, request_output, candidates in zip(batch_samples, last_outputs, collected_candidates):
            candidates = candidates[: args.num_candidates]

            if candidates:
                save_sample_outputs(sample, candidates, output_dir, args)
                if len(candidates) == args.num_candidates:
                    total_saved += 1
                else:
                    total_partial += 1
                tqdm.write(
                    f"[OK] {sample_base_name(sample['index'])}: saved {len(candidates)} candidate(s)"
                )
            else:
                save_prompt_text(sample, output_dir)
                save_sample_meta(sample, [], output_dir, args)
                if request_output is not None:
                    save_failed_raw_outputs(sample, request_output, output_dir)
                total_failed += 1
                tqdm.write(
                    f"[FAIL] {sample_base_name(sample['index'])}: no extractable SVG candidate"
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("=" * 80)
    print("Inference completed.")
    print(f"Fully saved samples    : {total_saved}")
    print(f"Partially saved samples: {total_partial}")
    print(f"Failed samples         : {total_failed}")
    print(f"Output dir             : {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
