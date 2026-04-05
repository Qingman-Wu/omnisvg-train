#!/usr/bin/env python3
"""
Run Qwen/Qwen2.5-VL baseline models on MMSVGBench text2svg with plain Transformers.

This script is intended as a fallback for failed or partial samples from the vLLM
benchmark pipeline. It keeps the same output layout so it can reuse the existing
output directory with --resume.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None

try:
    import pandas as pd
except ImportError:
    pd = None

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL_PATH = "/mnt/data/wuqingman/models/Qwen/Qwen2.5-VL-72B-Instruct"
DEFAULT_INPUT_PATH = "/mnt/data/wuqingman/datasets/OmniSVG/MMSVGBench/data/text2svg-00000-of-00001.parquet"
DEFAULT_OUTPUT_DIR = "/mnt/data3/wuqingman/omnisvg-train/baseline_qwen25vl72b_mmsvgbench_hf"
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
        description="Run Qwen/Qwen2.5-VL baseline on MMSVGBench text2svg with plain Transformers."
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
        "--device-map",
        type=str,
        default="auto",
        help="Transformers device_map, e.g. auto / balanced / sequential.",
    )
    parser.add_argument(
        "--max-memory-per-gpu-gib",
        type=int,
        default=None,
        help="Optional per-GPU max memory when loading with device_map.",
    )
    parser.add_argument(
        "--cpu-max-memory-gib",
        type=int,
        default=None,
        help="Optional CPU max memory for device_map offload.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Optional attention backend override.",
    )
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument(
        "--max-attempts-per-sample",
        type=int,
        default=12,
        help="Maximum generation attempts for each sample, including partial retries.",
    )
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
        help="Also save accepted and failed raw generations for debugging.",
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
        help="Skip SVG rendering validation and accept the first extractable candidates.",
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

        samples.append(
            {
                "index": len(samples),
                "text": prompt,
                "id": row.get("id"),
                "task_type": row.get("task_type"),
                "type": row.get("type"),
                "url": row.get("url"),
            }
        )

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


def normalize_candidate(raw_output: str, validate: bool) -> Optional[Dict[str, Any]]:
    svg_text = wrap_fragment_if_needed(extract_svg_code(raw_output))
    if not svg_text or len(svg_text) < MIN_SVG_LENGTH:
        return None
    if "<svg" not in svg_text.lower():
        return None

    png_image = render_svg_to_image(svg_text)
    if validate:
        if png_image is None:
            return None
        if np.array(png_image).mean() > EMPTY_THRESHOLD:
            return None

    return {
        "svg": svg_text,
        "raw_output": raw_output.strip(),
        "png_image": png_image,
    }


def list_existing_candidate_paths(output_dir: Path, sample_index: int) -> List[Path]:
    base = sample_base_name(sample_index)
    return sorted(output_dir.glob(f"{base}_c*.svg"))


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


def save_failed_raw_output(sample: Dict[str, Any], output_dir: Path, attempt: int, raw_output: str) -> None:
    base = sample_base_name(sample["index"])
    raw_path = output_dir / f"{base}_failed_attempt_{attempt:02d}.txt"
    raw_path.write_text(raw_output, encoding="utf-8")


def save_sample_meta(sample: Dict[str, Any], num_saved_candidates: int, output_dir: Path, args: argparse.Namespace) -> None:
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
        "num_saved_candidates": num_saved_candidates,
        "candidate_svg_files": [f"{base}_c{i}.svg" for i in range(num_saved_candidates)],
        "candidate_png_files": [f"{base}_c{i}.png" for i in range(num_saved_candidates)] if args.save_png else [],
    }
    (output_dir / f"{base}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_existing_pngs(sample: Dict[str, Any], output_dir: Path, args: argparse.Namespace) -> None:
    if not args.save_png:
        return

    base = sample_base_name(sample["index"])
    for idx, svg_path in enumerate(list_existing_candidate_paths(output_dir, sample["index"])):
        png_path = output_dir / f"{base}_c{idx}.png"
        if png_path.exists():
            continue
        try:
            svg_text = svg_path.read_text(encoding="utf-8")
        except Exception:
            continue
        png_image = render_svg_to_image(svg_text)
        if png_image is not None:
            png_image.save(png_path)


def save_candidate(sample: Dict[str, Any], candidate: Dict[str, Any], candidate_index: int, output_dir: Path, args: argparse.Namespace) -> None:
    base = sample_base_name(sample["index"])
    svg_path = output_dir / f"{base}_c{candidate_index}.svg"
    svg_path.write_text(candidate["svg"], encoding="utf-8")

    if args.save_png:
        png_image = candidate.get("png_image")
        if png_image is None:
            png_image = render_svg_to_image(candidate["svg"])
        if png_image is not None:
            png_image.save(output_dir / f"{base}_c{candidate_index}.png")

    if args.save_raw_output:
        raw_path = output_dir / f"{base}_c{candidate_index}_raw.txt"
        raw_path.write_text(candidate["raw_output"], encoding="utf-8")


def resolve_torch_dtype(dtype_name: str):
    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def build_max_memory_map(args: argparse.Namespace) -> Optional[Dict[Any, str]]:
    if args.max_memory_per_gpu_gib is None or not torch.cuda.is_available():
        return None

    max_memory: Dict[Any, str] = {
        gpu_idx: f"{int(args.max_memory_per_gpu_gib)}GiB" for gpu_idx in range(torch.cuda.device_count())
    }
    if args.cpu_max_memory_gib is not None:
        max_memory["cpu"] = f"{int(args.cpu_max_memory_gib)}GiB"
    return max_memory


def infer_model_cls(model_path: str):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    architectures = list(getattr(config, "architectures", []) or [])
    if "Qwen2_5_VLForConditionalGeneration" in architectures and Qwen2_5_VLForConditionalGeneration is not None:
        return Qwen2_5_VLForConditionalGeneration
    if "Qwen2VLForConditionalGeneration" in architectures and Qwen2VLForConditionalGeneration is not None:
        return Qwen2VLForConditionalGeneration
    return AutoModelForImageTextToText


def load_model_and_tokenizer(args: argparse.Namespace):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cls = infer_model_cls(args.model_path)
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
    }

    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    max_memory = build_max_memory_map(args)
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print("Loading Transformers model...")
    model = model_cls.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    return model, tokenizer


def get_model_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        cuda_devices = []
        for location in model.hf_device_map.values():
            if isinstance(location, str) and location.startswith("cuda:"):
                try:
                    cuda_devices.append(int(location.split(":", 1)[1]))
                except Exception:
                    continue
        if cuda_devices:
            return torch.device(f"cuda:{min(cuda_devices)}")
        if any(location == "cpu" for location in model.hf_device_map.values()):
            return torch.device("cpu")
    return next(model.parameters()).device


def build_generation_kwargs(tokenizer: AutoTokenizer, args: argparse.Namespace) -> Dict[str, Any]:
    do_sample = args.temperature > 0
    kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
        if args.top_k > 0:
            kwargs["top_k"] = args.top_k
    return kwargs


def generate_raw_output(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    input_device: torch.device,
    generation_kwargs: Dict[str, Any],
    seed: int,
) -> str:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(**encoded, **generation_kwargs)

    generated_ids = outputs[:, prompt_len:]
    raw_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return raw_text.strip()


def process_sample(
    sample: Dict[str, Any],
    prompt: str,
    model,
    tokenizer: AutoTokenizer,
    input_device: torch.device,
    generation_kwargs: Dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> str:
    save_prompt_text(sample, output_dir)
    ensure_existing_pngs(sample, output_dir, args)

    existing_svg_paths = list_existing_candidate_paths(output_dir, sample["index"])
    seen_svgs = set()
    for path in existing_svg_paths:
        try:
            seen_svgs.add(path.read_text(encoding="utf-8"))
        except Exception:
            continue

    num_existing = len(existing_svg_paths)
    if num_existing >= args.num_candidates:
        save_sample_meta(sample, args.num_candidates, output_dir, args)
        return "done"

    attempts = 0
    while num_existing < args.num_candidates and attempts < args.max_attempts_per_sample:
        attempts += 1
        seed = args.seed + sample["index"] * 1000 + attempts
        raw_output = generate_raw_output(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            input_device=input_device,
            generation_kwargs=generation_kwargs,
            seed=seed,
        )
        candidate = normalize_candidate(raw_output, validate=not args.no_validate)
        if candidate is None:
            if args.save_raw_output:
                save_failed_raw_output(sample, output_dir, attempts, raw_output)
            continue
        if candidate["svg"] in seen_svgs:
            if args.save_raw_output:
                save_failed_raw_output(sample, output_dir, attempts, raw_output)
            continue

        save_candidate(sample, candidate, num_existing, output_dir, args)
        seen_svgs.add(candidate["svg"])
        num_existing += 1

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_sample_meta(sample, num_existing, output_dir, args)
    if num_existing >= args.num_candidates:
        return "done"
    if num_existing > 0:
        return "partial"
    return "failed"


def main() -> None:
    args = parse_args()
    system_prompt = load_system_prompt(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Qwen/Qwen2.5-VL Transformers benchmark inference")
    print("=" * 80)
    print(f"Model path            : {args.model_path}")
    print(f"Input parquet         : {args.input_path}")
    print(f"Output dir            : {args.output_dir}")
    print(f"Prompt style          : {args.prompt_style}")
    print(f"Prompt template       : {args.prompt_template}")
    print(f"Device map            : {args.device_map}")
    print(f"Torch dtype           : {args.torch_dtype}")
    print(f"Max memory / GPU GiB  : {args.max_memory_per_gpu_gib}")
    print(f"CPU max memory GiB    : {args.cpu_max_memory_gib}")
    print(f"Num candidates        : {args.num_candidates}")
    print(f"Max attempts/sample   : {args.max_attempts_per_sample}")
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

    model, tokenizer = load_model_and_tokenizer(args)
    input_device = get_model_input_device(model)
    generation_kwargs = build_generation_kwargs(tokenizer, args)

    print("Preparing prompts...")
    prompt_style = resolve_prompt_style(args.model_path, args.prompt_style)
    print(f"Resolved prompt style : {prompt_style}")
    prompts = {
        sample["index"]: build_prompt(
            tokenizer=tokenizer,
            user_prompt=str(sample["text"]),
            prompt_style=prompt_style,
            prompt_template=args.prompt_template,
            system_prompt=system_prompt,
        )
        for sample in samples
    }

    total_saved = 0
    total_partial = 0
    total_failed = 0

    for sample in tqdm(samples, desc="Samples"):
        status = process_sample(
            sample=sample,
            prompt=prompts[sample["index"]],
            model=model,
            tokenizer=tokenizer,
            input_device=input_device,
            generation_kwargs=generation_kwargs,
            output_dir=output_dir,
            args=args,
        )
        existing_count = len(list_existing_candidate_paths(output_dir, sample["index"]))
        if status == "done":
            total_saved += 1
            tqdm.write(f"[OK] {sample_base_name(sample['index'])}: saved {existing_count} candidate(s)")
        elif status == "partial":
            total_partial += 1
            tqdm.write(f"[PARTIAL] {sample_base_name(sample['index'])}: saved {existing_count} candidate(s)")
        else:
            total_failed += 1
            tqdm.write(f"[FAIL] {sample_base_name(sample['index'])}: no extractable SVG candidate")

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
