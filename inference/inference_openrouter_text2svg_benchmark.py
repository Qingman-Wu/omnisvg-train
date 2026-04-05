#!/usr/bin/env python3
"""
Run OpenRouter API models on MMSVGBench text2svg benchmark.

Expected output layout (identical to the vLLM baseline):
  sample_0000.txt          (prompt text)
  sample_0000_meta.json    (metadata)
  sample_0000_c0.svg       (candidate 0)
  sample_0000_c0.png       (rendered candidate 0)
  ...
  sample_0000_c4.svg
  sample_0000_c4.png

Usage:
  python inference_openrouter_text2svg_benchmark.py \
      --model google/gemini-2.0-flash-001 \
      --output-dir /path/to/output \
      --save-png --resume
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    pd = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_INPUT_PATH = (
    "/mnt/data/wuqingman/datasets/OmniSVG/MMSVGBench/data/"
    "text2svg-00000-of-00001.parquet"
)
DEFAULT_OUTPUT_ROOT = "/mnt/data3/wuqingman/omnisvg-train"
DEFAULT_RENDER_SIZE = 512
MIN_SVG_LENGTH = 20
EMPTY_THRESHOLD = 250
DEFAULT_VIEWBOX = 200

OPENROUTER_MODELS = {
    # OpenAI
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-5.4": "openai/gpt-5.4",
    # Anthropic
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
    "claude-4-sonnet": "anthropic/claude-4-sonnet",
    # Google
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gemini-2.5-flash": "google/gemini-2.5-flash-preview",
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite": "google/gemini-3.1-flash-lite-preview",
    # xAI
    "grok-4": "x-ai/grok-4",
    "grok-4.20": "x-ai/grok-4.20-beta",
    # DeepSeek
    "deepseek-v3": "deepseek/deepseek-chat-v3-0324",
    "deepseek-v3.1": "deepseek/deepseek-chat-v3.1",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "deepseek-r1": "deepseek/deepseek-r1",
    # Qwen
    "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
    "qwen3-235b": "qwen/qwen3-235b-a22b",
    "qwen3-max": "qwen/qwen3-max",
    "qwen3-max-thinking": "qwen/qwen3-max-thinking",
    "qwen3.5-397b": "qwen/qwen3.5-397b-a17b",
    # Zhipu / GLM
    "glm-5": "z-ai/glm-5",
    "glm-5-turbo": "z-ai/glm-5-turbo",
    # Xiaomi / MiMo
    "mimo-v2-flash": "xiaomi/mimo-v2-flash",
    # Moonshot / Kimi
    "kimi-k2.5": "moonshotai/kimi-k2.5",
    # Meta
    "llama-4-maverick": "meta-llama/llama-4-maverick",
}

TOAPIS_MODELS = {
    # OpenAI
    "gpt-5.4": "gpt-5.4-official",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5": "gpt-5",
    # Anthropic
    "claude-4-sonnet": "claude-sonnet-4-6",
    "claude-4-opus": "claude-opus-4-6",
    "claude-4-haiku": "claude-haiku-4-5",
    # Google
    "gemini-2.5-pro": "gemini-2.5-pro-official",
    "gemini-2.5-flash": "gemini-2.5-flash-official",
    "gemini-3.1-pro": "gemini-3.1-pro-preview-official",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview-official",
}

YUNWU_MODELS = {
    "claude-4-sonnet": "claude-sonnet-4-6",
    "claude-4-opus": "claude-opus-4-6",
    "glm-5": "glm-5",
    "kimi-k2.5": "kimi-k2.5",
}

LEMON_MODELS = {
    "gemini-3.1-pro": "[L]gemini-3.1-pro-preview",
    "gemini-3-pro": "[L]gemini-3-pro-preview",
    "gemini-3-flash": "[L]gemini-3-flash-preview",
    "gemini-2.5-pro": "[L]gemini-2.5-pro",
}

APIBOX_MODELS = {
    "gpt-5.4": "gpt-5.4",
    "gpt-5": "gpt-5",
    "gemini-3.1-pro": "gemini-3.1-pro-high",
    "gemini-3-flash": "gemini-3-flash",
    "glm-5": "glm-5",
    "kimi-k2.5": "kimi-k2.5",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a benchmark-grade SVG author for text-to-SVG generation. "
    "Convert each natural-language description into one complete, self-contained "
    "SVG illustration. Return exactly one final answer: a single ```svg``` code "
    "block containing the full SVG, and nothing else.\n"
    "Hard requirements:\n"
    "1. The SVG must be valid XML and must start with "
    '`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">` '
    "or an equivalent full `<svg ...>` root using the same namespace and a "
    "200x200 viewBox.\n"
    "2. Generate a complete composition, not fragments. The main subject should "
    "be large, centered, and clearly recognizable, typically occupying about "
    "60% to 85% of the canvas.\n"
    "3. Use only reliable SVG elements and attributes: `svg`, `g`, `path`, "
    "`rect`, `circle`, `ellipse`, `polygon`, `polyline`, `line`, `fill`, "
    "`stroke`, `stroke-width`, `opacity`, and `transform`.\n"
    "4. Do not use `text`, CSS, JavaScript, animation, filters, masks, "
    "clipPath, pattern, foreignObject, raster images, or external resources.\n"
    "5. Prefer clean geometry, closed shapes, solid fills, simple strokes, "
    "and readable layering. Use concise paths and avoid tiny scattered marks, "
    "decorative noise, or nearly invisible details.\n"
    "6. Faithfully reflect the described subject, pose, color, and spatial "
    "layout. If the description is complex, simplify it into the most "
    "recognizable visual composition while preserving the core semantics.\n"
    "7. Ensure the result is renderable, non-empty, and not a blank or "
    "near-blank canvas.\n"
    "8. Do not output explanations, comments, reasoning, or Markdown outside "
    "the single ```svg``` block."
)

THINK_SVG_SYSTEM_PROMPT = (
    "You are a benchmark-grade SVG author for text-to-SVG generation. "
    "First think briefly about composition, geometry, color, and layering. "
    "Then output the final result in exactly this format:\n"
    "<think>\nbrief plan\n</think>\n```svg\n"
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">...</svg>\n'
    "```\n"
    "Hard requirements:\n"
    "1. The final SVG must be valid, self-contained, renderable, and non-empty.\n"
    "2. The main subject should be large, centered, and visually clear.\n"
    "3. Use only reliable SVG elements and attributes: `svg`, `g`, `path`, "
    "`rect`, `circle`, `ellipse`, `polygon`, `polyline`, `line`, `fill`, "
    "`stroke`, `stroke-width`, `opacity`, and `transform`.\n"
    "4. Do not use `text`, CSS, JavaScript, animation, filters, masks, "
    "clipPath, pattern, foreignObject, raster images, or external resources.\n"
    "5. Keep the drawing concise and structurally clean; if the scene is "
    "complex, simplify it into a strong recognizable composition.\n"
    "6. After the closing code fence, output nothing."
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run OpenRouter API models on MMSVGBench text2svg."
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model identifier. Use a short alias (e.g. 'gpt-4o', 'claude-3.5-sonnet') "
            "or a full OpenRouter model string (e.g. 'openai/gpt-4o')."
        ),
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key. Falls back to OPENROUTER_API_KEY env var.",
    )
    p.add_argument(
        "--api-base",
        type=str,
        default=None,
        help=(
            "Base URL for the chat completions endpoint. "
            "Default: https://openrouter.ai/api/v1/chat/completions. "
            "For other platforms pass e.g. https://toapis.com/v1/chat/completions"
        ),
    )
    p.add_argument("--input-path", type=str, default=DEFAULT_INPUT_PATH)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Auto-derived from model name if not set.",
    )
    p.add_argument(
        "--prompt-template",
        type=str,
        default="code_only",
        choices=["code_only", "think_svg"],
    )
    p.add_argument("--num-candidates", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max parallel API requests.")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-delay", type=float, default=2.0,
                   help="Base delay (seconds) between retries (exponential back-off).")
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--save-png", action="store_true", default=True)
    p.add_argument("--no-save-png", dest="save_png", action="store_false")
    p.add_argument("--save-raw-output", action="store_true", default=False)
    p.add_argument(
        "--sample-indices", type=int, nargs="+", default=None,
        help="Optional explicit sample indices to run.",
    )
    p.add_argument(
        "--no-validate", action="store_true", default=False,
        help="Skip SVG rendering validation.",
    )
    p.add_argument(
        "--list-models", action="store_true", default=False,
        help="List available model aliases and exit.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading (identical to vLLM reference)
# ---------------------------------------------------------------------------
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
        raise ValueError(
            "Input parquet must contain one of: text / description / prompt"
        )

    samples: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        prompt = row.get(text_column)
        if prompt is None:
            continue
        prompt = str(prompt).strip()
        if not prompt:
            continue
        samples.append({
            "index": len(samples),
            "text": prompt,
            "id": row.get("id"),
            "task_type": row.get("task_type"),
            "type": row.get("type"),
            "url": row.get("url"),
        })
    return samples


# ---------------------------------------------------------------------------
# SVG extraction & rendering (shared with vLLM reference)
# ---------------------------------------------------------------------------
def sample_base_name(idx: int) -> str:
    return f"sample_{idx:04d}"


def strip_reasoning(raw: str) -> str:
    return re.sub(
        r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def extract_svg_code(raw: str) -> str:
    text = strip_reasoning(raw)

    for pattern in [
        r"```svg\s*(.*?)```",
        r"```xml\s*(.*?)```",
        r"```html\s*(.*?)```",
        r"```\s*(.*?)```",
    ]:
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()

    m = re.search(r"(<svg[\s\S]*?</svg>)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return text.strip()


def wrap_fragment_if_needed(svg: str) -> str:
    svg = svg.strip()
    if not svg:
        return svg
    if "<svg" in svg.lower():
        return svg
    if any(
        tag in svg.lower()
        for tag in ("<path", "<g", "<rect", "<circle", "<ellipse", "<polygon", "<line")
    ):
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {DEFAULT_VIEWBOX} {DEFAULT_VIEWBOX}">{svg}</svg>'
        )
    return svg


def render_svg_to_image(
    svg_str: str, size: int = DEFAULT_RENDER_SIZE
) -> Optional[Image.Image]:
    try:
        import cairosvg

        png_data = cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            output_width=size,
            output_height=size,
        )
        img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_user_prompt(text: str, template: str) -> str:
    if template == "think_svg":
        return (
            "Create a single self-contained SVG illustration for the "
            "description below.\n"
            "Use a 200x200 canvas. First think briefly, then output one "
            "full SVG in a single ```svg``` block.\n\n"
            f"Description: {text}"
        )
    return (
        "Create a single self-contained SVG illustration for the "
        "description below.\n"
        "Use a 200x200 canvas and return exactly one complete SVG in a "
        "single ```svg``` block.\n\n"
        f"Description: {text}"
    )


def get_system_prompt(args: argparse.Namespace) -> str:
    if args.prompt_template == "think_svg":
        return THINK_SVG_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------
def resolve_model(model_arg: str, api_base: str = "") -> str:
    if "toapis.com" in api_base:
        if model_arg in TOAPIS_MODELS:
            return TOAPIS_MODELS[model_arg]
    if "apibox" in api_base:
        if model_arg in APIBOX_MODELS:
            return APIBOX_MODELS[model_arg]
    if "yunwu" in api_base:
        if model_arg in YUNWU_MODELS:
            return YUNWU_MODELS[model_arg]
    if "lemonapi" in api_base:
        if model_arg in LEMON_MODELS:
            return LEMON_MODELS[model_arg]
    if "qinzhiai" in api_base:
        if model_arg in TOAPIS_MODELS:
            return TOAPIS_MODELS[model_arg]
    if model_arg in OPENROUTER_MODELS:
        return OPENROUTER_MODELS[model_arg]
    return model_arg


def model_short_name(model_id: str) -> str:
    """Derive a filesystem-safe short name from the OpenRouter model ID."""
    short = model_id.replace("/", "_").replace(":", "_").replace(".", "-")
    return short


# ---------------------------------------------------------------------------
# HTTP session with connection-level retries
# ---------------------------------------------------------------------------
_thread_local = __import__("threading").local()


def _get_session() -> requests.Session:
    """Return a per-thread Session with automatic connection retries."""
    sess = getattr(_thread_local, "http_session", None)
    if sess is None:
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=4,
            pool_maxsize=4,
        )
        sess = requests.Session()
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _thread_local.http_session = sess
    return sess


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int,
    retry_delay: float,
    api_base: str = OPENROUTER_API_URL,
) -> Optional[str]:
    """Single API call -> single completion string (or None on failure)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter.ai" in api_base:
        headers["HTTP-Referer"] = "https://github.com/omnisvg"
        headers["X-Title"] = "MMSVGBench text2svg"

    is_claude = "claude" in model.lower()

    if is_claude:
        payload = {
            "model": model,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    else:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

    session = _get_session()

    for attempt in range(max_retries):
        try:
            resp = session.post(
                api_base,
                headers=headers,
                json=payload,
                timeout=180,
            )
            if resp.status_code == 429:
                wait = retry_delay * (2 ** attempt)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content
        except requests.exceptions.HTTPError as e:
            body = ""
            try:
                body = resp.json().get("error", {}).get("message", resp.text[:200])
            except Exception:
                body = resp.text[:200]
            if resp.status_code == 403:
                print(f"  [API 403] {body}")
                return None
            if resp.status_code in (500, 502, 503):
                wait = retry_delay * (2 ** attempt)
                print(f"  [API {resp.status_code}] {body}, retry in {wait:.1f}s")
                time.sleep(wait)
                continue
            print(f"  [API error {resp.status_code}] {body}")
            return None
        except Exception as e:
            wait = retry_delay * (2 ** attempt)
            print(f"  [Request error] {e}, retry in {wait:.1f}s")
            time.sleep(wait)

    print("  [API error] max retries exceeded")
    return None


# ---------------------------------------------------------------------------
# Candidate generation for one sample
# ---------------------------------------------------------------------------
def generate_candidates_for_sample(
    sample: Dict[str, Any],
    api_key: str,
    model: str,
    system_prompt: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    user_prompt = build_user_prompt(sample["text"], args.prompt_template)
    candidates: List[Dict[str, Any]] = []

    extra_budget = args.num_candidates + 4
    for attempt_idx in range(extra_budget):
        if len(candidates) >= args.num_candidates:
            break

        raw = call_openrouter(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            api_base=args._api_base,
        )
        if raw is None:
            continue

        svg = wrap_fragment_if_needed(extract_svg_code(raw))
        if not svg or len(svg) < MIN_SVG_LENGTH:
            continue
        if "<svg" not in svg.lower():
            continue

        png_image = render_svg_to_image(svg)
        if not args.no_validate:
            if png_image is None:
                continue
            if np.array(png_image).mean() > EMPTY_THRESHOLD:
                continue

        candidates.append({
            "svg": svg,
            "raw_output": raw,
            "png_image": png_image,
        })

    return candidates[: args.num_candidates]


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def is_sample_complete(
    sample: Dict[str, Any], output_dir: Path, args: argparse.Namespace
) -> bool:
    base = sample_base_name(sample["index"])
    txt_path = output_dir / f"{base}.txt"
    if not txt_path.exists():
        return False
    for i in range(args.num_candidates):
        if not (output_dir / f"{base}_c{i}.svg").exists():
            return False
    if args.save_png:
        for i in range(args.num_candidates):
            if not (output_dir / f"{base}_c{i}.png").exists():
                return False
    return True


def save_sample_outputs(
    sample: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    base = sample_base_name(sample["index"])
    (output_dir / f"{base}.txt").write_text(
        str(sample["text"]), encoding="utf-8"
    )

    for ci, cand in enumerate(candidates):
        (output_dir / f"{base}_c{ci}.svg").write_text(
            cand["svg"], encoding="utf-8"
        )
        if args.save_png:
            png_img = cand.get("png_image")
            if png_img is None:
                png_img = render_svg_to_image(cand["svg"])
            if png_img is not None:
                png_img.save(output_dir / f"{base}_c{ci}.png")
        if args.save_raw_output:
            (output_dir / f"{base}_c{ci}_raw.txt").write_text(
                cand["raw_output"], encoding="utf-8"
            )

    meta = {
        "index": sample["index"],
        "id": sample.get("id"),
        "text": sample.get("text"),
        "task_type": sample.get("task_type"),
        "type": sample.get("type"),
        "url": sample.get("url"),
        "model": args._resolved_model,
        "num_saved_candidates": len(candidates),
        "candidate_svg_files": [f"{base}_c{i}.svg" for i in range(len(candidates))],
        "candidate_png_files": (
            [f"{base}_c{i}.png" for i in range(len(candidates))]
            if args.save_png
            else []
        ),
    }
    (output_dir / f"{base}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_failed_sample(
    sample: Dict[str, Any], output_dir: Path, args: argparse.Namespace
) -> None:
    base = sample_base_name(sample["index"])
    (output_dir / f"{base}.txt").write_text(
        str(sample["text"]), encoding="utf-8"
    )
    meta = {
        "index": sample["index"],
        "id": sample.get("id"),
        "text": sample.get("text"),
        "model": args._resolved_model,
        "num_saved_candidates": 0,
        "candidate_svg_files": [],
        "candidate_png_files": [],
        "error": "no_valid_candidates",
    }
    (output_dir / f"{base}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Worker for thread pool
# ---------------------------------------------------------------------------
def process_one_sample(
    sample: Dict[str, Any],
    api_key: str,
    model: str,
    system_prompt: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    idx = sample["index"]
    try:
        candidates = generate_candidates_for_sample(
            sample, api_key, model, system_prompt, args
        )
        if candidates:
            save_sample_outputs(sample, candidates, output_dir, args)
            return {
                "index": idx,
                "status": "ok" if len(candidates) == args.num_candidates else "partial",
                "num_candidates": len(candidates),
            }
        else:
            save_failed_sample(sample, output_dir, args)
            return {"index": idx, "status": "fail", "num_candidates": 0}
    except Exception as e:
        traceback.print_exc()
        return {"index": idx, "status": "error", "error": str(e), "num_candidates": 0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if args.list_models:
        print("Available model aliases:")
        for alias, full_id in sorted(OPENROUTER_MODELS.items()):
            print(f"  {alias:25s} -> {full_id}")
        return

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "Provide --api-key or set OPENROUTER_API_KEY env var."
        )

    api_base = args.api_base or OPENROUTER_API_URL
    if not api_base.endswith("/chat/completions"):
        api_base = api_base.rstrip("/") + "/chat/completions"
    args._api_base = api_base

    resolved_model = resolve_model(args.model, api_base)
    args._resolved_model = resolved_model

    if args.output_dir is None:
        short = model_short_name(resolved_model)
        args.output_dir = os.path.join(
            DEFAULT_OUTPUT_ROOT,
            f"baseline_{short}_mmsvgbench_openrouter",
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = get_system_prompt(args)

    print("=" * 80)
    print("API text2svg benchmark inference")
    print("=" * 80)
    print(f"API base              : {api_base}")
    print(f"Model (resolved)      : {resolved_model}")
    print(f"Input parquet         : {args.input_path}")
    print(f"Output dir            : {args.output_dir}")
    print(f"Prompt template       : {args.prompt_template}")
    print(f"Num candidates        : {args.num_candidates}")
    print(f"Max tokens            : {args.max_tokens}")
    print(f"Temperature           : {args.temperature}")
    print(f"Top-p                 : {args.top_p}")
    print(f"Concurrency           : {args.concurrency}")
    print(f"Max retries           : {args.max_retries}")
    print(f"Resume                : {args.resume}")
    print(f"Save PNG              : {args.save_png}")
    print(f"Validate              : {not args.no_validate}")
    print("=" * 80)

    if not os.path.isfile(args.input_path):
        raise FileNotFoundError(f"Input parquet not found: {args.input_path}")

    samples = load_text2svg_samples(args.input_path)
    print(f"Loaded {len(samples)} text2svg samples from benchmark.")

    if args.sample_indices is not None:
        requested = set(args.sample_indices)
        samples = [s for s in samples if s["index"] in requested]
        print(f"Filtered to {len(samples)} requested samples.")

    if args.resume:
        remaining = [
            s for s in samples if not is_sample_complete(s, output_dir, args)
        ]
        skipped = len(samples) - len(remaining)
        print(
            f"Resume mode: skipping {skipped} completed, "
            f"running {len(remaining)} samples."
        )
        samples = remaining

    if not samples:
        print("Nothing to do.")
        return

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model": resolved_model,
                "model_alias": args.model,
                "prompt_template": args.prompt_template,
                "num_candidates": args.num_candidates,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "concurrency": args.concurrency,
                "validate": not args.no_validate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total_ok = 0
    total_partial = 0
    total_fail = 0

    pbar = tqdm(total=len(samples), desc="Samples")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                process_one_sample,
                sample,
                api_key,
                resolved_model,
                system_prompt,
                output_dir,
                args,
            ): sample
            for sample in samples
        }

        for future in as_completed(futures):
            result = future.result()
            idx = result["index"]
            status = result["status"]
            nc = result["num_candidates"]

            if status == "ok":
                total_ok += 1
                pbar.write(f"[OK]      sample_{idx:04d}: {nc} candidate(s)")
            elif status == "partial":
                total_partial += 1
                pbar.write(f"[PARTIAL] sample_{idx:04d}: {nc} candidate(s)")
            else:
                total_fail += 1
                err = result.get("error", "no valid SVG")
                pbar.write(f"[FAIL]    sample_{idx:04d}: {err}")

            pbar.update(1)

    pbar.close()

    print("=" * 80)
    print("Inference completed.")
    print(f"  Fully saved  : {total_ok}")
    print(f"  Partial      : {total_partial}")
    print(f"  Failed       : {total_fail}")
    print(f"  Output dir   : {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
