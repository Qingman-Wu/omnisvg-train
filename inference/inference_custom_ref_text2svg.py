#!/usr/bin/env python3
"""
Text2SVG with custom reference SVGs — HVM inference (multi-GPU).

Instead of automatic CLIP retrieval, the user supplies a folder of
reference SVG files.  The script:
  1. Parses each SVG → decomposes paths into groups (up to 4 per ref)
  2. Renders whole-image and per-group images
  3. Extracts post-merge features [256, 3584] via Qwen's frozen vision encoder
  4. Injects features into HVM memory and generates SVG candidates

Usage:
CUDA_VISIBLE_DEVICES=5,7 python inference/inference_custom_ref_text2svg.py \
    --prompt "A yellow SpongeBob SquarePants" \
    --ref_images_dir /mnt/a100_1_data2/wuqingman/omnisvg-train/paper_figures/custom_ref/haimian \
    --hvm_checkpoint /mnt/a100_1_data3/wuqingman/omnisvg-train/outputs_s9_full25w_top3part_12slot_nogist_edr_parttag_nozoom_last4/hvm_step_5000.pt \
    --output_dir /mnt/a100_1_data2/wuqingman/omnisvg-train/paper_figures/custom_ref_test \
    --num_candidates 5 \
    --save_png

CUDA_VISIBLE_DEVICES=0,1,2,3 python inference/inference_custom_ref_text2svg.py \
    --prompt_file prompts.txt \
    --ref_images_dir /path/to/my_ref_svgs \
    --hvm_checkpoint /path/to/hvm_step_5000.pt \
    --output_dir /mnt/a100_1_data2/wuqingman/omnisvg-train/paper_figures/custom_ref_test \
    --num_candidates 5 \
    --save_png
"""

import argparse
import gc
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from inference_hvm_s1_test import (
    SVG_CONFIG_PATH,
    SVGTokenizer,
    clear_hvm_memory,
    generate_svg,
    load_hvm_model,
    prepare_text_inputs,
    prepare_visual_prefix_inputs,
    render_svg_to_image,
    set_hvm_memory,
    split_indices,
    validate_candidate,
)

EXTRA_CANDIDATES_BUFFER = 0
TOKENS_PER_IMAGE = 256
GROUPS_PER_REFERENCE = 4
DEFAULT_VIEWBOX_SIZE = 200
IMAGE_SIZE = 448
MAX_GROUPS = 4

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
SVG_EXTENSIONS = {".svg"}


def _parse_viewbox(svg_string: str) -> Tuple[float, float, float, float]:
    """Extract viewBox from SVG string. Returns (x, y, width, height)."""
    m = re.search(r'viewBox\s*=\s*"([^"]*)"', svg_string)
    if m:
        parts = m.group(1).split()
        if len(parts) >= 4:
            return tuple(float(v) for v in parts[:4])
    m_w = re.search(r'width\s*=\s*"(\d+(?:\.\d+)?)"', svg_string)
    m_h = re.search(r'height\s*=\s*"(\d+(?:\.\d+)?)"', svg_string)
    w = float(m_w.group(1)) if m_w else DEFAULT_VIEWBOX_SIZE
    h = float(m_h.group(1)) if m_h else DEFAULT_VIEWBOX_SIZE
    return (0, 0, w, h)


def _resolve_path(path: str) -> str:
    """Auto-fix a100-1 native paths when running on a100-3."""
    if os.path.exists(path):
        return path
    for prefix, mapped in [
        ("/mnt/data3/", "/mnt/a100_1_data3/"),
        ("/mnt/data2/", "/mnt/a100_1_data2/"),
        ("/mnt/data/", "/mnt/a100_1_data/"),
    ]:
        if path.startswith(prefix):
            alt = mapped + path[len(prefix):]
            if os.path.exists(alt):
                return alt
    return path


# ============================================================================
# SVG parsing & path grouping (adapted from precompute_hvm_data.py)
# ============================================================================

def _parse_path_commands(d_str: str) -> List[Dict]:
    commands = []
    tokens = re.findall(r'([MLCQAZmlcqaz])([\s\d.,eE+-]*)', d_str)
    for cmd_char, coords_str in tokens:
        cmd_type = cmd_char.upper()
        numbers = re.findall(
            r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', coords_str
        )
        coords = [float(n) for n in numbers]
        if cmd_type == 'Z':
            commands.append({"type": "Z", "coords": []})
        elif cmd_type == 'M':
            for i in range(0, len(coords) - 1, 2):
                commands.append({"type": "M", "coords": coords[i:i + 2]})
        elif cmd_type == 'L':
            for i in range(0, len(coords) - 1, 2):
                commands.append({"type": "L", "coords": coords[i:i + 2]})
        elif cmd_type == 'C':
            for i in range(0, len(coords) - 5, 6):
                commands.append({"type": "C", "coords": coords[i:i + 6]})
        elif cmd_type == 'Q':
            for i in range(0, len(coords) - 3, 4):
                commands.append({"type": "Q", "coords": coords[i:i + 4]})
        elif cmd_type == 'A':
            for i in range(0, len(coords) - 6, 7):
                commands.append({"type": "A", "coords": coords[i:i + 7]})
    return commands


def _compute_path_bbox(
    commands: List[Dict], vb_w: float = 200, vb_h: float = 200,
) -> Tuple[float, float, float, float]:
    all_x, all_y = [], []
    for cmd in commands:
        coords = cmd["coords"]
        if cmd["type"] == "Z":
            continue
        elif cmd["type"] in ("M", "L"):
            if len(coords) >= 2:
                all_x.append(coords[0]); all_y.append(coords[1])
        elif cmd["type"] == "C":
            for i in range(0, len(coords) - 1, 2):
                all_x.append(coords[i]); all_y.append(coords[i + 1])
        elif cmd["type"] == "Q":
            for i in range(0, len(coords) - 1, 2):
                all_x.append(coords[i]); all_y.append(coords[i + 1])
        elif cmd["type"] == "A":
            if len(coords) >= 7:
                all_x.append(coords[5]); all_y.append(coords[6])
    if not all_x or not all_y:
        return (0, 0, vb_w, vb_h)
    return (min(all_x), min(all_y), max(all_x), max(all_y))


def _compute_path_complexity(
    commands: List[Dict], bbox: Tuple[float, float, float, float],
    vb_w: float = 200, vb_h: float = 200,
) -> float:
    cmd_weights = {"M": 0.5, "L": 1.0, "Q": 2.0, "C": 3.0, "A": 3.0, "Z": 0.2}
    cmd_score = sum(cmd_weights.get(c["type"], 1.0) for c in commands)
    total_area = vb_w * vb_h
    bbox_w = max(bbox[2] - bbox[0], 1.0)
    bbox_h = max(bbox[3] - bbox[1], 1.0)
    area_ratio = (bbox_w * bbox_h) / total_area
    area_weight = max(area_ratio, 0.05)
    return cmd_score * (0.5 + 0.5 * area_weight)


def parse_svg_paths(
    svg_string: str, vb_w: float = 200, vb_h: float = 200,
) -> List[Dict]:
    """Parse SVG paths, returning per-path metadata for grouping."""
    paths = []
    path_pattern = re.compile(r'<path\s+([^>]*)/?>', re.DOTALL)
    for match in path_pattern.finditer(svg_string):
        attrs_str = match.group(1)
        d_match = re.search(r'd="([^"]*)"', attrs_str)
        if not d_match:
            continue
        d_str = d_match.group(1).strip()
        if not d_str:
            continue
        fill_match = re.search(r'fill="([^"]*)"', attrs_str)
        fill = fill_match.group(1) if fill_match else "#000000"
        commands = _parse_path_commands(d_str)
        if not commands:
            continue
        bbox = _compute_path_bbox(commands, vb_w, vb_h)
        complexity = _compute_path_complexity(commands, bbox, vb_w, vb_h)
        paths.append({
            "d": d_str, "fill": fill, "commands": commands,
            "bbox": bbox, "complexity": complexity,
        })
    return paths


def _make_group(
    paths: List[Dict], indices: List[int], complexity: float,
    vb_w: float = 200, vb_h: float = 200,
) -> Dict:
    all_x0, all_y0, all_x1, all_y1 = [], [], [], []
    for i in indices:
        bbox = paths[i]["bbox"]
        all_x0.append(bbox[0]); all_y0.append(bbox[1])
        all_x1.append(bbox[2]); all_y1.append(bbox[3])
    merged_bbox = (min(all_x0), min(all_y0), max(all_x1), max(all_y1))
    bw = merged_bbox[2] - merged_bbox[0]
    bh = merged_bbox[3] - merged_bbox[1]
    pad = max(bw, bh) * 0.1
    padded_bbox = (
        max(merged_bbox[0] - pad, 0),
        max(merged_bbox[1] - pad, 0),
        min(merged_bbox[2] + pad, vb_w),
        min(merged_bbox[3] + pad, vb_h),
    )
    return {
        "path_indices": indices, "bbox": padded_bbox, "complexity": complexity,
    }


def group_paths_sequential(
    paths: List[Dict], num_groups: int,
    vb_w: float = 200, vb_h: float = 200,
) -> List[Dict]:
    """Split paths into *num_groups* groups by cumulative complexity."""
    if num_groups <= 0 or num_groups == 1 or len(paths) <= 1:
        total_c = sum(p["complexity"] for p in paths)
        return [_make_group(paths, list(range(len(paths))), total_c, vb_w, vb_h)]

    num_groups = min(num_groups, len(paths))
    complexities = [p["complexity"] for p in paths]
    remaining_complexity = sum(complexities)

    groups = []
    current_indices: List[int] = []
    current_sum = 0.0

    for i, (path, c) in enumerate(zip(paths, complexities)):
        current_indices.append(i)
        current_sum += c
        groups_still_needed = num_groups - len(groups)
        remaining_paths = len(paths) - i - 1
        target = (remaining_complexity / groups_still_needed
                  if groups_still_needed > 0 else remaining_complexity)
        force_split = (remaining_paths > 0
                       and remaining_paths == groups_still_needed - 1)
        normal_split = (current_sum >= target
                        and groups_still_needed > 1
                        and remaining_paths >= 1)
        if force_split or normal_split:
            groups.append(_make_group(paths, current_indices, current_sum, vb_w, vb_h))
            remaining_complexity -= current_sum
            current_indices = []
            current_sum = 0.0

    if current_indices:
        groups.append(_make_group(paths, current_indices, current_sum, vb_w, vb_h))
    return groups


def render_group_to_image(
    svg_string: str,
    group_path_indices: List[int],
    vb_x: float = 0, vb_y: float = 0,
    vb_w: float = 200, vb_h: float = 200,
    image_size: int = IMAGE_SIZE,
) -> Optional[Image.Image]:
    """Render only the paths in *group_path_indices*, using the actual viewBox."""
    import cairosvg

    path_pattern = re.compile(r'<path\s[^>]*?(?:/>|>\s*</path>)', re.DOTALL)
    all_path_tags = path_pattern.findall(svg_string)
    if not all_path_tags:
        return None

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{vb_x} {vb_y} {vb_w} {vb_h}" '
        f'width="{image_size}" height="{image_size}">',
        f'<rect x="{vb_x}" y="{vb_y}" width="{vb_w}" '
        f'height="{vb_h}" fill="white"/>',
    ]
    for pi in group_path_indices:
        if pi < len(all_path_tags):
            svg_lines.append(all_path_tags[pi])
    svg_lines.append('</svg>')

    try:
        png_bytes = cairosvg.svg2png(
            bytestring='\n'.join(svg_lines).encode('utf-8'),
            output_width=image_size, output_height=image_size,
        )
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception:
        return None


def render_whole_svg_to_image(
    svg_string: str, image_size: int = IMAGE_SIZE,
) -> Optional[Image.Image]:
    """Render full SVG to a PIL image."""
    import cairosvg
    try:
        png_bytes = cairosvg.svg2png(
            bytestring=svg_string.encode('utf-8'),
            output_width=image_size, output_height=image_size,
        )
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception:
        return None


# ============================================================================
# Load & decompose custom reference SVGs
# ============================================================================

def load_ref_svgs(ref_dir: str) -> List[Dict[str, Any]]:
    """Load reference SVG (and optional image) files from *ref_dir*.

    For each SVG:
      - reads the SVG source
      - renders the whole SVG → PIL image
      - parses paths → groups (up to MAX_GROUPS)
      - renders each group → PIL image
      - builds tag_meta (6-dim bbox + z-order)

    Also looks for optional ``<stem>.txt`` description files.

    Returns a list of dicts, one per reference, with keys:
      svg_str, whole_image, groups, group_images, tag_meta_rows,
      description, filename
    """
    ref_dir = Path(ref_dir)
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {ref_dir}")

    svg_files = sorted(
        e for e in ref_dir.iterdir() if e.suffix.lower() in SVG_EXTENSIONS
    )
    if not svg_files:
        raise FileNotFoundError(
            f"No .svg files found in {ref_dir}. "
            "Please place reference SVG files there."
        )

    refs: List[Dict[str, Any]] = []
    for svg_path in svg_files:
        svg_str = svg_path.read_text(encoding="utf-8")
        whole_img = render_whole_svg_to_image(svg_str)
        if whole_img is None:
            print(f"  WARNING: failed to render {svg_path.name}, skipping")
            continue

        vb_x, vb_y, vb_w, vb_h = _parse_viewbox(svg_str)
        print(f"    {svg_path.name}: viewBox={vb_x} {vb_y} {vb_w} {vb_h}")

        paths = parse_svg_paths(svg_str, vb_w, vb_h)
        num_paths = len(paths)

        if not paths:
            groups = [{"path_indices": [], "bbox": [0, 0, vb_w, vb_h],
                       "complexity": 0}]
        else:
            num_groups = min(MAX_GROUPS, num_paths)
            groups = group_paths_sequential(paths, num_groups, vb_w, vb_h)

        group_images: List[Image.Image] = []
        valid_group_indices: List[int] = []
        tag_meta_rows: List[List[float]] = []
        denom = max(num_paths - 1, 1)

        for g_idx, grp in enumerate(groups):
            pi_list = grp["path_indices"]
            if not pi_list:
                continue
            grp_img = render_group_to_image(
                svg_str, pi_list, vb_x, vb_y, vb_w, vb_h,
            )
            if grp_img is None:
                continue
            group_images.append(grp_img)
            valid_group_indices.append(g_idx)

            bbox = grp["bbox"]
            x0, y0, x1, y1 = [float(v) for v in bbox]
            cx = ((x0 + x1) * 0.5) / vb_w
            cy = ((y0 + y1) * 0.5) / vb_h
            w = max(x1 - x0, 0.0) / vb_w
            h = max(y1 - y0, 0.0) / vb_h
            z_start = min(pi_list) / denom
            z_end = max(pi_list) / denom
            tag_meta_rows.append([cx, cy, w, h, z_start, z_end])

        txt_path = svg_path.with_suffix(".txt")
        description = ""
        if txt_path.exists():
            description = txt_path.read_text(encoding="utf-8").strip()

        refs.append({
            "svg_str": svg_str,
            "whole_image": whole_img,
            "groups": groups,
            "group_images": group_images,
            "valid_group_indices": valid_group_indices,
            "tag_meta_rows": tag_meta_rows,
            "num_paths": num_paths,
            "description": description,
            "filename": svg_path.name,
        })

    if not refs:
        raise RuntimeError("No valid reference SVGs could be rendered.")

    return refs


# ============================================================================
# Feature extraction via Qwen vision encoder
# ============================================================================

def extract_post_merge_features(
    images: List[Image.Image],
    visual_module: torch.nn.Module,
    processor,
    device: str,
) -> List[torch.Tensor]:
    """Extract post-merge features [256, 3584] per image."""
    from qwen_vl_utils import process_vision_info

    all_pixel_values = []
    all_grid_thws = []

    for img in images:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "x"},
        ]}]
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text_input], images=image_inputs, return_tensors="pt"
        )
        all_pixel_values.append(inputs["pixel_values"])
        all_grid_thws.append(inputs["image_grid_thw"])

    pixel_values = torch.cat(all_pixel_values, dim=0).to(device, dtype=torch.float16)
    grid_thw = torch.cat(all_grid_thws, dim=0).to(device)

    with torch.no_grad():
        post_merge = visual_module(pixel_values, grid_thw=grid_thw)

    results: List[torch.Tensor] = []
    for b in range(len(images)):
        start = b * TOKENS_PER_IMAGE
        end = (b + 1) * TOKENS_PER_IMAGE
        results.append(post_merge[start:end].cpu().half())

    return results


def build_custom_ref_data(
    ref_records: List[Dict[str, Any]],
    visual_module: torch.nn.Module,
    processor,
    device: str,
) -> Dict[str, Any]:
    """Build HVM memory inputs from custom reference SVGs.

    Extracts features for:
      - whole images → ref_features  (list of [256, 3584], one per ref)
      - group images → group_features (list of [256, 3584], flattened across refs)
    Also produces proper tag_meta and group_ids with correct offsets.
    """
    whole_images = [r["whole_image"] for r in ref_records]
    print(f"    Extracting whole-image features for {len(whole_images)} refs ...")
    ref_features = extract_post_merge_features(
        whole_images, visual_module, processor, device
    )

    all_group_images: List[Image.Image] = []
    all_tag_meta: List[torch.Tensor] = []
    all_group_ids: List[torch.Tensor] = []
    group_image_to_ref: List[int] = []

    for ref_rank, rec in enumerate(ref_records):
        gi_list = rec["group_images"]
        tm_rows = rec["tag_meta_rows"]
        num_valid = len(gi_list)
        if num_valid == 0:
            continue
        all_group_images.extend(gi_list)
        group_image_to_ref.extend([ref_rank] * num_valid)
        all_tag_meta.append(torch.tensor(tm_rows, dtype=torch.float32))
        local_ids = torch.arange(num_valid, dtype=torch.long)
        all_group_ids.append(local_ids + ref_rank * GROUPS_PER_REFERENCE)

    group_features: List[torch.Tensor] = []
    if all_group_images:
        print(f"    Extracting group features for {len(all_group_images)} groups ...")
        group_features = extract_post_merge_features(
            all_group_images, visual_module, processor, device
        )

    if all_tag_meta:
        tag_meta = torch.cat(all_tag_meta, dim=0)
        group_ids = torch.cat(all_group_ids, dim=0)
    else:
        tag_meta = torch.zeros(0, 6, dtype=torch.float32)
        group_ids = torch.zeros(0, dtype=torch.long)

    descriptions = [r["description"] for r in ref_records]
    ref_text = " ".join(d for d in descriptions if d)

    return {
        "ref_features": ref_features,
        "group_features": group_features,
        "tag_meta": tag_meta,
        "group_ids": group_ids,
        "ref_text": ref_text,
    }


# ============================================================================
# Save reference material to output
# ============================================================================

def save_ref_material(
    prompt_dir: Path,
    ref_records: List[Dict[str, Any]],
) -> None:
    """Save reference images (whole + groups) and SVGs into output dir."""
    refs_dir = prompt_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    for rank, rec in enumerate(ref_records):
        whole_path = refs_dir / f"ref_{rank}.png"
        if not whole_path.exists():
            rec["whole_image"].save(str(whole_path))

        svg_path = refs_dir / f"ref_{rank}.svg"
        if not svg_path.exists():
            svg_path.write_text(rec["svg_str"], encoding="utf-8")

        for gi, grp_img in enumerate(rec["group_images"]):
            grp_path = refs_dir / f"ref_{rank}_group_{gi}.png"
            if not grp_path.exists():
                grp_img.save(str(grp_path))


# ============================================================================
# Prompts
# ============================================================================

def load_prompts(path: str):
    prompts: List[str] = []
    ids: List[int] = []
    has_ids = True
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\d+)\s*:\s*(.+)$", line)
            if m:
                ids.append(int(m.group(1)))
                prompts.append(m.group(2).strip())
            else:
                has_ids = False
                prompts.append(line)
    return (ids if has_ids and ids else None, prompts)


# ============================================================================
# Temperature schedule helpers
# ============================================================================

DEFAULT_TEMPERATURE_SCHEDULE = [
    (0.0, 2),
    (0.3, 2),
    (0.5, 5),
    (0.7, 2),
    (1.0, 2),
]


def _temp_label(temp: float) -> str:
    return f"t{temp:.1f}".replace(".", "")


def _count_existing(prompt_dir: Path, prefix: str, schedule) -> int:
    total = 0
    for temp, n in schedule:
        label = _temp_label(temp)
        for ci in range(n):
            if (prompt_dir / f"{prefix}_{label}_c{ci}.svg").exists():
                total += 1
    return total


def _generate_candidates_multi_temp(
    model, input_ids, attention_mask, token_config, svg_tokenizer,
    gen_kwargs_base, no_validate, prompt_dir, prefix, save_png, schedule,
):
    all_candidates = []
    total_elapsed = 0.0

    for temp, num_cand in schedule:
        label = _temp_label(temp)
        already_done = sum(
            1 for ci in range(num_cand)
            if (prompt_dir / f"{prefix}_{label}_c{ci}.svg").exists()
        )
        if already_done >= num_cand:
            all_candidates.append((temp, num_cand, 0.0))
            continue

        gk = dict(gen_kwargs_base)
        gk["temperature"] = temp if temp > 0 else 1e-7
        if temp == 0:
            gk["top_k"] = 1
            gk["top_p"] = 1.0

        t0 = time.time()
        candidates = generate_svg(
            model, input_ids, attention_mask,
            token_config, svg_tokenizer,
            num_return_sequences=num_cand + EXTRA_CANDIDATES_BUFFER,
            **gk,
        )
        elapsed = time.time() - t0
        total_elapsed += elapsed

        if candidates and not no_validate:
            valid = [c for c in candidates if validate_candidate(c["svg_str"])]
            candidates = valid[:num_cand]

        for ci, cand in enumerate(candidates[:num_cand]):
            (prompt_dir / f"{prefix}_{label}_c{ci}.svg").write_text(
                cand["svg_str"], encoding="utf-8"
            )
            if save_png:
                img = render_svg_to_image(cand["svg_str"])
                if img is not None:
                    img.save(str(prompt_dir / f"{prefix}_{label}_c{ci}.png"))

        all_candidates.append((temp, len(candidates[:num_cand]), round(elapsed, 2)))

    return all_candidates, round(total_elapsed, 2)


# ============================================================================
# Per-GPU worker
# ============================================================================

def run_on_single_gpu(
    local_rank: int,
    gpu_id: int,
    prompt_indices: List[int],
    prompts: List[str],
    ref_records: List[Dict[str, Any]],
    args: argparse.Namespace,
    prompt_ids: Optional[List[int]] = None,
):
    device = f"cuda:{gpu_id}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not prompt_indices:
        print(f"[GPU {gpu_id}] No prompts assigned, exiting.")
        return

    schedule = getattr(args, '_temperature_schedule', DEFAULT_TEMPERATURE_SCHEDULE)
    total_cand = sum(n for _, n in schedule)

    print(f"\n[GPU {gpu_id}] Assigned {len(prompt_indices)} prompts: "
          f"{prompt_indices[0]} ~ {prompt_indices[-1]}")

    # ---- Load HVM model ----
    hvm_model, tokenizer, processor, token_config, hvm_config = load_hvm_model(
        model_size=args.model_size,
        hvm_config_path=args.hvm_config,
        hvm_checkpoint_path=args.hvm_checkpoint,
        config_dir=args.config_dir,
        omnisvg_checkpoint=args.omnisvg_checkpoint,
        base_model_override=args.base_model,
        device=device,
    )
    transformer_for_generate = hvm_model.base_model.transformer
    svg_tokenizer = SVGTokenizer(SVG_CONFIG_PATH, model_size=args.model_size)

    # ---- Extract features from custom ref SVGs (once per worker) ----
    print(f"[GPU {gpu_id}] Building features from {len(ref_records)} "
          f"custom reference SVGs ...")
    visual_module = hvm_model.base_model.transformer.visual
    custom_data = build_custom_ref_data(
        ref_records, visual_module, processor, device
    )
    n_ref = len(custom_data["ref_features"])
    n_grp = len(custom_data["group_features"])
    print(f"[GPU {gpu_id}] Features ready: {n_ref} ref × [256,3584], "
          f"{n_grp} groups × [256,3584]")

    output_dir = Path(args.output_dir)
    gen_kwargs_base = dict(
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    prompt_offset = getattr(args, '_prompt_offset', 0)
    single_slug = getattr(args, '_single_prompt_slug', None)
    skipped = 0
    pbar = tqdm(prompt_indices, desc=f"[GPU {gpu_id}]", position=local_rank)
    for pi in pbar:
        prompt = prompts[pi]
        if single_slug is not None:
            prompt_dir = output_dir / single_slug
        elif prompt_ids is not None:
            prompt_dir = output_dir / f"prompt_{prompt_ids[pi]:06d}"
        else:
            prompt_dir = output_dir / f"prompt_{prompt_offset + pi:06d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        save_ref_material(prompt_dir, ref_records)

        base_done = _count_existing(prompt_dir, "base", schedule)
        hvm_done = _count_existing(prompt_dir, "hvm", schedule)
        need_base = base_done < total_cand
        need_hvm = hvm_done < total_cand

        if args.resume and not need_base and not need_hvm:
            skipped += 1
            continue

        input_ids, attention_mask = prepare_text_inputs(
            prompt, processor, token_config, device
        )

        ref_info = {
            "prompt": prompt,
            "ref_files": [r["filename"] for r in ref_records],
            "ref_descriptions": [r["description"] for r in ref_records],
            "ref_source": str(args.ref_images_dir),
            "groups_per_ref": [len(r["group_images"]) for r in ref_records],
        }
        (prompt_dir / "retrieval.json").write_text(
            json.dumps(ref_info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ---- Baseline (no HVM) ----
        base_elapsed = 0.0
        base_detail = []
        if need_base or not args.resume:
            clear_hvm_memory(hvm_model)
            base_detail, base_elapsed = _generate_candidates_multi_temp(
                transformer_for_generate, input_ids, attention_mask,
                token_config, svg_tokenizer, gen_kwargs_base,
                args.no_validate, prompt_dir, "base", args.save_png, schedule,
            )
            base_done = sum(n for _, n, _ in base_detail)

        # ---- HVM (with custom refs) ----
        hvm_elapsed = 0.0
        hvm_detail = []
        if need_hvm or not args.resume:
            set_hvm_memory(
                hvm_model,
                ref_features=custom_data["ref_features"],
                group_features=custom_data["group_features"],
                ref_text=custom_data["ref_text"],
                tokenizer=tokenizer,
                hvm_config=hvm_config,
                device=device,
                group_tag_meta=custom_data["tag_meta"],
                group_ids=custom_data["group_ids"],
            )

            gen_inputs_embeds = None
            gen_attn = attention_mask
            if (hvm_config.memory_mode == "visual_prefix"
                    and getattr(hvm_model, "_visual_prefix", None) is not None):
                gen_inputs_embeds, gen_attn = prepare_visual_prefix_inputs(
                    hvm_model, input_ids, attention_mask, device
                )

            hvm_detail, hvm_elapsed = _generate_candidates_multi_temp(
                transformer_for_generate,
                input_ids if gen_inputs_embeds is None else gen_inputs_embeds,
                gen_attn,
                token_config, svg_tokenizer, gen_kwargs_base,
                args.no_validate, prompt_dir, "hvm", args.save_png, schedule,
            )
            clear_hvm_memory(hvm_model)
            hvm_done = sum(n for _, n, _ in hvm_detail)

        summary = {
            "prompt": prompt,
            "temperature_schedule": [(t, n) for t, n in schedule],
            "base_total": base_done,
            "base_elapsed": base_elapsed,
            "base_detail": [
                {"temp": t, "count": n, "elapsed": e} for t, n, e in base_detail
            ],
            "hvm_total": hvm_done,
            "hvm_elapsed": hvm_elapsed,
            "hvm_detail": [
                {"temp": t, "count": n, "elapsed": e} for t, n, e in hvm_detail
            ],
        }
        (prompt_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        pbar.set_postfix(base=base_done, hvm=hvm_done)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[GPU {gpu_id}] Done. (skipped {skipped} already-complete prompts)")


def worker(
    local_rank: int,
    args: argparse.Namespace,
    index_splits: List[List[int]],
    prompts: List[str],
    ref_records: List[Dict[str, Any]],
    prompt_ids: Optional[List[int]] = None,
):
    run_on_single_gpu(
        local_rank=local_rank,
        gpu_id=local_rank,
        prompt_indices=index_splits[local_rank],
        prompts=prompts,
        ref_records=ref_records,
        args=args,
        prompt_ids=prompt_ids,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    p = argparse.ArgumentParser(
        description="Text2SVG with custom reference SVGs — HVM inference"
    )

    prompt_group = p.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt_file", type=str, default=None,
        help="Text file with prompts (one per line, optionally 'id: prompt').",
    )
    prompt_group.add_argument(
        "--prompt", type=str, default=None,
        help="Single prompt string.",
    )

    p.add_argument("--ref_images_dir", type=str, required=True,
                   help="Directory containing reference SVG files. "
                        "Optional per-ref descriptions via <stem>.txt files.")

    p.add_argument("--model_size", type=str, default="8B", choices=["4B", "8B"])
    p.add_argument("--config_dir", type=str, default=None)
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--hvm_checkpoint", type=str, required=True)
    p.add_argument("--hvm_config", type=str, default=None)
    p.add_argument("--omnisvg_checkpoint", type=str, default=None)
    p.add_argument("--output_dir", type=str,
                   default="./inference_results/custom_ref_text2svg")
    p.add_argument("--num_candidates", type=int, default=None,
                   help="Override: generate this many candidates at EACH temperature.")
    p.add_argument("--max_new_tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.90)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--save_png", action="store_true", default=False)
    p.add_argument("--no_validate", action="store_true", default=False)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--num_gpus", type=int, default=None)
    p.add_argument("--start_idx", type=int, default=None)
    p.add_argument("--end_idx", type=int, default=None)
    args = p.parse_args()

    args.ref_images_dir = _resolve_path(args.ref_images_dir)

    if args.num_candidates is not None:
        k = args.num_candidates
        args._temperature_schedule = [(t, k) for t, _ in DEFAULT_TEMPERATURE_SCHEDULE]
    else:
        args._temperature_schedule = list(DEFAULT_TEMPERATURE_SCHEDULE)
    schedule = args._temperature_schedule
    total_cand = sum(n for _, n in schedule)

    if args.omnisvg_checkpoint is None:
        _omni_default = _resolve_path(
            "/mnt/data2/wuqingman/models/OmniSVG/OmniSVG1.1_8B"
        )
        if os.path.exists(_omni_default):
            args.omnisvg_checkpoint = _omni_default

    if args.hvm_config is None:
        ckpt_dir = Path(args.hvm_checkpoint).parent
        for name in ("hvm_model_config.json", "hvm_config.json"):
            candidate = ckpt_dir / name
            if candidate.exists():
                args.hvm_config = str(candidate)
                break
        else:
            raise FileNotFoundError(
                f"Cannot find hvm_model_config.json in {ckpt_dir}. "
                "Please specify --hvm_config."
            )

    # ---- 1. Load & decompose reference SVGs ----
    print("\n=== Loading & decomposing custom reference SVGs ===")
    ref_records = load_ref_svgs(args.ref_images_dir)
    print(f"  Loaded {len(ref_records)} reference SVGs:")
    for i, rec in enumerate(ref_records):
        desc_preview = f" — {rec['description'][:60]}..." if rec["description"] else ""
        print(f"    [{i}] {rec['filename']}: "
              f"{rec['num_paths']} paths → {len(rec['group_images'])} groups"
              f"{desc_preview}")

    # ---- 2. Load prompts ----
    if args.prompt is not None:
        prompts = [args.prompt]
        prompt_ids = None
        prompt_offset = 0
        slug = re.sub(r"[^\w\s-]", "", args.prompt)[:60].strip().replace(" ", "_")
        timestamp = time.strftime("%m%d_%H%M%S")
        slug_with_ts = f"{slug}_{timestamp}"
        if (not args.output_dir
                or args.output_dir == "./inference_results/custom_ref_text2svg"):
            args.output_dir = f"./inference_results/{slug_with_ts}"
        args._single_prompt_slug = slug_with_ts
        print(f"\nSingle prompt mode: \"{args.prompt[:80]}\"")
    else:
        all_prompt_ids, all_prompts = load_prompts(args.prompt_file)
        total = len(all_prompts)
        start = args.start_idx if args.start_idx is not None else 0
        end = args.end_idx if args.end_idx is not None else total
        start = max(0, min(start, total))
        end = max(start, min(end, total))
        prompts = all_prompts[start:end]
        prompt_ids = all_prompt_ids[start:end] if all_prompt_ids else None
        prompt_offset = start
        id_info = " (with explicit IDs)" if prompt_ids else ""
        print(f"\nLoaded {total} prompts from {args.prompt_file}, "
              f"using [{start}, {end}) = {len(prompts)} prompts{id_info}")

    args._prompt_offset = prompt_offset if args.prompt is None else 0

    # ---- 3. Determine GPUs ----
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available == 0:
        raise RuntimeError("No CUDA GPU detected!")
    num_gpus = (min(args.num_gpus, num_gpus_available)
                if args.num_gpus else num_gpus_available)

    all_indices = list(range(len(prompts)))
    if args.resume:
        output_dir = Path(args.output_dir)
        pending = []
        single_slug = getattr(args, '_single_prompt_slug', None)
        for pi in all_indices:
            if single_slug is not None:
                pdir = output_dir / single_slug
            elif prompt_ids is not None:
                pdir = output_dir / f"prompt_{prompt_ids[pi]:06d}"
            else:
                pdir = output_dir / f"prompt_{prompt_offset + pi:06d}"
            if (_count_existing(pdir, "base", schedule) < total_cand
                    or _count_existing(pdir, "hvm", schedule) < total_cand):
                pending.append(pi)
        all_indices = pending
        print(f"Resume: {len(pending)} pending, "
              f"{len(prompts) - len(pending)} complete")

    index_splits = split_indices(all_indices, num_gpus)

    total_groups = sum(len(r["group_images"]) for r in ref_records)
    print("\n" + "=" * 70)
    print(f"Custom-Ref HVM Inference  --  {num_gpus} GPU(s)")
    print("=" * 70)
    print(f"  Prompts          : {len(prompts)} total, {len(all_indices)} pending")
    print(f"  Ref SVGs         : {len(ref_records)} refs, {total_groups} groups total")
    print(f"  HVM checkpoint   : {args.hvm_checkpoint}")
    print(f"  Candidates/prompt: {total_cand} ({schedule})")
    print(f"  Output dir       : {args.output_dir}")
    for i, split in enumerate(index_splits):
        label = f"[{split[0]}~{split[-1]}]" if split else "[empty]"
        print(f"  GPU {i}: {len(split)} prompts  {label}")
    print("=" * 70)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if num_gpus == 1:
        run_on_single_gpu(
            local_rank=0, gpu_id=0,
            prompt_indices=index_splits[0],
            prompts=prompts,
            ref_records=ref_records,
            args=args,
            prompt_ids=prompt_ids,
        )
    else:
        mp.start_processes(
            worker,
            args=(args, index_splits, prompts, ref_records, prompt_ids),
            nprocs=num_gpus,
            start_method="spawn",
        )

    print(f"\n{'=' * 70}")
    print(f"All done! Results saved to: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
