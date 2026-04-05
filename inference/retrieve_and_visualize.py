#!/usr/bin/env python3
"""
CLIP retrieval + SVG group visualization — standalone tool.

Given a text prompt, retrieves the top-K most similar samples from the
25w corpus via CLIP, then renders whole images and per-group PNGs.

Usage:
  CUDA_VISIBLE_DEVICES=0 python inference/retrieve_and_visualize.py \
      --prompt "A cartoon-style light purple cat" \
      --top_k 20

  CUDA_VISIBLE_DEVICES=0 python inference/retrieve_and_visualize.py \
      --prompt_file prompts.txt \
      --top_k 10 \
      --output_dir ./retrieval_results
"""

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

VIEWBOX_SIZE = 200.0
IMAGE_SIZE = 448


def _resolve_path(path: str) -> str:
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


DEFAULT_HVM_DIR = _resolve_path(
    "/mnt/data3/wuqingman/datasets/OmniSVG/"
    "MMSVG-Illustration/hvm_precomputed_22w_nozoom_top3part"
)
DEFAULT_CLIP_PATH = _resolve_path(
    "/mnt/data/wuqingman/models/openai/clip-vit-large-patch14"
)
DEFAULT_PARQUET_ROOT = _resolve_path(
    "/mnt/data3/wuqingman/datasets/OmniSVG/"
    "MMSVG-Illustration/data_process_train25_exclude_p25"
)


# ============================================================================
# CLIP retrieval
# ============================================================================

def clip_retrieve(
    prompts: List[str],
    hvm_dir: str,
    clip_model_path: str,
    top_k: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    from transformers import CLIPModel, CLIPTokenizer

    print(f"Loading CLIP model from {clip_model_path} ...")
    is_local = os.path.isdir(clip_model_path)
    clip_model = CLIPModel.from_pretrained(clip_model_path, local_files_only=is_local)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path, local_files_only=is_local)
    clip_model = clip_model.to(device).eval()

    embed_dim = clip_model.config.projection_dim
    embeddings = np.zeros((len(prompts), embed_dim), dtype=np.float32)

    for start in range(0, len(prompts), 64):
        end = min(start + 64, len(prompts))
        inputs = clip_tokenizer(
            prompts[start:end], padding=True, truncation=True,
            max_length=77, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            feats = clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings[start:end] = feats.cpu().numpy()

    del clip_model, clip_tokenizer
    torch.cuda.empty_cache()

    try:
        import faiss
        index_path = os.path.join(hvm_dir, "faiss_index.bin")
        print(f"Loading FAISS index from {index_path} ...")
        index = faiss.read_index(index_path)
        scores, indices = index.search(embeddings, top_k)
    except (ImportError, FileNotFoundError):
        embeddings_path = os.path.join(hvm_dir, "text_embeddings.npy")
        print("Falling back to brute-force cosine search ...")
        corpus_embeddings = np.load(embeddings_path, mmap_mode="r")
        query = torch.from_numpy(embeddings).to(device)
        query = query / query.norm(dim=-1, keepdim=True)

        top_scores = torch.full((len(prompts), top_k), float("-inf"), device=device)
        top_indices = torch.full((len(prompts), top_k), -1, dtype=torch.long, device=device)

        for cs in range(0, corpus_embeddings.shape[0], 16384):
            ce = min(cs + 16384, corpus_embeddings.shape[0])
            chunk = torch.from_numpy(
                np.asarray(corpus_embeddings[cs:ce], dtype=np.float32)
            ).to(device)
            chunk = chunk / chunk.norm(dim=-1, keepdim=True)
            sc = torch.matmul(query, chunk.T)
            local_k = min(top_k, sc.shape[1])
            ls, li = torch.topk(sc, k=local_k, dim=1)
            li = li + cs
            ms = torch.cat([top_scores, ls], dim=1)
            mi = torch.cat([top_indices, li], dim=1)
            top_scores, sel = torch.topk(ms, k=top_k, dim=1)
            top_indices = torch.gather(mi, 1, sel)

        scores = top_scores.cpu().numpy()
        indices = top_indices.cpu().numpy()

    results = []
    for i in range(len(prompts)):
        results.append({
            "ref_indices": [int(indices[i, j]) for j in range(top_k)],
            "ref_scores": [float(scores[i, j]) for j in range(top_k)],
        })
    return results


# ============================================================================
# Metadata & parquet loading
# ============================================================================

def load_jsonl_all(path: str) -> Dict[int, Dict[str, Any]]:
    subset: Dict[int, Dict[str, Any]] = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            subset[rec["idx"]] = rec
    return subset


def load_jsonl_subset(path: str, keep_indices: set) -> Dict[int, Dict[str, Any]]:
    subset: Dict[int, Dict[str, Any]] = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = rec["idx"]
            if idx in keep_indices:
                subset[idx] = rec
    return subset


class ParquetImageLoader:
    def __init__(self, parquet_root: str):
        self.parquet_root = parquet_root
        self._cache: Dict[str, dict] = {}

    def _load(self, parquet_file: str) -> dict:
        if parquet_file not in self._cache:
            import pyarrow.parquet as pq
            path = os.path.join(self.parquet_root, parquet_file)
            table = pq.read_table(path, columns=["svg", "image"])
            self._cache[parquet_file] = table.to_pydict()
        return self._cache[parquet_file]

    def get_svg_and_image(self, meta: Dict[str, Any]):
        pf = meta.get("parquet_file")
        pr = meta.get("parquet_row")
        if pf is None or pr is None:
            return None, None
        data = self._load(pf)
        svg_str = data["svg"][pr]
        img_dict = data["image"][pr]
        img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            img = None
        return svg_str, img


# ============================================================================
# Group rendering
# ============================================================================

def render_group_to_image(
    svg_string: str,
    group_path_indices: List[int],
    image_size: int = IMAGE_SIZE,
) -> Optional[Image.Image]:
    if not svg_string or not group_path_indices:
        return None
    try:
        import cairosvg
    except ImportError:
        return None

    path_pattern = re.compile(r'<path\s[^>]*?(?:/>|>\s*</path>)', re.DOTALL)
    all_path_tags = path_pattern.findall(svg_string)
    if not all_path_tags:
        return None

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {VIEWBOX_SIZE} {VIEWBOX_SIZE}" '
        f'width="{image_size}" height="{image_size}">',
        f'<rect x="0" y="0" width="{VIEWBOX_SIZE}" '
        f'height="{VIEWBOX_SIZE}" fill="white"/>',
    ]
    for pidx in group_path_indices:
        if 0 <= int(pidx) < len(all_path_tags):
            svg_lines.append(all_path_tags[int(pidx)])
    svg_lines.append("</svg>")

    try:
        png_bytes = cairosvg.svg2png(
            bytestring="\n".join(svg_lines).encode("utf-8"),
            output_width=image_size, output_height=image_size,
        )
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception:
        return None


# ============================================================================
# Main visualization pipeline
# ============================================================================

def visualize_retrieval(
    prompt: str,
    retrieval: Dict[str, Any],
    ref_meta_lookup: Dict[int, Dict[str, Any]],
    ref_groups_lookup: Dict[int, Dict[str, Any]],
    loader: ParquetImageLoader,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    ref_indices = retrieval["ref_indices"]
    ref_scores = retrieval["ref_scores"]

    summary_rows = []

    for rank, (ref_idx, score) in enumerate(zip(ref_indices, ref_scores)):
        meta = ref_meta_lookup.get(ref_idx, {})
        desc = meta.get("description", "")
        svg_str, img = loader.get_svg_and_image(meta)

        prefix = f"rank{rank:02d}_idx{ref_idx}"

        if img is not None:
            img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            img_resized.save(str(output_dir / f"{prefix}_whole.png"))

        if svg_str:
            (output_dir / f"{prefix}.svg").write_text(svg_str, encoding="utf-8")

        groups_record = ref_groups_lookup.get(ref_idx, {})
        groups = list(groups_record.get("groups", []))
        num_groups_rendered = 0

        for gi, group in enumerate(groups):
            path_indices = [int(v) for v in group.get("path_indices", [])]
            if not path_indices or svg_str is None:
                continue
            grp_img = render_group_to_image(svg_str, path_indices)
            if grp_img is not None:
                grp_img.save(str(output_dir / f"{prefix}_group{gi}.png"))
                num_groups_rendered += 1

        summary_rows.append({
            "rank": rank,
            "ref_idx": ref_idx,
            "score": round(score, 4),
            "description": desc,
            "num_groups": num_groups_rendered,
        })

        print(f"  [{rank:2d}] idx={ref_idx:>6d}  score={score:.4f}  "
              f"groups={num_groups_rendered}  {desc[:60]}")

    summary = {
        "prompt": prompt,
        "top_k": len(ref_indices),
        "results": summary_rows,
    }
    (output_dir / "retrieval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ============================================================================
# Entry point
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="CLIP retrieval + SVG group visualization"
    )
    prompt_group = p.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, default=None,
                              help="Single text prompt.")
    prompt_group.add_argument("--prompt_file", type=str, default=None,
                              help="Text file with prompts.")

    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--output_dir", type=str, default="./retrieval_results")
    p.add_argument("--hvm_dir", type=str, default=DEFAULT_HVM_DIR)
    p.add_argument("--clip_model_path", type=str, default=DEFAULT_CLIP_PATH)
    p.add_argument("--parquet_root", type=str, default=DEFAULT_PARQUET_ROOT)
    args = p.parse_args()

    args.hvm_dir = _resolve_path(args.hvm_dir)
    args.clip_model_path = _resolve_path(args.clip_model_path)
    args.parquet_root = _resolve_path(args.parquet_root)

    # ---- Load prompts ----
    if args.prompt is not None:
        prompts = [args.prompt]
        slug = re.sub(r"[^\w\s-]", "", args.prompt)[:60].strip().replace(" ", "_")
        prompt_labels = [slug]
    else:
        prompts = []
        prompt_labels = []
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"^(\d+)\s*:\s*(.+)$", line)
                if m:
                    prompts.append(m.group(2).strip())
                    prompt_labels.append(f"id_{m.group(1)}")
                else:
                    prompts.append(line)
                    slug = re.sub(r"[^\w\s-]", "", line)[:40].strip().replace(" ", "_")
                    prompt_labels.append(slug)

    print(f"Loaded {len(prompts)} prompt(s)")

    # ---- CLIP retrieval ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    retrieval_results = clip_retrieve(
        prompts, args.hvm_dir, args.clip_model_path, args.top_k, device
    )

    # ---- Gather all needed ref indices ----
    all_ref_ids = set()
    for r in retrieval_results:
        all_ref_ids.update(r["ref_indices"])

    print(f"\nLoading metadata for {len(all_ref_ids)} unique refs ...")
    metadata_path = os.path.join(args.hvm_dir, "metadata.jsonl")
    ref_meta_lookup = load_jsonl_subset(metadata_path, all_ref_ids)

    groups_path = os.path.join(args.hvm_dir, "groups_train_ref.jsonl")
    if not os.path.exists(groups_path):
        groups_path = os.path.join(args.hvm_dir, "groups.jsonl")
    ref_groups_lookup = load_jsonl_subset(groups_path, all_ref_ids)

    loader = ParquetImageLoader(args.parquet_root)

    # ---- Visualize ----
    for i, (prompt, label, retrieval) in enumerate(
        zip(prompts, prompt_labels, retrieval_results)
    ):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(prompts)}] \"{prompt[:80]}\"")
        print(f"{'='*60}")

        out_dir = Path(args.output_dir) / label
        visualize_retrieval(
            prompt, retrieval, ref_meta_lookup,
            ref_groups_lookup, loader, out_dir,
        )

    print(f"\nDone! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
