#!/usr/bin/env python3
"""
Image-to-image similarity search against the 25w dataset using CLIP vision features.

Two-step workflow:
  1. build-index (one-time, ~20min on 1 GPU): extract CLIP image embeddings for all 25w samples
  2. search: encode query image(s) and find top-K nearest neighbors

Usage:
  # Step 1: build index (only need to run once)
  CUDA_VISIBLE_DEVICES=0 python inference/query_similar_images.py build-index

  # Step 2: search
  python inference/query_similar_images.py search path/to/image.png
  python inference/query_similar_images.py search icon.svg --top_k 20
  python inference/query_similar_images.py search img1.png img2.svg img3.jpg
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

HVM_DIR = "/mnt/data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_22w_nozoom_top3part"
CLIP_MODEL = "/mnt/data/wuqingman/models/openai/clip-vit-large-patch14"
PARQUET_ROOT = "/mnt/data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_process_train25_exclude_p25"

for _prefix, _mapped in [
    ("/mnt/data3/", "/mnt/a100_1_data3/"),
    ("/mnt/data/", "/mnt/a100_1_data/"),
    ("/mnt/data2/", "/mnt/a100_1_data2/"),
]:
    for _var_name in ("HVM_DIR", "CLIP_MODEL", "PARQUET_ROOT"):
        _val = locals()[_var_name]
        if not os.path.exists(_val) and _val.startswith(_prefix):
            _alt = _mapped + _val[len(_prefix):]
            if os.path.exists(_alt):
                locals()[_var_name] = _alt


def load_metadata(hvm_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(hvm_dir, "metadata.jsonl")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render_svg_to_pil(svg_path: str, size: int = 224) -> Optional[Image.Image]:
    try:
        import cairosvg
    except ImportError:
        print("  [WARN] cairosvg not installed. pip install cairosvg")
        return None
    try:
        png_bytes = cairosvg.svg2png(
            bytestring=Path(svg_path).read_bytes(),
            output_width=size, output_height=size,
        )
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception as e:
        print(f"  [WARN] Failed to render SVG {svg_path}: {e}")
        return None


def load_query_image(path: str) -> Optional[Image.Image]:
    ext = Path(path).suffix.lower()
    if ext == ".svg":
        return render_svg_to_pil(path)
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"  [WARN] Failed to load {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# build-index
# ---------------------------------------------------------------------------

def cmd_build_index(args: argparse.Namespace) -> None:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    hvm_dir = args.hvm_dir
    parquet_root = args.parquet_root
    output_path = os.path.join(hvm_dir, "image_embeddings.npy")
    if os.path.exists(output_path) and not args.force:
        print(f"image_embeddings.npy already exists at {output_path}")
        print("Use --force to rebuild.")
        return

    metadata = load_metadata(hvm_dir)
    total = len(metadata)
    print(f"Total samples: {total}")

    parquet_files_ordered: List[str] = []
    parquet_rows_ordered: List[int] = []
    for row in metadata:
        parquet_files_ordered.append(row["parquet_file"])
        parquet_rows_ordered.append(int(row["parquet_row"]))

    unique_parquets = sorted(set(parquet_files_ordered))
    print(f"Unique parquet files: {len(unique_parquets)}")

    is_local = os.path.isdir(CLIP_MODEL)
    print(f"Loading CLIP model from {CLIP_MODEL} ...")
    model = CLIPModel.from_pretrained(CLIP_MODEL, local_files_only=is_local)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL, local_files_only=is_local)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    embed_dim = model.config.projection_dim
    print(f"  Device: {device}, embed_dim: {embed_dim}")

    all_embeddings = np.zeros((total, embed_dim), dtype=np.float32)
    batch_size = args.batch_size
    done = 0
    t0 = time.time()

    import pyarrow.parquet as pq

    for pf in unique_parquets:
        pf_path = os.path.join(parquet_root, pf)
        if not os.path.exists(pf_path):
            print(f"  [WARN] Missing parquet: {pf_path}, skipping")
            continue
        table = pq.read_table(pf_path, columns=["image"])
        image_col = table.column("image")

        indices_in_this_pf = [
            i for i in range(total) if parquet_files_ordered[i] == pf
        ]
        pil_batch: List[Image.Image] = []
        global_indices: List[int] = []

        for i in indices_in_this_pf:
            row_idx = parquet_rows_ordered[i]
            img_data = image_col[row_idx].as_py()
            img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (255, 255, 255))
            pil_batch.append(img)
            global_indices.append(i)

            if len(pil_batch) >= batch_size:
                inputs = processor(images=pil_batch, return_tensors="pt").to(device)
                with torch.no_grad():
                    feats = model.get_image_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                feats_np = feats.cpu().numpy()
                for bi, gi in enumerate(global_indices):
                    all_embeddings[gi] = feats_np[bi]
                done += len(pil_batch)
                elapsed = time.time() - t0
                speed = done / elapsed
                eta = (total - done) / speed if speed > 0 else 0
                print(
                    f"\r  [{pf}] {done}/{total}  "
                    f"{speed:.1f} img/s  ETA {eta/60:.1f}min",
                    end="", flush=True,
                )
                pil_batch.clear()
                global_indices.clear()

        if pil_batch:
            inputs = processor(images=pil_batch, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy()
            for bi, gi in enumerate(global_indices):
                all_embeddings[gi] = feats_np[bi]
            done += len(pil_batch)

        del table, image_col
        elapsed = time.time() - t0
        print(f"\r  [{pf}] done. {done}/{total} ({elapsed:.0f}s)")

    np.save(output_path, all_embeddings)
    elapsed = time.time() - t0
    print(f"\nSaved image_embeddings.npy ({total}, {embed_dim}) to {output_path}")
    print(f"Total time: {elapsed/60:.1f} min")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> None:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    hvm_dir = args.hvm_dir
    embeddings_path = os.path.join(hvm_dir, "image_embeddings.npy")
    if not os.path.exists(embeddings_path):
        print(f"ERROR: image_embeddings.npy not found at {embeddings_path}")
        print("Run `build-index` first:")
        print(f"  CUDA_VISIBLE_DEVICES=0 python {__file__} build-index")
        sys.exit(1)

    image_paths: List[str] = []
    pil_images: List[Image.Image] = []
    for path in args.images:
        img = load_query_image(path)
        if img is not None:
            image_paths.append(path)
            pil_images.append(img)
        else:
            print(f"  Skipping {path}")
    if not pil_images:
        print("No valid images to query.")
        sys.exit(1)

    is_local = os.path.isdir(CLIP_MODEL)
    print(f"Loading CLIP model from {CLIP_MODEL} ...")
    model = CLIPModel.from_pretrained(CLIP_MODEL, local_files_only=is_local)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL, local_files_only=is_local)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    query_emb = feats.cpu().numpy().astype(np.float32)

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading image embeddings from {embeddings_path} ...")
    corpus = np.load(embeddings_path, mmap_mode="r")
    print(f"  Corpus shape: {corpus.shape}")

    top_k = args.top_k
    corpus_f32 = np.asarray(corpus, dtype=np.float32)
    sims = query_emb @ corpus_f32.T
    indices = np.argsort(-sims, axis=-1)[:, :top_k]
    scores = np.take_along_axis(sims, indices, axis=-1)

    print(f"Loading metadata ...")
    metadata = load_metadata(hvm_dir)
    metadata_lookup = {row["idx"]: row for row in metadata}

    for qi in range(len(pil_images)):
        width = 80
        print("\n" + "=" * width)
        print(f"  Query image: {image_paths[qi]}")
        print("=" * width)
        print(f"{'─'*width}")
        print("  [CLIP Image→Image Similarity] (cosine)")
        print(f"{'─'*width}")
        for rank in range(top_k):
            idx = int(indices[qi, rank])
            score = float(scores[qi, rank])
            row = metadata_lookup.get(idx, {})
            desc = row.get("description", "N/A")
            src_id = row.get("id", "?")
            pf = row.get("parquet_file", "?")
            pr = row.get("parquet_row", "?")
            print(f"  #{rank+1:2d}  idx={idx:6d}  score={score:.4f}  id={src_id}")
            print(f"       {desc[:120]}")
            print(f"       parquet: {pf} row {pr}")
        print("=" * width + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Image-to-image similarity search against the 25w dataset"
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-index", help="Build CLIP image embeddings for the 25w dataset (one-time)")
    p_build.add_argument("--hvm_dir", type=str, default=HVM_DIR)
    p_build.add_argument("--parquet_root", type=str, default=PARQUET_ROOT)
    p_build.add_argument("--batch_size", type=int, default=64)
    p_build.add_argument("--force", action="store_true", help="Rebuild even if file exists")

    p_search = sub.add_parser("search", help="Search by query image(s)")
    p_search.add_argument("images", nargs="+", help="Query image paths (png/jpg/svg)")
    p_search.add_argument("--top_k", type=int, default=10)
    p_search.add_argument("--hvm_dir", type=str, default=HVM_DIR)

    args = p.parse_args()
    if args.command == "build-index":
        cmd_build_index(args)
    elif args.command == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()
