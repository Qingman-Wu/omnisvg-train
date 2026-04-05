#!/usr/bin/env python3
"""
Query the 25w dataset for the most similar text prompts.

Usage:
  python inference/query_similar_prompts.py "A green plant in a yellow flowerpot"
  python inference/query_similar_prompts.py "A cute cat" --top_k 20
  python inference/query_similar_prompts.py --prompt_file prompts.txt
"""

import argparse
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HVM_DIR = "/mnt/data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_22w_nozoom_top3part"
CLIP_MODEL = "/mnt/data/wuqingman/models/openai/clip-vit-large-patch14"

for _prefix, _mapped in [
    ("/mnt/data3/", "/mnt/a100_1_data3/"),
    ("/mnt/data/", "/mnt/a100_1_data/"),
]:
    if not os.path.exists(HVM_DIR) and os.path.exists(_mapped + HVM_DIR.split(_prefix, 1)[-1]):
        HVM_DIR = _mapped + HVM_DIR.split(_prefix, 1)[-1]
    if not os.path.exists(CLIP_MODEL) and os.path.exists(_mapped + CLIP_MODEL.split(_prefix, 1)[-1]):
        CLIP_MODEL = _mapped + CLIP_MODEL.split(_prefix, 1)[-1]


def load_metadata(hvm_dir: str) -> List[Dict]:
    path = os.path.join(hvm_dir, "metadata.jsonl")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def text_search(query: str, metadata: List[Dict], top_k: int) -> List[Tuple[int, float, Dict]]:
    query_lower = query.lower().strip()
    scored = []
    for row in metadata:
        desc = row.get("description", "")
        desc_lower = desc.lower().strip()
        if query_lower == desc_lower:
            score = 1.0
        else:
            score = SequenceMatcher(None, query_lower, desc_lower).ratio()
        scored.append((row["idx"], score, row))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def clip_search(queries: List[str], hvm_dir: str, top_k: int) -> List[List[Tuple[int, float]]]:
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    is_local = os.path.isdir(CLIP_MODEL)
    print(f"Loading CLIP model from {CLIP_MODEL} ...")
    model = CLIPModel.from_pretrained(CLIP_MODEL, local_files_only=is_local)
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL, local_files_only=is_local)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    inputs = tokenizer(
        queries, padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    embeddings = feats.cpu().numpy().astype(np.float32)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    faiss_path = os.path.join(hvm_dir, "faiss_index.bin")
    try:
        import faiss
        print(f"Loading FAISS index from {faiss_path} ...")
        index = faiss.read_index(faiss_path)
        print(f"  Index size: {index.ntotal}")
        scores, indices = index.search(embeddings, top_k)
    except ImportError:
        npy_path = os.path.join(hvm_dir, "text_embeddings.npy")
        print(f"faiss not available, brute-force search with {npy_path} ...")
        corpus = np.load(npy_path, mmap_mode="r").astype(np.float32)
        corpus_norm = corpus / np.linalg.norm(corpus, axis=-1, keepdims=True)
        sims = embeddings @ corpus_norm.T  # (Q, N)
        indices = np.argsort(-sims, axis=-1)[:, :top_k]
        scores = np.take_along_axis(sims, indices, axis=-1)

    results = []
    for qi in range(len(queries)):
        results.append([
            (int(indices[qi, j]), float(scores[qi, j]))
            for j in range(top_k)
        ])
    return results


def print_results(
    query: str,
    text_hits: List[Tuple[int, float, Dict]],
    clip_hits: List[Tuple[int, float]],
    metadata_lookup: Dict[int, Dict],
):
    width = 80
    print("\n" + "=" * width)
    print(f"  Query: {query}")
    print("=" * width)

    exact = [h for h in text_hits if h[1] >= 0.999]
    if exact:
        print(f"\n  *** EXACT MATCH FOUND (idx={exact[0][0]}) ***")

    print(f"\n{'─'*width}")
    print("  [Text Similarity] (SequenceMatcher ratio)")
    print(f"{'─'*width}")
    for rank, (idx, score, row) in enumerate(text_hits):
        tag = " <<<< EXACT" if score >= 0.999 else ""
        print(f"  #{rank+1:2d}  idx={idx:6d}  score={score:.4f}  id={row['id']}")
        print(f"       {row.get('description', '')[:120]}{tag}")

    print(f"\n{'─'*width}")
    print("  [CLIP Semantic Similarity] (cosine)")
    print(f"{'─'*width}")
    for rank, (idx, score) in enumerate(clip_hits):
        row = metadata_lookup.get(idx, {})
        desc = row.get("description", "N/A")
        print(f"  #{rank+1:2d}  idx={idx:6d}  score={score:.4f}  id={row.get('id','?')}")
        print(f"       {desc[:120]}")

    print("=" * width + "\n")


def main():
    p = argparse.ArgumentParser(description="Query similar prompts from the 25w dataset")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("prompt", nargs="?", default=None, help="Single prompt string")
    src.add_argument("--prompt_file", type=str, default=None,
                     help="Text file with prompts (one per line)")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--hvm_dir", type=str, default=HVM_DIR)
    p.add_argument("--no_clip", action="store_true", help="Skip CLIP search, only do text matching")
    args = p.parse_args()

    if args.prompt:
        queries = [args.prompt]
    else:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            queries = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    print(f"Loading metadata from {args.hvm_dir} ...")
    metadata = load_metadata(args.hvm_dir)
    metadata_lookup = {row["idx"]: row for row in metadata}
    print(f"  Loaded {len(metadata)} entries")

    if not args.no_clip:
        clip_results = clip_search(queries, args.hvm_dir, args.top_k)
    else:
        clip_results = [[] for _ in queries]

    for qi, query in enumerate(queries):
        text_hits = text_search(query, metadata, args.top_k)
        print_results(query, text_hits, clip_results[qi], metadata_lookup)


if __name__ == "__main__":
    main()
