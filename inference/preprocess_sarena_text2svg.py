#!/usr/bin/env python3
"""
Preprocess SArena text2svg.jsonl for HVM qualitative queue inference.

Converts the conversations-format JSONL into two files required by
inference_hvm_qualitative_queue.py:
  1. decoded.jsonl  – {id, text, svg}  (query source)
  2. retrieval.jsonl – {idx, ref_indices, ref_scores}  (retrieval results)

Retrieval is performed via CLIP text embeddings + FAISS / cosine similarity
against the HVM corpus (text_embeddings.npy or faiss_index.bin).

Usage:
  python preprocess_sarena_text2svg.py \
      --input_jsonl /path/to/text2svg.jsonl \
      --output_dir  /path/to/run_dir \
      --hvm_dir     /path/to/hvm_precomputed \
      --clip_model_path /path/to/clip-vit-large-patch14
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm


def parse_conversations_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """Parse SArena conversations JSONL → list of {id, text, svg}."""
    records: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = int(row["id"])

            human_msg = ""
            gpt_msg = None
            for turn in row.get("conversations", []):
                if turn["from"] == "human":
                    human_msg = turn["value"]
                elif turn["from"] == "gpt":
                    gpt_msg = turn["value"]

            match = re.search(r"Instruction:\s*(.+)", human_msg, re.DOTALL)
            text = match.group(1).strip() if match else human_msg.strip()

            rec: Dict[str, Any] = {"id": sample_id, "text": text}
            if gpt_msg:
                rec["svg"] = gpt_msg
            records.append(rec)

    return records


def compute_retrieval(
    records: List[Dict[str, Any]],
    hvm_dir: str,
    clip_model_path: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Compute CLIP-based top-K retrieval from the HVM corpus."""
    from transformers import CLIPModel, CLIPTokenizer

    print(f"Loading CLIP model from {clip_model_path} ...")
    is_local = os.path.isdir(clip_model_path)
    clip_model = CLIPModel.from_pretrained(clip_model_path, local_files_only=is_local)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path, local_files_only=is_local)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = clip_model.to(device).eval()

    texts = [r["text"] for r in records]
    embed_dim = clip_model.config.projection_dim
    embeddings = np.zeros((len(texts), embed_dim), dtype=np.float32)
    batch_size = 256

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        end = min(start + batch_size, len(texts))
        inputs = clip_tokenizer(
            texts[start:end],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            feats = clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings[start:end] = feats.cpu().numpy()

    del clip_model
    torch.cuda.empty_cache()

    scores_arr = None
    indices_arr = None

    try:
        import faiss

        index_path = os.path.join(hvm_dir, "faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        print(f"Loading FAISS index from {index_path} ...")
        index = faiss.read_index(index_path)
        print(f"FAISS index size: {index.ntotal}")
        print(f"Searching Top-{top_k} ...")
        scores_arr, indices_arr = index.search(embeddings, top_k)
    except (ImportError, FileNotFoundError) as exc:
        print(f"FAISS unavailable ({exc}), falling back to chunked cosine similarity ...")
        embeddings_path = os.path.join(hvm_dir, "text_embeddings.npy")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Neither faiss_index.bin nor text_embeddings.npy found in {hvm_dir}"
            )

        corpus_embeddings = np.load(embeddings_path, mmap_mode="r")
        query = torch.from_numpy(embeddings).to(device)
        query = query / query.norm(dim=-1, keepdim=True)

        num_queries = query.shape[0]
        top_scores = torch.full((num_queries, top_k), float("-inf"), device=device)
        top_indices = torch.full((num_queries, top_k), -1, dtype=torch.long, device=device)
        chunk_size = 16384

        for cs in tqdm(range(0, corpus_embeddings.shape[0], chunk_size), desc="Searching corpus"):
            ce = min(cs + chunk_size, corpus_embeddings.shape[0])
            chunk_np = np.asarray(corpus_embeddings[cs:ce], dtype=np.float32)
            chunk = torch.from_numpy(chunk_np).to(device)
            chunk = chunk / chunk.norm(dim=-1, keepdim=True)
            score_chunk = torch.matmul(query, chunk.transpose(0, 1))
            local_k = min(top_k, score_chunk.shape[1])
            local_scores, local_indices = torch.topk(score_chunk, k=local_k, dim=1)
            local_indices = local_indices + cs

            merged_scores = torch.cat([top_scores, local_scores], dim=1)
            merged_indices = torch.cat([top_indices, local_indices], dim=1)
            top_scores, select = torch.topk(merged_scores, k=top_k, dim=1)
            top_indices = torch.gather(merged_indices, 1, select)

            del chunk, score_chunk, local_scores, local_indices, merged_scores, merged_indices, select
            if device.type == "cuda":
                torch.cuda.empty_cache()

        scores_arr = top_scores.cpu().numpy()
        indices_arr = top_indices.cpu().numpy()

    results: List[Dict[str, Any]] = []
    for i in range(len(records)):
        results.append(
            {
                "idx": records[i]["id"],
                "ref_indices": [int(indices_arr[i, j]) for j in range(top_k)],
                "ref_scores": [float(scores_arr[i, j]) for j in range(top_k)],
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess SArena text2svg.jsonl for queue inference")
    parser.add_argument("--input_jsonl", type=str, required=True, help="SArena text2svg.jsonl path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory (same as run_dir)")
    parser.add_argument(
        "--hvm_dir",
        type=str,
        default="/mnt/a100_1_data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_22w_nozoom_top3part",
    )
    parser.add_argument(
        "--clip_model_path",
        type=str,
        default="/mnt/a100_1_data/wuqingman/models/openai/clip-vit-large-patch14",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of top references to retrieve")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Parsing {args.input_jsonl} ...")
    records = parse_conversations_jsonl(args.input_jsonl)
    print(f"  Total records: {len(records)}")

    decoded_path = os.path.join(args.output_dir, "decoded.jsonl")
    with open(decoded_path, "w", encoding="utf-8") as f:
        for r in records:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"  Decoded JSONL written: {decoded_path}")

    print("Computing CLIP-based retrieval ...")
    retrieval = compute_retrieval(records, args.hvm_dir, args.clip_model_path, args.top_k)

    retrieval_path = os.path.join(args.output_dir, "retrieval.jsonl")
    with open(retrieval_path, "w", encoding="utf-8") as f:
        for r in retrieval:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"  Retrieval JSONL written: {retrieval_path}")

    print("=" * 60)
    print(f"Preprocessing done.")
    print(f"  decoded.jsonl   : {decoded_path}  ({len(records)} records)")
    print(f"  retrieval.jsonl : {retrieval_path}  ({len(retrieval)} records)")
    print(f"Next: run prepare + worker with inference_hvm_qualitative_queue.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
