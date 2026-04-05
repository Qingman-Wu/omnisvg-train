#!/usr/bin/env python3
"""
为 decoded SVG 数据集离线检索并缓存 top-3 参考样本。

默认输入:
  query_jsonl:
    /mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k_decoded.jsonl
  hvm_dir:
    /mnt/data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_22w_nozoom_top3part

默认输出:
  /mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k_top3_retrieval.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from tqdm import tqdm


DEFAULT_QUERY_JSONL = Path(
    "/mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k_decoded.jsonl"
)
DEFAULT_HVM_DIR = Path(
    "/mnt/data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_22w_nozoom_top3part"
)
DEFAULT_OUTPUT_JSONL = Path(
    "/mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k_top3_retrieval.jsonl"
)
DEFAULT_CLIP_MODEL = Path("/mnt/data/wuqingman/models/openai/clip-vit-large-patch14")


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_texts_with_clip(
    texts: List[str],
    clip_model_path: str,
    batch_size: int,
) -> np.ndarray:
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_path, local_files_only=True)
    model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True).to(device).eval()
    embed_dim = model.config.projection_dim
    embeddings = np.zeros((len(texts), embed_dim), dtype=np.float32)

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding CLIP queries"):
        end = min(start + batch_size, len(texts))
        inputs = tokenizer(
            texts[start:end],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings[start:end] = feats.cpu().numpy()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


def build_query_text(record: Dict, text_key: str) -> str:
    text = (record.get(text_key) or "").strip()
    if not text:
        raise ValueError(f"missing query text field: {text_key}")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline top-3 retrieval cache builder.")
    parser.add_argument("--query_jsonl", type=Path, default=DEFAULT_QUERY_JSONL)
    parser.add_argument("--hvm_dir", type=Path, default=DEFAULT_HVM_DIR)
    parser.add_argument("--output_jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--clip_model_path", type=Path, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import faiss

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    queries = load_jsonl(args.query_jsonl)
    if args.limit is not None:
        queries = queries[: args.limit]
    print(f"Loaded {len(queries)} query samples from {args.query_jsonl}")

    corpus_meta_path = args.hvm_dir / "metadata.jsonl"
    index_path = args.hvm_dir / "faiss_index.bin"
    if not corpus_meta_path.exists():
        raise FileNotFoundError(f"Missing corpus metadata: {corpus_meta_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")

    corpus_meta = load_jsonl(corpus_meta_path)
    idx_to_meta = {row["idx"]: row for row in corpus_meta}
    print(f"Loaded {len(corpus_meta)} corpus metadata rows")

    print(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))
    print(f"FAISS ntotal: {index.ntotal}")

    query_texts = [build_query_text(row, args.text_key) for row in queries]
    query_embeddings = encode_texts_with_clip(
        query_texts,
        clip_model_path=str(args.clip_model_path),
        batch_size=args.batch_size,
    )

    print(f"Searching top-{args.top_k} ...")
    scores, indices = index.search(query_embeddings, args.top_k)

    written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for i, query in enumerate(queries):
            ref_indices = [int(v) for v in indices[i].tolist()]
            ref_scores = [float(v) for v in scores[i].tolist()]
            ref_ids = []
            ref_descriptions = []
            for ref_idx in ref_indices:
                meta = idx_to_meta.get(ref_idx, {})
                ref_ids.append(meta.get("id"))
                ref_descriptions.append(meta.get("description", ""))

            row = {
                "id": query["id"],
                "text": query_texts[i],
                "ref_indices": ref_indices,
                "ref_ids": ref_ids,
                "ref_scores": ref_scores,
                "ref_descriptions": ref_descriptions,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print("=" * 60)
    print(f"Queries processed : {written}")
    print(f"Output JSONL      : {args.output_jsonl}")
    print("=" * 60)


if __name__ == "__main__":
    main()
