#!/usr/bin/env python3
"""
Text-to-SVG benchmark inference for MMSVGBench parquet with HVM retrieval.

Pipeline:
1. Read text prompts from the benchmark parquet.
2. Retrieve Top-K similar training texts from the full corpus via CLIP + FAISS.
3. Load the retrieved refs' global features and part-group features from the
   precomputed HVM directory.
4. Inject retrieved memory into HVM and generate multiple SVG candidates.

Example:
CUDA_VISIBLE_DEVICES=4,5,7 python inference/inference_hvm_text2svg_benchmark.py \
    --base_model /mnt/a100_1_data/wuqingman/models/Qwen/Qwen2.5-VL-7B-Instruct \
    --omnisvg_checkpoint /mnt/a100_1_data2/wuqingman/models/OmniSVG/OmniSVG1.1_8B \
    --hvm_checkpoint /mnt/a100_1_data3/wuqingman/omnisvg-train/outputs_s9_full25w_top3part_12slot_nogist_edr_parttag_nozoom_last4/hvm_step_5000.pt \
    --parquet_path /mnt/a100_1_data2/wuqingman/datasets/OmniSVG/MMSVGBench/data/text2svg-00000-of-00001.parquet \
    --retrieval_hvm_dir /mnt/a100_1_data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_22w_nozoom_top3part \
    --output_dir ./inference_results/mmsvgbench_text2svg_hvm_step5000 \
    --num_candidates 5 \
    --save_png \
    --resume
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    pd = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from inference_hvm_s1_test import (  # noqa: E402
    EXTRA_CANDIDATES_BUFFER,
    SVG_CONFIG_PATH,
    SVGTokenizer,
    clear_hvm_memory,
    generate_svg,
    get_available_gpus,
    load_hvm_model,
    prepare_text_inputs,
    prepare_visual_prefix_inputs,
    render_svg_to_image,
    set_hvm_memory,
    split_indices,
    validate_candidate,
)

GROUPS_PER_REFERENCE = 4
DEFAULT_CLIP_MODEL_PATH = "/mnt/a100_1_data/wuqingman/models/openai/clip-vit-large-patch14"


def load_benchmark_records(parquet_path: str) -> List[Dict[str, Any]]:
    if pd is None:
        raise ImportError("pandas is required to read parquet benchmark files.")

    df = pd.read_parquet(parquet_path)
    if "text" not in df.columns:
        raise KeyError(f"'text' column not found in {parquet_path}")

    if "task_type" in df.columns:
        df = df[df["task_type"] == "text2svg"].reset_index(drop=True)

    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        records.append(
            {
                "id": row.get("id"),
                "text": text.strip(),
                "task_type": row.get("task_type"),
                "type": row.get("type"),
                "url": row.get("url"),
            }
        )

    return records


def build_retrieval_results(
    records: List[Dict[str, Any]],
    retrieval_hvm_dir: str,
    clip_model_path: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    from transformers import CLIPModel, CLIPTokenizer

    print(f"Loading CLIP retrieval model from {clip_model_path} ...")
    is_local_clip = os.path.isdir(clip_model_path)
    clip_model = CLIPModel.from_pretrained(
        clip_model_path,
        local_files_only=is_local_clip,
    )
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        clip_model_path,
        local_files_only=is_local_clip,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = clip_model.to(device).eval()

    texts = [r["text"] for r in records]
    embed_dim = clip_model.config.projection_dim
    embeddings = np.zeros((len(texts), embed_dim), dtype=np.float32)
    batch_size = 256

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding benchmark texts"):
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

    scores = None
    indices = None
    try:
        import faiss  # type: ignore

        index_path = os.path.join(retrieval_hvm_dir, "faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        print(f"Loading FAISS index from {index_path} ...")
        index = faiss.read_index(index_path)
        print(f"FAISS index size: {index.ntotal}")
        print(f"Searching Top-{top_k} refs from retrieval corpus with FAISS ...")
        scores, indices = index.search(embeddings, top_k)
    except ImportError:
        embeddings_path = os.path.join(retrieval_hvm_dir, "text_embeddings.npy")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"text_embeddings.npy not found for retrieval fallback: {embeddings_path}"
            )

        print("faiss is not installed; falling back to chunked matrix retrieval from text_embeddings.npy ...")
        corpus_embeddings = np.load(embeddings_path, mmap_mode="r")
        query = torch.from_numpy(embeddings).to(device)
        query = query / query.norm(dim=-1, keepdim=True)

        num_queries = query.shape[0]
        top_scores = torch.full((num_queries, top_k), float("-inf"), device=device)
        top_indices = torch.full((num_queries, top_k), -1, dtype=torch.long, device=device)
        chunk_size = 16384

        for start in tqdm(range(0, corpus_embeddings.shape[0], chunk_size), desc="Searching corpus chunks"):
            end = min(start + chunk_size, corpus_embeddings.shape[0])
            chunk_np = np.asarray(corpus_embeddings[start:end], dtype=np.float32)
            chunk = torch.from_numpy(chunk_np).to(device)
            chunk = chunk / chunk.norm(dim=-1, keepdim=True)
            score_chunk = torch.matmul(query, chunk.transpose(0, 1))
            local_k = min(top_k, score_chunk.shape[1])
            local_scores, local_indices = torch.topk(score_chunk, k=local_k, dim=1)
            local_indices = local_indices + start

            merged_scores = torch.cat([top_scores, local_scores], dim=1)
            merged_indices = torch.cat([top_indices, local_indices], dim=1)
            top_scores, select = torch.topk(merged_scores, k=top_k, dim=1)
            top_indices = torch.gather(merged_indices, 1, select)

            del chunk, score_chunk, local_scores, local_indices, merged_scores, merged_indices, select
            if device.type == "cuda":
                torch.cuda.empty_cache()

        scores = top_scores.cpu().numpy()
        indices = top_indices.cpu().numpy()

    results: List[Dict[str, Any]] = []
    for i in range(len(records)):
        results.append(
            {
                "ref_indices": [int(indices[i, j]) for j in range(top_k)],
                "ref_scores": [float(scores[i, j]) for j in range(top_k)],
            }
        )
    return results


def load_jsonl_subset(path: str, keep_indices: set[int]) -> Dict[int, Dict[str, Any]]:
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


class CorpusFeatureStore:
    def __init__(
        self,
        retrieval_hvm_dir: str,
        ref_meta_lookup: Dict[int, Dict[str, Any]],
        ref_groups_lookup: Dict[int, Dict[str, Any]],
        part_num_refs: int = 3,
    ):
        self.retrieval_hvm_dir = retrieval_hvm_dir
        self.features_dir = os.path.join(retrieval_hvm_dir, "features")
        self.group_features_dir = os.path.join(retrieval_hvm_dir, "group_features")
        self.ref_meta_lookup = ref_meta_lookup
        self.ref_groups_lookup = ref_groups_lookup
        self.part_num_refs = max(1, int(part_num_refs))

    def _ref_feat_path(self, ref_idx: int) -> str:
        return os.path.join(self.features_dir, f"{ref_idx // 1000:03d}", f"{ref_idx:06d}.pt")

    def _group_feat_path(self, ref_idx: int) -> str:
        return os.path.join(self.group_features_dir, f"{ref_idx // 1000:03d}", f"{ref_idx:06d}.pt")

    def _build_group_tag_meta(self, ref_idx: int, num_groups: Optional[int] = None) -> torch.Tensor:
        viewbox_size = 200.0
        group_record = self.ref_groups_lookup.get(ref_idx, {})
        groups = list(group_record.get("groups", []))
        if num_groups is not None:
            groups = groups[:num_groups]

        denom = max(int(group_record.get("num_paths", 0)) - 1, 1)
        tag_meta = []
        for group in groups:
            bbox = group.get("bbox", [0.0, 0.0, viewbox_size, viewbox_size])
            if len(bbox) != 4:
                bbox = [0.0, 0.0, viewbox_size, viewbox_size]
            x0, y0, x1, y1 = [float(v) for v in bbox]
            path_indices = [int(v) for v in group.get("path_indices", [])]
            if path_indices:
                z_start = min(path_indices) / denom
                z_end = max(path_indices) / denom
            else:
                z_start = 0.0
                z_end = 0.0

            cx = ((x0 + x1) * 0.5) / viewbox_size
            cy = ((y0 + y1) * 0.5) / viewbox_size
            w = max(x1 - x0, 0.0) / viewbox_size
            h = max(y1 - y0, 0.0) / viewbox_size
            tag_meta.append([cx, cy, w, h, z_start, z_end])

        if not tag_meta:
            return torch.zeros(0, 6, dtype=torch.float32)
        return torch.tensor(tag_meta, dtype=torch.float32)

    def load_ref_feature(self, ref_idx: int) -> torch.Tensor:
        return torch.load(self._ref_feat_path(ref_idx), map_location="cpu", weights_only=True)

    def load_group_features(self, ref_idx: int) -> Dict[str, Any]:
        raw = torch.load(self._group_feat_path(ref_idx), map_location="cpu", weights_only=False)

        part_mask = None
        tag_meta = None
        group_ids = None
        if isinstance(raw, dict):
            if "part_features" in raw:
                raw_features = raw["part_features"]
            elif "group_features" in raw:
                raw_features = raw["group_features"]
            elif "features" in raw:
                raw_features = raw["features"]
            else:
                raise KeyError(f"No feature tensor found in {self._group_feat_path(ref_idx)}")
            part_mask = raw.get("part_mask")
            tag_meta = raw.get("tag_meta")
            group_ids = raw.get("group_ids")
        else:
            raw_features = raw

        if isinstance(raw_features, torch.Tensor):
            if raw_features.ndim == 2:
                raw_features = raw_features.unsqueeze(0)
            if part_mask is not None:
                mask_tensor = torch.as_tensor(part_mask, dtype=torch.bool)
                raw_features = raw_features[mask_tensor]
            group_features = [raw_features[i] for i in range(raw_features.shape[0])]
        else:
            group_features = list(raw_features)
            if part_mask is not None:
                mask_tensor = torch.as_tensor(part_mask, dtype=torch.bool)
                group_features = [feat for feat, keep in zip(group_features, mask_tensor.tolist()) if keep]

        group_features = [torch.as_tensor(feat) for feat in group_features]
        num_groups = len(group_features)

        derived_tag_meta = self._build_group_tag_meta(ref_idx, num_groups=num_groups)
        if tag_meta is None:
            tag_meta = derived_tag_meta
        else:
            tag_meta = torch.as_tensor(tag_meta, dtype=torch.float32)
            if tag_meta.ndim == 1:
                tag_meta = tag_meta.unsqueeze(0)
            if tag_meta.shape[0] < num_groups:
                pad = derived_tag_meta[tag_meta.shape[0]:num_groups]
                tag_meta = torch.cat([tag_meta, pad], dim=0) if pad.numel() > 0 else tag_meta
            else:
                tag_meta = tag_meta[:num_groups]

        if group_ids is None:
            group_ids = torch.arange(num_groups, dtype=torch.long)
        else:
            group_ids = torch.as_tensor(group_ids, dtype=torch.long)[:num_groups]

        aligned_len = min(num_groups, tag_meta.shape[0], group_ids.shape[0])
        return {
            "group_features": group_features[:aligned_len],
            "tag_meta": tag_meta[:aligned_len],
            "group_ids": group_ids[:aligned_len],
        }

    def load_part_group_bundle(self, ref_indices: List[int]) -> Dict[str, Any]:
        merged_group_features: List[torch.Tensor] = []
        merged_tag_meta: List[torch.Tensor] = []
        merged_group_ids: List[torch.Tensor] = []

        for ref_rank, ref_idx in enumerate(ref_indices[:self.part_num_refs]):
            group_data = self.load_group_features(ref_idx)
            group_features = list(group_data["group_features"])
            tag_meta = torch.as_tensor(group_data["tag_meta"], dtype=torch.float32)
            group_ids = torch.as_tensor(group_data["group_ids"], dtype=torch.long)

            if not group_features:
                continue

            aligned_len = min(len(group_features), tag_meta.shape[0], group_ids.shape[0])
            if aligned_len <= 0:
                continue

            merged_group_features.extend(group_features[:aligned_len])
            merged_tag_meta.append(tag_meta[:aligned_len])
            merged_group_ids.append(group_ids[:aligned_len] + ref_rank * GROUPS_PER_REFERENCE)

        if not merged_group_features:
            return {
                "group_features": [],
                "tag_meta": torch.zeros(0, 6, dtype=torch.float32),
                "group_ids": torch.zeros(0, dtype=torch.long),
            }

        return {
            "group_features": merged_group_features,
            "tag_meta": torch.cat(merged_tag_meta, dim=0),
            "group_ids": torch.cat(merged_group_ids, dim=0),
        }

    def build_ref_text(self, ref_indices: List[int]) -> str:
        ref_texts = []
        for ref_idx in ref_indices:
            desc = self.ref_meta_lookup.get(ref_idx, {}).get("description", "")
            if desc:
                ref_texts.append(desc)
        return " ".join(ref_texts)


def save_sample_meta(
    output_dir: Path,
    idx: int,
    record: Dict[str, Any],
    retrieval: Dict[str, Any],
    ref_meta_lookup: Dict[int, Dict[str, Any]],
    candidate_files: List[str],
    elapsed_s: float,
    status: str,
) -> None:
    refs = []
    ref_indices = retrieval.get("ref_indices", [])
    ref_scores = retrieval.get("ref_scores", [])
    for pos, ref_idx in enumerate(ref_indices):
        ref_meta = ref_meta_lookup.get(ref_idx, {})
        refs.append(
            {
                "ref_index": ref_idx,
                "score": ref_scores[pos] if pos < len(ref_scores) else None,
                "description": ref_meta.get("description", ""),
                "detail": ref_meta.get("detail", ""),
                "parquet_file": ref_meta.get("parquet_file"),
                "parquet_row": ref_meta.get("parquet_row"),
            }
        )

    meta = {
        "index": idx,
        "id": record.get("id"),
        "text": record.get("text"),
        "task_type": record.get("task_type"),
        "type": record.get("type"),
        "url": record.get("url"),
        "retrieval": refs,
        "status": status,
        "elapsed_s": elapsed_s,
        "candidate_files": candidate_files,
    }
    (output_dir / f"sample_{idx:04d}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_on_single_gpu(
    local_rank: int,
    gpu_id: int,
    sample_indices: List[int],
    records: List[Dict[str, Any]],
    retrieval_results: List[Dict[str, Any]],
    ref_meta_lookup: Dict[int, Dict[str, Any]],
    ref_groups_lookup: Dict[int, Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    device = f"cuda:{gpu_id}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not sample_indices:
        print(f"[GPU {gpu_id}] No samples assigned, exiting.")
        return

    print(
        f"\n[GPU {gpu_id}] Assigned {len(sample_indices)} samples: "
        f"{sample_indices[0]} ~ {sample_indices[-1]}"
    )

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
    feature_store = CorpusFeatureStore(
        retrieval_hvm_dir=args.retrieval_hvm_dir,
        ref_meta_lookup=ref_meta_lookup,
        ref_groups_lookup=ref_groups_lookup,
        part_num_refs=args.retrieval_top_k,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_ok = 0
    total_fail = 0
    total_skipped = 0

    pbar = tqdm(sample_indices, desc=f"[GPU {gpu_id}] Inference", position=local_rank)
    for idx in pbar:
        if idx >= len(records):
            pbar.write(f"[GPU {gpu_id}] Skip sample {idx}: out of range")
            continue

        if args.resume:
            has_single = (output_dir / f"sample_{idx:04d}_hvm.svg").exists()
            if has_single:
                total_skipped += 1
                pbar.set_postfix(ok=total_ok, fail=total_fail, skip=total_skipped)
                continue
            min_cands = args.min_candidates or 1
            existing_count = sum(
                1 for ci in range(args.num_candidates)
                if (output_dir / f"sample_{idx:04d}_hvm_c{ci}.svg").exists()
            )
            if existing_count >= min_cands:
                total_skipped += 1
                pbar.set_postfix(ok=total_ok, fail=total_fail, skip=total_skipped)
                continue

        record = records[idx]
        retrieval = retrieval_results[idx]
        ref_indices = retrieval["ref_indices"]
        text = record["text"]
        t0 = time.time()

        input_ids, attention_mask = prepare_text_inputs(
            text, processor, token_config, device
        )

        ref_features = [feature_store.load_ref_feature(ref_idx) for ref_idx in ref_indices]
        ref_best_group_data = feature_store.load_part_group_bundle(ref_indices)
        ref_text = feature_store.build_ref_text(ref_indices)

        set_hvm_memory(
            hvm_model,
            ref_features=ref_features,
            group_features=ref_best_group_data["group_features"],
            ref_text=ref_text,
            tokenizer=tokenizer,
            hvm_config=hvm_config,
            device=device,
            group_tag_meta=ref_best_group_data["tag_meta"],
            group_ids=ref_best_group_data["group_ids"],
        )

        actual_num = args.num_candidates + EXTRA_CANDIDATES_BUFFER
        gen_inputs_embeds = None
        gen_attention_mask = attention_mask
        if hvm_config.memory_mode == "visual_prefix" and getattr(hvm_model, "_visual_prefix", None) is not None:
            gen_inputs_embeds, gen_attention_mask = prepare_visual_prefix_inputs(
                hvm_model, input_ids, attention_mask, device
            )
        candidates = generate_svg(
            transformer_for_generate,
            input_ids,
            gen_attention_mask,
            token_config,
            svg_tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=actual_num,
            inputs_embeds=gen_inputs_embeds,
        )

        clear_hvm_memory(hvm_model)
        elapsed = time.time() - t0

        if candidates and not args.no_validate:
            valid = []
            for cand in candidates:
                if validate_candidate(cand["svg_str"]):
                    valid.append(cand)
                    if len(valid) >= args.num_candidates:
                        break
            candidates = valid

        candidate_files: List[str] = []
        if candidates:
            total_ok += 1
            for ci, cand in enumerate(candidates):
                suffix = f"_c{ci}" if len(candidates) > 1 else ""
                base = f"sample_{idx:04d}_hvm{suffix}"

                svg_name = f"{base}.svg"
                (output_dir / svg_name).write_text(cand["svg_str"], encoding="utf-8")
                candidate_files.append(svg_name)

                if args.save_png:
                    img = render_svg_to_image(cand["svg_str"])
                    if img is not None:
                        img.save(str(output_dir / f"{base}.png"))

                if args.save_tokens:
                    (output_dir / f"{base}_tokens.json").write_text(
                        json.dumps(
                            {
                                "id": record.get("id"),
                                "text": text,
                                "retrieval": retrieval,
                                "tokens": cand["tokens"],
                                "num_tokens": len(cand["tokens"]),
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )

            save_sample_meta(
                output_dir,
                idx,
                record,
                retrieval,
                ref_meta_lookup,
                candidate_files,
                elapsed,
                status="ok",
            )
        else:
            total_fail += 1
            pbar.write(f"[GPU {gpu_id}] Sample {idx} FAILED ({elapsed:.1f}s)")
            save_sample_meta(
                output_dir,
                idx,
                record,
                retrieval,
                ref_meta_lookup,
                candidate_files,
                elapsed,
                status="failed",
            )

        (output_dir / f"sample_{idx:04d}.txt").write_text(text, encoding="utf-8")
        pbar.set_postfix(ok=total_ok, fail=total_fail, skip=total_skipped, last_t=f"{elapsed:.1f}s")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"\n[GPU {gpu_id}] Done: success={total_ok}, fail={total_fail}, "
        f"skipped={total_skipped}, total={len(sample_indices)}"
    )


def worker(
    local_rank: int,
    args: argparse.Namespace,
    index_splits: List[List[int]],
    records: List[Dict[str, Any]],
    retrieval_results: List[Dict[str, Any]],
    ref_meta_lookup: Dict[int, Dict[str, Any]],
    ref_groups_lookup: Dict[int, Dict[str, Any]],
) -> None:
    run_on_single_gpu(
        local_rank=local_rank,
        gpu_id=local_rank,
        sample_indices=index_splits[local_rank],
        records=records,
        retrieval_results=retrieval_results,
        ref_meta_lookup=ref_meta_lookup,
        ref_groups_lookup=ref_groups_lookup,
        args=args,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-to-SVG benchmark inference from parquet (multi-GPU)")

    p.add_argument("--model_size", type=str, default="8B", choices=["4B", "8B"])
    p.add_argument("--config_dir", type=str, default=None)
    p.add_argument("--base_model", type=str, default=None,
                   help="Qwen base model 路径，覆盖 tokenization.yaml 中的 base_model")
    p.add_argument("--hvm_checkpoint", type=str, required=True,
                   help="HVM checkpoint 路径")
    p.add_argument("--hvm_config", type=str, default=None,
                   help="HVM config 路径，默认自动从 checkpoint 目录读取")
    p.add_argument("--omnisvg_checkpoint", type=str, default=None,
                   help="OmniSVG checkpoint 路径，覆盖 tokenization.yaml 中的 checkpoint")

    p.add_argument(
        "--parquet_path",
        type=str,
        default="/mnt/a100_1_data2/wuqingman/datasets/OmniSVG/MMSVGBench/data/text2svg-00000-of-00001.parquet",
        help="MMSVGBench text2svg parquet 路径",
    )
    p.add_argument(
        "--retrieval_hvm_dir",
        type=str,
        default="/mnt/a100_1_data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_22w_nozoom_top3part",
        help="全量训练库的 HVM 预计算目录（含 faiss_index / features / group_features）",
    )
    p.add_argument(
        "--clip_model_path",
        type=str,
        default=DEFAULT_CLIP_MODEL_PATH,
        help="用于文本检索的 CLIP 模型路径",
    )
    p.add_argument(
        "--retrieval_top_k",
        type=int,
        default=3,
        help="从全量训练库中检索几个 top refs",
    )
    p.add_argument(
        "--sample_indices",
        type=int,
        nargs="+",
        default=None,
        help="要推理的样本索引，例如: --sample_indices $(seq 0 299)",
    )

    p.add_argument("--num_gpus", type=int, default=None,
                   help="使用几张卡。默认=所有 CUDA_VISIBLE_DEVICES 可见的卡数。")

    p.add_argument("--max_new_tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=0.90)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--num_candidates", type=int, default=5,
                   help="每个样本生成几个候选 SVG")
    p.add_argument("--no_validate", action="store_true", default=False,
                   help="跳过 SVG 验证 (渲染检查)")

    p.add_argument("--output_dir", type=str, default="./inference_results/mmsvgbench_text2svg",
                   help="推理结果保存目录")
    p.add_argument("--save_png", action="store_true", default=False,
                   help="保存 PNG 渲染图")
    p.add_argument("--save_tokens", action="store_true", default=False,
                   help="保存生成的 token 序列")
    p.add_argument("--resume", action="store_true", default=False,
                   help="跳过已经生成的样本 (断点续推)")
    p.add_argument("--min_candidates", type=int, default=None,
                   help="Resume 时要求已有候选数 >= 此值才跳过 (默认: 只要 c0 存在就跳过)")

    args = p.parse_args()

    if args.hvm_config is None:
        ckpt_dir = Path(args.hvm_checkpoint).parent
        candidate = ckpt_dir / "hvm_model_config.json"
        if candidate.exists():
            args.hvm_config = str(candidate)
        else:
            raise FileNotFoundError(
                f"Cannot find hvm_model_config.json in {ckpt_dir}. "
                "Please specify --hvm_config explicitly."
            )

    return args


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    records = load_benchmark_records(args.parquet_path)
    if not records:
        raise RuntimeError(f"No valid text2svg records found in {args.parquet_path}")

    retrieval_results = build_retrieval_results(
        records=records,
        retrieval_hvm_dir=args.retrieval_hvm_dir,
        clip_model_path=args.clip_model_path,
        top_k=args.retrieval_top_k,
    )
    retrieved_ref_ids = {
        ref_idx
        for result in retrieval_results
        for ref_idx in result["ref_indices"]
    }
    metadata_path = os.path.join(args.retrieval_hvm_dir, "metadata.jsonl")
    groups_path = os.path.join(args.retrieval_hvm_dir, "groups_train_ref.jsonl")
    if not os.path.exists(groups_path):
        groups_path = os.path.join(args.retrieval_hvm_dir, "groups.jsonl")

    print(f"Loading metadata subset for {len(retrieved_ref_ids)} retrieved refs ...")
    ref_meta_lookup = load_jsonl_subset(metadata_path, retrieved_ref_ids)
    ref_groups_lookup = load_jsonl_subset(groups_path, retrieved_ref_ids)

    if args.sample_indices is None:
        pending_indices = list(range(len(records)))
    else:
        pending_indices = list(args.sample_indices)

    available_gpus = get_available_gpus()
    num_gpus_available = len(available_gpus)
    if num_gpus_available == 0:
        raise RuntimeError("No CUDA GPU detected! Check CUDA_VISIBLE_DEVICES.")

    num_gpus = min(args.num_gpus, num_gpus_available) if args.num_gpus is not None else num_gpus_available

    if args.resume:
        output_dir = Path(args.output_dir)
        min_cands = args.min_candidates or 1
        completed = []
        for idx in pending_indices:
            has_single = (output_dir / f"sample_{idx:04d}_hvm.svg").exists()
            if has_single:
                completed.append(idx)
                continue
            existing_count = sum(
                1 for ci in range(args.num_candidates)
                if (output_dir / f"sample_{idx:04d}_hvm_c{ci}.svg").exists()
            )
            if existing_count >= min_cands:
                completed.append(idx)
        pending_indices = [i for i in pending_indices if i not in set(completed)]
    else:
        completed = []

    print("=" * 70)
    print("MMSVGBench Text2SVG Inference  [HVM + Retrieval]  --  Multi-GPU Data Parallel")
    print("=" * 70)
    print(f"  Model size       : {args.model_size}")
    print(f"  Mode             : retrieve top-{args.retrieval_top_k} refs + inject HVM memory")
    print(f"  HVM checkpoint   : {args.hvm_checkpoint}")
    print(f"  HVM config       : {args.hvm_config}")
    print(f"  Parquet path     : {args.parquet_path}")
    print(f"  Retrieval HVM dir: {args.retrieval_hvm_dir}")
    print(f"  Total records    : {len(records)}")
    print(f"  Selected samples : {len(pending_indices)}")
    print(f"  Output dir       : {args.output_dir}")
    print(f"  Num candidates   : {args.num_candidates}")
    print(f"  Available GPUs   : {num_gpus_available}  (using {num_gpus})")
    print(f"  Resume mode      : {args.resume}")
    if args.resume:
        print(f"  Resume summary   : {len(completed)} done, {len(pending_indices)} remaining"
              + (f" (min_candidates={min_cands})" if args.min_candidates else ""))
    print("=" * 70)

    index_splits = split_indices(pending_indices, num_gpus)
    for i, split in enumerate(index_splits):
        print(f"  GPU {i}: {len(split)} samples" + (f"  [{split[0]}~{split[-1]}]" if split else "  [empty]"))
    print("=" * 70)

    if num_gpus == 1:
        run_on_single_gpu(
            local_rank=0,
            gpu_id=0,
            sample_indices=index_splits[0],
            records=records,
            retrieval_results=retrieval_results,
            ref_meta_lookup=ref_meta_lookup,
            ref_groups_lookup=ref_groups_lookup,
            args=args,
        )
    else:
        mp.start_processes(
            worker,
            args=(args, index_splits, records, retrieval_results, ref_meta_lookup, ref_groups_lookup),
            nprocs=num_gpus,
            start_method="spawn",
        )

    print(f"\n{'=' * 70}")
    print(f"All done! Results saved to: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
