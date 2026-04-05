#!/usr/bin/env python3
"""
HVM 正式定性推理脚本（动态任务队列版）。

工作流：
1. prepare: 根据 decoded jsonl + offline top3 cache 生成待跑任务
2. worker:  单进程单卡/单进程共享卡 worker，动态抢任务直到跑完
3. rebuild-index: 从 pack 下的 records 重建 index.jsonl

设计目标：
- 支持 12 个独立进程动态抢任务，不固定绑卡
- 每个 sample 生成 4 个 candidate
- 结果直接按 pack 存储，pack 内固定容纳 3000 个 sample
- resume 时重新 prepare，只把未完成样本重新切成小任务
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:
    import pandas as pd
except ImportError:
    pd = None

INFERENCE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(INFERENCE_DIR, ".."))
sys.path.insert(0, INFERENCE_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:  # noqa: E402
    from inference_hvm_multigpu import (
        EXTRA_CANDIDATES_BUFFER,
        SVG_CONFIG_PATH,
        SVGTokenizer,
        generate_svg,
        load_hvm_model,
        prepare_text_inputs,
        render_svg_to_image,
        validate_candidate,
    )
except ModuleNotFoundError:  # noqa: E402
    from inference.inference_hvm_multigpu import (
        EXTRA_CANDIDATES_BUFFER,
        SVG_CONFIG_PATH,
        SVGTokenizer,
        generate_svg,
        load_hvm_model,
        prepare_text_inputs,
        render_svg_to_image,
        validate_candidate,
    )


DEFAULT_DECODED_JSONL = Path(
    "/mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k_decoded.jsonl"
)
DEFAULT_RETRIEVAL_JSONL = Path(
    "/mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k_top3_retrieval.jsonl"
)
DEFAULT_HVM_DIR = Path(
    "/mnt/data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_22w_nozoom_top3part"
)
DEFAULT_RUN_DIR = Path(
    "/mnt/data3/wuqingman/omnisvg-train/qualitative_runs/main_s9_queue"
)
DEFAULT_HVM_CONFIG = Path(
    "/mnt/data3/wuqingman/omnisvg-train/outputs_s9_full25w_top3part_12slot_nogist_edr_parttag_nozoom_last4/hvm_config.json"
)
DEFAULT_PARQUET_ROOT = Path(
    "/mnt/data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_process"
)
DEFAULT_CONFIG_DIR = Path("/mnt/data2/wuqingman/omnisvg-train/configs")

VIEWBOX_SIZE = 200.0
GROUPS_PER_REFERENCE = 4


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def safe_write_json(path: Path, obj: Dict[str, Any]) -> None:
    safe_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def format_sample_id(sample_id: int) -> str:
    return f"{int(sample_id):05d}"


def get_pack_name(sample_id: int, pack_size: int) -> str:
    return f"pack_{int(sample_id) // int(pack_size):04d}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunked(items: Sequence[int], chunk_size: int) -> Iterable[List[int]]:
    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


def detect_hvm_config_path(hvm_checkpoint: Optional[str], explicit_path: Optional[str]) -> str:
    if explicit_path:
        return explicit_path
    if not hvm_checkpoint:
        raise ValueError("hvm_checkpoint is required when hvm_config is not specified.")
    ckpt_dir = Path(hvm_checkpoint).resolve().parent
    for candidate_name in ("hvm_config.json", "hvm_model_config.json"):
        candidate = ckpt_dir / candidate_name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Cannot find hvm_config.json near checkpoint: {hvm_checkpoint}")


def normalize_query_row(row: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = row.get("idx", row.get("id"))
    if sample_id is None:
        raise KeyError("Query row must contain either 'id' or 'idx'.")

    text = row.get("text")
    if text is None:
        text = row.get("description")
    if text is None:
        raise KeyError("Query row must contain either 'text' or 'description'.")

    normalized: Dict[str, Any] = {
        "id": int(sample_id),
        "text": str(text),
    }

    source_id = row.get("source_id")
    raw_id = row.get("id")
    if source_id is not None:
        normalized["source_id"] = str(source_id)
    elif raw_id is not None and not isinstance(raw_id, int):
        normalized["source_id"] = str(raw_id)

    svg = row.get("svg")
    if svg is not None:
        normalized["svg"] = str(svg)

    parquet_file = row.get("parquet_file")
    parquet_row = row.get("parquet_row")
    if parquet_file is not None and parquet_row is not None:
        normalized["parquet_file"] = str(parquet_file)
        normalized["parquet_row"] = int(parquet_row)

    return normalized


def load_query_rows(path: Path) -> List[Dict[str, Any]]:
    return [normalize_query_row(row) for row in iter_jsonl(path)]


def normalize_retrieval_row(row: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = row.get("idx", row.get("id"))
    if sample_id is None:
        raise KeyError("Retrieval row must contain either 'id' or 'idx'.")

    ref_indices = row.get("ref_indices")
    if ref_indices is None:
        raise KeyError("Retrieval row must contain 'ref_indices'.")

    normalized: Dict[str, Any] = {
        "id": int(sample_id),
        "ref_indices": [int(v) for v in ref_indices],
    }

    text = row.get("text")
    if text is None:
        text = row.get("description")
    if text is not None:
        normalized["text"] = str(text)

    if "ref_scores" in row and row["ref_scores"] is not None:
        normalized["ref_scores"] = [float(v) for v in row["ref_scores"]]
    if "ref_ids" in row and row["ref_ids"] is not None:
        normalized["ref_ids"] = [str(v) for v in row["ref_ids"]]
    if "ref_descriptions" in row and row["ref_descriptions"] is not None:
        normalized["ref_descriptions"] = [str(v) for v in row["ref_descriptions"]]

    return normalized


def load_retrieval_rows(path: Path) -> List[Dict[str, Any]]:
    return [normalize_retrieval_row(row) for row in iter_jsonl(path)]


class QuerySVGStore:
    def __init__(self, parquet_root: Optional[Path]):
        self.parquet_root = parquet_root.resolve() if parquet_root else None
        self._cached_parquet_path: Optional[Path] = None
        self._cached_svg_column: Optional[List[str]] = None

    def get_svg(self, query: Dict[str, Any]) -> str:
        svg = query.get("svg")
        if svg is not None:
            return str(svg)

        parquet_file = query.get("parquet_file")
        parquet_row = query.get("parquet_row")
        if parquet_file is None or parquet_row is None:
            raise KeyError(f"Query id={query.get('id')} does not contain svg or parquet location.")
        if self.parquet_root is None:
            raise ValueError("parquet_root is required when query rows omit svg.")
        if pd is None:
            raise ImportError("pandas is required to load svg from parquet rows.")

        parquet_path = self.parquet_root / str(parquet_file)
        if self._cached_parquet_path != parquet_path:
            if not parquet_path.exists():
                raise FileNotFoundError(f"Missing parquet file for query svg lookup: {parquet_path}")
            df = pd.read_parquet(parquet_path, columns=["svg"])
            self._cached_parquet_path = parquet_path
            self._cached_svg_column = [str(v) for v in df["svg"].tolist()]

        row_idx = int(parquet_row)
        if self._cached_svg_column is None or row_idx < 0 or row_idx >= len(self._cached_svg_column):
            raise IndexError(
                f"Query id={query.get('id')} requests parquet_row={row_idx}, but cache size is "
                f"{0 if self._cached_svg_column is None else len(self._cached_svg_column)}."
            )
        return self._cached_svg_column[row_idx]


def resolve_hvm_dir_query_inputs(hvm_dir: Path) -> Tuple[Path, Path]:
    decoded_path = hvm_dir / "metadata.jsonl"
    retrieval_candidates = [
        hvm_dir / "rag_results.jsonl",
        hvm_dir / "rag_results_train.jsonl",
    ]
    retrieval_path = next((candidate for candidate in retrieval_candidates if candidate.exists()), None)

    if not decoded_path.exists():
        raise FileNotFoundError(f"Missing metadata.jsonl in hvm_dir: {hvm_dir}")
    if retrieval_path is None:
        raise FileNotFoundError(
            f"Missing rag_results.jsonl / rag_results_train.jsonl in hvm_dir: {hvm_dir}"
        )
    return decoded_path, retrieval_path


def resolve_retrieval_metadata(
    retrieval: Dict[str, Any],
    query_by_id: Dict[int, Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    ref_indices = [int(v) for v in retrieval.get("ref_indices", [])]
    raw_ref_ids = list(retrieval.get("ref_ids", []))
    raw_ref_descriptions = list(retrieval.get("ref_descriptions", []))

    resolved_ref_ids: List[str] = []
    resolved_ref_descriptions: List[str] = []
    for pos, ref_idx in enumerate(ref_indices):
        ref_query = query_by_id.get(int(ref_idx))

        ref_id = raw_ref_ids[pos] if pos < len(raw_ref_ids) else None
        if ref_id is None and ref_query is not None:
            ref_id = ref_query.get("source_id") or ref_query.get("id")

        ref_description = raw_ref_descriptions[pos] if pos < len(raw_ref_descriptions) else None
        if ref_description is None and ref_query is not None:
            ref_description = ref_query.get("text")

        if ref_id is not None:
            resolved_ref_ids.append(str(ref_id))
        if ref_description is not None:
            resolved_ref_descriptions.append(str(ref_description))

    return resolved_ref_ids, resolved_ref_descriptions


class ReferenceFeatureStore:
    def __init__(self, hvm_dir: Path):
        self.hvm_dir = hvm_dir
        self.features_dir = hvm_dir / "features"
        self.group_features_dir = hvm_dir / "group_features"

        groups_path = hvm_dir / "groups_train_ref.jsonl"
        if not groups_path.exists():
            groups_path = hvm_dir / "groups.jsonl"
        if not groups_path.exists():
            raise FileNotFoundError(f"Missing groups jsonl in {hvm_dir}")

        self.idx_to_groups = {int(row["idx"]): row for row in load_jsonl(groups_path)}

    def _ref_feat_path(self, ref_idx: int) -> Path:
        return self.features_dir / f"{ref_idx // 1000:03d}" / f"{ref_idx:06d}.pt"

    def _group_feat_path(self, ref_idx: int) -> Path:
        return self.group_features_dir / f"{ref_idx // 1000:03d}" / f"{ref_idx:06d}.pt"

    def load_ref_features(self, ref_indices: Sequence[int]) -> List[torch.Tensor]:
        feats = []
        for ref_idx in ref_indices:
            feat = torch.load(self._ref_feat_path(int(ref_idx)), map_location="cpu", weights_only=True)
            feats.append(torch.as_tensor(feat))
        return feats

    def _build_group_tag_meta(self, ref_idx: int, num_groups: Optional[int] = None) -> torch.Tensor:
        group_record = self.idx_to_groups.get(int(ref_idx), {})
        groups = list(group_record.get("groups", []))
        if num_groups is not None:
            groups = groups[:num_groups]

        denom = max(int(group_record.get("num_paths", 0)) - 1, 1)
        tag_meta: List[List[float]] = []
        for group in groups:
            bbox = group.get("bbox", [0.0, 0.0, VIEWBOX_SIZE, VIEWBOX_SIZE])
            if len(bbox) != 4:
                bbox = [0.0, 0.0, VIEWBOX_SIZE, VIEWBOX_SIZE]
            x0, y0, x1, y1 = [float(v) for v in bbox]
            path_indices = [int(v) for v in group.get("path_indices", [])]
            if path_indices:
                z_start = min(path_indices) / denom
                z_end = max(path_indices) / denom
            else:
                z_start = 0.0
                z_end = 0.0
            cx = ((x0 + x1) * 0.5) / VIEWBOX_SIZE
            cy = ((y0 + y1) * 0.5) / VIEWBOX_SIZE
            w = max(x1 - x0, 0.0) / VIEWBOX_SIZE
            h = max(y1 - y0, 0.0) / VIEWBOX_SIZE
            tag_meta.append([cx, cy, w, h, z_start, z_end])

        if not tag_meta:
            return torch.zeros(0, 6, dtype=torch.float32)
        return torch.tensor(tag_meta, dtype=torch.float32)

    def _load_group_features(self, ref_idx: int) -> Dict[str, Any]:
        gf_path = self._group_feat_path(int(ref_idx))
        raw = torch.load(gf_path, map_location="cpu", weights_only=False)

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
                raise KeyError(f"No feature tensor found in {gf_path}")
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

        derived_tag_meta = self._build_group_tag_meta(int(ref_idx), num_groups=num_groups)
        if tag_meta is None:
            tag_meta = derived_tag_meta
        else:
            tag_meta = torch.as_tensor(tag_meta, dtype=torch.float32)
            if tag_meta.ndim == 1:
                tag_meta = tag_meta.unsqueeze(0)
            if tag_meta.shape[0] < num_groups:
                pad = derived_tag_meta[tag_meta.shape[0] : num_groups]
                if pad.numel() > 0:
                    tag_meta = torch.cat([tag_meta, pad], dim=0)
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

    def load_part_group_bundle(self, ref_indices: Sequence[int], part_num_refs: int) -> Dict[str, torch.Tensor]:
        merged_group_features: List[torch.Tensor] = []
        merged_tag_meta: List[torch.Tensor] = []
        merged_group_ids: List[torch.Tensor] = []

        for ref_rank, ref_idx in enumerate(ref_indices[:part_num_refs]):
            group_data = self._load_group_features(int(ref_idx))
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
            raise RuntimeError("No part group features loaded for selected references.")

        part_features = torch.stack(merged_group_features, dim=0).unsqueeze(0)
        tag_meta = torch.cat(merged_tag_meta, dim=0).unsqueeze(0)
        group_ids = torch.cat(merged_group_ids, dim=0).unsqueeze(0)
        part_mask = torch.ones(1, part_features.shape[1], dtype=torch.bool)
        return {
            "part_features": part_features,
            "part_tag_meta": tag_meta,
            "part_group_ids": group_ids,
            "part_mask": part_mask,
        }


def clear_hvm_runtime_memory(model: torch.nn.Module) -> None:
    model._memory_ready = False
    model._gist_feats = None
    model._part_feats = None
    model._text_feats = None
    model._part_mask = None
    model._text_mask = None
    model._ref_feats = None
    model._detail_feats = None
    model._detail_slot_mask = None
    model._local_part_feats = None
    model._local_part_mask = None


def prepare_hvm_runtime_memory(
    model: torch.nn.Module,
    hvm_config,
    ref_feature_tensor: torch.Tensor,
    part_bundle: Dict[str, torch.Tensor],
) -> None:
    clear_hvm_runtime_memory(model)
    device = next(model.base_model.parameters()).device

    if model.gme is not None:
        hvm_dtype = next(model.gme.parameters()).dtype
    elif model.local_token_encoder is not None:
        hvm_dtype = next(model.local_token_encoder.parameters()).dtype
    elif len(model.pims) > 0:
        hvm_dtype = next(model.pims.parameters()).dtype
    else:
        raise RuntimeError("Cannot determine HVM dtype.")

    ref_feature_tensor = ref_feature_tensor.to(device=device, dtype=hvm_dtype)
    model._memory_ready = True

    if hvm_config.memory_mode == "gme":
        model._gist_feats = model.gme(ref_feature_tensor)
        return

    if hvm_config.memory_mode == "gme_dra":
        model._gist_feats = model.gme(ref_feature_tensor)
        model._ref_feats = ref_feature_tensor.view(ref_feature_tensor.shape[0], -1, ref_feature_tensor.shape[-1])
        return

    if hvm_config.memory_mode == "dense_global_local":
        part_features = part_bundle["part_features"].to(device=device, dtype=hvm_dtype)
        part_tag_meta = part_bundle["part_tag_meta"].to(device=device, dtype=hvm_dtype)
        part_group_ids = part_bundle["part_group_ids"].to(device=device)
        part_mask = part_bundle["part_mask"].to(device=device, dtype=torch.bool)
        model._ref_feats = ref_feature_tensor.view(ref_feature_tensor.shape[0], -1, ref_feature_tensor.shape[-1])
        model._local_part_feats, model._local_part_mask = model.local_token_encoder(
            part_features,
            part_tag_meta=part_tag_meta,
            part_group_ids=part_group_ids,
            part_mask=part_mask,
        )
        return

    if hvm_config.memory_mode in ("gme_cdm", "gme_cdm_edr"):
        part_features = part_bundle["part_features"].to(device=device, dtype=hvm_dtype)
        part_tag_meta = part_bundle["part_tag_meta"].to(device=device, dtype=hvm_dtype)
        part_group_ids = part_bundle["part_group_ids"].to(device=device)
        part_mask = part_bundle["part_mask"].to(device=device, dtype=torch.bool)
        model._gist_feats = model.gme(ref_feature_tensor)
        model._detail_feats = model.cdm(
            part_features,
            model._gist_feats.detach(),
            part_tag_meta=part_tag_meta,
            part_group_ids=part_group_ids,
            part_mask=part_mask,
        )
        model._detail_slot_mask = getattr(model.cdm, "last_detail_slot_mask", None)
        return

    raise NotImplementedError(
        f"Queue inference currently supports memory_mode in "
        f"['gme', 'gme_dra', 'gme_cdm', 'gme_cdm_edr', 'dense_global_local'], "
        f"got: {hvm_config.memory_mode}"
    )


def get_pack_paths(run_dir: Path, sample_id: int, pack_size: int) -> Dict[str, Path]:
    pack_dir = run_dir / "packs" / get_pack_name(sample_id, pack_size)
    return {
        "pack_dir": ensure_dir(pack_dir),
        "pred_svg_dir": ensure_dir(pack_dir / "pred_svg"),
        "pred_png_dir": ensure_dir(pack_dir / "pred_png"),
        "gt_png_dir": ensure_dir(pack_dir / "gt_png"),
        "gt_svg_dir": ensure_dir(pack_dir / "gt_svg"),
        "gt_txt_dir": ensure_dir(pack_dir / "gt_txt"),
        "records_dir": ensure_dir(pack_dir / "records"),
    }


def load_record(record_path: Path) -> Optional[Dict[str, Any]]:
    if not record_path.exists():
        return None
    return json.loads(record_path.read_text(encoding="utf-8"))


def list_existing_candidate_paths(pred_svg_dir: Path, sample_id: int) -> List[Path]:
    sid = format_sample_id(sample_id)
    return sorted(pred_svg_dir.glob(f"{sid}_c*.svg"))


def rebuild_pack_index(pack_dir: Path) -> None:
    records_dir = pack_dir / "records"
    index_path = pack_dir / "index.jsonl"
    lock_path = pack_dir / ".index.lock"
    ensure_dir(records_dir)

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        rows: List[Dict[str, Any]] = []
        for record_path in sorted(records_dir.glob("*.json")):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "id": record["id"],
                    "source_id": record.get("source_id"),
                    "status": record.get("status", "unknown"),
                    "text": record.get("text", ""),
                    "num_candidates_done": record.get("num_candidates_done", 0),
                    "target_num_candidates": record.get("target_num_candidates", 0),
                    "candidate_svg_files": record.get("candidate_svg_files", []),
                    "candidate_png_files": record.get("candidate_png_files", []),
                    "gt_png_file": record.get("gt_png_file"),
                    "ref_indices": record.get("ref_indices", []),
                    "ref_ids": record.get("ref_ids", []),
                    "ref_scores": record.get("ref_scores", []),
                    "last_worker": record.get("last_worker"),
                    "updated_at": record.get("updated_at"),
                }
            )
        tmp_path = index_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, index_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def scan_completed_sample_ids(run_dir: Path, target_num_candidates: int) -> set[int]:
    completed: set[int] = set()
    packs_dir = run_dir / "packs"
    if not packs_dir.exists():
        return completed

    for record_path in packs_dir.glob("pack_*/records/*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidate_svg_files = record.get("candidate_svg_files", [])
        if (
            record.get("status") == "done"
            and int(record.get("num_candidates_done", 0)) >= target_num_candidates
            and len(candidate_svg_files) >= target_num_candidates
        ):
            completed.add(int(record["id"]))
    return completed


def clear_task_files(tasks_dir: Path) -> None:
    ensure_dir(tasks_dir)
    for path in tasks_dir.glob("task_*"):
        if path.is_file():
            path.unlink()


def write_task_file(task_path: Path, sample_ids: Sequence[int]) -> None:
    with task_path.open("w", encoding="utf-8") as f:
        for sample_id in sample_ids:
            f.write(json.dumps({"id": int(sample_id)}) + "\n")


def claim_next_task(tasks_dir: Path, worker_name: str) -> Optional[Path]:
    pending_paths = sorted(tasks_dir.glob("task_*.pending.jsonl"))
    for pending_path in pending_paths:
        running_path = pending_path.with_name(
            pending_path.name.replace(".pending.jsonl", f".running.{worker_name}.jsonl")
        )
        try:
            os.replace(pending_path, running_path)
            return running_path
        except FileNotFoundError:
            continue
    return None


def finalize_task_file(running_path: Path) -> Path:
    done_path = running_path.with_name(running_path.name.split(".running.")[0] + ".done.jsonl")
    os.replace(running_path, done_path)
    return done_path


def load_task_sample_ids(task_path: Path) -> List[int]:
    return [int(row["id"]) for row in load_jsonl(task_path)]


def parse_sample_ids_arg(spec: Optional[str]) -> Optional[List[int]]:
    if not spec:
        return None
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def cmd_prepare(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    ensure_dir(run_dir)
    ensure_dir(run_dir / "packs")
    tasks_dir = ensure_dir(run_dir / "tasks")

    hvm_config_path = detect_hvm_config_path(args.hvm_checkpoint, args.hvm_config)
    hvm_config_payload = json.loads(Path(hvm_config_path).read_text(encoding="utf-8"))
    model_size = args.model_size or hvm_config_payload.get("model_size", "8B")
    part_num_refs = int(args.part_num_refs or hvm_config_payload.get("part_num_refs", 3))

    decoded_jsonl = args.decoded_jsonl
    retrieval_jsonl = args.retrieval_jsonl
    if args.use_hvm_dir_index:
        decoded_jsonl, retrieval_jsonl = resolve_hvm_dir_query_inputs(args.hvm_dir.resolve())

    query_rows = load_query_rows(decoded_jsonl)
    retrieval_rows = load_retrieval_rows(retrieval_jsonl)
    query_ids = {int(row["id"]) for row in query_rows}
    retrieval_ids = {int(row["id"]) for row in retrieval_rows}
    available_ids = sorted(query_ids & retrieval_ids)

    selected_ids = available_ids
    explicit_sample_ids = parse_sample_ids_arg(args.sample_ids)
    if explicit_sample_ids is not None:
        selected_ids = [sid for sid in explicit_sample_ids if sid in query_ids and sid in retrieval_ids]
    else:
        if args.offset:
            selected_ids = selected_ids[args.offset :]
        if args.limit is not None:
            selected_ids = selected_ids[: args.limit]

    completed_ids = set()
    if args.resume:
        completed_ids = scan_completed_sample_ids(run_dir, args.num_candidates)
    pending_ids = [sid for sid in selected_ids if sid not in completed_ids]

    run_config = {
        "created_at": now_iso(),
        "run_dir": str(run_dir),
        "decoded_jsonl": str(Path(decoded_jsonl).resolve()),
        "retrieval_jsonl": str(Path(retrieval_jsonl).resolve()),
        "parquet_root": str(args.parquet_root.resolve()) if args.parquet_root else None,
        "hvm_dir": str(args.hvm_dir.resolve()),
        "hvm_checkpoint": str(Path(args.hvm_checkpoint).resolve()),
        "hvm_config": str(Path(hvm_config_path).resolve()),
        "omnisvg_checkpoint": str(Path(args.omnisvg_checkpoint).resolve()) if args.omnisvg_checkpoint else None,
        "config_dir": str((args.config_dir or DEFAULT_CONFIG_DIR).resolve()),
        "model_size": model_size,
        "part_num_refs": part_num_refs,
        "task_size": int(args.task_size),
        "pack_size": int(args.pack_size),
        "num_candidates": int(args.num_candidates),
        "max_attempts_per_sample": int(args.max_attempts_per_sample),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "repetition_penalty": float(args.repetition_penalty),
        "no_validate": bool(args.no_validate),
        "save_png": not bool(args.no_save_png),
        "total_selected_samples": len(selected_ids),
        "pending_samples": len(pending_ids),
        "resume": bool(args.resume),
    }
    safe_write_json(run_dir / "config.json", run_config)

    clear_task_files(tasks_dir)
    if pending_ids:
        for chunk in chunked(pending_ids, int(args.task_size)):
            first_id = format_sample_id(chunk[0])
            last_id = format_sample_id(chunk[-1])
            task_path = tasks_dir / f"task_{first_id}_{last_id}.pending.jsonl"
            write_task_file(task_path, chunk)

    print("=" * 60)
    print(f"Run dir            : {run_dir}")
    print(f"Decoded jsonl      : {Path(decoded_jsonl).resolve()}")
    print(f"Retrieval jsonl    : {Path(retrieval_jsonl).resolve()}")
    print(f"Selected samples   : {len(selected_ids)}")
    print(f"Completed detected : {len(completed_ids)}")
    print(f"Pending samples    : {len(pending_ids)}")
    print(f"Task size          : {args.task_size}")
    print(f"Tasks written      : {len(list(tasks_dir.glob('task_*.pending.jsonl')))}")
    print(f"Config             : {run_dir / 'config.json'}")
    print("=" * 60)


def process_single_sample(
    sample_id: int,
    query_by_id: Dict[int, Dict[str, Any]],
    query_svg_store: QuerySVGStore,
    retrieval_by_id: Dict[int, Dict[str, Any]],
    ref_store: ReferenceFeatureStore,
    hvm_model,
    tokenizer,
    processor,
    token_config,
    hvm_config,
    transformer_for_generate,
    svg_tokenizer,
    run_config: Dict[str, Any],
    run_dir: Path,
    worker_name: str,
) -> Dict[str, Any]:
    query = query_by_id[int(sample_id)]
    retrieval = retrieval_by_id[int(sample_id)]
    ref_ids, ref_descriptions = resolve_retrieval_metadata(retrieval, query_by_id)
    pack_paths = get_pack_paths(run_dir, int(sample_id), int(run_config["pack_size"]))
    sid = format_sample_id(int(sample_id))
    record_path = pack_paths["records_dir"] / f"{sid}.json"
    existing_record = load_record(record_path) or {}

    existing_svg_paths = list_existing_candidate_paths(pack_paths["pred_svg_dir"], int(sample_id))
    existing_candidate_files = [f"pred_svg/{p.name}" for p in existing_svg_paths]
    existing_candidate_png_files = [
        f"pred_png/{sid}_c{idx}.png"
        for idx in range(len(existing_svg_paths))
        if (pack_paths["pred_png_dir"] / f"{sid}_c{idx}.png").exists()
    ]
    num_candidates_done = len(existing_svg_paths)
    target_num_candidates = int(run_config["num_candidates"])

    gt_png_rel = f"gt_png/{sid}.png"
    gt_png_path = pack_paths["gt_png_dir"] / f"{sid}.png"
    query_svg = query_svg_store.get_svg(query)
    if run_config.get("save_png", True) and not gt_png_path.exists() and query_svg:
        gt_img = render_svg_to_image(query_svg)
        if gt_img is not None:
            gt_img.save(gt_png_path)

    gt_svg_path = pack_paths["gt_svg_dir"] / f"{sid}.svg"
    if not gt_svg_path.exists() and query_svg:
        safe_write_text(gt_svg_path, query_svg)

    gt_txt_path = pack_paths["gt_txt_dir"] / f"{sid}.txt"
    if not gt_txt_path.exists():
        safe_write_text(gt_txt_path, query["text"])

    record: Dict[str, Any] = {
        "id": int(sample_id),
        "source_id": query.get("source_id"),
        "parquet_file": query.get("parquet_file"),
        "parquet_row": query.get("parquet_row"),
        "status": "running",
        "text": query["text"],
        "ref_indices": retrieval["ref_indices"],
        "ref_ids": ref_ids,
        "ref_scores": retrieval.get("ref_scores", []),
        "ref_descriptions": ref_descriptions,
        "ref_text": " ".join(ref_descriptions),
        "target_num_candidates": target_num_candidates,
        "num_candidates_done": num_candidates_done,
        "candidate_svg_files": existing_candidate_files,
        "candidate_png_files": existing_candidate_png_files,
        "gt_png_file": gt_png_rel if gt_png_path.exists() else None,
        "last_worker": worker_name,
        "attempts": int(existing_record.get("attempts", 0)),
        "updated_at": now_iso(),
        "errors": list(existing_record.get("errors", [])),
    }
    safe_write_json(record_path, record)

    if num_candidates_done >= target_num_candidates:
        record["status"] = "done"
        safe_write_json(record_path, record)
        rebuild_pack_index(pack_paths["pack_dir"])
        return record

    ref_indices = [int(v) for v in retrieval["ref_indices"][: int(run_config["part_num_refs"])]]
    ref_features = ref_store.load_ref_features(ref_indices)
    ref_feature_tensor = torch.stack(ref_features, dim=0).unsqueeze(0)
    part_bundle = ref_store.load_part_group_bundle(
        ref_indices,
        part_num_refs=int(run_config["part_num_refs"]),
    )

    device = next(hvm_model.base_model.parameters()).device
    input_ids, attention_mask = prepare_text_inputs(query["text"], processor, token_config, device=str(device))
    prepare_hvm_runtime_memory(
        hvm_model,
        hvm_config=hvm_config,
        ref_feature_tensor=ref_feature_tensor,
        part_bundle=part_bundle,
    )

    seen_svgs = set()
    for path in existing_svg_paths:
        try:
            seen_svgs.add(path.read_text(encoding="utf-8"))
        except Exception:
            continue

    try:
        while record["num_candidates_done"] < target_num_candidates and record["attempts"] < int(run_config["max_attempts_per_sample"]):
            remaining = target_num_candidates - int(record["num_candidates_done"])
            request_n = remaining + EXTRA_CANDIDATES_BUFFER
            record["attempts"] += 1
            record["updated_at"] = now_iso()
            safe_write_json(record_path, record)

            candidates = generate_svg(
                transformer_for_generate,
                input_ids,
                attention_mask,
                token_config,
                svg_tokenizer,
                max_new_tokens=int(run_config["max_new_tokens"]),
                temperature=float(run_config["temperature"]),
                top_p=float(run_config["top_p"]),
                top_k=int(run_config["top_k"]),
                repetition_penalty=float(run_config["repetition_penalty"]),
                num_return_sequences=request_n,
            )

            accepted_any = False
            for cand in candidates:
                svg_str = cand["svg_str"]
                if svg_str in seen_svgs:
                    continue
                if not run_config.get("no_validate", False) and not validate_candidate(svg_str):
                    continue

                cand_idx = int(record["num_candidates_done"])
                svg_filename = f"{sid}_c{cand_idx}.svg"
                png_filename = f"{sid}_c{cand_idx}.png"
                safe_write_text(pack_paths["pred_svg_dir"] / svg_filename, svg_str)
                record["candidate_svg_files"].append(f"pred_svg/{svg_filename}")

                if run_config.get("save_png", True):
                    pred_img = render_svg_to_image(svg_str)
                    pred_img.save(pack_paths["pred_png_dir"] / png_filename)
                    record["candidate_png_files"].append(f"pred_png/{png_filename}")

                record["num_candidates_done"] = cand_idx + 1
                record["updated_at"] = now_iso()
                safe_write_json(record_path, record)
                seen_svgs.add(svg_str)
                accepted_any = True

                if record["num_candidates_done"] >= target_num_candidates:
                    break

            if not accepted_any:
                record["errors"].append(
                    {
                        "attempt": int(record["attempts"]),
                        "stage": "generate",
                        "message": "no new valid candidates accepted",
                    }
                )
                safe_write_json(record_path, record)
                break

        if record["num_candidates_done"] >= target_num_candidates:
            record["status"] = "done"
        elif record["num_candidates_done"] > 0:
            record["status"] = "partial"
        else:
            record["status"] = "failed"
        record["updated_at"] = now_iso()
        safe_write_json(record_path, record)
        rebuild_pack_index(pack_paths["pack_dir"])
        return record
    except Exception as exc:  # noqa: BLE001
        record["status"] = "failed"
        record["updated_at"] = now_iso()
        record["errors"].append(
            {
                "attempt": int(record["attempts"]),
                "stage": "exception",
                "message": str(exc),
            }
        )
        safe_write_json(record_path, record)
        rebuild_pack_index(pack_paths["pack_dir"])
        raise
    finally:
        clear_hvm_runtime_memory(hvm_model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cmd_worker(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    run_config = load_run_config(run_dir)
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Missing tasks dir: {tasks_dir}")

    if not torch.cuda.is_available():
        raise RuntimeError("worker mode requires CUDA. Use prepare mode on CPU if needed.")

    worker_name = args.worker_name or f"{socket.gethostname()}_pid{os.getpid()}"
    device = "cuda:0"

    hvm_checkpoint = args.hvm_checkpoint or run_config["hvm_checkpoint"]
    hvm_config_path = detect_hvm_config_path(hvm_checkpoint, args.hvm_config or run_config.get("hvm_config"))
    model_size = args.model_size or run_config["model_size"]
    config_dir = args.config_dir or run_config["config_dir"]
    omnisvg_checkpoint = args.omnisvg_checkpoint or run_config.get("omnisvg_checkpoint")

    print("=" * 70)
    print(f"Starting queue worker: {worker_name}")
    print(f"  Run dir        : {run_dir}")
    print(f"  Device         : {device}")
    print(f"  HVM checkpoint : {hvm_checkpoint}")
    print(f"  HVM config     : {hvm_config_path}")
    print("=" * 70)

    hvm_model, tokenizer, processor, token_config, hvm_config = load_hvm_model(
        model_size=model_size,
        hvm_config_path=hvm_config_path,
        hvm_checkpoint_path=hvm_checkpoint,
        config_dir=config_dir,
        omnisvg_checkpoint=omnisvg_checkpoint,
        device=device,
    )
    transformer_for_generate = hvm_model.base_model.transformer
    svg_tokenizer = SVGTokenizer(SVG_CONFIG_PATH, model_size=model_size)

    query_by_id = {int(row["id"]): row for row in load_query_rows(Path(run_config["decoded_jsonl"]))}
    retrieval_by_id = {int(row["id"]): row for row in load_retrieval_rows(Path(run_config["retrieval_jsonl"]))}
    parquet_root = args.parquet_root or run_config.get("parquet_root")
    query_svg_store = QuerySVGStore(Path(parquet_root) if parquet_root else None)
    ref_store = ReferenceFeatureStore(Path(run_config["hvm_dir"]))

    total_done = 0
    total_partial = 0
    total_failed = 0

    while True:
        running_task = claim_next_task(tasks_dir, worker_name)
        if running_task is None:
            print(f"[{worker_name}] No pending task left, exiting.")
            break

        sample_ids = load_task_sample_ids(running_task)
        print(f"[{worker_name}] Claimed {running_task.name} ({len(sample_ids)} samples)")
        for sample_id in sample_ids:
            if sample_id not in query_by_id or sample_id not in retrieval_by_id:
                print(f"[{worker_name}] Skip missing sample id={sample_id}")
                continue
            t0 = time.time()
            try:
                record = process_single_sample(
                    sample_id=sample_id,
                    query_by_id=query_by_id,
                    query_svg_store=query_svg_store,
                    retrieval_by_id=retrieval_by_id,
                    ref_store=ref_store,
                    hvm_model=hvm_model,
                    tokenizer=tokenizer,
                    processor=processor,
                    token_config=token_config,
                    hvm_config=hvm_config,
                    transformer_for_generate=transformer_for_generate,
                    svg_tokenizer=svg_tokenizer,
                    run_config=run_config,
                    run_dir=run_dir,
                    worker_name=worker_name,
                )
                elapsed = time.time() - t0
                print(
                    f"[{worker_name}] sample={sample_id} status={record['status']} "
                    f"cand={record['num_candidates_done']}/{record['target_num_candidates']} "
                    f"time={elapsed:.1f}s"
                )
                if record["status"] == "done":
                    total_done += 1
                elif record["status"] == "partial":
                    total_partial += 1
                else:
                    total_failed += 1
            except Exception as exc:  # noqa: BLE001
                elapsed = time.time() - t0
                print(f"[{worker_name}] sample={sample_id} exception after {elapsed:.1f}s: {exc}")
                total_failed += 1

        finalize_task_file(running_task)

    print("=" * 70)
    print(
        f"[{worker_name}] finished: done={total_done}, partial={total_partial}, failed={total_failed}"
    )
    print("=" * 70)


def cmd_rebuild_index(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    packs_dir = run_dir / "packs"
    count = 0
    for pack_dir in sorted(packs_dir.glob("pack_*")):
        rebuild_pack_index(pack_dir)
        count += 1
    print(f"Rebuilt pack index for {count} packs under {packs_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HVM qualitative inference queue runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prepare = subparsers.add_parser("prepare", help="Prepare queue tasks from decoded jsonl + retrieval cache")
    p_prepare.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    p_prepare.add_argument(
        "--decoded_jsonl",
        type=Path,
        default=DEFAULT_DECODED_JSONL,
        help="Query source jsonl. Supports decoded jsonl or metadata.jsonl with parquet_file/parquet_row.",
    )
    p_prepare.add_argument(
        "--retrieval_jsonl",
        type=Path,
        default=DEFAULT_RETRIEVAL_JSONL,
        help="Retrieval source jsonl. Supports legacy top3 jsonl or rag_results.jsonl keyed by idx.",
    )
    p_prepare.add_argument(
        "--parquet_root",
        type=Path,
        default=DEFAULT_PARQUET_ROOT,
        help="Root directory for original training parquets. Used only when query rows omit svg.",
    )
    p_prepare.add_argument(
        "--use_hvm_dir_index",
        action="store_true",
        default=False,
        help="Directly use hvm_dir/metadata.jsonl and hvm_dir/rag_results*.jsonl as query/retrieval sources.",
    )
    p_prepare.add_argument("--hvm_dir", type=Path, default=DEFAULT_HVM_DIR)
    p_prepare.add_argument("--hvm_checkpoint", type=str, required=True)
    p_prepare.add_argument("--hvm_config", type=str, default=str(DEFAULT_HVM_CONFIG))
    p_prepare.add_argument("--omnisvg_checkpoint", type=str, default=None)
    p_prepare.add_argument("--config_dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p_prepare.add_argument("--model_size", type=str, default="8B")
    p_prepare.add_argument("--part_num_refs", type=int, default=None)
    p_prepare.add_argument("--task_size", type=int, default=100)
    p_prepare.add_argument("--pack_size", type=int, default=3000)
    p_prepare.add_argument("--num_candidates", type=int, default=4)
    p_prepare.add_argument("--max_attempts_per_sample", type=int, default=3)
    p_prepare.add_argument("--max_new_tokens", type=int, default=3000)
    p_prepare.add_argument("--temperature", type=float, default=0.5)
    p_prepare.add_argument("--top_p", type=float, default=0.9)
    p_prepare.add_argument("--top_k", type=int, default=50)
    p_prepare.add_argument("--repetition_penalty", type=float, default=1.05)
    p_prepare.add_argument("--no_validate", action="store_true", default=False)
    p_prepare.add_argument("--no_save_png", action="store_true", default=False)
    p_prepare.add_argument("--limit", type=int, default=None)
    p_prepare.add_argument("--offset", type=int, default=0)
    p_prepare.add_argument("--sample_ids", type=str, default=None)
    p_prepare.add_argument("--resume", action="store_true", default=False)

    p_worker = subparsers.add_parser("worker", help="Run one queue worker")
    p_worker.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    p_worker.add_argument("--worker_name", type=str, default=None)
    p_worker.add_argument("--parquet_root", type=Path, default=None)
    p_worker.add_argument("--hvm_checkpoint", type=str, default=None)
    p_worker.add_argument("--hvm_config", type=str, default=None)
    p_worker.add_argument("--omnisvg_checkpoint", type=str, default=None)
    p_worker.add_argument("--config_dir", type=str, default=None)
    p_worker.add_argument("--model_size", type=str, default=None)

    p_rebuild = subparsers.add_parser("rebuild-index", help="Rebuild pack index.jsonl from per-sample records")
    p_rebuild.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "worker":
        cmd_worker(args)
    elif args.command == "rebuild-index":
        cmd_rebuild_index(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
