"""
HVM-SVG Dataset
================
加载 OmniSVG 原始数据 (parquet) + HVM 预计算数据 (features, RAG, groups)。

每个样本返回:
  - text: 描述文本
  - pix_seq: SVG token 序列
  - ref_features: 3 张参考图的 pre-merge feature maps
  - ref_best_feature: Top-1 参考图的 feature map
  - ref_best_groups: Top-1 参考图的 path 分组信息
  - ref_text: 3 张参考图的拼接描述
"""

import io
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    from deepsvg.svglib.svg import SVG
    DEEPSVG_AVAILABLE = True
except ImportError:
    DEEPSVG_AVAILABLE = False
    print("[HVM Dataset] Warning: deepsvg not available. SVG tokenization will fail.")

from utils.config import TokenizationConfig, TrainConfig
from utils.dataset import SVGTokenizer


class HVMDataset(Dataset):
    """
    HVM-SVG 训练数据集。

    同时加载:
    1. 原始 parquet 数据 (SVG 字符串, 描述, 图像)
    2. 预计算的 HVM 数据 (vision features, RAG results, path groups)
    """

    def __init__(
        self,
        data_dir: str,
        hvm_dir: str,
        token_config: TokenizationConfig,
        train_config: Optional[TrainConfig] = None,
        max_len: int = 2048,
    ):
        """
        Args:
            data_dir: 原始 parquet 文件目录
            hvm_dir: HVM 预计算数据目录
            token_config: SVG tokenization 配置
            train_config: 训练配置
            max_len: 最大 SVG token 序列长度
        """
        self.data_dir = data_dir
        self.hvm_dir = hvm_dir
        self.max_len = max_len
        self.token_config = token_config
        self.train_config = train_config or TrainConfig()
        self.features_dir = os.path.join(hvm_dir, "features")

        # SVG tokenizer
        self.svg_tokenizer = SVGTokenizer(token_config)

        # 加载预计算数据
        print("[HVM Dataset] Loading precomputed data...")
        self.metadata = self._load_jsonl(os.path.join(hvm_dir, "metadata.jsonl"))
        self.rag_results = self._load_jsonl(os.path.join(hvm_dir, "rag_results.jsonl"))
        self.groups_data = self._load_jsonl(os.path.join(hvm_dir, "groups.jsonl"))

        # 建立 idx → metadata/rag/groups 的快速查找
        self.idx_to_meta = {r["idx"]: r for r in self.metadata}
        self.idx_to_rag = {r["idx"]: r for r in self.rag_results}
        self.idx_to_groups = {r["idx"]: r for r in self.groups_data}

        # 有效的样本 indices (必须同时有 metadata, rag, groups, features)
        self.valid_indices = self._build_valid_indices()

        # 加载 parquet tables（延迟加载）
        self._parquet_tables = {}

        print(f"[HVM Dataset] Ready: {len(self.valid_indices)} valid samples")

    def _load_jsonl(self, path: str) -> List[Dict]:
        records = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        print(f"  Loaded {len(records)} records from {os.path.basename(path)}")
        return records

    def _build_valid_indices(self) -> List[int]:
        """构建有效样本索引：必须同时有所有预计算数据"""
        valid = []
        for meta in self.metadata:
            idx = meta["idx"]
            # 检查 feature 文件是否存在
            feat_path = os.path.join(
                self.features_dir, f"{idx // 1000:03d}", f"{idx:06d}.pt"
            )
            if (
                idx in self.idx_to_rag
                and idx in self.idx_to_groups
                and os.path.exists(feat_path)
            ):
                # 检查参考样本的 features 是否也存在
                rag = self.idx_to_rag[idx]
                refs_ok = True
                for ref_idx in rag["ref_indices"]:
                    ref_feat_path = os.path.join(
                        self.features_dir, f"{ref_idx // 1000:03d}", f"{ref_idx:06d}.pt"
                    )
                    if not os.path.exists(ref_feat_path):
                        refs_ok = False
                        break
                if refs_ok:
                    valid.append(idx)

        return valid

    def _get_parquet_table(self, parquet_file: str):
        """延迟加载 parquet table"""
        if parquet_file not in self._parquet_tables:
            import pyarrow.parquet as pq
            full_path = os.path.join(self.data_dir, parquet_file)
            self._parquet_tables[parquet_file] = pq.read_table(full_path)
        return self._parquet_tables[parquet_file]

    def _load_feature(self, idx: int) -> torch.Tensor:
        """加载单个样本的 feature map"""
        feat_path = os.path.join(
            self.features_dir, f"{idx // 1000:03d}", f"{idx:06d}.pt"
        )
        return torch.load(feat_path, map_location="cpu", weights_only=True)  # [32, 32, 1280]

    def _tokenize_svg(self, svg_code: str) -> np.ndarray:
        """将 SVG 字符串 tokenize 为 token 序列"""
        if not svg_code or not DEEPSVG_AVAILABLE:
            return np.array([], dtype=np.int64)

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".svg", delete=False) as f:
                f.write(svg_code)
                temp_path = f.name

            svg = SVG.load_svg(temp_path)
            svg_tensors, color_tensors = svg.to_tensor(concat_groups=False, PAD_VAL=0)
            tokens = self.svg_tokenizer.tokenize_svg_tensors(svg_tensors, color_tensors)
            return tokens

        except Exception as e:
            # 静默处理错误，训练时会重试
            return np.array([], dtype=np.int64)

        finally:
            # 无论成功或异常，都清理临时文件
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns:
            dict with keys:
                text: str                              描述文本
                pix_seq: List[int]                     SVG token 序列 (含 BOS/EOS)
                ref_features: List[torch.Tensor]       3 × [32,32,1280]
                ref_best_groups: List[Tuple]            Top-1 参考的 bbox_feature 列表
                ref_text: str                          参考文本 (拼接)
        """
        max_retries = 10
        for retry in range(max_retries):
            try:
                actual_index = (index + retry) % len(self.valid_indices)
                idx = self.valid_indices[actual_index]
                return self._load_sample(idx)
            except Exception as e:
                if retry == max_retries - 1:
                    raise RuntimeError(f"Failed to load sample after {max_retries} retries: {e}")
                continue

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """加载单个样本的全部数据"""
        meta = self.idx_to_meta[idx]
        rag = self.idx_to_rag[idx]

        # ---- 从 parquet 加载原始数据 ----
        table = self._get_parquet_table(meta["parquet_file"])
        row = meta["parquet_row"]

        # 描述文本（随机选 detail 或 description）
        description = table.column("description")[row].as_py()
        detail = table.column("detail")[row].as_py() or ""
        if detail and random.random() < self.train_config.detail_prob:
            text = detail
        else:
            text = description
        text = str(text).strip()
        if len(text) < 5:
            text = detail or description
        if len(text) > 500:
            text = text[:500]

        # SVG tokenization
        svg_code = table.column("svg")[row].as_py()
        tokens = self._tokenize_svg(svg_code)

        if len(tokens) == 0:
            raise ValueError(f"Empty SVG tokens for idx={idx}")

        # 截断到 max_len
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]

        # 添加 BOS/EOS
        tokens = self.svg_tokenizer.add_special_tokens(tokens)
        pix_seq = tokens.tolist()

        # ---- 加载参考数据 ----
        ref_indices = rag["ref_indices"]  # [3]

        # 3 张参考图的 features
        ref_features = [self._load_feature(ri) for ri in ref_indices]

        # Top-1 参考的分组信息
        best_ref_idx = ref_indices[0]
        best_ref_groups = self.idx_to_groups.get(best_ref_idx, {})
        groups_list = best_ref_groups.get("groups", [])
        ref_best_groups = [
            tuple(g["bbox_feature"]) for g in groups_list
        ]
        if not ref_best_groups:
            # Fallback: 整图作为一个 group
            ref_best_groups = [(0, 32, 0, 32)]

        # 参考文本 (拼接 3 个参考的描述)
        ref_texts = []
        for ri in ref_indices:
            ref_meta = self.idx_to_meta.get(ri, {})
            ref_desc = ref_meta.get("description", "")
            if ref_desc:
                ref_texts.append(ref_desc)
        ref_text = " ".join(ref_texts)

        return {
            "text": text,
            "pix_seq": pix_seq,
            "ref_features": ref_features,      # List of 3 tensors
            "ref_best_groups": ref_best_groups,  # List of tuples
            "ref_text": ref_text,
        }


# ============================================================================
# Collate Function
# ============================================================================

def create_hvm_collate_fn(
    processor: Any,
    tokenizer: Any,
    token_config: TokenizationConfig,
    text_len: int = 800,
    max_seq_length: int = 2048,
    ref_text_max_length: int = 128,
):
    """
    创建 HVM 训练的 collate function。

    将 Dataset 返回的 raw samples 组装成 batch tensor:
    - input_ids, attention_mask, labels: 文本 + SVG 序列
    - ref_features: [B, 3, 32, 32, 1280]
    - ref_best_feature: [B, 32, 32, 1280]
    - groups_bbox_feature: List[List[Tuple]]
    - ref_text_ids, ref_text_mask: [B, N_t]
    """
    system_prompt = "You are an expert SVG code generator."
    pad_token_id = token_config.pad_token_id
    max_len = max_seq_length + text_len

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        texts = [s["text"] for s in batch]
        pix_seqs = [s["pix_seq"] for s in batch]
        ref_features_list = [s["ref_features"] for s in batch]
        ref_best_groups_list = [s["ref_best_groups"] for s in batch]
        ref_texts = [s["ref_text"] for s in batch]

        # ================================================================
        # 1. 构建 input_ids / attention_mask / labels
        # ================================================================
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for text, pix_seq in zip(texts, pix_seqs):
            # 构建 chat messages (text-to-SVG task)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Generate SVG code for this text description: {text}"}
                ]},
            ]

            # 用 processor 的 chat template tokenize
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text_input],
                padding=False,
                truncation=False,
                return_tensors=None,
            )

            base_input_ids = inputs["input_ids"][0]
            base_attention_mask = inputs["attention_mask"][0]

            # 拼接: [text_tokens] + [svg_tokens]
            current_input_ids = base_input_ids + pix_seq
            current_attention_mask = base_attention_mask + [1] * len(pix_seq)

            # Labels: text 部分为 -100, SVG 部分为 token IDs
            instruction_len = len(base_input_ids)
            current_labels = [-100] * instruction_len + pix_seq

            # Padding (左侧)
            pad_len = max_len - len(current_input_ids)
            if pad_len > 0:
                input_ids = [pad_token_id] * pad_len + current_input_ids
                attention_mask = [0] * pad_len + current_attention_mask
                labels = [-100] * pad_len + current_labels
            else:
                input_ids = current_input_ids[:max_len]
                attention_mask = current_attention_mask[:max_len]
                labels = current_labels[:max_len]

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))

        input_ids = torch.stack(batch_input_ids)          # [B, L]
        attention_mask = torch.stack(batch_attention_mask)  # [B, L]
        labels = torch.stack(batch_labels)                  # [B, L]

        # ================================================================
        # 2. 组装参考图 features
        # ================================================================
        # ref_features: [B, 3, 32, 32, 1280]
        ref_features = torch.stack([
            torch.stack(rf) for rf in ref_features_list
        ])  # [B, 3, 32, 32, 1280]

        # ref_best_feature: [B, 32, 32, 1280] (Top-1)
        ref_best_feature = ref_features[:, 0]

        # ================================================================
        # 3. 参考分组信息 (保持 List 格式，PME 内部逐样本处理)
        # ================================================================
        groups_bbox_feature = ref_best_groups_list

        # ================================================================
        # 4. 参考文本 tokenization
        # ================================================================
        ref_text_encoded = tokenizer(
            ref_texts,
            padding=True,
            truncation=True,
            max_length=ref_text_max_length,
            return_tensors="pt",
        )
        ref_text_ids = ref_text_encoded["input_ids"]         # [B, N_t]
        ref_text_mask = ref_text_encoded["attention_mask"]    # [B, N_t]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ref_features": ref_features,
            "ref_best_feature": ref_best_feature,
            "groups_bbox_feature": groups_bbox_feature,
            "ref_text_ids": ref_text_ids,
            "ref_text_mask": ref_text_mask,
        }

    return collate_fn
