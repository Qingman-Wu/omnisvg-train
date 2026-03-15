"""
HVM-SVG Dataset
================
加载 OmniSVG 原始数据 (parquet) + HVM 预计算数据 (features, RAG, groups)。

每个样本返回:
  - text: 描述文本
  - pix_seq: SVG token 序列
  - ref_features: 3 张参考图的 post-merge features [256, 3584] (for GME)
  - ref_best_group_features: 选中 part refs 的逐 group 独立渲染特征 list of [256, 3584]
  - ref_best_group_tag_meta: 选中 part refs 的结构 tag [G, 6]
  - ref_best_group_ids: 选中 part refs 的全局 group_id [G]
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

VIEWBOX_SIZE = 200.0
GROUPS_PER_REFERENCE = 4
DEFAULT_MAX_GROUPS = GROUPS_PER_REFERENCE


class HVMDataset(Dataset):
    """
    HVM-SVG 训练/验证数据集。

    同时加载:
    1. 原始 parquet 数据 (SVG 字符串, 描述, 图像)
    2. 预计算的 HVM 数据 (vision features, RAG results, path groups)

    eval 模式 (is_eval=True):
      val/test 数据的 features 路径格式不同于训练集：
      - 自身: features/val_{idx//1000:03d}/val_{idx:06d}.pt
      - ref:  features/ref_{ridx//1000:03d}/ref_{ridx:06d}.pt
      - group_features: group_features/{ridx//1000:03d}/{ridx:06d}.pt (与训练集相同)
    """

    def __init__(
        self,
        data_dir: str,
        hvm_dir: str,
        token_config: TokenizationConfig,
        train_config: Optional[TrainConfig] = None,
        max_len: int = 2048,
        shuffle_rag: bool = False,
        shuffle_gme: bool = False,
        shuffle_cdm: bool = False,
        is_eval: bool = False,
        split: str = "train",  # 新增: train, val, test, test_holdout
        part_num_refs: int = 1,
    ):
        """
        Args:
            data_dir: 原始 parquet 文件目录
            hvm_dir: HVM 预计算数据目录
            token_config: SVG tokenization 配置
            train_config: 训练配置
            max_len: 最大 SVG token 序列长度
            shuffle_rag: 是否全局打乱 ref 对应关系 (ablation)
            is_eval: 是否为 eval 模式 (val/test)，影响 features 路径格式
            split: 数据集划分 (train / val / test / test_holdout)
            part_num_refs: part-grounded 分支使用前多少个 refs（默认 Top-1）
        """
        self.data_dir = data_dir
        self.hvm_dir = hvm_dir
        self.max_len = max_len
        self.token_config = token_config
        self.train_config = train_config or TrainConfig()
        self.shuffle_rag = shuffle_rag
        self.shuffle_gme = shuffle_gme
        self.shuffle_cdm = shuffle_cdm
        self.is_eval = is_eval
        self.split = split
        self.part_num_refs = max(1, int(part_num_refs))
        self.features_dir = os.path.join(hvm_dir, "features")
        self.group_features_dir = os.path.join(hvm_dir, "group_features")

        # SVG tokenizer
        self.svg_tokenizer = SVGTokenizer(token_config)

        # 加载预计算数据
        print(f"[HVM Dataset] Loading precomputed data ({split})...")
        self.metadata = self._load_jsonl(os.path.join(hvm_dir, "metadata.jsonl"))
        
        # 根据 split 自动选择对应的 jsonl 文件
        if split in ("train", "val", "test_holdout"):
            rag_file = "rag_results_train.jsonl"
            groups_file = "groups_train_ref.jsonl"
        else:
            rag_file = "rag_results.jsonl"
            groups_file = "groups.jsonl"
            
        self.rag_results = self._load_jsonl(os.path.join(hvm_dir, rag_file))
        self.groups_data = self._load_jsonl(os.path.join(hvm_dir, groups_file))

        # 建立 idx → metadata/rag/groups 的快速查找
        self.idx_to_meta = {r["idx"]: r for r in self.metadata}
        self.idx_to_rag = {r["idx"]: r for r in self.rag_results}
        self.idx_to_groups = {r["idx"]: r for r in self.groups_data}

        # eval 模式下，ref 的 metadata 来自全库（用于获取 ref 描述文本）
        # 在 __init__（主进程）中预加载，fork 后 worker 自动共享，不再重复加载
        self._ref_meta_cache: Dict[int, Dict] = {}
        if is_eval:
            ref_meta_path = os.path.join(
                os.path.dirname(hvm_dir), "hvm_precomputed_1w", "metadata.jsonl"
            )
            if os.path.exists(ref_meta_path):
                print(f"[HVM Dataset] Pre-loading full ref metadata...")
                with open(ref_meta_path) as f:
                    for line in f:
                        r = json.loads(line)
                        self._ref_meta_cache[r["idx"]] = r
                print(f"[HVM Dataset] Loaded {len(self._ref_meta_cache)} ref metadata entries")

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

    def _self_feat_path(self, idx: int) -> str:
        """val/test 自身 features 路径"""
        if self.is_eval:
            return os.path.join(
                self.features_dir, f"val_{idx // 1000:03d}", f"val_{idx:06d}.pt"
            )
        return os.path.join(
            self.features_dir, f"{idx // 1000:03d}", f"{idx:06d}.pt"
        )

    def _ref_feat_path(self, ref_idx: int) -> str:
        """ref features 路径"""
        if self.is_eval:
            return os.path.join(
                self.features_dir, f"ref_{ref_idx // 1000:03d}", f"ref_{ref_idx:06d}.pt"
            )
        return os.path.join(
            self.features_dir, f"{ref_idx // 1000:03d}", f"{ref_idx:06d}.pt"
        )

    def _build_valid_indices(self) -> List[int]:
        """构建有效样本索引：必须同时有所有预计算数据"""
        valid = []
        for meta in self.metadata:
            idx = meta["idx"]
            if idx not in self.idx_to_rag:
                continue

            # 检查自身 feature 文件是否存在
            if not os.path.exists(self._self_feat_path(idx)):
                continue

            # 检查参考样本的 features 和 group_features 是否也存在
            rag = self.idx_to_rag[idx]
            refs_ok = True
            for ref_idx in rag["ref_indices"]:
                if not os.path.exists(self._ref_feat_path(ref_idx)):
                    refs_ok = False
                    break

            # 检查 part 分支所需 refs 的 group_features 是否存在
            if refs_ok:
                for part_ref_idx in rag["ref_indices"][:self.part_num_refs]:
                    gf_path = os.path.join(
                        self.group_features_dir,
                        f"{part_ref_idx // 1000:03d}",
                        f"{part_ref_idx:06d}.pt",
                    )
                    if not os.path.exists(gf_path):
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

    def _get_ref_meta(self, ref_idx: int) -> Dict:
        """获取 ref 的 metadata。训练模式查 idx_to_meta，eval 模式查预加载的 _ref_meta_cache。"""
        if not self.is_eval:
            return self.idx_to_meta.get(ref_idx, {})
        return self._ref_meta_cache.get(ref_idx, {})

    def _load_self_feature(self, idx: int) -> torch.Tensor:
        """加载样本自身的整图 post-merge feature"""
        return torch.load(self._self_feat_path(idx), map_location="cpu", weights_only=True)

    def _load_ref_feature(self, ref_idx: int) -> torch.Tensor:
        """加载 ref 的整图 post-merge feature (for GME)"""
        return torch.load(self._ref_feat_path(ref_idx), map_location="cpu", weights_only=True)

    def _build_group_tag_meta(
        self,
        ref_idx: int,
        num_groups: Optional[int] = None,
    ) -> torch.Tensor:
        """根据 groups_train_ref.jsonl 在线构造 [cx, cy, w, h, z_start, z_end]。"""
        group_record = self.idx_to_groups.get(ref_idx, {})
        groups = list(group_record.get("groups", []))
        if num_groups is not None:
            groups = groups[:num_groups]

        denom = max(int(group_record.get("num_paths", 0)) - 1, 1)
        tag_meta = []
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

    def _load_group_features(self, idx: int) -> Dict[str, Any]:
        """加载单个样本的逐 group 渲染特征，并兼容新旧存储格式。"""
        gf_path = os.path.join(
            self.group_features_dir, f"{idx // 1000:03d}", f"{idx:06d}.pt"
        )
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

        derived_tag_meta = self._build_group_tag_meta(idx, num_groups=num_groups)
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
        group_features = group_features[:aligned_len]
        tag_meta = tag_meta[:aligned_len]
        group_ids = group_ids[:aligned_len]

        return {
            "group_features": group_features,
            "tag_meta": tag_meta,
            "group_ids": group_ids,
        }

    def _load_part_group_bundle(self, ref_indices: List[int]) -> Dict[str, Any]:
        """
        将前 K 个 refs 的 group features 串联成一个统一的 part bank。

        group_id 采用最小改动方案:
          global_group_id = ref_rank * 4 + local_group_id
        例如 Top-3 时范围为 0..11。
        """
        merged_group_features: List[torch.Tensor] = []
        merged_tag_meta: List[torch.Tensor] = []
        merged_group_ids: List[torch.Tensor] = []

        for ref_rank, ref_idx in enumerate(ref_indices[:self.part_num_refs]):
            group_data = self._load_group_features(ref_idx)
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
                ref_features: List[torch.Tensor]       3 × [256, 3584] (for GME, post-merge)
                ref_best_group_features: List[Tensor]  part refs 的逐 group 渲染特征, list of [256, 3584]
                ref_best_group_tag_meta: Tensor        part refs 各 group 的 tag [G, 6]
                ref_best_group_ids: Tensor             part refs 各 group 的全局 group_id [G]
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

        # 描述文本（训练时随机选 detail 或 description，eval 时固定用 description）
        description = table.column("description")[row].as_py()
        detail = table.column("detail")[row].as_py() or ""
        if not self.is_eval and detail and random.random() < self.train_config.detail_prob:
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

        # Shuffle RAG ablation: 随机采样另一个样本的 ref，保证 100% 错配
        if self.shuffle_rag:
            donor_idx = idx
            while donor_idx == idx:
                donor_idx = random.choice(self.valid_indices)
            donor_rag = self.idx_to_rag[donor_idx]
            ref_indices = donor_rag["ref_indices"]

        # GME 用的 ref indices（可能被单独 shuffle）
        gme_ref_indices = ref_indices
        if self.shuffle_gme and not self.shuffle_rag:
            gme_donor_idx = idx
            while gme_donor_idx == idx:
                gme_donor_idx = random.choice(self.valid_indices)
            gme_ref_indices = self.idx_to_rag[gme_donor_idx]["ref_indices"]

        # CDM 用的 ref indices（可能被单独 shuffle）
        cdm_ref_indices = ref_indices
        if self.shuffle_cdm and not self.shuffle_rag:
            cdm_donor_idx = idx
            while cdm_donor_idx == idx:
                cdm_donor_idx = random.choice(self.valid_indices)
            cdm_ref_indices = self.idx_to_rag[cdm_donor_idx]["ref_indices"]

        # 3 张参考图的整图 features (for GME)
        ref_features = [self._load_ref_feature(ri) for ri in gme_ref_indices]

        # part refs 的逐 group 渲染特征 (for PME / part-grounded CDM)
        ref_best_group_data = self._load_part_group_bundle(cdm_ref_indices)

        # 参考文本 (拼接 3 个参考的描述)
        ref_texts = []
        for ri in gme_ref_indices:
            ref_meta = self._get_ref_meta(ri)
            ref_desc = ref_meta.get("description", "")
            if ref_desc:
                ref_texts.append(ref_desc)
        ref_text = " ".join(ref_texts)

        return {
            "text": text,
            "pix_seq": pix_seq,
            "ref_features": ref_features,                    # List of 3 × [256, 3584]
            "ref_best_group_features": ref_best_group_data["group_features"],  # List of [256, 3584]
            "ref_best_group_tag_meta": ref_best_group_data["tag_meta"],        # [G, 6]
            "ref_best_group_ids": ref_best_group_data["group_ids"],            # [G]
            "ref_text": ref_text,
            "part_ref_indices": ref_indices[:self.part_num_refs],
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
    ref_text_max_length: int = 128,#1111111111具体参数待定
):
    """
    创建 HVM 训练的 collate function。

    将 Dataset 返回的 raw samples 组装成 batch tensor:
    - input_ids, attention_mask, labels: 文本 + SVG 序列
    - ref_features: [B, 3, 256, 3584]  (for GME, post-merge)
    - group_features_list: List[List[Tensor]]  (for PME, 每个样本 1~K*4 个 [256, 3584])
    - part_features: [B, G, 256, 3584]         (for part-grounded CDM, G 可为 4/12/...)
    - part_tag_meta: [B, G, 6]                 (for layout tags)
    - part_group_ids: [B, G]                   (for group-id embedding)
    - part_mask: [B, G]                        (有效 group)
    - ref_text_ids, ref_text_mask: [B, N_t]
    """
    system_prompt = "You are an expert SVG code generator."
    pad_token_id = token_config.pad_token_id
    max_len = max_seq_length + text_len

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        texts = [s["text"] for s in batch]
        pix_seqs = [s["pix_seq"] for s in batch]
        ref_features_list = [s["ref_features"] for s in batch]
        group_features_list = [s["ref_best_group_features"] for s in batch]
        group_tag_meta_list = [s["ref_best_group_tag_meta"] for s in batch]
        group_ids_list = [s["ref_best_group_ids"] for s in batch]
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
            pad_len = max_len - len(current_input_ids) # 这部分参考omnisvg的train.py的_process_sample函数
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
        # 2. 组装参考图 features (for GME)
        # ================================================================
        # ref_features: [B, 3, 256, 3584]  (post-merge, LLM-aligned)
        ref_features = torch.stack([
            torch.stack(rf) for rf in ref_features_list
        ])  # [B, 3, 256, 3584]

        # ================================================================
        # 3. Group features (for PME) — 保持 List 格式，PME 内部逐样本处理
        # ================================================================
        # group_features_list: List[List[Tensor]]
        # group_features_list[b] = [tensor_g0, tensor_g1, ...], 每个 [256, 3584]
        max_groups = max(
            DEFAULT_MAX_GROUPS,
            max((len(gfs) for gfs in group_features_list), default=0),
        )
        token_dim = ref_features.shape[-1]
        num_tokens = ref_features.shape[2]
        group_dtype = ref_features.dtype

        part_features = torch.zeros(
            len(batch), max_groups, num_tokens, token_dim, dtype=group_dtype
        )
        part_tag_meta = torch.zeros(len(batch), max_groups, 6, dtype=torch.float32)
        part_group_ids = torch.zeros(len(batch), max_groups, dtype=torch.long)
        part_mask = torch.zeros(len(batch), max_groups, dtype=torch.bool)

        for b_idx, (sample_gfs, sample_meta, sample_group_ids) in enumerate(
            zip(group_features_list, group_tag_meta_list, group_ids_list)
        ):
            num_valid = min(len(sample_gfs), max_groups)
            if num_valid == 0:
                continue
            part_features[b_idx, :num_valid] = torch.stack(sample_gfs[:num_valid], dim=0)
            part_tag_meta[b_idx, :num_valid] = sample_meta[:num_valid]
            part_group_ids[b_idx, :num_valid] = sample_group_ids[:num_valid]
            part_mask[b_idx, :num_valid] = True

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
            "group_features_list": group_features_list,
            "part_features": part_features,
            "part_tag_meta": part_tag_meta,
            "part_group_ids": part_group_ids,
            "part_mask": part_mask,
            "ref_text_ids": ref_text_ids,
            "ref_text_mask": ref_text_mask,
        }

    return collate_fn
