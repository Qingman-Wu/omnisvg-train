"""
HVM-SVG: Hierarchical Visual Memory Modules
=============================================
所有 HVM-SVG 的核心神经网络模块：
- MultiHeadAttention: 高效的多头注意力（支持 bottleneck）
- QFormerLayer / QFormer: 用于 GME 和 PME 的 Query Transformer
- GistMemoryEncoder (GME): 全局场景记忆编码器，32 queries
- PartMemoryEncoder (PME): 局部零件记忆编码器，每组 4 queries
- AdaptiveGate: 双层门控（层级 + token 级）
- SimpleGMEInjectionModule: Stage1 简化注入模块（GME-only + fixed scale）
- PrefrontalInjectionModule (PIM): 前额叶注入模块，4 步融合 + gate
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HVMConfig:
    """HVM-SVG 超参数配置"""

    # === 维度 ===
    d_model: int = 3584          # LLM hidden dim (Qwen2.5-7B)
    d_vision: int = 3584         # Vision encoder post-merge dim (GME & PME 统一使用)
    d_qformer: int = 1024        # QFormer 内部维度
    d_pim_inner: int = 512       # PIM attention bottleneck 维度

    # === GME ===
    gme_num_queries: int = 32
    gme_num_layers: int = 6
    gme_num_heads: int = 8
    gme_ff_mult: int = 4         # FFN 扩展倍数

    # === PME ===
    pme_queries_per_group: int = 4
    pme_num_layers: int = 4
    pme_num_heads: int = 8
    pme_ff_mult: int = 4
    pme_max_groups: int = 4
    pme_max_tokens: int = 16     # 4 groups × 4 queries

    # === CDM (Complementary Detail Memory) ===
    cdm_num_queries: int = 16
    cdm_num_layers: int = 6
    cdm_num_heads: int = 8
    cdm_ff_mult: int = 4
    cdm_layout: str = "global"      # global: 所有 group tokens 共用一个 CDM, groupwise: 每个 group 单独过共享 CDM
    cdm_group_queries_per_group: int = 4
    cdm_detail_source: str = "ref"   # ref: 3x256 raw ref tokens, part: 4x256 tagged part tokens
    cdm_tag_meta_dim: int = 6        # [cx, cy, w, h, z_start, z_end]
    cdm_use_tag_meta: bool = True
    cdm_use_group_id: bool = True
    cdm_disable_gist: bool = False   # 关闭 CDM 内部 gist cross-attn（EDR gist 路仍可保留）

    # === EDR (Execution-aware Detail Routing) ===
    edr_d_router: int = 256
    edr_top_k: int = 2
    edr_disable_conf: bool = False
    edr_disable_gist: bool = False
    edr_detail_layer_indices_override: Optional[List[int]] = None

    # === DRA (Direct Reference Attention) ===
    dra_d_inner: int = 128           # DRA ref path bottleneck 维度
    dra_n_heads: int = 4             # DRA ref path attention heads

    # === PIM ===
    pim_num_heads: int = 8
    pim_layer_interval: int = 4  # 每隔 N 层插入一个 PIM
    num_decoder_layers: int = 28
    gate_alpha_init: float = 0.05  # 冷启动更平滑: tanh(0.05)≈0.05，避免 PIM 主干梯度过弱
    memory_mode: str = "full"      # full: GME+PME+full PIM, gme: Stage1 简化路径
    inject_mode: str = "adaptive"  # adaptive: gate 注入, fixed: 固定缩放注入
    inject_scale: float = 0.1      # inject_mode=fixed 时生效
    pim_layer_indices_override: Optional[List[int]] = None
    delta_ln: bool = False           # delta 上加 LayerNorm 稳定 scale

    # === RAG ===
    num_references: int = 3
    ref_text_max_length: int = 128

    @property
    def pim_layer_indices(self) -> List[int]:
        """PIM 插入位置（0-indexed，在该层之后插入）"""
        if self.pim_layer_indices_override is not None:
            return list(self.pim_layer_indices_override)
        return list(range(
            self.pim_layer_interval - 1,
            self.num_decoder_layers,
            self.pim_layer_interval
        ))

    @property
    def num_pims(self) -> int:
        return len(self.pim_layer_indices)

    @property
    def edr_detail_layer_indices(self) -> List[int]:
        """EDR detail 路实际生效的 decoder 层。默认与 PIM 层一致。"""
        if self.edr_detail_layer_indices_override is None:
            return list(self.pim_layer_indices)
        return list(self.edr_detail_layer_indices_override)


def build_cdm_tag_embedding(
    config: HVMConfig,
    tag_meta_mlp: nn.Module,
    group_id_embedding: nn.Embedding,
    part_tag_meta: Optional[torch.Tensor],
    part_group_ids: Optional[torch.Tensor],
    part_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
    num_groups: int,
    out_dim: int,
) -> torch.Tensor:
    if config.cdm_use_tag_meta and part_tag_meta is not None:
        tag_emb = tag_meta_mlp(part_tag_meta.to(device=device, dtype=dtype))
    else:
        tag_emb = torch.zeros(batch_size, num_groups, out_dim, device=device, dtype=dtype)

    if config.cdm_use_group_id and part_group_ids is not None:
        safe_group_ids = part_group_ids.clamp(min=0, max=config.pme_max_groups - 1)
        tag_emb = tag_emb + group_id_embedding(safe_group_ids.to(device=device))

    if part_mask is not None:
        tag_emb = tag_emb * part_mask.to(device=device, dtype=dtype).unsqueeze(-1)

    return tag_emb


def build_detail_slot_mask(
    part_mask: Optional[torch.Tensor],
    slots_per_group: int,
    max_slots: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if part_mask is None:
        return torch.ones(batch_size, max_slots, device=device, dtype=torch.bool)

    slot_mask = part_mask.to(device=device, dtype=torch.bool).unsqueeze(-1)
    slot_mask = slot_mask.expand(-1, -1, slots_per_group).reshape(part_mask.shape[0], -1)

    if slot_mask.shape[1] < max_slots:
        pad = torch.zeros(slot_mask.shape[0], max_slots - slot_mask.shape[1], device=device, dtype=torch.bool)
        slot_mask = torch.cat([slot_mask, pad], dim=1)
    elif slot_mask.shape[1] > max_slots:
        slot_mask = slot_mask[:, :max_slots]

    empty_rows = ~slot_mask.any(dim=-1)
    if empty_rows.any():
        slot_mask = slot_mask.clone()
        slot_mask[empty_rows, 0] = True
    return slot_mask


# ============================================================================
# Attention Primitives
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    通用多头注意力模块，支持 bottleneck（Q/K/V 投影到 d_inner < d_model）。

    参数:
        d_model: 输入/输出维度
        n_heads: 注意力头数
        d_inner: 内部维度（None 则等于 d_model）
    """

    def __init__(self, d_model: int, n_heads: int, d_inner: Optional[int] = None):
        super().__init__()
        self.d_inner = d_inner or d_model
        self.n_heads = n_heads
        self.d_head = self.d_inner // n_heads
        assert self.d_inner % n_heads == 0, f"d_inner({self.d_inner}) must be divisible by n_heads({n_heads})"

        self.to_q = nn.Linear(d_model, self.d_inner, bias=False)
        self.to_k = nn.Linear(d_model, self.d_inner, bias=False)
        self.to_v = nn.Linear(d_model, self.d_inner, bias=False)
        self.to_out = nn.Linear(self.d_inner, d_model, bias=False)
        self.capture_attn = False
        self.last_attn = None

    def _capture_attention_probs(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        kv_mask: Optional[torch.Tensor],
    ) -> None:
        if not self.capture_attn:
            self.last_attn = None
            return

        with torch.no_grad():
            score = torch.matmul(Q.float(), K.float().transpose(-1, -2)) * (self.d_head ** -0.5)
            if kv_mask is not None:
                valid_mask = kv_mask.to(device=score.device, dtype=torch.bool)
                if valid_mask.ndim != 2:
                    raise ValueError(f"kv_mask must be [B, Nk], got shape={tuple(valid_mask.shape)}")
                if valid_mask.shape[0] != score.shape[0] or valid_mask.shape[1] != score.shape[-1]:
                    raise ValueError(
                        "kv_mask shape mismatch: "
                        f"mask={tuple(valid_mask.shape)} vs attn={tuple(score.shape)}"
                    )
                valid_mask = valid_mask.clone()
                empty_rows = ~valid_mask.any(dim=-1)
                if empty_rows.any():
                    valid_mask[empty_rows, 0] = True
                score = score.masked_fill(~valid_mask[:, None, None, :], float("-inf"))

            self.last_attn = torch.softmax(score, dim=-1).detach().float()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: [B, Nq, d_model]
            k: [B, Nk, d_model]
            v: [B, Nk, d_model]
            kv_mask: [B, Nk] bool, True=有效, False=padding

        Returns:
            [B, Nq, d_model]
        """
        B, Nq, _ = q.shape
        Nk = k.shape[1]

        Q = self.to_q(q).reshape(B, Nq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.to_k(k).reshape(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        V = self.to_v(v).reshape(B, Nk, self.n_heads, self.d_head).transpose(1, 2)
        # Q: [B, H, Nq, d_head], K/V: [B, H, Nk, d_head]

        # 构造 attention mask（SDPA 格式：0=有效, -inf=mask）
        attn_mask = None
        if kv_mask is not None:
            # kv_mask: [B, Nk] True=valid → [B, 1, 1, Nk]
            attn_mask = kv_mask[:, None, None, :].to(dtype=Q.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        self._capture_attention_probs(Q, K, kv_mask)
        # out: [B, H, Nq, d_head]
        out = out.transpose(1, 2).reshape(B, Nq, self.d_inner)
        return self.to_out(out)


# ============================================================================
# QFormer
# ============================================================================

class QFormerLayer(nn.Module):
    """
    QFormer 单层: Self-Attention → Cross-Attention → FFN
    全部使用 Pre-Norm（更稳定的训练）。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        # Self-attention among queries
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm_sa = nn.LayerNorm(d_model)

        # Cross-attention: queries attend to KV (image features)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.norm_ca = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, Nq, D]
            kv: [B, Nk, D]  (image features, already projected)
            kv_mask: [B, Nk] bool, True=valid

        Returns:
            [B, Nq, D]
        """
        # Self-attention
        q_norm = self.norm_sa(queries)
        queries = queries + self.self_attn(q_norm, q_norm, q_norm)

        # Cross-attention
        q_norm = self.norm_ca(queries)
        queries = queries + self.cross_attn(q_norm, kv, kv, kv_mask=kv_mask)

        # FFN
        queries = queries + self.ffn(self.norm_ff(queries))

        return queries


class QFormer(nn.Module):
    """
    Query Transformer: 可学习 queries 从 image features 中提取固定数量的表示。

    输入: image features [B, N_img, d_vision]
    输出: query embeddings [B, N_query, d_model]
    """

    def __init__(
        self,
        num_queries: int,
        num_layers: int,
        d_qformer: int,
        d_vision: int,
        d_out: int,
        n_heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.d_qformer = d_qformer #1024

        # 可学习 queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_qformer) * 0.02)

        # 输入投影: d_vision → d_qformer
        self.input_proj = nn.Linear(d_vision, d_qformer, bias=False)
        self.input_norm = nn.LayerNorm(d_qformer)

        # QFormer layers
        d_ff = d_qformer * ff_mult
        self.layers = nn.ModuleList([
            QFormerLayer(d_qformer, n_heads, d_ff)
            for _ in range(num_layers)
        ])

        # 输出投影: d_qformer → d_out (d_model of LLM)
        self.output_proj = nn.Linear(d_qformer, d_out, bias=False)
        self.output_norm = nn.LayerNorm(d_out)

    def forward(
        self,
        image_feats: torch.Tensor,
        feat_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            image_feats: [B, N_img, d_vision]
            feat_mask: [B, N_img] bool, True=valid

        Returns:
            [B, num_queries, d_out]
        """
        B = image_feats.shape[0]

        # 统一 dtype (features 可能是 float16, 权重可能是 float32/bfloat16)
        image_feats = image_feats.to(dtype=self.input_proj.weight.dtype)

        # 投影 image features
        kv = self.input_norm(self.input_proj(image_feats))  # [B, N_img, d_qformer]

        # 扩展 queries 到 batch
        queries = self.queries.expand(B, -1, -1)  # [B, N_query, d_qformer] b,32,1024

        # 过 QFormer layers
        for layer in self.layers:
            queries = layer(queries, kv, kv_mask=feat_mask)

        # 输出投影
        return self.output_norm(self.output_proj(queries))  # [B, N_query, d_out] b,32,3584


# ============================================================================
# Gist Memory Encoder (GME)
# ============================================================================

class GistMemoryEncoder(nn.Module):
    """
    全局场景记忆编码器。
    认知对应: Scene Gist — 快速、压缩、持久的整体结构记忆。

    将 3 张参考图的 post-merge features 压缩成 32 个 gist tokens。
    输入: 3 张参考图的 post-merge features, 各 [B, 256, 3584]
    输出: gist_feats [B, 32, d_model]

    """

    def __init__(self, config: HVMConfig):
        super().__init__()
        self.config = config
        self.qformer = QFormer(
            num_queries=config.gme_num_queries,      # 32
            num_layers=config.gme_num_layers,         # 6
            d_qformer=config.d_qformer,               # 1024
            d_vision=config.d_vision,                  # 3584
            d_out=config.d_model,                      # 3584
            n_heads=config.gme_num_heads,              # 8
            ff_mult=config.gme_ff_mult,                # 4
        )

    def forward(self, ref_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_features: [B, num_refs, 256, d_vision]
                         3 张参考图的 post-merge features (已对齐 LLM 空间)

        Returns:
            gist_feats: [B, 32, d_model]
        """
        B, N_ref, T, D = ref_features.shape
        # 展平所有参考图: [B, N_ref * T, D] = [B, 3*256, 3584] = [B, 768, 3584]
        flat_feats = ref_features.reshape(B, N_ref * T, D)
        # 过 QFormer: 768 tokens → 32 queries
        return self.qformer(flat_feats)  # [B, 32, d_model]


# ============================================================================
# CDM (Complementary Detail Memory) Encoder
# ============================================================================

class CDMQFormerLayer(nn.Module):
    """
    CDM QFormer 单层: SelfAttn → CrossAttn(gist) → CrossAttn(ref) → FFN
    比标准 QFormerLayer 多一路 gist cross-attention，实现"残差提取"。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm_sa = nn.LayerNorm(d_model)

        self.gist_cross_attn = MultiHeadAttention(d_model, n_heads)
        self.norm_gist_ca = nn.LayerNorm(d_model)

        self.ref_cross_attn = MultiHeadAttention(d_model, n_heads)
        self.norm_ref_ca = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.capture_vis = False
        self.last_ref_attn = None

    def forward(
        self,
        queries: torch.Tensor,
        gist_kv: Optional[torch.Tensor],
        ref_kv: torch.Tensor,
        ref_kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, Nq, D]
            gist_kv: [B, 32, D] or None  gist features (detached)
            ref_kv:  [B, 768, D]  raw ref features (projected)
            ref_kv_mask: [B, N_ref] bool, True=有效
        """
        self.last_ref_attn = None
        q_norm = self.norm_sa(queries)
        queries = queries + self.self_attn(q_norm, q_norm, q_norm)

        if gist_kv is not None:
            q_norm = self.norm_gist_ca(queries)
            queries = queries + self.gist_cross_attn(q_norm, gist_kv, gist_kv)

        q_norm = self.norm_ref_ca(queries)
        self.ref_cross_attn.capture_attn = bool(self.capture_vis)
        queries = queries + self.ref_cross_attn(q_norm, ref_kv, ref_kv, kv_mask=ref_kv_mask)
        if self.capture_vis and self.ref_cross_attn.last_attn is not None:
            self.last_ref_attn = self.ref_cross_attn.last_attn.detach()

        queries = queries + self.ffn(self.norm_ff(queries))
        return queries


class CDMEncoder(nn.Module):
    """
    互补细节记忆编码器 (Complementary Detail Memory)。

    16 个 learnable queries 先了解 gist 覆盖了什么，
    再从 detail source 中提取 gist 遗漏的互补细节。

    输入:
      - ref_features: [B, 768, 3584]      (旧版 CDM: 展平后的 3 张参考图 vision features)
      - part_features: [B, G, 256, 3584]  (新版 CDM: Top-1 ref 的 group 特征)
      - part_tag_meta: [B, G, 6]          (layout/order tag)
      - part_group_ids: [B, G]            (group id: 0,1,2,3)
      - part_mask: [B, G]                 (有效 group)
      - gist_feats: [B, 32, 3584]         (GME 输出, detached)
    输出:
      - detail_feats: [B, 16, 3584]
    """

    def __init__(self, config: HVMConfig):
        super().__init__()
        self.config = config
        d_q = config.d_qformer  # 1024

        self.queries = nn.Parameter(torch.randn(1, config.cdm_num_queries, d_q) * 0.02)

        self.ref_proj = nn.Linear(config.d_vision, d_q, bias=False)
        self.ref_norm = nn.LayerNorm(d_q)

        self.gist_proj = nn.Linear(config.d_model, d_q, bias=False)
        self.gist_norm = nn.LayerNorm(d_q)

        self.group_id_embedding = nn.Embedding(config.pme_max_groups, config.d_vision)
        self.tag_meta_mlp = nn.Sequential(
            nn.Linear(config.cdm_tag_meta_dim, d_q),
            nn.GELU(),
            nn.Linear(d_q, config.d_vision),
        )

        d_ff = d_q * config.cdm_ff_mult
        self.layers = nn.ModuleList([
            CDMQFormerLayer(d_q, config.cdm_num_heads, d_ff)
            for _ in range(config.cdm_num_layers)
        ])

        self.output_proj = nn.Linear(d_q, config.d_model, bias=False)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.capture_vis = False
        self.last_vis = {}
        self.last_detail_slot_mask = None

    def forward(
        self,
        ref_features: torch.Tensor,
        gist_feats: torch.Tensor,
        part_tag_meta: Optional[torch.Tensor] = None,
        part_group_ids: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            ref_features: [B, 768, d_vision] or [B, G, 256, d_vision]
            gist_feats:   [B, 32, d_model]  (should be detached by caller)
            part_tag_meta: [B, G, 6]，仅 part-grounded CDM 使用
            part_group_ids: [B, G]，仅 part-grounded CDM 使用
            part_mask: [B, G]，仅 part-grounded CDM 使用
        Returns:
            detail_feats: [B, cdm_num_queries, d_model]
        """
        self.last_vis = {}
        B = ref_features.shape[0]
        dtype = self.ref_proj.weight.dtype
        gist_kv = None
        if not self.config.cdm_disable_gist:
            gist_kv = self.gist_norm(self.gist_proj(gist_feats.to(dtype=dtype)))
        ref_kv_mask = None
        group_token_count = None
        self.last_detail_slot_mask = torch.ones(
            B,
            self.config.cdm_num_queries,
            device=ref_features.device,
            dtype=torch.bool,
        )

        if ref_features.ndim == 4:
            # part-grounded CDM: [B, G, 256, 3584] → 加 tag 后展平为 [B, G*256, 3584]
            group_tokens = ref_features.to(dtype=dtype)
            group_token_count = int(group_tokens.shape[2])
            tag_emb = build_cdm_tag_embedding(
                config=self.config,
                tag_meta_mlp=self.tag_meta_mlp,
                group_id_embedding=self.group_id_embedding,
                part_tag_meta=part_tag_meta,
                part_group_ids=part_group_ids,
                part_mask=part_mask,
                dtype=dtype,
                device=group_tokens.device,
                batch_size=group_tokens.shape[0],
                num_groups=group_tokens.shape[1],
                out_dim=group_tokens.shape[-1],
            )
            tagged_tokens = group_tokens + tag_emb.unsqueeze(2)
            flat_tokens = tagged_tokens.view(B, -1, tagged_tokens.shape[-1])

            if part_mask is not None:
                ref_kv_mask = part_mask.to(device=flat_tokens.device, dtype=torch.bool)
                ref_kv_mask = ref_kv_mask.unsqueeze(-1).expand(-1, -1, group_tokens.shape[2]).reshape(B, -1)
                empty_rows = ~ref_kv_mask.any(dim=1)
                if empty_rows.any():
                    ref_kv_mask = ref_kv_mask.clone()
                    ref_kv_mask[empty_rows, 0] = True
            ref_kv = self.ref_norm(self.ref_proj(flat_tokens))
        else:
            ref_kv = self.ref_norm(self.ref_proj(ref_features.to(dtype=dtype)))

        queries = self.queries.expand(B, -1, -1)
        ref_attn_per_layer = []

        for layer in self.layers:
            layer.capture_vis = bool(self.capture_vis)
            queries = layer(queries, gist_kv, ref_kv, ref_kv_mask=ref_kv_mask)
            if self.capture_vis and layer.last_ref_attn is not None:
                ref_attn_per_layer.append(layer.last_ref_attn.detach())

        detail_feats = self.output_norm(self.output_proj(queries))
        if self.capture_vis:
            self.last_vis = {
                "detail_source": "part" if ref_features.ndim == 4 else "ref",
                "detail_layout": "global",
                "ref_attn_per_layer": ref_attn_per_layer,
                "group_token_count": group_token_count,
                "ref_kv_mask": ref_kv_mask.detach().clone() if ref_kv_mask is not None else None,
                "part_mask": part_mask.detach().clone() if part_mask is not None else None,
                "part_group_ids": part_group_ids.detach().clone() if part_group_ids is not None else None,
                "part_tag_meta": part_tag_meta.detach().clone() if part_tag_meta is not None else None,
                "detail_slot_mask": self.last_detail_slot_mask.detach().clone(),
                "detail_feats": detail_feats.detach(),
            }
        return detail_feats


class GroupwiseCDMEncoder(nn.Module):
    """
    Group-wise CDM:
      - 每个 group 的 256 个视觉 tokens 单独经过一套共享权重的 CDM
      - 每组输出固定 4 个 slots
      - 4 组 concat 成最终 16 个 detail slots
      - group_id / tag_meta 在每组 slots 编码完成后再注入（Version B）
    """

    def __init__(self, config: HVMConfig):
        super().__init__()
        self.config = config
        d_q = config.d_qformer

        if config.cdm_group_queries_per_group * config.pme_max_groups != config.cdm_num_queries:
            raise ValueError(
                "groupwise CDM requires "
                f"cdm_group_queries_per_group({config.cdm_group_queries_per_group}) * "
                f"pme_max_groups({config.pme_max_groups}) == cdm_num_queries({config.cdm_num_queries})"
            )

        self.queries = nn.Parameter(
            torch.randn(1, config.cdm_group_queries_per_group, d_q) * 0.02
        )

        self.ref_proj = nn.Linear(config.d_vision, d_q, bias=False)
        self.ref_norm = nn.LayerNorm(d_q)

        self.gist_proj = nn.Linear(config.d_model, d_q, bias=False)
        self.gist_norm = nn.LayerNorm(d_q)

        self.group_id_embedding = nn.Embedding(config.pme_max_groups, config.d_model)
        self.tag_meta_mlp = nn.Sequential(
            nn.Linear(config.cdm_tag_meta_dim, d_q),
            nn.GELU(),
            nn.Linear(d_q, config.d_model),
        )

        d_ff = d_q * config.cdm_ff_mult
        self.layers = nn.ModuleList([
            CDMQFormerLayer(d_q, config.cdm_num_heads, d_ff)
            for _ in range(config.cdm_num_layers)
        ])

        self.output_proj = nn.Linear(d_q, config.d_model, bias=False)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.slot_tag_norm = nn.LayerNorm(config.d_model)
        self.capture_vis = False
        self.last_vis = {}
        self.last_detail_slot_mask = None

    def _stitch_group_attn(
        self,
        local_attn: torch.Tensor,
        batch_size: int,
        num_groups: int,
        group_token_count: int,
    ) -> torch.Tensor:
        _, num_heads, _, _ = local_attn.shape
        slots_per_group = self.config.cdm_group_queries_per_group
        local_attn = local_attn.view(batch_size, num_groups, num_heads, slots_per_group, group_token_count)
        global_attn = torch.zeros(
            batch_size,
            num_heads,
            num_groups * slots_per_group,
            num_groups * group_token_count,
            device=local_attn.device,
            dtype=local_attn.dtype,
        )
        for group_idx in range(num_groups):
            q_start = group_idx * slots_per_group
            q_end = q_start + slots_per_group
            k_start = group_idx * group_token_count
            k_end = k_start + group_token_count
            global_attn[:, :, q_start:q_end, k_start:k_end] = local_attn[:, group_idx]
        return global_attn

    def forward(
        self,
        ref_features: torch.Tensor,
        gist_feats: torch.Tensor,
        part_tag_meta: Optional[torch.Tensor] = None,
        part_group_ids: Optional[torch.Tensor] = None,
        part_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ref_features.ndim != 4:
            raise ValueError(
                f"GroupwiseCDMEncoder expects part features [B, G, T, D], got {tuple(ref_features.shape)}"
            )

        self.last_vis = {}
        B, G, T, _ = ref_features.shape
        dtype = self.ref_proj.weight.dtype
        device = ref_features.device
        group_tokens = ref_features.to(dtype=dtype)

        if part_mask is None:
            part_mask = torch.ones(B, G, device=device, dtype=torch.bool)
        else:
            part_mask = part_mask.to(device=device, dtype=torch.bool)

        ref_kv = self.ref_norm(self.ref_proj(group_tokens.reshape(B * G, T, -1)))
        ref_kv_mask = part_mask.reshape(B * G, 1).expand(-1, T)
        empty_rows = ~ref_kv_mask.any(dim=1)
        if empty_rows.any():
            ref_kv_mask = ref_kv_mask.clone()
            ref_kv_mask[empty_rows, 0] = True

        gist_kv = None
        if not self.config.cdm_disable_gist:
            gist_base = self.gist_norm(self.gist_proj(gist_feats.to(dtype=dtype)))
            gist_kv = gist_base.unsqueeze(1).expand(-1, G, -1, -1).reshape(B * G, gist_base.shape[1], -1)

        queries = self.queries.expand(B * G, -1, -1)
        ref_attn_per_layer = []
        for layer in self.layers:
            layer.capture_vis = bool(self.capture_vis)
            queries = layer(queries, gist_kv, ref_kv, ref_kv_mask=ref_kv_mask)
            if self.capture_vis and layer.last_ref_attn is not None:
                ref_attn_per_layer.append(
                    self._stitch_group_attn(
                        layer.last_ref_attn.detach(),
                        batch_size=B,
                        num_groups=G,
                        group_token_count=T,
                    )
                )

        detail_slots = self.output_norm(self.output_proj(queries))
        detail_slots = detail_slots.view(B, G, self.config.cdm_group_queries_per_group, -1)

        tag_emb = build_cdm_tag_embedding(
            config=self.config,
            tag_meta_mlp=self.tag_meta_mlp,
            group_id_embedding=self.group_id_embedding,
            part_tag_meta=part_tag_meta,
            part_group_ids=part_group_ids,
            part_mask=part_mask,
            dtype=detail_slots.dtype,
            device=detail_slots.device,
            batch_size=B,
            num_groups=G,
            out_dim=detail_slots.shape[-1],
        )
        detail_slots = self.slot_tag_norm(detail_slots + tag_emb.unsqueeze(2))
        detail_slots = detail_slots * part_mask.to(dtype=detail_slots.dtype).unsqueeze(-1).unsqueeze(-1)

        detail_feats = detail_slots.reshape(B, -1, detail_slots.shape[-1])
        if detail_feats.shape[1] < self.config.cdm_num_queries:
            pad_slots = self.config.cdm_num_queries - detail_feats.shape[1]
            pad = torch.zeros(B, pad_slots, detail_feats.shape[-1], device=device, dtype=detail_feats.dtype)
            detail_feats = torch.cat([detail_feats, pad], dim=1)
        elif detail_feats.shape[1] > self.config.cdm_num_queries:
            detail_feats = detail_feats[:, :self.config.cdm_num_queries]

        self.last_detail_slot_mask = build_detail_slot_mask(
            part_mask=part_mask,
            slots_per_group=self.config.cdm_group_queries_per_group,
            max_slots=self.config.cdm_num_queries,
            device=device,
            batch_size=B,
        )
        detail_feats = detail_feats * self.last_detail_slot_mask.to(dtype=detail_feats.dtype).unsqueeze(-1)

        if self.capture_vis:
            flat_ref_mask = part_mask.unsqueeze(-1).expand(-1, -1, T).reshape(B, -1)
            self.last_vis = {
                "detail_source": "part",
                "detail_layout": "groupwise",
                "ref_attn_per_layer": ref_attn_per_layer,
                "group_token_count": T,
                "ref_kv_mask": flat_ref_mask.detach().clone(),
                "part_mask": part_mask.detach().clone(),
                "part_group_ids": part_group_ids.detach().clone() if part_group_ids is not None else None,
                "part_tag_meta": part_tag_meta.detach().clone() if part_tag_meta is not None else None,
                "detail_slot_mask": self.last_detail_slot_mask.detach().clone(),
                "detail_feats": detail_feats.detach(),
            }

        return detail_feats


# ============================================================================
# Part Memory Encoder (PME)
# ============================================================================

class PartMemoryEncoder(nn.Module):
    """
    局部零件记忆编码器。
    认知对应: Object/Part Memory — 容量有限(4±1)、高精度的零件级表征。

    接收 Top-1 参考图的逐 group 独立渲染后提取的 vision features，
    每组通过 QFormer 压缩为 4 个 tokens。

    输入: group_features_list — batch of [List of [256, 3584] tensors]
          每个 tensor 是一个 group 独立渲染后通过 ViT+Merger 提取的 post-merge 特征。
          group 图片是：只包含该 group 的 paths，viewbox 设为 group 的 bbox，
          渲染成 448x448 后过完整 vision pipeline (encoder + merger) 得到的。
          特征已对齐 LLM 空间 (3584 维)。
    输出: part_feats [B, max_tokens, d_model] + part_mask [B, max_tokens]
    """

    def __init__(self, config: HVMConfig):
        super().__init__()
        self.config = config
        self.max_tokens = config.pme_max_tokens  # 16

        self.qformer = QFormer(
            num_queries=config.pme_queries_per_group,  # 4
            num_layers=config.pme_num_layers,           # 4
            d_qformer=config.d_qformer,                 # 1024
            d_vision=config.d_vision,                    # 3584 (post-merge, LLM-aligned, 与 GME 统一)
            d_out=config.d_model,                        # 3584
            n_heads=config.pme_num_heads,                # 8
            ff_mult=config.pme_ff_mult,                  # 4
        )

    def forward(
        self,
        group_features_list: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            group_features_list: batch of group feature lists.
                group_features_list[b] = [feat_g0, feat_g1, ...]
                每个 feat_gX: [256, 3584] — 该 group 独立渲染后的 post-merge 特征
                长度 1~4，对应 1~4 个 group。

        Returns:
            part_feats: [B, max_tokens(16), d_model]  padded
            part_mask:  [B, max_tokens(16)]  bool, True=有效
        """
        B = len(group_features_list)
        # 从第一个非空样本获取 device 和 dtype
        device = None
        dtype = None
        for gfl in group_features_list:
            if gfl:
                device = gfl[0].device
                dtype = gfl[0].dtype
                break
        if device is None:
            # 全空，用模块自身参数的 device/dtype
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype

        all_part_feats = []
        all_masks = []

        for b in range(B):
            group_feats_list = group_features_list[b]  # List of [256, D] 256，3584
            group_outs = []

            for group_feat in group_feats_list:
                # group_feat: [256, 3584] (post-merge, LLM-aligned)
                # 过 QFormer: [1, 256, 3584] → [1, 4, 3584]
                group_input = group_feat.unsqueeze(0)  # [1, 256, 3584]
                with torch.set_grad_enabled(self.training):
                    group_out = self.qformer(group_input)
                group_outs.append(group_out.squeeze(0))  # [4, d_model]

            if group_outs:
                # Concat 所有组: [N_groups * 4, d_model]
                sample_feats = torch.cat(group_outs, dim=0)
                actual_len = sample_feats.shape[0]
            else:
                sample_feats = torch.zeros(0, self.config.d_model, device=device, dtype=dtype)
                actual_len = 0

            # Pad 到 max_tokens
            #1111111111这部分设计的是q=16，一个svg最多分为4group，每个group占4q，所以需要pad，但是后续更想改为引入top3的svg的局部信息，q超出16，再proj回来
            pad_len = self.max_tokens - actual_len #max16
            if pad_len > 0: 
                padding = torch.zeros(pad_len, self.config.d_model, device=device, dtype=dtype)
                sample_feats = torch.cat([sample_feats, padding], dim=0)
            else:
                sample_feats = sample_feats[:self.max_tokens]
                actual_len = self.max_tokens

            all_part_feats.append(sample_feats)

            # 构建 mask
            mask = torch.zeros(self.max_tokens, dtype=torch.bool, device=device)
            mask[:actual_len] = True
            all_masks.append(mask)

        part_feats = torch.stack(all_part_feats, dim=0)  # [B, 16, d_model]
        part_mask = torch.stack(all_masks, dim=0)          # [B, 16]

        return part_feats, part_mask


# ============================================================================
# Adaptive Gate
# ============================================================================

class AdaptiveGate(nn.Module):
    """
    双层自适应门控。
    认知对应: 前额叶注意力门控 — 选择性维持相关信息。

    Layer gate: Flamingo 式 tanh(alpha)，控制该 PIM 整体注入强度
    Token gate: GateRA 式 input-dependent sigmoid，每个 token 位置独立决定

    关键: 初始化确保训练初期 PIM 几乎不影响 decoder。
    """

    def __init__(self, d_model: int, base_alpha_init: float = 0.0):
        super().__init__()

        # 层级 gate: tanh(alpha), alpha 初始化为 0 → tanh(0) = 0
        self.base_alpha = nn.Parameter(torch.tensor(float(base_alpha_init)))

        # Token 级 gate: input-dependent
        d_gate = d_model // 4
        self.token_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_gate),
            nn.GELU(),
            nn.Linear(d_gate, 1),
        )

        # 最后一层 Linear 初始化为 0，确保 sigmoid(0)=0.5
        nn.init.zeros_(self.token_gate[-1].weight)
        nn.init.zeros_(self.token_gate[-1].bias)

        # 最近一次 forward 的诊断统计（用于训练日志）
        self.last_stats = {}

    def forward(self, hidden_state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, L, d_model]  decoder block 输出
            delta: [B, L, d_model]  PIM 计算的修正量

        Returns:
            [B, L, d_model]  修正后的 hidden_state
        """
        # 层级 gate
        layer_gate = torch.tanh(self.base_alpha)  # 标量, 初始 ≈ 0

        # Token 级 gate
        gate_input = torch.cat([hidden_state, delta], dim=-1)  # [B, L, 2*d_model]
        token_gate = torch.sigmoid(self.token_gate(gate_input))  # [B, L, 1]

        # 组合注入
        injection = layer_gate * token_gate * delta

        # 记录诊断统计（detach，避免额外反向图）
        with torch.no_grad():
            hidden_rms = hidden_state.detach().float().pow(2).mean().sqrt()
            delta_rms = delta.detach().float().pow(2).mean().sqrt()
            inject_rms = injection.detach().float().pow(2).mean().sqrt()
            self.last_stats = {
                "layer_gate": layer_gate.detach().float(),
                "token_gate_mean": token_gate.detach().float().mean(),
                "token_gate_std": token_gate.detach().float().std(unbiased=False),
                "delta_rms": delta_rms,
                "inject_rms": inject_rms,
                "inject_hidden_ratio": inject_rms / (hidden_rms + 1e-6),
            }

        return hidden_state + injection


# ============================================================================
# Layer-Gated GME Injection Module (Stage1.5)
# ============================================================================

class LayerGatedGMEInjectionModule(nn.Module):
    """
    GME-only + layer gate 注入模块：
      - 仅使用 gist memory（GME 输出）
      - 单次 hidden × gist cross-attention
      - Layer gate: 可学习标量 tanh(alpha) 控制注入强度
      - 可选 delta LayerNorm 稳定 scale（config.delta_ln=True）

    相比 SimpleGMEInjectionModule: fixed scale → learnable layer gate
    相比 AdaptiveGate: 去掉 token-level gate，只保留 layer-level gate
    """

    def __init__(self, config: HVMConfig):
        super().__init__()
        d = config.d_model
        self.hidden_norm = nn.LayerNorm(d)
        self.hidden_gist_cross_attn = MultiHeadAttention(
            d,
            config.pim_num_heads,
            d_inner=config.d_pim_inner,
        )
        self.delta_norm = nn.LayerNorm(d) if config.delta_ln else None
        self.base_alpha = nn.Parameter(torch.tensor(float(config.gate_alpha_init)))
        self.last_stats = {}

    def forward(
        self,
        hidden_state: torch.Tensor,
        gist_feats: torch.Tensor,
    ) -> torch.Tensor:
        query = self.hidden_norm(hidden_state)
        delta = self.hidden_gist_cross_attn(
            q=query,
            k=gist_feats,
            v=gist_feats,
        )
        if self.delta_norm is not None:
            delta = self.delta_norm(delta)

        layer_gate = torch.tanh(self.base_alpha)
        injection = layer_gate * delta

        with torch.no_grad():
            hidden_rms = hidden_state.detach().float().pow(2).mean().sqrt()
            delta_rms = delta.detach().float().pow(2).mean().sqrt()
            inject_rms = injection.detach().float().pow(2).mean().sqrt()
            self.last_stats = {
                "layer_gate": layer_gate.detach().float(),
                "delta_rms": delta_rms,
                "inject_rms": inject_rms,
                "inject_hidden_ratio": inject_rms / (hidden_rms + 1e-6),
            }

        return hidden_state + injection


# ============================================================================
# Layer-Gated GME+PME Injection Module (Stage2a)
# ============================================================================

class LayerGatedGMEPMEInjectionModule(nn.Module):
    """
    GME+PME 双路独立 cross-attention + layer gate 注入模块：
      - Gist 路: hidden × gist cross-attention（全局风格信息）
      - Part 路: hidden × part cross-attention（局部零件信息，带 kv_mask）
      - 两路 delta 加法融合
      - Layer gate: tanh(alpha) 控制总注入强度

    相比 LayerGatedGMEInjectionModule: 新增 Part 路
    消融对照: 后续 Stage2c 的 PrefrontalInjectionModule（层次化 4 步融合）
    """

    def __init__(self, config: HVMConfig):
        super().__init__()
        d = config.d_model
        self.hidden_norm = nn.LayerNorm(d)
        self.hidden_gist_cross_attn = MultiHeadAttention(
            d,
            config.pim_num_heads,
            d_inner=config.d_pim_inner,
        )
        self.hidden_part_cross_attn = MultiHeadAttention(
            d,
            config.pim_num_heads,
            d_inner=config.d_pim_inner,
        )
        self.base_alpha = nn.Parameter(torch.tensor(float(config.gate_alpha_init)))
        self.last_stats = {}

    def forward(
        self,
        hidden_state: torch.Tensor,
        gist_feats: torch.Tensor,
        part_feats: torch.Tensor,
        part_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.hidden_norm(hidden_state)

        delta_gist = self.hidden_gist_cross_attn(
            q=query, k=gist_feats, v=gist_feats,
        )
        delta_part = self.hidden_part_cross_attn(
            q=query, k=part_feats, v=part_feats,
            kv_mask=part_mask,
        )

        delta = delta_gist + delta_part
        layer_gate = torch.tanh(self.base_alpha)
        injection = layer_gate * delta

        with torch.no_grad():
            hidden_rms = hidden_state.detach().float().pow(2).mean().sqrt()
            delta_gist_rms = delta_gist.detach().float().pow(2).mean().sqrt()
            delta_part_rms = delta_part.detach().float().pow(2).mean().sqrt()
            delta_rms = delta.detach().float().pow(2).mean().sqrt()
            inject_rms = injection.detach().float().pow(2).mean().sqrt()
            self.last_stats = {
                "layer_gate": layer_gate.detach().float(),
                "delta_gist_rms": delta_gist_rms,
                "delta_part_rms": delta_part_rms,
                "delta_rms": delta_rms,
                "inject_rms": inject_rms,
                "inject_hidden_ratio": inject_rms / (hidden_rms + 1e-6),
            }

        return hidden_state + injection


# ============================================================================
# Dual-Gated GME+PME Injection Module (Stage2b)
# ============================================================================

class DualGatedGMEPMEInjectionModule(nn.Module):
    """
    GME+PME 双路独立 cross-attention + 独立 gate 注入模块：
      - Gist 路: hidden × gist cross-attention + tanh(alpha_gist)
      - Part 路: hidden × part cross-attention + tanh(alpha_part)
      - 两路分别 gate 后再相加注入

    相比 LayerGatedGMEPMEInjectionModule:
      - 共享 gate → 独立 gate，解耦 Gist/Part 的 scale 差异
      - 可通过 alpha_gist / alpha_part 的值诊断两路各自的贡献
    """

    def __init__(self, config: HVMConfig):
        super().__init__()
        d = config.d_model
        self.hidden_norm = nn.LayerNorm(d)
        self.hidden_gist_cross_attn = MultiHeadAttention(
            d,
            config.pim_num_heads,
            d_inner=config.d_pim_inner,
        )
        self.hidden_part_cross_attn = MultiHeadAttention(
            d,
            config.pim_num_heads,
            d_inner=config.d_pim_inner,
        )
        self.alpha_gist = nn.Parameter(torch.tensor(float(config.gate_alpha_init)))
        self.alpha_part = nn.Parameter(torch.tensor(float(config.gate_alpha_init)))
        self.last_stats = {}

    def forward(
        self,
        hidden_state: torch.Tensor,
        gist_feats: torch.Tensor,
        part_feats: torch.Tensor,
        part_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.hidden_norm(hidden_state)

        delta_gist = self.hidden_gist_cross_attn(
            q=query, k=gist_feats, v=gist_feats,
        )
        delta_part = self.hidden_part_cross_attn(
            q=query, k=part_feats, v=part_feats,
            kv_mask=part_mask,
        )

        gate_gist = torch.tanh(self.alpha_gist)
        gate_part = torch.tanh(self.alpha_part)

        inject_gist = gate_gist * delta_gist
        inject_part = gate_part * delta_part
        injection = inject_gist + inject_part

        with torch.no_grad():
            hidden_rms = hidden_state.detach().float().pow(2).mean().sqrt()
            delta_gist_rms = delta_gist.detach().float().pow(2).mean().sqrt()
            delta_part_rms = delta_part.detach().float().pow(2).mean().sqrt()
            inject_gist_rms = inject_gist.detach().float().pow(2).mean().sqrt()
            inject_part_rms = inject_part.detach().float().pow(2).mean().sqrt()
            delta_rms = (delta_gist + delta_part).detach().float().pow(2).mean().sqrt()
            inject_rms = injection.detach().float().pow(2).mean().sqrt()
            self.last_stats = {
                "gate_gist": gate_gist.detach().float(),
                "gate_part": gate_part.detach().float(),
                "delta_gist_rms": delta_gist_rms,
                "delta_part_rms": delta_part_rms,
                "inject_gist_rms": inject_gist_rms,
                "inject_part_rms": inject_part_rms,
                "delta_rms": delta_rms,
                "inject_rms": inject_rms,
                "inject_hidden_ratio": inject_rms / (hidden_rms + 1e-6),
            }

        return hidden_state + injection


# ============================================================================
# Hierarchical GME+PME Injection Module (Stage2c)
# ============================================================================

class HierarchicalGMEPMEInjectionModule(nn.Module):
    """
    GME+PME 层次化融合 + 双独立 gate 注入模块：
      - Part 预处理:
        Step 1: Part × Gist cross-attention → Part 获得全局上下文
        Step 2: Part self-attention → 各组带着全局上下文互相整合
      - 注入:
        Gist 路: hidden × gist cross-attention + tanh(alpha_gist)  (与 DualGated 完全一致)
        Part 路: hidden × refined_part cross-attention + tanh(alpha_part)

    相比 DualGatedGMEPMEInjectionModule:
      - 唯一区别: Part features 先经过 Part×Gist + Part self-attn 预处理
      - Gist 路完全不变, gate 机制完全不变
      - 消融目标: 验证层次化 Part 融合是否让 Part 变得有用
    """

    def __init__(self, config: HVMConfig):
        super().__init__()
        d = config.d_model

        # Part 预处理 (新增)
        self.part_gist_cross_attn = MultiHeadAttention(
            d, config.pim_num_heads, d_inner=config.d_pim_inner,
        )
        self.part_gist_norm = nn.LayerNorm(d)
        self.part_self_attn = MultiHeadAttention(
            d, config.pim_num_heads, d_inner=config.d_pim_inner,
        )
        self.part_self_norm = nn.LayerNorm(d)

        # 注入 (与 DualGated 完全一致)
        self.hidden_norm = nn.LayerNorm(d)
        self.hidden_gist_cross_attn = MultiHeadAttention(
            d, config.pim_num_heads, d_inner=config.d_pim_inner,
        )
        self.hidden_part_cross_attn = MultiHeadAttention(
            d, config.pim_num_heads, d_inner=config.d_pim_inner,
        )
        self.alpha_gist = nn.Parameter(torch.tensor(float(config.gate_alpha_init)))
        self.alpha_part = nn.Parameter(torch.tensor(float(config.gate_alpha_init)))
        self.last_stats = {}

    def forward(
        self,
        hidden_state: torch.Tensor,
        gist_feats: torch.Tensor,
        part_feats: torch.Tensor,
        part_mask: torch.Tensor,
    ) -> torch.Tensor:
        # === Part 预处理: 层次化融合 ===
        # Step 1: Part × Gist cross-attention — Part 在全局中定位自己
        step1_out = self.part_gist_cross_attn(
            q=part_feats, k=gist_feats, v=gist_feats,
        )
        context_part = self.part_gist_norm(step1_out + part_feats)

        # Step 2: Part self-attention — 各组带着全局上下文互相整合
        step2_out = self.part_self_attn(
            q=context_part, k=context_part, v=context_part,
            kv_mask=part_mask,
        )
        refined_part = self.part_self_norm(step2_out + context_part)

        # === 双 Gate 注入 (与 DualGated 完全一致) ===
        query = self.hidden_norm(hidden_state)

        delta_gist = self.hidden_gist_cross_attn(
            q=query, k=gist_feats, v=gist_feats,
        )
        delta_part = self.hidden_part_cross_attn(
            q=query, k=refined_part, v=refined_part,
            kv_mask=part_mask,
        )

        gate_gist = torch.tanh(self.alpha_gist)
        gate_part = torch.tanh(self.alpha_part)

        inject_gist = gate_gist * delta_gist
        inject_part = gate_part * delta_part
        injection = inject_gist + inject_part

        with torch.no_grad():
            hidden_rms = hidden_state.detach().float().pow(2).mean().sqrt()
            delta_gist_rms = delta_gist.detach().float().pow(2).mean().sqrt()
            delta_part_rms = delta_part.detach().float().pow(2).mean().sqrt()
            inject_gist_rms = inject_gist.detach().float().pow(2).mean().sqrt()
            inject_part_rms = inject_part.detach().float().pow(2).mean().sqrt()
            delta_rms = (delta_gist + delta_part).detach().float().pow(2).mean().sqrt()
            inject_rms = injection.detach().float().pow(2).mean().sqrt()
            self.last_stats = {
                "gate_gist": gate_gist.detach().float(),
                "gate_part": gate_part.detach().float(),
                "delta_gist_rms": delta_gist_rms,
                "delta_part_rms": delta_part_rms,
                "inject_gist_rms": inject_gist_rms,
                "inject_part_rms": inject_part_rms,
                "delta_rms": delta_rms,
                "inject_rms": inject_rms,
                "inject_hidden_ratio": inject_rms / (hidden_rms + 1e-6),
            }

        return hidden_state + injection


# ============================================================================
# Single-Path Hierarchical GME+PME Injection Module (Stage2d)
# ============================================================================

class SinglePathHierarchicalInjectionModule(nn.Module):
    """
    GME+PME 层次化融合 + 单路注入模块：
      - Part 预处理 (与 Hierarchical 完全一致):
        Step 1: Part × Gist cross-attention → Part 获得全局上下文
        Step 2: Part self-attention → 各组带着全局上下文互相整合
      - 单路注入:
        Hidden × refined_part cross-attention → delta
        tanh(alpha) 层级 gate

    相比 HierarchicalGMEPMEInjectionModule:
      - 去掉独立的 Gist 注入路径 (hidden × gist)
      - Gist 信息只通过 Step 1 融入 Part 后间接注入
      - 单 gate (无 alpha_gist / alpha_part 分离)
      - 消融目标: 验证 gist+part 融合后单路注入 vs 双路分别注入
    """

    def __init__(self, config: HVMConfig):
        super().__init__()
        d = config.d_model

        # Part 预处理 (与 Hierarchical 完全一致)
        self.part_gist_cross_attn = MultiHeadAttention(
            d, config.pim_num_heads, d_inner=config.d_pim_inner,
        )
        self.part_gist_norm = nn.LayerNorm(d)
        self.part_self_attn = MultiHeadAttention(
            d, config.pim_num_heads, d_inner=config.d_pim_inner,
        )
        self.part_self_norm = nn.LayerNorm(d)

        # 单路注入
        self.hidden_norm = nn.LayerNorm(d)
        self.hidden_aligned_cross_attn = MultiHeadAttention(
            d, config.pim_num_heads, d_inner=config.d_pim_inner,
        )
        self.base_alpha = nn.Parameter(torch.tensor(float(config.gate_alpha_init)))
        self.last_stats = {}

    def forward(
        self,
        hidden_state: torch.Tensor,
        gist_feats: torch.Tensor,
        part_feats: torch.Tensor,
        part_mask: torch.Tensor,
    ) -> torch.Tensor:
        # === Part 预处理: 层次化融合 ===
        step1_out = self.part_gist_cross_attn(
            q=part_feats, k=gist_feats, v=gist_feats,
        )
        context_part = self.part_gist_norm(step1_out + part_feats)

        step2_out = self.part_self_attn(
            q=context_part, k=context_part, v=context_part,
            kv_mask=part_mask,
        )
        refined_part = self.part_self_norm(step2_out + context_part)

        # === 单路注入: hidden × refined_part ===
        query = self.hidden_norm(hidden_state)
        delta = self.hidden_aligned_cross_attn(
            q=query, k=refined_part, v=refined_part,
            kv_mask=part_mask,
        )

        layer_gate = torch.tanh(self.base_alpha)
        injection = layer_gate * delta

        with torch.no_grad():
            hidden_rms = hidden_state.detach().float().pow(2).mean().sqrt()
            delta_rms = delta.detach().float().pow(2).mean().sqrt()
            inject_rms = injection.detach().float().pow(2).mean().sqrt()
            self.last_stats = {
                "layer_gate": layer_gate.detach().float(),
                "delta_rms": delta_rms,
                "inject_rms": inject_rms,
                "inject_hidden_ratio": inject_rms / (hidden_rms + 1e-6),
            }

        return hidden_state + injection


# ============================================================================
# DRA (Direct Reference Attention) Injection Module (Stage3)
# ============================================================================

class DRAInjectionModule(nn.Module):
    """
    GME + Direct Reference Attention 注入模块：
      - Gist 路: hidden × gist cross-attn (d_inner=512, 8 heads) + tanh(α_gist)
      - Ref 路:  hidden × ref_features cross-attn (d_inner=128, 4 heads) + tanh(α_ref)
      - 两路分别 gate 后相加注入

    核心假设检验: 模型能否利用未经 QFormer 压缩的原始 ref_features？
      - α_ref > 0 → 模型能用细粒度信息，QFormer 压缩有信息损失
      - α_ref → 0 → 32 gist tokens 已充分，不需要更细粒度信息
    """

    def __init__(self, config: HVMConfig):
        super().__init__()
        d = config.d_model

        self.hidden_norm = nn.LayerNorm(d)

        # Gist 路 (与 LayerGatedGMEInjectionModule 一致)
        self.gist_cross_attn = MultiHeadAttention(
            d, config.pim_num_heads, d_inner=config.d_pim_inner,
        )
        self.alpha_gist = nn.Parameter(torch.tensor(float(config.gate_alpha_init)))

        # Ref 路 (极窄 bottleneck)
        self.ref_cross_attn = MultiHeadAttention(
            d, config.dra_n_heads, d_inner=config.dra_d_inner,
        )
        self.alpha_ref = nn.Parameter(torch.tensor(float(config.gate_alpha_init)))

        self.last_stats = {}

    def forward(
        self,
        hidden_state: torch.Tensor,
        gist_feats: torch.Tensor,
        ref_feats: torch.Tensor,
    ) -> torch.Tensor:
        query = self.hidden_norm(hidden_state)

        delta_gist = self.gist_cross_attn(q=query, k=gist_feats, v=gist_feats)
        delta_ref = self.ref_cross_attn(q=query, k=ref_feats, v=ref_feats)

        gate_gist = torch.tanh(self.alpha_gist)
        gate_ref = torch.tanh(self.alpha_ref)

        inject_gist = gate_gist * delta_gist
        inject_ref = gate_ref * delta_ref
        injection = inject_gist + inject_ref

        with torch.no_grad():
            hidden_rms = hidden_state.detach().float().pow(2).mean().sqrt()
            delta_gist_rms = delta_gist.detach().float().pow(2).mean().sqrt()
            delta_ref_rms = delta_ref.detach().float().pow(2).mean().sqrt()
            inject_gist_rms = inject_gist.detach().float().pow(2).mean().sqrt()
            inject_ref_rms = inject_ref.detach().float().pow(2).mean().sqrt()
            delta_rms = (delta_gist + delta_ref).detach().float().pow(2).mean().sqrt()
            inject_rms = injection.detach().float().pow(2).mean().sqrt()
            self.last_stats = {
                "gate_gist": gate_gist.detach().float(),
                "gate_ref": gate_ref.detach().float(),
                "delta_gist_rms": delta_gist_rms,
                "delta_ref_rms": delta_ref_rms,
                "inject_gist_rms": inject_gist_rms,
                "inject_ref_rms": inject_ref_rms,
                "delta_rms": delta_rms,
                "inject_rms": inject_rms,
                "inject_hidden_ratio": inject_rms / (hidden_rms + 1e-6),
            }

        return hidden_state + injection


# ============================================================================
# EDR (Execution-aware Detail Routing) Injection Module
# ============================================================================

class DetailRouter(nn.Module):
    """
    最小版 detail router：
      - 用当前 hidden_state 对 detail_bank 中的每个 slot 打分
      - 通过 softmax 熵得到 router 置信度 conf
      - 只保留 top-k slots 并加权聚合成 routed_detail

    输入:
      - hidden_state: [B, L, d_model]
      - detail_bank:  [B, N_detail, d_model]
    输出:
      - routed_detail: [B, L, d_model]
      - conf:          [B, L, 1]
      - router_stats:  标量诊断信息
    """

    def __init__(self, d_model: int, d_router: int, top_k: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_router, bias=False)
        self.k_proj = nn.Linear(d_model, d_router, bias=False)
        self.scale = d_router ** -0.5
        self.top_k = top_k
        self.capture_trace = False
        self.last_trace = {}

    def forward(
        self,
        hidden_state: torch.Tensor,
        detail_bank: torch.Tensor,
        detail_slot_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        self.last_trace = {}
        score = torch.matmul(
            self.q_proj(hidden_state),
            self.k_proj(detail_bank).transpose(-1, -2),
        ) * self.scale  # [B, L, N_detail]

        score_fp32 = score.float()
        valid_slot_mask = None
        num_slots = detail_bank.shape[1]
        if detail_slot_mask is not None:
            valid_slot_mask = detail_slot_mask.to(device=score_fp32.device, dtype=torch.bool)
            if valid_slot_mask.ndim != 2:
                raise ValueError(
                    f"detail_slot_mask must be [B, N_detail], got shape={tuple(valid_slot_mask.shape)}"
                )
            if valid_slot_mask.shape[0] != score_fp32.shape[0] or valid_slot_mask.shape[1] != num_slots:
                raise ValueError(
                    "detail_slot_mask shape mismatch: "
                    f"mask={tuple(valid_slot_mask.shape)} vs detail_bank={tuple(detail_bank.shape)}"
                )
            valid_slot_mask = valid_slot_mask.clone()
            empty_rows = ~valid_slot_mask.any(dim=-1)
            if empty_rows.any():
                valid_slot_mask[empty_rows, 0] = True
            score_fp32 = score_fp32.masked_fill(~valid_slot_mask[:, None, :], float("-inf"))

        full_prob = torch.softmax(score_fp32, dim=-1)
        entropy = -(full_prob * torch.log(full_prob.clamp_min(1e-8))).sum(dim=-1, keepdim=True)

        if valid_slot_mask is not None:
            valid_slot_count = valid_slot_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            valid_slot_count = valid_slot_count.to(device=entropy.device, dtype=entropy.dtype).view(-1, 1, 1)
            valid_slot_log = valid_slot_count.clamp_min(2).log()
            conf = torch.ones_like(entropy)
            multi_slot_rows = valid_slot_count > 1
            conf = torch.where(
                multi_slot_rows,
                1.0 - entropy / valid_slot_log,
                conf,
            )
        elif num_slots > 1:
            conf = 1.0 - entropy / math.log(num_slots)
        else:
            conf = torch.ones_like(entropy)
        conf = conf.clamp_(0.0, 1.0)

        if valid_slot_mask is not None:
            min_valid_slots = int(valid_slot_mask.sum(dim=-1).min().item())
            k = min(self.top_k, max(min_valid_slots, 1))
        else:
            k = min(self.top_k, num_slots)
        topk_vals, topk_idx = torch.topk(score_fp32, k=k, dim=-1)
        topk_prob = torch.softmax(topk_vals, dim=-1)

        sparse_prob = torch.zeros_like(score_fp32)
        sparse_prob.scatter_(-1, topk_idx, topk_prob)
        routed_detail = torch.matmul(sparse_prob.to(dtype=detail_bank.dtype), detail_bank)
        if self.capture_trace:
            with torch.no_grad():
                self.last_trace = {
                    "score": score_fp32.detach(),
                    "full_prob": full_prob.detach(),
                    "topk_idx": topk_idx.detach(),
                    "topk_prob": topk_prob.detach(),
                    "selected_slot": topk_idx[..., 0].detach() if k > 0 else None,
                    "selected_prob": topk_prob[..., 0].detach() if k > 0 else None,
                    "sparse_prob": sparse_prob.detach(),
                    "conf": conf.detach().float(),
                    "detail_slot_mask": valid_slot_mask.detach().clone() if valid_slot_mask is not None else None,
                    "num_slots": num_slots,
                    "top_k": k,
                }

        if num_slots > 1:
            slot_usage = sparse_prob.detach().mean(dim=(0, 1))
            slot_usage_entropy = -(
                slot_usage * torch.log(slot_usage.clamp_min(1e-8))
            ).sum() / math.log(num_slots)
        else:
            slot_usage_entropy = torch.tensor(0.0, device=detail_bank.device)

        router_stats = {
            "router_conf_mean": conf.detach().float().mean(),
            "router_entropy_mean": entropy.detach().float().mean(),
            "router_top1_prob_mean": full_prob.max(dim=-1).values.detach().float().mean(),
            "router_active_ratio": (conf.detach().float() > 0.5).float().mean(),
            "router_slot_usage_entropy": slot_usage_entropy.detach().float(),
        }
        return routed_detail, conf.to(dtype=detail_bank.dtype), router_stats


class EDRInjectionModule(nn.Module):
    """
    GME + CDM + EDR 注入模块：
      - Gist 路保持原始 cross-attn 注入
      - Detail 路先对 CDM 的 16 个 detail slots 做 router
      - 再用 routed_detail 经 conf 调制后注入 hidden_state
    """

    def __init__(self, config: HVMConfig, enable_detail: bool = True):
        super().__init__()
        d = config.d_model

        self.hidden_norm = nn.LayerNorm(d)
        self.gist_cross_attn = MultiHeadAttention(
            d, config.pim_num_heads, d_inner=config.d_pim_inner,
        )
        self.detail_router = DetailRouter(
            d_model=d,
            d_router=config.edr_d_router,
            top_k=config.edr_top_k,
        )
        self.disable_conf = bool(config.edr_disable_conf)
        self.enable_gist = not bool(config.edr_disable_gist)
        self.enable_detail = bool(enable_detail)

        self.alpha_gist = nn.Parameter(torch.tensor(float(config.gate_alpha_init)))
        self.alpha_detail = nn.Parameter(torch.tensor(float(config.gate_alpha_init)))
        self.last_stats = {}
        self.capture_vis = False
        self.last_trace = {}

    def forward(
        self,
        hidden_state: torch.Tensor,
        gist_feats: torch.Tensor,
        detail_feats: torch.Tensor,
        detail_slot_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.last_trace = {}
        self.detail_router.capture_trace = bool(self.capture_vis)
        self.detail_router.last_trace = {}
        query = self.hidden_norm(hidden_state)

        if self.enable_gist:
            delta_gist = self.gist_cross_attn(q=query, k=gist_feats, v=gist_feats)
            gate_gist = torch.tanh(self.alpha_gist)
            inject_gist = gate_gist * delta_gist
        else:
            delta_gist = torch.zeros_like(hidden_state)
            gate_gist = torch.zeros(
                (),
                device=hidden_state.device,
                dtype=hidden_state.dtype,
            )
            inject_gist = torch.zeros_like(hidden_state)

        if self.enable_detail:
            routed_detail, conf, router_stats = self.detail_router(
                query,
                detail_feats,
                detail_slot_mask=detail_slot_mask,
            )
            effective_conf = torch.ones_like(conf) if self.disable_conf else conf
            delta_detail = routed_detail
            gate_detail = torch.tanh(self.alpha_detail)
            inject_detail = gate_detail * effective_conf * delta_detail
        else:
            gate_detail = torch.zeros_like(gate_gist)
            delta_detail = torch.zeros_like(hidden_state)
            inject_detail = torch.zeros_like(hidden_state)
            effective_conf = torch.zeros(
                hidden_state.shape[0],
                hidden_state.shape[1],
                1,
                device=hidden_state.device,
                dtype=hidden_state.dtype,
            )
            router_stats = {
                "router_conf_mean": torch.tensor(0.0, device=hidden_state.device),
                "router_entropy_mean": torch.tensor(0.0, device=hidden_state.device),
                "router_top1_prob_mean": torch.tensor(0.0, device=hidden_state.device),
                "router_active_ratio": torch.tensor(0.0, device=hidden_state.device),
                "router_slot_usage_entropy": torch.tensor(0.0, device=hidden_state.device),
            }

        injection = inject_gist + inject_detail

        with torch.no_grad():
            hidden_rms = hidden_state.detach().float().pow(2).mean().sqrt()
            delta_gist_rms = delta_gist.detach().float().pow(2).mean().sqrt()
            delta_detail_rms = delta_detail.detach().float().pow(2).mean().sqrt()
            inject_gist_rms = inject_gist.detach().float().pow(2).mean().sqrt()
            inject_detail_rms = inject_detail.detach().float().pow(2).mean().sqrt()
            delta_rms = (delta_gist + delta_detail).detach().float().pow(2).mean().sqrt()
            inject_rms = injection.detach().float().pow(2).mean().sqrt()
            self.last_stats = {
                "gate_gist": gate_gist.detach().float(),
                "gate_detail": gate_detail.detach().float(),
                "gist_enabled": float(self.enable_gist),
                "detail_enabled": float(self.enable_detail),
                "delta_gist_rms": delta_gist_rms,
                "delta_detail_rms": delta_detail_rms,
                "inject_gist_rms": inject_gist_rms,
                "inject_detail_rms": inject_detail_rms,
                "delta_rms": delta_rms,
                "inject_rms": inject_rms,
                "inject_hidden_ratio": inject_rms / (hidden_rms + 1e-6),
                "router_effective_conf_mean": effective_conf.detach().float().mean(),
                **router_stats,
            }
            if self.capture_vis:
                router_trace = dict(self.detail_router.last_trace)
                self.last_trace = {
                    "enable_gist": self.enable_gist,
                    "enable_detail": self.enable_detail,
                    "effective_conf": effective_conf.detach().float(),
                    "router_stats": {
                        key: value.detach().float() if torch.is_tensor(value) else value
                        for key, value in router_stats.items()
                    },
                    **router_trace,
                }

        return hidden_state + injection


# ============================================================================
# CDM Injection Module (Stage3)
# ============================================================================

class CDMInjectionModule(nn.Module):
    """
    GME + CDM 注入模块：
      - Gist 路:   hidden × gist cross-attn (d_inner=512, 8 heads) + tanh(α_gist)
      - Detail 路: hidden × detail cross-attn (d_inner=512, 8 heads) + tanh(α_detail)
      - 两路分别 gate 后相加注入

    与 DRA 的区别: Detail KV 是 CDM 编码器的 16 tokens 而非原始 768 tokens，
    所以 d_inner 不需要收窄，和 Gist 路规格一致。
    """

    def __init__(self, config: HVMConfig):
        super().__init__()
        d = config.d_model

        self.hidden_norm = nn.LayerNorm(d)

        self.gist_cross_attn = MultiHeadAttention(
            d, config.pim_num_heads, d_inner=config.d_pim_inner,
        )
        self.alpha_gist = nn.Parameter(torch.tensor(float(config.gate_alpha_init)))

        self.detail_cross_attn = MultiHeadAttention(
            d, config.pim_num_heads, d_inner=config.d_pim_inner,
        )
        self.alpha_detail = nn.Parameter(torch.tensor(float(config.gate_alpha_init)))

        self.last_stats = {}

    def forward(
        self,
        hidden_state: torch.Tensor,
        gist_feats: torch.Tensor,
        detail_feats: torch.Tensor,
    ) -> torch.Tensor:
        query = self.hidden_norm(hidden_state)

        delta_gist = self.gist_cross_attn(q=query, k=gist_feats, v=gist_feats)
        delta_detail = self.detail_cross_attn(q=query, k=detail_feats, v=detail_feats)

        gate_gist = torch.tanh(self.alpha_gist)
        gate_detail = torch.tanh(self.alpha_detail)

        inject_gist = gate_gist * delta_gist
        inject_detail = gate_detail * delta_detail
        injection = inject_gist + inject_detail

        with torch.no_grad():
            hidden_rms = hidden_state.detach().float().pow(2).mean().sqrt()
            delta_gist_rms = delta_gist.detach().float().pow(2).mean().sqrt()
            delta_detail_rms = delta_detail.detach().float().pow(2).mean().sqrt()
            inject_gist_rms = inject_gist.detach().float().pow(2).mean().sqrt()
            inject_detail_rms = inject_detail.detach().float().pow(2).mean().sqrt()
            delta_rms = (delta_gist + delta_detail).detach().float().pow(2).mean().sqrt()
            inject_rms = injection.detach().float().pow(2).mean().sqrt()
            self.last_stats = {
                "gate_gist": gate_gist.detach().float(),
                "gate_detail": gate_detail.detach().float(),
                "delta_gist_rms": delta_gist_rms,
                "delta_detail_rms": delta_detail_rms,
                "inject_gist_rms": inject_gist_rms,
                "inject_detail_rms": inject_detail_rms,
                "delta_rms": delta_rms,
                "inject_rms": inject_rms,
                "inject_hidden_ratio": inject_rms / (hidden_rms + 1e-6),
            }

        return hidden_state + injection


# ============================================================================
# Simple GME Injection Module (Stage1)
# ============================================================================

class SimpleGMEInjectionModule(nn.Module):
    """
    Stage1 最简注入模块：
      - 仅使用 gist memory（GME 输出）
      - 单次 hidden × gist cross-attention
      - 固定缩放 residual 注入（无 gate）
    """

    def __init__(self, config: HVMConfig):
        super().__init__()
        d = config.d_model
        self.hidden_norm = nn.LayerNorm(d)
        self.hidden_gist_cross_attn = MultiHeadAttention(
            d,
            config.pim_num_heads,
            d_inner=config.d_pim_inner,
        )
        self.inject_scale = float(config.inject_scale)
        self.last_stats = {}

    def forward(
        self,
        hidden_state: torch.Tensor,
        gist_feats: torch.Tensor,
    ) -> torch.Tensor:
        query = self.hidden_norm(hidden_state)
        delta = self.hidden_gist_cross_attn(
            q=query,
            k=gist_feats,
            v=gist_feats,
        )
        injection = self.inject_scale * delta

        with torch.no_grad():
            hidden_rms = hidden_state.detach().float().pow(2).mean().sqrt()
            delta_rms = delta.detach().float().pow(2).mean().sqrt()
            inject_rms = injection.detach().float().pow(2).mean().sqrt()
            self.last_stats = {
                "inject_scale": float(self.inject_scale),
                "delta_rms": delta_rms,
                "inject_rms": inject_rms,
                "inject_hidden_ratio": inject_rms / (hidden_rms + 1e-6),
            }

        return hidden_state + injection


# ============================================================================
# Prefrontal Injection Module (PIM)
# ============================================================================

class PrefrontalInjectionModule(nn.Module):
    """
    前额叶注入模块 — 核心创新。
    认知对应: 前额叶层次化执行控制。

    4 步融合:
        Step 1: Part × Gist Cross-Attention  (零件在整体中定位)
        Step 2: Part Self-Attention           (零件互相整合)
        Step 3: Visual × Text Cross-Attention (视觉语义对齐)
        Step 4: Hidden × Aligned Cross-Attention (提取修正量 delta)
    + 第 5 步: Adaptive Gated Injection
    """

    def __init__(self, config: HVMConfig):
        super().__init__()
        d = config.d_model

        # Step 1: Part × Gist Cross-Attention
        self.part_gist_cross_attn = MultiHeadAttention(d, config.pim_num_heads, d_inner=config.d_pim_inner)#512
        self.norm1 = nn.LayerNorm(d)

        # Step 2: Part Self-Attention
        self.part_self_attn = MultiHeadAttention(d, config.pim_num_heads, d_inner=config.d_pim_inner)
        self.norm2 = nn.LayerNorm(d)

        # Step 3: Visual × Text Cross-Attention
        self.visual_text_cross_attn = MultiHeadAttention(d, config.pim_num_heads, d_inner=config.d_pim_inner)
        self.norm3 = nn.LayerNorm(d)

        # Step 4: Hidden × Aligned Cross-Attention (bottleneck: 大序列)
        self.hidden_aligned_cross_attn = MultiHeadAttention(d, config.pim_num_heads, d_inner=config.d_pim_inner)

        # Step 5: Adaptive Gate
        self.gate = AdaptiveGate(d, base_alpha_init=config.gate_alpha_init)

    def forward(
        self,
        hidden_state: torch.Tensor,
        gist_feats: torch.Tensor,
        part_feats: torch.Tensor,
        text_feats: torch.Tensor,
        part_mask: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, L, d_model]   decoder block 输出
            gist_feats:   [B, 32, d_model]  GME 输出 (缓存)
            part_feats:   [B, 16, d_model]  PME 输出 (缓存)
            text_feats:   [B, Nt, d_model]  text embeddings (缓存)
            part_mask:    [B, 16]           bool, True=有效
            text_mask:    [B, Nt]           bool, True=有效

        Returns:
            [B, L, d_model]
        """
        # Step 1: Part × Gist Cross-Attention
        # "零件在整体中定位自己"
        step1_out = self.part_gist_cross_attn(
            q=part_feats, k=gist_feats, v=gist_feats
        )
        context_part = self.norm1(step1_out + part_feats)  # [B, 16, d]

        # Step 2: Part Self-Attention
        # "带全局上下文的零件互相整合"
        step2_out = self.part_self_attn(
            q=context_part, k=context_part, v=context_part,
            kv_mask=part_mask,
        )
        refined_part = self.norm2(step2_out + context_part)  # [B, 16, d]

        # Step 3: Visual × Text Cross-Attention
        # "视觉信息与文本语义对齐"
        #修复 PIM 第 3 步（Visual×Text）未使用文本 mask 的问题，避免 padding 文本参与注意力。必须传入 kv_mask=text_mask
        #不加 kv_mask，attention 会把 pad 位当成有效 key/value，稀释文本语义对齐质量
        step3_out = self.visual_text_cross_attn(
            q=refined_part, k=text_feats, v=text_feats,
            kv_mask=text_mask,
        )
        aligned_feats = self.norm3(step3_out + refined_part)  # [B, 16, d]

        # Step 4: Hidden × Aligned Cross-Attention
        # "LLM 的每个 token 从融合信息中提取修正量"
        delta = self.hidden_aligned_cross_attn(
            q=hidden_state, k=aligned_feats, v=aligned_feats,
            kv_mask=part_mask,
        )  # [B, L, d]

        # Step 5: Adaptive Gated Injection
        return self.gate(hidden_state, delta)  # [B, L, d]


# ============================================================================
# Utility: 参数统计
# ============================================================================

def count_parameters(module: nn.Module) -> int:
    """统计可训练参数数量"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def print_hvm_parameter_summary(config: HVMConfig):
    """打印 HVM 模块参数量概要"""
    gme = GistMemoryEncoder(config)
    if config.memory_mode == "gme" and config.inject_mode == "fixed":
        pme_params = 0
        pim = SimpleGMEInjectionModule(config)
        gate_params = 0
    else:
        pme = PartMemoryEncoder(config)
        pme_params = count_parameters(pme)
        pim = PrefrontalInjectionModule(config)
        gate_params = count_parameters(pim.gate)
    pim_params = count_parameters(pim)

    print("=" * 60)
    print("HVM-SVG Parameter Summary")
    print("=" * 60)
    print(f"  GME:                {count_parameters(gme):>12,}")
    print(f"  PME:                {pme_params:>12,}")
    print(f"  PIM × {config.num_pims}:")
    print(f"    Per PIM:          {pim_params:>12,}")
    print(f"    - Gate per PIM:   {gate_params:>12,}")
    print(f"    Total PIMs:       {pim_params * config.num_pims:>12,}")
    total = count_parameters(gme) + pme_params + pim_params * config.num_pims
    print(f"  {'─' * 40}")
    print(f"  Total HVM params:   {total:>12,}  ({total / 1e6:.1f}M)")
    print(f"  Base model (7B):    ~7,600,000,000")
    print(f"  HVM / Base:         {total / 7.6e9 * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    config = HVMConfig()
    print(f"PIM layer indices: {config.pim_layer_indices}")
    print(f"Num PIMs: {config.num_pims}")
    print()
    print_hvm_parameter_summary(config)
