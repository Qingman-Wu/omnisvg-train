"""
HVM-SVG: Hierarchical Visual Memory Modules
=============================================
所有 HVM-SVG 的核心神经网络模块：
- MultiHeadAttention: 高效的多头注意力（支持 bottleneck）
- QFormerLayer / QFormer: 用于 GME 和 PME 的 Query Transformer
- GistMemoryEncoder (GME): 全局场景记忆编码器，32 queries
- PartMemoryEncoder (PME): 局部零件记忆编码器，每组 4 queries
- AdaptiveGate: 双层门控（层级 + token 级）
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

    # === PIM ===
    pim_num_heads: int = 8
    pim_layer_interval: int = 4  # 每隔 N 层插入一个 PIM
    num_decoder_layers: int = 28

    # === RAG ===
    num_references: int = 3
    ref_text_max_length: int = 128

    @property
    def pim_layer_indices(self) -> List[int]:
        """PIM 插入位置（0-indexed，在该层之后插入）"""
        return list(range(
            self.pim_layer_interval - 1,
            self.num_decoder_layers,
            self.pim_layer_interval
        ))

    @property
    def num_pims(self) -> int:
        return len(self.pim_layer_indices)


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
        self.d_qformer = d_qformer

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
        queries = self.queries.expand(B, -1, -1)  # [B, N_query, d_qformer]

        # 过 QFormer layers
        for layer in self.layers:
            queries = layer(queries, kv, kv_mask=feat_mask)

        # 输出投影
        return self.output_norm(self.output_proj(queries))  # [B, N_query, d_out]


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

    相比旧版 pre-merge [32,32,1280]：
      - QFormer 输入从 3×1024=3072 tokens 降到 3×256=768 tokens，快 4 倍
      - 复用 Qwen 训练好的 merger 投影，特征质量更好
      - 与 PME 统一 d_vision=3584，架构更简洁
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
            group_feats_list = group_features_list[b]  # List of [1024, D]
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
            pad_len = self.max_tokens - actual_len
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

    def __init__(self, d_model: int):
        super().__init__()

        # 层级 gate: tanh(alpha), alpha 初始化为 0 → tanh(0) = 0
        self.base_alpha = nn.Parameter(torch.tensor(0.0))

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
        return hidden_state + layer_gate * token_gate * delta


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
        self.part_gist_cross_attn = MultiHeadAttention(d, config.pim_num_heads, d_inner=config.d_pim_inner)
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
        self.gate = AdaptiveGate(d)

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
    pme = PartMemoryEncoder(config)
    pim = PrefrontalInjectionModule(config)
    gate_params = count_parameters(pim.gate)
    pim_params = count_parameters(pim)

    print("=" * 60)
    print("HVM-SVG Parameter Summary")
    print("=" * 60)
    print(f"  GME:                {count_parameters(gme):>12,}")
    print(f"  PME:                {count_parameters(pme):>12,}")
    print(f"  PIM × {config.num_pims}:")
    print(f"    Per PIM:          {pim_params:>12,}")
    print(f"    - Gate per PIM:   {gate_params:>12,}")
    print(f"    Total PIMs:       {pim_params * config.num_pims:>12,}")
    total = count_parameters(gme) + count_parameters(pme) + pim_params * config.num_pims
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
