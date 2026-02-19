# HVM-SVG 详细架构图

本文档提供 HVM-SVG 所有核心组件的详细架构图，包含每一层的操作和维度变化。

## 目录
1. [整体架构](#整体架构)
2. [GME (Gist Memory Encoder)](#gme-gist-memory-encoder)
3. [PME (Part Memory Encoder)](#pme-part-memory-encoder)
4. [PIM (Prefrontal Injection Module)](#pim-prefrontal-injection-module)
5. [AdaptiveGate](#adaptivegate)
6. [QFormer](#qformer)
7. [MultiHeadAttention](#multiheadattention)

---

## 整体架构

```
输入数据流:
├─ 3张参考图 (全图) → Vision Pipeline → [B, 3, 256, 3584]
│                                          ↓
│                                        GME (Gist Memory Encoder)
│                                          ↓
│                                   [B, 32, 3584] gist_feats (缓存)
│
├─ Top-1 参考图的 4 个 group (独立渲染) → Vision Pipeline
│                                          ↓
│                                   List[[256, 3584], [256, 3584], ...]
│                                          ↓
│                                        PME (Part Memory Encoder)
│                                          ↓
│                              [B, 16, 3584] part_feats + [B, 16] mask (缓存)
│
└─ 3张参考图的文本描述 → Text Tokenizer → Text Embeddings
                                          ↓
                                   [B, Nt, 3584] text_feats (缓存)

解码阶段:
LLM Decoder Layers (28层)
  ├─ Layer 0-3:   标准 Transformer Block
  ├─ Layer 3:     输出 → PIM #0 (注入)
  ├─ Layer 4-7:   标准 Transformer Block
  ├─ Layer 7:     输出 → PIM #1 (注入)
  ├─ Layer 8-11:  标准 Transformer Block
  ├─ Layer 11:    输出 → PIM #2 (注入)
  ...
  └─ Layer 27:    输出 → PIM #6 (注入)
```

**配置参数:**
- `d_model = 3584` (Qwen2.5-7B hidden dim)
- `d_vision = 3584` (Vision features post-merge)
- `d_qformer = 1024` (QFormer 内部维度)
- `d_pim_inner = 512` (PIM attention bottleneck)
- `num_decoder_layers = 28`
- `pim_layer_interval = 4` → 7 个 PIMs (插入在 layer 3, 7, 11, 15, 19, 23, 27)

---

## GME (Gist Memory Encoder)

**作用:** 压缩 3 张参考图的全局信息为 32 个 gist tokens

**详细架构:**

```
输入: ref_features [B, 3, 256, 3584]
                    │
                    ├─ Reshape
                    ↓
         [B, 768, 3584]  (3×256 = 768 tokens)
                    │
                    ↓
        ┌───────────────────────────────────────┐
        │         QFormer (GME)                 │
        │  num_queries = 32                     │
        │  num_layers = 6                       │
        │  d_qformer = 1024                     │
        │  n_heads = 8                          │
        │  ff_mult = 4                          │
        └───────────────────────────────────────┘
                    │
                    ↓
            [B, 32, 3584]
             gist_feats (缓存用于所有 PIMs)

QFormer 内部结构 (见下方 QFormer 详图):
  ├─ Input Proj: [B, 768, 3584] → [B, 768, 1024]
  ├─ Learnable Queries: [B, 32, 1024]
  ├─ 6× QFormerLayer (Self-Attn → Cross-Attn → FFN)
  ├─ Output Proj: [B, 32, 1024] → [B, 32, 3584]
  └─ Output Norm
```

**参数量: ~32.8M**

---

## PME (Part Memory Encoder)

**作用:** 将 Top-1 参考图的 1~4 个 group 分别编码为 4 个 tokens (共 4~16 tokens)

**详细架构:**

```
输入: group_features_list (batch)
      每个样本: List[[256, 3584], [256, 3584], ...] (1~4 个 groups)
                     │
                     ├─ 对每个 group 独立处理
                     ↓
         ┌─────────────────────────────────────────┐
         │      QFormer (PME)                      │
         │  num_queries = 4                        │
         │  num_layers = 4                         │
         │  d_qformer = 1024                       │
         │  n_heads = 8                            │
         │  ff_mult = 4                            │
         └─────────────────────────────────────────┘
                     │
                     ↓
              [1, 4, 3584]  (每个 group)
                     │
                     ├─ Concat 所有 groups
                     ↓
         [N_groups×4, 3584]  (N_groups ∈ [1,4])
                     │
                     ├─ Pad 到 max_tokens=16
                     ↓
              [16, 3584]
                     │
                     ├─ Stack batch
                     ↓
输出: part_feats [B, 16, 3584]
     part_mask  [B, 16]  (True=有效, False=padding)

示例:
  样本1: 3 groups → 12 tokens 有效, 4 tokens padding
  样本2: 2 groups → 8 tokens 有效, 8 tokens padding
  样本3: 4 groups → 16 tokens 有效, 0 padding
```

**参数量: ~21.9M**

---

## PIM (Prefrontal Injection Module)

**作用:** 4 步注意力融合 + 自适应门控注入

**完整架构:**

```
输入:
  ├─ hidden_state [B, L, 3584]   (decoder block 输出)
  ├─ gist_feats [B, 32, 3584]    (GME 缓存)
  ├─ part_feats [B, 16, 3584]    (PME 缓存)
  ├─ text_feats [B, Nt, 3584]    (文本缓存)
  ├─ part_mask [B, 16]
  └─ text_mask [B, Nt]

┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Part × Gist Cross-Attention                           │
│  "零件在整体中定位"                                              │
│                                                                  │
│  Q = part_feats [B, 16, 3584]                                   │
│  K = V = gist_feats [B, 32, 3584]                               │
│        │                                                         │
│        ├─ MultiHeadAttention(n_heads=8, d_inner=512)           │
│        │   ├─ to_q: [B, 16, 3584] → [B, 16, 512]               │
│        │   ├─ to_k: [B, 32, 3584] → [B, 32, 512]               │
│        │   ├─ to_v: [B, 32, 3584] → [B, 32, 512]               │
│        │   ├─ Reshape: [B, 8, 16, 64] × [B, 8, 32, 64]         │
│        │   ├─ SDPA: [B, 8, 16, 32] @ [B, 8, 32, 64]            │
│        │   │         → [B, 8, 16, 64]                           │
│        │   └─ to_out: [B, 16, 512] → [B, 16, 3584]             │
│        ↓                                                         │
│  step1_out [B, 16, 3584]                                        │
│        │                                                         │
│        ├─ Residual + LayerNorm                                  │
│        ↓                                                         │
│  context_part [B, 16, 3584]                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Part Self-Attention                                    │
│  "零件互相整合"                                                  │
│                                                                  │
│  Q = K = V = context_part [B, 16, 3584]                         │
│  kv_mask = part_mask [B, 16]                                    │
│        │                                                         │
│        ├─ MultiHeadAttention(n_heads=8, d_inner=512)           │
│        │   ├─ to_qkv: [B, 16, 3584] → [B, 16, 512] ×3          │
│        │   ├─ Reshape: [B, 8, 16, 64] each                     │
│        │   ├─ 构建 attn_mask: [B, 16] → [B, 1, 1, 16]          │
│        │   │   padding 位置 → -inf, 有效位置 → 0                │
│        │   ├─ SDPA with mask: [B, 8, 16, 16]                   │
│        │   └─ to_out: [B, 16, 512] → [B, 16, 3584]             │
│        ↓                                                         │
│  step2_out [B, 16, 3584]                                        │
│        │                                                         │
│        ├─ Residual + LayerNorm                                  │
│        ↓                                                         │
│  refined_part [B, 16, 3584]                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Visual × Text Cross-Attention                         │
│  "视觉信息与文本语义对齐"                                        │
│                                                                  │
│  Q = refined_part [B, 16, 3584]                                 │
│  K = V = text_feats [B, Nt, 3584]                               │
│  kv_mask = text_mask [B, Nt]                                    │
│        │                                                         │
│        ├─ MultiHeadAttention(n_heads=8, d_inner=512)           │
│        │   ├─ to_q: [B, 16, 3584] → [B, 16, 512]               │
│        │   ├─ to_k: [B, Nt, 3584] → [B, Nt, 512]               │
│        │   ├─ to_v: [B, Nt, 3584] → [B, Nt, 512]               │
│        │   ├─ 构建 attn_mask from text_mask                     │
│        │   ├─ SDPA: [B, 8, 16, Nt] @ [B, 8, Nt, 64]            │
│        │   │         → [B, 8, 16, 64]                           │
│        │   └─ to_out: [B, 16, 512] → [B, 16, 3584]             │
│        ↓                                                         │
│  step3_out [B, 16, 3584]                                        │
│        │                                                         │
│        ├─ Residual + LayerNorm                                  │
│        ↓                                                         │
│  aligned_feats [B, 16, 3584]  (视觉-文本对齐特征)               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Hidden × Aligned Cross-Attention                      │
│  "LLM 每个 token 提取修正量"                                    │
│                                                                  │
│  Q = hidden_state [B, L, 3584]  (L 可能很大, 如 2048)          │
│  K = V = aligned_feats [B, 16, 3584]                            │
│  kv_mask = part_mask [B, 16]                                    │
│        │                                                         │
│        ├─ MultiHeadAttention(n_heads=8, d_inner=512)           │
│        │   ├─ to_q: [B, L, 3584] → [B, L, 512]                 │
│        │   ├─ to_k: [B, 16, 3584] → [B, 16, 512]               │
│        │   ├─ to_v: [B, 16, 3584] → [B, 16, 512]               │
│        │   ├─ SDPA: [B, 8, L, 16] @ [B, 8, 16, 64]             │
│        │   │         → [B, 8, L, 64]                            │
│        │   └─ to_out: [B, L, 512] → [B, L, 3584]               │
│        ↓                                                         │
│  delta [B, L, 3584]  (修正量)                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Adaptive Gated Injection                              │
│  (见下方 AdaptiveGate 详图)                                     │
│                                                                  │
│  输入:                                                           │
│    ├─ hidden_state [B, L, 3584]                                │
│    └─ delta [B, L, 3584]                                        │
│        │                                                         │
│        ├─ AdaptiveGate (双层门控)                               │
│        ↓                                                         │
│  输出: [B, L, 3584]                                             │
└─────────────────────────────────────────────────────────────────┘

输出: hidden_state [B, L, 3584]  (注入后)
```

**参数量 (每个 PIM): ~40.5M**
- Step 1-4 的 4 个 MultiHeadAttention: 每个约 9.2M
- AdaptiveGate: ~2.6M

---

## AdaptiveGate

**作用:** 双层自适应门控 (Layer-level + Token-level)

**详细架构:**

```
输入:
  ├─ hidden_state [B, L, 3584]  (decoder 输出)
  └─ delta [B, L, 3584]         (PIM 计算的修正量)

┌─────────────────────────────────────────────────────────────────┐
│  Layer-level Gate (Flamingo 式)                                │
│                                                                  │
│  base_alpha: Parameter(0.0)  (初始化为 0)                       │
│        │                                                         │
│        ├─ tanh(base_alpha)                                      │
│        ↓                                                         │
│  layer_gate: scalar  (初始 ≈ 0, 训练中逐渐增大)                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Token-level Gate (GateRA 式, input-dependent)                 │
│                                                                  │
│  gate_input = concat([hidden_state, delta], dim=-1)            │
│               [B, L, 3584] + [B, L, 3584]                       │
│               → [B, L, 7168]                                    │
│        │                                                         │
│        ├─ token_gate MLP:                                       │
│        │   ├─ Linear: [B, L, 7168] → [B, L, 896]  (d//4)       │
│        │   ├─ GELU                                              │
│        │   ├─ Linear: [B, L, 896] → [B, L, 1]                  │
│        │   │   (权重和偏置初始化为 0)                            │
│        │   └─ Sigmoid → [B, L, 1]                               │
│        │       (初始 sigmoid(0) = 0.5)                          │
│        ↓                                                         │
│  token_gate [B, L, 1]                                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  组合注入                                                        │
│                                                                  │
│  output = hidden_state + layer_gate × token_gate × delta       │
│           [B, L, 3584] + scalar × [B, L, 1] × [B, L, 3584]     │
│                        → [B, L, 3584]                           │
│                                                                  │
│  训练初期:                                                       │
│    layer_gate ≈ 0 → 几乎无注入 (保护预训练 LLM)                │
│  训练后期:                                                       │
│    layer_gate ∈ [0.1, 0.5] → token_gate 自适应选择注入位置     │
└─────────────────────────────────────────────────────────────────┘

输出: [B, L, 3584]
```

**参数量: ~2.6M**

**关键设计:**
- `base_alpha` 初始化为 0 → 训练初期 gate ≈ 0
- `token_gate` 最后一层初始化为 0 → 初始输出 sigmoid(0) = 0.5
- 双层控制: 全局强度 × 局部选择

---

## QFormer

**作用:** 可学习 queries 从大量 tokens 中提取固定数量的表示

**详细架构:**

```
超参数示例 (GME):
  ├─ num_queries = 32
  ├─ num_layers = 6
  ├─ d_qformer = 1024
  ├─ d_vision = 3584
  ├─ d_out = 3584
  ├─ n_heads = 8
  └─ ff_mult = 4

输入: image_feats [B, N_img, 3584]
      feat_mask [B, N_img]  (optional)

┌─────────────────────────────────────────────────────────────────┐
│  输入投影层                                                      │
│                                                                  │
│  image_feats [B, N_img, 3584]                                   │
│        │                                                         │
│        ├─ input_proj: Linear(3584, 1024, bias=False)           │
│        ├─ input_norm: LayerNorm(1024)                           │
│        ↓                                                         │
│  kv [B, N_img, 1024]                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  可学习 Queries                                                  │
│                                                                  │
│  queries: Parameter([1, 32, 1024]) × 0.02                       │
│        │                                                         │
│        ├─ expand to batch                                       │
│        ↓                                                         │
│  queries [B, 32, 1024]                                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  QFormer Layers (6 层)                                          │
│                                                                  │
│  for layer in layers:                                           │
│      queries = QFormerLayer(queries, kv, kv_mask)              │
│                                                                  │
│  (见下方 QFormerLayer 详图)                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  输出投影层                                                      │
│                                                                  │
│  queries [B, 32, 1024]                                          │
│        │                                                         │
│        ├─ output_proj: Linear(1024, 3584, bias=False)          │
│        ├─ output_norm: LayerNorm(3584)                          │
│        ↓                                                         │
│  output [B, 32, 3584]                                           │
└─────────────────────────────────────────────────────────────────┘
```

### QFormerLayer (单层)

```
输入:
  ├─ queries [B, Nq, 1024]
  ├─ kv [B, Nk, 1024]
  └─ kv_mask [B, Nk]

┌─────────────────────────────────────────────────────────────────┐
│  Self-Attention (queries 之间)                                  │
│                                                                  │
│  q_norm = norm_sa(queries)  [B, Nq, 1024]                       │
│        │                                                         │
│        ├─ MultiHeadAttention(d_model=1024, n_heads=8)          │
│        │   Q = K = V = q_norm                                   │
│        │   ├─ to_qkv: [B, Nq, 1024] → [B, Nq, 1024] ×3         │
│        │   ├─ Reshape: [B, 8, Nq, 128] each                    │
│        │   ├─ SDPA: [B, 8, Nq, Nq]                             │
│        │   └─ to_out: [B, Nq, 1024]                            │
│        ↓                                                         │
│  queries = queries + self_attn_out  (Residual)                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Cross-Attention (queries attend to image features)            │
│                                                                  │
│  q_norm = norm_ca(queries)  [B, Nq, 1024]                       │
│        │                                                         │
│        ├─ MultiHeadAttention(d_model=1024, n_heads=8)          │
│        │   Q = q_norm, K = V = kv                               │
│        │   kv_mask applied                                      │
│        │   ├─ to_q: [B, Nq, 1024] → [B, Nq, 1024]              │
│        │   ├─ to_k: [B, Nk, 1024] → [B, Nk, 1024]              │
│        │   ├─ to_v: [B, Nk, 1024] → [B, Nk, 1024]              │
│        │   ├─ 构建 attn_mask from kv_mask                       │
│        │   ├─ SDPA: [B, 8, Nq, Nk]                             │
│        │   └─ to_out: [B, Nq, 1024]                            │
│        ↓                                                         │
│  queries = queries + cross_attn_out  (Residual)                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Feed-Forward Network                                           │
│                                                                  │
│  q_norm = norm_ff(queries)  [B, Nq, 1024]                       │
│        │                                                         │
│        ├─ FFN:                                                   │
│        │   ├─ Linear: [B, Nq, 1024] → [B, Nq, 4096]  (×4)      │
│        │   ├─ GELU                                              │
│        │   └─ Linear: [B, Nq, 4096] → [B, Nq, 1024]            │
│        ↓                                                         │
│  queries = queries + ffn_out  (Residual)                        │
└─────────────────────────────────────────────────────────────────┘

输出: queries [B, Nq, 1024]
```

---

## MultiHeadAttention

**作用:** 通用多头注意力，支持 bottleneck (d_inner < d_model)

**详细架构:**

```
超参数示例:
  ├─ d_model = 3584
  ├─ n_heads = 8
  └─ d_inner = 512  (bottleneck)

计算:
  d_head = d_inner / n_heads = 512 / 8 = 64

输入:
  ├─ q [B, Nq, 3584]
  ├─ k [B, Nk, 3584]
  ├─ v [B, Nk, 3584]
  └─ kv_mask [B, Nk]  (optional, bool, True=有效)

┌─────────────────────────────────────────────────────────────────┐
│  线性投影层 (降维到 bottleneck)                                 │
│                                                                  │
│  Q = to_q(q)  [B, Nq, 3584] → [B, Nq, 512]                     │
│  K = to_k(k)  [B, Nk, 3584] → [B, Nk, 512]                     │
│  V = to_v(v)  [B, Nk, 3584] → [B, Nk, 512]                     │
│                                                                  │
│  (无 bias, 减少参数)                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  多头 Reshape                                                    │
│                                                                  │
│  Q: [B, Nq, 512] → [B, Nq, 8, 64] → [B, 8, Nq, 64]             │
│  K: [B, Nk, 512] → [B, Nk, 8, 64] → [B, 8, Nk, 64]             │
│  V: [B, Nk, 512] → [B, Nk, 8, 64] → [B, 8, Nk, 64]             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  构建 Attention Mask (如果提供 kv_mask)                         │
│                                                                  │
│  kv_mask [B, Nk]  (True=有效, False=padding)                   │
│        │                                                         │
│        ├─ [:, None, None, :]  → [B, 1, 1, Nk]                  │
│        ├─ to(dtype=Q.dtype)                                     │
│        ├─ masked_fill(mask==0, -inf)  (padding → -inf)         │
│        ├─ masked_fill(mask==1, 0.0)   (有效 → 0)               │
│        ↓                                                         │
│  attn_mask [B, 1, 1, Nk]                                        │
│  (SDPA 格式: 0=有效, -inf=mask)                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Scaled Dot-Product Attention (Flash Attention)                │
│                                                                  │
│  Q: [B, 8, Nq, 64]                                              │
│  K: [B, 8, Nk, 64]                                              │
│  V: [B, 8, Nk, 64]                                              │
│  attn_mask: [B, 1, 1, Nk]                                       │
│        │                                                         │
│        ├─ scores = Q @ K^T / sqrt(64)                           │
│        │          [B, 8, Nq, Nk]                                │
│        ├─ scores = scores + attn_mask  (广播)                   │
│        ├─ attn_weights = softmax(scores, dim=-1)               │
│        │                 [B, 8, Nq, Nk]                         │
│        ├─ out = attn_weights @ V                                │
│        │       [B, 8, Nq, Nk] @ [B, 8, Nk, 64]                 │
│        ↓       → [B, 8, Nq, 64]                                 │
│  out [B, 8, Nq, 64]                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  合并多头 & 输出投影                                            │
│                                                                  │
│  out: [B, 8, Nq, 64]                                            │
│        │                                                         │
│        ├─ transpose(1, 2) → [B, Nq, 8, 64]                     │
│        ├─ reshape → [B, Nq, 512]                                │
│        ├─ to_out(out) → [B, Nq, 3584]                          │
│        │   (升维回 d_model)                                     │
│        ↓                                                         │
│  output [B, Nq, 3584]                                           │
└─────────────────────────────────────────────────────────────────┘
```

**参数量计算 (d_model=3584, d_inner=512):**
- `to_q`: 3584 × 512 = 1,835,008
- `to_k`: 3584 × 512 = 1,835,008
- `to_v`: 3584 × 512 = 1,835,008
- `to_out`: 512 × 3584 = 1,835,008
- **总计: 7,340,032 ≈ 7.3M**

**Bottleneck 优势:**
- 无 bottleneck (d_inner=3584): 4 × (3584²) ≈ 51.4M
- 有 bottleneck (d_inner=512): 4 × (3584 × 512) ≈ 7.3M
- **节省 86% 参数!**

---

## 维度流动总结

### GME 数据流
```
[B, 3, 256, 3584]  参考图 post-merge features
       ↓ reshape
[B, 768, 3584]     展平
       ↓ input_proj
[B, 768, 1024]     投影到 QFormer 空间
       ↓ + learnable queries [B, 32, 1024]
       ↓ 6× QFormerLayer
[B, 32, 1024]      提取的 gist queries
       ↓ output_proj
[B, 32, 3584]      gist_feats (对齐 LLM 空间)
```

### PME 数据流
```
List[[256, 3584], ...]  每个 group 的 post-merge features
       ↓ 逐个过 QFormer
List[[4, 3584], ...]    每个 group → 4 tokens
       ↓ concat
[N_groups×4, 3584]      1~4 groups → 4~16 tokens
       ↓ pad to 16
[B, 16, 3584]           part_feats
```

### PIM 数据流
```
hidden [B, L, 3584]    decoder 输出
gist   [B, 32, 3584]   GME 缓存
part   [B, 16, 3584]   PME 缓存
text   [B, Nt, 3584]   文本缓存

Step 1: part × gist → [B, 16, 3584]
Step 2: part self   → [B, 16, 3584]
Step 3: part × text → [B, 16, 3584]
Step 4: hidden × aligned → [B, L, 3584]  (delta)
Step 5: gate(hidden, delta) → [B, L, 3584]
```

---

## 参数量统计

```
HVM-SVG Parameter Summary
============================================================
  GME:                      32,804,864  (32.8M)
  PME:                      21,871,616  (21.9M)
  PIM × 7:
    Per PIM:                40,531,968  (40.5M)
    - Gate per PIM:          2,583,553  (2.6M)
    Total PIMs:            283,723,776  (283.7M)
  ────────────────────────────────────
  Total HVM params:        338,400,256  (338.4M)
  Base model (7B):      ~7,600,000,000
  HVM / Base:                    4.5%
============================================================
```

**关键设计:**
- HVM 只增加 4.5% 参数量
- 多数参数在 PIMs (每层深度融合)
- Bottleneck attention 大幅节省参数
- 训练初期 gates=0 保护预训练权重

---

## 认知科学对应

| 模块 | 认知概念 | 功能 |
|------|---------|------|
| **GME** | Scene Gist | 快速、压缩、持久的整体结构记忆 |
| **PME** | Working Memory | 容量有限(4±1)、高精度的零件表征 |
| **PIM** | Prefrontal Control | 层次化执行控制，多模态信息融合 |
| **AdaptiveGate** | Attention Control | 选择性注意，过滤无关信息 |

---

## 训练特性

1. **Warm Start**: `base_alpha` 初始化为 0，训练初期 PIM 几乎无影响
2. **Gradual Injection**: 学习率 warmup 期间，gates 逐渐打开
3. **Adaptive**: Token-level gate 根据输入动态调整注入强度
4. **Efficient**: Bottleneck attention 节省 86% 参数和计算

---

生成时间: 2026-02-18
配置: Qwen2.5-7B, 28 layers, 7 PIMs
