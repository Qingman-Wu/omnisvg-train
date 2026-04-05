# Fetch, Frame, and Focus: Retrieval-Augmented Panoramic-Spotlight Memory for Scalable Vector Graphics Generation

## 完整方案文档（可直接复制到新对话用于论文写作）

---

## 1. Paper Metadata

**Title**: Fetch, Frame, and Focus: Retrieval-Augmented Panoramic-Spotlight Memory for Scalable Vector Graphics Generation


---

## 2. Problem Statement

Autoregressive text-to-SVG models (e.g., OmniSVG) decode long SVG token sequences conditioned solely on text prompts. As structural complexity grows, this text-only decoding leads to three failure modes:
1. **Part omission** — structurally complex components are dropped
2. **Layout drift** — spatial arrangement degrades over long generation
3. **Repetitive strokes** — the model falls into repetitive patterns

We trace these failures to the absence of **visual grounding** during generation. We recast text-to-SVG as **grounded visual-program synthesis** and introduce *Fetch, Frame, and Focus*.

---

## 3. Method Overview

| Stage | Paper Name | Core Function | Output |
|-------|-----------|--------------|--------|
| **Fetch** | Reference Retrieval + Decomposition | Retrieve top-K similar SVGs, decompose into semantic groups | Scene Tokens, Detail Tokens |
| **Frame** | Panoramic Memory Encoding | Compress global scene context via Q-Former | Panoramic Memory [B, 32, d] |
| **Focus** | Spotlight Memory Encoding + Routing | Extract per-group detail slots, dynamically route to decoder | Spotlight Memory [B, 12, d] → Routed Detail [B, L, d] |

These memories are injected into the frozen autoregressive decoder's **last 4 layers** (Layer 25-28, 1-indexed) through a dual-path injection module.

### Architecture Diagram Description (最终版)

架构图分为四个区域：
- **左：Fetch** — TEXT → cross-modal 相似度检索 → 3 张参考图 → 每张分解为 4 个 semantic groups → Image Tokenizer (冻结 CLIP ViT) → 上路输出 Scene Tokens (完整参考图特征), 下路输出 Detail Tokens (group 子图特征)
- **中左：Frame (蓝色)** — 32 个 Learnable Queries → Q-Former Layer (Self-Attn, Cross-Attn, FFN) ← Scene Tokens 作为 K,V → 输出 Panoramic Memory
- **中右：Focus (红/粉色)** — 12 个 Learnable Queries → Q-Former Layer (Self-Attn, Cross-Attn, FFN) ← Detail Tokens 作为 K,V + Positional Encoding (Group ID 0-11) → 输出 Spotlight Memory
- **底部：Memory Injection (黄色)** — Autoregressive Decoder (Layer 1 → ... → Layer 25-28 → SVG Detokenizer), 最后 4 层各插入一个 Memory Injection Layer:
  - 左路: Panoramic Memory → Cross-Attn ← Hidden State (标准多头交叉注意力, soft, 全量访问)
  - 右路: Spotlight Memory → Spotlight Router (Q-K Scorer + Top-1 选择) ← Hidden State (稀疏访问)
  - 两路各乘以 adaptive gate 后相加注入 hidden state
- **最右：Spotlight Router 详细面板** — 展示 3 个生成步骤 (画猫耳、画尾巴、画胡须) 的动态路由行为: 左侧 Spotlight Memory 中不同 slot 被选中 (稀疏, 细箭头), 右侧 Panoramic Memory 全量注入 (密集, 粗箭头)

---

## 4. Module Details

### 4.1 Fetch: Reference Retrieval and Decomposition

**目的**: 为给定 text prompt 检索视觉上相似的参考 SVG, 并将其结构化分解。

**流程**:
1. **Text Embedding**: CLIP-ViT-L/14 text encoder → text embedding
2. **Image Embedding**: 训练集所有 SVG 渲染为 PNG 后预计算 CLIP image embedding, 存入 FAISS index
3. **kNN Search**: 用 text embedding 在 FAISS index 中检索 **Top-3** 最相似的参考 SVG
4. **SVG Decomposition**: 每张参考 SVG 通过预计算的语义分组 (基于 path 的空间聚类), 分为 **4 个 groups** (semantic parts)
   - 3 refs x 4 groups/ref = **12 groups** 总计
   - 每个 group 记录 path_indices (SVG 中哪些 path 属于该 group)
   - 每个 group 有 spatial tag metadata: [cx, cy, w, h, z_start, z_end] (中心坐标、宽高、深度范围)
5. **Image Tokenizer (冻结)**: CLIP ViT-L/14 image encoder (通过 Qwen2.5-VL visual module)
   - 完整参考图 → **Scene Tokens** [B, 3, 256, 3584] (3 张完整参考图, 每张 256 tokens, 经 Qwen VLM vision merger 后维度 3584)
   - Group 子图 → **Detail Tokens** [B, 12, 256, 3584] (12 个 group 子图, 每个 256 tokens)

**代码对应**: 预计算阶段完成, 推理时从 hvm_precomputed 目录加载。Image Tokenizer 使用 Qwen2.5-VL 自带的 visual 模块 (冻结)。

---

### 4.2 Frame: Panoramic Memory Encoding (GME)

**目的**: 将 3 张参考图的全局场景信息压缩为固定长度的全局记忆。

**架构**: GistMemoryEncoder (GME) — 标准 Q-Former

**详细配置**:
- **Learnable Queries**: 32 个, 维度 d_qformer = 1024
- **Q-Former Layers**: 6 层, 每层: Self-Attn → Cross-Attn (queries attend to Scene Tokens as K,V) → FFN (4x expansion)
- **输入**: Scene Tokens [B, 3x256, 3584] = [B, 768, 3584] (3 张参考图展平)
  - 先通过 ref_proj: Linear(3584 → 1024) 投影到 Q-Former 内部维度
- **输出**: output_proj: Linear(1024 → 3584) + LayerNorm → **Panoramic Memory** [B, 32, 3584]
- **参数量**: ~102M

**关键设计**: 768 个 vision tokens 压缩为 32 个 queries, 实现 24:1 高比例信息压缩, 保留全局场景结构。

**代码对应**: hvm_modules.py → class GistMemoryEncoder (L422-459)

---

### 4.3 Focus: Spotlight Memory Encoding (CDM) + Routing (EDR)

#### 4.3.1 Spotlight Memory Encoding — GroupwiseCDMEncoder

**目的**: 从 12 个 group 的视觉特征中提取可寻址的局部细节 slot。

**架构**: GroupwiseCDMEncoder — 共享权重的 group-wise Q-Former

**详细配置**:
- **Learnable Queries**: 1 个/group (共享权重), 维度 d_qformer = 1024
  - 12 groups x 1 query/group = **12 detail slots**
- **Q-Former Layers**: 6 层, 共享权重 (所有 group 共用同一套 Q-Former), 每层:
  - Self-Attn
  - Cross-Attn (gist): **当前禁用** (cdm_disable_gist=true)
  - Cross-Attn (ref): queries attend to 该 group 的 256 个 Detail Tokens
  - FFN
- **输入**: Detail Tokens [B, 12, 256, 3584]
  - 处理方式: 每个 group 独立过共享 CDM → [Bx12, 1, 1024] → reshape → [B, 12, 1024]
- **Spatial Tags (Group ID + Tag Metadata)**:
  - group_id_embedding: Embedding(12, 3584), 全局编号 0-11
    - ref0 → group 0,1,2,3; ref1 → group 4,5,6,7; ref2 → group 8,9,10,11
  - tag_meta_mlp: Linear(6 → 1024 → 3584), 输入 [cx, cy, w, h, z_start, z_end]
  - 注入方式: CDM 输出后加到每个 slot 上 → slot_tag_norm(slot + tag_emb)
- **输出**: output_proj: Linear(1024 → 3584) + LayerNorm → **Spotlight Memory** [B, 12, 3584]
- **参数量**: ~82M

**关键设计**:
1. Group-wise 处理确保每个 slot 只关注自己 group 的 256 tokens, 不被其他 group 干扰
2. 共享权重减少参数, 通过 Group ID 区分不同 group
3. 每个 group 只输出 1 个 slot, 实现极致压缩

**代码对应**: hvm_modules.py → class GroupwiseCDMEncoder (L664-843)

#### 4.3.2 Spotlight Router — DetailRouter

**目的**: 在解码时, 让每个 token 动态选择最相关的 detail slot, 实现按需访问。

**架构**: DetailRouter — 可学习的 dot-product scorer + top-k 稀疏选择

**详细流程**:

```
输入: hidden_state [B, L, 3584], detail_bank (Spotlight Memory) [B, 12, 3584]

1.  query = LayerNorm(hidden_state)                          [B, L, 3584]
2.  q = q_proj(query)                                        [B, L, 256]     ← Linear(3584→256, no bias)
3.  k = k_proj(detail_bank)                                  [B, 12, 256]    ← Linear(3584→256, no bias)
4.  score = (q * kT) / sqrt(256)                             [B, L, 12]      ← dot-product scoring
5.  full_prob = softmax(score, dim=-1)                       [B, L, 12]      ← 完整概率分布
6.  entropy = -sum(p * log(p))                               [B, L, 1]
7.  conf = clamp(1 - entropy / log(12), 0, 1)               [B, L, 1]       ← 路由置信度
8.  topk_vals, topk_idx = topk(score, k=1, dim=-1)          对每个 token 选最相关 slot
9.  topk_prob = softmax(topk_vals, dim=-1)                   [B, L, 1]       ← top-k 重归一化 (k=1时为1.0)
10. sparse_prob = scatter(zeros, topk_idx, topk_prob)        [B, L, 12]      ← 稀疏概率矩阵
11. routed_detail = sparse_prob * detail_bank                [B, L, 3584]    ← 加权聚合 (实际是选中slot的特征)
```

**与 Cross-Attention 的关键区别**:
- 没有 V 投影: 直接使用原始 slot 向量, 不经过 V 投影
- 没有多头: 单头 dot-product 打分
- Hard top-k 选择: 不是 soft attention weighted sum, 而是 top-1 稀疏选择
- Confidence 调制: 通过 entropy-based conf 衡量路由确定性

**参数量**: q_proj + k_proj = 2 x (3584 x 256) = ~1.8M/层 x 4 层 = ~7.3M

**代码对应**: hvm_modules.py → class DetailRouter (L1708-1854)

---

### 4.4 Memory Injection Layer (PIM) — EDRInjectionModule

**目的**: 将 Panoramic Memory 和路由后的 Spotlight detail 注入冻结 decoder 的 hidden state。

**插入位置**: Decoder Layer 24, 25, 26, 27 (0-indexed) = Layer 25, 26, 27, 28 (1-indexed), 即**最后 4 层**

**架构**: EDRInjectionModule — 双路注入模块

**详细流程**:

```
输入: hidden_state [B, L, 3584], panoramic_memory [B, 32, 3584], spotlight_memory [B, 12, 3584]

query = LayerNorm(hidden_state)                                          [B, L, 3584]

[Panoramic 路 (左路) — Cross-Attention]
delta_gist = MultiHeadAttention(Q=query, K=panoramic, V=panoramic)       [B, L, 3584]
    - 8 heads, d_inner=512 (bottleneck)
    - 标准 cross-attention: 每个 token soft attend 所有 32 panoramic queries
gate_gist = tanh(alpha_gist)                                              标量 (可学习, 初始化 0.05)
inject_gist = gate_gist * delta_gist                                     [B, L, 3584]

[Spotlight 路 (右路) — Router]
routed_detail, conf, stats = DetailRouter(query, spotlight_memory)        [B, L, 3584], [B, L, 1]
    - 每个 token 独立选 top-1 slot
    - conf in [0,1] 衡量路由置信度
gate_detail = tanh(alpha_detail)                                          标量 (可学习, 初始化 0.05)
inject_detail = gate_detail * conf * routed_detail                       [B, L, 3584]

[汇合]
injection = inject_gist + inject_detail                                   [B, L, 3584]
hidden_state = hidden_state + injection                                   残差连接
```

**两路的核心对比**:

| 属性 | Panoramic 路 (左) | Spotlight 路 (右) |
|------|-------------------|-------------------|
| 机制 | Multi-Head Cross-Attention | Dot-Product Router + Top-1 |
| 访问范围 | 所有 32 个 panoramic tokens | 12 个 detail slots 中选 1 个 |
| 访问方式 | Soft (每个 token 加权所有) | Hard Sparse (每个 token 只取 1 个) |
| V 投影 | 有 | 无 (直接用原始 slot) |
| 多头 | 8 heads | 单头 |
| Conf 调制 | 无 | 有 (entropy-based confidence) |
| 语义 | 广播全局场景上下文 | 按需读取局部细节参考 |
| 对应隐喻 | 全景观看 (Panoramic) | 聚焦放大 (Spotlight) |

**参数量**: 每个 PIM ~25M x 4 层 = ~100M

**代码对应**: hvm_modules.py → class EDRInjectionModule (L1857-1975)

---

## 5. Design Principles (Three Design Principles)

通过系统性 ablation 得出三条设计原则:

1. **Structurally Decomposed > Flattened**: Reference information should be decomposed into semantic groups (4 groups/ref) rather than flattened into a single long sequence. Group-wise processing preserves part boundaries.

2. **Selectively Accessed > Uniformly Broadcast**: Detail memory should be selectively accessed via a router (top-k) rather than uniformly broadcast via attention to all slots. The router lets each decoding token fetch only the most relevant reference detail.

3. **Late Layer Injection > Early Layer Injection**: Memory injection in late decoder layers (last 4 of 28) is more effective than early layers. Late layers handle fine-grained spatial decisions where visual grounding is most needed.

---

## 6. Naming Convention Mapping (Paper Term <-> Code)

| Paper Term | Code Variable/Class | Shape/Value |
|---|---|---|
| Image Tokenizer | Qwen2.5-VL visual module (frozen CLIP ViT) | - |
| Scene Tokens | ref_features (full images) | [B, 3, 256, 3584] |
| Detail Tokens | part_features (group images) | [B, 12, 256, 3584] |
| Frame / Panoramic Memory Encoder | GistMemoryEncoder (GME) | - |
| Learnable Queries (Frame, x32) | gme.qformer.queries | [1, 32, 1024] |
| Panoramic Memory | gist_feats | [B, 32, 3584] |
| Focus / Spotlight Memory Encoder | GroupwiseCDMEncoder (CDM) | - |
| Learnable Queries (Focus, x12) | cdm.queries (1/group, shared) | [1, 1, 1024] |
| Group ID (0-11) | group_id_embedding | Embedding(12, 3584) |
| Spatial Tags [cx,cy,w,h,z_s,z_e] | tag_meta_mlp | Linear(6->1024->3584) |
| Spotlight Memory | detail_feats | [B, 12, 3584] |
| Memory Injection Layer | EDRInjectionModule (PIM) | x4 layers |
| Cross-Attn (Panoramic path) | gist_cross_attn (MultiHeadAttention) | 8 heads, d_inner=512 |
| Q-K Scorer (Spotlight Router) | detail_router.q_proj, detail_router.k_proj | Linear(3584->256) |
| Spotlight Router | DetailRouter | top_k=1, d_router=256 |
| Adaptive Gate | alpha_gist, alpha_detail | scalar, init=0.05 |
| Routing Confidence | conf = 1 - entropy/log(12) | [B, L, 1] |
| Autoregressive Decoder | Qwen2.5-7B transformer (28 layers, frozen) | 7.6B params |
| SVG Detokenizer | lm_head -> SVG token decoding | - |

---

## 7. Training Configuration

| Item | Value |
|------|-------|
| Backbone | Qwen2.5-7B VLM (frozen, 7,615M params) |
| Trainable Params | ~285M (GME ~102M + CDM ~82M + PIMs x4 ~100M) |
| Trainable Ratio | **3.7%** of total ~7,900M params |
| GPUs | 8x A100 80GB |
| Batch Size | 4 per GPU |
| Gradient Accumulation | 4 |
| Effective Batch Size | 4 x 4 x 8 = **128** |
| Learning Rate | 5e-4 |
| Warmup Steps | 200 |
| Weight Decay | 0.01 |
| Max Grad Norm | 1.0 |
| Mixed Precision | bf16 |
| Optimizer | AdamW (via DeepSpeed ZeRO-2) |
| Training Data | 250K SVG samples (25 parquet files from MMSVG-Illustration) |
| Validation | Separate val split, eval every 500 steps |
| Gate Init | alpha = 0.05, tanh(0.05) approx 0.05 (cold start) |

**Memory Mode**: gme_cdm_edr (GME + Groupwise CDM + EDR routing)

**Key Flags**:
- cdm_disable_gist = true (CDM 内部不使用 gist cross-attention)
- cdm_detail_source = "part" (CDM 输入来自 group 子图特征)
- cdm_layout = "groupwise" (每个 group 独立过共享 CDM)
- edr_top_k = 1 (路由时每个 token 只选 1 个 slot)
- edr_disable_conf = false (使用 entropy-based confidence)
- pim_layer_indices = [24,25,26,27] (注入最后 4 层)

---

## 8. Benchmark Results (MMSVGBench)

### 8.1 Main Results (Full Test Set, 1000 samples, 5 candidates)

| Method | FID (min, lower=better) | FID (all, lower=better) | CLIP-T (trim, higher=better) | Aesthetic (trim, higher=better) | HPS (trim, higher=better) |
|--------|-----------|-----------|----------------|-------------------|-------------|
| OmniSVG (baseline) | 71.14 | 60.65 | 0.2704 | 4.473 | 0.2431 |
| **Ours (Fetch-Frame-Focus)** | **70.01** | **59.12** | **0.2727** | **0.4.497** | **0.2449** |
| Delta | -1.13 (-1.6%) | -1.53 (-2.5%) | +0.0023 (+0.9%) | +0.024 (+0.5%) | +0.0018 (+0.7%) |

- Success rate: Ours 99.8% vs OmniSVG 99.1%
- 所有指标上 Ours 均优于 baseline

### 8.2 MMSVGBench 300-sample Subset (Step 5000 checkpoint)

**Icon split (150 samples)**:

| Method | FID (min) | CLIP-T (trim) | Aesthetic (trim) | HPS (trim) |
|--------|-----------|---------------|-----------------|------------|
| OmniSVG | 146.51 | 0.2672 | 4.569 | 0.2402 |
| Ours (step5000) | 148.34 | 0.2690 | 4.615 | 0.2389 |

**Illustration split (150 samples)**:

| Method | FID (min) | CLIP-T (trim) | Aesthetic (trim) | HPS (trim) |
|--------|-----------|---------------|-----------------|------------|
| OmniSVG | 153.92 | 0.2141 | 4.561 | 0.2231 |
| Ours (step5000) | 150.70 | 0.2145 | 4.542 | 0.2254 |

---

## 9. Four Contributions

1. **Problem Identification**: We identify a central bottleneck of text-to-SVG generation as the lack of visual grounding during decoding. As SVGs grow in complexity, text-only decoders suffer from structural drift, missing parts, and repetition collapse, suggesting that text-to-SVG should be viewed as a problem of grounded visual-program decoding.

2. **Reference-Grounded Decoding Framework**: We introduce a retrieval-augmented framework that transforms retrieved reference images into two complementary forms of visual evidence: a global scene context (Panoramic Memory, via Q-Former compression) that stabilizes overall composition, and addressable detail slots (Spotlight Memory, via group-wise encoding + spatial tagging) with a token-conditioned router that allows each generation step to access only the most relevant visual detail.

3. **Three Design Principles**: Through systematic ablations, we derive three design principles for effective visual grounding in SVG generation: (i) structurally decomposed rather than flattened, (ii) selectively accessed rather than uniformly broadcast, and (iii) injected in late decoder layers rather than early ones.

4. **Lightweight and Architecture-Preserving**: The backbone VLM remains entirely frozen, and visual grounding is introduced through a small set of late-layer modules, adding merely 285M trainable parameters (3.7% of the 7.6B frozen backbone). The approach is architecture-preserving and readily transferable to other autoregressive generation backbones.

---

## 10. Abstract (Refined Draft)

Autoregressive text to SVG models can produce compact, resolution-independent vector graphics, yet they decode long token sequences with text conditioning alone. As structural complexity grows, this text-only decoding leads to part omission, layout drift, and repetitive strokes — symptoms we trace to the absence of visual grounding during generation. We recast text to SVG as a problem of grounded visual program synthesis and introduce Fetch, Frame, and Focus, a retrieval-augmented framework that supplies the decoder with compositionally structured visual evidence at every layer. Given a text prompt, the Fetch stage retrieves the top-K reference SVGs from a large corpus via cross-modal similarity search and decomposes each into semantically coherent part groups. The retrieved references are then encoded into two complementary memory forms: a Panoramic Memory (Frame) that captures global scene context through Q-Former compression, and a bank of spatially tagged Spotlight Memory slots (Focus), from which a lightweight dot-product router selects the most relevant entries conditioned on the current decoder hidden state. Through systematic ablation, we distill three design principles for effective visual grounding in SVG generation: reference information should be structurally decomposed rather than flattened, selectively accessed rather than uniformly broadcast, and injected in late decoder layers rather than early ones. The entire grounding module adds only 285M trainable parameters (3.7% of the frozen 7.6B backbone), requires no multimodal retraining of the base VLM, and is architecture-preserving, making it readily transferable to other autoregressive generation backbones. Experiments on MMSVGBench show that our method improves FID, aesthetic quality, and human preference scores over the ungrounded baseline while remaining competitive with or surpassing existing text to SVG methods across both icon and illustration domains.

---

## 11. Ablation Study Dimensions (for paper)

1. **Memory mode**: gme_only vs gme_cdm vs gme_cdm_edr
2. **CDM layout**: global vs groupwise
3. **CDM detail source**: ref (full image tokens) vs part (group sub-image tokens)
4. **CDM disable gist**: with/without gist cross-attn in CDM
5. **Spatial tags**: with/without group_id, with/without tag_meta
6. **EDR top-k**: k=1 vs k=2 vs no routing (uniform)
7. **EDR confidence**: with/without conf modulation
8. **PIM layer positions**: last4 (24-27) vs every4 (3,7,11,...,27) vs first4
9. **Retrieval shuffle**: correct refs vs random refs
10. **GME/CDM shuffle**: independently shuffle GME/CDM inputs

---

## 12. Related Work Categories

1. **Text-to-SVG Generation**: SVG-VAE, DeepSVG, Im2Vec, DiffSketcher, Chat2SVG, InternSVG, IntroSVG, SVGen, SGP-RL, OmniSVG
2. **Retrieval-Augmented Generation**: RAG for LLMs, RETRO, kNN-LM
3. **Visual Memory / Grounding**: BLIP-2 Q-Former, Flamingo, Visual Token Merging
4. **Mixture of Experts / Sparse Routing**: MoE, Switch Transformer, Slot Attention, Top-k routing
5. **Parameter-Efficient Adaptation**: LoRA, Adapter, Prefix Tuning

---

## 13. Method Section Naming (Final)

- 3.1 Overall Framework
- 3.2 Fetch: Compositional Reference Retrieval
- 3.3 Frame and Focus: Panoramic-Spotlight Memory Injection
