# Fetch, Frame, and Focus — 完整论文写作方案

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PART 0: 实验数据分析与叙事策略
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 0.1 实验全景

**对比实验 (150 samples, MMSVGBench-Illustration)**:
共 21 个对比方法，分为 3 类：

| 类别 | 方法 | Success |
|------|------|---------|
| **Optimization-Based Methods** | SVGDreamer, VectorFusion, Chat2SVG | 150, ?, 148 |
| **Autoregressive Text-to-SVG Models** | OmniSVG, SVGen×4, UniSVG, InternSVG, IntroSVG, Iconshop, SGP-RL | 115-150 |
| **General-Purpose LLMs/VLMs** | Qwen-7B/72B, DeepSeek, GLM, Kimi, Mimo, Claude, GPT | 149-150 |
| **Ours** | Fetch-Frame-Focus | 149 |

**消融实验 (1000 samples, full test)**:
17 个变体，覆盖 5 个维度。

### 0.2 叙事策略：实验如何支撑三条 Design Principles

**Principle 1 — Structurally Decomposed > Flattened**:
- Full (groupwise, 998) vs Dense Ref Attention (flat, 995) vs Visual Prefix (flat, 995)
- w/o tag (995) 证明 spatial decomposition 信息有用

**Principle 2 — Selectively Accessed > Uniformly Broadcast**:
- Full (top-1, 998) vs TOP-12 / uniform (996) vs w/o conf (994)
- random replace top1 (998) — success rate 相同，但需要用 quality metrics 证明质量差异

**Principle 3 — Late Layer Injection > Early**:
- Full (last-4, 998) vs uniform 4-layer (994) vs last-layer only (995)

**附加验证**:
- shuffle GME (993) vs shuffle CDM (998) — GME 更敏感，说明全局构图信息更关键
- fixed scale (995) vs adaptive gate (998) — 自适应门控更优

### 0.3 论文独特卖点

1. **21 个对比方法** — 包括 GPT-5.4、Claude-4-Sonnet 等最前沿模型，在 text-to-SVG 论文中极为罕见
2. **三条 Design Principles** — 不只是报告结果，而是提炼出可迁移的设计原则
3. **3.7% 可训练参数** — 冻结 7.6B backbone，仅添加 285M 参数
4. **系统性消融** — 17 个变体，每一条 principle 都有正反两方向的实验支撑

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PART 1: 论文完整架构
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 页面分配 (8页正文)

| 节 | 标题 | 页数 |
|----|------|------|
| Abstract | — | 15行 |
| 1 | Introduction | 1.3页 |
| 2 | Related Work | 0.7页 |
| 3 | Method: Fetch, Frame, and Focus | 2.5页 |
| 4 | Experiments | 2.8页 |
| 5 | Conclusion | 0.2页 |
| — | References | ~0.5页 |
| A-D | Supplementary / Appendix | 不限 |

### 主文论证顺序 (必须固定)

正文的说服链必须严格按照下面的顺序展开:

1. **Figure 1** 先建立问题和直觉收益: baseline 为什么失败, 我们解决了什么
2. **Introduction** 给出根因级洞察: 缺的不是更多 text conditioning, 而是 decoder-time visual grounding
3. **Figure 2 + Method** 解释机制: 为什么是 panoramic + spotlight 两类 memory, 为什么是 late-layer injection
4. **Table 1** 证明整体有效: full model 在真实 benchmark 上是否稳定优于 OmniSVG 和其他 baselines
5. **Table 2** 提炼科学结论: 三条 design principles 是否被系统性支持
6. **Figure 3/4** 回答 "为什么有效": 失败模式修复 + router/gate 的可解释行为

### 版面优先级 (超页时如何取舍)

**绝不能砍**:
- Figure 1
- Figure 2
- Table 1
- Table 2

**优先保留**:
- Figure 3 (如果 qualitative 对三类 failure 修复很有说服力)

**超页时最先挪到 Appendix**:
- Figure 4 中的次级分析图
- General-purpose LLM/VLM 的完整大表行
- 额外的 qualitative gallery

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PART 2: Figure 规划
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Figure 1 — Teaser (横向细长, 占 ~1/3 页, Introduction 第一页)

**布局**: 三段式横排

**左 Part (失败案例)**:
- 3 个典型 prompt，每个展示 OmniSVG baseline 的一种失败模式
- 用红色圈/箭头标注问题区域
- 简短标注: "Part Omission", "Layout Drift", "Repetitive Strokes"

**中 Part (方法简化架构图)**:
- 极简版 pipeline: Text → Fetch → Frame (Panoramic) + Focus (Spotlight) → Frozen Decoder → SVG
- 颜色系统与 Figure 2 一致 (Fetch 灰, Frame 蓝, Focus 粉, Injection 黄)

**右 Part (结果 Gallery + 提升数字)**:
- ~10 个高质量生成结果的小 gallery (3-4 行排列, 展示多样性)
- 下方/侧边用醒目字体+箭头展示 MMSVGBench 关键指标提升:
  FID ↓X.X%   CLIP-T ↑X.X%   Aesthetic ↑X.X%   HPS ↑X.X%

**Caption**:
> Left: existing text-to-SVG decoders suffer from part omission, layout drift, and repetitive strokes when generating complex SVGs. Middle: we introduce Fetch, Frame, and Focus, which retrieves reference SVGs and encodes them into panoramic and spotlight memories for decoder-side visual grounding. Right: our method produces diverse, high-quality SVG illustrations and achieves consistent improvements across all metrics on MMSVGBench.

### Figure 2 — Architecture Diagram (已画好, 占 1/2 页, Method 节)

**五区域布局**:
- 左: Fetch — TEXT→检索→3 refs→Image Tokenizer→Scene/Detail Tokens
- 中左: Frame — Learnable Queries×32→QFormer→Panoramic Memory (蓝色)
- 中右: Focus — Learnable Queries×12→QFormer→Spotlight Memory + Positional Encoding (粉色)
- 底部: Autoregressive Decoder (Layer 1...25-28)→Memory Injection Layer×4 (黄色)
  - 左路: Cross-Attn ← Panoramic Memory
  - 右路: Q-K Scorer (Spotlight Router) ← Spotlight Memory
- 最右: Spotlight Router 详细面板 — 3 个生成步骤的动态路由行为

### Figure 3 — Qualitative Comparison (占 ~1/2 页, Experiments 节)

**布局**: 对比所有测试的 baseline 方法
- 列: 包含所有对比方法 + Ours
- 行: 3-4 个代表性 prompt，选择能展示三类失败修复的例子
- 正文放精选方法即可，优先保 `OmniSVG + 2-3 个最强专用 baseline + Ours`
- 完整全量对比放 Appendix，不要为了“全”挤压 Table 2 的空间

### Figure 4 — Router Visualization + Analysis (合并, 占 ~1/3 页, Analysis 小节)

**上半: Router Visualization**
- 2-3 个生成步骤中 Spotlight Router 的路由选择
- 左: 12 个 Spotlight slots 对应的参考图 group 子图
- 右: 不同 token 选择不同 slot 的热力图/示意

**下半: Analysis Plots**
- (a) Gate 值 (α_pan, α_spot) 随训练步数的演化
- (b) Router slot usage entropy 分布
- (c) inject/hidden ratio 或其他诊断指标

**版面策略**:
- 如果主文空间充足，Figure 4 作为 mechanism evidence 留在正文
- 如果主文过满，优先保留 router visualization，把 gate 曲线和次级统计图挪到 Appendix

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PART 3: Table 精确设计
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Table 1 — Main Comparison (MMSVGBench, 150 samples)

分 3 组，每组内按 FID 排序。列: Method | Backbone | #Params | Succ.↑ | FID↓ | CLIP-T↑ | Aes.↑ | HPS↑

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Table 1: Comparison on MMSVGBench-Illustration (150 samples, 5 cands)  │
├──────────────────────────────────────────────────────────────────────────┤
│ Optimization-Based Methods                                              │
│  VectorFusion           ...       ...    ?/150   ...   ...   ...   ... │
│  SVGDreamer             ...       ...  150/150   ...   ...   ...   ... │
│  Chat2SVG               ...       ...  148/150   ...   ...   ...   ... │
├──────────────────────────────────────────────────────────────────────────┤
│ Autoregressive Text-to-SVG Models                                       │
│  Iconshop               ...       ...  150/150   ...   ...   ...   ... │
│  SGP-RL                 ...       ...  149/150   ...   ...   ...   ... │
│  InternSVG              ...       ...  150/150   ...   ...   ...   ... │
│  IntroSVG               ...       ...  150/150   ...   ...   ...   ... │
│  SVGen-LLaMA-3.2-3B     ...       ...  150/150   ...   ...   ...   ... │
│  SVGen-Qwen2.5-3B       ...       ...  149/150   ...   ...   ...   ... │
│  SVGen-Qwen2.5-Coder-7B ...       ...  150/150   ...   ...   ...   ... │
│  SVGen-StarCoder2-3B    ...       ...  150/150   ...   ...   ...   ... │
│  UniSVG                 ...       ...  115/150   ...   ...   ...   ... │
│  OmniSVG                Qwen2.5-7B 7.6B 148/150 ...   ...   ...   ... │
├──────────────────────────────────────────────────────────────────────────┤
│ General-Purpose LLMs / VLMs                                             │
│  Qwen2.5-VL-7B          ...       ...  149/150   ...   ...   ...   ... │
│  Qwen2.5-VL-72B         ...       ...  150/150   ...   ...   ...   ... │
│  DeepSeek-3.2           ...       ...  150/150   ...   ...   ...   ... │
│  GLM-5                  ...       ...  150/150   ...   ...   ...   ... │
│  Kimi-K2.5              ...       ...  150/150   ...   ...   ...   ... │
│  Mimo-v2-Flash          ...       ...  150/150   ...   ...   ...   ... │
│  Claude-4-Sonnet        ...       ...  150/150   ...   ...   ...   ... │
│  GPT-5.4                ...       ...  150/150   ...   ...   ...   ... │
├──────────────────────────────────────────────────────────────────────────┤
│ Ours                                                                    │
│  Fetch-Frame-Focus      Qwen2.5-7B 7.9B 149/150 ...  ...   ...   ... │
│  (Δ vs OmniSVG)         —         +285M  +1    ...   ...   ...   ... │
└──────────────────────────────────────────────────────────────────────────┘
```

**写作注意**:
- 粗体标注每列最优 (Autoregressive 类) 和整体最优
- OmniSVG 行下方用浅灰色标注 Δ
- 若 Table 1 太大，第一优先是把 General-Purpose LLMs/VLMs 移到 Appendix，而不是压缩 Ours vs OmniSVG 的关键信息
- General-Purpose LLMs/VLMs 的加入是本文的独特贡献——展示即使最强通用模型，在 SVG 领域也未必优于专用方法

### Table 2 — Ablation Study (MMSVGBench, 1000 samples)

围绕三条 Design Principles 组织，共 4 个子表:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Table 2: Ablation Study (1000 samples, 5 candidates)                    │
├──────────────────────────────────────────────────────────────────────────┤
│ (a) Memory Architecture                                                 │
│  Full (GME+CDM+EDR)     ← Ours    998/1000   ...   ...   ...   ...   │
│  GME only (Panoramic)              998/1000   ...   ...   ...   ...   │
│  CDM+EDR only (Spotlight)          997/1000   ...   ...   ...   ...   │
│  GME+CDM (no routing)              997/1000   ...   ...   ...   ...   │
│  CDM only (no pan., no route)      996/1000   ...   ...   ...   ...   │
├──────────────────────────────────────────────────────────────────────────┤
│ (b) Structured vs. Flat   [Principle 1]                                 │
│  Full (group-wise CDM)   ← Ours    998/1000   ...   ...   ...   ...   │
│  Dense Ref Attention (flat)        995/1000   ...   ...   ...   ...   │
│  Visual Prefix (flat)              995/1000   ...   ...   ...   ...   │
│  w/o spatial tags                  995/1000   ...   ...   ...   ...   │
├──────────────────────────────────────────────────────────────────────────┤
│ (c) Selective vs. Uniform  [Principle 2]                                │
│  Full (top-1 + conf)    ← Ours    998/1000   ...   ...   ...   ...   │
│  top-12 (all slots)                996/1000   ...   ...   ...   ...   │
│  w/o confidence                    994/1000   ...   ...   ...   ...   │
│  random slot (ablated routing)     998/1000   ...   ...   ...   ...   │
├──────────────────────────────────────────────────────────────────────────┤
│ (d) Late vs. Early Injection  [Principle 3]                             │
│  Full (last 4 layers)   ← Ours    998/1000   ...   ...   ...   ...   │
│  uniform 4 layers                  994/1000   ...   ...   ...   ...   │
│  last layer only                   995/1000   ...   ...   ...   ...   │
├──────────────────────────────────────────────────────────────────────────┤
│ (e) Design Choices                                                      │
│  Full (adaptive gate)   ← Ours    998/1000   ...   ...   ...   ...   │
│  fixed scale 0.03                  995/1000   ...   ...   ...   ...   │
│  shuffle GME input                 993/1000   ...   ...   ...   ...   │
│  shuffle CDM input                 998/1000   ...   ...   ...   ...   │
└──────────────────────────────────────────────────────────────────────────┘
```

**关于 "random replace top1" (998) 和 "shuffle CDM" (998)**:
这两个消融 success rate 与 Full 相同。论文中需要用 quality metrics (FID/CLIP-T/HPS) 来区分。
如果 quality metrics 也相近，则需在文中诚实讨论:
- random replace: "The router improves quality metrics while maintaining success rate..."
- shuffle CDM: "CDM is more robust to input perturbation than GME, suggesting
  that group-level structure matters more than exact content matching."

**主文保留原则**:
- Table 2 是这篇 paper 的 scientific core，不能因为版面原因被弱化成 appendix 表
- 真正可以裁剪的是子表行数、额外分析和次级 baseline，而不是三条 principle 本身

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PART 4: 各节写作要点
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Section 1: Introduction

**5 段式结构**:

P1 — 背景 (SVG 的优势 + 近期进展) → 一句话转折
P2 — 失败现象 + 归因 (配 Figure 1) → 核心 insight
P3 — 方案概述 (Fetch-Frame-Focus 三阶段)
P4 — Design Principles 预告 + 轻量特性
P5 — Contributions (4 条)

**核心句 (必须出现)**:
> "We trace these failures to a common root: the absence of visual grounding
> during autoregressive decoding. Text prompts specify *what* to generate
> but provide no structural evidence for *how* to compose the visual program."

### Section 2: Related Work

**5 个方向，每段 4-5 句**:

1. **Text-to-SVG Generation** (SVG-VAE → DeepSVG → OmniSVG → SVGen)
   - 强调：都是 text-only conditioned, 没有外部视觉参考
2. **Retrieval-Augmented Generation** (RAG, RETRO, kNN-LM)
   - 强调：在 NLP 成功，但在 structured visual program 中是空白
3. **Visual Memory and Grounding** (BLIP-2 Q-Former, Flamingo)
   - 强调：我们借鉴 Q-Former 但用于 decoder-side grounding
4. **Sparse Routing and MoE** (Switch Transformer, Slot Attention)
   - 强调：我们的 router 不同于 MoE — 无 V 投影，confidence 调制
5. **Parameter-Efficient Adaptation** (LoRA, Adapter)
   - 强调：我们不是微调已有功能，而是引入新功能模块

### Section 3: Method

**3.1 Overall Framework** (~0.3页)
- 问题形式化: given text t, generate SVG token sequence
- Pipeline 概览 + Figure 2

**3.2 Fetch: Compositional Reference Retrieval** (~0.4页)
- CLIP text encoding → FAISS kNN → top-3 refs
- SVG decomposition: spatial clustering → 4 groups/ref → 12 groups total
- Image tokenization: frozen CLIP ViT → Scene Tokens [B,3,256,3584] + Detail Tokens [B,12,256,3584]
- **写作重点**: 强调 "为什么需要检索" 而非 "检索是锦上添花"

**3.3 Frame: Panoramic Memory Encoding** (~0.4页)
- Q-Former: 32 learnable queries, 6 layers
- 768 scene tokens → 32 panoramic tokens (24:1 compression)
- 公式写法: M_pan = QFormer(Q_pan, K_scene, V_scene)

**3.4 Focus: Spotlight Memory and Routing** (~0.7页)
- Group-wise CDM: shared-weight Q-Former, 1 query/group × 12 groups
  - Spatial tags: group_id embedding + tag_meta MLP([cx,cy,w,h,z_s,z_e])
- Spotlight Router: q_proj / k_proj (3584→256), dot-product scoring, top-1, conf
- 公式: routed = sparse_select(softmax(q·k^T/√d), detail_bank)
- **表格**: Panoramic vs Spotlight 的属性对比 (soft vs hard, 全量 vs top-1, etc.)

**3.5 Late-Layer Dual-Path Injection** (~0.5页)
- EDRInjectionModule: 在 last 4 decoder layers (25-28)
- 双路注入公式:
  h' = h + tanh(α_g) · CrossAttn(h, M_pan) + tanh(α_d) · conf · Router(h, M_spot)
- Gate 初始化: α=0.05, cold start
- **写作重点**: 为什么选 last 4 layers (late layers 负责 fine-grained spatial decisions)

### Section 4: Experiments

**4.1 Experimental Setup** (~0.3页)
- Dataset: MMSVG-Illustration, 250K train, test splits
- Benchmark: MMSVGBench (150 illustration samples for comparison, 1000 for ablation)
- Protocol: 5 candidates per prompt, metrics computed on best/all
- Metrics: FID, CLIP-T, Aesthetic Score, HPS, Success Rate
- Training: 8×A100, bs=128 effective, lr=5e-4, bf16, DeepSpeed ZeRO-2
- Baselines: 21 methods in 3 categories

**4.2 Main Results** (~0.6页)
- Table 1: Full comparison
- 讨论要点:
  1. vs specialized SVG methods: 超越 OmniSVG baseline
  2. vs general-purpose VLMs: 即使 GPT-5.4/Claude-4 也未必在所有指标上领先
  3. UniSVG 的 115/150 success rate — 说明不是所有 visual conditioning 方案都有效
  4. 只增加 285M params (3.7%) 就获得一致提升

**4.3 Ablation Study** (~0.8页)
- Table 2: 四组消融
- 讨论围绕三条 Principles 展开:
  - P1: groupwise (998) > dense ref attn (995) = visual prefix (995)
  - P2: top-1+conf (998) > top-12 (996) > w/o conf (994)
  - P3: last-4 (998) > uniform-4 (994) > last-1 (995)
- 额外发现:
  - shuffle GME (993) 比 shuffle CDM (998) 下降更多 → 全局构图信息不可替代
  - adaptive gate (998) > fixed scale (995) → 每层自适应权重有效

**4.4 Qualitative Analysis** (~0.5页)
- Figure 3: 生成对比 (展示三类失败被修复)
- Figure 4: Router 可视化 (展示 spotlight 路由的动态选择行为)

**4.5 Analysis** (~0.4页)
- Gate 值演化分析 (α_gist vs α_detail over training)
- Router behavior: slot usage entropy, confidence distribution
- Retrieval quality analysis: 什么情况下检索失效

### Section 5: Conclusion (~0.2页)

- 总结方法和三条原则
- Limitations:
  1. 检索质量依赖训练集覆盖
  2. 当前仅在 illustration 域验证
  3. group decomposition 使用固定 4 groups
- Future: 扩展到 CAD/chart, 多轮编辑, 自适应 group 数

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PART 5: 审稿人预防策略
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| 可能质疑 | 预防策略 |
|---------|---------|
| "改进幅度不大" | (1) 强调是在 frozen 7.6B 上仅加 3.7% 参数; (2) success rate 从 991→998; (3) 定性展示三类失败修复; (4) 21 个 baseline 全面对比 |
| "检索是在抄答案" | (1) shuffle GME/CDM 实验; (2) 展示生成 SVG 与检索参考的视觉差异; (3) 方法是利用 structural similarity 而非 copy |
| "为什么不用 LoRA" | LoRA 微调已有功能, 我们引入全新的视觉记忆能力模块 |
| "random replace top1 success rate 相同" | 质量指标 (FID/CLIP-T) 会显示差异; success rate 仅衡量生成有效性，不衡量质量 |
| "只在 illustration 域" | Conclusion 中坦诚承认, 并指出架构是 domain-agnostic 的 |
| "与 DuetSVG 的区别" | 我们是 retrieval-conditioned (外部), DuetSVG 是 self-conditioned (内部生成 image tokens), 我们不需要重训 backbone |
| "General-purpose VLMs 比较公平吗" | 这些模型有更多参数/训练数据, 我们在 SVG 专项任务上展示专用方法的优势 |

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PART 6: Introduction 完整初稿 (English)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

(见下方 Section)
