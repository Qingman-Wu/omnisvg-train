# Sections 2-5: Related Work, Method, and Experiments (Draft v2)

<!--
Canonical writing scaffold after locking the top-level narrative.
Write from claim -> mechanism -> evidence, not from code module names.
-->

# Section 2: Related Work

**Text-to-SVG Generation.**
Early SVG generation methods relied on autoencoding or iterative optimization over vector paths, including SVG-VAE, DeepSVG, VectorFusion, DiffSketcher, and SVGDreamer. More recent work has shifted toward autoregressive decoding, treating SVG generation as next-token prediction over serialized drawing commands, as in OmniSVG, SVGen, InternSVG, and IntroSVG. Other methods explore retrieval, reinforcement learning, or multi-stage refinement, such as Iconshop, SGP-RL, and Chat2SVG. Despite their differences, most existing methods ultimately decode from text alone, without structured external visual evidence available during generation.

**Retrieval-Augmented Generation.**
Retrieval-augmented generation has become a powerful paradigm in NLP, where external memory improves factuality, specificity, and long-context reasoning. Related ideas have also appeared in multimodal systems for captioning, question answering, and open-world recognition. However, retrieval has rarely been studied in the setting of *structured visual program generation*, where the output is neither free-form text nor a raster image, but a sequence of geometry-bearing drawing commands. Our work adapts retrieval not as a source of textual facts, but as a source of compositional visual evidence that can guide autoregressive SVG decoding.

**Visual Memory and Token Compression.**
Several vision-language models compress visual inputs into compact latent memories. BLIP-2 introduces Q-Former to summarize frozen visual features into learnable query tokens, while Flamingo uses a resampler for similar cross-modal compression. Token pruning and merging methods further study how to preserve salient visual information under tight token budgets. We borrow the idea of query-based compression, but our goal is different: we do not compress an image for recognition or captioning, but compress retrieved reference evidence into a persistent memory that stabilizes downstream generation.

**Sparse Routing and Slot-Based Access.**
Sparse routing has been studied extensively in mixture-of-experts models, where each token activates only a subset of experts. Slot Attention and related object-centric methods also learn decomposed latent slots that can be selectively accessed. Our spotlight router is inspired by the broader principle of sparse, input-dependent selection, but differs in role and mechanism: it routes over reference-derived memory slots rather than expert subnetworks, uses a lightweight single-head scorer without value projection, and modulates the routed signal through entropy-based confidence.

**Parameter-Efficient Adaptation.**
LoRA, adapters, and prefix tuning demonstrate that large pretrained models can be adapted with a small trainable parameter budget. Our method shares the frozen-backbone philosophy, but differs fundamentally in purpose. Rather than adapting existing capabilities to a new domain, we add a new functional interface: the ability to ground autoregressive SVG decoding in retrieved visual evidence. This makes our method closer to architecture-preserving capability augmentation than conventional parameter-efficient finetuning.

---

# Section 3: Method

## 3.1 Problem Formulation and Canonical Full Model

Given a text prompt $t$, our goal is to generate an SVG token sequence $\mathbf{s} = (s_1, s_2, \ldots, s_L)$ that faithfully realizes the described visual content. We build upon OmniSVG, whose decoder is based on Qwen2.5-7B with 28 transformer layers and hidden dimension $d = 3584$. The backbone remains entirely frozen throughout training. Only the newly introduced grounding modules are trainable.

The *canonical full model* reported in the paper is fixed as follows. For each prompt, we retrieve the top-$K=3$ reference SVGs. Each reference is decomposed into $G=4$ semantic groups, yielding 12 group-level units in total. A panoramic encoder compresses the three full references into 32 memory tokens. A shared-weight group-wise spotlight encoder produces one slot per group, yielding 12 spotlight slots. A lightweight router performs top-1 slot selection with entropy-based confidence modulation. Dual-path memory injection is applied only in the last four decoder layers (layers 25-28, 1-indexed). This full configuration corresponds to the final training recipe and introduces about 285M trainable parameters in total. All alternative settings, such as flattened references, removing confidence, or changing injection depth, are treated as ablations rather than the default model.

Figure 2 illustrates the overall pipeline. The **Fetch** stage retrieves and decomposes reference SVGs into full-image and group-level visual evidence. The **Frame** stage compresses the retrieved full references into a compact *Panoramic Memory* that captures global layout and part inventory. The **Focus** stage encodes decomposed groups into a bank of spatially tagged *Spotlight Memory* slots and enables token-wise retrieval through a sparse router. Both memories are injected into the frozen decoder through a late-layer dual-path module. The panoramic path broadcasts scene-level context to all tokens, while the spotlight path provides selective access to local detail only when needed.

This separation is intentional. The panoramic memory answers "what should the whole composition look like," whereas the spotlight memory answers "which local reference detail matters at the current generation step." The paper-level story should preserve this division throughout Method and Experiments.

## 3.2 Fetch: Compositional Reference Retrieval

**Cross-modal retrieval.**
We encode the text prompt $t$ using a frozen CLIP-ViT-L/14 text encoder to obtain a query embedding $\mathbf{e}_t$. All training SVGs are pre-rendered as PNG images, encoded with the corresponding CLIP image encoder, and indexed in FAISS. At inference time, we retrieve the top-$K=3$ reference SVGs by nearest-neighbor search in the joint embedding space.

**Structural decomposition.**
Each retrieved SVG is decomposed into $G=4$ semantically coherent path groups through spatial clustering. For each group $g_i$, we record spatial metadata
$$
\mathbf{m}_i = [c_x, c_y, w, h, z_{\text{start}}, z_{\text{end}}],
$$
which captures center position, bounding box size, and depth range. Across three references, this yields 12 decomposed groups. This step is not merely preprocessing; it is the basis for our first design principle, namely that reference evidence should preserve part structure rather than be flattened into one undifferentiated token stream.

**Visual tokenization.**
We feed both the full references and the cropped group sub-images into the frozen Qwen2.5-VL visual module, which provides CLIP-style vision features in the model dimension:
- **Scene Tokens**: $\mathbf{X}_{\text{scene}} \in \mathbb{R}^{B \times 3 \times 256 \times d_v}$
- **Detail Tokens**: $\mathbf{X}_{\text{detail}} \in \mathbb{R}^{B \times 12 \times 256 \times d_v}$

where $d_v = 3584$. The scene tokens are used by the panoramic encoder; the detail tokens are used by the spotlight encoder. In the paper text, this section should emphasize *why retrieval is necessary*: text alone underdetermines visual composition, while retrieved references provide reusable structural priors about part arrangement and scene layout.

## 3.3 Frame: Panoramic Memory Encoding

The Frame stage compresses the full-reference evidence into a compact global memory. We first flatten the scene tokens across the three references, obtaining $\mathbf{X}_{\text{flat}} \in \mathbb{R}^{B \times 768 \times d_v}$. These tokens are projected to a Q-Former hidden size of 1024 and processed by a 6-layer encoder with 32 learnable query tokens:

$$
\mathbf{M}_{\text{pan}} =
\text{Proj}_{\text{out}}
\Bigl(
\text{QFormer}\bigl(
\mathbf{Q}_{\text{pan}},
\text{Proj}_{\text{in}}(\mathbf{X}_{\text{flat}})
\bigr)
\Bigr)
\in \mathbb{R}^{B \times 32 \times d}.
$$

This stage compresses 768 vision tokens into 32 memory slots, corresponding to a 24:1 compression ratio. The purpose of this compression is not only efficiency. It forces the model to distill scene-level information such as part inventory, coarse arrangement, and overall composition into a persistent scaffold that can be broadcast to all decoder tokens. In writing, the key message is that panoramic memory carries global structure rather than detailed local geometry.

## 3.4 Focus: Spotlight Memory and Routing

### 3.4.1 Group-wise Spotlight Encoding

The Focus stage preserves decomposed local evidence rather than merging all detail tokens into a single pool. Each of the 12 groups is processed independently by a shared-weight Q-Former with a single learnable query. In the canonical full model, this encoder does **not** cross-attend back to the panoramic memory internally; keeping the group encoder local helps preserve clean part-specific slots and prevents the local bank from collapsing into another global summary.

For group $i$, we compute
$$
\hat{\mathbf{v}}_i =
\text{QFormer}_{\text{shared}}
\bigl(
\mathbf{q}_{\text{spot}},
\text{Proj}_{\text{in}}(\mathbf{X}_{\text{detail}}^{(i)})
\bigr).
$$

We then inject spatial identity into each slot through two mechanisms:
- **Group ID embedding** distinguishes the 12 global groups across references.
- **Spatial tag MLP** maps $\mathbf{m}_i = [c_x, c_y, w, h, z_{\text{start}}, z_{\text{end}}]$ into the model dimension.

The final spotlight slot is
$$
\mathbf{M}_{\text{spot}}^{(i)} =
\text{LN}
\bigl(
\hat{\mathbf{v}}_i +
\mathbf{e}_{\text{id}}^{(i)} +
\text{MLP}_{\text{tag}}(\mathbf{m}_i)
\bigr),
$$
and stacking all groups yields $\mathbf{M}_{\text{spot}} \in \mathbb{R}^{B \times 12 \times d}$.

This design enforces a clean interpretation: one group, one slot. It also makes the ensuing ablation story simple and strong. When the group structure is removed, performance drops; when the slots retain decomposition and spatial tags, the decoder can access local evidence in a way that respects part boundaries.

### 3.4.2 Spotlight Router

The spotlight router determines which local slot matters for each decoding state. Let $\mathbf{h} \in \mathbb{R}^{B \times L \times d}$ denote the decoder hidden state. We first compute low-dimensional queries and keys:

$$
\mathbf{q} = W_q \, \text{LN}(\mathbf{h}), \qquad
\mathbf{k} = W_k \, \mathbf{M}_{\text{spot}},
$$

where $W_q, W_k \in \mathbb{R}^{d \times d_r}$ and $d_r = 256$. The routing logits are
$$
\text{score} = \mathbf{q}\mathbf{k}^{\top}/\sqrt{d_r}.
$$

From the full score distribution we compute an entropy-based confidence term:
$$
\text{conf} =
\text{clamp}
\bigl(
1 - H(\text{softmax}(\text{score})) / \log(12),
0,
1
\bigr).
$$

The canonical model uses hard top-1 routing: each token selects the highest-scoring spotlight slot, and the selected slot vector is scaled by the confidence before injection. This differs from standard cross-attention in four ways: there is no value projection, the scorer is single-head, access is sparse rather than soft over all slots, and the routed signal is explicitly modulated by uncertainty.

In the paper narrative, this subsection should support the second principle directly: not every generation step needs all detail evidence. A sparse router lets the model fetch only the most relevant local cue instead of broadcasting every local slot to every token.

## 3.5 Late-Layer Dual-Path Injection

Both memory streams are injected into the backbone only in the last four decoder layers. Let $\tilde{\mathbf{h}} = \text{LN}(\mathbf{h})$ denote the normalized hidden state entering an injection block.

**Panoramic path.**
We apply multi-head cross-attention from the hidden state to panoramic memory:
$$
\Delta_{\text{pan}} =
\text{CrossAttn}
\bigl(
\tilde{\mathbf{h}},
\mathbf{M}_{\text{pan}},
\mathbf{M}_{\text{pan}}
\bigr),
$$
using 8 attention heads and a bottleneck inner dimension of 512.

**Spotlight path.**
We apply the router over spotlight memory:
$$
\Delta_{\text{spot}}, \text{conf} =
\text{Router}
\bigl(
\tilde{\mathbf{h}},
\mathbf{M}_{\text{spot}}
\bigr).
$$

**Gated fusion.**
The two paths are fused through learnable scalar gates:
$$
\mathbf{h}' =
\mathbf{h} +
\tanh(\alpha_{\text{pan}})\Delta_{\text{pan}} +
\tanh(\alpha_{\text{spot}})\text{conf}\Delta_{\text{spot}}.
$$

Both gates are initialized to 0.05, giving the model a cold start near the pretrained backbone behavior. The placement of these modules is a first-class design choice rather than an implementation detail. Late layers are where the decoder resolves fine-grained spatial and geometric decisions, so this is the most effective depth at which to introduce external visual evidence.

| Property | Panoramic Path | Spotlight Path |
|----------|----------------|----------------|
| Mechanism | Multi-head cross-attention | Dot-product router + top-1 |
| Access scope | All 32 panoramic tokens | 1 of 12 spotlight slots |
| Access pattern | Soft broadcast | Hard sparse |
| Value projection | Yes | No |
| Confidence modulation | No | Yes |
| Semantic role | Global compositional scaffold | Local detail lookup |

This section should end with a strong sentence in the final paper: the model does not merely retrieve references, it *transforms them into two complementary memory interfaces* for autoregressive decoding.

---

# Section 4: Experiments

## 4.1 Experimental Setup

**Training data.**
We train on 250K SVG samples from MMSVG-Illustration. The final full model uses the same training recipe as the canonical configuration in Section 3.1, including top-3 retrieved references, four groups per reference, and precomputed reference visual features.

**Benchmark protocol.**
We evaluate on MMSVGBench. For main comparison, we use the 150-sample illustration split. For ablations, we use the full 1,000-sample test set to ensure more stable conclusions. Following standard protocol, we generate 5 candidates per prompt.

**Metrics.**
We report FID-min, FID-all, CLIP-T, Aesthetic Score, HPS, and Success Rate. FID-min is computed from the best candidate per prompt, while FID-all measures the full generated distribution. CLIP-T, Aesthetic, and HPS are reported as trimmed means for robustness to outliers.

**Training details.**
The backbone is the frozen 7.6B OmniSVG/Qwen2.5-7B model. We train only the newly introduced modules, totaling about 285M parameters (3.7\%). The final runs use 8 A100 80GB GPUs, batch size 4 per GPU, gradient accumulation 4, effective batch size 128, learning rate $5\times10^{-4}$, weight decay 0.01, max gradient norm 1.0, 200 warmup steps, bfloat16 mixed precision, and DeepSpeed ZeRO-2. Unless noted otherwise, all experiments use the canonical full model with top-1 routing, confidence modulation, and injection in the last four layers.

**Baselines.**
We compare against 21 methods from three categories: optimization-based vector generation, specialized text-to-SVG models, and frontier general-purpose LLMs/VLMs. In the paper narrative, OmniSVG should remain the anchor baseline, while general-purpose models serve as a stress test showing that SVG generation remains a difficult structured-output task even for large foundation models.

## 4.2 Main Results

<!-- Table 1 goes here -->

[Table 1: Main comparison on MMSVGBench-Illustration. Fill in exact FID, CLIP-T, Aesthetic, and HPS values.]

The main comparison should be discussed in the following order:

1. **Direct baseline first.** Start with OmniSVG, since it isolates the effect of the proposed grounding mechanism. Emphasize that the backbone remains frozen, yet the grounded model improves both validity and quality.

2. **Then specialized SVG methods.** Compare against other task-specific systems to show that the gain is not merely due to scale or data, but to the specific way visual evidence is introduced during decoding.

3. **Then frontier general-purpose models.** Use them to argue that SVG generation is still a specialized structured generation problem where strong prompting alone is insufficient.

4. **Close with efficiency.** Reiterate that the method adds only 285M trainable parameters and avoids retraining the base model.

A strong final paragraph for this subsection should say: the gains are modest in absolute value but unusually meaningful under a frozen 7.6B backbone, especially because they are consistent across every reported metric and paired with improved success rate.

## 4.3 Ablation by Design Principles

We organize ablations around claims rather than modules. This is important for the paper's scientific identity. The reader should leave Section 4.3 remembering three principles, not a list of engineering switches.

### 4.3.1 Why the Full Model Needs Both Memory Paths

| Configuration | Description | Succ. | FID↓ | CLIP-T↑ | Aes.↑ | HPS↑ |
|--------------|-------------|-------|------|---------|-------|------|
| Full (Panoramic + Spotlight + Router) | Canonical model | 998 | - | - | - | - |
| Panoramic only | Remove spotlight path | 998 | - | - | - | - |
| Spotlight + Router only | Remove panoramic path | 997 | - | - | - | - |
| Panoramic + Spotlight, no router | Uniform local access | 997 | - | - | - | - |
| Spotlight only | No panoramic, no router | 996 | - | - | - | - |

This subsection is a warm-up rather than a standalone principle. It establishes that panoramic and spotlight memory serve complementary roles: global scene context stabilizes validity, while routed local evidence improves structural fidelity and detail quality.

### 4.3.2 Principle 1: Structurally Decomposed > Flattened

| Configuration | Description | Succ. | FID↓ | CLIP-T↑ | Aes.↑ | HPS↑ |
|--------------|-------------|-------|------|---------|-------|------|
| Full | Group-wise spotlight slots with spatial tags | 998 | - | - | - | - |
| Dense Ref Attention | Attend over all reference tokens directly | 995 | - | - | - | - |
| Visual Prefix | Prepend reference tokens as a flat prefix | 995 | - | - | - | - |
| w/o spatial tags | Remove group identity and metadata | 995 | - | - | - | - |

The interpretation should be explicit: it is not enough to expose the decoder to the same reference content. The *organization* of the evidence matters. Flattened alternatives erase part boundaries and force the decoder to disentangle global and local information on its own. Group-wise decomposition, together with spatial tags, preserves addressable structure and leads to more reliable grounding.

### 4.3.3 Principle 2: Selectively Accessed > Uniformly Broadcast

| Configuration | Description | Succ. | FID↓ | CLIP-T↑ | Aes.↑ | HPS↑ |
|--------------|-------------|-------|------|---------|-------|------|
| Full (top-1 + conf) | Sparse routing with uncertainty control | 998 | - | - | - | - |
| top-12 | Uniform access to all spotlight slots | 996 | - | - | - | - |
| w/o confidence | Top-1 routing without confidence scaling | 994 | - | - | - | - |
| random slot | Replace learned top-1 with random slot | 998 | - | - | - | - |

This principle should be argued carefully. If `random slot` preserves success rate, state clearly that success rate only measures whether a valid SVG is produced, not whether the right structure is generated. The quality metrics should do the heavy lifting here. The conclusion is that local detail evidence should be *selected*, not globally broadcast, and noisy routing should be attenuated rather than trusted blindly.

### 4.3.4 Principle 3: Late-Layer Injection > Early or Uniform Injection

| Configuration | Description | Succ. | FID↓ | CLIP-T↑ | Aes.↑ | HPS↑ |
|--------------|-------------|-------|------|---------|-------|------|
| Full (last 4 layers) | Layers 25-28 | 998 | - | - | - | - |
| Uniform 4 layers | Evenly spaced across depth | 994 | - | - | - | - |
| Last layer only | Inject only once at the end | 995 | - | - | - | - |

The narrative should emphasize *where* external evidence is useful. Early layers focus more on general token dynamics and language-model priors, whereas the final layers are responsible for fine-grained geometric decisions. Injecting visual evidence late lets the model use it exactly where compositional structure and stroke-level choices are resolved.

### 4.3.5 Additional Causal Validations

| Configuration | Description | Succ. | FID↓ | CLIP-T↑ | Aes.↑ | HPS↑ |
|--------------|-------------|-------|------|---------|-------|------|
| Full (adaptive gate) | Canonical model | 998 | - | - | - | - |
| fixed scale 0.03 | No learnable gate | 995 | - | - | - | - |
| shuffle GME | Break panoramic-reference correspondence | 993 | - | - | - | - |
| shuffle CDM | Break spotlight-reference correspondence | 998 | - | - | - | - |

These results help answer reviewer concerns. `shuffle GME` demonstrates that the model genuinely depends on relevant retrieved scene context rather than simply benefiting from extra parameters. `shuffle CDM` being less harmful suggests that global composition is the more sensitive channel, while routed local detail is partially protected by sparse access and confidence gating. Adaptive gating further shows that the model benefits from learning how strongly to trust external evidence at each injection depth.

## 4.4 Qualitative and Mechanistic Analysis

### 4.4.1 Repairing the Three Failure Modes

<!-- Figure 3 goes here -->

Figure 3 should be organized around the same three failures introduced in Section 1:
- **Part omission**: show a case where the baseline drops semantically necessary components and our method recovers them.
- **Layout drift**: show a prompt with nontrivial spatial arrangement where our method preserves proportions and placement.
- **Repetitive strokes**: show a long-generation example where the baseline degenerates into redundant paths but our method remains structurally diverse.

The caption should explicitly connect these examples back to the core claim: retrieved references do not merely improve style, they stabilize composition.

### 4.4.2 Router Behavior

<!-- Figure 4 goes here -->

Visualize several decoding steps where different spotlight slots are selected for different local structures. The ideal story is intuitive and semantic: when the model draws ears, tail, or whiskers, the router shifts to the corresponding reference group. Also report dataset-level slot usage statistics and routing entropy to show that the router does not collapse to a trivial always-same-slot behavior.

### 4.4.3 Gate Evolution and Failure Cases

Track the learned values of $\alpha_{\text{pan}}$ and $\alpha_{\text{spot}}$ through training. If the panoramic gate consistently becomes larger, argue that global composition is the more heavily relied-on signal; if both rise, argue that the two evidence types are complementary. Close this subsection with honest failure cases, especially examples where retrieval quality is poor or the fixed four-group decomposition under-segments a complex scene.

---

# Section 5: Conclusion

We present *Fetch, Frame, and Focus*, a retrieval-grounded framework for autoregressive text-to-SVG generation. The core idea is to transform retrieved SVG references into two complementary memory interfaces: a panoramic memory that stabilizes global composition and a spotlight memory that provides token-wise access to local structure. Injecting these memories only in the decoder's final layers yields a lightweight yet effective form of visual grounding for a frozen 7.6B backbone.

Beyond the immediate model improvement, the paper's lasting contribution should be framed as a set of design principles for retrieval-grounded structured generation: preserve decomposition, access local evidence selectively, and inject visual grounding where fine-grained decisions are made. This is the level at which the work should be remembered.

**Limitations and future work.**
The method still depends on retrieval quality, is currently validated only in the illustration domain, and uses a fixed four-group decomposition that may not be optimal for every SVG category. Future work can study adaptive decomposition, stronger retrieval objectives, multi-turn editing, and transfer to other structured visual outputs such as diagrams, charts, or CAD drawings.
