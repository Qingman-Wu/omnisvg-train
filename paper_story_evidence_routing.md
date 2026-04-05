# Evidence Routing Story

## Final Recommendation

- `总概念`：`Evidence Routing`
- `问题定义`：`reference-free decoding`
- `方法名`：`LateBind`
- `主标题`：`LateBind: Dual-Channel Evidence Routing for Text-to-SVG Generation`
- `副标题族`：围绕 `evidence routing`、`dual-channel`、`late-stage injection` 展开

我最终不建议把这篇论文的主线继续建立在 `grounding`、`memory`、`anchoring`、`pilot` 或 `adapter` 上。

更稳、更像顶会方法论文的写法是：

> This paper is not about building a better visual encoder; it is about routing the right external visual evidence into long-horizon SVG decoding.

---

## 1. Core Story

### 问题

现有 `text-to-SVG` 方法在解码时要么没有视觉参考，要么默认所有 token 在所有时刻看同一份视觉信息。

对于长程 SVG 解码，这会导致：

- `structural drift`
- `missing parts`
- `repetition collapse`

因此，这篇论文的核心问题不应再写成“SVG token modeling 不够好”，而应写成：

> Text-to-SVG generation suffers from `reference-free decoding`: the decoder must generate a long visual program without external visual evidence during generation.

### 核心 insight

最深的 scientific insight 不是“加了一个 memory 模块”，而是：

> Not all decoding steps need the same visual evidence.

更正式一点：

> Effective long-horizon SVG decoding requires a dual-channel evidence interface: a persistent scene channel for global composition and a selective detail channel for step-specific local structure.

也就是说，你的方法本质上是在做：

- `scene evidence broadcast`
- `detail evidence routing`
- `late evidential fusion`

而不是泛泛地“存视觉记忆”。

---

## 2. Method Name and Title Family

### Locked Method Name

`LateBind`

选择理由：

- 和你最终方案最对齐的是“最后四层注入”
- 不像产品名，也不太像工程脚本
- 比 `TwinStream` 更学术，比 `ReRoute` 更稳，比 `Conduit` 更具体
- 可以让真正的主概念留给副标题里的 `Evidence Routing`

### Primary Title

`LateBind: Dual-Channel Evidence Routing for Text-to-SVG Generation`

### Strong Alternates

1. `LateBind: Retrieval-Conditioned Evidence Routing for Text-to-SVG Generation`
2. `LateBind: Structured and Selective Visual Evidence for Text-to-SVG Generation`
3. `LateBind: Routing Retrieved Visual Evidence for Long-Horizon SVG Decoding`
4. `LateBind: Late-Stage Evidence Routing for Text-to-SVG Generation`

### Title Strategy

标题不要一次把所有贡献都塞进去。最稳的结构是：

`方法名: 核心机制 + 任务`

这里的核心机制就是：

- `Dual-Channel`
- `Evidence Routing`

而不是：

- `working memory`
- `grounding`
- `anchor-and-retrieve`

---

## 3. One-Sentence Claim

推荐的一句话 claim：

> We show that long-horizon text-to-SVG generation requires not a single visual prompt, but dual-channel external evidence: persistent scene evidence for global composition and selectively routed detail evidence for step-specific structure, injected only in the late layers of a frozen decoder.

更短一点的版本：

> Long-horizon SVG decoding works best when retrieved visual references are routed as dual-channel evidence rather than uniformly injected as a single dense condition.

---

## 4. Abstract Opening

### Version A

> Text-to-SVG generation requires decoding long visual programs, yet existing methods often perform decoding without external visual evidence or condition all decoding steps on the same visual signal. As generation becomes longer and more structurally complex, this leads to structural drift, missing parts, and repetition collapse. We argue that the core bottleneck is `reference-free decoding` rather than SVG token modeling alone. To address this problem, we propose `LateBind`, a retrieval-conditioned framework that routes external visual evidence into a frozen decoder through two complementary channels: a persistent scene channel for global composition and a selective detail channel for step-specific local structure.

### Version B

> Existing text-to-SVG decoders largely draw blind: they generate long SVG programs without access to external visual evidence during decoding. We show that effective SVG decoding requires a dual-channel evidence interface, where low-bandwidth scene evidence is broadcast to stabilize global composition while high-selectivity detail evidence is routed on demand to resolve local structure. Based on this observation, we propose `LateBind`, a lightweight retrieval-conditioned method that injects structured visual evidence only in the late layers of a frozen decoder.

---

## 5. Contributions

推荐直接写成下面这 4 条：

1. `We identify reference-free decoding as a central bottleneck of text-to-SVG generation.` As SVG outputs become longer and more compositionally complex, text-only decoders suffer from structural drift, missing parts, and repetition collapse, showing that the main difficulty lies in decoding long visual programs without external visual evidence.
2. `We propose LateBind, a dual-channel evidence routing framework for text-to-SVG generation.` Retrieved references are transformed into two complementary forms of visual evidence: persistent scene evidence for global composition and selectively addressed detail evidence for step-specific local structure.
3. `We derive three design principles for effective visual evidence injection in long-horizon SVG decoding:` structured evidence outperforms flat visual tokens, selective access outperforms uniform broadcasting, and late-layer injection outperforms early conditioning.
4. `We provide a lightweight alternative to end-to-end multimodal retraining.` LateBind introduces retrieval-conditioned evidence routing through a small set of late-layer modules on top of a frozen backbone, adding about `285.5M` trainable parameters, or `3.76%` of a `7.6B` base model, while preserving the original decoder architecture.

如果你后面确认 benchmark 提升数字，第四条最后一句可以再补：

> and delivers consistent gains on text-to-SVG benchmarks over text-only and uniformly conditioned baselines.

---

## 6. Module Naming

建议统一改成下面这套：


| Code Name                            | Paper Name                      | Figure Label        |
| ------------------------------------ | ------------------------------- | ------------------- |
| `GistMemoryEncoder`                  | `Scene Evidence Encoder`        | `Scene Evidence`    |
| `GroupwiseCDMEncoder` / `CDMEncoder` | `Detail Evidence Encoder`       | `Detail Evidence`   |
| `DetailRouter`                       | `State-Based Detail Addressing` | `Detail Addressing` |
| `EDRInjectionModule`                 | `Late Evidential Fusion`        | `Evidence Fusion`   |


说明：

- `Scene Evidence Encoder` 比 `Scene Context Encoder` 更贴主线
- `Detail Evidence Encoder` 比 `Compositional Detail Encoder` 更统一
- `State-Based Detail Addressing` 继续保留，这是目前最好的名字
- `Late Evidential Fusion` 比 `Gated Reference Fusion` 更适合 Evidence Routing 主线

如果你想保守一点，也可以把 `Late Evidential Fusion` 改成：

- `Late Evidence Fusion`

这个更自然一些，也更少术语负担。

---

## 7. Figure 1 Story

Figure 1 不要讲成“我们的 detail 分支很重要”，而要讲成：

> Existing SVG generators decode without the right visual evidence interface.

### Panel Layout

#### Left: Reference-Free Decoding

- `OmniSVG / text-only decoder`
- 没有外部视觉证据
- 随着解码变长，出现：
  - `structural drift`
  - `missing parts`
  - `repetition collapse`

图上文字建议：

- `Reference-Free Decoding`
- `Long decoding without external visual evidence`
- `Structural drift and repetition collapse`

#### Middle: Uniform Visual Conditioning Is Not Enough

这里可以用一个消融示意，不一定点名现有模型。

- 一份统一的 dense visual condition 被所有 token 共享
- 全局上可能有帮助
- 但局部步骤需要的细节证据被噪声淹没

图上文字建议：

- `Uniform Visual Conditioning`
- `All tokens read the same evidence`
- `Global signal helps, local precision remains weak`

#### Right: LateBind

- `Scene evidence` 广播给所有 token
- `Detail evidence` 按 hidden state 选择性读取
- 只在后四层融合

图上文字建议：

- `Dual-Channel Evidence Routing`
- `Broadcast scene evidence`
- `Select step-specific detail evidence`
- `Fuse in late decoder layers`

### Caption Direction

> Existing text-to-SVG decoders either generate without external visual evidence or expose all decoding steps to the same visual condition. LateBind instead routes retrieved references through two complementary channels: broadcast scene evidence for global composition and selectively addressed detail evidence for local structure, fused only in the late layers of decoding.

---

## 8. Differentiation vs. Prior Paradigms

不要再用太口语化的 `Blind / Imagine / Remember` 作为正式 taxonomy。

更适合论文正文的是：


| Paradigm                                 | Representative                 | Visual Signal During Decoding      |
| ---------------------------------------- | ------------------------------ | ---------------------------------- |
| `Text-only decoding`                     | OmniSVG, LLM4SVG               | none                               |
| `Optimization-based guidance`            | VectorFusion / related methods | iterative external optimization    |
| `Self-conditioned decoding`              | DuetSVG                        | internally generated visual tokens |
| `Retrieval-conditioned evidence routing` | `LateBind`                     | retrieved external visual evidence |


可以直接用的一段 differentiation：

> OmniSVG performs text-to-SVG generation without visual evidence during decoding. DuetSVG addresses this limitation by internalizing visual guidance through jointly generated image tokens, but requires architectural coupling and large-scale retraining. In contrast, LateBind externalizes visual guidance through retrieval-conditioned evidence routing, introducing structured and selective visual evidence into a frozen decoder with lightweight late-layer modules.

---

## 9. Reviewer-Facing Proof Points

这部分是给你写 introduction / method / experiment discussion 时对齐 reviewer 的。

### 9.1 What the Code Already Supports

- `retrieval-conditioned inference`
  - `inference/inference_hvm_text2svg_benchmark.py`
  - CLIP + FAISS 检索文本最近邻，并装载预计算 reference features
- `dual-channel evidence`
  - `hvm_modules.py`
  - `GistMemoryEncoder` 负责 scene-level compressed evidence
  - `DetailRouter` + `EDRInjectionModule` 负责 detail evidence routing
- `late-layer injection into a frozen decoder`
  - `hvm_decoder.py`
  - base model frozen
  - evidence injected with hooks into decoder layers
- `final configuration`
  - `run_train_hvm_a100_1_all.sh`
  - `memory_mode="gme_cdm_edr"`
  - `cdm_layout="groupwise"`
  - `edr_top_k=1`
  - `pim_layer_indices="24,25,26,27"`

### 9.2 Hard Numbers You Can Safely Say

- final trainable params: `285,504,008`
- base model size used in repo summary: `~7.6B`
- trainable ratio: `3.76%`
- injection layers: last `4` decoder layers
- detail routing: `top-1`
- final detail slots in the training script: `12`

### 9.3 Experiments the Paper Must Make Explicit

这几条最好在实验图表和正文里明确写出来：

1. `structured > flat`
  - raw dense visual tokens vs structured scene/detail evidence
2. `selective > uniform`
  - routed detail evidence vs broadcast-all detail conditioning
3. `late > early`
  - last four layers vs earlier or evenly spaced injection
4. `retrieval-conditioned vs self-conditioned`
  - 和 DuetSVG 的定位差异要讲清楚是方法论上的，不只是数字对比
5. `failure mode analysis`
  - 什么时候 retrieval 失效
  - 什么时候 detail routing 选错 slot
  - 什么时候文本本身已经足够，不需要额外视觉证据

### 9.4 Benchmark Gains

仓库里当前没有直接定位到现成的 benchmark 汇总文件，因此论文定稿时请补：

- `main text2svg benchmark gains`
- `per-category gains`
- `complex / long-horizon subset gains`
- `qualitative reduction in repetition collapse`

如果你后面拿到数字，建议优先强调：

- `complex prompts`
- `multi-part objects`
- `long SVG sequences`

因为这三类最能支撑 `evidence routing` 叙事。

---

## 10. Wording Guardrails

建议避免以下 headline 级词汇：

- `grounding`
  - 在 CV 里负载太重，容易被理解成 referring / localization
- `working memory`
  - 容易被批评为认知隐喻过强
- `anchor / pilot / copilot`
  - 比喻味太重，不够像正式方法名
- `adapter`
  - 你已经明确不想用，而且会把 reviewer 思路带到 PEFT 老范式里

### 更稳的用词

- `reference-free decoding`
- `retrieval-conditioned`
- `external visual evidence`
- `dual-channel evidence`
- `selective detail addressing`
- `late evidence fusion`

---

## 11. Final Package

如果现在直接往论文里落，我建议你先统一成下面这套：

- `method name`: `LateBind`
- `title`: `LateBind: Dual-Channel Evidence Routing for Text-to-SVG Generation`
- `problem`: `reference-free decoding`
- `core concept`: `evidence routing`
- `key insight`: `not all decoding steps need the same visual evidence`
- `module names`:
  - `Scene Evidence Encoder`
  - `Detail Evidence Encoder`
  - `State-Based Detail Addressing`
  - `Late Evidence Fusion`

最重要的一句话：

> This paper is not about storing more visual information; it is about routing the right external visual evidence to the right decoding steps.

