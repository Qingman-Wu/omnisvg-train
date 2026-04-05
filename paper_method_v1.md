% ============================================================================
% Abstract
% ============================================================================

\begin{abstract}
Scalable Vector Graphics (SVGs) represent visual content as sequences of text-based drawing commands and can therefore be viewed as visual programs. Recent autoregressive text-to-SVG models decode SVG tokens directly, but rely on text alone and therefore receive no explicit visual grounding during generation. The prompt specifies semantic content, yet provides limited guidance for organizing the resulting SVG program.

We present \emph{Fetch, Frame, and Focus} (\textbf{F$^{3}$SVG}), a retrieval-augmented framework for decoder-side visual grounding in text-to-SVG generation. Given a prompt, \textbf{Fetch} retrieves relevant reference SVGs via CLIP-based text similarity search, decomposes each reference into ordered part groups, and derives scene-level and group-level visual tokens. \textbf{Frame} compresses full-reference evidence into a \emph{Panoramic Memory} that provides persistent compositional guidance. \textbf{Focus} encodes group-level evidence into \emph{Spotlight Memory} slots that preserve part-specific structure and are accessed through a lightweight token-conditioned router. Both memories are injected only into the late layers of a frozen autoregressive decoder, preserving single-pass decoding and the backbone architecture.

Systematic ablations identify three principles for retrieval-grounded SVG generation: reference evidence should remain structurally decomposed rather than flattened, part-level evidence should be accessed selectively rather than broadcast uniformly, and visual grounding is most effective when injected into late decoder layers. On MMSVGBench, F$^{3}$SVG consistently improves upon OmniSVG across quality metrics while adding only 257M trainable parameters (3.3\% of a frozen 7.6B backbone), and remains competitive with frontier general-purpose models. Together, these results establish retrieval-grounded memory injection as a lightweight, architecture-preserving route to stronger autoregressive SVG generation.
\end{abstract}


% ============================================================================
% Section 2: Related Work
% ============================================================================

\section{Related Work}

\paragraph{SVG Generation.}
Research on SVG generation spans several paradigms. Earlier work such as DeepSVG studied latent and hierarchical representations for vector graphics, while later text-conditioned methods moved toward direct SVG program generation with autoregressive transformers or large language models, including IconShop, StarVector, SVGen, InternSVG, IntroSVG, and OmniSVG~\cite{DeepSVG, Iconshop, StarVector, SVGen, InternSVG, IntroSVG, OmniSVG}. A separate line relies on optimization over rendered outputs, using differentiable rendering, diffusion priors, or reward-based feedback, as in VectorFusion, SVGDreamer, Chat2SVG, and the more recent RL-based SGP-RL~\cite{VectorFusion, SVGDreamer, Chat2SVG, SGP-RL}. Recent multimodal SVG efforts such as UniSVG and OmniSVG broaden the scope of SVG generation and understanding to more unified multimodal settings~\cite{UniSVG, OmniSVG}, but in the text-to-SVG setting they still do not provide explicit external visual evidence during decoding. DuetSVG moves closer to decoder-time visual guidance by jointly generating image and SVG tokens and using the model's own visual predictions as internal guidance~\cite{DuetSVG}. In contrast, our method grounds a frozen autoregressive decoder with explicit, externally retrieved, and structurally organized visual evidence while preserving single-pass decoding.

\paragraph{Retrieval-Augmented and Memory-Grounded Generation.}
Retrieval-augmented generation improves language and multimodal models by conditioning generation on external non-parametric memory rather than parametric weights alone~\cite{RAG, RETRO}. However, prior retrieval-based systems typically retrieve textual knowledge or global context and fuse it into a single conditioning stream, often at the input level. In parallel, query-based visual interfaces such as the Q-Former in BLIP-2 and the Perceiver Resampler in Flamingo compress high-dimensional visual features into compact token sets for frozen language models~\cite{BLIP-2, Flamingo}. Our work is related to both lines but differs in setting and mechanism. SVG generation requires evidence that is visual rather than textual, structured rather than flat, and useful at multiple granularities over a long decoding horizon. We therefore turn retrieved SVG references into two complementary memories: a scene-level memory that provides persistent compositional guidance and a part-level memory that preserves decomposed local structure and is accessed selectively during decoding. This makes our approach closer to retrieval-conditioned evidence routing for structured visual program generation than to standard RAG or generic multimodal prefix conditioning.


% ============================================================================
% Section 3: Method
% ============================================================================

\section{Method}
\subsection{Overview}
\label{sec:overview}

Given a text prompt $t$, our goal is to autoregressively generate an SVG token sequence $\mathbf{s} = (s_1, s_2, \ldots, s_L)$ that faithfully realizes the described visual content. We build upon OmniSVG~\cite{OmniSVG}, a pretrained autoregressive SVG generator based on Qwen2.5-VL-7B~\cite{Qwen2.5-VL} with 28 transformer layers and hidden dimension $d{=}3584$. The backbone remains \emph{entirely frozen} throughout training; only the newly introduced memory modules are trainable.

Figure~\ref{fig:framework} illustrates the overall pipeline. Given a text prompt, the \textbf{Fetch} stage (Sec.~\ref{sec:fetch}) retrieves the top-$K{=}3$ reference SVGs via CLIP-based text similarity search and decomposes each into $G{=}4$ part groups by partitioning paths along drawing order, producing two sets of visual tokens: \emph{Scene Tokens} from full reference images and \emph{Detail Tokens} from individual group sub-images. The \textbf{Frame} stage (Sec.~\ref{sec:frame}) compresses the Scene Tokens into a 32-token \emph{Panoramic Memory} $\mathbf{M}_{\mathrm{pan}} \in \mathbb{R}^{32 \times d}$ via a Q-Former encoder. The \textbf{Focus} stage (Sec.~\ref{sec:focus}) independently encodes each of the $K{\times}G{=}12$ groups through a shared-weight group-wise Q-Former, producing a bank of spatially tagged \emph{Spotlight Memory} slots $\mathbf{M}_{\mathrm{spot}} \in \mathbb{R}^{12 \times d}$, from which a dot-product router dynamically selects the most relevant entry conditioned on the decoder's hidden state. Both memories are injected into the frozen decoder's last four layers (layers 25--28) through dual-path memory injection modules with adaptive scalar gating (Sec.~\ref{sec:injection}).

The panoramic memory acts as a persistent compositional scaffold broadcast to all decoding tokens, while the spotlight memory provides on-demand access to fine-grained structural detail. The entire framework introduces approximately 257M trainable parameters---about 3.3\% of the frozen 7.6B backbone.


\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figure/framework}
  \caption{Overview of \textbf{F$^3$SVG} (Fetch, Frame, and Focus). Given a text prompt, the \textbf{Fetch} stage retrieves and decomposes reference SVGs into part groups. The \textbf{Frame} stage compresses full reference images into a Panoramic Memory via a Q-Former encoder. The \textbf{Focus} stage encodes each part group independently into Spotlight Memory slots, from which a dot-product router selects the most relevant entry. Both memories are injected into the frozen decoder's last four layers through adaptive dual-path modules.}
  \Description{Architecture diagram of the F3SVG framework showing the three-stage pipeline.}
  \label{fig:framework}
\end{figure*}




\subsection{Fetch: Retrieved Visual Evidence}
\label{sec:fetch}


\paragraph{Offline reference retrieval.}
We encode all SVG descriptions in a reference corpus with a frozen CLIP-ViT-L/14 text encoder, L2-normalize the resulting embeddings, and index them with FAISS~\cite{FAISS} using inner-product search. Given a prompt $t$, we retrieve the top-$K{=}3$ reference SVGs by text-to-text cosine similarity.

During training, this retrieval is amortized offline rather than repeated inside every optimization step. Concretely, we precompute the top-3 reference IDs for each training sample once, and cache on disk both the retrieval results and the corresponding visual features required by later stages. The dataloader therefore reads cached reference IDs, full-reference features, and group features directly, instead of issuing a new nearest-neighbor query during training. At inference time, the same retrieval rule is applied once before decoding begins.

\paragraph{Ordered group decomposition.}
An SVG is an ordered sequence of \texttt{<path>} elements. We assign each path a complexity score that combines a type-weighted count of its drawing commands with the spatial extent of its bounding box. The path sequence is then partitioned into $G{=}4$ contiguous groups by a single front-to-back scan: we accumulate path complexities and place a split whenever the running sum reaches the per-group target, defined as the remaining total complexity divided by the number of groups still to be formed. This greedy rule yields four non-overlapping, order-preserving segments whose complexities are approximately balanced.

For each group $g_i$, we record a spatial tag
\begin{equation}
\mathbf{m}_i = [c_x,\, c_y,\, w,\, h,\, z_{\mathrm{start}},\, z_{\mathrm{end}}],
\label{eq:spatial_tag}
\end{equation}
where $(c_x, c_y, w, h)$ denote the normalized center and size of the group's bounding box, and $z_{\mathrm{start}}, z_{\mathrm{end}} \in [0,1]$ encode the group's depth range within the SVG drawing order, with $z{=}0$ corresponding to the first path drawn and $z{=}1$ to the last. With $K{=}3$ references each decomposed into $G{=}4$ groups, Fetch produces $N_s{=}12$ groups in total.


\paragraph{Dual visual evidence construction.}
From the same retrieved references, we derive two token streams using the frozen Qwen2.5-VL visual module~\cite{Qwen2.5-VL}. The scene-level view is obtained by encoding the full reference images into \emph{Scene Tokens}
\begin{equation}
\mathbf{X}_{\mathrm{scene}} \in \mathbb{R}^{K \times T \times d},
\qquad T{=}256.
\label{eq:scene_tokens}
\end{equation}
The group-level view is obtained by rendering each decomposed group separately and encoding the resulting images into \emph{Detail Tokens}
\begin{equation}
\mathbf{X}_{\mathrm{detail}} \in \mathbb{R}^{N_s \times T \times d}.
\label{eq:detail_tokens}
\end{equation}
Each group is rendered by retaining only its assigned paths while preserving the original SVG viewBox, rather than cropping or zooming to its local bounding box. This preserves the group's native scale and canvas position, while the spatial tag in Eq.~\ref{eq:spatial_tag} further specifies its absolute location. Scene Tokens capture scene-level compositional context, whereas Detail Tokens preserve group-level structural evidence tied to the ordered part groups. Because both views are computed once before decoding begins, Fetch provides external visual grounding while preserving single-pass autoregressive decoding.





% ============================================================================
\subsection{Frame: Panoramic Memory Encoding}
\label{sec:frame}

Each retrieved reference provides a complete but reference-specific view of how the described content might be structured as an SVG. Part inventory, spatial layout, and expected complexity tend to recur across the retrieved set, while reference-specific details such as exact stroke geometry naturally diverge. Frame distills the shared scene-level patterns into a compact \emph{Panoramic Memory}, leaving part-level detail to be handled selectively by Focus.

We flatten the Scene Tokens from all $K{=}3$ references into $\mathbf{X}_{\mathrm{flat}} \in \mathbb{R}^{KT \times d}$ ($KT{=}768$). These are projected to dimension $d_q{=}1024$ and jointly encoded by a 6-layer Q-Former~\cite{BLIP-2} with $N_p{=}32$ learnable queries, 8 attention heads, and $4{\times}$ feed-forward expansion:
\begin{equation}
\mathbf{M}_{\mathrm{pan}} =
\mathrm{Proj}_{\mathrm{out}}
\Bigl(
\mathrm{QFormer}\bigl(
\mathbf{Q}_{\mathrm{pan}},\,
\mathrm{Proj}_{\mathrm{in}}(\mathbf{X}_{\mathrm{flat}})
\bigr)
\Bigr)
\;\in\; \mathbb{R}^{32 \times d}.
\label{eq:panoramic}
\end{equation}
Joint encoding across references allows the queries to attend to all 768 scene tokens simultaneously, reinforcing shared compositional cues while averaging out reference-specific variation. The 24:1 compression ($768 \rightarrow 32$) further acts as an information bottleneck: layout, part inventory, and coarse spatial arrangement survive the compression; fine-grained stroke geometry does not and is instead captured by Focus (\S\ref{sec:focus}). The resulting panoramic memory is broadcast to every decoding position as persistent scene-level context.


% ============================================================================
\subsection{Focus: Spotlight Memory and Routing}
\label{sec:focus}

\subsubsection{Group-Wise Spotlight Encoding}
\label{sec:spotlight_enc}

Frame preserves what is shared across the retrieved references at the scene level, whereas Focus preserves what should not be averaged away at the part level. The panoramic memory stabilizes composition, but its very compression discards much of the fine-grained geometry needed for token-level drawing decisions. Focus is therefore introduced to retain the ordered group decomposition from Fetch and expose it as addressable part-level memory.

Each of the $N_s{=}12$ retrieved groups is encoded independently by a shared-weight 6-layer Q-Former encoder with the same 8-head, $4\times$ feed-forward configuration as Frame. Unlike the panoramic encoder, which uses 32 learnable queries to summarize scene-level context, the spotlight encoder uses a single learnable query per group, yielding exactly one slot for each retrieved group. This one-slot-per-group design is deliberate: the retrieved groups are heterogeneous, often occupying different positions in the SVG drawing order and carrying distinct local geometric signatures. Mixing them too early would blur part boundaries and collapse heterogeneous local evidence into another scene-level summary. Preserving one slot for one ordered retrieved group instead maintains a one-to-one correspondence between part-level evidence and an addressable memory slot.

For group $i$, we compute
\begin{equation}
\hat{\mathbf{v}}_i =
\mathrm{Proj}_{\mathrm{out}}
\Bigl(
\mathrm{QFormer}_{\mathrm{shared}}
\bigl(
\mathbf{q}_{\mathrm{spot}},\,
\mathrm{Proj}_{\mathrm{in}}(\mathbf{X}_{\mathrm{detail}}^{(i)})
\bigr)
\Bigr),
\label{eq:spotlight_raw}
\end{equation}
where $\mathbf{q}_{\mathrm{spot}}$ denotes the shared spotlight query. To preserve slot identity after shared encoding, each slot is further augmented with spatial identity. A learned group-ID embedding $\mathbf{e}_{\mathrm{id}}^{(i)}$ identifies group $i$ in the 12-slot bank, and a two-layer MLP maps the spatial tag $\mathbf{m}_i$ in Eq.~\ref{eq:spatial_tag} to the decoder dimension. The final spotlight slot is
\begin{equation}
\mathbf{M}_{\mathrm{spot}}^{(i)}
=
\mathrm{LN}
\bigl(
\hat{\mathbf{v}}_i
+ \mathbf{e}_{\mathrm{id}}^{(i)}
+ \mathrm{MLP}_{\mathrm{tag}}(\mathbf{m}_i)
\bigr),
\label{eq:spotlight_slot}
\end{equation}
and stacking the resulting slots yields the Spotlight Memory bank $\mathbf{M}_{\mathrm{spot}} \in \mathbb{R}^{12 \times d}$.

\subsubsection{Spotlight Router}
\label{sec:router}

Even after local evidence has been organized into slots, the decoder should not consume all slots at every step. At a given decoding position, most retrieved groups are irrelevant to the current drawing decision, and soft aggregation over all slots would re-mix the very part structure that Focus is designed to preserve. We therefore use a lightweight router that selects the single most relevant slot for each decoding position.

Let $\mathbf{h} \in \mathbb{R}^{L \times d}$ denote the decoder hidden states at an injection layer. The router first projects the hidden states and the Spotlight Memory into a shared routing space:
\begin{equation}
\mathbf{q} = W_q\,\mathrm{LN}(\mathbf{h}),
\qquad
\mathbf{k} = W_k\,\mathbf{M}_{\mathrm{spot}},
\label{eq:router_proj}
\end{equation}
where $W_q, W_k \in \mathbb{R}^{d \times d_r}$ are learned projections and $d_r{=}256$ is the routing dimension. The routing scores and routing distribution are then
\begin{equation}
\mathrm{score}_{\ell j}
=
\frac{\mathbf{q}_\ell \cdot \mathbf{k}_j}{\sqrt{d_r}},
\qquad
\mathbf{p}_\ell = \mathrm{softmax}(\mathrm{score}_{\ell :}),
\label{eq:router_score}
\end{equation}
where $\mathbf{p}_\ell$ is the routing distribution over the $N_s{=}12$ spotlight slots for decoding position $\ell$. The router performs hard top-1 selection,
\begin{equation}
i^\star_\ell = \arg\max_j \mathrm{score}_{\ell j},
\qquad
\Delta_{\mathrm{spot},\ell} = \mathbf{M}_{\mathrm{spot}}^{(i^\star_\ell)},
\label{eq:router_top1}
\end{equation}
and passes the selected slot directly to the subsequent injection path, without value projection. We further compute an entropy-based confidence
\begin{equation}
\mathrm{conf}_\ell
=
1 - \frac{H(\mathbf{p}_\ell)}{\log N_s},
\label{eq:router_conf}
\end{equation}
where $H(\cdot)$ denotes Shannon entropy. This confidence measures how decisively the current decoder state favors a single spotlight slot. When the routing distribution is sharp, the selected part-level evidence is injected with high confidence; when it is diffuse, the local pathway is attenuated in the subsequent injection stage, preventing ambiguous detail from perturbing decoding.

Focus is therefore sparse and token-conditioned by construction. Frame supplies persistent scene-level guidance, whereas Focus retrieves only the part-level slot that is most relevant to the current decoding state. In this way, part-level evidence remains structurally decomposed in memory and selectively accessed during generation.


% ============================================================================
\subsection{Dual-Path Memory Injection}
\label{sec:injection}

Having constructed a scene-level Panoramic Memory and a part-level Spotlight Memory, the remaining question is how these memories should intervene in a frozen autoregressive decoder without collapsing their distinct roles. We address this with late-layer dual-path memory injection: Panoramic Memory is injected as persistent scene-level guidance, whereas Spotlight Memory is injected through a sparse, token-conditioned pathway. Both are introduced only in the last four decoder layers, where retrieved evidence can most directly shape concrete spatial and geometric decisions during SVG generation.

Let $\tilde{\mathbf{h}}=\mathrm{LN}(\mathbf{h})$ denote the normalized hidden state at an injection layer.

The panoramic path provides persistent scene-level guidance through cross-attention:
\begin{equation}
\Delta_{\mathrm{pan}}
=
\mathrm{MHA}
\bigl(
\tilde{\mathbf{h}},\;
\mathbf{M}_{\mathrm{pan}},\;
\mathbf{M}_{\mathrm{pan}}
\bigr),
\label{eq:pan_inject}
\end{equation}
where $\mathrm{MHA}$ denotes standard multi-head cross-attention. Because every decoding position can attend to all 32 panoramic tokens, this path broadcasts a stable compositional scaffold throughout decoding.

The spotlight path provides part-level guidance through the router defined in Sec.~\ref{sec:router}:
\begin{equation}
\Delta_{\mathrm{spot}},\ \mathrm{conf}
=
\mathrm{Router}
\bigl(
\tilde{\mathbf{h}},\;
\mathbf{M}_{\mathrm{spot}}
\bigr).
\label{eq:spot_inject}
\end{equation}
Unlike the panoramic path, this route is sparse and token-conditioned. Each decoding position consults a single retrieved part slot, and the selected slot enters the injection path directly, without a separate value projection. The associated confidence determines how strongly that evidence should influence the decoder.

The two paths are fused through learnable scalar gates and added residually:
\begin{equation}
\mathbf{h}'
=
\mathbf{h}
+ \tanh(\alpha_{\mathrm{pan}})\,\Delta_{\mathrm{pan}}
+ \tanh(\alpha_{\mathrm{spot}})\,\mathrm{conf}\,\Delta_{\mathrm{spot}},
\label{eq:gated_fusion}
\end{equation}
where $\alpha_{\mathrm{pan}}$ and $\alpha_{\mathrm{spot}}$ are independent per-layer scalars initialized to $0.05$. This cold-start initialization keeps the augmented decoder close to the pretrained backbone at the start of training, while allowing it to learn how strongly scene-level and part-level evidence should intervene at each depth.

Late-layer dual-path injection therefore aligns the two memories with their intended roles. Panoramic Memory is broadcast persistently to stabilize overall composition, whereas Spotlight Memory is injected selectively and only when the decoder has a confident part-level preference.


image.png
% ============================================================================
% Section 4: Experiments
% ============================================================================

\section{Experiments}

\subsection{Experimental Setup}
\label{sec:setup}

\paragraph{Training.}
The final model is trained on 250K SVG samples from MMSVG-Illustration~\cite{OmniSVG}. Unless otherwise stated, all main-result comparisons use the canonical configuration defined in Sec.~\ref{sec:overview}. The OmniSVG/Qwen2.5-VL-7B backbone remains entirely frozen, and only the proposed memory modules are optimized. Training uses 8 A100 80\,GB GPUs with per-GPU batch size 4 and gradient accumulation 4, yielding an effective batch size of 128. We use AdamW~\cite{AdamW} with learning rate $5{\times}10^{-4}$, weight decay 0.01, 200 warmup steps, gradient clipping at 1.0, bfloat16 mixed precision, and DeepSpeed ZeRO-2~\cite{DeepSpeed}. Reference retrieval and visual feature extraction are precomputed and cached offline.

\paragraph{Evaluation.}
We evaluate on the text-to-SVG task of MMSVGBench~\cite{OmniSVG}, which comprises two sub-benchmarks: MMSVG-Icon and MMSVG-Illustration. We report Success Rate, CLIP-T~\cite{CLIP}, Aesthetic Score~\cite{LAION-Aesthetic}, HPS v2~\cite{HPSv2}, and FID~\cite{FID}. Success Rate measures the fraction of prompts yielding a valid SVG, while CLIP-T, Aesthetic, and HPS are computed from rendered SVG outputs against the input prompt. Because this benchmark does not provide a paired target image for each prompt in our evaluation setting, FID is computed against a fixed reference image set rendered from a small subset of the training SVG corpus.

\paragraph{Baselines.}
We compare against three categories of baselines: optimization-based vector generation methods, autoregressive text-to-SVG models, and frontier general-purpose LLM/VLM systems. Together, these categories span the major paradigms currently used for SVG generation: iterative optimization, task-specific autoregressive decoding, and general-purpose multimodal generation. OmniSVG serves as the primary anchor baseline, since it shares the same frozen backbone and cleanly isolates the contribution of the proposed retrieval-grounded memory modules. The complete method list is reported in Table~\ref{tab:main_results}.

\paragraph{Ablation protocol.}
Because the ablation suite spans many design choices, all ablation experiments are conducted under a reduced-scale protocol using 10K training samples and a 1K held-out test set. This protocol is used only for controlled relative comparison among architectural variants; the final full model and all main comparisons use the full-scale training setup described above.



% ============================================================================
% 4.2 Main Results
% ============================================================================

\subsection{Main Results}
\label{sec:main_results}

Table~\ref{tab:main_results} compares F$^3$SVG with 21 baselines spanning general-purpose VLMs/LLMs, optimization-based methods, and autoregressive text-to-SVG models on both MMSVG-Icon and MMSVG-Illustration. Because SVG is a textual format, all visual quality metrics (FID, CLIP-T, Aesthetic, HPS) are computed on PNG images rendered from the generated SVGs, enabling direct cross-method comparison regardless of the underlying generation paradigm.

Since OmniSVG shares the same frozen backbone and training data as F$^3$SVG, the performance gap directly measures the contribution of our memory modules. F$^3$SVG reduces FID by ${\sim}$40\% on both benchmarks (152.1\,$\to$\,90.9 on Icon; 153.9\,$\to$\,90.0 on Illustration) and improves all quality metrics by 8--23\%, through only 257M additional parameters (3.3\%) without any backbone modification.

Beyond this controlled comparison, F$^3$SVG leads all autoregressive methods on every metric and achieves the lowest FID overall on both benchmarks, surpassing even frontier general-purpose models. On Icon, our method attains the highest Aesthetic score among all methods (4.97) and nearly matches GPT-5.4 on HPS (0.2711 vs.\ 0.2717). Optimization-based methods achieve high CLIP-T via direct objective optimization but suffer severely inflated FID (SVGDreamer: 227.1, over 2.5$\times$ ours). General-purpose models produce semantically aligned outputs---Claude-4-Sonnet leads CLIP-T, Aesthetic, and HPS on Illustration---yet its FID of 180.0 is exactly twice ours, revealing distributional divergence from real SVGs. UniSVG, the only other visually conditioned method, collapses to 54\% success on Illustration, confirming that naive visual conditioning without structural decomposition fails. These results show that F$^3$SVG balances semantic quality and distributional fidelity without sacrificing single-pass efficiency, offering a parameter-efficient alternative to scaling model size.
