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




\subsection{Frame: Panoramic Memory Encoding}
\label{sec:frame}

The retrieved full-reference images provide rich scene-level evidence, but directly exposing all of their tokens to the decoder would make the global pathway unnecessarily large and overly sensitive to local visual variation. The role of Frame is therefore to distill the retrieved set into a compact memory that preserves the information that should remain stable throughout generation, including overall layout, part inventory, and coarse spatial arrangement.

We first flatten the Scene Tokens from the $K{=}3$ retrieved references into a single sequence $\mathbf{X}_{\mathrm{flat}} \in \mathbb{R}^{KT \times d}$, where $T{=}256$ and thus $KT{=}768$ in the full model. These tokens are then jointly encoded by a 6-layer Q-Former~\cite{BLIP-2} with 32 learnable queries. Joint encoding is important here: it allows the query set to aggregate evidence across the retrieved references rather than treating them as three isolated scene memories. The resulting query states form the \emph{Panoramic Memory}
\begin{equation}
\mathbf{M}_{\mathrm{pan}}
=
\mathrm{QFormer}
\bigl(
\mathbf{Q}_{\mathrm{pan}},
\mathbf{X}_{\mathrm{flat}}
\bigr)
\in \mathbb{R}^{32 \times d}.
\label{eq:panoramic}
\end{equation}
where $\mathbf{Q}_{\mathrm{pan}}$ denotes the learnable panoramic queries.

Compressing 768 scene tokens into 32 memory tokens is deliberate. Frame is not designed to preserve fine-grained part geometry or stroke-level detail; these are handled later by Focus. Instead, it produces a persistent panoramic scaffold that summarizes the global structure of the retrieved set and stabilizes the overall composition during decoding.


\subsection{Focus: Spotlight Memory and Routing}
\label{sec:focus}

\subsubsection{Group-Wise Spotlight Encoding}
\label{sec:spotlight_enc}

Frame consolidates scene-level regularities across the retrieved references, but accurate SVG generation also depends on part-specific evidence. When the decoder is drawing a particular component, it needs to access the retrieved detail associated with that component rather than another global summary. Focus is designed for this role. It preserves the ordered group decomposition established in Fetch and converts each retrieved group into a separately addressable memory slot.

Given the $N_s{=}12$ group-level Detail Token sets from Fetch, we encode each group independently with a shared-weight Q-Former using a single learnable query per group. This produces one slot for each retrieved group, forming a 12-slot \emph{Spotlight Memory} bank. Independent per-group encoding is essential here: it preserves part boundaries in memory rather than mixing all local evidence into another dense representation. We also keep the spotlight encoder local, without conditioning it on the panoramic memory, so that each slot remains anchored to its own retrieved group.

For group $i$, the raw spotlight representation is
\begin{equation}
\hat{\mathbf{v}}_i
=
\mathrm{QFormer}_{\mathrm{shared}}
\bigl(
\mathbf{q}_{\mathrm{spot}},
\mathbf{X}_{\mathrm{detail}}^{(i)}
\bigr)
\label{eq:spotlight_raw}
\end{equation}
where $\mathbf{q}_{\mathrm{spot}}$ denotes the shared spotlight query. We then inject retrieved-group identity and spatial information through a learned group-ID embedding $\mathbf{e}_{\mathrm{id}}^{(i)}$ and an MLP over the spatial tag $\mathbf{m}_i$, yielding
\begin{equation}
\mathbf{M}_{\mathrm{spot}}^{(i)}
=
\mathrm{LN}
\bigl(
\hat{\mathbf{v}}_i
+ \mathbf{e}_{\mathrm{id}}^{(i)}
+ \mathrm{MLP}_{\mathrm{tag}}(\mathbf{m}_i)
\bigr).
\label{eq:spotlight_slot}
\end{equation}
Stacking the resulting slots gives $\mathbf{M}_{\mathrm{spot}} \in \mathbb{R}^{12 \times d}$. Using a single slot per group is deliberate: each slot is forced to summarize one ordered part group, so the local evidence remains decomposed and part-aligned rather than collapsing back into another scene-level memory.


\subsubsection{Spotlight Router}
\label{sec:router}

Once the retrieved groups have been encoded into slots, the decoder still should not consume all of them at every step. When one component is being generated, most spotlight slots are irrelevant, and dense access would blur part-specific evidence. We therefore route Spotlight access from the current decoder state, so that each decoding position consults only the most relevant retrieved part.

Let $\mathbf{h} \in \mathbb{R}^{L \times d}$ denote the decoder hidden states at an injection layer. The router scores each spotlight slot against the current hidden state in a learned routing space:
\begin{equation}
\mathrm{score}
=
\frac{
W_q\,\mathrm{LN}(\mathbf{h})
\left(W_k\,\mathbf{M}_{\mathrm{spot}}\right)^{\top}
}{\sqrt{d_r}},
\qquad
\mathbf{p} = \mathrm{softmax}(\mathrm{score}),
\label{eq:router_score}
\end{equation}
where $\mathbf{p}$ is the routing distribution over the 12 spotlight slots. For each decoding position $\ell$, the router performs hard top-1 selection,
\begin{equation}
i^\star_\ell = \arg\max_j \mathrm{score}_{\ell j},
\qquad
\Delta_{\mathrm{spot},\ell} = \mathbf{M}_{\mathrm{spot}}^{(i^\star_\ell)},
\label{eq:router_top1}
\end{equation}
and computes an entropy-based routing confidence
\begin{equation}
\mathrm{conf}_\ell
=
1 - \frac{H(\mathbf{p}_\ell)}{\log N_s},
\label{eq:router_conf}
\end{equation}
which downweights ambiguous selections. The selected slot and its confidence are then passed to the injection module in Sec.~\ref{sec:injection}.

Focus is therefore sparse and token-conditioned by construction. Frame provides persistent scene-level context to all decoding positions, whereas Focus exposes only the retrieved part slot that best matches the decoder's current state. In this way, part-level evidence remains structurally decomposed in memory and selectively accessed during generation.


\subsection{Late-Layer Dual-Path Injection}
\label{sec:injection}

We inject both memories only into the last four decoder layers, namely layers 24, 25, 26, and 27 in 0-indexed notation. Let $\tilde{\mathbf{h}} = \mathrm{LN}(\mathbf{h})$ denote the normalized hidden state at an injection point.

\paragraph{Panoramic path.}
The panoramic stream is injected by standard cross-attention from the hidden state to the panoramic memory:
\begin{equation}
\Delta_{\mathrm{pan}}
=
\mathrm{MHA}
\bigl(
\tilde{\mathbf{h}},
\mathbf{M}_{\mathrm{pan}},
\mathbf{M}_{\mathrm{pan}}
\bigr),
\label{eq:pan_inject}
\end{equation}
using 8 attention heads and an inner bottleneck dimension of 512. This path softly broadcasts scene-level evidence to every decoding position.

\paragraph{Spotlight path.}
The detail stream is injected through the spotlight router:
\begin{equation}
\Delta_{\mathrm{spot}},
\mathrm{conf}
=
\mathrm{Router}
\bigl(
\tilde{\mathbf{h}},
\mathbf{M}_{\mathrm{spot}}
\bigr).
\label{eq:spot_inject}
\end{equation}
Unlike the panoramic path, this route is sparse, token-specific, and confidence-modulated.

\paragraph{Adaptive residual fusion.}
The two paths are fused through learnable scalar gates and added residually to the frozen decoder:
\begin{equation}
\mathbf{h}'
=
\mathbf{h}
+ \tanh(\alpha_{\mathrm{pan}})\,\Delta_{\mathrm{pan}}
+ \tanh(\alpha_{\mathrm{spot}})\,\mathrm{conf}\,\Delta_{\mathrm{spot}}.
\label{eq:gated_fusion}
\end{equation}
The gates $\alpha_{\mathrm{pan}}$ and $\alpha_{\mathrm{spot}}$ are independent for each injection layer and are initialized to $0.05$, so training starts close to the original pretrained backbone behavior and only gradually increases the strength of external grounding.

\begin{table}[t]
\centering
\caption{Comparison of the two injection paths.}
\label{tab:dual_path}
\small
\begin{tabular}{lcc}
\hline
 & \textbf{Panoramic} & \textbf{Spotlight} \\
\hline
Mechanism & Cross-attention & Router + top-1 \\
Scope & 32 memory tokens & Up to 12 slots \\
Access pattern & Soft broadcast & Hard sparse access \\
Value projection & Yes & No \\
Confidence modulation & No & Yes \\
Primary role & Scene scaffold & Detail lookup \\
\hline
\end{tabular}
\end{table}

This late-layer placement is a deliberate architectural decision rather than a minor implementation choice. Earlier decoder layers mainly preserve token-level language-model dynamics, whereas the final layers are where fine-grained spatial and geometric decisions become most actionable by external visual evidence. Panoramic memory therefore provides a persistent compositional scaffold, while spotlight memory supplies selectively accessed local structure exactly where the decoder is making the most consequential drawing decisions.
