# HVM-SVG 架构详解

本文档描述当前仓库实现的 HVM-SVG 架构（与代码一致），重点是模块职责、张量流和注入机制。

---

## 1. 总体设计

目标：在不改 OmniSVG base decoder 主体代码的前提下，引入层次化视觉记忆，减少复杂 SVG 生成退化。

策略：

1. 训练前离线构建 RAG 参考与视觉特征
2. 训练时由 HVM 模块构建可注入记忆
3. 通过 decoder layer hook 间隔注入 PIM

---

## 2. 模块组成

### 2.1 Base Model（冻结）

- `SketchDecoder`（OmniSVG）
- 全参数冻结，不参与训练更新
- 提供：
  - decoder layers（被 hook）
  - `embed_tokens`（生成参考文本 embedding）

### 2.2 HVM Trainable Modules

- `GistMemoryEncoder`（GME）
- `PartMemoryEncoder`（PME）
- `PrefrontalInjectionModule`（PIM）× N
- `AdaptiveGate`（每个 PIM 内）

配置入口：`hvm_modules.py` 的 `HVMConfig`。

---

## 3. 数据输入与语义

训练一个 batch 的核心输入：

- `input_ids`, `attention_mask`, `labels`
- `ref_features`: `[B, 3, 16, 16, 4, 1280]`
- `ref_best_feature`: `[B, 16, 16, 4, 1280]`
- `groups_bbox_feature`: 每样本若干 `(r1, r2, c1, c2)`
- `ref_text_ids`, `ref_text_mask`

对应来源：

- `hvm_dataset.py` + `create_hvm_collate_fn()`
- 视觉特征来自 `precompute_hvm_data.py --stage features`
- 分组 bbox 来自 `--stage groups`

---

## 4. 视觉记忆编码

### 4.1 GME（全局记忆）

输入：

- 3 张参考图拼接后的 pre-merge 特征（展平）

过程：

- QFormer（6 层）
- 32 个 learnable query
- `d_vision -> d_qformer -> d_model`

输出：

- `gist_feats [B, 32, 3584]`

### 4.2 PME（局部记忆）

输入：

- Top-1 参考图 feature map
- 分组 bbox（最多 4 组）

过程：

1. 按 bbox 裁剪 `[16,16,4,D]`
2. 展平为 token 序列
3. 每组过 QFormer（4 query）
4. 拼接后 pad 到 16 token

输出：

- `part_feats [B, 16, 3584]`
- `part_mask [B, 16]`（True 表示有效）

---

## 5. PIM 内部 5 步

单个 PIM 的 forward：

1. Part × Gist Cross-Attention
2. Part Self-Attention
3. Visual × Text Cross-Attention（使用 `text_mask`）
4. Hidden × Aligned Cross-Attention 得到 `delta`
5. AdaptiveGate 注入：`hidden + gate * delta`

张量主维度：

- `hidden_state`: `[B, L, 3584]`
- `gist_feats`: `[B, 32, 3584]`
- `part_feats`: `[B, 16, 3584]`
- `text_feats`: `[B, Nt, 3584]`
- `delta`: `[B, L, 3584]`

---

## 6. Adaptive Gate 设计

结构：

- Layer gate：`tanh(base_alpha)`（标量）
- Token gate：MLP(`cat(hidden, delta)`) + sigmoid（逐 token）

注入公式：

`output = hidden_state + tanh(alpha) * token_gate * delta`

初始化特性：

- `alpha=0`，初始注入接近 0
- token gate 最后一层置零，初始输出约 0.5

目的：训练早期尽量不扰动 base decoder，再逐步学会注入。

---

## 7. Hook 注入机制（关键）

实现位置：`hvm_decoder.py`

核心思路：

1. 在 `base_model.transformer.model.layers[layer_idx]` 注册 forward hook
2. 每个 hook 读取当前缓存：
   - `_gist_feats`
   - `_part_feats`
   - `_text_feats`
   - `_part_mask`
   - `_text_mask`
3. 调用对应 PIM，替换 layer 输出 hidden states

为什么要缓存：

- hook 回调拿不到训练 batch 的自定义参数
- 必须在 `forward` 入口先计算并缓存，再供每层 hook 读取

---

## 8. 训练时的执行顺序

单次 forward：

1. 计算并缓存 HVM memory（GME/PME/text embedding）
2. 调用 base model forward
3. 经过被 hook 的层时执行 PIM 注入
4. 输出 logits

注意：

- 不在 forward 结束后清理缓存
- 因为 gradient checkpointing 可能在 backward 触发重算 forward，hook 仍需读取这些缓存

---

## 9. 参数规模与训练边界

- Base model：冻结
- HVM 模块：可训练（约数亿级，取决于配置）
- 优化器只接收 `model.get_trainable_parameters()`

这保证了：

- 训练显存可控
- 迭代 focus 在 HVM 行为学习上

---

## 10. 离线 RAG 与在线注入的关系

离线阶段解决：

- 检索参考样本（文本相似度）
- 抽取参考图视觉特征
- 生成局部分组 bbox

在线训练阶段只做：

- 特征读取
- 记忆压缩
- 注入学习

好处：训练不需要实时跑 vision encoder 与检索，吞吐更稳定。

---

## 11. 当前实现边界（已知）

1. PIM 注入强度早期很小（gate 初始化策略决定）
2. 训练是否“看起来收敛”对 warmup 非常敏感
3. path complexity 与分组策略仍有可优化空间

---

## 12. 调优优先级建议

按收益优先：

1. warmup / lr 策略
2. optimizer param groups（尤其 gate 与 norm/bias 的 weight decay）
3. 分组策略与 complexity 公式
4. PIM 插入层间隔与 bottleneck 维度

