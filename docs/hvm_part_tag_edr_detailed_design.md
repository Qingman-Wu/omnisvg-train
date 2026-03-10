#  s5_gme_cdm_edr_parttag_topk1 详细实验方案

- 主实验：`part-tag + nozoom + GME + CDM + EDR(top1) + last4`
- 消融实验：`part-no-tag + nozoom + GME + CDM + EDR(top1) + last4`

对应脚本：

- `run_train_hvm_a100_1_1.sh`
- `run_train_hvm_a100_1_2.sh`

## 1. 一句话概括当前方案

当前方案的核心思想是：

1. 先用 `GME` 从 3 张参考图提取全局 gist memory。
2. 再把 Top-1 参考图按 SVG path 顺序切成 4 个 group。
3. 每个 group 在原始整张画布坐标系下单独渲染成图，提取出 `part features`。
4. 对这些 `part features` 加上显式结构标签：
   `group_id` 与 `[cx, cy, w, h, z_start, z_end]`。
5. 用 `CDM` 的第二个 cross-attention 从这些 tagged part tokens 中提取 16 个 detail slots。
6. 用 `EDR` 根据当前 decoder hidden state，从 16 个 detail slots 中动态选择最相关的细节，并用置信度抑制不可靠调用。
7. 在 decoder 的最后 4 层把 gist 路和 routed detail 路注入回冻结的 OmniSVG / Qwen2.5-VL 解码器。

一句更精确的话是：

> 当前方案本质上是在保留 `try1` 的 `CDM + EDR` 后半段不变的前提下，把 `CDM` 第二个 cross-attention 的 `K/V` 从旧的 `raw ref features` 换成新的 `tagged part features`。

---

## 2. 这版方案和旧版 try1 的关系

旧版 `try1` 可以理解为：

```text
3 ref images
  -> GME -> gist_feats
  -> flatten raw ref tokens -> CDM -> detail_feats
  -> EDR(router) -> routed_detail/conf
  -> inject hidden_state
```

当前主实验改成了：

```text
3 ref images
  -> GME -> gist_feats

Top-1 ref grouped parts
  -> nozoom render
  -> part_features
  -> + tag_meta + group_id
  -> CDM -> detail_feats
  -> EDR(router) -> routed_detail/conf
  -> inject hidden_state
```

真正变化的是：

- 旧版 CDM 的 detail source：`ref`
- 新版 CDM 的 detail source：`part`
- 新版 `part` 可以带显式 tag

保持不变的部分有：

- `GME`
- `CDM` 的基本结构：learnable queries + gist cross-attn + detail-source cross-attn
- `EDR` 路由逻辑
- `last4` 注入层位置
- 冻结 base model、只训练外接记忆模块的训练方式ce="part"`

---

## 3. 整体系统图

```text
                           ┌────────────────────────────┐
                           │ Frozen OmniSVG / Qwen2.5-VL│
                           │ 28 decoder layers          │
                           └─────────────┬──────────────┘
                                         │
                           hook after layers 24/25/26/27
                                         │
                      ┌──────────────────┴──────────────────┐
                      │                                     │
                 gist injection                        detail injection
                      │                                     │
                from GME gist_feats                 from EDR routed_detail
                      │                                     │
                      └──────────────────┬──────────────────┘
                                         │
                                  updated hidden_state
视觉记忆侧:
3 ref images
  -> ref_features [B, 3, 256, 3584]
  -> GME
  -> gist_feats [B, 32, 3584]
Top-1 ref grouped parts
  -> part_features [B, G, 256, 3584]
  -> (+ tag_meta [B, G, 6])
  -> (+ group_id [B, G])
  -> CDM
  -> detail_feats [B, 16, 3584]
  -> EDR(router)
  -> routed_detail [B, L, 3584], conf [B, L, 1]
```

---

## 4. 在线数据加载阶段

在线数据加载由 `hvm_dataset.py` 完成。

### 4.1 每个样本主要返回什么

当前每个样本会返回：

- `text`
- `pix_seq`
- `ref_features`
- `ref_best_group_features`
- `ref_best_group_tag_meta`
- `ref_best_group_ids`
- `ref_text`

其中最关键的几项是：

- `ref_features`
  - 3 张参考图整图特征
  - shape: `3 x [256, 3584]`
  - 供 `GME` 使用

- `ref_best_group_features`
  - Top-1 参考图的 4 个 group 特征
  - 每个 group 是 `[256, 3584]`
  - 供 `CDM(part)` 使用

- `ref_best_group_tag_meta`
  - shape: `[G, 6]`
  - 每组对应 `[cx, cy, w, h, z_start, z_end]`

- `ref_best_group_ids`
  - shape: `[G]`
  - 默认是 `0..G-1`

### 4.2 tag_meta 是如何在线构造的

`_build_group_tag_meta(ref_idx, num_groups)` 会从 `groups_train_ref.jsonl` 里读对应 reference 的 group 结构，然后计算：

```text
cx      = ((x0 + x1) / 2) / 200
cy      = ((y0 + y1) / 2) / 200
w       = (x1 - x0) / 200
h       = (y1 - y0) / 200
z_start = min(path_indices) / (num_paths - 1)
z_end   = max(path_indices) / (num_paths - 1)
```

### 4.3 group_id 是什么

```python
group_ids = torch.arange(num_groups)
```

也就是第 0 个 group 对应 id 0，第 1 个 group 对应 id 1，以此类推。

---

## 5. batch 组装阶段

`create_hvm_collate_fn()` 把每个样本组装成 batch。

对 `GME` 的输入：

```text
ref_features: [B, 3, 256, 3584]
```

对 `CDM(part)` 的输入：

```text
part_features:  [B, G, 256, 3584]
part_tag_meta:  [B, G, 6]
part_group_ids: [B, G]
part_mask:      [B, G]
```

---

## 6. CDM 模型结构：

CDM 使用 16 个 learnable queries 做 detail 提取

```text
Self-Attn(queries)
-> Cross-Attn(queries, gist_kv)
-> Cross-Attn(queries, detail_source_kv)
-> FFN
```

其中真正最关键的是第三步：

> `queries` 与 `detail_source_kv` 的 cross-attention

当前这一步在主实验中不再看 `ref features`，而是看 `tagged part features`

---

## 7. part-tag CDM 的详细流程

### 7.1 输入

在 `cdm_detail_source="part"` 时，CDM 收到：

```text
part_features:  [B, G, 256, 3584]
part_tag_meta:  [B, G, 6]
part_group_ids: [B, G]
part_mask:      [B, G]
gist_feats:     [B, 32, 3584]
```

### 7.2 第一步：构造 tag embedding

当前代码里，tag 由两部分组成：

1. `tag_meta_mlp(part_tag_meta)`
2. `group_id_embedding(part_group_ids)`

具体是：

```python
tag_emb = MLP([cx, cy, w, h, z_start, z_end])   # [B, G, 3584]
tag_emb = tag_emb + E_group_id(group_id)        # [B, G, 3584]
```

然后再对 padding group 清零：

```python
tag_emb = tag_emb * part_mask.unsqueeze(-1)
```

### 7.3 第二步：把 tag 加到 group tokens 上

每个 group 有 256 个视觉 token。同一个 group 的 256 个 token 共享同一份结构 tag：

```python
tagged_tokens = group_tokens + tag_emb.unsqueeze(2)
```

shape 变化：

```text
group_tokens: [B, G, 256, 3584]
tag_emb:      [B, G, 3584]
unsqueeze:    [B, G, 1, 3584]
broadcast:    [B, G, 256, 3584]
```

### 7.4 第三步：展平为 CDM 第二个 cross-attn 的 `K/V`

```python
flat_tokens = tagged_tokens.view(B, -1, 3584)
```

于是：

```text
[B, G, 256, 3584] -> [B, G*256, 3584]
```

如果 `G=4`，就是：

```text
[B, 1024, 3584]
```

### 7.5 第四步：构造 attention mask

`part_mask` 原来是 group 级别的：

```text
[B, G]
```

现在需要扩成 token 级别：

```python
ref_kv_mask = part_mask.unsqueeze(-1).expand(-1, -1, 256).reshape(B, -1)
```

结果是：

```text
[B, G*256]
```

语义：

- 对真实 group 的 256 个 token，都标记为有效
- 对 padding group 的 256 个 token，都标记为无效

### 7.6 第五步：投影到 QFormer 空间

CDM 的内部工作维度是 `d_qformer = 1024`，所以需要先投影：

```python
ref_kv = LayerNorm(Linear(flat_tokens))   # [B, G*256, 1024]
gist_kv = LayerNorm(Linear(gist_feats))   # [B, 32, 1024]
```

### 7.7 第六步：16 个 learnable queries 逐层提取 detail

初始化：

```text
queries: [1, 16, 1024] -> expand -> [B, 16, 1024]
```

每层做：

```text
queries
  -> self attention
  -> attend gist_kv
  -> attend ref_kv (= 当前 tagged part tokens)
  -> FFN
```

最终输出：

```text
detail_feats [B, 16, 3584]
```

这 16 个 slots 可以理解为：

- 不是原始 patch token
- 而是经过 gist-aware 筛选后的 detail memory bank

---

## 8. 可解释性链条：slot -> group -> SVG 区域

当前方案相比旧版 `ref` CDM 最大的额外优势之一，是它更容易建立下面这条解释链：

```text
hidden token
  -> EDR selected detail slot
  -> CDM slot attends to which group
  -> group corresponds to which SVG region
```

### 8.1 `hidden token -> detail slot`

由 `EDR` 的 `score [B, L, 16]` 决定。

可以看：

- 哪些 token 置信度高
- 哪个 slot 被 top1 选中
- 各层 slot usage 是否分工明确

### 8.2 `detail slot -> group`

虽然 `detail_feats` 是 16 个抽象 slots，但它们来自 `CDM` 第二个 cross-attn 对 `part tokens` 的聚合。

因为每个 group 有 256 个 token，所以如果拿到这一路 attention 权重：

```text
[B, heads, 16, G*256]
```

就可以按每 256 个 token 为一组求和，得到：

```text
[B, heads, 16, G]
```

这就能估计：

- 每个 detail slot 主要在关注哪个 group

### 8.3 `group -> SVG region`

这个映射由 `groups_train_ref.jsonl` 直接提供：

- `group_id`
- `bbox`
- `path_indices`

所以 group 最终是可以回到原始 SVG 区域上的