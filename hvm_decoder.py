"""
HVM-SVG Decoder Wrapper
========================
在 OmniSVG (SketchDecoder) 基础上，通过 forward hook 在 decoder layers 间注入 PIM。

核心机制:
  1. 冻结 base model (OmniSVG) 全部参数
  2. 训练前: GME/PME 从参考图提取 gist_feats / part_feats
  3. 运行 base model forward，hook 在指定 decoder layer 后自动调用 PIM
  4. PIM 将视觉记忆注入 hidden_state，不改变 base model 的任何代码
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from hvm_modules import (
    HVMConfig,
    GistMemoryEncoder,
    PartMemoryEncoder,
    PrefrontalInjectionModule,
    SimpleGMEInjectionModule,
    LayerGatedGMEInjectionModule,
    LayerGatedGMEPMEInjectionModule,
    DualGatedGMEPMEInjectionModule,
    HierarchicalGMEPMEInjectionModule,
    SinglePathHierarchicalInjectionModule,
    DRAInjectionModule,
    CDMEncoder,
    CDMInjectionModule,
    EDRInjectionModule,
    count_parameters,
)


class HVMSketchDecoder(nn.Module):
    """
    HVM-SVG 模型: 基于 OmniSVG + 层次化视觉记忆注入。

    Architecture:
        OmniSVG (frozen)
          ├── transformer.model.embed_tokens  → 用于 text_feats
          ├── transformer.model.layers[0..27] → decoder layers
          │     ├── [hook after layer 3]  → PIM #0
          │     ├── [hook after layer 7]  → PIM #1
          │     ├── ...
          │     └── [hook after layer 27] → PIM #6
          └── transformer.lm_head            → logits

        HVM Modules (trainable):
          ├── GME (gist_feats from 3 ref images)
          ├── PME (part_feats from top-1 ref image crop)
          └── PIMs × 7 (inject into decoder layers)
    """

    def __init__(
        self,
        base_model: nn.Module,
        hvm_config: HVMConfig,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        """
        Args:
            base_model: 已加载 OmniSVG checkpoint 的 SketchDecoder
            hvm_config: HVM 超参数
            tokenizer: 用于 tokenize 参考文本
        """
        super().__init__()
        self.hvm_config = hvm_config
        self.tokenizer = tokenizer

        # ---- Base model (frozen) ----
        self.base_model = base_model
        self._freeze_base_model()

        # ---- HVM Modules (trainable) ----
        self.gme = GistMemoryEncoder(hvm_config)
        self.cdm = None
        if hvm_config.memory_mode == "gme" and hvm_config.inject_mode == "fixed":
            self.pme = None
            self.pims = nn.ModuleList([
                SimpleGMEInjectionModule(hvm_config)
                for _ in range(hvm_config.num_pims)
            ])
        elif hvm_config.memory_mode == "gme" and hvm_config.inject_mode == "adaptive":
            self.pme = None
            self.pims = nn.ModuleList([
                LayerGatedGMEInjectionModule(hvm_config)
                for _ in range(hvm_config.num_pims)
            ])
        elif hvm_config.memory_mode == "gme_pme":
            self.pme = PartMemoryEncoder(hvm_config)
            self.pims = nn.ModuleList([
                LayerGatedGMEPMEInjectionModule(hvm_config)
                for _ in range(hvm_config.num_pims)
            ])
        elif hvm_config.memory_mode == "gme_pme_dual":
            self.pme = PartMemoryEncoder(hvm_config)
            self.pims = nn.ModuleList([
                DualGatedGMEPMEInjectionModule(hvm_config)
                for _ in range(hvm_config.num_pims)
            ])
        elif hvm_config.memory_mode == "gme_pme_hier":
            self.pme = PartMemoryEncoder(hvm_config)
            self.pims = nn.ModuleList([
                HierarchicalGMEPMEInjectionModule(hvm_config)
                for _ in range(hvm_config.num_pims)
            ])
        elif hvm_config.memory_mode == "gme_pme_single":
            self.pme = PartMemoryEncoder(hvm_config)
            self.pims = nn.ModuleList([
                SinglePathHierarchicalInjectionModule(hvm_config)
                for _ in range(hvm_config.num_pims)
            ])
        elif hvm_config.memory_mode == "gme_dra":
            self.pme = None
            self.pims = nn.ModuleList([
                DRAInjectionModule(hvm_config)
                for _ in range(hvm_config.num_pims)
            ])
        elif hvm_config.memory_mode == "gme_cdm":
            self.pme = None
            self.cdm = CDMEncoder(hvm_config)
            self.pims = nn.ModuleList([
                CDMInjectionModule(hvm_config)
                for _ in range(hvm_config.num_pims)
            ])
        elif hvm_config.memory_mode == "gme_cdm_edr":
            self.pme = None
            self.cdm = CDMEncoder(hvm_config)
            detail_layers = set(hvm_config.edr_detail_layer_indices)
            self.pims = nn.ModuleList([
                EDRInjectionModule(
                    hvm_config,
                    enable_detail=(layer_idx in detail_layers),
                )
                for layer_idx in hvm_config.pim_layer_indices
            ])
        else:
            self.pme = PartMemoryEncoder(hvm_config)
            self.pims = nn.ModuleList([
                PrefrontalInjectionModule(hvm_config)
                for _ in range(hvm_config.num_pims)
            ])

        # ---- 统一 dtype: HVM 模块与 base model 一致 (bfloat16) ----
        # HVM 参数保持 bf16 以节省显存（428M params × 2 bytes vs × 4 bytes = 省 ~3.4GB/GPU）
        # 训练精度由 DeepSpeed ZeRO-2 保证：DS 内部自动维护 float32 optimizer states
        # （master weights + momentum + variance），即使模型参数是 bf16，
        # optimizer 更新也在 float32 上进行，避免小梯度 round to zero
        base_dtype = next(self.base_model.parameters()).dtype
        self.gme = self.gme.to(dtype=base_dtype)
        if self.pme is not None:
            self.pme = self.pme.to(dtype=base_dtype)
        if self.cdm is not None:
            self.cdm = self.cdm.to(dtype=base_dtype)
        self.pims = self.pims.to(dtype=base_dtype)
        print(f"[HVM] HVM modules dtype set to {base_dtype}")

        # ---- Hook management ----
        self._hooks = []
        # 当前 forward 的记忆缓存（每次 forward 开始时设置）
        self._gist_feats = None
        self._part_feats = None
        self._text_feats = None
        self._part_mask = None
        self._text_mask = None
        self._ref_feats = None
        self._detail_feats = None

        # PIM layer index → PIM module index 的映射
        self._pim_map: Dict[int, int] = {}
        for pim_idx, layer_idx in enumerate(hvm_config.pim_layer_indices):
            self._pim_map[layer_idx] = pim_idx

        # 安装 hooks
        self._install_hooks()

        # 打印信息
        self._print_info()

    def _freeze_base_model(self):
        """冻结 base model 全部参数，并修复 gradient checkpointing"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        print(f"[HVM] Frozen base model: {sum(p.numel() for p in self.base_model.parameters()) / 1e6:.0f}M params")

        # 关键: 用 use_reentrant=False 重新启用 gradient checkpointing
        # 原因: use_reentrant=True (默认) 要求输入 tensor 必须 requires_grad
        #       但 frozen base model 的 hidden_states 初始不需要 grad。
        #       PIM hook 在 decoder layer 后注入梯度，
        #       use_reentrant=False 才能正确传播这些梯度。
        transformer = self.base_model.transformer
        if hasattr(transformer, 'gradient_checkpointing_disable'):
            transformer.gradient_checkpointing_disable()
        if hasattr(transformer, 'gradient_checkpointing_enable'):
            transformer.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("[HVM] Re-enabled gradient checkpointing with use_reentrant=False")

    def _install_hooks(self):
        """在指定 decoder layer 后注册 forward hook"""
        qwen_model = self.base_model.transformer.model  # Qwen2_5_VLModel
        
        # 兼容 Qwen2.5-VL 和普通 Qwen2
        decoder_layers = None
        if hasattr(qwen_model, "layers"):
            decoder_layers = qwen_model.layers
        elif hasattr(qwen_model, "language_model"):
            # Qwen2.5-VL 可能将 layers 放在 language_model 中
            if hasattr(qwen_model.language_model, "layers"):
                decoder_layers = qwen_model.language_model.layers
            elif hasattr(qwen_model.language_model, "model") and hasattr(qwen_model.language_model.model, "layers"):
                decoder_layers = qwen_model.language_model.model.layers
            else:
                # Fallback: search for layers in language_model
                decoder_layers = getattr(qwen_model.language_model, "layers", None)
        
        if decoder_layers is None:
            raise AttributeError(f"Could not find 'layers' in {type(qwen_model).__name__}. Available attributes: {dir(qwen_model)}")

        for layer_idx, pim_idx in self._pim_map.items():
            if layer_idx >= len(decoder_layers):
                print(f"[HVM] Warning: layer {layer_idx} out of range (total {len(decoder_layers)} layers), skipping")
                continue

            hook = decoder_layers[layer_idx].register_forward_hook(
                self._make_pim_hook(pim_idx)
            )
            self._hooks.append(hook)

        print(f"[HVM] Installed {len(self._hooks)} PIM hooks at layers {list(self._pim_map.keys())}")

    def _make_pim_hook(self, pim_idx: int):
        """
        创建 PIM forward hook。
        当 decoder layer forward 完成后，hook 会修改 hidden_states。

        decoder layer 输出格式: tuple(hidden_states, ...)
        """
        def hook_fn(module, input, output):
            # 如果记忆未设置（非 HVM forward），跳过
            if self._gist_feats is None:
                return output

            # 提取 hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # generate() with num_return_sequences>1 会扩展 batch，
            # 需要将缓存的 HVM 特征扩展到匹配的 batch size
            B = hidden_states.shape[0]
            gist_feats = self._gist_feats.expand(B, -1, -1)
            if self.hvm_config.memory_mode == "gme":
                hidden_states = self.pims[pim_idx](
                    hidden_states,
                    gist_feats,
                )
            elif self.hvm_config.memory_mode == "gme_dra":
                ref_feats = self._ref_feats.expand(B, -1, -1)
                hidden_states = self.pims[pim_idx](
                    hidden_states,
                    gist_feats,
                    ref_feats,
                )
            elif self.hvm_config.memory_mode in ("gme_cdm", "gme_cdm_edr"):
                detail_feats = self._detail_feats.expand(B, -1, -1)
                hidden_states = self.pims[pim_idx](
                    hidden_states,
                    gist_feats,
                    detail_feats,
                )
            elif self.hvm_config.memory_mode in ("gme_pme", "gme_pme_dual", "gme_pme_hier", "gme_pme_single"):
                part_feats = self._part_feats.expand(B, -1, -1)
                part_mask = self._part_mask.expand(B, -1)
                hidden_states = self.pims[pim_idx](
                    hidden_states,
                    gist_feats,
                    part_feats,
                    part_mask,
                )
            else:
                part_feats = self._part_feats.expand(B, -1, -1)
                text_feats = self._text_feats.expand(B, -1, -1)
                part_mask = self._part_mask.expand(B, -1)
                text_mask = self._text_mask.expand(B, -1) if self._text_mask is not None else None
                hidden_states = self.pims[pim_idx](
                    hidden_states,
                    gist_feats,
                    part_feats,
                    text_feats,
                    part_mask,
                    text_mask,
                )

            # 返回修改后的 output
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states

        return hook_fn

    def remove_hooks(self):
        """移除所有 hooks（清理用）"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _prepare_text_feats(
        self,
        ref_text_ids: torch.Tensor,
        ref_text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        用 base model 的 embed_tokens 查找参考文本的 raw embedding。

        Args:
            ref_text_ids: [B, N_t]  tokenized 参考文本 IDs
            ref_text_mask: [B, N_t] attention mask, 1=valid

        Returns:
            text_feats: [B, N_t, d_model]  (padding 位置的 embedding 会被 mask 掉，但不影响 PIM 的 cross-attention)
        """
        embed_tokens = self.base_model.transformer.model.embed_tokens
        with torch.no_grad():
            text_feats = embed_tokens(ref_text_ids)  # [B, N_t, d_model]
        return text_feats

    def forward(
        self,
        # Base model inputs (same as SketchDecoder)
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        # HVM-specific inputs
        ref_features: Optional[torch.Tensor] = None,
        group_features_list: Optional[List[List[torch.Tensor]]] = None,
        ref_text_ids: Optional[torch.Tensor] = None,
        ref_text_mask: Optional[torch.Tensor] = None,
        # Original model inputs (for compatibility)
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        HVM-SVG Forward Pass.

        Base model inputs:
            input_ids:      [B, L]  text prompt + SVG tokens
            attention_mask: [B, L]
            labels:         [B, L]  (-100 for text, token IDs for SVG)

        HVM inputs:
            ref_features:        [B, 3, 256, 3584]       3 张参考图 post-merge features (for GME)
            group_features_list: List[List[Tensor]]       Top-1 参考的逐 group 渲染特征 (for PME)
                                 group_features_list[b] = [feat_g0, feat_g1, ...],
                                 每个 feat_gX: [256, 3584] (post-merge, LLM-aligned)
            ref_text_ids:        [B, N_t]                 参考文本 token IDs
            ref_text_mask:       [B, N_t]                 参考文本 attention mask

        Returns:
            Same as SketchDecoder.forward (loss, logits, etc.)
        """
        device = input_ids.device

        # ================================================================
        # 1. 计算 HVM 视觉记忆 (只算一次，全程缓存给 hooks 使用)
        # ================================================================
        if ref_features is not None:
            # 确定 HVM 模块的 dtype（与 base model 一致，通常为 bfloat16）
            hvm_dtype = next(self.gme.parameters()).dtype

            # GME: 3 张参考图 → 32 个 gist tokens
            self._gist_feats = self.gme(ref_features.to(device=device, dtype=hvm_dtype))

            if self.hvm_config.memory_mode == "gme":
                self._ref_feats = None
                self._part_feats = None
                self._part_mask = None
                self._text_feats = None
                self._text_mask = None
            elif self.hvm_config.memory_mode == "gme_dra":
                # DRA: 将 [B, 3, 256, 3584] reshape 为 [B, 768, 3584] 作为原始 ref tokens
                B_ref = ref_features.shape[0]
                self._ref_feats = ref_features.to(device=device, dtype=hvm_dtype).view(B_ref, -1, ref_features.shape[-1])
                self._part_feats = None
                self._part_mask = None
                self._text_feats = None
                self._text_mask = None
            elif self.hvm_config.memory_mode in ("gme_cdm", "gme_cdm_edr"):
                # CDM / EDR: gist_feats detach 后和展平的 ref_features 一起送入 CDM 编码器
                B_ref = ref_features.shape[0]
                flat_ref = ref_features.to(device=device, dtype=hvm_dtype).view(B_ref, -1, ref_features.shape[-1])
                self._detail_feats = self.cdm(flat_ref, self._gist_feats.detach())
                self._ref_feats = None
                self._part_feats = None
                self._part_mask = None
                self._text_feats = None
                self._text_mask = None
            elif self.hvm_config.memory_mode in ("gme_pme", "gme_pme_dual", "gme_pme_hier", "gme_pme_single"):
                gfl_on_device = [
                    [gf.to(device=device, dtype=hvm_dtype) for gf in sample_gfs]
                    for sample_gfs in group_features_list
                ]
                self._part_feats, self._part_mask = self.pme(
                    gfl_on_device,
                )  # [B, 16, d_model], [B, 16]
                self._text_feats = None
                self._text_mask = None
            else:
                gfl_on_device = [
                    [gf.to(device=device, dtype=hvm_dtype) for gf in sample_gfs]
                    for sample_gfs in group_features_list
                ]
                self._part_feats, self._part_mask = self.pme(
                    gfl_on_device,
                )  # [B, 16, d_model], [B, 16]
                self._text_mask = ref_text_mask.to(device=device, dtype=torch.bool)
                self._text_feats = self._prepare_text_feats(
                    ref_text_ids.to(device),
                    self._text_mask,
                )  # [B, N_t, d_model]
        else:
            # 没有 HVM 输入，退化为普通 OmniSVG
            self._gist_feats = None
            self._ref_feats = None
            self._detail_feats = None
            self._part_feats = None
            self._text_feats = None
            self._part_mask = None
            self._text_mask = None

        # ================================================================
        # 2. 运行 base model forward (hooks 自动注入 PIM)
        # ================================================================
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **kwargs,
        )

        # 注意: 不在此处清理缓存！
        # gradient checkpointing 在 backward 时会重新计算 forward，
        # 此时 hooks 需要访问这些缓存的 memory tensors。
        # 缓存会在下一次 forward 调用时被自然覆盖，内存开销很小（~5MB/sample）。

        return outputs

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """返回所有可训练参数（只有 HVM 模块）"""
        params = []
        params.extend(self.gme.parameters())
        if self.pme is not None:
            params.extend(self.pme.parameters())
        if self.cdm is not None:
            params.extend(self.cdm.parameters())
        params.extend(self.pims.parameters())
        return params

    def get_trainable_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """返回所有可训练的 named parameters"""
        named_params = []
        for name, param in self.gme.named_parameters():
            named_params.append((f"gme.{name}", param))
        if self.pme is not None:
            for name, param in self.pme.named_parameters():
                named_params.append((f"pme.{name}", param))
        if self.cdm is not None:
            for name, param in self.cdm.named_parameters():
                named_params.append((f"cdm.{name}", param))
        for name, param in self.pims.named_parameters():
            named_params.append((f"pims.{name}", param))
        return named_params

    def save_hvm_checkpoint(self, save_path: str):
        """只保存 HVM 模块的 checkpoint"""
        state_dict = {}
        for name, param in self.gme.named_parameters():
            state_dict[f"gme.{name}"] = param.data
        if self.pme is not None:
            for name, param in self.pme.named_parameters():
                state_dict[f"pme.{name}"] = param.data
        if self.cdm is not None:
            for name, param in self.cdm.named_parameters():
                state_dict[f"cdm.{name}"] = param.data
        for name, param in self.pims.named_parameters():
            state_dict[f"pims.{name}"] = param.data
        torch.save(state_dict, save_path)
        print(f"[HVM] Saved HVM checkpoint to {save_path} ({len(state_dict)} keys)")

    def load_hvm_checkpoint(self, load_path: str):
        """加载 HVM 模块的 checkpoint"""
        state_dict = torch.load(load_path, map_location="cpu", weights_only=True)
        gme_dict = {k.replace("gme.", ""): v for k, v in state_dict.items() if k.startswith("gme.")}
        pme_dict = {k.replace("pme.", ""): v for k, v in state_dict.items() if k.startswith("pme.")}
        cdm_dict = {k.replace("cdm.", ""): v for k, v in state_dict.items() if k.startswith("cdm.")}
        pims_dict = {k.replace("pims.", ""): v for k, v in state_dict.items() if k.startswith("pims.")}

        self.gme.load_state_dict(gme_dict, strict=True)
        if self.pme is not None:
            self.pme.load_state_dict(pme_dict, strict=True)
        elif pme_dict:
            print("[HVM] Warning: checkpoint contains PME weights, but current mode disables PME. Skipping PME load.")
        if self.cdm is not None:
            self.cdm.load_state_dict(cdm_dict, strict=True)
        elif cdm_dict:
            print("[HVM] Warning: checkpoint contains CDM weights, but current mode disables CDM. Skipping CDM load.")
        self.pims.load_state_dict(pims_dict, strict=True)
        print(f"[HVM] Loaded HVM checkpoint from {load_path}")

    def _print_info(self):
        """打印模型信息摘要"""
        frozen_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())
        total_params = frozen_params + trainable_params

        print(f"\n[HVM] Model Summary:")
        print(f"  Base model (frozen): {frozen_params / 1e6:.0f}M params")
        print(f"  HVM modules (train): {trainable_params / 1e6:.1f}M params")
        print(f"    GME: {count_parameters(self.gme) / 1e6:.1f}M")
        pme_params = count_parameters(self.pme) / 1e6 if self.pme is not None else 0.0
        print(f"    PME: {pme_params:.1f}M")
        cdm_params = count_parameters(self.cdm) / 1e6 if self.cdm is not None else 0.0
        if cdm_params > 0:
            print(f"    CDM: {cdm_params:.1f}M")
        print(f"    PIMs×{self.hvm_config.num_pims}: {sum(count_parameters(p) for p in self.pims) / 1e6:.1f}M")
        print(f"  Memory mode: {self.hvm_config.memory_mode}")
        print(f"  Inject mode: {self.hvm_config.inject_mode}")
        if self.hvm_config.inject_mode == "fixed":
            print(f"  Inject scale: {self.hvm_config.inject_scale}")
        print(f"  Total: {total_params / 1e6:.0f}M params")
        print(f"  Trainable ratio: {trainable_params / total_params * 100:.1f}%")
        print(f"  PIM insertion layers: {self.hvm_config.pim_layer_indices}\n")
        if self.hvm_config.memory_mode == "gme_cdm_edr":
            print(f"  EDR detail layers: {self.hvm_config.edr_detail_layer_indices}\n")
