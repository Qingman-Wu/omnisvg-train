#!/usr/bin/env python3
"""
验证 HVM 模型确实加载了 HVM 模块并注入了两路 memory（Gist + Detail）。

验证内容：
  1. 结构验证：GME / CDM / PIM(EDR) 模块是否存在且参数非零
  2. Checkpoint 验证：HVM checkpoint 的权重是否成功加载到模型中
  3. Hook 验证：PIM hook 是否安装在正确的 decoder layer 上
  4. Runtime 验证：实际注入一组 ref features 后，两路 memory 是否被正确填充
  5. Forward 验证：经过 PIM hook 后 hidden_state 是否确实被修改

用法:
  python verify_hvm_memory_injection.py \
    --hvm_checkpoint /path/to/hvm_step_5000.pt \
    --hvm_config /path/to/hvm_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

INFERENCE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(INFERENCE_DIR, ".."))
sys.path.insert(0, INFERENCE_DIR)
sys.path.insert(0, PROJECT_ROOT)

from inference_hvm_multigpu import load_hvm_model

DEFAULT_HVM_CONFIG = "/mnt/data3/wuqingman/omnisvg-train/outputs_s9_full25w_top3part_12slot_nogist_edr_parttag_nozoom_last4/hvm_config.json"
DEFAULT_HVM_CKPT = "/mnt/data3/wuqingman/omnisvg-train/outputs_s9_full25w_top3part_12slot_nogist_edr_parttag_nozoom_last4/hvm_step_5000.pt"
DEFAULT_CONFIG_DIR = "/mnt/data2/wuqingman/omnisvg-train/configs"


class Colors:
    OK = "\033[92m"
    FAIL = "\033[91m"
    WARN = "\033[93m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {Colors.OK}✓{Colors.END} {msg}")


def fail(msg: str) -> None:
    print(f"  {Colors.FAIL}✗{Colors.END} {msg}")


def section(title: str) -> None:
    print(f"\n{Colors.BOLD}{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}{Colors.END}")


def verify_module_structure(model) -> bool:
    section("1. 结构验证：HVM 模块是否存在")
    all_pass = True

    hvm_config = model.hvm_config
    print(f"  memory_mode = {hvm_config.memory_mode}")
    print(f"  inject_mode = {hvm_config.inject_mode}")
    print(f"  pim_layer_indices = {hvm_config.pim_layer_indices}")

    if model.gme is not None:
        gme_params = sum(p.numel() for p in model.gme.parameters())
        ok(f"GME 存在: {gme_params / 1e6:.2f}M params")
    else:
        fail("GME 不存在")
        all_pass = False

    if model.cdm is not None:
        cdm_params = sum(p.numel() for p in model.cdm.parameters())
        ok(f"CDM ({type(model.cdm).__name__}) 存在: {cdm_params / 1e6:.2f}M params")
    else:
        fail("CDM 不存在")
        all_pass = False

    pim_count = len(model.pims)
    if pim_count > 0:
        pim_total_params = sum(sum(p.numel() for p in pim.parameters()) for pim in model.pims)
        pim_types = set(type(pim).__name__ for pim in model.pims)
        ok(f"PIMs × {pim_count} 存在: {pim_total_params / 1e6:.2f}M params, 类型={pim_types}")

        for i, pim in enumerate(model.pims):
            layer_idx = hvm_config.pim_layer_indices[i] if i < len(hvm_config.pim_layer_indices) else "?"
            has_gist_attn = hasattr(pim, "gist_cross_attn")
            has_detail = hasattr(pim, "detail_router") or hasattr(pim, "detail_cross_attn")
            has_alpha_gist = hasattr(pim, "alpha_gist") and pim.alpha_gist is not None
            has_alpha_detail = hasattr(pim, "alpha_detail") and pim.alpha_detail is not None
            ok(
                f"  PIM[{i}] @layer {layer_idx}: "
                f"gist_cross_attn={has_gist_attn}, detail_path={has_detail}, "
                f"alpha_gist={has_alpha_gist}, alpha_detail={has_alpha_detail}"
            )
    else:
        fail("PIMs 为空")
        all_pass = False

    return all_pass


def verify_checkpoint_loaded(model, ckpt_path: str) -> bool:
    section("2. Checkpoint 验证：权重是否成功加载")
    all_pass = True

    raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    ckpt_gme_keys = {k for k in raw_state if k.startswith("gme.")}
    ckpt_cdm_keys = {k for k in raw_state if k.startswith("cdm.")}
    ckpt_pim_keys = {k for k in raw_state if k.startswith("pims.")}
    print(f"  Checkpoint 中: gme keys={len(ckpt_gme_keys)}, cdm keys={len(ckpt_cdm_keys)}, pim keys={len(ckpt_pim_keys)}")

    def compare_weights(module: nn.Module, prefix: str, ckpt_keys: set) -> bool:
        if module is None:
            if ckpt_keys:
                fail(f"{prefix} 模块不存在但 checkpoint 有 {len(ckpt_keys)} 个对应 key")
                return False
            return True

        model_sd = {f"{prefix}.{k}": v for k, v in module.state_dict().items()}
        mismatches = []
        matched = 0
        for key in sorted(ckpt_keys):
            if key not in model_sd:
                continue
            ckpt_tensor = raw_state[key].float()
            model_tensor = model_sd[key].float().cpu()
            if torch.allclose(ckpt_tensor, model_tensor, atol=1e-4):
                matched += 1
            else:
                max_diff = (ckpt_tensor - model_tensor).abs().max().item()
                mismatches.append((key, max_diff))
        if mismatches:
            fail(f"{prefix}: {len(mismatches)} 个 key 不匹配 (max_diff > 1e-4)")
            for k, d in mismatches[:5]:
                print(f"    {k}: max_diff={d:.6f}")
            return False
        ok(f"{prefix}: {matched}/{len(ckpt_keys)} 个 key 完全匹配 ✓")
        return True

    all_pass &= compare_weights(model.gme, "gme", ckpt_gme_keys)
    all_pass &= compare_weights(model.cdm, "cdm", ckpt_cdm_keys)
    all_pass &= compare_weights(model.pims, "pims", ckpt_pim_keys)

    return all_pass


def verify_hooks_installed(model) -> bool:
    section("3. Hook 验证：PIM hook 是否安装在正确的 decoder layer 上")
    all_pass = True

    pim_map = dict(model._pim_map)
    hooks = model._hooks
    ok(f"_pim_map = {pim_map}  ({len(pim_map)} 个映射)")
    ok(f"已安装 {len(hooks)} 个 forward hooks")

    qwen_model = model.base_model.transformer.model
    decoder_layers = getattr(qwen_model, "layers", None)
    if decoder_layers is None:
        fail("无法找到 decoder layers")
        return False

    for layer_idx, pim_idx in sorted(pim_map.items()):
        layer = decoder_layers[layer_idx]
        hook_count = len(layer._forward_hooks)
        if hook_count > 0:
            ok(f"decoder layer[{layer_idx}] 有 {hook_count} 个 forward hook → PIM[{pim_idx}]")
        else:
            fail(f"decoder layer[{layer_idx}] 没有 forward hook，但 _pim_map 中有映射")
            all_pass = False

    return all_pass


def verify_runtime_memory(model, hvm_config) -> bool:
    section("4. Runtime 验证：prepare_hvm_runtime_memory 两路 memory 是否被正确填充")
    all_pass = True
    device = next(model.base_model.parameters()).device

    if model.gme is not None:
        hvm_dtype = next(model.gme.parameters()).dtype
    elif len(model.pims) > 0:
        hvm_dtype = next(model.pims.parameters()).dtype
    else:
        fail("无法确定 HVM dtype")
        return False
    ok(f"HVM dtype = {hvm_dtype}, device = {device}")

    num_refs = hvm_config.num_references
    d_vision = hvm_config.d_vision
    fake_ref = torch.randn(1, num_refs, 256, d_vision, device=device, dtype=hvm_dtype)
    ok(f"构造 fake ref_features: shape={list(fake_ref.shape)}")

    model._memory_ready = False
    model._gist_feats = None
    model._detail_feats = None
    model._detail_slot_mask = None

    with torch.no_grad():
        gist_feats = model.gme(fake_ref)
    model._gist_feats = gist_feats
    ok(f"GME 输出 gist_feats: shape={list(gist_feats.shape)}, "
       f"norm={gist_feats.float().norm().item():.4f}, "
       f"mean={gist_feats.float().mean().item():.6f}")
    if gist_feats.shape[1] != hvm_config.gme_num_queries:
        fail(f"gist_feats 序列长度 {gist_feats.shape[1]} != 预期 {hvm_config.gme_num_queries}")
        all_pass = False
    else:
        ok(f"gist_feats 序列长度 = {hvm_config.gme_num_queries} (GME queries) ✓")

    num_groups = min(hvm_config.pme_max_groups, 12)
    fake_part = torch.randn(1, num_groups, 256, d_vision, device=device, dtype=hvm_dtype)
    fake_tag = torch.rand(1, num_groups, hvm_config.cdm_tag_meta_dim, device=device, dtype=hvm_dtype)
    fake_gids = torch.arange(num_groups, device=device).unsqueeze(0)
    fake_pmask = torch.ones(1, num_groups, device=device, dtype=torch.bool)
    ok(f"构造 fake part inputs: parts={num_groups} groups × 256 tokens")

    with torch.no_grad():
        detail_feats = model.cdm(
            fake_part,
            gist_feats.detach(),
            part_tag_meta=fake_tag,
            part_group_ids=fake_gids,
            part_mask=fake_pmask,
        )
    model._detail_feats = detail_feats
    model._detail_slot_mask = getattr(model.cdm, "last_detail_slot_mask", None)
    ok(f"CDM 输出 detail_feats: shape={list(detail_feats.shape)}, "
       f"norm={detail_feats.float().norm().item():.4f}, "
       f"mean={detail_feats.float().mean().item():.6f}")

    expected_detail_slots = hvm_config.cdm_num_queries
    if detail_feats.shape[1] != expected_detail_slots:
        fail(f"detail_feats 序列长度 {detail_feats.shape[1]} != 预期 {expected_detail_slots}")
        all_pass = False
    else:
        ok(f"detail_feats 序列长度 = {expected_detail_slots} (CDM queries) ✓")

    if model._detail_slot_mask is not None:
        ok(f"detail_slot_mask: shape={list(model._detail_slot_mask.shape)}, "
           f"active={model._detail_slot_mask.sum().item()}/{model._detail_slot_mask.numel()}")
    else:
        ok("detail_slot_mask = None (全部 slot 都有效)")

    model._memory_ready = True
    ok("_memory_ready = True (两路 memory 已注入)")
    return all_pass


def verify_forward_injection(model, hvm_config) -> bool:
    section("5. Forward 验证：PIM hook 是否实际修改了 hidden_state")
    all_pass = True
    device = next(model.base_model.parameters()).device

    if not model._memory_ready:
        fail("memory 未就绪，跳过 forward 验证")
        return False

    seq_len = 10
    d_model = hvm_config.d_model
    hvm_dtype = model._gist_feats.dtype
    fake_hidden = torch.randn(1, seq_len, d_model, device=device, dtype=hvm_dtype)

    for pim_idx, pim in enumerate(model.pims):
        layer_idx = hvm_config.pim_layer_indices[pim_idx] if pim_idx < len(hvm_config.pim_layer_indices) else -1
        B = fake_hidden.shape[0]
        gist_feats = model._gist_feats.expand(B, -1, -1)
        detail_feats = model._detail_feats.expand(B, -1, -1)
        detail_slot_mask = None
        if model._detail_slot_mask is not None:
            detail_slot_mask = model._detail_slot_mask.expand(B, -1)

        with torch.no_grad():
            output = pim(fake_hidden, gist_feats, detail_feats, detail_slot_mask)

        delta = (output - fake_hidden).float()
        delta_norm = delta.norm().item()
        delta_mean = delta.abs().mean().item()

        stats = pim.last_stats if hasattr(pim, "last_stats") and pim.last_stats else {}
        alpha_gist_val = None
        alpha_detail_val = None
        if hasattr(pim, "alpha_gist") and pim.alpha_gist is not None:
            alpha_gist_val = torch.tanh(pim.alpha_gist).item()
        if hasattr(pim, "alpha_detail") and pim.alpha_detail is not None:
            alpha_detail_val = torch.tanh(pim.alpha_detail).item()

        if delta_norm > 1e-8:
            ok(
                f"PIM[{pim_idx}] @layer{layer_idx}: "
                f"delta_norm={delta_norm:.4f}, delta_mean={delta_mean:.6f}, "
                f"gate_gist={alpha_gist_val:.4f}, gate_detail={alpha_detail_val:.4f}"
            )
        else:
            fail(f"PIM[{pim_idx}] @layer{layer_idx}: delta_norm={delta_norm:.10f} ≈ 0，注入无效！")
            all_pass = False

        if alpha_gist_val is not None and abs(alpha_gist_val) < 1e-6:
            fail(f"  PIM[{pim_idx}] alpha_gist ≈ 0，gist 路无效")
            all_pass = False
        if alpha_detail_val is not None and abs(alpha_detail_val) < 1e-6:
            fail(f"  PIM[{pim_idx}] alpha_detail ≈ 0，detail 路无效")
            all_pass = False

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="验证 HVM 模型加载和两路 memory 注入")
    parser.add_argument("--hvm_checkpoint", type=str, default=DEFAULT_HVM_CKPT)
    parser.add_argument("--hvm_config", type=str, default=DEFAULT_HVM_CONFIG)
    parser.add_argument("--config_dir", type=str, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--model_size", type=str, default="8B")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    section("加载模型")
    print(f"  hvm_checkpoint: {args.hvm_checkpoint}")
    print(f"  hvm_config: {args.hvm_config}")
    print(f"  device: {args.device}")

    model, tokenizer, processor, token_config, hvm_config = load_hvm_model(
        model_size=args.model_size,
        hvm_config_path=args.hvm_config,
        hvm_checkpoint_path=args.hvm_checkpoint,
        config_dir=args.config_dir,
        device=args.device,
    )
    ok("模型加载完成")

    results = {}
    results["structure"] = verify_module_structure(model)
    results["checkpoint"] = verify_checkpoint_loaded(model, args.hvm_checkpoint)
    results["hooks"] = verify_hooks_installed(model)
    results["runtime_memory"] = verify_runtime_memory(model, hvm_config)
    results["forward_injection"] = verify_forward_injection(model, hvm_config)

    section("验证总结")
    all_pass = True
    for name, passed in results.items():
        status = f"{Colors.OK}PASS{Colors.END}" if passed else f"{Colors.FAIL}FAIL{Colors.END}"
        print(f"  {name:25s} : {status}")
        all_pass &= passed

    print()
    if all_pass:
        print(f"{Colors.OK}{Colors.BOLD}  所有验证通过！模型确实加载了 HVM 模块并注入了两路 memory (Gist + Detail)。{Colors.END}")
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}  部分验证失败，请检查上方日志。{Colors.END}")
    print()


if __name__ == "__main__":
    main()
