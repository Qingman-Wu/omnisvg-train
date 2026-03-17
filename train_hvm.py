"""
HVM-SVG Training Script
========================
基于 OmniSVG 的 HVM-SVG 训练:
  - 冻结 OmniSVG base model
  - 只训练 HVM 模块 (GME, PME, PIMs)
  - 使用预计算的 vision features 和 RAG 检索结果

Usage:
    # 单 GPU 测试
    python train_hvm.py --batch_size 1

    # 多 GPU 分布式训练 (DeepSpeed ZeRO-2)
    accelerate launch --config_file configs/ds_zero2_hvm.yaml train_hvm.py --batch_size 2

    # 从完整 checkpoint 恢复训练 (optimizer/scheduler/step 全部恢复)
    accelerate launch ... train_hvm.py --resume_from ./outputs_hvm/checkpoint-epoch-5

    # 从 HVM 权重初始化 (仅加载模型权重, 不恢复训练状态)
    accelerate launch ... train_hvm.py --hvm_checkpoint ./outputs_hvm/hvm_epoch_5.pt
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import swanlab
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator

# Local imports
from utils.config import OmniSVGConfig, TokenizationConfig, TrainConfig
from decoder import SketchDecoder
from hvm_modules import HVMConfig, count_parameters
from hvm_decoder import HVMSketchDecoder
from hvm_dataset import HVMDataset, create_hvm_collate_fn

# ============================================================================
# Monkey-patch for DeepSpeed compatibility (same as train.py)
# ============================================================================
import torch.autograd.graph as _torch_ag_graph
_original = getattr(_torch_ag_graph, '_get_grad_fn_or_grad_acc', None)
if _original is not None:
    def _safe(t):
        if t.requires_grad and t.grad_fn is None:
            view = t.view_as(t)
            if view.grad_fn is None:
                return None
            return view.grad_fn.next_functions[0][0]
        else:
            return t.grad_fn
    _torch_ag_graph._get_grad_fn_or_grad_acc = _safe


# ============================================================================
# Model Loading
# ============================================================================

# Checkpoint paths (same as train.py) 只有在config/tokenization.yaml文件中没有指定checkpoint时使用
MODEL_DEFAULTS = {
    "8B": {
        "base_model": "/mnt/data/wuqingman/models/Qwen/Qwen2.5-VL-7B-Instruct",
        "checkpoint": "/mnt/data2/wuqingman/models/OmniSVG/OmniSVG1.1_8B",
    },
}


def load_omnisvg_base(
    model_size: str,
    token_config: TokenizationConfig,
    checkpoint_path: Optional[str] = None,
) -> SketchDecoder:
    """加载 OmniSVG base model"""
    defaults = MODEL_DEFAULTS[model_size]
    base_model_path = token_config.base_model or defaults["base_model"]
    default_ckpt = token_config.checkpoint or defaults["checkpoint"]

    print(f"Loading OmniSVG {model_size} from {base_model_path}")

    model = SketchDecoder(
        pix_len=2048,
        text_len=800,
        model_path=base_model_path,
        vocab_size=token_config.extended_vocab_size,
        bos_token_id=token_config.bos_token_id,
        eos_token_id=token_config.eos_token_id,
        pad_token_id=token_config.pad_token_id,
    )

    # 加载 OmniSVG checkpoint
    ckpt_path = checkpoint_path or default_ckpt
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading OmniSVG checkpoint from {ckpt_path}")
        from train import load_checkpoint_state_dict, find_checkpoint_file

        if os.path.isdir(ckpt_path):
            ckpt_file = find_checkpoint_file(ckpt_path)
            if ckpt_file:
                state_dict = load_checkpoint_state_dict(
                    ckpt_file if os.path.isfile(ckpt_file) else ckpt_path
                )
            else:
                print(f"Warning: No checkpoint file found in {ckpt_path}")
                state_dict = None
        else:
            state_dict = load_checkpoint_state_dict(ckpt_path)

        if state_dict:
            # Remap keys for transformers >=4.57 where Qwen2_5_VL restructured:
            #   model.X → model.language_model.X  (language parts)
            #   visual.X → model.visual.X
            model_keys = set(model.state_dict().keys())
            sample_ckpt_key = next(iter(state_dict))
            if sample_ckpt_key not in model_keys:
                remapped = {}
                for k, v in state_dict.items():
                    new_k = k
                    if k.startswith("transformer.visual."):
                        new_k = k.replace("transformer.visual.", "transformer.model.visual.", 1)
                    elif k.startswith("transformer.model.") and not k.startswith("transformer.model.language_model.") and not k.startswith("transformer.model.visual."):
                        new_k = k.replace("transformer.model.", "transformer.model.language_model.", 1)
                    remapped[new_k] = v
                if next(iter(remapped)) in model_keys:
                    print(f"  Remapped {len(remapped)} checkpoint keys for transformers compat")
                    state_dict = remapped

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print(f"Warning: OmniSVG checkpoint not found at {ckpt_path}")

    return model


# ============================================================================
# Loss Computation
# ============================================================================

def compute_loss(
    outputs: Any,
    labels: torch.Tensor,
) -> torch.Tensor:
    """计算 SVG token 的 cross-entropy loss"""
    logits = outputs.logits[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous().to(logits.device)

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )
    return loss


def _param_grad_norm(param: Optional[torch.nn.Parameter]) -> Optional[float]:
    """返回参数梯度 L2 norm（若当前步无梯度则返回 None）。"""
    if param is None or param.grad is None:
        return None
    return float(param.grad.detach().float().norm().item())


def _scalar_tensor_to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().item())
    return float(x)


def parse_pim_layer_indices(spec: Optional[str], num_decoder_layers: int) -> Optional[List[int]]:
    """
    解析逗号分隔的 PIM 层索引字符串，支持 -1 表示最后一层。
    例如: "-1" / "3,7,11"
    """
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None

    indices: List[int] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid pim layer index '{token}', expected integer.") from exc

        if idx == -1:
            idx = num_decoder_layers - 1
        if idx < 0 or idx >= num_decoder_layers:
            raise ValueError(
                f"PIM layer index {idx} out of range [0, {num_decoder_layers - 1}] "
                f"(original token: '{token}')."
            )
        if idx not in indices:
            indices.append(idx)

    if not indices:
        raise ValueError("pim_layer_indices is empty after parsing.")
    return indices


def collect_hvm_diagnostics(unwrapped_model: nn.Module) -> Dict[str, float]:
    """
    采集 HVM 训练诊断信息：
      - 代表性梯度范数（用于判断是否只有 gate 在学习）
      - 每个 PIM 的 gate / token_gate / 注入强度
    """
    stats: Dict[str, float] = {}

    # 代表性梯度（避免遍历全部参数，减小开销）
    grad_targets = {}
    if getattr(unwrapped_model, "gme", None) is not None:
        grad_targets["grad/gme_input_proj"] = unwrapped_model.gme.qformer.input_proj.weight
    if getattr(unwrapped_model, "pme", None) is not None:
        grad_targets["grad/pme_input_proj"] = unwrapped_model.pme.qformer.input_proj.weight
    if getattr(unwrapped_model, "local_token_encoder", None) is not None:
        grad_targets["grad/local_token_tag_mlp"] = unwrapped_model.local_token_encoder.tag_meta_mlp[0].weight

    if len(unwrapped_model.pims) > 0:
        pim0 = unwrapped_model.pims[0]
        pim_last = unwrapped_model.pims[-1]
        if hasattr(pim0, "hidden_aligned_cross_attn"):
            grad_targets["grad/pim0_hidden_attn_q"] = pim0.hidden_aligned_cross_attn.to_q.weight
        elif hasattr(pim0, "hidden_gist_cross_attn"):
            grad_targets["grad/pim0_hidden_gist_attn_q"] = pim0.hidden_gist_cross_attn.to_q.weight
        elif hasattr(pim0, "global_cross_attn"):
            grad_targets["grad/pim0_global_attn_q"] = pim0.global_cross_attn.to_q.weight
            grad_targets["grad/pim0_local_attn_q"] = pim0.local_cross_attn.to_q.weight
        if hasattr(pim0, "gate"):
            grad_targets["grad/pim0_gate_alpha"] = pim0.gate.base_alpha
        if hasattr(pim_last, "gate"):
            grad_targets["grad/pim_last_gate_alpha"] = pim_last.gate.base_alpha

    for key, param in grad_targets.items():
        grad_norm = _param_grad_norm(param)
        if grad_norm is not None:
            stats[key] = grad_norm

    # PIM 运行时统计
    # - PrefrontalInjectionModule (full mode): gate.base_alpha + gate.last_stats
    # - LayerGatedGMEInjectionModule (gme+adaptive): pim.base_alpha + pim.last_stats
    # - SimpleGMEInjectionModule (gme+fixed): pim.last_stats with inject_scale
    for pim_idx, pim in enumerate(unwrapped_model.pims):
        if hasattr(pim, "gate"):
            gate = pim.gate
            stats[f"gate/pim_{pim_idx}_tanh_alpha"] = float(torch.tanh(gate.base_alpha.detach()).item())
            last_stats = getattr(gate, "last_stats", None) or {}
            for src_key, dst_key in [
                ("token_gate_mean", f"gate/pim_{pim_idx}_token_mean"),
                ("token_gate_std", f"gate/pim_{pim_idx}_token_std"),
                ("delta_rms", f"gate/pim_{pim_idx}_delta_rms"),
                ("inject_rms", f"gate/pim_{pim_idx}_inject_rms"),
                ("inject_hidden_ratio", f"gate/pim_{pim_idx}_inject_hidden_ratio"),
            ]:
                val = _scalar_tensor_to_float(last_stats.get(src_key))
                if val is not None:
                    stats[dst_key] = val
        elif hasattr(pim, "alpha_detail"):
            gist_enabled = bool(getattr(pim, "enable_gist", True))
            stats[f"gate/pim_{pim_idx}_tanh_alpha_gist"] = (
                float(torch.tanh(pim.alpha_gist.detach()).item()) if gist_enabled else 0.0
            )
            stats[f"gate/pim_{pim_idx}_gist_enabled"] = 1.0 if gist_enabled else 0.0
            detail_enabled = bool(getattr(pim, "enable_detail", True))
            stats[f"gate/pim_{pim_idx}_tanh_alpha_detail"] = (
                float(torch.tanh(pim.alpha_detail.detach()).item()) if detail_enabled else 0.0
            )
            stats[f"gate/pim_{pim_idx}_detail_enabled"] = 1.0 if detail_enabled else 0.0
            last_stats = getattr(pim, "last_stats", None) or {}
            for src_key, dst_key in [
                ("gate_gist", f"gate/pim_{pim_idx}_gate_gist"),
                ("gate_detail", f"gate/pim_{pim_idx}_gate_detail"),
                ("delta_gist_rms", f"gate/pim_{pim_idx}_delta_gist_rms"),
                ("delta_detail_rms", f"gate/pim_{pim_idx}_delta_detail_rms"),
                ("inject_gist_rms", f"gate/pim_{pim_idx}_inject_gist_rms"),
                ("inject_detail_rms", f"gate/pim_{pim_idx}_inject_detail_rms"),
                ("delta_rms", f"gate/pim_{pim_idx}_delta_rms"),
                ("inject_rms", f"gate/pim_{pim_idx}_inject_rms"),
                ("inject_hidden_ratio", f"gate/pim_{pim_idx}_inject_hidden_ratio"),
                ("router_conf_mean", f"gate/pim_{pim_idx}_router_conf_mean"),
                ("router_effective_conf_mean", f"gate/pim_{pim_idx}_router_effective_conf_mean"),
                ("router_entropy_mean", f"gate/pim_{pim_idx}_router_entropy_mean"),
                ("router_top1_prob_mean", f"gate/pim_{pim_idx}_router_top1_prob_mean"),
                ("router_active_ratio", f"gate/pim_{pim_idx}_router_active_ratio"),
                ("router_slot_usage_entropy", f"gate/pim_{pim_idx}_router_slot_usage_entropy"),
            ]:
                val = _scalar_tensor_to_float(last_stats.get(src_key))
                if val is not None:
                    stats[dst_key] = val
        elif hasattr(pim, "alpha_global"):
            stats[f"gate/pim_{pim_idx}_tanh_alpha_global"] = float(torch.tanh(pim.alpha_global.detach()).item())
            stats[f"gate/pim_{pim_idx}_tanh_alpha_local"] = float(torch.tanh(pim.alpha_local.detach()).item())
            last_stats = getattr(pim, "last_stats", None) or {}
            for src_key, dst_key in [
                ("gate_global", f"gate/pim_{pim_idx}_gate_global"),
                ("gate_local", f"gate/pim_{pim_idx}_gate_local"),
                ("delta_global_rms", f"gate/pim_{pim_idx}_delta_global_rms"),
                ("delta_local_rms", f"gate/pim_{pim_idx}_delta_local_rms"),
                ("inject_global_rms", f"gate/pim_{pim_idx}_inject_global_rms"),
                ("inject_local_rms", f"gate/pim_{pim_idx}_inject_local_rms"),
                ("delta_rms", f"gate/pim_{pim_idx}_delta_rms"),
                ("inject_rms", f"gate/pim_{pim_idx}_inject_rms"),
                ("inject_hidden_ratio", f"gate/pim_{pim_idx}_inject_hidden_ratio"),
            ]:
                val = _scalar_tensor_to_float(last_stats.get(src_key))
                if val is not None:
                    stats[dst_key] = val
        elif hasattr(pim, "alpha_ref"):
            stats[f"gate/pim_{pim_idx}_tanh_alpha_gist"] = float(torch.tanh(pim.alpha_gist.detach()).item())
            stats[f"gate/pim_{pim_idx}_tanh_alpha_ref"] = float(torch.tanh(pim.alpha_ref.detach()).item())
            last_stats = getattr(pim, "last_stats", None) or {}
            for src_key, dst_key in [
                ("gate_gist", f"gate/pim_{pim_idx}_gate_gist"),
                ("gate_ref", f"gate/pim_{pim_idx}_gate_ref"),
                ("delta_gist_rms", f"gate/pim_{pim_idx}_delta_gist_rms"),
                ("delta_ref_rms", f"gate/pim_{pim_idx}_delta_ref_rms"),
                ("inject_gist_rms", f"gate/pim_{pim_idx}_inject_gist_rms"),
                ("inject_ref_rms", f"gate/pim_{pim_idx}_inject_ref_rms"),
                ("delta_rms", f"gate/pim_{pim_idx}_delta_rms"),
                ("inject_rms", f"gate/pim_{pim_idx}_inject_rms"),
                ("inject_hidden_ratio", f"gate/pim_{pim_idx}_inject_hidden_ratio"),
            ]:
                val = _scalar_tensor_to_float(last_stats.get(src_key))
                if val is not None:
                    stats[dst_key] = val
        elif hasattr(pim, "alpha_gist"):
            stats[f"gate/pim_{pim_idx}_tanh_alpha_gist"] = float(torch.tanh(pim.alpha_gist.detach()).item())
            stats[f"gate/pim_{pim_idx}_tanh_alpha_part"] = float(torch.tanh(pim.alpha_part.detach()).item())
            last_stats = getattr(pim, "last_stats", None) or {}
            for src_key, dst_key in [
                ("gate_gist", f"gate/pim_{pim_idx}_gate_gist"),
                ("gate_part", f"gate/pim_{pim_idx}_gate_part"),
                ("delta_gist_rms", f"gate/pim_{pim_idx}_delta_gist_rms"),
                ("delta_part_rms", f"gate/pim_{pim_idx}_delta_part_rms"),
                ("inject_gist_rms", f"gate/pim_{pim_idx}_inject_gist_rms"),
                ("inject_part_rms", f"gate/pim_{pim_idx}_inject_part_rms"),
                ("delta_rms", f"gate/pim_{pim_idx}_delta_rms"),
                ("inject_rms", f"gate/pim_{pim_idx}_inject_rms"),
                ("inject_hidden_ratio", f"gate/pim_{pim_idx}_inject_hidden_ratio"),
            ]:
                val = _scalar_tensor_to_float(last_stats.get(src_key))
                if val is not None:
                    stats[dst_key] = val
        elif hasattr(pim, "base_alpha"):
            stats[f"gate/pim_{pim_idx}_tanh_alpha"] = float(torch.tanh(pim.base_alpha.detach()).item())
            last_stats = getattr(pim, "last_stats", None) or {}
            for src_key, dst_key in [
                ("delta_gist_rms", f"gate/pim_{pim_idx}_delta_gist_rms"),
                ("delta_part_rms", f"gate/pim_{pim_idx}_delta_part_rms"),
                ("delta_rms", f"gate/pim_{pim_idx}_delta_rms"),
                ("inject_rms", f"gate/pim_{pim_idx}_inject_rms"),
                ("inject_hidden_ratio", f"gate/pim_{pim_idx}_inject_hidden_ratio"),
            ]:
                val = _scalar_tensor_to_float(last_stats.get(src_key))
                if val is not None:
                    stats[dst_key] = val
        else:
            last_stats = getattr(pim, "last_stats", None) or {}
            inject_scale = _scalar_tensor_to_float(last_stats.get("inject_scale"))
            if inject_scale is not None:
                stats[f"inject/pim_{pim_idx}_scale"] = inject_scale
            for src_key, dst_key in [
                ("delta_rms", f"gate/pim_{pim_idx}_delta_rms"),
                ("inject_rms", f"gate/pim_{pim_idx}_inject_rms"),
                ("inject_hidden_ratio", f"gate/pim_{pim_idx}_inject_hidden_ratio"),
            ]:
                val = _scalar_tensor_to_float(last_stats.get(src_key))
                if val is not None:
                    stats[dst_key] = val

    return stats


@torch.no_grad()
def evaluate_val_loss(
    model: nn.Module,
    val_dataloader,
    accelerator: "Accelerator",
    disable_hvm: bool = False,
) -> float:
    """在验证集上计算平均 loss。"""
    model.eval()
    all_losses = []

    for batch in val_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        if disable_hvm:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ref_features=batch["ref_features"],
                group_features_list=batch["group_features_list"],
                part_features=batch["part_features"],
                part_tag_meta=batch["part_tag_meta"],
                part_group_ids=batch["part_group_ids"],
                part_mask=batch["part_mask"],
                ref_text_ids=batch["ref_text_ids"],
                ref_text_mask=batch["ref_text_mask"],
            )

        loss = compute_loss(outputs, labels)
        gathered_loss = accelerator.gather(loss.unsqueeze(0))
        all_losses.extend(gathered_loss.cpu().tolist())

    model.train()
    return float(np.mean(all_losses)) if all_losses else 0.0


def get_git_metadata(workdir: Optional[Path] = None) -> Dict[str, Any]:
    """
    获取当前代码版本信息（用于实验可复现性）。
    返回字段:
      - git_commit: 完整 commit hash 或 None
      - git_commit_short: 短 hash 或 None
      - git_branch: 分支名或 None
      - git_dirty: bool / None (非 git 仓库或获取失败时为 None)
    """
    cwd = str(workdir) if workdir is not None else None

    def _run_git(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="replace").strip()
        except Exception:
            return None

    commit = _run_git(["rev-parse", "HEAD"])
    commit_short = _run_git(["rev-parse", "--short", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--porcelain"])

    git_dirty: Optional[bool]
    if status is None:
        git_dirty = None
    else:
        git_dirty = len(status.strip()) > 0

    return {
        "git_commit": commit,
        "git_commit_short": commit_short,
        "git_branch": branch,
        "git_dirty": git_dirty,
    }


# ============================================================================
# Training
# ============================================================================

def train(args):
    """Main training function"""

    # ---- Config ----
    config = OmniSVGConfig(
        config_dir=args.config_dir,
        model_size=args.model_size,
    )
    token_config = config.tokenization
    if args.base_model:
        token_config.base_model = args.base_model
    hvm_config = HVMConfig(
        d_model=3584,
        d_vision=3584,       # post-merge dim (GME & PME 统一)
        d_qformer=args.d_qformer, #1024，QFormer 内部维度
        d_pim_inner=args.d_pim_inner, #512，PIM attention bottleneck 维度
        pim_layer_interval=args.pim_layer_interval, #4，每隔 N 层插入 PIM
        num_decoder_layers=28,
        gate_alpha_init=args.gate_alpha_init,
        gme_num_queries=args.gme_num_queries,
        pme_max_groups=args.pme_max_groups,
        memory_mode=args.memory_mode,
        inject_mode=args.inject_mode,
        inject_scale=args.inject_scale,
        pim_layer_indices_override=args.pim_layer_indices,
        delta_ln=args.delta_ln,
        dra_d_inner=args.dra_d_inner,
        dra_n_heads=args.dra_n_heads,
        cdm_num_queries=args.cdm_num_queries,
        cdm_num_layers=args.cdm_num_layers,
        cdm_layout=args.cdm_layout,
        cdm_group_queries_per_group=args.cdm_group_queries_per_group,
        cdm_detail_source=args.cdm_detail_source,
        cdm_use_tag_meta=not args.cdm_disable_tag_meta,
        cdm_use_group_id=not args.cdm_disable_group_id,
        cdm_disable_gist=args.cdm_disable_gist,
        edr_d_router=args.edr_d_router,
        edr_top_k=args.edr_top_k,
        edr_disable_conf=args.edr_disable_conf,
        edr_disable_gist=args.edr_disable_gist,
        edr_random_replace_top1=args.edr_random_replace_top1,
        edr_detail_layer_indices_override=args.edr_detail_layer_indices,
    )

    # ---- Accelerator ----
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    # DeepSpeed ZeRO-2 handles gradient accumulation internally;
    # its reduce_scatter is incompatible with accelerate's no_sync().
    import contextlib
    from accelerate.utils import DistributedType
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.no_sync = lambda model: contextlib.nullcontext()
    set_seed(args.seed)

    # ---- Tokenizer & Processor ----
    base_model_path = token_config.base_model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="left")
    processor = AutoProcessor.from_pretrained(base_model_path, padding_side="left")
    processor.tokenizer.padding_side = "left"

    # ---- Dataset ----
    accelerator.print("Loading HVM dataset...")
    dataset = HVMDataset(
        data_dir=args.data_dir,
        hvm_dir=args.hvm_dir,
        token_config=token_config,
        train_config=config.training,
        max_len=config.training.max_seq_length,
        shuffle_rag=args.shuffle_rag,
        shuffle_gme=args.shuffle_gme,
        shuffle_cdm=args.shuffle_cdm,
        part_num_refs=args.part_num_refs,
    )

    collate_fn = create_hvm_collate_fn(
        processor=processor,
        tokenizer=tokenizer,
        token_config=token_config,
        text_len=config.training.text_max_length,
        max_seq_length=config.training.max_seq_length,
        ref_text_max_length=hvm_config.ref_text_max_length,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # ---- Validation Dataset (optional) ----
    val_dataloader = None
    if args.val_data_dir and args.val_hvm_dir:
        accelerator.print("Loading validation dataset...")
        val_dataset = HVMDataset(
            data_dir=args.val_data_dir,
            hvm_dir=args.val_hvm_dir,
            token_config=token_config,
            train_config=config.training,
            max_len=config.training.max_seq_length,
            shuffle_rag=False,
            is_eval=True,
            part_num_refs=args.part_num_refs,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            collate_fn=collate_fn,
            drop_last=False,
        )
        accelerator.print(f"  Val samples: {len(val_dataset)}")

    # ---- Model ----
    accelerator.print("Loading OmniSVG base model...")
    base_model = load_omnisvg_base(
        model_size=args.model_size,
        token_config=token_config,
        checkpoint_path=args.omnisvg_checkpoint,
    )

    accelerator.print("Building HVM model...")
    model = HVMSketchDecoder(
        base_model=base_model,
        hvm_config=hvm_config,
        tokenizer=tokenizer,
    )

    # 加载 HVM checkpoint (如果有)
    if args.hvm_checkpoint and os.path.exists(args.hvm_checkpoint):
        accelerator.print(f"Loading HVM checkpoint from {args.hvm_checkpoint}")
        model.load_hvm_checkpoint(args.hvm_checkpoint)

    # ---- Optimizer (只优化 HVM 参数) ----
    trainable_params = model.get_trainable_parameters()
    accelerator.print(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ---- Prepare with Accelerator ----
    # 注意: 必须先 prepare dataloader，再计算 steps，
    # 因为 accelerator.prepare 会给 dataloader 加 DistributedSampler，
    # 改变 len(dataloader)。
    if val_dataloader is not None:
        model, optimizer, dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, dataloader, val_dataloader
        )
    else:
        model, optimizer, dataloader = accelerator.prepare(
            model, optimizer, dataloader
        )

    # ---- Scheduler (在 prepare 之后计算 steps) ----
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    total_training_steps = num_update_steps_per_epoch * args.epochs

    # Warmup steps: 默认 10% of total, 但至少 10 步
    if args.warmup_steps is None:
        warmup_steps = max(10, int(total_training_steps * 0.1))
    else:
        warmup_steps = args.warmup_steps

    accelerator.print(f"  Steps per epoch: {num_update_steps_per_epoch} (dataloader batches: {len(dataloader)})")
    accelerator.print(f"  Total training steps: {total_training_steps}, warmup: {warmup_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    # 注意: 不要 accelerator.prepare(lr_scheduler)!
    # 因为 steps 已经基于 prepare 后的 dataloader 计算（已是 per-GPU 视角），
    # 如果再 prepare scheduler，AcceleratedScheduler 会内部再除以 num_processes，
    # 导致 lr schedule 被压缩 8 倍，几个 epoch 就衰减到 0。

    # ---- Resume from full training checkpoint ----
    start_epoch = 0
    global_step = 0
    resume_step_in_epoch = 0  # 当前 epoch 中已完成的 optimizer steps

    if args.resume_from and os.path.exists(args.resume_from):
        accelerator.print(f"Resuming full training state from {args.resume_from}")
        # 所有 rank 参与: 恢复 model (DeepSpeed engine) + optimizer states
        accelerator.load_state(args.resume_from)

        # 读取训练元信息 (所有 rank 都需要)
        metadata_path = os.path.join(args.resume_from, "training_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)
        global_step = metadata["global_step"]
        start_epoch = metadata["epoch"]

        # 恢复 lr_scheduler state (不经过 accelerator.prepare, 手动保存/加载)
        scheduler_path = os.path.join(args.resume_from, "lr_scheduler.pt")
        if os.path.exists(scheduler_path):
            lr_scheduler.load_state_dict(
                torch.load(scheduler_path, map_location="cpu", weights_only=True)
            )

        # 计算当前 epoch 中需要跳过的 batches
        resume_step_in_epoch = global_step - start_epoch * num_update_steps_per_epoch

        accelerator.print(
            f"  Resumed: epoch={start_epoch + 1}/{args.epochs}, "
            f"global_step={global_step}/{total_training_steps}, "
            f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
        )
        if resume_step_in_epoch > 0:
            accelerator.print(
                f"  Will skip {resume_step_in_epoch} steps "
                f"({resume_step_in_epoch * args.gradient_accumulation_steps} batches) "
                f"in epoch {start_epoch + 1}"
            )

    # ---- Output dir & SwanLab ----
    output_dir = Path(args.output_dir)
    git_meta = get_git_metadata(Path.cwd())
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

        accelerator.print(
            f"Git: branch={git_meta.get('git_branch')} "
            f"commit={git_meta.get('git_commit_short')} "
            f"dirty={git_meta.get('git_dirty')}"
        )

        # 初始化 SwanLab
        swanlab.init(
            project="HVM-SVG",
            experiment_name=args.swanlab_run_name or f"hvm-{args.model_size}-lr{args.learning_rate}",
            description="HVM-SVG: Hierarchical Visual Memory for SVG Generation",
            config={
                **vars(args),
                "hvm_config": hvm_config.__dict__,
                "num_pims": hvm_config.num_pims,
                "pim_layers": hvm_config.pim_layer_indices,
                "edr_detail_layers": hvm_config.edr_detail_layer_indices,
                "trainable_params_M": sum(p.numel() for p in trainable_params) / 1e6,
                **git_meta,
            },
            logdir=str(output_dir / "swanlog"),
            mode=args.swanlab_mode,
        )

        # 保存配置
        run_config = {
            **vars(args),
            **git_meta,
        }
        with open(output_dir / "hvm_config.json", "w") as f:
            json.dump(run_config, f, indent=2)
        with open(output_dir / "hvm_model_config.json", "w") as f:
            # __dict__ 不包含 @property，需手动补充
            config_dict = {
                **hvm_config.__dict__,
                "pim_layer_indices": hvm_config.pim_layer_indices,
                "num_pims": hvm_config.num_pims,
                "edr_detail_layer_indices": hvm_config.edr_detail_layer_indices,
            }
            json.dump(config_dict, f, indent=2, default=str)

    # ---- Training Loop ----
    accelerator.print("=" * 60)
    if args.disable_hvm:
        accelerator.print("BASELINE MODE: HVM injection DISABLED (frozen OmniSVG only)")
    else:
        accelerator.print("Starting HVM-SVG Training")
    accelerator.print(f"  Epochs: {args.epochs}")
    accelerator.print(f"  Batch size: {args.batch_size}")
    accelerator.print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    accelerator.print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
    accelerator.print(f"  Learning rate: {args.learning_rate}")
    accelerator.print(f"  Total steps: {total_training_steps}")
    accelerator.print(f"  Warmup steps: {warmup_steps}")
    if not args.disable_hvm:
        accelerator.print(f"  Memory mode: {hvm_config.memory_mode}")
        accelerator.print(f"  Inject mode: {hvm_config.inject_mode}")
        accelerator.print(f"  Part refs used: {args.part_num_refs}")
        accelerator.print(f"  PME max groups: {hvm_config.pme_max_groups}")
        if hvm_config.memory_mode in ("gme_cdm", "gme_cdm_edr"):
            accelerator.print(f"  CDM layout: {hvm_config.cdm_layout}")
            accelerator.print(f"  CDM detail source: {hvm_config.cdm_detail_source}")
            accelerator.print(f"  CDM disable gist: {hvm_config.cdm_disable_gist}")
            accelerator.print(f"  CDM use tag meta: {hvm_config.cdm_use_tag_meta}")
            accelerator.print(f"  CDM use group id: {hvm_config.cdm_use_group_id}")
        if hvm_config.memory_mode == "dense_global_local":
            accelerator.print("  Dense baseline: raw global refs + raw local parts")
            accelerator.print(f"  Local use tag meta: {hvm_config.cdm_use_tag_meta}")
            accelerator.print(f"  Local use group id: {hvm_config.cdm_use_group_id}")
        if hvm_config.memory_mode == "gme_cdm_edr":
            accelerator.print(f"  EDR d_router: {hvm_config.edr_d_router}")
            accelerator.print(f"  EDR top-k: {hvm_config.edr_top_k}")
            accelerator.print(f"  EDR disable conf: {hvm_config.edr_disable_conf}")
            accelerator.print(f"  EDR disable gist: {hvm_config.edr_disable_gist}")
            accelerator.print(f"  EDR random replace top1: {hvm_config.edr_random_replace_top1}")
            accelerator.print(f"  EDR detail layers: {hvm_config.edr_detail_layer_indices}")
        if hvm_config.inject_mode == "fixed":
            accelerator.print(f"  Inject scale: {hvm_config.inject_scale}")
        accelerator.print(f"  PIM layers: {hvm_config.pim_layer_indices}")
    if args.resume_from:
        accelerator.print(f"  Resuming from: step {global_step}, epoch {start_epoch + 1}")
    if val_dataloader is not None:
        accelerator.print(f"  Val eval every: {args.eval_every} steps")
    if args.shuffle_rag:
        accelerator.print(f"  *** SHUFFLE RAG ABLATION: ref loaded from random donor sample (global, 100% mismatch) ***")
    if args.shuffle_gme:
        accelerator.print(f"  *** SHUFFLE GME ABLATION: GME ref_features from random donor, CDM uses correct refs ***")
    if args.shuffle_cdm:
        accelerator.print(f"  *** SHUFFLE CDM ABLATION: CDM part_features from random donor, GME uses correct refs ***")
    accelerator.print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []  # 每 epoch 重置，用于真实 epoch 平均

        # 恢复时跳过当前 epoch 已完成的 batches
        if epoch == start_epoch and resume_step_in_epoch > 0:
            batches_to_skip = resume_step_in_epoch * args.gradient_accumulation_steps
            active_dataloader = accelerator.skip_first_batches(dataloader, batches_to_skip)
            steps_this_epoch = num_update_steps_per_epoch - resume_step_in_epoch
        else:
            active_dataloader = dataloader
            steps_this_epoch = num_update_steps_per_epoch

        progress_bar = tqdm(
            total=steps_this_epoch,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for batch in active_dataloader:
            with accelerator.accumulate(model):
                # Move to device
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                # Forward pass
                if args.disable_hvm:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                else:
                    ref_features = batch["ref_features"]
                    group_features_list = batch["group_features_list"]
                    ref_text_ids = batch["ref_text_ids"]
                    ref_text_mask = batch["ref_text_mask"]

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        ref_features=ref_features,
                        group_features_list=group_features_list,
                        part_features=batch["part_features"],
                        part_tag_meta=batch["part_tag_meta"],
                        part_group_ids=batch["part_group_ids"],
                        part_mask=batch["part_mask"],
                        ref_text_ids=ref_text_ids,
                        ref_text_mask=ref_text_mask,
                    )

                # Compute loss
                loss = compute_loss(outputs, labels)
                epoch_losses.append(loss.item())

                if args.disable_hvm:
                    # Baseline 模式: loss 来自全冻结模型，无梯度图。
                    # 加一个零值 dummy 项连接可训练参数，满足 DeepSpeed 的 requires_grad 断言。
                    # 梯度为 0，不影响任何参数。
                    dummy_param = next(p for p in model.parameters() if p.requires_grad)
                    loss = loss + 0.0 * dummy_param.sum()

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    next_step = global_step + 1
                    should_log = (next_step % args.log_every == 0)
                    should_print_gates = (next_step % (args.log_every * 10) == 0)

                    diag_stats = None
                    if (should_log or should_print_gates) and not args.disable_hvm:
                        unwrapped = accelerator.unwrap_model(model)
                        diag_stats = collect_hvm_diagnostics(unwrapped)

                    # Gradient clipping (只遍历可训练参数，跳过冻结的 8.6B base model)
                    accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=args.max_grad_norm,
                    )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1

                    # Update progress bar
                    recent_avg = np.mean(epoch_losses[-50:]) if epoch_losses else 0
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "avg": f"{recent_avg:.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    })

                    # ---- Logging ----
                    if should_log and accelerator.is_main_process:
                        avg = np.mean(epoch_losses[-args.log_every:])

                        log_dict = {
                            "train/loss": avg,
                            "train/loss_step": loss.item(),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                        }

                        if diag_stats:
                            log_dict.update(diag_stats)

                        swanlab.log(log_dict, step=global_step)

                    # ---- Val evaluation ----
                    # Ensure ALL ranks participate in evaluation to avoid collective timeout
                    if (val_dataloader is not None
                            and global_step % args.eval_every == 0):
                        val_loss = evaluate_val_loss(
                            model, val_dataloader, accelerator,
                            disable_hvm=args.disable_hvm,
                        )
                        
                        if accelerator.is_main_process:
                            swanlab.log({"val/loss": val_loss}, step=global_step)
                            accelerator.print(
                                f"  Step {global_step} | Val loss: {val_loss:.4f}"
                            )

                    # ---- Save checkpoint ----
                    if global_step % args.save_every == 0 and not args.disable_hvm:
                        # 保存完整训练状态 (所有 rank 参与)
                        ckpt_dir = str(output_dir / f"checkpoint-step-{global_step}")
                        accelerator.save_state(ckpt_dir)
                        if accelerator.is_main_process:
                            # 训练元信息 (global_step, epoch)
                            with open(os.path.join(ckpt_dir, "training_metadata.json"), "w") as f:
                                json.dump({"global_step": global_step, "epoch": epoch}, f, indent=2)
                            # LR scheduler state (未经 accelerator.prepare, 需手动保存)
                            torch.save(lr_scheduler.state_dict(),
                                       os.path.join(ckpt_dir, "lr_scheduler.pt"))
                            # 轻量 HVM-only checkpoint (用于推理/部署)
                            unwrapped = accelerator.unwrap_model(model)
                            unwrapped.save_hvm_checkpoint(
                                str(output_dir / f"hvm_step_{global_step}.pt"))
                        accelerator.wait_for_everyone()

                    # ---- Print gate status ----
                    if should_print_gates and not args.disable_hvm:
                        if diag_stats is None:
                            unwrapped = accelerator.unwrap_model(model)
                            diag_stats = collect_hvm_diagnostics(unwrapped)

                        gate_vals = []
                        inject_ratios = []
                        for pim_idx in range(hvm_config.num_pims):
                            gate_key = f"gate/pim_{pim_idx}_tanh_alpha"
                            scale_key = f"inject/pim_{pim_idx}_scale"
                            ratio_key = f"gate/pim_{pim_idx}_inject_hidden_ratio"
                            if gate_key in diag_stats:
                                gate_vals.append(f"{diag_stats[gate_key]:.4f}")
                            elif scale_key in diag_stats:
                                gate_vals.append(f"fixed={diag_stats[scale_key]:.4f}")
                            if ratio_key in diag_stats:
                                inject_ratios.append(f"{diag_stats[ratio_key]:.4e}")

                        grad_summary_keys = [
                            "grad/gme_input_proj",
                            "grad/pme_input_proj",
                            "grad/pim0_hidden_attn_q",
                            "grad/pim0_hidden_gist_attn_q",
                            "grad/pim0_gate_alpha",
                            "grad/pim_last_gate_alpha",
                        ]
                        grad_summary = ", ".join(
                            f"{k.split('/')[-1]}={diag_stats[k]:.2e}"
                            for k in grad_summary_keys if diag_stats and k in diag_stats
                        )

                        gate_label = "Scales" if hvm_config.inject_mode == "fixed" else "Gates"
                        msg = f"  Step {global_step} | {gate_label}: [{', '.join(gate_vals)}]"
                        if inject_ratios:
                            msg += f" | InjectRatio: [{', '.join(inject_ratios)}]"
                        if grad_summary:
                            msg += f" | Grads: {grad_summary}"
                        accelerator.print(msg)

        progress_bar.close()

        # End of epoch: 记录真实 epoch 平均 loss + val loss
        # evaluate_val_loss 包含 accelerator.gather()，所有 rank 必须参与
        accelerator.wait_for_everyone()
        epoch_avg_loss = np.mean(epoch_losses) if epoch_losses else 0

        val_loss_epoch = None
        if val_dataloader is not None:
            val_loss_epoch = evaluate_val_loss(
                model, val_dataloader, accelerator,
                disable_hvm=args.disable_hvm,
            )

        if accelerator.is_main_process:
            log_dict = {"train/epoch_loss": epoch_avg_loss}
            if val_loss_epoch is not None:
                log_dict["val/epoch_loss"] = val_loss_epoch
                accelerator.print(
                    f"  Epoch {epoch + 1} | Train loss: {epoch_avg_loss:.4f} | Val loss: {val_loss_epoch:.4f}"
                )
            else:
                accelerator.print(
                    f"  Epoch {epoch + 1} | Train loss: {epoch_avg_loss:.4f}"
                )
            swanlab.log(log_dict, step=global_step)

        torch.cuda.empty_cache()

    # Cleanup
    if accelerator.is_main_process:
        swanlab.finish()
    accelerator.print("Training complete!")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="HVM-SVG Training")

    # Model
    parser.add_argument("--model_size", type=str, default="8B", choices=["8B"])
    parser.add_argument("--config_dir", type=str, default="./configs")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Override base_model path (e.g. Qwen2.5-VL) from tokenization.yaml")
    parser.add_argument("--omnisvg_checkpoint", type=str, default=None,
                        help="OmniSVG checkpoint path (None=use default)")

    # HVM architecture
    parser.add_argument("--d_qformer", type=int, default=1024)
    parser.add_argument("--d_pim_inner", type=int, default=512)
    parser.add_argument("--pim_layer_interval", type=int, default=4)
    parser.add_argument("--pim_layer_indices", type=str, default=None,
                        help="Comma-separated decoder layer indices for PIM hooks. "
                             "Supports -1 for last layer, e.g. '-1' or '3,7,11'.")
    parser.add_argument("--memory_mode", type=str, default="full",
                        choices=["full", "gme", "gme_pme", "gme_pme_dual", "gme_pme_hier", "gme_pme_single", "gme_dra", "gme_cdm", "gme_cdm_edr", "dense_global_local"],
                        help="Memory pipeline: full (GME+PME+Text), gme (GME-only), gme_pme (GME+PME shared gate), "
                             "gme_pme_dual (GME+PME dual gate), gme_pme_hier (GME+PME hierarchical fusion + dual gate), "
                             "gme_pme_single (GME+PME hierarchical fusion + single path injection), "
                             "gme_dra (GME + Direct Reference Attention), "
                             "gme_cdm (GME + Complementary Detail Memory), "
                             "gme_cdm_edr (GME + CDM + Execution-aware Detail Router), "
                             "dense_global_local (raw global/local visual tokens direct attention).")
    parser.add_argument("--inject_mode", type=str, default="adaptive", choices=["adaptive", "fixed"],
                        help="Injection mode: adaptive gate (full) or fixed scale (simple).")
    parser.add_argument("--inject_scale", type=float, default=0.1,
                        help="Fixed injection scale for inject_mode=fixed.")
    parser.add_argument("--gate_alpha_init", type=float, default=0.05,
                        help="Initial value for AdaptiveGate base_alpha (default: 0.05)")
    parser.add_argument("--gme_num_queries", type=int, default=32,
                        help="Number of GME QFormer queries (default: 32)")
    parser.add_argument("--part_num_refs", type=int, default=1,
                        help="Number of retrieved refs used by the part-grounded branch (default: 1, Top-1).")
    parser.add_argument("--pme_max_groups", type=int, default=4,
                        help="Maximum number of part groups per sample. "
                             "Use 4 for Top-1 and 12 for Top-3 with 4 groups/reference.")
    parser.add_argument("--delta_ln", action="store_true", default=False,
                        help="Add LayerNorm on delta before gating (stabilize delta scale).")
    parser.add_argument("--dra_d_inner", type=int, default=128,
                        help="DRA ref cross-attn bottleneck dimension (default: 128)")
    parser.add_argument("--dra_n_heads", type=int, default=4,
                        help="DRA ref cross-attn number of heads (default: 4)")
    parser.add_argument("--cdm_num_queries", type=int, default=16,
                        help="CDM number of learnable queries (default: 16)")
    parser.add_argument("--cdm_num_layers", type=int, default=6,
                        help="CDM QFormer number of layers (default: 6)")
    parser.add_argument("--cdm_layout", type=str, default="global", choices=["global", "groupwise"],
                        help="CDM layout: global=all part tokens share one CDM, groupwise=each group runs through a shared CDM independently.")
    parser.add_argument("--cdm_group_queries_per_group", type=int, default=4,
                        help="Number of detail slots produced per group in groupwise CDM (default: 4).")
    parser.add_argument("--cdm_detail_source", type=str, default="ref",
                        choices=["ref", "part"],
                        help="CDM detail source: ref=flattened 3x256 raw ref tokens, "
                             "part=top-k grouped tokens with layout tags.")
    parser.add_argument("--cdm_disable_gist", action="store_true", default=False,
                        help="Disable CDM gist cross-attention while keeping the rest of the pipeline unchanged.")
    parser.add_argument("--cdm_disable_tag_meta", action="store_true", default=False,
                        help="Disable CDM layout tag meta ([cx, cy, w, h, z_start, z_end]) when using part detail source.")
    parser.add_argument("--cdm_disable_group_id", action="store_true", default=False,
                        help="Disable CDM group-id embedding when using part detail source.")
    parser.add_argument("--edr_d_router", type=int, default=256,
                        help="EDR router bottleneck dimension (default: 256)")
    parser.add_argument("--edr_top_k", type=int, default=2,
                        help="EDR sparse top-k detail slots (default: 2)")
    parser.add_argument("--edr_disable_conf", action="store_true", default=False,
                        help="Disable EDR confidence suppression and force conf=1 during detail injection.")
    parser.add_argument("--edr_disable_gist", action="store_true", default=False,
                        help="Disable EDR gist injection and keep only routed detail injection.")
    parser.add_argument("--edr_random_replace_top1", action="store_true", default=False,
                        help="For top-k=1 ablation: replace the selected detail slot with a random valid slot before detail injection.")
    parser.add_argument("--edr_detail_layer_indices", type=str, default=None,
                        help="Comma-separated decoder layer indices where EDR detail path is enabled. "
                             "Defaults to all PIM layers; gist path still runs on every PIM layer.")

    # Data
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/data/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test2",
                        help="Directory containing parquet files referenced in metadata.jsonl")
    parser.add_argument("--hvm_dir", type=str,
                        default="/mnt/data/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed")
    parser.add_argument("--val_data_dir", type=str, default=None,
                        help="Val parquet directory (None=skip val)")
    parser.add_argument("--val_hvm_dir", type=str, default=None,
                        help="Val HVM precomputed directory (None=skip val)")
    parser.add_argument("--eval_every", type=int, default=200,
                        help="Evaluate val loss every N optimizer steps")

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Warmup steps (default: 10%% of total steps)")
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--output_dir", type=str, default="./outputs_hvm")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--swanlab_run_name", type=str, default=None,
                        help="SwanLab experiment name (auto-generated if None)")
    parser.add_argument("--swanlab_mode", type=str, default="local",
                        choices=["cloud", "local", "disabled"],
                        help="SwanLab mode: cloud (需要API key), local (本地), disabled")

    # DataLoader
    parser.add_argument("--num_workers", type=int, default=4)

    # Resume
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Full training checkpoint directory to resume from "
                             "(includes optimizer/scheduler/step state)")
    parser.add_argument("--hvm_checkpoint", type=str, default=None,
                        help="HVM-only weights (.pt) for fine-tuning "
                             "(does NOT restore optimizer/scheduler/step)")

    # Ablation / Baseline
    parser.add_argument("--disable_hvm", action="store_true", default=False,
                        help="Disable HVM injection (baseline: frozen OmniSVG only). "
                             "HVM modules are still created but hooks skip injection.")
    parser.add_argument("--shuffle_rag", action="store_true", default=False,
                        help="Shuffle ref_features within batch (ablation: break RAG correspondence). "
                             "If GME still helps with shuffled refs, the benefit is from extra params, not RAG info.")
    parser.add_argument("--shuffle_gme", action="store_true", default=False,
                        help="Only shuffle GME input (ref_features from random donor). "
                             "CDM still receives correct part features. Isolates GME path contribution.")
    parser.add_argument("--shuffle_cdm", action="store_true", default=False,
                        help="Only shuffle CDM input (part_features from random donor). "
                             "GME still receives correct ref_features. Isolates CDM/EDR path contribution.")

    args = parser.parse_args()

    # 互斥检查: --resume_from 恢复完整训练状态，--hvm_checkpoint 只加载权重
    if args.resume_from and args.hvm_checkpoint:
        parser.error("--resume_from and --hvm_checkpoint are mutually exclusive. "
                     "Use --resume_from for full training resume, "
                     "--hvm_checkpoint for HVM weight initialization only.")

    try:
        args.pim_layer_indices = parse_pim_layer_indices(args.pim_layer_indices, num_decoder_layers=28)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.edr_detail_layer_indices = parse_pim_layer_indices(args.edr_detail_layer_indices, num_decoder_layers=28)
    except ValueError as exc:
        parser.error(str(exc))

    if args.inject_scale <= 0:
        parser.error("--inject_scale must be > 0.")
    if args.part_num_refs <= 0:
        parser.error("--part_num_refs must be > 0.")
    if args.part_num_refs > 3:
        parser.error("--part_num_refs currently supports at most 3 retrieved refs.")
    if args.pme_max_groups <= 0:
        parser.error("--pme_max_groups must be > 0.")
    if args.edr_d_router <= 0:
        parser.error("--edr_d_router must be > 0.")
    if args.edr_top_k <= 0:
        parser.error("--edr_top_k must be > 0.")
    if args.cdm_detail_source == "part" and args.pme_max_groups < args.part_num_refs * 4:
        parser.error(
            "--pme_max_groups is too small for the selected part refs: "
            f"need at least {args.part_num_refs * 4}, got {args.pme_max_groups}."
        )

    if args.memory_mode == "full" and args.inject_mode != "adaptive":
        parser.error("memory_mode='full' currently supports only inject_mode='adaptive'.")
    if args.memory_mode == "gme_pme" and args.inject_mode != "adaptive":
        parser.error("memory_mode='gme_pme' currently supports only inject_mode='adaptive'.")
    if args.memory_mode == "gme_pme_dual" and args.inject_mode != "adaptive":
        parser.error("memory_mode='gme_pme_dual' currently supports only inject_mode='adaptive'.")
    if args.memory_mode == "gme_pme_hier" and args.inject_mode != "adaptive":
        parser.error("memory_mode='gme_pme_hier' currently supports only inject_mode='adaptive'.")
    if args.memory_mode == "gme_pme_single" and args.inject_mode != "adaptive":
        parser.error("memory_mode='gme_pme_single' currently supports only inject_mode='adaptive'.")
    if args.memory_mode == "gme_dra" and args.inject_mode != "adaptive":
        parser.error("memory_mode='gme_dra' currently supports only inject_mode='adaptive'.")
    if args.memory_mode == "gme_cdm" and args.inject_mode != "adaptive":
        parser.error("memory_mode='gme_cdm' currently supports only inject_mode='adaptive'.")
    if args.memory_mode == "gme_cdm_edr" and args.inject_mode != "adaptive":
        parser.error("memory_mode='gme_cdm_edr' currently supports only inject_mode='adaptive'.")
    if args.memory_mode == "dense_global_local" and args.inject_mode != "adaptive":
        parser.error("memory_mode='dense_global_local' currently supports only inject_mode='adaptive'.")
    if args.cdm_disable_gist and args.memory_mode not in ("gme_cdm", "gme_cdm_edr"):
        parser.error("--cdm_disable_gist is only supported when memory_mode is 'gme_cdm' or 'gme_cdm_edr'.")
    if args.cdm_layout == "groupwise" and args.cdm_detail_source != "part":
        parser.error("--cdm_layout groupwise requires --cdm_detail_source part.")
    if args.cdm_layout == "groupwise" and args.cdm_num_queries != args.cdm_group_queries_per_group * args.pme_max_groups:
        parser.error(
            "--cdm_layout groupwise requires "
            "--cdm_num_queries == --cdm_group_queries_per_group * --pme_max_groups "
            f"(got {args.cdm_num_queries} vs {args.cdm_group_queries_per_group}*{args.pme_max_groups})."
        )
    if args.edr_disable_gist and args.memory_mode not in ("gme_cdm_edr", "gme_cdm"):
        parser.error("--edr_disable_gist is only supported when memory_mode is 'gme_cdm_edr' or 'gme_cdm'.")
    if args.edr_random_replace_top1:
        if args.memory_mode != "gme_cdm_edr":
            parser.error("--edr_random_replace_top1 is only supported when memory_mode='gme_cdm_edr'.")
        if args.edr_top_k != 1:
            parser.error("--edr_random_replace_top1 requires --edr_top_k 1.")
    if args.edr_detail_layer_indices is not None:
        if args.memory_mode != "gme_cdm_edr":
            parser.error("--edr_detail_layer_indices is only supported when memory_mode='gme_cdm_edr'.")
        if args.pim_layer_indices is None:
            active_pim_layers = list(range(args.pim_layer_interval - 1, 28, args.pim_layer_interval))
        else:
            active_pim_layers = list(args.pim_layer_indices)
        invalid_detail_layers = [idx for idx in args.edr_detail_layer_indices if idx not in active_pim_layers]
        if invalid_detail_layers:
            parser.error("--edr_detail_layer_indices must be a subset of active PIM layers. "
                         f"Invalid detail layers: {invalid_detail_layers}; active PIM layers: {active_pim_layers}.")

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
