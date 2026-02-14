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

    # 多 GPU 分布式训练
    accelerate launch --num_processes 8 train_hvm.py --batch_size 2

    # 从 HVM checkpoint 恢复
    accelerate launch train_hvm.py --hvm_checkpoint ./outputs_hvm/hvm_step_10000.pt
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import math
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

# Checkpoint paths (same as train.py)
MODEL_DEFAULTS = {
    "8B": {
        "base_model": "/mnt/data/wuqingman/models/Qwen/Qwen2.5-VL-7B-Instruct",
        "checkpoint": "/mnt/data/wuqingman/models/OmniSVG/OmniSVG1.1_8B",
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
    hvm_config = HVMConfig(
        d_model=3584,
        d_vision=1280,
        d_qformer=args.d_qformer,
        d_pim_inner=args.d_pim_inner,
        pim_layer_interval=args.pim_layer_interval,
        num_decoder_layers=28,
    )

    # ---- Accelerator ----
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
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

    # ---- Scheduler ----
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    total_training_steps = num_update_steps_per_epoch * args.epochs

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )

    # ---- Prepare with Accelerator ----
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # ---- Output dir & SwanLab ----
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

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
                "trainable_params_M": sum(p.numel() for p in trainable_params) / 1e6,
            },
            logdir=str(output_dir / "swanlog"),
            mode=args.swanlab_mode,
        )

        # 保存配置
        with open(output_dir / "hvm_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        with open(output_dir / "hvm_model_config.json", "w") as f:
            json.dump(hvm_config.__dict__, f, indent=2, default=str)

    # ---- Training Loop ----
    accelerator.print("=" * 60)
    accelerator.print("Starting HVM-SVG Training")
    accelerator.print(f"  Epochs: {args.epochs}")
    accelerator.print(f"  Batch size: {args.batch_size}")
    accelerator.print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    accelerator.print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
    accelerator.print(f"  Learning rate: {args.learning_rate}")
    accelerator.print(f"  Total steps: {total_training_steps}")
    accelerator.print(f"  Warmup steps: {args.warmup_steps}")
    accelerator.print(f"  PIM layers: {hvm_config.pim_layer_indices}")
    accelerator.print("=" * 60)

    global_step = 0
    running_losses = []

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for batch in dataloader:
            with accelerator.accumulate(model):
                # Move to device
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                ref_features = batch["ref_features"]
                ref_best_feature = batch["ref_best_feature"]
                groups_bbox_feature = batch["groups_bbox_feature"]
                ref_text_ids = batch["ref_text_ids"]
                ref_text_mask = batch["ref_text_mask"]

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ref_features=ref_features,
                    ref_best_feature=ref_best_feature,
                    groups_bbox_feature=groups_bbox_feature,
                    ref_text_ids=ref_text_ids,
                    ref_text_mask=ref_text_mask,
                )

                # Compute loss
                loss = compute_loss(outputs, labels)
                running_losses.append(loss.item())

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # Gradient clipping
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        max_norm=args.max_grad_norm,
                    )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1

                    # Update progress bar
                    avg_loss = np.mean(running_losses[-50:]) if running_losses else 0
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "avg": f"{avg_loss:.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    })

                    # ---- Logging ----
                    if global_step % args.log_every == 0 and accelerator.is_main_process:
                        avg = np.mean(running_losses[-args.log_every:])

                        log_dict = {
                            "train/loss": avg,
                            "train/loss_step": loss.item(),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                        }

                        # Log gate values
                        unwrapped = accelerator.unwrap_model(model)
                        for pim_idx, pim in enumerate(unwrapped.pims):
                            alpha = pim.gate.base_alpha.item()
                            log_dict[f"gate/pim_{pim_idx}_tanh_alpha"] = torch.tanh(torch.tensor(alpha)).item()

                        swanlab.log(log_dict, step=global_step)
                        running_losses = running_losses[-100:]

                    # ---- Save checkpoint ----
                    if global_step % args.save_every == 0:
                        if accelerator.is_main_process:
                            unwrapped = accelerator.unwrap_model(model)
                            save_path = str(output_dir / f"hvm_step_{global_step}.pt")
                            unwrapped.save_hvm_checkpoint(save_path)

                    # ---- Print gate status ----
                    if global_step % (args.log_every * 10) == 0:
                        unwrapped = accelerator.unwrap_model(model)
                        gate_vals = []
                        for pim in unwrapped.pims:
                            gate_vals.append(f"{torch.tanh(pim.gate.base_alpha).item():.4f}")
                        accelerator.print(
                            f"  Step {global_step} | Gates: [{', '.join(gate_vals)}]"
                        )

        progress_bar.close()

        # End of epoch save
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            save_path = str(output_dir / f"hvm_epoch_{epoch + 1}.pt")
            unwrapped.save_hvm_checkpoint(save_path)

            # Log epoch-level metrics
            epoch_avg_loss = np.mean(running_losses) if running_losses else 0
            swanlab.log({"train/epoch_loss": epoch_avg_loss}, step=global_step)

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
    parser.add_argument("--omnisvg_checkpoint", type=str, default=None,
                        help="OmniSVG checkpoint path (None=use default)")

    # HVM architecture
    parser.add_argument("--d_qformer", type=int, default=1024)
    parser.add_argument("--d_pim_inner", type=int, default=512)
    parser.add_argument("--pim_layer_interval", type=int, default=4)

    # Data
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/data/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test2",
                        help="Directory containing parquet files referenced in metadata.jsonl")
    parser.add_argument("--hvm_dir", type=str,
                        default="/mnt/data/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed")

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=500)
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
    parser.add_argument("--hvm_checkpoint", type=str, default=None,
                        help="HVM checkpoint to resume from")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
