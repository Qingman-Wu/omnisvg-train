#!/usr/bin/env python3
"""
HVM-SVG Inference Script  (Multi-GPU Data Parallel Version)
============================================================
每张卡加载完整模型，独立处理自己那份数据，卡间零通信。

适配 Stage1 架构: memory_mode=gme, inject_mode=fixed, pim_layer=[27]
  - GME-only: 只使用 gist memory (3 ref images → 32 gist tokens)
  - 无 PME: 不需要 group_features
  - SimpleGMEInjectionModule: hidden × gist cross-attention + fixed scale
  - 无 AdaptiveGate: 固定缩放注入

用法:
CUDA_VISIBLE_DEVICES=0,1,2,3,6 python inference/inference_hvm_s1.py \
    --base_model /mnt/a100_1_data/wuqingman/models/Qwen/Qwen2.5-VL-7B-Instruct \
    --omnisvg_checkpoint /mnt/a100_1_data2/wuqingman/models/OmniSVG/OmniSVG1.1_8B \
    --hvm_checkpoint outputs_s1_fixed0.03/hvm_step_8000.pt \
    --sample_indices $(seq 0 999) \
    --output_dir inference/output_s1_fixed0.03_8000step \
    --save_png --save_gt --save_refs \
    --with_baseline --resume


"""

import argparse
import gc
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, AutoProcessor

from decoder import SketchDecoder
from hvm_decoder import HVMSketchDecoder
from hvm_modules import HVMConfig
from hvm_dataset import HVMDataset
from utils.config import OmniSVGConfig, TokenizationConfig, TrainConfig, MODEL_DEFAULTS
from train import load_checkpoint_state_dict, find_checkpoint_file

from tokenizer import SVGTokenizer

# ---------------------------------------------------------------------------
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

EXTRA_CANDIDATES_BUFFER = 4
MIN_SVG_LENGTH = 20
EMPTY_THRESHOLD = 250

SYSTEM_PROMPT = (
    "You are an expert SVG code generator. "
    "Generate precise, valid SVG path commands that accurately represent "
    "the described scene or object."
)

SVG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

TARGET_IMAGE_SIZE = 448
RENDER_SIZE = 512


# ============================================================================
# 多卡工具函数
# ============================================================================

def get_available_gpus() -> List[int]:
    """返回当前进程可见的 GPU 编号列表（考虑 CUDA_VISIBLE_DEVICES）"""
    n = torch.cuda.device_count()
    return list(range(n))


def split_indices(indices: List[int], num_parts: int) -> List[List[int]]:
    """将 indices 尽量均等地切分成 num_parts 份"""
    indices = list(indices)
    size = len(indices)
    base, rem = divmod(size, num_parts)
    parts = []
    start = 0
    for i in range(num_parts):
        end = start + base + (1 if i < rem else 0)
        parts.append(indices[start:end])
        start = end
    return parts


# ============================================================================
# Model loading
# ============================================================================

def load_base_model_only(
    model_size: str,
    config_dir: str = None,
    omnisvg_checkpoint: str = None,
    base_model_override: str = None,
    device: str = "cuda",
):
    if config_dir is None:
        config_dir = os.path.join(PROJECT_ROOT, "configs")

    config = OmniSVGConfig(config_dir=config_dir, model_size=model_size)
    token_config = config.tokenization
    defaults = MODEL_DEFAULTS[model_size]
    base_model_path = base_model_override or token_config.base_model or defaults["base_model"]

    print(f"[{device}][1/2] Loading tokenizer from {base_model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, padding_side="left", trust_remote_code=True, local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(
        base_model_path, padding_side="left", trust_remote_code=True, local_files_only=True
    )
    processor.tokenizer.padding_side = "left"

    print(f"[{device}][2/2] Loading OmniSVG {model_size} base model (NO HVM) ...")
    base_model = SketchDecoder(
        pix_len=6000,
        text_len=800,
        model_path=base_model_path,
        vocab_size=token_config.extended_vocab_size,
        bos_token_id=token_config.bos_token_id,
        eos_token_id=token_config.eos_token_id,
        pad_token_id=token_config.pad_token_id,
    )

    ckpt_path = omnisvg_checkpoint or token_config.checkpoint or defaults["checkpoint"]
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"  [{device}] Loading OmniSVG checkpoint from {ckpt_path}")
        if os.path.isdir(ckpt_path):
            ckpt_file = find_checkpoint_file(ckpt_path)
            state_dict = load_checkpoint_state_dict(ckpt_file) if ckpt_file else None
        else:
            state_dict = load_checkpoint_state_dict(ckpt_path)
        if state_dict:
            missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
            print(f"  [{device}] Loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    base_model = base_model.to(device).eval()
    if hasattr(base_model.transformer, "gradient_checkpointing_disable"):
        base_model.transformer.gradient_checkpointing_disable()

    return base_model, tokenizer, processor, token_config


def load_hvm_model(
    model_size: str,
    hvm_config_path: str,
    hvm_checkpoint_path: str,
    config_dir: str = None,
    omnisvg_checkpoint: str = None,
    base_model_override: str = None,
    device: str = "cuda",
):
    if config_dir is None:
        config_dir = os.path.join(PROJECT_ROOT, "configs")

    config = OmniSVGConfig(config_dir=config_dir, model_size=model_size)
    token_config = config.tokenization
    defaults = MODEL_DEFAULTS[model_size]
    base_model_path = base_model_override or token_config.base_model or defaults["base_model"]

    with open(hvm_config_path, "r") as f:
        hvm_cfg_dict = json.load(f)
    hvm_config = HVMConfig(**{
        k: v for k, v in hvm_cfg_dict.items()
        if k in HVMConfig.__dataclass_fields__
    })

    print(f"[{device}][1/4] Loading tokenizer from {base_model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, padding_side="left", trust_remote_code=True, local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(
        base_model_path, padding_side="left", trust_remote_code=True, local_files_only=True
    )
    processor.tokenizer.padding_side = "left"

    print(f"[{device}][2/4] Loading OmniSVG {model_size} base model ...")
    base_model = SketchDecoder(
        pix_len=6000,
        text_len=800,
        model_path=base_model_path,
        vocab_size=token_config.extended_vocab_size,
        bos_token_id=token_config.bos_token_id,
        eos_token_id=token_config.eos_token_id,
        pad_token_id=token_config.pad_token_id,
    )

    ckpt_path = omnisvg_checkpoint or token_config.checkpoint or defaults["checkpoint"]
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"  [{device}] Loading OmniSVG checkpoint from {ckpt_path}")
        if os.path.isdir(ckpt_path):
            ckpt_file = find_checkpoint_file(ckpt_path)
            state_dict = load_checkpoint_state_dict(ckpt_file) if ckpt_file else None
        else:
            state_dict = load_checkpoint_state_dict(ckpt_path)
        if state_dict:
            missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
            print(f"  [{device}] Loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    print(f"[{device}][3/4] Building HVM model ...")
    model = HVMSketchDecoder(
        base_model=base_model,
        hvm_config=hvm_config,
        tokenizer=tokenizer,
    )

    print(f"[{device}][4/4] Loading HVM checkpoint from {hvm_checkpoint_path} ...")
    model.load_hvm_checkpoint(hvm_checkpoint_path)

    model = model.to(device).eval()
    transformer = model.base_model.transformer
    if hasattr(transformer, "gradient_checkpointing_disable"):
        transformer.gradient_checkpointing_disable()

    return model, tokenizer, processor, token_config, hvm_config


# ============================================================================
# Input preparation
# ============================================================================

def prepare_text_inputs(text, processor, token_config, device="cuda"):
    instruction = (
        f"Generate an SVG illustration for: {text}\n\n"
        "Requirements:\n"
        "- Create complete SVG path commands\n"
        "- Include proper coordinates and colors\n"
        "- Maintain visual clarity and composition"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": instruction}]},
    ]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input], padding=True, truncation=True, return_tensors="pt"
    )

    bos = torch.full((1, 1), token_config.bos_token_id, dtype=torch.long)
    input_ids = torch.cat([inputs["input_ids"], bos], dim=1).to(device)
    attention_mask = torch.cat(
        [inputs["attention_mask"], torch.ones(1, 1, dtype=torch.long)], dim=1
    ).to(device)
    return input_ids, attention_mask


def set_hvm_memory(model, ref_features, group_features, ref_text,
                   tokenizer, hvm_config, device="cuda"):
    """将参考特征注入 HVM 模型。

    Stage1 (gme + fixed): 只需要 gist_feats，不需要 PME 和 text_feats。
    Full mode: 需要 gist_feats + part_feats + text_feats。
    """
    hvm_dtype = next(model.gme.parameters()).dtype

    # GME: 3 ref images → 32 gist tokens
    ref_feat_tensor = torch.stack(ref_features).unsqueeze(0).to(device=device, dtype=hvm_dtype)
    model._gist_feats = model.gme(ref_feat_tensor)

    # PME: 仅在 full mode 下使用
    if model.pme is not None:
        gfl_on_device = [[gf.to(device=device, dtype=hvm_dtype) for gf in group_features]]
        model._part_feats, model._part_mask = model.pme(gfl_on_device)
    else:
        # gme + fixed 模式: 不需要 part_feats
        model._part_feats = None
        model._part_mask = None

    # Text feats: 仅在 full mode 下使用
    if hvm_config.memory_mode == "gme" and hvm_config.inject_mode == "fixed":
        model._text_feats = None
        model._text_mask = None
    else:
        ref_text_encoded = tokenizer(
            [ref_text], padding=True, truncation=True,
            max_length=hvm_config.ref_text_max_length, return_tensors="pt",
        )
        ref_text_ids = ref_text_encoded["input_ids"].to(device)
        ref_text_mask = ref_text_encoded["attention_mask"].to(device)
        model._text_mask = ref_text_mask.to(dtype=torch.bool)
        model._text_feats = model._prepare_text_feats(ref_text_ids, model._text_mask)


def clear_hvm_memory(model):
    model._gist_feats = None
    model._part_feats = None
    model._text_feats = None
    model._part_mask = None
    model._text_mask = None


# ============================================================================
# Generation & decoding
# ============================================================================

@torch.no_grad()
def generate_svg(
    transformer_model,
    input_ids, attention_mask,
    token_config, svg_tokenizer,
    max_new_tokens=3000, temperature=0.5, top_p=0.90,
    top_k=50, repetition_penalty=1.05, num_return_sequences=1,
):
    gen_cfg = dict(
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=int(top_k),
        repetition_penalty=repetition_penalty,
        eos_token_id=token_config.eos_token_id,
        pad_token_id=token_config.pad_token_id,
        bos_token_id=token_config.bos_token_id,
        use_cache=True,
    )

    results = transformer_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_cfg,
    )

    input_len = input_ids.shape[1]
    generated_ids_batch = results[:, input_len:]

    candidates = []
    BLACK_COLOR_TOKEN = svg_tokenizer.COLOR_TOKEN_START_RAW + 2

    for i in range(generated_ids_batch.shape[0]):
        try:
            current_ids = generated_ids_batch[i:i+1].cpu()
            fake_wrapper = torch.cat([
                torch.full((1, 1), token_config.bos_token_id),
                current_ids,
                torch.full((1, 1), token_config.eos_token_id),
            ], dim=1)

            generated_xy = svg_tokenizer.process_generated_tokens(fake_wrapper)
            if len(generated_xy) == 0:
                continue

            svg_tensors, color_tensors = svg_tokenizer.raster_svg(generated_xy)
            if not svg_tensors or not svg_tensors[0]:
                continue

            num_paths = len(svg_tensors[0])
            while len(color_tensors) < num_paths:
                color_tensors.append(BLACK_COLOR_TOKEN)

            svg_obj = svg_tokenizer.apply_colors_to_svg(svg_tensors[0], color_tensors)
            svg_str = svg_obj.to_str()

            if "width=" not in svg_str:
                svg_str = svg_str.replace(
                    "<svg",
                    f'<svg width="{TARGET_IMAGE_SIZE}" height="{TARGET_IMAGE_SIZE}"',
                    1,
                )

            candidates.append({
                "svg_str": svg_str,
                "num_paths": num_paths,
                "tokens": current_ids.squeeze(0).tolist(),
            })
        except Exception as e:
            print(f"  Candidate {i} decode error: {e}")
            continue

    return candidates


def render_svg_to_image(svg_str: str, size: int = RENDER_SIZE):
    try:
        import cairosvg
        from PIL import Image

        png_data = cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            output_width=size, output_height=size,
        )
        img = Image.open(io.BytesIO(png_data)).convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    except Exception as e:
        print(f"  Render error: {e}")
        return None


def validate_candidate(svg_str: str) -> bool:
    if not svg_str or len(svg_str) < MIN_SVG_LENGTH:
        return False
    if "<svg" not in svg_str:
        return False
    img = render_svg_to_image(svg_str)
    if img is None:
        return False
    img_array = np.array(img)
    if img_array.mean() > EMPTY_THRESHOLD:
        return False
    return True


# ============================================================================
# GT SVG & reference data helpers
# ============================================================================

def load_gt_svg(dataset: HVMDataset, dataset_index: int) -> Optional[str]:
    try:
        idx = dataset.valid_indices[dataset_index]
        meta = dataset.idx_to_meta[idx]
        table = dataset._get_parquet_table(meta["parquet_file"])
        row = meta["parquet_row"]
        return table.column("svg")[row].as_py()
    except Exception as e:
        print(f"  Warning: failed to load GT SVG: {e}")
        return None


def load_ref_info(dataset: HVMDataset, dataset_index: int) -> Dict[str, Any]:
    try:
        idx = dataset.valid_indices[dataset_index]
        rag = dataset.idx_to_rag[idx]
        ref_indices = rag["ref_indices"]
        ref_scores = rag.get("ref_scores", [])

        refs = []
        for ri_pos, ri in enumerate(ref_indices):
            ref_meta = dataset.idx_to_meta.get(ri, {})
            ref_entry = {
                "ref_index": ri,
                "score": ref_scores[ri_pos] if ri_pos < len(ref_scores) else None,
                "description": ref_meta.get("description", ""),
                "detail": ref_meta.get("detail", ""),
            }
            try:
                ref_table = dataset._get_parquet_table(ref_meta["parquet_file"])
                ref_row = ref_meta["parquet_row"]
                ref_entry["svg"] = ref_table.column("svg")[ref_row].as_py()
            except Exception:
                ref_entry["svg"] = None
            refs.append(ref_entry)

        return {"ref_indices": ref_indices, "refs": refs}
    except Exception as e:
        print(f"  Warning: failed to load ref info: {e}")
        return {"ref_indices": [], "refs": []}


def save_gt(output_dir: Path, idx: int, gt_svg: str, save_png: bool):
    base = f"sample_{idx:04d}_gt"
    svg_str = gt_svg
    if "width=" not in svg_str:
        svg_str = svg_str.replace(
            "<svg",
            f'<svg width="{TARGET_IMAGE_SIZE}" height="{TARGET_IMAGE_SIZE}"',
            1,
        )
    (output_dir / f"{base}.svg").write_text(svg_str, encoding="utf-8")
    print(f"  -> {base}.svg (GT)")
    if save_png:
        img = render_svg_to_image(svg_str)
        if img is not None:
            img.save(str(output_dir / f"{base}.png"))


def save_refs(output_dir: Path, idx: int, ref_info: Dict, ref_text: str, save_png: bool):
    base = f"sample_{idx:04d}_refs"
    meta_for_json = {
        "ref_text": ref_text,
        "refs": [
            {
                "ref_index": r["ref_index"],
                "score": r["score"],
                "description": r["description"],
                "detail": r["detail"][:200] if r.get("detail") else "",
            }
            for r in ref_info["refs"]
        ],
    }
    (output_dir / f"{base}.json").write_text(
        json.dumps(meta_for_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  -> {base}.json (ref metadata)")

    for ri, ref in enumerate(ref_info["refs"]):
        ref_svg = ref.get("svg")
        if not ref_svg:
            continue
        ref_base = f"sample_{idx:04d}_ref{ri}"
        if "width=" not in ref_svg:
            ref_svg = ref_svg.replace(
                "<svg",
                f'<svg width="{TARGET_IMAGE_SIZE}" height="{TARGET_IMAGE_SIZE}"',
                1,
            )
        (output_dir / f"{ref_base}.svg").write_text(ref_svg, encoding="utf-8")
        if save_png:
            img = render_svg_to_image(ref_svg)
            if img is not None:
                img.save(str(output_dir / f"{ref_base}.png"))

    print(f"  -> {len(ref_info['refs'])} reference SVGs saved")


# ============================================================================
# 单卡推理主逻辑（每个子进程都跑这个函数）
# ============================================================================

def run_on_single_gpu(
    local_rank: int,
    gpu_id: int,
    sample_indices: List[int],
    args: argparse.Namespace,
):
    device = f"cuda:{gpu_id}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not sample_indices:
        print(f"[GPU {gpu_id}] No samples assigned, exiting.")
        return

    print(f"\n[GPU {gpu_id}] Assigned {len(sample_indices)} samples: "
          f"{sample_indices[0]} ~ {sample_indices[-1]}")

    # --- Load model ---
    hvm_config = None
    if args.no_hvm:
        base_model, tokenizer, processor, token_config = load_base_model_only(
            model_size=args.model_size,
            config_dir=args.config_dir,
            omnisvg_checkpoint=args.omnisvg_checkpoint,
            base_model_override=args.base_model,
            device=device,
        )
        transformer_for_generate = base_model.transformer
        hvm_model = None
    else:
        hvm_model, tokenizer, processor, token_config, hvm_config = load_hvm_model(
            model_size=args.model_size,
            hvm_config_path=args.hvm_config,
            hvm_checkpoint_path=args.hvm_checkpoint,
            config_dir=args.config_dir,
            omnisvg_checkpoint=args.omnisvg_checkpoint,
            base_model_override=args.base_model,
            device=device,
        )
        transformer_for_generate = hvm_model.base_model.transformer

    # --- SVG tokenizer ---
    svg_tokenizer = SVGTokenizer(SVG_CONFIG_PATH, model_size=args.model_size)

    # --- Dataset ---
    dataset = HVMDataset(
        data_dir=args.data_dir,
        hvm_dir=args.hvm_dir,
        token_config=token_config,
        train_config=TrainConfig(model_size=args.model_size),
        max_len=6000,
    )
    print(f"[GPU {gpu_id}] Dataset size: {len(dataset)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_ok = 0
    total_fail = 0
    total_skipped = 0

    for idx in sample_indices:
        if idx >= len(dataset):
            print(f"[GPU {gpu_id}] Skip sample {idx}: out of range")
            continue

        # --resume: 检查输出文件是否已存在
        if args.resume:
            if args.no_hvm:
                expected_tags = ["base"]
            elif args.with_baseline:
                expected_tags = ["hvm", "base"]
            else:
                expected_tags = ["hvm"]
            all_exist = all(
                (output_dir / f"sample_{idx:04d}_{tag}.svg").exists()
                for tag in expected_tags
            )
            if all_exist:
                total_skipped += 1
                continue

        sample = dataset[idx]
        text = sample["text"]
        print(f"\n[GPU {gpu_id}] {'='*60}")
        print(f"[GPU {gpu_id}] Sample {idx}: {text[:80]}{'...' if len(text) > 80 else ''}")
        if not args.no_hvm:
            print(f"[GPU {gpu_id}]   ref_features  : {len(sample['ref_features'])}")
            print(f"[GPU {gpu_id}]   group_features: {len(sample['ref_best_group_features'])}")
            print(f"[GPU {gpu_id}]   ref_text      : {sample['ref_text'][:60]}...")

        t0 = time.time()

        # 1) 保存 GT SVG
        if args.save_gt:
            gt_svg = load_gt_svg(dataset, idx)
            if gt_svg:
                save_gt(output_dir, idx, gt_svg, args.save_png)

        # 2) 保存参考信息
        if args.save_refs and not args.no_hvm:
            ref_info = load_ref_info(dataset, idx)
            save_refs(output_dir, idx, ref_info, sample["ref_text"], args.save_png)

        # 3) 准备输入
        input_ids, attention_mask = prepare_text_inputs(
            text, processor, token_config, device
        )

        # 4) 决定运行哪些模式
        if args.no_hvm:
            run_modes = [("base", False)]
        elif args.with_baseline:
            run_modes = [("hvm", True), ("base", False)]
        else:
            run_modes = [("hvm", True)]

        sample_ok = False
        for tag, use_hvm in run_modes:
            t_gen = time.time()

            if use_hvm:
                set_hvm_memory(
                    hvm_model,
                    ref_features=sample["ref_features"],
                    group_features=sample["ref_best_group_features"],
                    ref_text=sample["ref_text"],
                    tokenizer=tokenizer,
                    hvm_config=hvm_config,
                    device=device,
                )
            elif hvm_model is not None:
                clear_hvm_memory(hvm_model)

            actual_num = args.num_candidates + EXTRA_CANDIDATES_BUFFER
            candidates = generate_svg(
                transformer_for_generate,
                input_ids, attention_mask,
                token_config, svg_tokenizer,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                num_return_sequences=actual_num,
            )

            if use_hvm and hvm_model is not None:
                clear_hvm_memory(hvm_model)

            gen_elapsed = time.time() - t_gen

            # 验证候选
            if candidates and not args.no_validate:
                valid = []
                for ci, cand in enumerate(candidates):
                    if validate_candidate(cand["svg_str"]):
                        valid.append(cand)
                        if len(valid) >= args.num_candidates:
                            break
                    else:
                        print(f"[GPU {gpu_id}]   [{tag}] Candidate {ci}: invalid")
                rejected = len(candidates) - len(valid)
                candidates = valid
                if rejected > 0:
                    print(f"[GPU {gpu_id}]   [{tag}] Validated: {len(valid)} valid, {rejected} rejected")

            if candidates:
                sample_ok = True
                print(f"[GPU {gpu_id}]   [{tag}] {len(candidates)} candidate(s) in {gen_elapsed:.1f}s")

                for ci, cand in enumerate(candidates):
                    suffix = f"_c{ci}" if len(candidates) > 1 else ""
                    base = f"sample_{idx:04d}_{tag}{suffix}"

                    svg_path = output_dir / f"{base}.svg"
                    svg_path.write_text(cand["svg_str"], encoding="utf-8")
                    print(f"[GPU {gpu_id}]     -> {svg_path.name} ({cand['num_paths']} paths)")

                    if args.save_png:
                        img = render_svg_to_image(cand["svg_str"])
                        if img is not None:
                            img.save(str(output_dir / f"{base}.png"))

                    if args.save_tokens:
                        tok_path = output_dir / f"{base}_tokens.json"
                        tok_path.write_text(json.dumps({
                            "text": text,
                            "ref_text": sample.get("ref_text", ""),
                            "tokens": cand["tokens"],
                            "num_tokens": len(cand["tokens"]),
                        }, indent=2), encoding="utf-8")
            else:
                print(f"[GPU {gpu_id}]   [{tag}] FAILED ({gen_elapsed:.1f}s)")

        # 保存 prompt 文本
        (output_dir / f"sample_{idx:04d}.txt").write_text(text, encoding="utf-8")

        if sample_ok:
            total_ok += 1
        else:
            total_fail += 1

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[GPU {gpu_id}] Done: success={total_ok}, fail={total_fail}, "
          f"skipped={total_skipped}, total={len(sample_indices)}")


# ============================================================================
# 多进程入口
# ============================================================================

def worker(local_rank: int, args: argparse.Namespace, index_splits: List[List[int]]):
    run_on_single_gpu(
        local_rank=local_rank,
        gpu_id=local_rank,
        sample_indices=index_splits[local_rank],
        args=args,
    )


# ============================================================================
# Args
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="HVM-SVG Inference (Multi-GPU)")

    # model
    p.add_argument("--model_size", type=str, default="8B", choices=["4B", "8B"])
    p.add_argument("--config_dir", type=str, default=None)
    p.add_argument("--base_model", type=str, default=None,
                   help="Qwen base model 路径，覆盖 tokenization.yaml 中的 base_model")
    p.add_argument("--hvm_checkpoint", type=str, required=True)
    p.add_argument("--hvm_config", type=str, default=None)
    p.add_argument("--omnisvg_checkpoint", type=str, default=None,
                   help="OmniSVG checkpoint 路径，覆盖 tokenization.yaml 中的 checkpoint")

    # mode
    p.add_argument("--no_hvm", action="store_true", default=False)
    p.add_argument("--with_baseline", action="store_true", default=False)

    # data
    p.add_argument("--data_dir", type=str,
                   default="/mnt/a100_1_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test")
    p.add_argument("--hvm_dir", type=str,
                   default="/mnt/a100_1_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed")
    p.add_argument("--sample_indices", type=int, nargs="+", default=[0])

    # multi-gpu
    p.add_argument("--num_gpus", type=int, default=None,
                   help="使用几张卡。默认=所有 CUDA_VISIBLE_DEVICES 可见的卡数。")

    # generation
    p.add_argument("--max_new_tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=0.90)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--num_candidates", type=int, default=5)
    p.add_argument("--no_validate", action="store_true", default=False)

    # output
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--save_png", action="store_true", default=False)
    p.add_argument("--save_tokens", action="store_true", default=False)
    p.add_argument("--save_gt", action="store_true", default=False)
    p.add_argument("--save_refs", action="store_true", default=False)
    p.add_argument("--resume", action="store_true", default=False)

    args = p.parse_args()

    # auto-detect hvm_config
    if args.hvm_config is None:
        ckpt_dir = Path(args.hvm_checkpoint).parent
        candidate = ckpt_dir / "hvm_model_config.json"
        if candidate.exists():
            args.hvm_config = str(candidate)
        elif not args.no_hvm:
            raise FileNotFoundError(
                f"Cannot find hvm_model_config.json in {ckpt_dir}. "
                "Please specify --hvm_config explicitly."
            )

    return args


# ============================================================================
# Main
# ============================================================================

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    available_gpus = get_available_gpus()
    num_gpus_available = len(available_gpus)

    if num_gpus_available == 0:
        raise RuntimeError("No CUDA GPU detected! Check CUDA_VISIBLE_DEVICES.")

    if args.num_gpus is not None:
        num_gpus = min(args.num_gpus, num_gpus_available)
    else:
        num_gpus = num_gpus_available

    mode_str = "BASELINE" if args.no_hvm else ("HVM + BASELINE" if args.with_baseline else "HVM")

    print("=" * 70)
    print(f"HVM-SVG Inference  [{mode_str}]  --  Multi-GPU Data Parallel")
    print("=" * 70)
    print(f"  Model size       : {args.model_size}")
    print(f"  Mode             : {mode_str}")
    if not args.no_hvm:
        print(f"  HVM checkpoint   : {args.hvm_checkpoint}")
    print(f"  Available GPUs   : {num_gpus_available}  (using {num_gpus})")
    print(f"  Total samples    : {len(args.sample_indices)}")
    print(f"  Output dir       : {args.output_dir}")
    print("=" * 70)

    index_splits = split_indices(args.sample_indices, num_gpus)
    for i, split in enumerate(index_splits):
        print(f"  GPU {i}: {len(split)} samples"
              + (f"  [{split[0]}~{split[-1]}]" if split else "  [empty]"))
    print("=" * 70)

    if num_gpus == 1:
        run_on_single_gpu(
            local_rank=0,
            gpu_id=0,
            sample_indices=index_splits[0],
            args=args,
        )
    else:
        mp.start_processes(
            worker,
            args=(args, index_splits),
            nprocs=num_gpus,
            start_method="spawn",
        )

    print(f"\n{'='*70}")
    print(f"All done! Results saved to: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
