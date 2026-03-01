#!/usr/bin/env python3
"""
HVM-SVG Inference Script
========================
参考原始 OmniSVG 推理代码，使用 transformer.generate() + SVGTokenizer 完成推理。

功能:
  - HVM 推理:  OmniSVG + HVM modules (默认)
  - Baseline:  仅 OmniSVG (--no_hvm)
  - 保存 GT:   从 parquet 读取原始 SVG 并保存 (--save_gt)
  - 保存参考:  保存 RAG 检索到的参考文本/参考图 (--save_refs)

用法:
    # HVM 推理 + 保存 GT + 参考 + PNG
    CUDA_VISIBLE_DEVICES=0 python inference_hvm.py \\
        --hvm_checkpoint ../outputs_hvm/hvm_step_1000.pt \\
        --sample_indices 0 1 2 3 4 \\
        --output_dir ./output \\
        --save_png --save_gt --save_refs

    # Baseline 对比（仅原始 OmniSVG，不加载 HVM）
    CUDA_VISIBLE_DEVICES=0 python inference_hvm.py \\
        --hvm_checkpoint ../outputs_hvm/hvm_step_1000.pt \\
        --sample_indices 0 1 2 3 4 \\
        --output_dir ./output_baseline \\
        --save_png --save_gt \\
        --no_hvm

CUDA_VISIBLE_DEVICES=4 python inference/inference_hvm.py \
    --hvm_checkpoint outputs_hvm_2026_02_23_00_18/hvm_step_4000.pt \
    --sample_indices $(seq 0 999) \
    --output_dir inference/output_hvm_2026_02_23_00_18-2000222 \
    --save_png --save_gt --save_refs \
    --with_baseline \
    --resume
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
# Model loading
# ============================================================================

def load_base_model_only(
    model_size: str,
    config_dir: str = None,
    omnisvg_checkpoint: str = None,
    device: str = "cuda",
):
    """仅加载原始 OmniSVG 模型（Baseline 对比用）"""

    if config_dir is None:
        config_dir = os.path.join(PROJECT_ROOT, "configs")

    config = OmniSVGConfig(config_dir=config_dir, model_size=model_size)
    token_config = config.tokenization
    defaults = MODEL_DEFAULTS[model_size]
    base_model_path = token_config.base_model or defaults["base_model"]

    print(f"[1/2] Loading tokenizer from {base_model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="left", trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(base_model_path, padding_side="left", trust_remote_code=True, local_files_only=True)
    processor.tokenizer.padding_side = "left"

    print(f"[2/2] Loading OmniSVG {model_size} base model (NO HVM) ...")
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
        print(f"  Loading OmniSVG checkpoint from {ckpt_path}")
        if os.path.isdir(ckpt_path):
            ckpt_file = find_checkpoint_file(ckpt_path)
            state_dict = load_checkpoint_state_dict(ckpt_file) if ckpt_file else None
        else:
            state_dict = load_checkpoint_state_dict(ckpt_path)
        if state_dict:
            missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    base_model = base_model.to(device).eval()
    if hasattr(base_model.transformer, "gradient_checkpointing_disable"):
        base_model.transformer.gradient_checkpointing_disable()

    print(f"\nBaseline model loaded on {device}")
    return base_model, tokenizer, processor, token_config


def load_hvm_model(
    model_size: str,
    hvm_config_path: str,
    hvm_checkpoint_path: str,
    config_dir: str = None,
    omnisvg_checkpoint: str = None,
    device: str = "cuda",
):
    """加载完整的 HVM 模型（OmniSVG base + HVM modules）"""

    if config_dir is None:
        config_dir = os.path.join(PROJECT_ROOT, "configs")

    config = OmniSVGConfig(config_dir=config_dir, model_size=model_size)
    token_config = config.tokenization
    defaults = MODEL_DEFAULTS[model_size]
    base_model_path = token_config.base_model or defaults["base_model"]

    with open(hvm_config_path, "r") as f:
        hvm_cfg_dict = json.load(f)
    hvm_config = HVMConfig(**{
        k: v for k, v in hvm_cfg_dict.items()
        if k in HVMConfig.__dataclass_fields__
    })

    print(f"[1/4] Loading tokenizer from {base_model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="left", trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(base_model_path, padding_side="left", trust_remote_code=True, local_files_only=True)
    processor.tokenizer.padding_side = "left"

    print(f"[2/4] Loading OmniSVG {model_size} base model ...")
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
        print(f"  Loading OmniSVG checkpoint from {ckpt_path}")
        if os.path.isdir(ckpt_path):
            ckpt_file = find_checkpoint_file(ckpt_path)
            state_dict = load_checkpoint_state_dict(ckpt_file) if ckpt_file else None
        else:
            state_dict = load_checkpoint_state_dict(ckpt_path)
        if state_dict:
            missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    print(f"[3/4] Building HVM model ...")
    model = HVMSketchDecoder(
        base_model=base_model,
        hvm_config=hvm_config,
        tokenizer=tokenizer,
    )

    print(f"[4/4] Loading HVM checkpoint from {hvm_checkpoint_path} ...")
    model.load_hvm_checkpoint(hvm_checkpoint_path)

    model = model.to(device).eval()
    transformer = model.base_model.transformer
    if hasattr(transformer, "gradient_checkpointing_disable"):
        transformer.gradient_checkpointing_disable()

    print(f"\nHVM model loaded on {device}, dtype={DTYPE}")
    return model, tokenizer, processor, token_config, hvm_config


# ============================================================================
# Input preparation
# ============================================================================

def prepare_text_inputs(text, processor, token_config, device="cuda"):
    """构造文本 prompt 的 input_ids（chat template + BOS）"""
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
    """
    调用 transformer.generate() 生成 SVG token，再用 SVGTokenizer 解码。

    Args:
        transformer_model: Qwen2_5_VLForConditionalGeneration 实例
                           (HVM 模式: model.base_model.transformer;
                            Baseline 模式: base_model.transformer)
    """
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
    """用 cairosvg 渲染 SVG → PIL Image"""
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
    """Render SVG and check if the image is non-empty and valid."""
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
    """从 parquet 读取原始 GT SVG 字符串"""
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
    """加载 RAG 参考样本的详细信息（描述、SVG 等）"""
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

            # 尝试读取参考 SVG
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
    """保存 GT SVG 及可选 PNG"""
    base = f"sample_{idx:04d}_gt"

    # 确保 SVG 有 width/height
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
    """保存参考样本信息"""
    base = f"sample_{idx:04d}_refs"

    # JSON 元信息（不包含大段 SVG 字符串，保持可读）
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

    # 保存每张参考的 SVG / PNG
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
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="HVM-SVG Inference")

    # model
    p.add_argument("--model_size", type=str, default="8B", choices=["4B", "8B"])
    p.add_argument("--config_dir", type=str, default=None)
    p.add_argument("--hvm_checkpoint", type=str, required=True,
                    help="Path to hvm_step_*.pt")
    p.add_argument("--hvm_config", type=str, default=None,
                    help="Path to hvm_model_config.json (auto-detect)")
    p.add_argument("--omnisvg_checkpoint", type=str, default=None,
                    help="Override OmniSVG base checkpoint path")

    # mode
    p.add_argument("--no_hvm", action="store_true", default=False,
                    help="Baseline mode: only use OmniSVG without HVM modules")
    p.add_argument("--with_baseline", action="store_true", default=False,
                    help="Also run baseline (pure OmniSVG) for each sample in the same pass")

    # data
    p.add_argument("--data_dir", type=str,
                    default="/mnt/a100_1_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test")
    p.add_argument("--hvm_dir", type=str,
                    default="/mnt/a100_1_data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed")
    p.add_argument("--sample_indices", type=int, nargs="+", default=[0],
                    help="Dataset sample indices to run inference on")

    # generation
    p.add_argument("--max_new_tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=0.90)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--num_candidates", type=int, default=5,
                    help="Num candidates to generate (with extra buffer for validation)")
    p.add_argument("--no_validate", action="store_true", default=False,
                    help="Skip candidate validation (render + empty check)")

    # output
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--save_png", action="store_true", default=False)
    p.add_argument("--save_tokens", action="store_true", default=False)
    p.add_argument("--save_gt", action="store_true", default=False,
                    help="Save ground-truth SVG from parquet for comparison")
    p.add_argument("--save_refs", action="store_true", default=False,
                    help="Save RAG reference images/text for comparison")
    p.add_argument("--resume", action="store_true", default=False,
                    help="Skip samples whose output SVGs already exist in output_dir")
    p.add_argument("--device", type=str, default="cuda")

    args = p.parse_args()

    # auto-detect hvm_config (even in --no_hvm mode, still needed for dataset)
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


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    if args.no_hvm:
        mode_str = "BASELINE (OmniSVG only)"
    elif args.with_baseline:
        mode_str = "HVM + BASELINE"
    else:
        mode_str = "HVM"

    print("=" * 70)
    print(f"HVM-SVG Inference  [{mode_str}]")
    print("=" * 70)
    print(f"  Model size     : {args.model_size}")
    print(f"  Mode           : {mode_str}")
    if not args.no_hvm:
        print(f"  HVM checkpoint : {args.hvm_checkpoint}")
    print(f"  Data dir       : {args.data_dir}")
    print(f"  Sample indices : {args.sample_indices}")
    print(f"  Num candidates : {args.num_candidates}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Max new tokens : {args.max_new_tokens}")
    print(f"  Save GT        : {args.save_gt}")
    print(f"  Save refs      : {args.save_refs}")
    print(f"  Device         : {args.device}")
    print("=" * 70)

    # --- Load model ---
    hvm_config = None
    if args.no_hvm:
        base_model, tokenizer, processor, token_config = load_base_model_only(
            model_size=args.model_size,
            config_dir=args.config_dir,
            omnisvg_checkpoint=args.omnisvg_checkpoint,
            device=args.device,
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
            device=args.device,
        )
        transformer_for_generate = hvm_model.base_model.transformer

    # --- SVG tokenizer (for decoding) ---
    svg_tokenizer = SVGTokenizer(SVG_CONFIG_PATH, model_size=args.model_size)

    # --- Dataset ---
    print("\nLoading dataset ...")
    dataset = HVMDataset(
        data_dir=args.data_dir,
        hvm_dir=args.hvm_dir,
        token_config=token_config,
        train_config=TrainConfig(model_size=args.model_size),
        max_len=6000,
    )
    print(f"  Dataset size: {len(dataset)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Run inference ---
    total_ok = 0
    total_fail = 0
    total_skipped = 0

    for idx in args.sample_indices:
        if idx >= len(dataset):
            print(f"\n[Skip] sample {idx} out of range (dataset size={len(dataset)})")
            continue

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
        print(f"\n{'='*70}")
        print(f"Sample {idx}: {text[:100]}{'...' if len(text) > 100 else ''}")
        if not args.no_hvm:
            print(f"  ref_features  : {len(sample['ref_features'])} refs")
            print(f"  group_features: {len(sample['ref_best_group_features'])} groups")
            print(f"  ref_text      : {sample['ref_text'][:80]}...")
        print(f"{'='*70}")

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

        # 3) 准备文本 input_ids
        input_ids, attention_mask = prepare_text_inputs(
            text, processor, token_config, args.device
        )
        print(f"  Input tokens: {input_ids.shape[1]}")

        # 确定本样本要跑哪些模式
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
                    device=args.device,
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

            # Validate candidates (filter out empty/broken SVGs)
            if candidates and not args.no_validate:
                valid = []
                for ci, cand in enumerate(candidates):
                    if validate_candidate(cand["svg_str"]):
                        valid.append(cand)
                        if len(valid) >= args.num_candidates:
                            break
                    else:
                        print(f"    [{tag}] Candidate {ci}: invalid (empty or render failed)")
                rejected = len(candidates) - len(valid)
                candidates = valid
                if rejected > 0:
                    print(f"    [{tag}] Validated: {len(valid)} valid, {rejected} rejected")

            if candidates:
                sample_ok = True
                print(f"  [{tag}] Generated {len(candidates)} candidate(s) in {gen_elapsed:.1f}s")

                for ci, cand in enumerate(candidates):
                    suffix = f"_c{ci}" if len(candidates) > 1 else ""
                    base = f"sample_{idx:04d}_{tag}{suffix}"

                    svg_path = output_dir / f"{base}.svg"
                    svg_path.write_text(cand["svg_str"], encoding="utf-8")
                    print(f"    -> {svg_path.name} ({cand['num_paths']} paths)")

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
                print(f"  [{tag}] FAILED to generate valid SVG ({gen_elapsed:.1f}s)")

        # 保存 prompt 文本
        txt_path = output_dir / f"sample_{idx:04d}.txt"
        txt_path.write_text(text, encoding="utf-8")

        elapsed = time.time() - t0

        if sample_ok:
            total_ok += 1
        else:
            total_fail += 1

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"Inference Complete!  [{mode_str}]")
    if total_skipped > 0:
        print(f"  Skipped: {total_skipped} (already done)")
    print(f"  Success: {total_ok}/{len(args.sample_indices)}")
    print(f"  Failed : {total_fail}/{len(args.sample_indices)}")
    print(f"  Output : {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
