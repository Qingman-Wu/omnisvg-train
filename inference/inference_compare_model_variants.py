#!/usr/bin/env python3
"""
Run multiple text-to-SVG model variants from a prompt txt file.

Each prompt is evaluated by all variants listed in a tab-separated txt file.
This is intended for qualitative comparison figures such as:
  OmniSVG / Panoramic only / Panoramic + Spotlight / Full model /
  Shuffled panoramic.

Variant file format (tab-separated, '#' comments allowed):
  key<TAB>label<TAB>kind<TAB>checkpoint<TAB>hvm_config<TAB>flags

Example:
  omnisvg    OmniSVG    baseline    /path/to/full/hvm_step_5000.pt        -
  full_model Full model hvm         /path/to/full/hvm_step_5000.pt        -
  shuffled_panoramic  Shuffled panoramic hvm /path/to/full/hvm_step_5000.pt    shuffle_gme
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

INFERENCE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(INFERENCE_DIR, ".."))
sys.path.insert(0, INFERENCE_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:  # noqa: E402
    from inference_compare_text2svg import (
        DEFAULT_PARQUET_ROOT,
        DEFAULT_TEMPERATURE_SCHEDULE,
        RefImageSaver,
        _build_candidate_grid,
        _count_existing,
        _resolve_path,
        build_retrieval_for_prompts,
        load_prompts,
    )
    from inference_hvm_s1_test import (
        EXTRA_CANDIDATES_BUFFER,
        SVG_CONFIG_PATH,
        SVGTokenizer,
        clear_hvm_memory,
        generate_svg,
        load_hvm_model,
        prepare_text_inputs,
        prepare_visual_prefix_inputs,
        render_svg_to_image,
        set_hvm_memory,
        split_indices,
        validate_candidate,
    )
    from inference_hvm_text2svg_benchmark import CorpusFeatureStore, load_jsonl_subset
except ModuleNotFoundError:  # noqa: E402
    from inference.inference_compare_text2svg import (
        DEFAULT_PARQUET_ROOT,
        DEFAULT_TEMPERATURE_SCHEDULE,
        RefImageSaver,
        _build_candidate_grid,
        _count_existing,
        _resolve_path,
        build_retrieval_for_prompts,
        load_prompts,
    )
    from inference.inference_hvm_s1_test import (
        EXTRA_CANDIDATES_BUFFER,
        SVG_CONFIG_PATH,
        SVGTokenizer,
        clear_hvm_memory,
        generate_svg,
        load_hvm_model,
        prepare_text_inputs,
        prepare_visual_prefix_inputs,
        render_svg_to_image,
        set_hvm_memory,
        split_indices,
        validate_candidate,
    )
    from inference.inference_hvm_text2svg_benchmark import CorpusFeatureStore, load_jsonl_subset


DEFAULT_VARIANT_FILE = os.path.join(INFERENCE_DIR, "compare_model_variants.txt")


@dataclass
class ModelVariant:
    key: str
    label: str
    kind: str
    hvm_checkpoint: str
    hvm_config: str
    flags: List[str]
    shuffle_gme: bool = False


def _temp_label(temp: float) -> str:
    return f"t{temp:.1f}".replace(".", "")


def detect_hvm_config_path(
    hvm_checkpoint: str,
    explicit_path: Optional[str] = None,
) -> str:
    if explicit_path:
        explicit_path = _resolve_path(explicit_path)
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Explicit hvm_config not found: {explicit_path}")

    ckpt_dir = Path(_resolve_path(hvm_checkpoint)).resolve().parent
    for candidate_name in ("hvm_model_config.json", "hvm_config.json"):
        candidate = ckpt_dir / candidate_name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Cannot find hvm config near checkpoint: {hvm_checkpoint}")


def parse_flags(flags_str: str) -> List[str]:
    flags_str = (flags_str or "").strip()
    if not flags_str or flags_str == "-":
        return []
    return [flag.strip() for flag in flags_str.split(",") if flag.strip()]


def load_model_variants(path: str) -> List[ModelVariant]:
    variants: List[ModelVariant] = []
    seen_keys = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            parts = [part.strip() for part in line.split("\t")]
            while len(parts) < 6:
                parts.append("")
            key, label, kind, checkpoint, hvm_config, flags_str = parts[:6]

            if not key:
                raise ValueError(f"{path}:{line_no} missing variant key")
            if key in seen_keys:
                raise ValueError(f"{path}:{line_no} duplicate variant key: {key}")
            if kind not in {"baseline", "hvm"}:
                raise ValueError(f"{path}:{line_no} invalid kind '{kind}' (expected baseline or hvm)")
            if not checkpoint:
                raise ValueError(f"{path}:{line_no} missing checkpoint path")

            resolved_ckpt = _resolve_path(checkpoint)
            if not os.path.exists(resolved_ckpt):
                raise FileNotFoundError(f"{path}:{line_no} checkpoint not found: {resolved_ckpt}")

            flags = parse_flags(flags_str)
            variant = ModelVariant(
                key=key,
                label=label or key,
                kind=kind,
                hvm_checkpoint=resolved_ckpt,
                hvm_config=detect_hvm_config_path(resolved_ckpt, hvm_config or None),
                flags=flags,
                shuffle_gme=("shuffle_gme" in flags),
            )
            variants.append(variant)
            seen_keys.add(key)

    if not variants:
        raise ValueError(f"No variants loaded from: {path}")
    return variants


def load_corpus_ref_indices(metadata_path: str) -> List[int]:
    ref_indices: List[int] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ref_indices.append(int(record["idx"]))
    if not ref_indices:
        raise ValueError(f"No corpus ref indices found in: {metadata_path}")
    return ref_indices


def sample_random_corpus_refs(
    corpus_ref_indices: List[int],
    exclude_ref_indices: List[int],
    sample_count: int,
    seed: int,
    sample_key: int,
) -> List[int]:
    exclude_set = set(int(v) for v in exclude_ref_indices)
    candidates = [ref_idx for ref_idx in corpus_ref_indices if ref_idx not in exclude_set]
    if len(candidates) < sample_count:
        raise ValueError(
            f"Not enough corpus refs to sample {sample_count} random refs after exclusion."
        )
    rng = random.Random(seed + int(sample_key) * 1000003)
    return rng.sample(candidates, sample_count)


def resolve_prompt_dir(
    output_dir: Path,
    prompt_index: int,
    prompt_offset: int,
    prompt_ids: Optional[List[int]] = None,
    single_slug: Optional[str] = None,
) -> Path:
    if single_slug is not None:
        return output_dir / single_slug
    if prompt_ids is not None:
        return output_dir / f"prompt_{prompt_ids[prompt_index]:06d}"
    return output_dir / f"prompt_{prompt_offset + prompt_index:06d}"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_prompt_metadata(
    prompt_dir: Path,
    prompt: str,
    retrieval: Dict[str, Any],
    ref_meta_lookup: Dict[int, Dict[str, Any]],
) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    retrieval_meta = {
        "prompt": prompt,
        "ref_indices": retrieval["ref_indices"],
        "ref_scores": retrieval["ref_scores"],
        "ref_descriptions": [
            ref_meta_lookup.get(ri, {}).get("description", "")
            for ri in retrieval["ref_indices"]
        ],
    }
    write_json(prompt_dir / "retrieval.json", retrieval_meta)


def generate_candidates_multi_temp(
    transformer_model,
    input_ids,
    attention_mask,
    inputs_embeds,
    token_config,
    svg_tokenizer,
    gen_kwargs_base,
    no_validate,
    prompt_dir: Path,
    prefix: str,
    save_png: bool,
    schedule,
):
    all_candidates = []
    total_elapsed = 0.0

    for temp, num_cand in schedule:
        label = _temp_label(temp)
        already_done = sum(
            1 for ci in range(num_cand)
            if (prompt_dir / f"{prefix}_{label}_c{ci}.svg").exists()
        )
        if already_done >= num_cand:
            all_candidates.append((temp, num_cand, 0.0))
            continue

        gen_kwargs = dict(gen_kwargs_base)
        gen_kwargs["temperature"] = temp if temp > 0 else 1e-7
        if temp == 0:
            gen_kwargs["top_k"] = 1
            gen_kwargs["top_p"] = 1.0

        t0 = time.time()
        candidates = generate_svg(
            transformer_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_config=token_config,
            svg_tokenizer=svg_tokenizer,
            num_return_sequences=num_cand + EXTRA_CANDIDATES_BUFFER,
            inputs_embeds=inputs_embeds,
            **gen_kwargs,
        )
        elapsed = time.time() - t0
        total_elapsed += elapsed

        if candidates and not no_validate:
            valid = [cand for cand in candidates if validate_candidate(cand["svg_str"])]
            candidates = valid[:num_cand]

        for ci, cand in enumerate(candidates[:num_cand]):
            (prompt_dir / f"{prefix}_{label}_c{ci}.svg").write_text(
                cand["svg_str"], encoding="utf-8"
            )
            if save_png:
                img = render_svg_to_image(cand["svg_str"])
                if img is not None:
                    img.save(str(prompt_dir / f"{prefix}_{label}_c{ci}.png"))

        all_candidates.append((temp, len(candidates[:num_cand]), round(elapsed, 2)))

    return all_candidates, round(total_elapsed, 2)


def run_variants_on_single_gpu(
    local_rank: int,
    gpu_id: int,
    prompt_indices: List[int],
    prompts: List[str],
    retrieval_results: List[Dict[str, Any]],
    ref_meta_lookup: Dict[int, Dict[str, Any]],
    ref_groups_lookup: Dict[int, Dict[str, Any]],
    corpus_ref_indices: List[int],
    args: argparse.Namespace,
    variants: List[ModelVariant],
    prompt_ids: Optional[List[int]] = None,
):
    device = f"cuda:{gpu_id}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not prompt_indices:
        print(f"[GPU {gpu_id}] No prompts assigned, exiting.")
        return

    schedule = getattr(args, "_temperature_schedule", DEFAULT_TEMPERATURE_SCHEDULE)
    total_cand = sum(n for _, n in schedule)
    output_dir = Path(args.output_dir)
    prompt_offset = getattr(args, "_prompt_offset", 0)
    single_slug = getattr(args, "_single_prompt_slug", None)
    svg_tokenizer = SVGTokenizer(SVG_CONFIG_PATH, model_size=args.model_size)

    feature_store = CorpusFeatureStore(
        retrieval_hvm_dir=args.retrieval_hvm_dir,
        ref_meta_lookup=ref_meta_lookup,
        ref_groups_lookup=ref_groups_lookup,
        part_num_refs=args.retrieval_top_k,
    )
    ref_image_saver = RefImageSaver(
        ref_meta_lookup=ref_meta_lookup,
        ref_groups_lookup=ref_groups_lookup,
        parquet_root=args.parquet_root,
    )

    gen_kwargs_base = dict(
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    for variant in variants:
        pending_prompt_indices = []
        for pi in prompt_indices:
            prompt_dir = resolve_prompt_dir(
                output_dir=output_dir,
                prompt_index=pi,
                prompt_offset=prompt_offset,
                prompt_ids=prompt_ids,
                single_slug=single_slug,
            )
            done = _count_existing(prompt_dir, variant.key, schedule)
            if not (args.resume and done >= total_cand):
                pending_prompt_indices.append(pi)

        if not pending_prompt_indices:
            print(f"[GPU {gpu_id}] Variant {variant.label}: nothing pending, skip.")
            continue

        print(
            f"[GPU {gpu_id}] Loading variant {variant.label} "
            f"(kind={variant.kind}, ckpt={variant.hvm_checkpoint})"
        )
        model, tokenizer, processor, token_config, hvm_config = load_hvm_model(
            model_size=args.model_size,
            hvm_config_path=variant.hvm_config,
            hvm_checkpoint_path=variant.hvm_checkpoint,
            config_dir=args.config_dir,
            omnisvg_checkpoint=args.omnisvg_checkpoint,
            base_model_override=args.base_model,
            device=device,
        )
        transformer_for_generate = model.base_model.transformer

        pbar = tqdm(
            pending_prompt_indices,
            desc=f"[GPU {gpu_id}] {variant.key}",
            position=local_rank,
        )
        for pi in pbar:
            prompt = prompts[pi]
            prompt_dir = resolve_prompt_dir(
                output_dir=output_dir,
                prompt_index=pi,
                prompt_offset=prompt_offset,
                prompt_ids=prompt_ids,
                single_slug=single_slug,
            )
            retrieval = retrieval_results[pi]
            ref_indices = retrieval["ref_indices"]
            prompt_key = (
                prompt_ids[pi]
                if prompt_ids is not None else prompt_offset + pi
            )

            ensure_prompt_metadata(prompt_dir, prompt, retrieval, ref_meta_lookup)
            ref_image_saver.save_ref_images(prompt_dir, ref_indices)

            input_ids, attention_mask = prepare_text_inputs(
                prompt, processor, token_config, device
            )

            variant_detail = []
            variant_elapsed = 0.0
            donor_meta = None

            try:
                clear_hvm_memory(model)

                if variant.kind == "baseline":
                    variant_detail, variant_elapsed = generate_candidates_multi_temp(
                        transformer_model=transformer_for_generate,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        inputs_embeds=None,
                        token_config=token_config,
                        svg_tokenizer=svg_tokenizer,
                        gen_kwargs_base=gen_kwargs_base,
                        no_validate=args.no_validate,
                        prompt_dir=prompt_dir,
                        prefix=variant.key,
                        save_png=args.save_png,
                        schedule=schedule,
                    )
                else:
                    gme_ref_features = [
                        feature_store.load_ref_feature(ri) for ri in ref_indices
                    ]
                    if variant.shuffle_gme:
                        donor_ref_indices = sample_random_corpus_refs(
                            corpus_ref_indices=corpus_ref_indices,
                            exclude_ref_indices=ref_indices,
                            sample_count=args.retrieval_top_k,
                            seed=args.shuffle_seed,
                            sample_key=prompt_key,
                        )
                        gme_ref_features = [
                            feature_store.load_ref_feature(ri)
                            for ri in donor_ref_indices
                        ]
                        donor_meta = {
                            "random_ref_indices": donor_ref_indices,
                            "exclude_ref_indices": ref_indices,
                            "pool_size": len(corpus_ref_indices),
                        }

                    group_data = feature_store.load_part_group_bundle(ref_indices)
                    ref_text = feature_store.build_ref_text(ref_indices)
                    set_hvm_memory(
                        model,
                        ref_features=gme_ref_features,
                        group_features=group_data["group_features"],
                        ref_text=ref_text,
                        tokenizer=tokenizer,
                        hvm_config=hvm_config,
                        device=device,
                        group_tag_meta=group_data["tag_meta"],
                        group_ids=group_data["group_ids"],
                    )

                    inputs_embeds = None
                    gen_attention_mask = attention_mask
                    if (
                        hvm_config.memory_mode == "visual_prefix"
                        and getattr(model, "_visual_prefix", None) is not None
                    ):
                        inputs_embeds, gen_attention_mask = prepare_visual_prefix_inputs(
                            model, input_ids, attention_mask, device
                        )

                    variant_detail, variant_elapsed = generate_candidates_multi_temp(
                        transformer_model=transformer_for_generate,
                        input_ids=input_ids,
                        attention_mask=gen_attention_mask,
                        inputs_embeds=inputs_embeds,
                        token_config=token_config,
                        svg_tokenizer=svg_tokenizer,
                        gen_kwargs_base=gen_kwargs_base,
                        no_validate=args.no_validate,
                        prompt_dir=prompt_dir,
                        prefix=variant.key,
                        save_png=args.save_png,
                        schedule=schedule,
                    )
            finally:
                clear_hvm_memory(model)

            summary = {
                "variant_key": variant.key,
                "variant_label": variant.label,
                "variant_kind": variant.kind,
                "variant_flags": variant.flags,
                "hvm_checkpoint": variant.hvm_checkpoint,
                "hvm_config": variant.hvm_config,
                "memory_mode": getattr(hvm_config, "memory_mode", None),
                "inject_mode": getattr(hvm_config, "inject_mode", None),
                "prompt": prompt,
                "ref_indices": ref_indices,
                "temperature_schedule": [(t, n) for t, n in schedule],
                "generated_total": sum(n for _, n, _ in variant_detail),
                "elapsed": variant_elapsed,
                "detail": [
                    {"temp": t, "count": n, "elapsed": e}
                    for t, n, e in variant_detail
                ],
            }
            if donor_meta is not None:
                summary["shuffle_gme_donor"] = donor_meta

            write_json(prompt_dir / f"summary_{variant.key}.json", summary)

            if args.save_png:
                grid_img = _build_candidate_grid(prompt_dir, variant.key, schedule)
                if grid_img is not None:
                    grid_img.save(str(prompt_dir / f"{variant.key}_grid.png"))

            pbar.set_postfix(cand=sum(n for _, n, _ in variant_detail))

            gc.collect()
            torch.cuda.empty_cache()

        del model, tokenizer, processor, token_config, hvm_config, transformer_for_generate
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[GPU {gpu_id}] Done.")


def worker(
    local_rank: int,
    args: argparse.Namespace,
    index_splits: List[List[int]],
    prompts: List[str],
    retrieval_results: List[Dict[str, Any]],
    ref_meta_lookup: Dict[int, Dict[str, Any]],
    ref_groups_lookup: Dict[int, Dict[str, Any]],
    corpus_ref_indices: List[int],
    variant_payloads: List[Dict[str, Any]],
    prompt_ids: Optional[List[int]] = None,
):
    variants = [ModelVariant(**payload) for payload in variant_payloads]
    run_variants_on_single_gpu(
        local_rank=local_rank,
        gpu_id=local_rank,
        prompt_indices=index_splits[local_rank],
        prompts=prompts,
        retrieval_results=retrieval_results,
        ref_meta_lookup=ref_meta_lookup,
        ref_groups_lookup=ref_groups_lookup,
        corpus_ref_indices=corpus_ref_indices,
        args=args,
        variants=variants,
        prompt_ids=prompt_ids,
    )


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(
        description="Run multiple model variants on prompt txt for qualitative comparison."
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Text file with prompts (one per line, optionally prefixed with 'id: ').",
    )
    prompt_group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt string. Output dir name is derived from the prompt text.",
    )
    parser.add_argument(
        "--variant_file",
        type=str,
        default=DEFAULT_VARIANT_FILE,
        help="Tab-separated variant definition file.",
    )
    parser.add_argument("--model_size", type=str, default="8B", choices=["4B", "8B"])
    parser.add_argument("--config_dir", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--omnisvg_checkpoint", type=str, default=None)
    parser.add_argument(
        "--retrieval_hvm_dir",
        type=str,
        default=_resolve_path(
            "/mnt/data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/"
            "hvm_precomputed_22w_nozoom_top3part"
        ),
    )
    parser.add_argument(
        "--clip_model_path",
        type=str,
        default=_resolve_path("/mnt/data/wuqingman/models/openai/clip-vit-large-patch14"),
    )
    parser.add_argument("--retrieval_top_k", type=int, default=3)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results/compare_model_variants",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=None,
        help="Override: generate this many candidates at EACH non-zero temperature. "
        "Temperature 0.0 still uses 2 candidates.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--save_png", action="store_true", default=False)
    parser.add_argument("--no_validate", action="store_true", default=False)
    parser.add_argument(
        "--parquet_root",
        type=str,
        default=DEFAULT_PARQUET_ROOT,
        help="Root directory containing parquet files for loading ref SVG/images.",
    )
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=1234,
        help="Seed used to pick donor prompts for shuffle_gme variants.",
    )
    args = parser.parse_args()

    args.variant_file = _resolve_path(args.variant_file)
    args.retrieval_hvm_dir = _resolve_path(args.retrieval_hvm_dir)
    args.clip_model_path = _resolve_path(args.clip_model_path)
    args.parquet_root = _resolve_path(args.parquet_root)

    variants = load_model_variants(args.variant_file)
    if args.num_candidates is not None:
        k = args.num_candidates
        args._temperature_schedule = [
            (temp, 2 if temp == 0.0 else k)
            for temp, _ in DEFAULT_TEMPERATURE_SCHEDULE
        ]
    else:
        args._temperature_schedule = list(DEFAULT_TEMPERATURE_SCHEDULE)
    schedule = args._temperature_schedule
    total_cand = sum(n for _, n in schedule)

    if args.omnisvg_checkpoint is None:
        omnisvg_default = _resolve_path("/mnt/data2/wuqingman/models/OmniSVG/OmniSVG1.1_8B")
        if os.path.exists(omnisvg_default):
            args.omnisvg_checkpoint = omnisvg_default

    if args.prompt is not None:
        prompts = [args.prompt]
        prompt_ids = None
        prompt_offset = 0
        slug = re.sub(r"[^\w\s-]", "", args.prompt)[:60].strip().replace(" ", "_")
        timestamp = time.strftime("%m%d_%H%M%S")
        slug_with_ts = f"{slug}_{timestamp}"
        if not args.output_dir or args.output_dir == "./inference_results/compare_model_variants":
            args.output_dir = f"./inference_results/{slug_with_ts}"
        args._single_prompt_slug = slug_with_ts
        print(f'Single prompt mode: "{args.prompt[:80]}..."')
    else:
        all_prompt_ids, all_prompts = load_prompts(args.prompt_file)
        total = len(all_prompts)
        start = args.start_idx if args.start_idx is not None else 0
        end = args.end_idx if args.end_idx is not None else total
        start = max(0, min(start, total))
        end = max(start, min(end, total))
        prompts = all_prompts[start:end]
        prompt_ids = all_prompt_ids[start:end] if all_prompt_ids else None
        prompt_offset = start
        id_info = " (with explicit IDs)" if prompt_ids else ""
        print(
            f"Loaded {total} prompts from {args.prompt_file}, "
            f"using [{start}, {end}) = {len(prompts)} prompts{id_info}"
        )

    if not prompts:
        raise ValueError("No prompts to process.")

    print("\n=== Stage 1: CLIP Retrieval ===")
    retrieval_results = build_retrieval_for_prompts(
        prompts,
        args.retrieval_hvm_dir,
        args.clip_model_path,
        args.retrieval_top_k,
        torch.device("cuda:0"),
    )
    retrieved_ref_ids = {
        ref_idx for retrieval in retrieval_results for ref_idx in retrieval["ref_indices"]
    }
    metadata_path = os.path.join(args.retrieval_hvm_dir, "metadata.jsonl")
    groups_path = os.path.join(args.retrieval_hvm_dir, "groups_train_ref.jsonl")
    if not os.path.exists(groups_path):
        groups_path = os.path.join(args.retrieval_hvm_dir, "groups.jsonl")
    print(f"Loading metadata for {len(retrieved_ref_ids)} retrieved refs ...")
    ref_meta_lookup = load_jsonl_subset(metadata_path, retrieved_ref_ids)
    ref_groups_lookup = load_jsonl_subset(groups_path, retrieved_ref_ids)
    corpus_ref_indices = load_corpus_ref_indices(metadata_path)
    print(f"Loaded {len(corpus_ref_indices)} corpus refs for shuffle_gme sampling.")

    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available == 0:
        raise RuntimeError("No CUDA GPU detected!")
    num_gpus = min(args.num_gpus, num_gpus_available) if args.num_gpus else num_gpus_available

    all_indices = list(range(len(prompts)))
    args._prompt_offset = prompt_offset
    if args.resume:
        output_dir = Path(args.output_dir)
        pending = []
        for pi in all_indices:
            prompt_dir = resolve_prompt_dir(
                output_dir=output_dir,
                prompt_index=pi,
                prompt_offset=prompt_offset,
                prompt_ids=prompt_ids,
                single_slug=getattr(args, "_single_prompt_slug", None),
            )
            prompt_pending = False
            for variant in variants:
                done = _count_existing(prompt_dir, variant.key, schedule)
                if done < total_cand:
                    prompt_pending = True
                    break
            if prompt_pending:
                pending.append(pi)
        all_indices = pending
        print(
            f"Resume: {len(pending)} prompts pending, "
            f"{len(list(range(len(prompts)))) - len(pending)} already complete"
        )

    index_splits = split_indices(all_indices, num_gpus)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    manifest = {
        "variant_file": args.variant_file,
        "variants": [asdict(variant) for variant in variants],
        "temperature_schedule": [(t, n) for t, n in schedule],
        "prompt_count": len(prompts),
        "shuffle_seed": args.shuffle_seed,
        "corpus_ref_count": len(corpus_ref_indices),
    }
    write_json(Path(args.output_dir) / "variant_manifest.json", manifest)

    print("\n" + "=" * 70)
    print(f"Model Variant Comparison  --  {num_gpus} GPU(s)")
    print("=" * 70)
    print(f"  Prompt count     : {len(prompts)} total, {len(all_indices)} pending")
    print(f"  Variant file     : {args.variant_file}")
    print(f"  Variants         : {', '.join(variant.key for variant in variants)}")
    print(f"  Num candidates   : {args.num_candidates}")
    print(f"  Output dir       : {args.output_dir}")
    for i, split in enumerate(index_splits):
        label = f"[{split[0]}~{split[-1]}]" if split else "[empty]"
        print(f"  GPU {i}: {len(split)} prompts  {label}")
    print("=" * 70)

    variant_payloads = [asdict(variant) for variant in variants]
    if num_gpus == 1:
        run_variants_on_single_gpu(
            local_rank=0,
            gpu_id=0,
            prompt_indices=index_splits[0],
            prompts=prompts,
            retrieval_results=retrieval_results,
            ref_meta_lookup=ref_meta_lookup,
            ref_groups_lookup=ref_groups_lookup,
            corpus_ref_indices=corpus_ref_indices,
            args=args,
            variants=variants,
            prompt_ids=prompt_ids,
        )
    else:
        mp.start_processes(
            worker,
            args=(
                args,
                index_splits,
                prompts,
                retrieval_results,
                ref_meta_lookup,
                ref_groups_lookup,
                corpus_ref_indices,
                variant_payloads,
                prompt_ids,
            ),
            nprocs=num_gpus,
            start_method="spawn",
        )

    print(f"\n{'=' * 70}")
    print(f"All done! Results saved to: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
