#!/usr/bin/env python3
"""
Paired baseline vs HVM text2svg comparison (multi-GPU).

Reads prompts from a plain text file (one prompt per line), then:
  1. CLIP retrieval (main process, single GPU) — finds top-K refs per prompt.
  2. Spawns one worker per GPU, each loads its own HVM model copy.
  3. Each worker generates N baseline + N HVM candidates for its assigned prompts.

Usage (multi-GPU):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python inference/inference_compare_text2svg.py \
    --prompt_file /mnt/data2/wuqingman/omnisvg-train/paper_figures/selected_8_prompts.txt \
    --hvm_checkpoint /mnt/data3/wuqingman/omnisvg-train/outputs_s9_full25w_top3part_12slot_nogist_edr_parttag_nozoom_last4/hvm_step_5000.pt \
    --start_idx 0 \
    --end_idx 500 \
    --output_dir /mnt/data2/wuqingman/omnisvg-train/paper_figures/selected_8_prompts222 \
    --num_candidates 20 \
    --save_png \
    --resume

CUDA_VISIBLE_DEVICES=0 python inference/inference_compare_text2svg.py \
    --prompt "A colorful cartoon-style illustration of a bus with rounded edges and large windows." \
    --hvm_checkpoint /mnt/data3/wuqingman/omnisvg-train/outputs_s9_full25w_top3part_12slot_nogist_edr_parttag_nozoom_last4/hvm_step_5000.pt \
    --output_dir /mnt/data2/wuqingman/omnisvg-train/paper_figures/single_test \
    --num_candidates 20 \
    --save_png

A blond, gentle-looking cartoon boy wearing a red vest, with a round and cute style.
A beautiful pink flower
A round-faced, big-eyed, and adorable cartoon tiger head in orange and yellow colors.

A cartoon-style long-haired girl with an angry expression, wearing a yellow dress.
Calculator icon, blue screen, red buttons

A cartoon-style boy character wearing a golden crown, with a beard, and dressed in a blue shirt.
A cartoon-style girl character, wearing a golden crown, smiling, and dressed in light gray clothes.
A book with a heart pattern, with a pen next to it.
A cartoon illustration of a person wearing a headset with a microphone.

A cute, tech-themed lab flask
A illustration of a plant growing from soil.
An orange cat with long whiskers

A man facing us sat cross-legged with his hands on his knees.
A boy with cartoon illustrations is sitting on a computer and working hard.
Cartoon hamster illustration, round face, orange hair, white chin, dark brown eye block.
A cartoon-style pink baby bottle
A red circular background has an open yellow folder with a magnifying glass icon on it to indicate the query.
A cartoon bear head


3275：Illustration of a fire hydrant with a red bottle body
3315：A piece of watermelon with red pulp and green skin with some black seeds on it.
3422：A ripe ear of wheat
3478：A man with a side face full of science and technology
3526：A yellow truck with two wheels and a black front window.
3711：Cartoon mushroom SVG, red round cap with white dots.
3834：Cartoon ice cream SVG, the cone is an orange-red slender triangle, and the spherical ice cream is superimposed in two colors.
3862：Cartoon cat head SVG, round mint green background, milky white fluffy hair, orange pointed ears, blue big round eyes.
3886：An illustration of a cartoon raccoon face, with an orange head with a brown outline, two pink inner ear parts, two dark brown oval eyes, a small black nose, a white lower face and a dark brown smile on a white background.
3878：An anthropomorphic yellow pumpkin with a green stem and an unhappy expression on it.
3947：An illustration consists of three urban buildings with different heights, with a chimney-like structure at the top.
4072：An illustration with a gray delivery truck on a red circular background.
4446：Blue earth illustration
4493：Stylistic illustration of a bank building, with purple at the bottom and top, golden building in the middle and a dollar symbol in the center.
4566：A computer monitor with a blue square in the middle and a cartoon head in the middle.
4766！！：A purple chinchilla with a white striped scarf and a long beard.


6077！！：Two pins SVG, one large and one small, are placed in a staggered way. The pins consist of a red disk, a golden neck and a purple needle body.

！！！A cartoon-style light purple cat, white belly, black eyes, two light purple ears, pink triangles inside the ears and a long black beard.




!!!104743 A computer screen displays a lock icon, signifying data security and protection.

105346  A green vehicle is driving, with three blue windows and gray wheels.

105758 A flat design icon of an ATM machine with a yellow circular background, featuring a blue screen displaying a credit card with red, green, and yellow dots.
！！！108233： A flat design illustration of a yellow rabbit in side profile sitting pose, with red inner ears and a small red tail, minimalist geometric style

110169：A church with a golden dome.
110886  A green plant in a yellow flowerpot
11637A blue ambulance with a cross on the side.

111672：A stylized illustration of a Wi-Fi router with two antennas and a signal strength indicator.
112640：A colorful cartoon-style illustration of a bus with rounded edges and large windows.

113065：A flat design yellow cat face emoji with a wide grinning smile showing teeth, closed crescent-shaped eyes, orange nose and ear tips, cartoon style on white background.





113524：A yellow star-shaped character with large blue eyes, pink cheeks, and black eyebrows is depicted. It has three colorful triangular flagsâpurple, pink, and orangeâattached to one side. The flags feature small white sparkles and round dots, adding decorative elements. The background is entirely white, emphasizing the vibrant colors of the star and the flags.


117100：A teal-colored crown with yellow gemstones is depicted in a minimalist style.



123809：A yellow star-shaped object with two large navy blue circular eyes, one pink heart-shaped nose, two small pink cheeks, two navy blue eyebrows, and several white decorative elements including stars, circles, and lines scattered across its surface. The white background surrounds the star, making it the focal point of the image.

127528：An illustration of a computer screen displaying a web page with elements such as buttons, text boxes, and icons, set against a yellow circular background.

128399：A flat design illustration of a wooden signpost with three arrow-shaped signs pointing in different directions — two yellow arrows pointing right and one red arrow pointing left — mounted on a dark pole with green grass at the base.
128594：A pink hand holding a lime-green rectangular card with three dark green circles at each corner, positioned above a yellow rectangular grid patterned card with one central dark green circle. The entire scene has a white background.


130491：An image features four colored puzzle pieces shaped like human profiles, each in a distinct colorâpurple, green, pink, and yellowâforming a group. A hand with a dark blue sleeve is placing the green puzzle piece into the arrangement, positioned slightly above and to the right of the others. The background is completely white, emphasizing the vibrant colors of the puzzle pieces.

130861：A stylized illustration of a smartphone displaying a date with colorful decorative elements around it.


131694：A stylized illustration of a camera with a large lens and various colored components.

6284：A smartphone displaying a blue phone icon with a yellow handset.

6819：A stylized emoji depicting a camera with a circular lens and a red top cap.

6842：A smartphone with a pie chart and a speech bubble is displayed.


7018：A computer monitor displaying a red fingerprint icon on a dark background.

7020：A cartoon-style green vehicle with a pink paw print and cross symbol on its side.


7405：A computer screen displays a person icon with a speech bubble containing a question mark.

8132：A stylized, abstract representation of a planet with a blue and purple color scheme.

8488：A magnifying glass focuses on a person's profile picture displayed on a computer screen.




9592：A smartphone with a pie chart icon featuring a dollar sign inside it.




9808：A teal-colored spray bottle with a red nozzle is depicted alongside its cap, surrounded by small bubbles and sparkles.



100464：An anthropomorphic seagull wearing a light blue sailor's cap with a black anchor emblem stands upright against a white background. The bird has a white body with gray wings and a black tail. Its orange beak and feet are prominent, and it wears a light blue scarf around its neck. The facial features include closed eyes depicted as simple curved lines and an orange triangular nose area.



113749！！！ An illustration of a cute, stylized robot with a pink rectangular body, blue legs, a purple arm, and a yellow antenna. The robot has a black face with two white circular eyes and a small curved mouth, giving it a simple yet friendly appearance. The background is entirely white, emphasizing the vibrant colors of the robot.


















"""

import argparse
import gc
import math
import io
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from inference_hvm_s1_test import (
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
from inference_hvm_text2svg_benchmark import (
    CorpusFeatureStore,
    load_jsonl_subset,
)

EXTRA_CANDIDATES_BUFFER = 0

GROUP_RENDER_VIEWBOX = 200.0


def _resolve_path(path: str) -> str:
    """Auto-fix a100-1 native paths when running on a100-3."""
    if os.path.exists(path):
        return path
    for prefix, mapped in [
        ("/mnt/data3/", "/mnt/a100_1_data3/"),
        ("/mnt/data2/", "/mnt/a100_1_data2/"),
        ("/mnt/data/", "/mnt/a100_1_data/"),
    ]:
        if path.startswith(prefix):
            alt = mapped + path[len(prefix):]
            if os.path.exists(alt):
                return alt
    return path


DEFAULT_PARQUET_ROOT = _resolve_path(
    "/mnt/data3/wuqingman/datasets/OmniSVG/"
    "MMSVG-Illustration/data_process_train25_exclude_p25"
)


def _render_group_to_image(
    svg_string: str, group_path_indices: List[int], image_size: int = 448,
) -> Optional[Image.Image]:
    if not svg_string or not group_path_indices:
        return None
    try:
        import cairosvg
    except ImportError:
        return None

    path_pattern = re.compile(r'<path\s[^>]*?(?:/>|>\s*</path>)', re.DOTALL)
    all_path_tags = path_pattern.findall(svg_string)
    if not all_path_tags:
        return None

    svg_lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {GROUP_RENDER_VIEWBOX} {GROUP_RENDER_VIEWBOX}" '
        f'width="{image_size}" height="{image_size}">',
        f'<rect x="0" y="0" width="{GROUP_RENDER_VIEWBOX}" '
        f'height="{GROUP_RENDER_VIEWBOX}" fill="white"/>',
    ]
    for pidx in group_path_indices:
        if 0 <= int(pidx) < len(all_path_tags):
            svg_lines.append(all_path_tags[int(pidx)])
    svg_lines.append("</svg>")

    try:
        png_bytes = cairosvg.svg2png(
            bytestring="\n".join(svg_lines).encode("utf-8"),
            output_width=image_size, output_height=image_size,
        )
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception:
        return None


class RefImageSaver:
    def __init__(
        self,
        ref_meta_lookup: Dict[int, Dict[str, Any]],
        ref_groups_lookup: Dict[int, Dict[str, Any]],
        parquet_root: str = DEFAULT_PARQUET_ROOT,
        image_size: int = 448,
    ):
        self.ref_meta_lookup = ref_meta_lookup
        self.ref_groups_lookup = ref_groups_lookup
        self.parquet_root = parquet_root
        self.image_size = image_size
        self._parquet_cache: Dict[str, dict] = {}

    def _load_parquet(self, parquet_file: str) -> dict:
        if parquet_file not in self._parquet_cache:
            import pyarrow.parquet as pq
            path = os.path.join(self.parquet_root, parquet_file)
            table = pq.read_table(path, columns=["svg", "image"])
            self._parquet_cache[parquet_file] = table.to_pydict()
        return self._parquet_cache[parquet_file]

    def _get_svg_and_image(self, ref_idx: int):
        meta = self.ref_meta_lookup.get(ref_idx, {})
        pf = meta.get("parquet_file")
        pr = meta.get("parquet_row")
        if pf is None or pr is None:
            return None, None
        data = self._load_parquet(pf)
        svg_str = data["svg"][pr]
        img_dict = data["image"][pr]
        img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            img = None
        return svg_str, img

    def save_ref_images(
        self,
        prompt_dir: Path,
        ref_indices: List[int],
    ) -> None:
        refs_dir = prompt_dir / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)

        for rank, ref_idx in enumerate(ref_indices):
            whole_path = refs_dir / f"ref_{rank}.png"
            if whole_path.exists():
                continue

            svg_str, img = self._get_svg_and_image(ref_idx)

            if img is not None:
                img_resized = img.resize(
                    (self.image_size, self.image_size), Image.LANCZOS,
                )
                img_resized.save(str(whole_path))

            groups_record = self.ref_groups_lookup.get(ref_idx, {})
            groups = list(groups_record.get("groups", []))
            for gi, group in enumerate(groups):
                group_path = refs_dir / f"ref_{rank}_group_{gi}.png"
                if group_path.exists():
                    continue
                path_indices = [int(v) for v in group.get("path_indices", [])]
                if not path_indices or svg_str is None:
                    continue
                group_img = _render_group_to_image(
                    svg_str, path_indices, self.image_size,
                )
                if group_img is not None:
                    group_img.save(str(group_path))


def load_prompts(path: str):
    """Load prompts from a text file.

    Supports two formats:
      - Plain:   ``A cat sitting on a mat``
      - With ID: ``612: A cat sitting on a mat``

    Returns (prompt_ids, prompts) where prompt_ids is a list of int
    (parsed IDs) or None (plain format).
    """
    prompts: List[str] = []
    ids: List[int] = []
    has_ids = True

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\d+)\s*:\s*(.+)$", line)
            if m:
                ids.append(int(m.group(1)))
                prompts.append(m.group(2).strip())
            else:
                has_ids = False
                prompts.append(line)

    return (ids if has_ids and ids else None, prompts)


def build_retrieval_for_prompts(
    prompts: List[str],
    retrieval_hvm_dir: str,
    clip_model_path: str,
    top_k: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    from transformers import CLIPModel, CLIPTokenizer

    print(f"Loading CLIP model from {clip_model_path} ...")
    is_local = os.path.isdir(clip_model_path)
    clip_model = CLIPModel.from_pretrained(clip_model_path, local_files_only=is_local)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path, local_files_only=is_local)
    clip_model = clip_model.to(device).eval()

    embed_dim = clip_model.config.projection_dim
    embeddings = np.zeros((len(prompts), embed_dim), dtype=np.float32)
    batch_size = 64

    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        inputs = clip_tokenizer(
            prompts[start:end], padding=True, truncation=True,
            max_length=77, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            feats = clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings[start:end] = feats.cpu().numpy()

    del clip_model, clip_tokenizer
    torch.cuda.empty_cache()

    try:
        import faiss
        index_path = os.path.join(retrieval_hvm_dir, "faiss_index.bin")
        print(f"Loading FAISS index from {index_path} ...")
        index = faiss.read_index(index_path)
        print(f"FAISS index size: {index.ntotal}")
        scores, indices = index.search(embeddings, top_k)
    except ImportError:
        embeddings_path = os.path.join(retrieval_hvm_dir, "text_embeddings.npy")
        print("faiss not installed; falling back to brute-force search ...")
        corpus_embeddings = np.load(embeddings_path, mmap_mode="r")
        query = torch.from_numpy(embeddings).to(device)
        query = query / query.norm(dim=-1, keepdim=True)

        top_scores = torch.full((len(prompts), top_k), float("-inf"), device=device)
        top_indices = torch.full((len(prompts), top_k), -1, dtype=torch.long, device=device)
        chunk_size = 16384

        for cs in range(0, corpus_embeddings.shape[0], chunk_size):
            ce = min(cs + chunk_size, corpus_embeddings.shape[0])
            chunk = torch.from_numpy(np.asarray(corpus_embeddings[cs:ce], dtype=np.float32)).to(device)
            chunk = chunk / chunk.norm(dim=-1, keepdim=True)
            sc = torch.matmul(query, chunk.T)
            local_k = min(top_k, sc.shape[1])
            ls, li = torch.topk(sc, k=local_k, dim=1)
            li = li + cs
            ms = torch.cat([top_scores, ls], dim=1)
            mi = torch.cat([top_indices, li], dim=1)
            top_scores, sel = torch.topk(ms, k=top_k, dim=1)
            top_indices = torch.gather(mi, 1, sel)

        scores = top_scores.cpu().numpy()
        indices = top_indices.cpu().numpy()

    results = []
    for i in range(len(prompts)):
        results.append({
            "ref_indices": [int(indices[i, j]) for j in range(top_k)],
            "ref_scores": [float(scores[i, j]) for j in range(top_k)],
        })
    return results


# ============================================================================
# Per-GPU worker
# ============================================================================

DEFAULT_TEMPERATURE_SCHEDULE = [
    (0.0, 2),
    (0.3, 2),
    (0.5, 5),
    (0.7, 2),
    (1.0, 2),
]


def _temp_label(temp: float) -> str:
    return f"t{temp:.1f}".replace(".", "")


def _count_existing(prompt_dir: Path, prefix: str, schedule) -> int:
    """Count existing candidate SVGs for a given prefix (e.g. 'base' or 'hvm')."""
    total = 0
    for temp, n in schedule:
        label = _temp_label(temp)
        for ci in range(n):
            if (prompt_dir / f"{prefix}_{label}_c{ci}.svg").exists():
                total += 1
    return total


def _build_candidate_grid(
    prompt_dir: Path, prefix: str, schedule, cell_size: int = 256, padding: int = 4,
) -> Optional[Image.Image]:
    """Collect all candidate PNGs for *prefix* and arrange them in a compact grid.

    Each cell is labelled with its temperature and candidate index.
    Returns None when no images are found.
    """
    entries: list = []
    for temp, n in schedule:
        label = _temp_label(temp)
        for ci in range(n):
            png_path = prompt_dir / f"{prefix}_{label}_c{ci}.png"
            if png_path.exists():
                entries.append((f"{label}_c{ci}", png_path))

    if not entries:
        return None

    total = len(entries)
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)

    grid_w = cols * (cell_size + padding) + padding
    grid_h = rows * (cell_size + padding) + padding
    grid_img = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
    except ImportError:
        draw = None
        font = None

    for idx, (label, png_path) in enumerate(entries):
        r, c = divmod(idx, cols)
        x = padding + c * (cell_size + padding)
        y = padding + r * (cell_size + padding)
        try:
            img = Image.open(png_path).convert("RGB")
            img = img.resize((cell_size, cell_size), Image.LANCZOS)
            grid_img.paste(img, (x, y))
        except Exception:
            pass
        if draw is not None:
            draw.text((x + 4, y + 2), label, fill=(255, 60, 60), font=font)

    return grid_img


def _generate_candidates_multi_temp(
    model, input_ids, attention_mask, token_config, svg_tokenizer,
    gen_kwargs_base, no_validate, prompt_dir, prefix, save_png, schedule,
):
    """Run generation at each temperature in the schedule, save results."""
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

        gk = dict(gen_kwargs_base)
        gk["temperature"] = temp if temp > 0 else 1e-7
        if temp == 0:
            gk["top_k"] = 1
            gk["top_p"] = 1.0

        t0 = time.time()
        candidates = generate_svg(
            model, input_ids, attention_mask,
            token_config, svg_tokenizer,
            num_return_sequences=num_cand + EXTRA_CANDIDATES_BUFFER,
            **gk,
        )
        elapsed = time.time() - t0
        total_elapsed += elapsed

        if candidates and not no_validate:
            valid = [c for c in candidates if validate_candidate(c["svg_str"])]
            candidates = valid[:num_cand]

        for ci, cand in enumerate(candidates[:num_cand]):
            (prompt_dir / f"{prefix}_{label}_c{ci}.svg").write_text(cand["svg_str"], encoding="utf-8")
            if save_png:
                img = render_svg_to_image(cand["svg_str"])
                if img is not None:
                    img.save(str(prompt_dir / f"{prefix}_{label}_c{ci}.png"))

        all_candidates.append((temp, len(candidates[:num_cand]), round(elapsed, 2)))

    return all_candidates, round(total_elapsed, 2)


def run_on_single_gpu(
    local_rank: int,
    gpu_id: int,
    prompt_indices: List[int],
    prompts: List[str],
    retrieval_results: List[Dict[str, Any]],
    ref_meta_lookup: Dict[int, Dict[str, Any]],
    ref_groups_lookup: Dict[int, Dict[str, Any]],
    args: argparse.Namespace,
    prompt_ids: Optional[List[int]] = None,
):
    device = f"cuda:{gpu_id}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not prompt_indices:
        print(f"[GPU {gpu_id}] No prompts assigned, exiting.")
        return

    schedule = getattr(args, '_temperature_schedule', DEFAULT_TEMPERATURE_SCHEDULE)
    total_cand = sum(n for _, n in schedule)

    print(f"\n[GPU {gpu_id}] Assigned {len(prompt_indices)} prompts: "
          f"{prompt_indices[0]} ~ {prompt_indices[-1]}")
    print(f"  Temperature schedule: {[(t, n) for t, n in schedule]} = {total_cand} candidates/prompt")

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

    output_dir = Path(args.output_dir)

    gen_kwargs_base = dict(
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    prompt_offset = getattr(args, '_prompt_offset', 0)
    single_slug = getattr(args, '_single_prompt_slug', None)
    skipped = 0
    pbar = tqdm(prompt_indices, desc=f"[GPU {gpu_id}]", position=local_rank)
    for pi in pbar:
        prompt = prompts[pi]
        if single_slug is not None:
            prompt_dir = output_dir / single_slug
        elif prompt_ids is not None:
            prompt_dir = output_dir / f"prompt_{prompt_ids[pi]:06d}"
        else:
            prompt_dir = output_dir / f"prompt_{prompt_offset + pi:06d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        base_done = _count_existing(prompt_dir, "base", schedule)
        hvm_done = _count_existing(prompt_dir, "hvm", schedule)
        need_base = base_done < total_cand
        need_hvm = hvm_done < total_cand

        if args.resume and not need_base and not need_hvm:
            skipped += 1
            continue

        input_ids, attention_mask = prepare_text_inputs(
            prompt, processor, token_config, device
        )

        retrieval = retrieval_results[pi]
        ref_indices = retrieval["ref_indices"]

        retrieval_meta = {
            "prompt": prompt,
            "ref_indices": ref_indices,
            "ref_scores": retrieval["ref_scores"],
            "ref_descriptions": [
                ref_meta_lookup.get(ri, {}).get("description", "")
                for ri in ref_indices
            ],
        }
        (prompt_dir / "retrieval.json").write_text(
            json.dumps(retrieval_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        ref_image_saver.save_ref_images(prompt_dir, ref_indices)

        # ---- Baseline (no HVM) ----
        base_elapsed = 0.0
        base_detail = []
        if need_base or not args.resume:
            clear_hvm_memory(hvm_model)
            base_detail, base_elapsed = _generate_candidates_multi_temp(
                transformer_for_generate, input_ids, attention_mask,
                token_config, svg_tokenizer, gen_kwargs_base,
                args.no_validate, prompt_dir, "base", args.save_png, schedule,
            )
            base_done = sum(n for _, n, _ in base_detail)

        # ---- HVM (with retrieval) ----
        hvm_elapsed = 0.0
        hvm_detail = []
        if need_hvm or not args.resume:
            ref_features = [feature_store.load_ref_feature(ri) for ri in ref_indices]
            group_data = feature_store.load_part_group_bundle(ref_indices)
            ref_text = feature_store.build_ref_text(ref_indices)

            set_hvm_memory(
                hvm_model,
                ref_features=ref_features,
                group_features=group_data["group_features"],
                ref_text=ref_text,
                tokenizer=tokenizer,
                hvm_config=hvm_config,
                device=device,
                group_tag_meta=group_data["tag_meta"],
                group_ids=group_data["group_ids"],
            )

            gen_inputs_embeds = None
            gen_attn = attention_mask
            if hvm_config.memory_mode == "visual_prefix" and getattr(hvm_model, "_visual_prefix", None) is not None:
                gen_inputs_embeds, gen_attn = prepare_visual_prefix_inputs(
                    hvm_model, input_ids, attention_mask, device
                )
            hvm_detail, hvm_elapsed = _generate_candidates_multi_temp(
                transformer_for_generate,
                input_ids if gen_inputs_embeds is None else gen_inputs_embeds,
                gen_attn,
                token_config, svg_tokenizer, gen_kwargs_base,
                args.no_validate, prompt_dir, "hvm", args.save_png, schedule,
            )
            clear_hvm_memory(hvm_model)
            hvm_done = sum(n for _, n, _ in hvm_detail)

        summary = {
            "prompt": prompt,
            "temperature_schedule": [(t, n) for t, n in schedule],
            "base_total": base_done,
            "base_elapsed": base_elapsed,
            "base_detail": [{"temp": t, "count": n, "elapsed": e} for t, n, e in base_detail],
            "hvm_total": hvm_done,
            "hvm_elapsed": hvm_elapsed,
            "hvm_detail": [{"temp": t, "count": n, "elapsed": e} for t, n, e in hvm_detail],
        }
        (prompt_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if args.save_png:
            for grid_prefix in ("base", "hvm"):
                grid_img = _build_candidate_grid(prompt_dir, grid_prefix, schedule)
                if grid_img is not None:
                    grid_img.save(str(prompt_dir / f"{grid_prefix}_grid.png"))

        pbar.set_postfix(base=base_done, hvm=hvm_done)

        gc.collect()
        torch.cuda.empty_cache()

    print(f"[GPU {gpu_id}] Done. (skipped {skipped} already-complete prompts)")



def worker(
    local_rank: int,
    args: argparse.Namespace,
    index_splits: List[List[int]],
    prompts: List[str],
    retrieval_results: List[Dict[str, Any]],
    ref_meta_lookup: Dict[int, Dict[str, Any]],
    ref_groups_lookup: Dict[int, Dict[str, Any]],
    prompt_ids: Optional[List[int]] = None,
):
    run_on_single_gpu(
        local_rank=local_rank,
        gpu_id=local_rank,
        prompt_indices=index_splits[local_rank],
        prompts=prompts,
        retrieval_results=retrieval_results,
        ref_meta_lookup=ref_meta_lookup,
        ref_groups_lookup=ref_groups_lookup,
        args=args,
        prompt_ids=prompt_ids,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    p = argparse.ArgumentParser(description="Paired baseline vs HVM text2svg comparison (multi-GPU)")
    prompt_group = p.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt_file", type=str, default=None,
                              help="Text file with prompts (one per line, optionally prefixed with 'id: ').")
    prompt_group.add_argument("--prompt", type=str, default=None,
                              help="Single prompt string. Output dir name is derived from the prompt text.")
    p.add_argument("--model_size", type=str, default="8B", choices=["4B", "8B"])
    p.add_argument("--config_dir", type=str, default=None)
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--hvm_checkpoint", type=str, required=True)
    p.add_argument("--hvm_config", type=str, default=None)
    p.add_argument("--omnisvg_checkpoint", type=str, default=None)
    p.add_argument("--retrieval_hvm_dir", type=str,
                   default=_resolve_path("/mnt/data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_22w_nozoom_top3part"))
    p.add_argument("--clip_model_path", type=str,
                   default=_resolve_path("/mnt/data/wuqingman/models/openai/clip-vit-large-patch14"))
    p.add_argument("--retrieval_top_k", type=int, default=3)
    p.add_argument("--output_dir", type=str, default="./inference_results/compare_baseline_vs_hvm")
    p.add_argument("--num_candidates", type=int, default=None,
                   help="Override: generate this many candidates at EACH temperature. "
                        "If not set, use the built-in schedule (2/2/5/2/2).")
    p.add_argument("--max_new_tokens", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.90)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--save_png", action="store_true", default=False)
    p.add_argument("--no_validate", action="store_true", default=False)
    p.add_argument("--parquet_root", type=str, default=DEFAULT_PARQUET_ROOT,
                   help="Root directory containing parquet files for loading ref SVG/images.")
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--num_gpus", type=int, default=None,
                   help="Number of GPUs to use. Default = all visible GPUs.")
    p.add_argument("--start_idx", type=int, default=None,
                   help="Start prompt index (0-based, inclusive). Default = 0.")
    p.add_argument("--end_idx", type=int, default=None,
                   help="End prompt index (0-based, exclusive). Default = len(prompts).")
    args = p.parse_args()

    args.retrieval_hvm_dir = _resolve_path(args.retrieval_hvm_dir)
    args.clip_model_path = _resolve_path(args.clip_model_path)
    args.parquet_root = _resolve_path(args.parquet_root)

    if args.num_candidates is not None:
        k = args.num_candidates
        args._temperature_schedule = [(t, 2 if t == 0.0 else k) for t, _ in DEFAULT_TEMPERATURE_SCHEDULE]
    else:
        args._temperature_schedule = list(DEFAULT_TEMPERATURE_SCHEDULE)
    schedule = args._temperature_schedule
    total_cand = sum(n for _, n in schedule)
    if args.omnisvg_checkpoint is None:
        _omni_default = _resolve_path("/mnt/data2/wuqingman/models/OmniSVG/OmniSVG1.1_8B")
        if os.path.exists(_omni_default):
            args.omnisvg_checkpoint = _omni_default

    if args.hvm_config is None:
        ckpt_dir = Path(args.hvm_checkpoint).parent
        candidate = ckpt_dir / "hvm_model_config.json"
        if candidate.exists():
            args.hvm_config = str(candidate)
        else:
            candidate2 = ckpt_dir / "hvm_config.json"
            if candidate2.exists():
                args.hvm_config = str(candidate2)
            else:
                raise FileNotFoundError(
                    f"Cannot find hvm_model_config.json or hvm_config.json in {ckpt_dir}. "
                    "Please specify --hvm_config explicitly."
                )

    # ---- 1. Load prompts ----
    if args.prompt is not None:
        prompts = [args.prompt]
        prompt_ids = None
        prompt_offset = 0
        slug = re.sub(r"[^\w\s-]", "", args.prompt)[:60].strip().replace(" ", "_")
        timestamp = time.strftime("%m%d_%H%M%S")
        slug_with_ts = f"{slug}_{timestamp}"
        if not args.output_dir or args.output_dir == "./inference_results/compare_baseline_vs_hvm":
            args.output_dir = f"./inference_results/{slug_with_ts}"
        args._single_prompt_slug = slug_with_ts
        print(f"Single prompt mode: \"{args.prompt[:80]}...\"")
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
        id_info = f" (with explicit IDs)" if prompt_ids else ""
        print(f"Loaded {total} prompts from {args.prompt_file}, using [{start}, {end}) = {len(prompts)} prompts{id_info}")

    # ---- 2. CLIP retrieval (main process, GPU 0) ----
    print("\n=== Stage 1: CLIP Retrieval ===")
    retrieval_results = build_retrieval_for_prompts(
        prompts, args.retrieval_hvm_dir, args.clip_model_path,
        args.retrieval_top_k, torch.device("cuda:0"),
    )
    retrieved_ref_ids = {
        ref_idx for r in retrieval_results for ref_idx in r["ref_indices"]
    }
    metadata_path = os.path.join(args.retrieval_hvm_dir, "metadata.jsonl")
    groups_path = os.path.join(args.retrieval_hvm_dir, "groups_train_ref.jsonl")
    if not os.path.exists(groups_path):
        groups_path = os.path.join(args.retrieval_hvm_dir, "groups.jsonl")
    print(f"Loading metadata for {len(retrieved_ref_ids)} retrieved refs ...")
    ref_meta_lookup = load_jsonl_subset(metadata_path, retrieved_ref_ids)
    ref_groups_lookup = load_jsonl_subset(groups_path, retrieved_ref_ids)

    # ---- 3. Determine GPUs ----
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
            gidx = prompt_ids[pi] if prompt_ids else (prompt_offset + pi)
            prompt_dir = output_dir / f"prompt_{gidx:06d}"
            if not prompt_dir.exists():
                prompt_dir = output_dir / f"prompt_{gidx:03d}"
            base_done = _count_existing(prompt_dir, "base", schedule)
            hvm_done = _count_existing(prompt_dir, "hvm", schedule)
            if base_done < total_cand or hvm_done < total_cand:
                pending.append(pi)
        all_indices = pending
        print(f"Resume: {len(pending)} prompts pending, {len(list(range(len(prompts)))) - len(pending)} already complete")

    index_splits = split_indices(all_indices, num_gpus)

    print("\n" + "=" * 70)
    print(f"Baseline vs HVM Comparison  --  {num_gpus} GPU(s)")
    print("=" * 70)
    print(f"  Prompts          : {len(prompts)} total, {len(all_indices)} pending")
    print(f"  HVM checkpoint   : {args.hvm_checkpoint}")
    print(f"  Num candidates   : {args.num_candidates}")
    print(f"  Output dir       : {args.output_dir}")
    for i, split in enumerate(index_splits):
        label = f"[{split[0]}~{split[-1]}]" if split else "[empty]"
        print(f"  GPU {i}: {len(split)} prompts  {label}")
    print("=" * 70)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- 4. Launch workers ----
    if num_gpus == 1:
        run_on_single_gpu(
            local_rank=0, gpu_id=0,
            prompt_indices=index_splits[0],
            prompts=prompts,
            retrieval_results=retrieval_results,
            ref_meta_lookup=ref_meta_lookup,
            ref_groups_lookup=ref_groups_lookup,
            args=args,
            prompt_ids=prompt_ids,
        )
    else:
        mp.start_processes(
            worker,
            args=(args, index_splits, prompts, retrieval_results,
                  ref_meta_lookup, ref_groups_lookup, prompt_ids),
            nprocs=num_gpus,
            start_method="spawn",
        )

    print(f"\n{'='*70}")
    print(f"All done! Results saved to: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
