#!/usr/bin/env python3
"""
将 GT SVG 与离线检索到的 top-3 参考 SVG 渲染并拼接成质量检查图。
"""

from __future__ import annotations

import argparse
import io
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


DEFAULT_DECODED_JSONL = Path(
    "/mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k_decoded.jsonl"
)
DEFAULT_RETRIEVAL_JSONL = Path(
    "/mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k_top3_retrieval.jsonl"
)
DEFAULT_HVM_DIR = Path(
    "/mnt/data3/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed_22w_nozoom_top3part"
)
DEFAULT_CORPUS_DATA_DIR = Path(
    "/mnt/data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_process"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/data2/wuqingman/omnisvg-train/data/sft_training_data2_encode_modified_10k_top3_contact_sheets"
)


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render_svg_to_image(svg_str: str, size: int) -> Image.Image:
    import cairosvg

    png_data = cairosvg.svg2png(
        bytestring=svg_str.encode("utf-8"),
        output_width=size,
        output_height=size,
    )
    img = Image.open(io.BytesIO(png_data)).convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def load_svg_from_corpus(
    ref_idx: int,
    idx_to_meta: Dict[int, Dict],
    corpus_data_dir: Path,
    table_cache: Dict[str, object],
) -> Optional[str]:
    import pyarrow.parquet as pq

    meta = idx_to_meta.get(ref_idx)
    if meta is None:
        return None
    parquet_file = meta["parquet_file"]
    parquet_path = corpus_data_dir / parquet_file
    if parquet_file not in table_cache:
        table_cache[parquet_file] = pq.read_table(parquet_path)
    table = table_cache[parquet_file]
    return table.column("svg")[meta["parquet_row"]].as_py()


def make_placeholder(size: int, title: str, subtitle: str = "") -> Image.Image:
    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, size - 1, size - 1), outline=(180, 180, 180), width=2)
    lines = [title]
    if subtitle:
        lines.extend(textwrap.wrap(subtitle, width=28)[:4])
    y = 16
    for line in lines:
        draw.text((12, y), line, fill=(60, 60, 60), font=font)
        y += 16
    return img


def make_panel(
    image: Image.Image,
    title_lines: List[str],
    panel_size: int,
    caption_height: int,
) -> Image.Image:
    canvas = Image.new("RGB", (panel_size, panel_size + caption_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    if image.size != (panel_size, panel_size):
        image = image.resize((panel_size, panel_size))
    canvas.paste(image, (0, 0))
    draw.rectangle((0, 0, panel_size - 1, panel_size - 1), outline=(180, 180, 180), width=2)

    y = panel_size + 8
    for line in title_lines:
        draw.text((8, y), line, fill=(20, 20, 20), font=font)
        y += 14
    return canvas


def build_contact_sheet(
    sample_id: int,
    query_text: str,
    gt_svg: str,
    retrieval_row: Dict,
    idx_to_meta: Dict[int, Dict],
    corpus_data_dir: Path,
    table_cache: Dict[str, object],
    panel_size: int,
) -> Image.Image:
    caption_height = 56
    margin = 16
    gap = 12
    header_height = 92
    num_panels = 4
    width = margin * 2 + num_panels * panel_size + (num_panels - 1) * gap
    height = header_height + panel_size + caption_height + margin
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    header_title = f"sample_id={sample_id:05d}"
    header_text = textwrap.fill(query_text, width=100)
    draw.text((margin, 12), header_title, fill=(10, 10, 10), font=font)
    draw.text((margin, 30), header_text, fill=(30, 30, 30), font=font)

    panels: List[Image.Image] = []

    try:
        gt_image = render_svg_to_image(gt_svg, size=panel_size)
    except Exception as exc:  # noqa: BLE001
        gt_image = make_placeholder(panel_size, "GT render failed", str(exc))
    panels.append(make_panel(gt_image, ["GT"], panel_size, caption_height))

    ref_indices = retrieval_row["ref_indices"]
    ref_scores = retrieval_row.get("ref_scores", [])
    ref_ids = retrieval_row.get("ref_ids", [])
    ref_descriptions = retrieval_row.get("ref_descriptions", [])

    for ref_pos, ref_idx in enumerate(ref_indices[:3]):
        ref_meta = idx_to_meta.get(ref_idx, {})
        ref_id = ref_ids[ref_pos] if ref_pos < len(ref_ids) else ref_meta.get("id", "unknown")
        score = ref_scores[ref_pos] if ref_pos < len(ref_scores) else None
        description = ref_descriptions[ref_pos] if ref_pos < len(ref_descriptions) else ref_meta.get("description", "")
        try:
            ref_svg = load_svg_from_corpus(ref_idx, idx_to_meta, corpus_data_dir, table_cache)
            if not ref_svg:
                raise ValueError("empty ref svg")
            ref_image = render_svg_to_image(ref_svg, size=panel_size)
        except Exception as exc:  # noqa: BLE001
            ref_image = make_placeholder(panel_size, f"Ref{ref_pos + 1} render failed", str(exc))

        score_text = f"score={score:.4f}" if score is not None else "score=NA"
        desc_lines = textwrap.wrap(description, width=28)[:1]
        title_lines = [f"Ref{ref_pos + 1}: {ref_id}", score_text] + desc_lines
        panels.append(make_panel(ref_image, title_lines, panel_size, caption_height))

    x = margin
    y = header_height
    for panel in panels:
        canvas.paste(panel, (x, y))
        x += panel.width + gap
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GT and top-3 retrieval contact sheets.")
    parser.add_argument("--decoded_jsonl", type=Path, default=DEFAULT_DECODED_JSONL)
    parser.add_argument("--retrieval_jsonl", type=Path, default=DEFAULT_RETRIEVAL_JSONL)
    parser.add_argument("--hvm_dir", type=Path, default=DEFAULT_HVM_DIR)
    parser.add_argument("--corpus_data_dir", type=Path, default=DEFAULT_CORPUS_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--panel_size", type=int, default=320)
    parser.add_argument("--sample_ids", type=str, default=None, help="Comma-separated sample ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decoded_rows = load_jsonl(args.decoded_jsonl)
    retrieval_rows = load_jsonl(args.retrieval_jsonl)
    corpus_meta = load_jsonl(args.hvm_dir / "metadata.jsonl")

    decoded_by_id = {int(row["id"]): row for row in decoded_rows}
    retrieval_by_id = {int(row["id"]): row for row in retrieval_rows}
    idx_to_meta = {int(row["idx"]): row for row in corpus_meta}
    table_cache: Dict[str, object] = {}

    if args.sample_ids:
        target_ids = [int(x) for x in args.sample_ids.split(",") if x.strip()]
    else:
        sorted_ids = sorted(decoded_by_id.keys())
        target_ids = sorted_ids[args.offset : args.offset + args.limit]

    index_rows = []
    ok = 0
    for sample_id in target_ids:
        decoded = decoded_by_id.get(sample_id)
        retrieved = retrieval_by_id.get(sample_id)
        if decoded is None or retrieved is None:
            continue

        sheet = build_contact_sheet(
            sample_id=sample_id,
            query_text=decoded["text"],
            gt_svg=decoded["svg"],
            retrieval_row=retrieved,
            idx_to_meta=idx_to_meta,
            corpus_data_dir=args.corpus_data_dir,
            table_cache=table_cache,
            panel_size=args.panel_size,
        )
        output_path = args.output_dir / f"{sample_id:05d}.png"
        sheet.save(output_path)
        index_rows.append(
            {
                "id": sample_id,
                "image": str(output_path),
                "text": decoded["text"],
                "ref_indices": retrieved["ref_indices"],
                "ref_ids": retrieved.get("ref_ids", []),
                "ref_scores": retrieved.get("ref_scores", []),
            }
        )
        ok += 1
        if ok % 10 == 0 or ok == len(target_ids):
            print(f"[{ok}/{len(target_ids)}] saved {output_path.name}")

    index_path = args.output_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"Saved contact sheets : {ok}")
    print(f"Output dir           : {args.output_dir}")
    print(f"Index file           : {index_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
