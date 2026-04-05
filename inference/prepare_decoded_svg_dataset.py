#!/usr/bin/env python3
"""
将 encode SVG 的 JSON 数据集解码为仅包含文本和可渲染 SVG 的 JSONL。

默认输入:
  /mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k.json

默认输出:
  /mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k_decoded.jsonl

失败日志:
  /mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k_decode_errors.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import re
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image


TOKEN_MAP = {
    "svg": ("<|SVG_START|>", "<|SVG_END|>"),
    "g": ("<|GROUP_START|>", "<|GROUP_END|>"),
    "path": ("<|PATH_START|>", "<|PATH_END|>"),
    "circle": ("<|CIRCLE_START|>", "<|CIRCLE_END|>"),
    "rect": ("<|RECT_START|>", "<|RECT_END|>"),
    "ellipse": ("<|ELLIPSE_START|>", "<|ELLIPSE_END|>"),
    "M": "<|MOVE_TO_ABS|>",
    "L": "<|LINE_TO_ABS|>",
    "H": "<|HORIZONTAL_LINE_ABS|>",
    "V": "<|VERTICAL_LINE_ABS|>",
    "C": "<|CUBIC_CURVE_ABS|>",
    "S": "<|SMOOTH_CUBIC_ABS|>",
    "Q": "<|QUAD_CURVE_ABS|>",
    "T": "<|SMOOTH_QUAD_ABS|>",
    "A": "<|ELLIPTICAL_ARC_ABS|>",
    "Z": "<|PATH_CLOSE|>",
    "m": "<|MOVE_TO_REL|>",
    "l": "<|LINE_TO_REL|>",
    "h": "<|HORIZONTAL_LINE_REL|>",
    "v": "<|VERTICAL_LINE_REL|>",
    "c": "<|CUBIC_CURVE_REL|>",
    "s": "<|SMOOTH_CUBIC_REL|>",
    "q": "<|QUAD_CURVE_REL|>",
    "t": "<|SMOOTH_QUAD_REL|>",
    "a": "<|ELLIPTICAL_ARC_REL|>",
    "z": "<|PATH_CLOSE|>",
}

SVG_SPECIAL_TOKENS = {
    "root": "<|SVG_START_ROOT|>",
    "viewbox": "<|SVG_VIEWBOX|>",
}

ATTRIBUTE_TOKEN_MAP = {
    "d": "<|PATH_DATA|>",
    "fill": "<|FILL_COLOR|>",
    "fill-opacity": "<|FILL_OPACITY|>",
    "stroke": "<|STROKE_COLOR|>",
    "stroke-width": "<|STROKE_WIDTH|>",
    "cx": "<|CENTER_X|>",
    "cy": "<|CENTER_Y|>",
    "r": "<|RADIUS|>",
    "stroke-linecap": "<|STROKE_LINECAP|>",
    "stroke-linejoin": "<|STROKE_LINEJOIN|>",
    "opacity": "<|OPACITY|>",
    "rx": "<|RADIUS_X|>",
    "fill-rule": "<|FILL_RULE|>",
    "y": "<|Y_COORD|>",
    "x": "<|X_COORD|>",
    "ry": "<|RADIUS_Y|>",
    "stroke-opacity": "<|STROKE_OPACITY|>",
}

REVERSE_TOKEN_MAP = {v: k for k, v in TOKEN_MAP.items() if isinstance(v, str)}
REVERSE_ATTRIBUTE_MAP = {v: k for k, v in ATTRIBUTE_TOKEN_MAP.items()}
REVERSE_STRUCTURE_MAP = {v[0]: k for k, v in TOKEN_MAP.items() if isinstance(v, tuple)}
REVERSE_STRUCTURE_MAP.update({v[1]: k for k, v in TOKEN_MAP.items() if isinstance(v, tuple)})

DECODE_TOKEN_REGEX = re.compile(r'<\|[^|]+\|>|"[^"]*"|\S+')

DEFAULT_INPUT = Path(
    "/mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k.json"
)
DEFAULT_OUTPUT = Path(
    "/mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k_decoded.jsonl"
)
DEFAULT_ERROR_LOG = Path(
    "/mnt/data2/wuqingman/LLaMA-Factory/data/sft_training_data2_encode_modified_10k_decode_errors.jsonl"
)
DEFAULT_RENDER_SIZE = 512


def detokenize_path_d(tokenized_d_string: str) -> str:
    cleaned_string = re.sub(r"\s+\.?$", "", tokenized_d_string.strip())
    parts = cleaned_string.split()
    return " ".join(REVERSE_TOKEN_MAP.get(part, part) for part in parts)


def decode_svg(token_string: str) -> str:
    if token_string.count('"') % 2 != 0:
        token_string += '"'

    tokens = DECODE_TOKEN_REGEX.findall(token_string.strip())
    if (
        len(tokens) < 4
        or tokens[0] != "<|SVG_START|>"
        or tokens[1] != SVG_SPECIAL_TOKENS["root"]
        or tokens[2] != SVG_SPECIAL_TOKENS["viewbox"]
    ):
        raise ValueError("invalid svg token prefix")

    viewbox_value = tokens[3].strip('"')
    root = ET.Element("svg")
    root.set("viewBox", viewbox_value)
    root.set("xmlns", "http://www.w3.org/2000/svg")
    element_stack = [root]

    i = 4
    while i < len(tokens):
        token = tokens[i]
        current_element = element_stack[-1]

        if token.endswith("_START|>") and token in REVERSE_STRUCTURE_MAP:
            tag = REVERSE_STRUCTURE_MAP[token]
            new_element = ET.Element(tag)
            current_element.append(new_element)
            element_stack.append(new_element)
        elif token.endswith("_END|>") and token in REVERSE_STRUCTURE_MAP:
            if len(element_stack) > 1:
                element_stack.pop()
        elif token in REVERSE_ATTRIBUTE_MAP:
            if i + 1 < len(tokens):
                attr_name = REVERSE_ATTRIBUTE_MAP[token]
                attr_value = tokens[i + 1].strip('"')
                if attr_name == "d":
                    current_element.set(attr_name, detokenize_path_d(attr_value))
                else:
                    current_element.set(attr_name, attr_value)
                i += 1
        elif "=" in token:
            name, value = token.split("=", 1)
            current_element.set(name, value.strip('"'))
        i += 1

    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    return ET.tostring(root, encoding="unicode", method="xml")


def render_svg_to_image(svg_str: str, size: int = DEFAULT_RENDER_SIZE) -> Image.Image:
    import cairosvg

    png_data = cairosvg.svg2png(
        bytestring=svg_str.encode("utf-8"),
        output_width=size,
        output_height=size,
    )
    image = Image.open(io.BytesIO(png_data)).convert("RGBA")
    image.load()
    return image


def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a list in {path}, got {type(data).__name__}")
    return data


def iter_records(records: List[Dict], limit: int | None) -> Iterable[Tuple[int, Dict]]:
    if limit is None:
        yield from enumerate(records)
    else:
        for idx, record in enumerate(records[:limit]):
            yield idx, record


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode tokenized SVG dataset and validate by rendering.")
    parser.add_argument("--input_json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--error_log", type=Path, default=DEFAULT_ERROR_LOG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--render_size", type=int, default=DEFAULT_RENDER_SIZE)
    parser.add_argument("--progress_every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.error_log.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(args.input_json)
    total = min(len(records), args.limit) if args.limit is not None else len(records)
    print(f"Loaded {len(records)} records from {args.input_json}")
    print(f"Processing {total} records")

    decoded_rows: List[Dict] = []
    error_rows: List[Dict] = []

    for processed, (idx, record) in enumerate(iter_records(records, args.limit), start=1):
        sample_id = idx
        text = record.get("input", "")
        instruction = record.get("instruction", "")
        encoded_svg = record.get("output", "")

        try:
            svg = decode_svg(encoded_svg)
        except Exception as exc:  # noqa: BLE001
            error_rows.append(
                {
                    "id": sample_id,
                    "stage": "decode",
                    "error": str(exc),
                    "text": text,
                }
            )
            if processed % args.progress_every == 0 or processed == total:
                print(
                    f"[{processed}/{total}] ok={len(decoded_rows)} "
                    f"fail={len(error_rows)} last=decode_error id={sample_id}"
                )
            continue

        try:
            render_svg_to_image(svg, size=args.render_size)
        except Exception as exc:  # noqa: BLE001
            error_rows.append(
                {
                    "id": sample_id,
                    "stage": "render",
                    "error": str(exc),
                    "text": text,
                    "traceback": traceback.format_exc(limit=1),
                }
            )
            if processed % args.progress_every == 0 or processed == total:
                print(
                    f"[{processed}/{total}] ok={len(decoded_rows)} "
                    f"fail={len(error_rows)} last=render_error id={sample_id}"
                )
            continue

        decoded_rows.append(
            {
                "id": sample_id,
                "instruction": instruction,
                "text": text,
                "svg": svg,
            }
        )

        if processed % args.progress_every == 0 or processed == total:
            print(
                f"[{processed}/{total}] ok={len(decoded_rows)} "
                f"fail={len(error_rows)} last=ok id={sample_id}"
            )

    ok_count = write_jsonl(args.output_jsonl, decoded_rows)
    fail_count = write_jsonl(args.error_log, error_rows)

    print("=" * 60)
    print(f"Decoded+rendered OK : {ok_count}")
    print(f"Failed              : {fail_count}")
    print(f"Output JSONL        : {args.output_jsonl}")
    print(f"Error log           : {args.error_log}")
    print("=" * 60)


if __name__ == "__main__":
    main()
