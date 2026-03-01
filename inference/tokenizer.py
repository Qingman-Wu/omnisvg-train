"""
SVGTokenizer for inference: token IDs → SVG string.

Pipeline:
    process_generated_tokens(output_ids)  → xy pairs (np.ndarray)
    raster_svg(xy_pairs)                  → svg_tensors, color_tensors
    apply_colors_to_svg(svg_tensors, colors) → SVG object  (.to_str() → string)
"""

import numpy as np
import torch
import yaml
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox


class SVGTokenizer:
    """SVG tokenizer — supports both 8B and 4B models via config.yaml."""

    def __init__(self, config_path: str = "./config.yaml", model_size: str = None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_size = model_size or self.config.get("default_model_size", "8B")
        if self.model_size not in self.config.get("models", {}):
            raise ValueError(
                f"Invalid model_size: {self.model_size}. "
                f"Available: {list(self.config['models'].keys())}"
            )

        self._load_config()
        self.pixel2xy = self._create_pixel2xy_mapping()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def _get_model_specific_config(self, *keys):
        model_cfg = self.config.get("models", {}).get(self.model_size, {})
        value = model_cfg
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = None
                break
        if value is None:
            value = self.config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
        return value

    def _load_config(self):
        self.NUM_MASK_AND_EOM = self._get_model_specific_config("tokens", "num_mask_and_eom")
        self.BASE_OFFSET = self._get_model_specific_config("tokens", "base_offset")

        tokens_cfg = self.config["tokens"]
        self.NUM_SVG_END = tokens_cfg["svg_end"]
        self.NUM_END_TOKEN = tokens_cfg["num_end_token"]

        self.PIX_PAD = self._get_model_specific_config("coordinates", "pix_pad_offset")
        self.COORD_PAD = self._get_model_specific_config("coordinates", "coord_pad_offset")

        coords_cfg = self.config["coordinates"]
        self.BBOX = coords_cfg["bbox"]

        colors_cfg = self.config["colors"]
        self.COLOR_TOKEN_START_RAW = colors_cfg["color_token_start"]
        self.MAX_COLOR_TOKENS = colors_cfg["max_color_tokens"]
        self.COLOR_START_OFFSET = self._get_model_specific_config("colors", "color_start_offset")
        self.COLOR_END_OFFSET = self._get_model_specific_config("colors", "color_end_offset")

        commands_cfg = self.config["svg_commands"]
        self.CMD_MOVE = commands_cfg["move"]
        self.CMD_LINE = commands_cfg["line"]
        self.CMD_CURVE = commands_cfg["curve"]
        self.CMD_ARC = commands_cfg["arc"]
        self.CMD_CLOSE = commands_cfg["close"]

        model_cfg = self.config["model"]
        self.BOS_TOKEN_ID = model_cfg["bos_token_id"]
        self.EOS_TOKEN_ID = model_cfg["eos_token_id"]
        self.PAD_TOKEN_ID = model_cfg["pad_token_id"]

        arc_cfg = self.config.get("arc", {})
        self.ARC_PARAM_OFFSET = arc_cfg.get("param_offset", 44500)
        self.ARC_PARAM_RANGE = arc_cfg.get("param_range", 100)
        self.ARC_PARAM_START = self.ARC_PARAM_OFFSET + self.BASE_OFFSET

        # Derived
        self.PIXEL_OFFSET = (
            self.NUM_MASK_AND_EOM - self.BASE_OFFSET + self.NUM_SVG_END - self.CMD_MOVE
        )
        self.CMD_TOKEN_START = self.NUM_MASK_AND_EOM + self.NUM_SVG_END
        self.CMD_TOKEN_END = self.PIX_PAD + self.NUM_SVG_END
        self.COORD_TOKEN_START = self.PIX_PAD + self.NUM_SVG_END
        self.COLOR_COORD_BOUNDARY = self.COLOR_TOKEN_START_RAW + 1 + self.BASE_OFFSET
        self.COLOR_THRESHOLD = self.COLOR_TOKEN_START_RAW - self.PIXEL_OFFSET + 1

    def _create_pixel2xy_mapping(self) -> Dict[int, np.ndarray]:
        pixel2xy: Dict[int, np.ndarray] = {}
        x = np.linspace(0, self.BBOX - 1, self.BBOX)
        y = np.linspace(0, self.BBOX - 1, self.BBOX)
        xx, yy = np.meshgrid(x, y)
        xy_grid = np.array((xx.ravel(), yy.ravel())).T.astype(int)
        for pixel, xy in enumerate(xy_grid):
            pixel2xy[pixel] = xy + self.COORD_PAD + self.NUM_SVG_END
        return pixel2xy

    # ------------------------------------------------------------------
    # Token → colour
    # ------------------------------------------------------------------
    def token_to_color(self, color_token: int) -> str:
        try:
            if color_token == self.COLOR_TOKEN_START_RAW:
                return "none"
            if color_token == self.COLOR_TOKEN_START_RAW + 1:
                return "currentColor"
            color_index = color_token - (self.COLOR_TOKEN_START_RAW + 2)
            if color_index < 0 or color_index >= self.MAX_COLOR_TOKENS:
                return "#808080"
            r = (color_index >> 8) & 0xF
            g = (color_index >> 4) & 0xF
            b = color_index & 0xF
            r = (r << 4) | r
            g = (g << 4) | g
            b = (b << 4) | b
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return "#808080"

    # ------------------------------------------------------------------
    # Step 1: raw tokens → xy pairs
    # ------------------------------------------------------------------
    def process_generated_tokens(self, output_ids: torch.Tensor) -> np.ndarray:
        """
        Args:
            output_ids: [1, L] tensor **with BOS at front and EOS at back**.
        Returns:
            np.ndarray of shape [N, 2]  (xy pairs).
        """
        generated_pixels = output_ids[:, 1:-1].cpu().numpy().flatten()
        sample_xys: list = []

        for pixel in generated_pixels:
            try:
                if self.CMD_TOKEN_START <= pixel < self.CMD_TOKEN_END:
                    xy = np.array([pixel - self.BASE_OFFSET,
                                   pixel - self.BASE_OFFSET], dtype=int)
                    sample_xys.append(xy)
                elif self.COORD_TOKEN_START <= pixel < self.COLOR_COORD_BOUNDARY:
                    pixel_index = pixel - self.COORD_TOKEN_START
                    if pixel_index in self.pixel2xy:
                        xy = self.pixel2xy[pixel_index] - self.BASE_OFFSET
                        sample_xys.append(xy)
                elif self.ARC_PARAM_START + 1 <= pixel < self.ARC_PARAM_START + 1 + self.ARC_PARAM_RANGE:
                    value = pixel - self.ARC_PARAM_START - 1
                    sample_xys.append(np.array([value, value], dtype=int))
                elif self.COLOR_COORD_BOUNDARY <= pixel < self.ARC_PARAM_START:
                    xy = np.array([pixel - self.BASE_OFFSET,
                                   pixel - self.BASE_OFFSET], dtype=int)
                    sample_xys.append(xy)
            except Exception:
                continue

        if sample_xys:
            return np.vstack(sample_xys)
        return np.array([], dtype=int).reshape(0, 2)

    # ------------------------------------------------------------------
    # Step 2: xy pairs → SVG tensors + colour tokens
    # ------------------------------------------------------------------
    def raster_svg(
        self, pixels: np.ndarray
    ) -> Tuple[List[List[torch.Tensor]], List[int]]:
        try:
            if len(pixels) == 0:
                return [[]], []

            pixels = pixels - self.PIXEL_OFFSET
            svg_tensors: List[torch.Tensor] = []
            color_tensors: List[int] = []
            path_tensor: list = []

            i = 0
            while i < len(pixels):
                try:
                    pix = pixels[i]

                    if pix[0] == self.CMD_MOVE:
                        if i + 2 >= len(pixels):
                            break
                        cmd = np.zeros(14)
                        cmd[0] = 0
                        cmd[12:14] = pixels[i + 2]
                        path_tensor.append(cmd.tolist())
                        i += 3

                    elif pix[0] == self.CMD_LINE:
                        if i + 1 >= len(pixels):
                            break
                        cmd = np.zeros(14)
                        cmd[0] = 1
                        cmd[12:14] = pixels[i + 1]
                        path_tensor.append(cmd.tolist())
                        i += 2

                    elif pix[0] == self.CMD_CURVE:
                        if i + 3 >= len(pixels):
                            break
                        cmd = np.zeros(14)
                        cmd[0] = 2
                        cmd[8:10] = pixels[i + 1]
                        cmd[10:12] = pixels[i + 2]
                        cmd[12:14] = pixels[i + 3]
                        path_tensor.append(cmd.tolist())
                        i += 4

                    elif pix[0] == self.CMD_ARC:
                        if i + 5 >= len(pixels):
                            break
                        cmd = np.zeros(14)
                        cmd[0] = 3
                        cmd[1:3] = pixels[i + 1]
                        cmd[3] = pixels[i + 2][0] + self.PIXEL_OFFSET
                        cmd[4] = pixels[i + 3][0] + self.PIXEL_OFFSET
                        cmd[5] = pixels[i + 4][0] + self.PIXEL_OFFSET
                        cmd[12:14] = pixels[i + 5]
                        path_tensor.append(cmd.tolist())
                        i += 6

                    elif pix[0] == self.CMD_CLOSE:
                        if i + 1 >= len(pixels):
                            break
                        cmd = np.zeros(14)
                        cmd[0] = 6
                        cmd[12:14] = pixels[i + 1]
                        path_tensor.append(cmd.tolist())
                        i += 2

                    elif pix[0] >= self.COLOR_THRESHOLD:
                        if path_tensor:
                            svg_tensors.append(torch.tensor(path_tensor))
                            color_tensors.append(int(pix[0] + self.PIXEL_OFFSET - 1))
                            path_tensor = []
                        i += 1
                    else:
                        i += 1

                except (IndexError, TypeError):
                    break

            if path_tensor:
                svg_tensors.append(torch.tensor(path_tensor))

            return [svg_tensors], color_tensors

        except Exception as e:
            print(f"Error in raster_svg: {e}")
            return [[]], []

    # ------------------------------------------------------------------
    # Step 3: SVG tensors + colours → SVG object
    # ------------------------------------------------------------------
    def apply_colors_to_svg(
        self,
        svg_tensors: List[torch.Tensor],
        colors: Optional[List[int]],
    ) -> SVG:
        paths = []
        if not svg_tensors:
            raise ValueError("No valid SVG tensors")
        colors = colors or []

        for i, path_tensor in enumerate(svg_tensors):
            try:
                path = SVGTensor.from_data(path_tensor)
                path = SVG.from_tensor(path.data, viewbox=Bbox(self.BBOX))
                actual_color = self.token_to_color(colors[i]) if i < len(colors) else "none"
                for pg in path:
                    pg.color = actual_color
                    pg.stroke_color = "none"
                path.fill_(True)
                paths.append(path)
            except Exception as e:
                print(f"Warning: skip path {i}: {e}")
                continue

        if not paths:
            raise ValueError("No valid paths generated")

        path_groups = paths[0].svg_path_groups
        for p in paths[1:]:
            path_groups.extend(p.svg_path_groups)
        return SVG(path_groups, viewbox=Bbox(self.BBOX))
