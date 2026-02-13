"""
SVG Processing Module for OmniSVG.

This module provides the core SVG class for loading, manipulating, and converting
SVG graphics to tensor representations for machine learning models.

Key Features:
    - Load SVG from files or strings
    - Convert SVG paths to tensor format
    - Color quantization and tokenization
    - SVG manipulation (translate, rotate, scale, normalize)
    - Path operations (split, merge, simplify)

Example:
    >>> svg = SVG.load_svg("icon.svg")
    >>> tensors, colors = svg.to_tensor()
    >>> svg.normalize().save_svg("normalized.svg")
"""

from __future__ import annotations

import io
import math
import os
import random
import re
from typing import List, Union, Optional, Tuple, Dict, Any
from xml.dom import expatbuilder

import cairosvg
import IPython.display as ipd
import networkx as nx
import numpy as np
import torch
from moviepy.editor import ImageClip, concatenate_videoclips, ipython_display
from PIL import Image

from .geom import Bbox, Point, Angle, union_bbox
from .svg_command import SVGCommandBezier
from .svg_path import SVGPath, Filling, Orientation
from .svg_primitive import (
    SVGPathGroup, 
    SVGRectangle, 
    SVGCircle, 
    SVGEllipse, 
    SVGLine, 
    SVGPolyline, 
    SVGPolygon
)

# Type alias for numeric values
Num = Union[int, float]


# =============================================================================
# Constants and Configuration
# =============================================================================

class ColorTokenConfig:
    """
    Configuration for color tokenization.
    
    This class centralizes all constants related to color token encoding,
    making it easy to adjust the tokenization scheme.
    
    Token Layout:
        - Token 0: No color (None, 'none')
        - Token 1: currentColor
        - Tokens 2-4096: Quantized RGB colors (12-bit, 4096 values)
        - Token 4097: Gradient placeholder
    
    Attributes:
        BASE_VOCAB_SIZE: Base vocabulary size of the language model.
        COLOR_TOKEN_OFFSET: Offset for color tokens in the vocabulary.
        COLOR_TOKEN_START: Starting token ID for colors.
        MAX_COLOR_TOKENS: Maximum number of color tokens.
        TOKEN_NONE: Token ID offset for 'none' color.
        TOKEN_CURRENTCOLOR: Token ID offset for 'currentColor'.
        TOKEN_GRADIENT: Token ID offset for gradient colors.
    """
    
    # Base vocabulary configuration (matches tokenization.yaml)
    BASE_VOCAB_SIZE: int = 152064
    COLOR_TOKEN_OFFSET: int = 40011

    # Derived constants
    COLOR_TOKEN_START: int = COLOR_TOKEN_OFFSET + BASE_VOCAB_SIZE  # 191946
    MAX_COLOR_TOKENS: int = 4098
    
    # Special token offsets (relative to COLOR_TOKEN_START)
    TOKEN_NONE: int = 0           # No color / transparent
    TOKEN_CURRENTCOLOR: int = 1   # CSS currentColor
    TOKEN_GRADIENT: int = 4097    # Gradient placeholder (exceeds normal range)
    
    # Color quantization
    BITS_PER_CHANNEL: int = 4     # 4 bits per RGB channel = 12 bits total
    COLORS_PER_CHANNEL: int = 16  # 2^4 = 16 values per channel
    TOTAL_QUANTIZED_COLORS: int = 4096  # 16^3 = 4096 unique colors


# CSS Named Colors Mapping
# Reference: https://www.w3.org/TR/css-color-3/#svg-color
CSS_NAMED_COLORS: Dict[str, str] = {
    # Basic colors
    'white': '#ffffff',
    'black': '#000000',
    'red': '#ff0000',
    'green': '#008000',
    'blue': '#0000ff',
    'yellow': '#ffff00',
    'cyan': '#00ffff',
    'magenta': '#ff00ff',
    
    # Gray shades
    'silver': '#c0c0c0',
    'gray': '#808080',
    'grey': '#808080',
    
    # Extended colors
    'maroon': '#800000',
    'olive': '#808000',
    'lime': '#00ff00',
    'aqua': '#00ffff',
    'teal': '#008080',
    'navy': '#000080',
    'fuchsia': '#ff00ff',
    'purple': '#800080',
    'orange': '#ffa500',
    'pink': '#ffc0cb',
    'brown': '#a52a2a',
    'gold': '#ffd700',
    'violet': '#ee82ee',
    'indigo': '#4b0082',
    'tan': '#d2b48c',
    'beige': '#f5f5dc',
    'coral': '#ff7f50',
    'crimson': '#dc143c',
    'khaki': '#f0e68c',
    'lavender': '#e6e6fa',
    'salmon': '#fa8072',
    'turquoise': '#40e0d0',
}

# Random colors for visualization
VISUALIZATION_COLORS: List[str] = [
    "deepskyblue", "lime", "deeppink", "gold", "coral", "darkviolet",
    "royalblue", "darkmagenta", "teal", "gold", "green", "maroon",
    "aqua", "grey", "steelblue", "lime", "orange"
]


# =============================================================================
# Color Processing Utilities
# =============================================================================

class ColorProcessor:
    """
    Utility class for color parsing, normalization, and quantization.
    
    Handles various color formats:
        - Hex: #fff, #ffffff, #FFFFFF
        - RGB: rgb(255, 255, 255), rgb(100%, 100%, 100%)
        - Named: white, black, red, etc.
        - Special: none, currentColor, url(#gradient)
    """
    
    # Regex pattern for RGB color format
    RGB_PATTERN = re.compile(
        r'rgb\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)',
        re.IGNORECASE
    )
    
    @staticmethod
    def parse_rgb(color_str: str) -> Optional[str]:
        """
        Parse RGB format color to hex.
        
        Supports:
            - Integer values: rgb(255, 128, 0)
            - Percentage values: rgb(100%, 50%, 0%)
        
        Args:
            color_str: RGB color string to parse.
            
        Returns:
            Hex color string (#rrggbb) or None if parsing fails.
            
        Example:
            >>> ColorProcessor.parse_rgb("rgb(255, 128, 0)")
            '#ff8000'
            >>> ColorProcessor.parse_rgb("rgb(100%, 50%, 0%)")
            '#ff8000'
        """
        match = ColorProcessor.RGB_PATTERN.match(color_str.strip())
        if not match:
            return None
        
        r_str, g_str, b_str = match.groups()
        
        try:
            # Handle percentage format
            if '%' in r_str:
                r = int(float(r_str.rstrip('%')) * 255 / 100)
                g = int(float(g_str.rstrip('%')) * 255 / 100)
                b = int(float(b_str.rstrip('%')) * 255 / 100)
            else:
                # Handle integer format
                r = int(float(r_str))
                g = int(float(g_str))
                b = int(float(b_str))
            
            # Clamp values to valid range
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            
            return f'#{r:02x}{g:02x}{b:02x}'
            
        except (ValueError, TypeError) as e:
            print(f"Warning: Failed to parse RGB values from '{color_str}': {e}")
            return None
    
    @staticmethod
    def normalize_hex(color_hex: str) -> str:
        """
        Normalize hex color to standard 6-digit lowercase format.
        
        Args:
            color_hex: Hex color string (e.g., '#fff', '#FFF', '#ffffff').
            
        Returns:
            Normalized hex string (e.g., '#ffffff').
            
        Example:
            >>> ColorProcessor.normalize_hex("#fff")
            '#ffffff'
            >>> ColorProcessor.normalize_hex("#ABC")
            '#aabbcc'
        """
        if not color_hex.startswith('#'):
            return color_hex
        
        hex_code = color_hex[1:].lower()
        
        # Expand 3-digit format to 6-digit
        if len(hex_code) == 3:
            hex_code = ''.join([c * 2 for c in hex_code])
        
        if len(hex_code) == 6:
            return '#' + hex_code
        
        return color_hex
    
    @staticmethod
    def normalize(color_value: Any) -> str:
        """
        Normalize any color format to standard hex.
        
        Handles:
            - Hex colors (#fff, #ffffff)
            - RGB colors (rgb(r, g, b))
            - CSS named colors (white, black)
            - Pass-through for unrecognized formats
        
        Args:
            color_value: Color in any supported format.
            
        Returns:
            Normalized hex color string or original value if unrecognized.
        """
        if not isinstance(color_value, str):
            return str(color_value)
        
        color_str = color_value.strip()
        
        # Handle RGB format
        if color_str.lower().startswith('rgb('):
            parsed = ColorProcessor.parse_rgb(color_str)
            return parsed if parsed else color_str
        
        # Handle hex format
        if color_str.startswith('#'):
            return ColorProcessor.normalize_hex(color_str)
        
        # Handle CSS named colors
        color_lower = color_str.lower()
        if color_lower in CSS_NAMED_COLORS:
            return CSS_NAMED_COLORS[color_lower]
        
        return color_str
    
    @staticmethod
    def quantize(color_hex: str) -> int:
        """
        Quantize 24-bit RGB color to 12-bit (0-4095).
        
        Reduces each 8-bit channel to 4-bit by taking the high nibble,
        then combines into a single 12-bit value.
        
        Formula:
            quantized = (r >> 4) << 8 | (g >> 4) << 4 | (b >> 4)
        
        Args:
            color_hex: Normalized hex color string (#rrggbb).
            
        Returns:
            Quantized color value (0-4095).
            
        Example:
            >>> ColorProcessor.quantize("#ff8000")  # Orange
            3968  # (15 << 8) + (8 << 4) + 0
        """
        try:
            # Handle empty string as black (#000000)
            if not color_hex or (isinstance(color_hex, str) and color_hex.strip() == ''):
                return 0  # Black
            
            # Normalize first
            color_hex = ColorProcessor.normalize(color_hex)
            
            # Handle empty string after normalization (edge case)
            if not color_hex or color_hex.strip() == '':
                return 0  # Black
            
            if isinstance(color_hex, str) and color_hex.startswith('#') and len(color_hex) == 7:
                # Extract and quantize each channel (8-bit to 4-bit)
                r = int(color_hex[1:3], 16) >> 4
                g = int(color_hex[3:5], 16) >> 4
                b = int(color_hex[5:7], 16) >> 4
                
                # Combine into 12-bit value
                return (r << 8) + (g << 4) + b
            else:
                print(f"Warning: Non-standard color format '{color_hex}'. Using default (black).")
                return 0
                
        except (ValueError, TypeError) as e:
            print(f"Warning: Unable to parse color '{color_hex}': {e}. Using default (black).")
            return 0
    
    @staticmethod
    def to_token(color_value: Any, config: ColorTokenConfig = None) -> int:
        """
        Convert color to token ID.
        
        Token mapping:
            - None/'none' -> TOKEN_NONE
            - 'currentColor' -> TOKEN_CURRENTCOLOR
            - 'url(...)' (gradient) -> TOKEN_GRADIENT
            - Valid colors -> quantized value + 2
        
        Args:
            color_value: Color in any supported format.
            config: Token configuration (uses default if None).
            
        Returns:
            Token ID for the color.
        """
        if config is None:
            config = ColorTokenConfig()
        
        start = config.COLOR_TOKEN_START
        
        # Handle None or 'none'
        if color_value is None:
            return start + config.TOKEN_NONE
        
        if isinstance(color_value, str):
            color_str = color_value.strip().lower()
            
            if color_str == 'none':
                return start + config.TOKEN_NONE
            
            if color_str == 'currentcolor':
                return start + config.TOKEN_CURRENTCOLOR
            
            if color_str.startswith('url('):
                print(f"Warning: Gradient color detected '{color_value}'. Using gradient token.")
                return start + config.TOKEN_GRADIENT
            
            # Quantize and map to token
            color_id = ColorProcessor.quantize(color_value)
            # Offset by 2 because 0 and 1 are special tokens
            return start + 2 + min(color_id, config.MAX_COLOR_TOKENS - 3)
        
        # Handle other types by converting to string
        color_id = ColorProcessor.quantize(str(color_value))
        return start + 2 + min(color_id, config.MAX_COLOR_TOKENS - 3)


# =============================================================================
# Main SVG Class
# =============================================================================

class SVG:
    """
    Main class for SVG processing and manipulation.
    
    This class provides comprehensive functionality for:
        - Loading SVG from files or strings
        - Converting to/from tensor representations
        - Geometric transformations (translate, rotate, scale)
        - Path operations (split, merge, simplify)
        - Visualization and export
    
    Attributes:
        svg_path_groups: List of path groups in the SVG.
        viewbox: Bounding box defining the SVG coordinate system.
    
    Example:
        >>> # Load and process SVG
        >>> svg = SVG.load_svg("icon.svg")
        >>> svg.normalize()
        >>> tensors, colors = svg.to_tensor()
        
        >>> # Create from tensor
        >>> new_svg = SVG.from_tensor(tensors)
        >>> new_svg.save_svg("output.svg")
    """
    
    # Supported SVG primitives and their parser classes
    SUPPORTED_PRIMITIVES = {
        "path": SVGPath,
        "rect": SVGRectangle,
        "circle": SVGCircle,
        "ellipse": SVGEllipse,
        "line": SVGLine,
        "polyline": SVGPolyline,
        "polygon": SVGPolygon
    }
    
    def __init__(
        self, 
        svg_path_groups: List[SVGPathGroup], 
        viewbox: Optional[Bbox] = None
    ):
        """
        Initialize SVG instance.
        
        Args:
            svg_path_groups: List of SVGPathGroup objects.
            viewbox: Bounding box for the SVG. Defaults to 24x24.
        """
        self.svg_path_groups = svg_path_groups
        self.viewbox = viewbox if viewbox is not None else Bbox(24)
        
        # Color processing configuration
        self._color_config = ColorTokenConfig()
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def paths(self):
        """
        Iterator over all paths in all path groups.
        
        Yields:
            SVGPath objects from all groups.
        """
        for path_group in self.svg_path_groups:
            for path in path_group.svg_paths:
                yield path
    
    @property
    def start_pos(self) -> Point:
        """Get the starting position (always origin)."""
        return Point(0.)
    
    @property
    def end_pos(self) -> Point:
        """Get the end position of the last path group."""
        if not self.svg_path_groups:
            return Point(0.)
        return self.svg_path_groups[-1].end_pos
    
    # =========================================================================
    # Magic Methods
    # =========================================================================
    
    def __add__(self, other: SVG) -> SVG:
        """Combine two SVGs by concatenating their path groups."""
        svg = self.copy()
        svg.svg_path_groups.extend(other.svg_path_groups)
        return svg
    
    def __getitem__(self, idx):
        """
        Access path groups by index.
        
        Args:
            idx: Integer index or tuple (group_idx, path_idx).
            
        Returns:
            SVGPathGroup or SVGPath depending on index type.
        """
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise IndexError("Dimension out of range")
            i, j = idx
            return self.svg_path_groups[i][j]
        return self.svg_path_groups[idx]
    
    def __len__(self) -> int:
        """Return number of path groups."""
        return len(self.svg_path_groups)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        path_strs = ",\n".join([f"\t{pg}" for pg in self.svg_path_groups])
        return f"SVG[{self.viewbox}](\n{path_strs}\n)"
    
    # =========================================================================
    # Loading Methods
    # =========================================================================
    
    @staticmethod
    def load_svg(file_path: str) -> SVG:
        """
        Load SVG from file.
        
        Args:
            file_path: Path to the SVG file.
            
        Returns:
            Parsed SVG instance.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If SVG parsing fails.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return SVG.from_str(f.read())
    
    @staticmethod
    def from_str(svg_str: str) -> SVG:
        """
        Parse SVG from string.
        
        Args:
            svg_str: SVG content as string.
            
        Returns:
            Parsed SVG instance.
        """
        svg_path_groups = []
        svg_dom = expatbuilder.parseString(svg_str, False)
        svg_root = svg_dom.getElementsByTagName('svg')[0]

        # Parse viewBox
        viewbox_str = svg_root.getAttribute("viewBox")
        viewbox_list = list(map(float, viewbox_str.split(" ")))
        view_box = Bbox(*viewbox_list)

        # Parse all supported primitives
        for tag, PrimitiveClass in SVG.SUPPORTED_PRIMITIVES.items():
            for element in svg_dom.getElementsByTagName(tag):
                svg_path_groups.append(PrimitiveClass.from_xml(element))

        return SVG(svg_path_groups, view_box)
    
    @staticmethod
    def load_splineset(
        spline_str: str, 
        width: Num, 
        height: Num, 
        add_closing: bool = True
    ) -> SVG:
        """
        Load SVG from FontForge SplineSet format.
        
        Args:
            spline_str: SplineSet string content.
            width: Width of the glyph.
            height: Height of the glyph.
            add_closing: Whether to add closing command.
            
        Returns:
            Parsed SVG instance.
            
        Raises:
            ValueError: If not a valid SplineSet or empty.
        """
        if "SplineSet" not in spline_str:
            raise ValueError("Not a SplineSet")

        spline = spline_str[
            spline_str.index('SplineSet') + 10:
            spline_str.index('EndSplineSet')
        ]
        svg_str = SVG._spline_to_svg_str(spline, height)

        if not svg_str:
            raise ValueError("Empty SplineSet")

        svg_path_group = SVGPath.from_str(svg_str, add_closing=add_closing)
        return SVG([svg_path_group], viewbox=Bbox(width, height))
    
    @staticmethod
    def _spline_to_svg_str(
        spline_str: str, 
        height: Num, 
        replace_with_prev: bool = False
    ) -> str:
        """
        Convert SplineSet format to SVG path string.
        
        Args:
            spline_str: SplineSet content.
            height: Height for Y-coordinate transformation.
            replace_with_prev: Whether to replace first control point with previous.
            
        Returns:
            SVG path d attribute string.
        """
        path = []
        prev_xy = []
        
        for line in spline_str.splitlines():
            if not line:
                continue
                
            tokens = line.split(' ')
            cmd = tokens[-2]
            
            if cmd not in 'cml':
                raise ValueError(f"Command not recognized: {cmd}")
            
            args = [float(x) for x in tokens[:-2] if x]

            if replace_with_prev and cmd == 'c':
                args[:2] = prev_xy
            prev_xy = args[-2:]

            # Transform Y coordinates (flip vertically)
            new_args = []
            for i, a in enumerate(args):
                if i % 2 == 1:
                    new_args.append(str(height - a))
                else:
                    new_args.append(str(a))

            path.extend([cmd.upper()] + new_args)
            
        return " ".join(path)
    
    # =========================================================================
    # Tensor Conversion
    # =========================================================================
    
    def to_tensor(
        self, 
        concat_groups: bool = True, 
        PAD_VAL: int = 0,
        skip_gradients: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert SVG to tensor representation.
        
        Each path is converted to a tensor of commands and coordinates,
        along with a color token.
        
        Args:
            concat_groups: If True, concatenate all groups into single tensor.
            PAD_VAL: Padding value for tensor.
            skip_gradients: If True, skip paths with gradient fills.
            
        Returns:
            Tuple of (path_tensors, color_tokens):
                - If concat_groups: (Tensor, Tensor)
                - If not: (List[Tensor], List[int])
        """
        group_tensors = []
        color_tensors = []
        
        for path_group in self.svg_path_groups:
            fill_color = path_group.path.fill
            
            # Skip gradient fills if requested
            if skip_gradients:
                if isinstance(fill_color, str) and fill_color.strip().lower().startswith('url('):
                    # Not implement for gradient fill
                    continue
            
            group_tensors.append(path_group.to_tensor(PAD_VAL=PAD_VAL))
            color_tensors.append(ColorProcessor.to_token(fill_color, self._color_config))

        if concat_groups:
            if group_tensors:
                return torch.cat(group_tensors, dim=0), torch.tensor(color_tensors)
            else:
                return torch.tensor([]), torch.tensor([])
        
        return group_tensors, color_tensors
    
    def to_fillings(self) -> List[Filling]:
        """Get filling types for all path groups."""
        return [p.path.filling for p in self.svg_path_groups]
    
    @staticmethod
    def from_tensor(
        tensor: torch.Tensor, 
        viewbox: Optional[Bbox] = None,
        allow_empty: bool = False
    ) -> SVG:
        """
        Create SVG from tensor representation.
        
        Args:
            tensor: Path tensor.
            viewbox: Bounding box. Defaults to 24x24.
            allow_empty: Whether to allow empty paths.
            
        Returns:
            SVG instance.
        """
        if viewbox is None:
            viewbox = Bbox(24)

        svg = SVG(
            [SVGPath.from_tensor(tensor, allow_empty=allow_empty)], 
            viewbox=viewbox
        )
        return svg
    
    @staticmethod
    def from_tensors(
        tensors: List[torch.Tensor], 
        viewbox: Optional[Bbox] = None,
        allow_empty: bool = False
    ) -> SVG:
        """
        Create SVG from multiple path tensors.
        
        Args:
            tensors: List of path tensors.
            viewbox: Bounding box. Defaults to 24x24.
            allow_empty: Whether to allow empty paths.
            
        Returns:
            SVG instance.
        """
        if viewbox is None:
            viewbox = Bbox(24)

        svg = SVG(
            [SVGPath.from_tensor(t, allow_empty=allow_empty) for t in tensors],
            viewbox=viewbox
        )
        return svg
    
    # =========================================================================
    # Export Methods
    # =========================================================================
    
    def save_svg(self, file_path: str) -> None:
        """
        Save SVG to file.
        
        Args:
            file_path: Output file path.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.to_str())
    
    def save_png(self, file_path: str) -> None:
        """
        Save SVG as PNG image.
        
        Args:
            file_path: Output file path.
        """
        cairosvg.svg2png(bytestring=self.to_str(), write_to=file_path)
    
    def to_str(
        self, 
        fill: bool = False,
        with_points: bool = False,
        with_handles: bool = False,
        with_bboxes: bool = False,
        with_markers: bool = False,
        color_firstlast: bool = False,
        with_moves: bool = True
    ) -> str:
        """
        Convert SVG to string representation.
        
        Args:
            fill: Whether to fill paths.
            with_points: Show control points.
            with_handles: Show bezier handles.
            with_bboxes: Show bounding boxes.
            with_markers: Show direction markers.
            color_firstlast: Color first/last points differently.
            with_moves: Include move commands.
            
        Returns:
            SVG string.
        """
        viz_elements = self._get_viz_elements(
            with_points, with_handles, with_bboxes, color_firstlast, with_moves
        )
        
        newline = "\n"
        paths_str = newline.join(
            pg.to_str(fill=fill, with_markers=with_markers) 
            for pg in [*self.svg_path_groups, *viz_elements]
        )
        
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{self.viewbox.to_str()}" height="200px" width="200px">'
            f'{self._markers() if with_markers else ""}'
            f'{paths_str}'
            '</svg>'
        )
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def draw(
        self, 
        fill: bool = False,
        file_path: Optional[str] = None,
        do_display: bool = True,
        return_png: bool = False,
        with_points: bool = False,
        with_handles: bool = False,
        with_bboxes: bool = False,
        with_markers: bool = False,
        color_firstlast: bool = False,
        with_moves: bool = True
    ) -> Optional[Image.Image]:
        """
        Draw and optionally display/save the SVG.
        
        Args:
            fill: Whether to fill paths.
            file_path: Optional path to save (supports .svg, .png).
            do_display: Whether to display in notebook.
            return_png: Whether to return PIL Image.
            with_points: Show control points.
            with_handles: Show bezier handles.
            with_bboxes: Show bounding boxes.
            with_markers: Show direction markers.
            color_firstlast: Color first/last points differently.
            with_moves: Include move commands.
            
        Returns:
            PIL Image if return_png=True, else None.
        """
        # Save to file if path provided
        if file_path is not None:
            _, ext = os.path.splitext(file_path)
            if ext == ".svg":
                self.save_svg(file_path)
            elif ext == ".png":
                self.save_png(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        svg_str = self.to_str(
            fill=fill,
            with_points=with_points,
            with_handles=with_handles,
            with_bboxes=with_bboxes,
            with_markers=with_markers,
            color_firstlast=color_firstlast,
            with_moves=with_moves
        )

        # Display in notebook
        if do_display:
            ipd.display(ipd.SVG(svg_str))

        # Return PNG image
        if return_png:
            if file_path is None:
                img_data = cairosvg.svg2png(bytestring=svg_str)
                return Image.open(io.BytesIO(img_data))
            else:
                _, ext = os.path.splitext(file_path)
                if ext == ".svg":
                    img_data = cairosvg.svg2png(url=file_path)
                    return Image.open(io.BytesIO(img_data))
                else:
                    return Image.open(file_path)
        
        return None
    
    def draw_colored(self, *args, **kwargs) -> None:
        """Draw SVG with random colors for each path."""
        self.copy().normalize().split_paths().set_color("random").draw(*args, **kwargs)
    
    def _get_viz_elements(
        self,
        with_points: bool = False,
        with_handles: bool = False,
        with_bboxes: bool = False,
        color_firstlast: bool = False,
        with_moves: bool = True
    ) -> List:
        """Get visualization helper elements."""
        viz_elements = []
        for svg_path_group in self.svg_path_groups:
            viz_elements.extend(
                svg_path_group._get_viz_elements(
                    with_points, with_handles, with_bboxes, color_firstlast, with_moves
                )
            )
        return viz_elements
    
    def _markers(self) -> str:
        """SVG markers definition for direction arrows."""
        return (
            '<defs>'
            '<marker id="arrow" viewBox="0 0 10 10" markerWidth="4" markerHeight="4" '
            'refX="0" refY="3" orient="auto" markerUnits="strokeWidth">'
            '<path d="M0,0 L0,6 L9,3 z" fill="#f00" />'
            '</marker>'
            '</defs>'
        )
    
    # =========================================================================
    # Animation
    # =========================================================================
    
    def to_video(
        self, 
        wrapper, 
        color: str = "grey"
    ) -> Tuple[List, List]:
        """
        Convert SVG to video frames.
        
        Args:
            wrapper: Function to wrap image arrays.
            color: Color for path strokes.
            
        Returns:
            Tuple of (clips, svg_commands).
        """
        clips, svg_commands = [], []

        # Initial empty frame
        im = SVG([]).draw(do_display=False, return_png=True)
        clips.append(wrapper(np.array(im)))

        # Add frames for each path
        for svg_path in self.paths:
            clips, svg_commands = svg_path.to_video(
                wrapper, clips, svg_commands, color=color
            )

        # Final complete frame
        im = self.draw(do_display=False, return_png=True)
        clips.append(wrapper(np.array(im)))

        return clips
    
    def animate(
        self, 
        file_path: Optional[str] = None,
        frame_duration: float = 0.1,
        do_display: bool = True
    ) -> None:
        """
        Create and display animation of SVG drawing process.
        
        Args:
            file_path: Optional path to save GIF.
            frame_duration: Duration of each frame in seconds.
            do_display: Whether to display in notebook.
        """
        clips = self.to_video(
            lambda img: ImageClip(img).set_duration(frame_duration)
        )

        clip = concatenate_videoclips(
            clips, method="compose", bg_color=(255, 255, 255)
        )

        if file_path is not None:
            clip.write_gif(file_path, fps=24, verbose=False, logger=None)

        if do_display:
            src = clip if file_path is None else file_path
            ipd.display(ipython_display(
                src, fps=24, rd_kwargs=dict(logger=None), autoplay=1, loop=1
            ))
    
    # =========================================================================
    # Geometric Transformations
    # =========================================================================
    
    def translate(self, vec: Point) -> SVG:
        """
        Translate SVG by vector.
        
        Args:
            vec: Translation vector.
            
        Returns:
            Self for chaining.
        """
        return self._apply_to_paths("translate", vec)
    
    def rotate(
        self, 
        angle: Angle, 
        center: Optional[Point] = None
    ) -> SVG:
        """
        Rotate SVG around a center point.
        
        Args:
            angle: Rotation angle.
            center: Center of rotation. Defaults to viewbox center.
            
        Returns:
            Self for chaining.
        """
        if center is None:
            center = self.viewbox.center

        self.translate(-self.viewbox.center)
        self._apply_to_paths("rotate", angle)
        self.translate(center)

        return self
    
    def zoom(
        self, 
        factor: Num, 
        center: Optional[Point] = None
    ) -> SVG:
        """
        Zoom (scale) SVG around a center point.
        
        Args:
            factor: Scale factor.
            center: Center of scaling. Defaults to viewbox center.
            
        Returns:
            Self for chaining.
        """
        if center is None:
            center = self.viewbox.center

        self.translate(-self.viewbox.center)
        self._apply_to_paths("scale", factor)
        self.translate(center)

        return self
    
    def normalize(self, viewbox: Optional[Bbox] = None) -> SVG:
        """
        Normalize SVG to fit within a viewbox.
        
        Scales the SVG to fit within the target viewbox while
        maintaining aspect ratio.
        
        Args:
            viewbox: Target viewbox. Defaults to 24x24.
            
        Returns:
            Self for chaining.
        """
        if viewbox is None:
            viewbox = Bbox(24)

        size = self.viewbox.size
        scale_factor = viewbox.size.min() / size.max()
        self.zoom(scale_factor, viewbox.center)
        self.viewbox = viewbox

        return self
    
    def numericalize(self, n: int = 256) -> SVG:
        """
        Quantize coordinates to integer grid.
        
        Args:
            n: Grid size.
            
        Returns:
            Self for chaining.
        """
        self.normalize(viewbox=Bbox(n))
        return self._apply_to_paths("numericalize", n)
    
    # =========================================================================
    # Path Operations
    # =========================================================================
    
    def copy(self) -> SVG:
        """Create a deep copy of the SVG."""
        return SVG(
            [pg.copy() for pg in self.svg_path_groups],
            self.viewbox.copy()
        )
    
    def total_length(self) -> int:
        """Get total length (number of commands) across all paths."""
        return sum([pg.get_length() for pg in self.svg_path_groups])
    
    def empty(self) -> bool:
        """Check if SVG has no path groups."""
        return len(self.svg_path_groups) == 0
    
    def split_paths(self) -> SVG:
        """Split compound paths into separate path groups."""
        path_groups = []
        for path_group in self.svg_path_groups:
            path_groups.extend(path_group.split_paths())
        self.svg_path_groups = path_groups
        return self
    
    def merge_groups(self) -> SVG:
        """Merge all path groups into one."""
        if not self.svg_path_groups:
            return self
            
        path_group = self.svg_path_groups[0]
        for pg in self.svg_path_groups[1:]:
            path_group.svg_paths.extend(pg.svg_paths)
        self.svg_path_groups = [path_group]
        return self
    
    def drop_z(self) -> SVG:
        """Remove close (Z) commands from all paths."""
        return self._apply_to_paths("drop_z")
    
    def filter_empty(self) -> SVG:
        """Remove empty paths and path groups."""
        self._apply_to_paths("filter_empty")
        self.svg_path_groups = [
            pg for pg in self.svg_path_groups if pg.svg_paths
        ]
        return self
    
    def filter_consecutives(self) -> SVG:
        """Remove consecutive duplicate commands."""
        return self._apply_to_paths("filter_consecutives")
    
    def filter_duplicates(self) -> SVG:
        """Remove duplicate paths."""
        return self._apply_to_paths("filter_duplicates")
    
    def reverse(self) -> SVG:
        """Reverse direction of all paths."""
        self._apply_to_paths("reverse")
        return self
    
    def reverse_non_closed(self) -> SVG:
        """Reverse direction of non-closed paths only."""
        self._apply_to_paths("reverse_non_closed")
        return self
    
    def duplicate_extremities(self) -> SVG:
        """Duplicate first and last points of paths."""
        self._apply_to_paths("duplicate_extremities")
        return self
    
    def reorder(self) -> SVG:
        """Reorder path commands."""
        return self._apply_to_paths("reorder")
    
    def compute_filling(self) -> SVG:
        """Compute filling type for all paths."""
        return self._apply_to_paths("compute_filling")
    
    def recompute_origins(self) -> None:
        """Recompute origin points for all path groups."""
        origin = self.start_pos

        for path_group in self.svg_path_groups:
            path_group.set_origin(origin.copy())
            origin = path_group.end_pos
    
    # =========================================================================
    # Simplification
    # =========================================================================
    
    def simplify(
        self, 
        tolerance: float = 0.1,
        epsilon: float = 0.1,
        angle_threshold: float = 179.0,
        force_smooth: bool = False
    ) -> SVG:
        """
        Simplify paths by reducing control points.
        
        Args:
            tolerance: Simplification tolerance.
            epsilon: Epsilon for point comparison.
            angle_threshold: Angle threshold for smoothing.
            force_smooth: Force smooth curves.
            
        Returns:
            Self for chaining.
        """
        self._apply_to_paths(
            "simplify",
            tolerance=tolerance,
            epsilon=epsilon,
            angle_threshold=angle_threshold,
            force_smooth=force_smooth
        )
        self.recompute_origins()
        return self
    
    def simplify_arcs(self) -> SVG:
        """Convert arcs to bezier curves."""
        return self._apply_to_paths("simplify_arcs")
    
    def simplify_heuristic(
        self, 
        tolerance: float = 0.1,
        force_smooth: bool = False
    ) -> SVG:
        """
        Apply heuristic simplification.
        
        Combines splitting and simplification for better results.
        
        Args:
            tolerance: Simplification tolerance.
            force_smooth: Force smooth curves.
            
        Returns:
            Simplified SVG copy.
        """
        return (
            self.copy()
            .split(max_dist=2, include_lines=False)
            .simplify(
                tolerance=tolerance,
                epsilon=0.2,
                angle_threshold=150,
                force_smooth=force_smooth
            )
            .split(max_dist=7.5)
        )
    
    def simplify_heuristic2(self) -> SVG:
        """Alternative heuristic simplification."""
        return (
            self.copy()
            .split(max_dist=2, include_lines=False)
            .simplify(tolerance=0.2, epsilon=0.2, angle_threshold=150)
            .split(max_dist=7.5)
        )
    
    def split(
        self, 
        n: Optional[int] = None,
        max_dist: Optional[float] = None,
        include_lines: bool = True
    ) -> SVG:
        """
        Split paths into smaller segments.
        
        Args:
            n: Number of segments per curve.
            max_dist: Maximum distance between points.
            include_lines: Whether to split line segments.
            
        Returns:
            Self for chaining.
        """
        return self._apply_to_paths("split", n=n, max_dist=max_dist, include_lines=include_lines)
    
    # =========================================================================
    # Canonicalization
    # =========================================================================
    
    def canonicalize(self, normalize: bool = False) -> SVG:
        """
        Canonicalize SVG to standard form.
        
        Performs:
            1. Convert to paths and simplify arcs
            2. Optionally normalize
            3. Split paths
            4. Filter and reorder
            5. Canonicalize individual paths
            6. Drop close commands
        
        Args:
            normalize: Whether to normalize coordinates.
            
        Returns:
            Self for chaining.
        """
        self.to_path().simplify_arcs()

        if normalize:
            self.normalize()

        self.split_paths()
        self.filter_consecutives()
        self.filter_empty()
        self._apply_to_paths("reorder")
        
        # Sort by position (bottom-to-top, left-to-right)
        self.svg_path_groups = sorted(
            self.svg_path_groups, 
            key=lambda x: x.start_pos.tolist()[::-1]
        )
        
        self._apply_to_paths("canonicalize")
        self.recompute_origins()
        self.drop_z()

        return self
    
    def canonicalize_new(self, normalize: bool = False) -> SVG:
        """
        Canonicalize with filling computation.
        
        Same as canonicalize() but also computes filling types.
        
        Args:
            normalize: Whether to normalize coordinates.
            
        Returns:
            Self for chaining.
        """
        self.to_path().simplify_arcs()
        self.compute_filling()

        if normalize:
            self.normalize()

        self.split_paths()
        self.filter_consecutives()
        self.filter_empty()
        self._apply_to_paths("reorder")
        
        self.svg_path_groups = sorted(
            self.svg_path_groups,
            key=lambda x: x.start_pos.tolist()[::-1]
        )
        
        self._apply_to_paths("canonicalize")
        self.recompute_origins()
        self.drop_z()

        return self
    
    def to_path(self) -> SVG:
        """Convert all primitives to paths."""
        for i, path_group in enumerate(self.svg_path_groups):
            self.svg_path_groups[i] = path_group.to_path()
        return self
    
    # =========================================================================
    # Styling
    # =========================================================================
    
    def set_color(self, color: Union[str, List[str]]) -> SVG:
        """
        Set color for all path groups.
        
        Args:
            color: Color string, list of colors, "random", or "random_random".
            
        Returns:
            Self for chaining.
        """
        colors = VISUALIZATION_COLORS.copy()

        if color == "random_random":
            random.shuffle(colors)

        if isinstance(color, list):
            colors = color

        for i, path_group in enumerate(self.svg_path_groups):
            if color in ("random", "random_random") or isinstance(color, list):
                c = colors[i % len(colors)]
            else:
                c = color
            path_group.color = c
            
        return self
    
    def fill_(self, fill: bool = True) -> SVG:
        """Set fill property for all paths."""
        return self._apply_to_paths("fill_", fill)
    
    # =========================================================================
    # Geometry
    # =========================================================================
    
    def bbox(self) -> Bbox:
        """Get bounding box encompassing all paths."""
        return union_bbox([pg.bbox() for pg in self.svg_path_groups])
    
    def to_points(self, sort: bool = True) -> np.ndarray:
        """
        Extract all control points.
        
        Args:
            sort: Whether to sort and deduplicate points.
            
        Returns:
            Array of points with shape (N, 2).
        """
        points = np.concatenate([
            pg.to_points() for pg in self.svg_path_groups
        ])

        if sort:
            ind = np.lexsort((points[:, 0], points[:, 1]))
            points = points[ind]

            # Remove duplicates
            row_mask = np.append([True], np.any(np.diff(points, axis=0), 1))
            points = points[row_mask]

        return points
    
    # =========================================================================
    # Path Group Management
    # =========================================================================
    
    def add_path_group(self, path_group: SVGPathGroup) -> SVG:
        """
        Add a path group to the SVG.
        
        Args:
            path_group: Path group to add.
            
        Returns:
            Self for chaining.
        """
        path_group.set_origin(self.end_pos.copy())
        self.svg_path_groups.append(path_group)
        return self
    
    def add_path_groups(self, path_groups: List[SVGPathGroup]) -> SVG:
        """
        Add multiple path groups.
        
        Args:
            path_groups: List of path groups to add.
            
        Returns:
            Self for chaining.
        """
        for path_group in path_groups:
            self.add_path_group(path_group)
        return self
    
    def permute(self, indices: Optional[List[int]] = None) -> SVG:
        """
        Reorder path groups by indices.
        
        Args:
            indices: New order of path groups.
            
        Returns:
            Self for chaining.
        """
        if indices is not None:
            self.svg_path_groups = [self.svg_path_groups[i] for i in indices]
        return self
    
    # =========================================================================
    # Graph Analysis
    # =========================================================================
    
    def overlap_graph(
        self, 
        threshold: float = 0.95,
        draw: bool = False
    ) -> nx.DiGraph:
        """
        Build overlap graph between path groups.
        
        Creates a directed graph where edges represent overlap relationships
        (one shape contained within another).
        
        Args:
            threshold: Minimum overlap ratio to create edge.
            draw: Whether to visualize the graph.
            
        Returns:
            NetworkX DiGraph.
        """
        G = nx.DiGraph()
        shapes = [group.to_shapely() for group in self.svg_path_groups]

        for i, group1 in enumerate(shapes):
            G.add_node(i)

            if self.svg_path_groups[i].path.filling != Filling.OUTLINE:
                for j, group2 in enumerate(shapes):
                    if i != j and self.svg_path_groups[j].path.filling == Filling.FILL:
                        overlap = group1.intersection(group2).area / group1.area
                        if overlap > threshold:
                            G.add_edge(j, i, weight=overlap)

        if draw:
            pos = nx.spring_layout(G)
            nx.draw_networkx(G, pos, with_labels=True)
            labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
            
        return G
    
    def group_overlapping_paths(self) -> SVG:
        """
        Group overlapping paths based on containment.
        
        Uses the overlap graph to determine which paths should be
        grouped together (e.g., a filled shape with a hole).
        
        Returns:
            New SVG with grouped paths.
        """
        G = self.overlap_graph()

        path_groups = []
        root_nodes = [i for i, d in G.in_degree() if d == 0]

        for root in root_nodes:
            if self[root].path.filling == Filling.FILL:
                current = [root]

                while current:
                    n = current.pop(0)

                    fill_neighbors, erase_neighbors = [], []
                    for m in G.neighbors(n):
                        if G.in_degree(m) == 1:
                            if self[m].path.filling == Filling.ERASE:
                                erase_neighbors.append(m)
                            else:
                                fill_neighbors.append(m)
                    G.remove_node(n)

                    path_group = SVGPathGroup(
                        [self[n].path.copy().set_orientation(Orientation.CLOCKWISE)],
                        fill=True
                    )
                    
                    if erase_neighbors:
                        for en in erase_neighbors:
                            neighbor = self[en].path.copy().set_orientation(
                                Orientation.COUNTER_CLOCKWISE
                            )
                            path_group.append(neighbor)
                        G.remove_nodes_from(erase_neighbors)

                    path_groups.append(path_group)
                    current.extend(fill_neighbors)

        # Add outlines at the end
        for path_group in self.svg_path_groups:
            if path_group.path.filling == Filling.OUTLINE:
                path_groups.append(path_group)

        return SVG(path_groups)
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @staticmethod
    def unit_circle() -> SVG:
        """
        Create a unit circle SVG.
        
        Returns:
            SVG containing a circle with radius 0.5 centered at (0.5, 0.5).
        """
        d = 2 * (math.sqrt(2) - 1) / 3

        circle = SVGPath([
            SVGCommandBezier(Point(.5, 0.), Point(.5 + d, 0.), Point(1., .5 - d), Point(1., .5)),
            SVGCommandBezier(Point(1., .5), Point(1., .5 + d), Point(.5 + d, 1.), Point(.5, 1.)),
            SVGCommandBezier(Point(.5, 1.), Point(.5 - d, 1.), Point(0., .5 + d), Point(0., .5)),
            SVGCommandBezier(Point(0., .5), Point(0., .5 - d), Point(.5 - d, 0.), Point(.5, 0.))
        ]).to_group()

        return SVG([circle], viewbox=Bbox(1))
    
    @staticmethod
    def unit_square() -> SVG:
        """
        Create a unit square SVG.
        
        Returns:
            SVG containing a 1x1 square at origin.
        """
        square = SVGPath.from_str("m 0,0 h1 v1 h-1 v-1")
        return SVG([square], viewbox=Bbox(1))
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _apply_to_paths(self, method: str, *args, **kwargs) -> SVG:
        """
        Apply a method to all path groups.
        
        Args:
            method: Method name to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
            
        Returns:
            Self for chaining.
        """
        for path_group in self.svg_path_groups:
            getattr(path_group, method)(*args, **kwargs)
        return self
    
    # =========================================================================
    # Legacy Color Methods (Deprecated - use ColorProcessor instead)
    # =========================================================================
    
    def parse_rgb_color(self, color_str: str) -> Optional[str]:
        """
        Parse RGB color format.
        
        Deprecated: Use ColorProcessor.parse_rgb() instead.
        """
        return ColorProcessor.parse_rgb(color_str)
    
    def normalize_color_hex(self, color_hex: str) -> str:
        """
        Normalize color to standard hex format.
        
        Deprecated: Use ColorProcessor.normalize() instead.
        """
        return ColorProcessor.normalize(color_hex)
    
    def quantize_color(self, color_hex: str) -> int:
        """
        Quantize color to 12-bit value.
        
        Deprecated: Use ColorProcessor.quantize() instead.
        """
        return ColorProcessor.quantize(color_hex)
    
    def color_to_token(self, color_hex: Any) -> int:
        """
        Convert color to token ID.
        
        Deprecated: Use ColorProcessor.to_token() instead.
        """
        return ColorProcessor.to_token(color_hex, self._color_config)

