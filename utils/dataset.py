"""
Dataset classes for OmniSVG training.
Supports both local files and HuggingFace datasets.
"""

import os
import io
import re
import random
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
from functools import lru_cache

# Optional imports
try:
    from datasets import load_dataset, Dataset as HFDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from deepsvg.svglib.svg import SVG
    DEEPSVG_AVAILABLE = True
except ImportError:
    DEEPSVG_AVAILABLE = False
    print("Warning: deepsvg not available. SVG parsing may be limited.")

from .config import TokenizationConfig, TrainConfig


class SVGTokenizer:
    """
    Tokenizer for converting SVG tensors to token sequences.
    Uses configuration from TokenizationConfig.
    """
    
    def __init__(self, config: Optional[TokenizationConfig] = None):
        self.config = config or TokenizationConfig()
        
        # Build pixel to xy mapping
        self._build_pixel_mapping()
        
        # Command token IDs
        self.CMD_MOVE = self.config.cmd_move
        self.CMD_LINE = self.config.cmd_line
        self.CMD_CURVE = self.config.cmd_curve
        self.CMD_ARC = self.config.cmd_arc
        self.CMD_CLOSE = self.config.cmd_close
        
        # Offsets
        self.PIX_PAD = self.config.pix_pad_offset
        self.COORD_PAD = self.config.coord_pad_offset
        self.ARC_PARAM_START = self.config.arc_param_start
        self.BBOX = self.config.bbox_size
        
        # Special tokens
        self.BOS_TOKEN = self.config.bos_token_id
        self.EOS_TOKEN = self.config.eos_token_id
        self.MASK_TOKEN = self.config.mask_token
    
    def _build_pixel_mapping(self):
        """Build pixel to xy coordinate mapping."""
        bbox = self.config.bbox_size
        x = np.linspace(0, bbox - 1, bbox)
        y = np.linspace(0, bbox - 1, bbox)
        xx, yy = np.meshgrid(x, y)
        xy_grid = (np.array((xx.ravel(), yy.ravel())).T).astype(int)
        
        self.pixel2xy = {}
        offset = self.config.coord_pad_offset + self.config.num_svg_end
        for pixel, xy in enumerate(xy_grid):
            self.pixel2xy[pixel] = xy + offset
    
    def coord_to_index(self, coord: np.ndarray) -> int:
        """Convert 2D coordinate to linear index."""
        return int(coord[0] + coord[1] * self.BBOX)
    
   
    def tokenize_svg_tensors(
        self, 
        svg_tensors: List[torch.Tensor], 
        color_tensors: List[int]
    ) -> np.ndarray:
        """
        Convert SVG tensors to token sequence.
        
        Args:
            svg_tensors: List of path tensors from SVG
            color_tensors: List of color tokens
        
        Returns:
            Token sequence as numpy array
        """
        all_tokens = []
        
        for path_tensor, color_token in zip(svg_tensors, color_tensors):
            path_tensor = path_tensor.round().int()
            path_tensor = torch.clip(path_tensor, min=0, max=self.BBOX - 1)
            path_tokens = []
            
            for i, cmd_arg_tensor in enumerate(path_tensor):
                cmd = cmd_arg_tensor[0].item()
                start_pos = cmd_arg_tensor[6:8].numpy()
                control1 = cmd_arg_tensor[8:10].numpy()
                control2 = cmd_arg_tensor[10:12].numpy()
                end_pos = cmd_arg_tensor[12:14].numpy()
                
                if cmd == 0:  # Move
                    path_tokens.append(self.CMD_MOVE)
                    if i == 0:
                        path_tokens.append(self.coord_to_index(end_pos) + self.PIX_PAD + self.config.num_svg_end)
                        path_tokens.append(self.coord_to_index(end_pos) + self.PIX_PAD + self.config.num_svg_end)
                    else:
                        path_tokens.append(self.coord_to_index(start_pos) + self.PIX_PAD + self.config.num_svg_end)
                        path_tokens.append(self.coord_to_index(end_pos) + self.PIX_PAD + self.config.num_svg_end)
                        
                elif cmd == 1:  # Line
                    path_tokens.append(self.CMD_LINE)
                    path_tokens.append(self.coord_to_index(end_pos) + self.PIX_PAD + self.config.num_svg_end)
                    
                elif cmd == 2:  # Curve (Cubic Bezier)
                    path_tokens.append(self.CMD_CURVE)
                    path_tokens.append(self.coord_to_index(control1) + self.PIX_PAD + self.config.num_svg_end)
                    path_tokens.append(self.coord_to_index(control2) + self.PIX_PAD + self.config.num_svg_end)
                    path_tokens.append(self.coord_to_index(end_pos) + self.PIX_PAD + self.config.num_svg_end)
                    
                elif cmd == 3:  # Arc
                    radius = cmd_arg_tensor[1:3].numpy()
                    x_axis_rot = cmd_arg_tensor[3].item()
                    large_arc_flag = cmd_arg_tensor[4].item()
                    sweep_flag = cmd_arg_tensor[5].item()
                    
                    path_tokens.append(self.CMD_ARC)
                    path_tokens.append(self.coord_to_index(radius) + self.PIX_PAD + self.config.num_svg_end)
                    path_tokens.append(int(x_axis_rot) + self.ARC_PARAM_START + self.config.num_svg_end)
                    path_tokens.append(int(large_arc_flag) + self.ARC_PARAM_START + self.config.num_svg_end)
                    path_tokens.append(int(sweep_flag) + self.ARC_PARAM_START + self.config.num_svg_end)
                    path_tokens.append(self.coord_to_index(end_pos) + self.PIX_PAD + self.config.num_svg_end)
                    
                elif cmd == 6:  # Close
                    path_tokens.append(self.CMD_CLOSE)
                    path_tokens.append(self.coord_to_index(end_pos) + self.PIX_PAD + self.config.num_svg_end)
            
            # Add color token
            path_tokens.append(color_token)
            all_tokens.extend(path_tokens)
        
        return np.array(all_tokens, dtype=np.int64)
    def add_special_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """Add BOS and EOS tokens to sequence."""
        tokens = np.insert(tokens, 0, self.BOS_TOKEN)
        tokens = np.append(tokens, self.EOS_TOKEN)
        return tokens


class OmniSVGDataset(Dataset):
    """
    Dataset for OmniSVG training.
    Supports both local files and HuggingFace datasets.
    """
    
    def __init__(
        self,
        # Data source (choose one)
        meta_file: Optional[str] = None,
        svg_folder: Optional[str] = None,
        png_folder: Optional[str] = None,
        hf_dataset: Optional[Any] = None,
        
        # Configuration
        max_len: int = 2048,
        text_len: int = 800,
        tokenizer: Optional[Any] = None,
        processor: Optional[Any] = None,
        
        # Config objects
        token_config: Optional[TokenizationConfig] = None,
        train_config: Optional[TrainConfig] = None,
        
        # Options
        target_image_size: int = 448,
    ):
        """
        Initialize OmniSVG dataset.
        
        Args:
            meta_file: Path to metadata CSV (for local files)
            svg_folder: Path to SVG files folder
            png_folder: Path to PNG files folder
            hf_dataset: Pre-loaded HuggingFace dataset
            max_len: Maximum sequence length
            text_len: Maximum text length
            tokenizer: Text tokenizer
            processor: Image processor
            token_config: Tokenization configuration
            train_config: Training configuration
            target_image_size: Target image size
        """
        self.max_len = max_len
        self.text_len = text_len
        self.tokenizer = tokenizer
        self.processor = processor
        self.target_image_size = target_image_size
        
        # Load configs
        self.token_config = token_config or TokenizationConfig()
        self.train_config = train_config or TrainConfig()
        
        # Initialize SVG tokenizer
        self.svg_tokenizer = SVGTokenizer(self.token_config)
        
        # Load data
        if hf_dataset is not None:
            self._init_from_hf_dataset(hf_dataset)
        elif meta_file is not None:
            self._init_from_local_files(meta_file, svg_folder, png_folder)
        else:
            raise ValueError("Must provide either hf_dataset or meta_file")
        
        # Apply data balancing
        self._apply_data_balancing()
        
        # Causal masking setup
        self._setup_causal_masking()
        
        # Tracking
        self.skipped_samples = set()
        
        print(f"Dataset initialized with {len(self)} samples")
    
    def _init_from_hf_dataset(self, dataset):
        """Initialize from HuggingFace dataset."""
        self.data_source = "huggingface"
        self.hf_dataset = dataset
        
        # Filter by token length
        self.hf_dataset = self.hf_dataset.filter(
            lambda x: 0 < x['token_len'] <= self.max_len
        )
        
        self.original_indices = list(range(len(self.hf_dataset)))
        print(f"Loaded {len(self.original_indices)} samples from HuggingFace dataset")
    
    def _init_from_local_files(self, meta_file: str, svg_folder: str, png_folder: str):
        """Initialize from local files."""
        self.data_source = "local"
        self.svg_folder = svg_folder
        self.png_folder = png_folder
        
        # Load metadata
        self.meta_df = self._load_meta_file(meta_file)
        
        # Filter by token length
        self.meta_df = self.meta_df[
            (0 < self.meta_df['len_pix']) & 
            (self.meta_df['len_pix'] <= self.max_len)
        ]
        
        # Build file indices for fast lookup
        if svg_folder:
            self.svg_index = self._build_file_index(svg_folder, '.svg')
        if png_folder:
            self.png_index = self._build_file_index(png_folder, '.png')
        
        self.original_indices = list(range(len(self.meta_df)))
        print(f"Loaded {len(self.original_indices)} samples from local files")
    
    def _load_meta_file(self, filepath: str) -> pd.DataFrame:
        """Load metadata CSV with encoding handling."""
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                print(f"Loaded metadata with {encoding} encoding")
                return df
            except Exception:
                continue
        
        raise ValueError(f"Could not read {filepath} with any encoding")
    
    def _build_file_index(self, folder: str, extension: str) -> Dict[str, List[str]]:
        """Build file index for fast lookup."""
        index = {}
        
        for filename in os.listdir(folder):
            if filename.endswith(extension):
                match = re.match(r'^(\d+)', filename)
                if match:
                    prefix = match.group(1)
                    if prefix not in index:
                        index[prefix] = []
                    index[prefix].append(filename)
        
        return index
    
    def _apply_data_balancing(self):
        """Apply data balancing based on sequence length."""
        # Length-based duplication factors
        length_intervals = [
            (0, 512, 1.0),
            (512, 1024, 1.2),
            (1024, 2048, 1.3),
            (2048, float('inf'), 1.0),
        ]
        
        self.duplicated_indices = []
        random.seed(42)
        
        for i in self.original_indices:
            # Get token length
            if self.data_source == "huggingface":
                token_len = self.hf_dataset[i]['token_len']
            else:
                token_len = self.meta_df.iloc[i]['len_pix']
            
            # Find duplication factor
            factor = 1.0
            for min_len, max_len, f in length_intervals:
                if min_len <= token_len < max_len:
                    factor = f
                    break
            
            # Apply duplication
            base_copies = int(factor)
            fractional = factor - base_copies
            
            for _ in range(base_copies):
                self.duplicated_indices.append(i)
            
            if fractional > 0 and random.random() < fractional:
                self.duplicated_indices.append(i)
        
        print(f"Data balancing: {len(self.original_indices)} -> {len(self.duplicated_indices)}")
    
    def _setup_causal_masking(self):
        """Setup causal masking for pre-training."""
        self.mask_probability = 0.1
        self.sentinel_tokens = [self.token_config.mask_token]
        self.eos_token = self.token_config.eos_token_id
    
    def __len__(self) -> int:
        return len(self.duplicated_indices)
    
    def __getitem__(self, index: int) -> Tuple[str, Image.Image, List[int], str]:
        """Get a single sample."""
        max_retries = 20
        
        for _ in range(max_retries):
            try:
                original_idx = self.duplicated_indices[index]
                
                if self.data_source == "huggingface":
                    return self._get_hf_sample(original_idx)
                else:
                    return self._get_local_sample(original_idx)
                    
            except Exception as e:
                print(f"Error loading sample {index}: {e}")
                index = (index + 1) % len(self)
        
        raise RuntimeError(f"Failed to load sample after {max_retries} retries")
    
    def _get_hf_sample(self, idx: int) -> Tuple[str, Image.Image, List[int], str]:
        """Get sample from HuggingFace dataset. Returns (text, image, tokens, original_svg)."""
        sample = self.hf_dataset[idx]
        
        # Get text
        if random.random() < self.train_config.detail_prob:
            text = sample.get('detail', '')
        else:
            text = sample.get('description', '')
        
        text = str(text).strip()
        if len(text) < 5:
            text = sample.get('detail', sample.get('description', ''))
        
        # Truncate long text
        if len(text) > 500:
            text = self._truncate_text(text, 500)
        
        # Get image
        image_data = sample.get('image')
        if isinstance(image_data, dict) and 'bytes' in image_data:
            image = Image.open(io.BytesIO(image_data['bytes']))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            # Create blank image as fallback
            image = Image.new('RGB', (self.target_image_size, self.target_image_size), 'white')
        
        # Process image
        image = self._process_image(image)
        
        # Get SVG tokens
        svg_code = sample.get('svg', '')
        tokens = self._tokenize_svg_code(svg_code)
        
        # Apply causal masking
        tokens = self._apply_masking(tokens)
        
        # Add special tokens
        tokens = self.svg_tokenizer.add_special_tokens(tokens)
        
        # Return original SVG code as well
        return text, image, tokens.tolist(), svg_code
    
    def _get_local_sample(self, idx: int) -> Tuple[str, Image.Image, List[int], str]:
        """Get sample from local files. Returns (text, image, tokens, original_svg)."""
        row = self.meta_df.iloc[idx]
        uid = row['id']
        
        # Get text
        if random.random() < self.train_config.detail_prob:
            text = str(row.get('detail', '')).strip()
        else:
            text = str(row.get('desc_en', '')).strip()
        
        if len(text) < 5:
            text = str(row.get('detail', '')).strip()
        
        if len(text) > 500:
            text = self._truncate_text(text, 500)
        
        # Load image
        png_path = self._find_file(uid, '.png', self.png_folder, self.png_index)
        if png_path and os.path.exists(png_path):
            image = Image.open(png_path)
        else:
            image = Image.new('RGB', (self.target_image_size, self.target_image_size), 'white')
        
        image = self._process_image(image)
        
        # Load and tokenize SVG
        svg_path = self._find_file(uid, '.svg', self.svg_folder, self.svg_index)
        svg_code = ""
        if svg_path and os.path.exists(svg_path):
            # Read original SVG code
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_code = f.read()
            
            if DEEPSVG_AVAILABLE:
                svg = SVG.load_svg(svg_path)
                svg_tensors, color_tensors = svg.to_tensor(concat_groups=False, PAD_VAL=0)
                tokens = self.svg_tokenizer.tokenize_svg_tensors(svg_tensors, color_tensors)
            else:
                tokens = np.array([], dtype=np.int64)
        else:
            # Return empty tokens as fallback
            tokens = np.array([], dtype=np.int64)
        
        # Apply masking and special tokens
        tokens = self._apply_masking(tokens)
        tokens = self.svg_tokenizer.add_special_tokens(tokens)
        
        return text, image, tokens.tolist(), svg_code
    
    def _process_image(self, image: Image.Image) -> Image.Image:
        """Process image to target format."""
        # Convert to RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Resize
        image = image.resize(
            (self.target_image_size, self.target_image_size),
            Image.Resampling.LANCZOS
        )
        
        # Add white background
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
        
        return image.convert('RGB')
    
    def _truncate_text(self, text: str, max_len: int) -> str:
        """Truncate text at sentence boundary."""
        if len(text) <= max_len:
            return text
        
        text = text[:max_len]
        for delimiter in ['. ', ', ', ' ']:
            idx = text.rfind(delimiter)
            if idx > max_len * 0.6:
                return text[:idx + 1]
        
        return text
    
    def _find_file(
        self, 
        uid: str, 
        extension: str, 
        folder: str, 
        index: Dict[str, List[str]]
    ) -> Optional[str]:
        """Find file by UID."""
        # Direct path
        direct_path = os.path.join(folder, f"{uid}{extension}")
        if os.path.exists(direct_path):
            return direct_path
        
        # Search by prefix
        match = re.match(r'^(\d+)', str(uid))
        if match and index:
            prefix = match.group(1)
            if prefix in index:
                for filename in index[prefix]:
                    return os.path.join(folder, filename)
        
        return None
    
    def _apply_masking(self, tokens: np.ndarray) -> np.ndarray:
        """Apply causal masking with given probability."""
        if len(tokens) == 0 or random.random() >= self.mask_probability:
            return tokens
        
        # Random span masking
        start = random.random()
        end = random.random()
        if end < start:
            start, end = end, start
        
        start_idx = int(start * len(tokens))
        end_idx = int(end * len(tokens) + 0.5)
        
        if start_idx >= end_idx:
            return tokens
        
        # Create masked sequence
        masked = np.concatenate([
            tokens[:start_idx],
            [self.token_config.mask_token],
            tokens[start_idx:end_idx],
            [self.eos_token],
            tokens[end_idx:]
        ])
        
        return masked
    
    def _tokenize_svg_code(self, svg_code: str) -> np.ndarray:
        """Tokenize SVG code string (for HuggingFace dataset)."""
        if not svg_code or not DEEPSVG_AVAILABLE:
            return np.array([], dtype=np.int64)
        
        try:
            # Save to temp file and load with deepsvg
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
                f.write(svg_code)
                temp_path = f.name
            
            svg = SVG.load_svg(temp_path)
            svg_tensors, color_tensors = svg.to_tensor(concat_groups=False, PAD_VAL=0)
            tokens = self.svg_tokenizer.tokenize_svg_tensors(svg_tensors, color_tensors)
            
            os.unlink(temp_path)
            return tokens
            
        except Exception as e:
            print(f"SVG tokenization error: {e}")
            return np.array([], dtype=np.int64)


def create_dataloader(
    dataset: OmniSVGDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 8,
    collate_fn: Optional[callable] = None,
) -> torch.utils.data.DataLoader:
    """Create DataLoader with optimized settings."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn,
    )

