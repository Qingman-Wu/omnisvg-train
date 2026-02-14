"""
Configuration loading and management for OmniSVG.
Supports both 4B and 8B model variants.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from pathlib import Path


def load_yaml(filepath: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge multiple configuration dictionaries."""
    result = {}
    for config in configs:
        if config is None:
            continue
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    return result


# Model size type
ModelSize = Literal["4B", "8B"]


# Model-specific defaults (fallback if config file not found)
MODEL_DEFAULTS = {
    "4B": {
        "base_vocab_size": 151936,
        "extended_vocab_size": 197000,
        "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "checkpoint": "OmniSVG/OmniSVG1.1_4B",
        "pad_token_id": 151643,
        "bos_token_id": 196998,
        "eos_token_id": 196999,
        "mask_token": 151937,
        "eom_token": 151938,
        "cmd_move": 151938,
        "cmd_line": 151939,
        "cmd_curve": 151940,
        "cmd_arc": 151941,
        "cmd_close": 151942,
        "pix_pad_offset": 151943,
        "coord_pad_offset": 151943,
        "arc_param_start": 196436,
    },
    "8B": {
        "base_vocab_size": 152064,
        "extended_vocab_size": 197000,
        "base_model": "/mnt/data/wuqingman/models/Qwen/Qwen2.5-VL-7B-Instruct",
        "checkpoint": "/mnt/data/wuqingman/models/OmniSVG/OmniSVG1.1_8B",
        "pad_token_id": 151643,
        "bos_token_id": 196998,
        "eos_token_id": 196999,
        "mask_token": 152065,
        "eom_token": 152066,
        "cmd_move": 152067,
        "cmd_line": 152068,
        "cmd_curve": 152069,
        "cmd_arc": 152070,
        "cmd_close": 152071,
        "pix_pad_offset": 152071,
        "coord_pad_offset": 152071,
        "arc_param_start": 196564,
    },
}


@dataclass
class TokenizationConfig:
    """Tokenization configuration for SVG encoding."""
    
    # Model size
    model_size: ModelSize = "4B"
    
    # Base vocabulary
    base_vocab_size: int = 151936
    extended_vocab_size: int = 197000
    
    # Model paths
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    checkpoint: str = "OmniSVG/OmniSVG1.1_4B"
    
    # Special tokens
    pad_token_id: int = 151643
    bos_token_id: int = 196998
    eos_token_id: int = 196999
    
    # SVG tokens
    num_svg_end: int = 1
    mask_token: int = 151937
    eom_token: int = 151938
    
    # Command tokens
    cmd_move: int = 151938
    cmd_line: int = 151939
    cmd_curve: int = 151940
    cmd_arc: int = 151941
    cmd_close: int = 151942
    
    # Coordinate encoding
    bbox_size: int = 200
    pix_pad_offset: int = 151943
    coord_pad_offset: int = 151943
    
    # Arc parameters
    arc_param_start: int = 196436
    
    # Color tokens
    color_token_offset: int = 40010
    max_color_tokens: int = 4098
    
    @classmethod
    def from_yaml(cls, filepath: str, model_size: ModelSize = "4B") -> "TokenizationConfig":
        """Load tokenization config from YAML file for specified model size."""
        config = load_yaml(filepath)
        
        # Get model-specific config
        models = config.get('models', {})
        model_config = models.get(model_size, {})
        
        if not model_config:
            print(f"Warning: Model size {model_size} not found in config, using defaults")
            model_config = MODEL_DEFAULTS.get(model_size, MODEL_DEFAULTS["4B"])
        
        special = model_config.get('special_tokens', {})
        svg = model_config.get('svg_tokens', {})
        coords = svg.get('coordinates', {})
        arc = svg.get('arc_params', {})
        commands = svg.get('commands', {})
        colors = config.get('color_tokens', {})
        
        # Get defaults for this model size
        defaults = MODEL_DEFAULTS.get(model_size, MODEL_DEFAULTS["4B"])
        
        return cls(
            model_size=model_size,
            base_vocab_size=model_config.get('base_vocab_size', defaults['base_vocab_size']),
            extended_vocab_size=model_config.get('extended_vocab_size', defaults['extended_vocab_size']),
            base_model=model_config.get('base_model', defaults['base_model']),
            checkpoint=model_config.get('checkpoint', defaults['checkpoint']),
            pad_token_id=special.get('pad_token_id', defaults['pad_token_id']),
            bos_token_id=special.get('bos_token_id', defaults['bos_token_id']),
            eos_token_id=special.get('eos_token_id', defaults['eos_token_id']),
            num_svg_end=svg.get('num_svg_end', 1),
            mask_token=svg.get('mask_token_offset', defaults['mask_token']),
            eom_token=svg.get('eom_token_offset', defaults['eom_token']),
            cmd_move=commands.get('move', defaults['cmd_move']),
            cmd_line=commands.get('line', defaults['cmd_line']),
            cmd_curve=commands.get('curve', defaults['cmd_curve']),
            cmd_arc=commands.get('arc', defaults['cmd_arc']),
            cmd_close=commands.get('close', defaults['cmd_close']),
            bbox_size=coords.get('bbox_size', 200),
            pix_pad_offset=coords.get('pix_pad_offset', defaults['pix_pad_offset']),
            coord_pad_offset=coords.get('coord_pad_offset', defaults['coord_pad_offset']),
            arc_param_start=arc.get('start_offset', defaults['arc_param_start']),
            color_token_offset=colors.get('color_token_offset', 40010),
            max_color_tokens=colors.get('max_color_tokens', 4098),
        )
    
    @classmethod
    def from_model_size(cls, model_size: ModelSize) -> "TokenizationConfig":
        """Create tokenization config from model size using defaults."""
        defaults = MODEL_DEFAULTS.get(model_size, MODEL_DEFAULTS["4B"])
        return cls(
            model_size=model_size,
            **{k: v for k, v in defaults.items() if k in cls.__dataclass_fields__}
        )
    
    @property
    def num_mask_and_eom(self) -> int:
        """Calculate num_mask_and_eom offset."""
        return 2 + self.base_vocab_size


@dataclass
class TrainConfig:
    """Training configuration."""
    
    # Model
    model_size: ModelSize = "4B"
    model_path: str = ""  # Will be set based on model_size
    torch_dtype: str = "bfloat16"
    use_flash_attn: bool = True
    
    # Data
    data_dir: str = "./data"
    target_image_size: int = 448
    text_max_length: int = 800
    max_seq_length: int = 2048
    
    # Text source probabilities
    detail_prob: float = 0.60
    
    # Training
    learning_rate: float = 1e-5
    weight_decay: float = 0.015
    max_grad_norm: float = 0.8
    warmup_steps: int = 50000
    epochs: int = 150
    gradient_accumulation_steps: int = 4
    
    # Task balance
    text_only_ratio: float = 0.50
    text_loss_weight: float = 1.5
    image_loss_weight: float = 1.0
    
    # Logging
    log_every: int = 10
    save_every: int = 3000
    val_every: int = 5000
    
    # DataLoader
    num_workers: int = 8
    
    # Seed
    seed: int = 2023
    
    def __post_init__(self):
        """Set model path based on model size if not specified."""
        if not self.model_path:
            self.model_path = MODEL_DEFAULTS[self.model_size]["base_model"]
    
    @classmethod
    def from_yaml(cls, filepath: str, model_size: Optional[ModelSize] = None) -> "TrainConfig":
        """Load training config from YAML file."""
        config = load_yaml(filepath)
        
        model = config.get('model', {})
        data = config.get('data', {})
        training = config.get('training', {})
        task = training.get('task_balance', {})
        loss = training.get('loss_weights', {})
        scheduler = training.get('scheduler', {})
        logging = config.get('logging', {})
        dataloader = config.get('dataloader', {})
        text_probs = data.get('text_source_probabilities', {})
        
        # Use provided model_size or fall back to config
        effective_model_size = model_size or model.get('size', '4B')
        defaults = MODEL_DEFAULTS.get(effective_model_size, MODEL_DEFAULTS["4B"])
        
        return cls(
            model_size=effective_model_size,
            model_path=defaults['base_model'],
            torch_dtype=model.get('torch_dtype', 'bfloat16'),
            use_flash_attn=model.get('use_flash_attn', True),
            data_dir=data.get('data_dir', './data'),
            target_image_size=data.get('target_image_size', 448),
            text_max_length=data.get('text_max_length', 800),
            max_seq_length=data.get('max_seq_length', 2048),
            detail_prob=text_probs.get('detail_description', 0.60),
            learning_rate=training.get('learning_rate', 1e-5),
            weight_decay=training.get('weight_decay', 0.015),
            max_grad_norm=training.get('max_grad_norm', 0.8),
            warmup_steps=scheduler.get('warmup_steps', 50000),
            epochs=training.get('epochs', 150),
            gradient_accumulation_steps=training.get('gradient_accumulation_steps', 4),
            text_only_ratio=task.get('initial_text_only_ratio', 0.50),
            text_loss_weight=loss.get('text_task', 1.5),
            image_loss_weight=loss.get('image_task', 1.0),
            log_every=logging.get('log_every', 10),
            save_every=logging.get('save_every', 3000),
            val_every=logging.get('val_every', 5000),
            num_workers=dataloader.get('num_workers', 8),
            seed=config.get('seed', 2023),
        )


@dataclass  
class DataConfig:
    """Data configuration for dataset downloading and processing."""
    
    # Dataset sources
    illustration_dataset: str = "OmniSVG/MMSVG-Illustration"
    icon_dataset: str = "OmniSVG/MMSVG-Icon"
    
    # Directory structure (relative to data_dir)
    # data_dir/
    #   ├── train_meta.csv
    #   ├── val_meta.csv  
    #   ├── svg/
    #   └── png/
    data_dir: str = "./data"
    
    # Download settings
    cache_dir: str = "./data/cache"
    
    # Train/val split
    train_ratio: float = 0.95
    val_ratio: float = 0.05
    
    # Filtering
    max_token_length: int = 2048
    min_token_length: int = 1
    
    @property
    def train_meta_file(self) -> str:
        return os.path.join(self.data_dir, "train_meta.csv")
    
    @property
    def val_meta_file(self) -> str:
        return os.path.join(self.data_dir, "val_meta.csv")
    
    @property
    def svg_folder(self) -> str:
        return os.path.join(self.data_dir, "svg")
    
    @property
    def png_folder(self) -> str:
        return os.path.join(self.data_dir, "png")
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DataConfig":
        """Create DataConfig from dictionary."""
        return cls(
            illustration_dataset=config.get('illustration_dataset', cls.illustration_dataset),
            icon_dataset=config.get('icon_dataset', cls.icon_dataset),
            data_dir=config.get('data_dir', cls.data_dir),
            cache_dir=config.get('cache_dir', cls.cache_dir),
            train_ratio=config.get('train_ratio', cls.train_ratio),
            val_ratio=config.get('val_ratio', cls.val_ratio),
            max_token_length=config.get('max_token_length', cls.max_token_length),
            min_token_length=config.get('min_token_length', cls.min_token_length),
        )


class OmniSVGConfig:
    """
    Main configuration class that combines all config components.
    Supports both 4B and 8B model variants.
    """
    
    def __init__(
        self,
        config_dir: str = "./configs",
        tokenization_file: str = "tokenization.yaml",
        train_file: str = "train_config.yaml",
        model_size: ModelSize = "4B",
    ):
        self.config_dir = Path(config_dir)
        self.model_size = model_size
        
        # Load tokenization config
        token_path = self.config_dir / tokenization_file
        if token_path.exists():
            self.tokenization = TokenizationConfig.from_yaml(str(token_path), model_size)
        else:
            print(f"Warning: {token_path} not found, using defaults for {model_size}")
            self.tokenization = TokenizationConfig.from_model_size(model_size)
        
        # Load training config
        train_path = self.config_dir / train_file
        if train_path.exists():
            self.training = TrainConfig.from_yaml(str(train_path), model_size)
        else:
            print(f"Warning: {train_path} not found, using defaults")
            self.training = TrainConfig(model_size=model_size)
        
        # Data config
        self.data = DataConfig(data_dir=self.training.data_dir)
    
    @property
    def base_model_path(self) -> str:
        """Get the base model path for this configuration."""
        return self.tokenization.base_model
    
    @property
    def checkpoint_path(self) -> str:
        """Get the checkpoint path for this configuration."""
        return self.tokenization.checkpoint
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all configs to a dictionary."""
        return {
            'model_size': self.model_size,
            'tokenization': self.tokenization.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
        }
    
    def save(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_args(cls, args) -> "OmniSVGConfig":
        """Create config from command line arguments."""
        model_size = getattr(args, 'model_size', '4B')
        config = cls(
            config_dir=getattr(args, 'config_dir', './configs'),
            model_size=model_size,
        )
        
        # Override with command line arguments
        if hasattr(args, 'model_path') and args.model_path:
            config.training.model_path = args.model_path
        if hasattr(args, 'max_seq_length') and args.max_seq_length:
            config.training.max_seq_length = args.max_seq_length
        if hasattr(args, 'data_dir') and args.data_dir:
            config.training.data_dir = args.data_dir
            config.data.data_dir = args.data_dir
        if hasattr(args, 'use_flash_attn'):
            config.training.use_flash_attn = args.use_flash_attn
            
        return config


def get_default_config(model_size: ModelSize = "4B") -> OmniSVGConfig:
    """Get default configuration for specified model size."""
    return OmniSVGConfig(model_size=model_size)


def print_model_info():
    """Print information about available models."""
    print("\n" + "=" * 60)
    print("OmniSVG Available Models")
    print("=" * 60)
    
    for size in ["4B", "8B"]:
        defaults = MODEL_DEFAULTS[size]
        print(f"\n{size} Model:")
        print(f"  Base Model:      {defaults['base_model']}")
        print(f"  Checkpoint:      {defaults['checkpoint']}")
        print(f"  Base Vocab Size: {defaults['base_vocab_size']}")
        print(f"  Extended Vocab:  {defaults['extended_vocab_size']}")
    
    print("\n" + "=" * 60)