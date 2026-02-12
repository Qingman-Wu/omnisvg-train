"""
Data downloading and preprocessing utilities for OmniSVG.
Supports downloading from HuggingFace Hub and local processing.
"""

import os
import io
import json
import random
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image

try:
    from datasets import load_dataset, Dataset, concatenate_datasets
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Run: pip install datasets")


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    repo_id: str
    num_parquets: int
    description: str


# Dataset registry
DATASET_REGISTRY = {
    "illustration": DatasetInfo(
        name="MMSVG-Illustration",
        repo_id="OmniSVG/MMSVG-Illustration",
        num_parquets=26,
        description="SVG illustrations dataset"
    ),
    "icon": DatasetInfo(
        name="MMSVG-Icon", 
        repo_id="OmniSVG/MMSVG-Icon",
        num_parquets=91,
        description="SVG icons dataset"
    ),
}


class HuggingFaceDataLoader:
    """
    Handles downloading and processing datasets from HuggingFace Hub or local parquet files.
    """
    
    def __init__(
        self,
        cache_dir: str = "./data/cache",
        processed_dir: str = "./data/processed",
    ):
        if not HF_AVAILABLE:
            raise RuntimeError("Please install datasets: pip install datasets")
        
        self.cache_dir = Path(cache_dir)
        self.processed_dir = Path(processed_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_from_local_parquet(
        self,
        parquet_dir: str,
        pattern: str = "*.parquet",
    ) -> Dataset:
        """
        Load dataset from local parquet files.
        
        Args:
            parquet_dir: Directory containing parquet files
            pattern: Glob pattern for parquet files (e.g., "train-*.parquet")
        
        Returns:
            HuggingFace Dataset object
        """
        from datasets import Dataset as HFDataset
        import glob
        
        parquet_dir = Path(parquet_dir)
        parquet_files = sorted(glob.glob(str(parquet_dir / pattern)))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir} with pattern {pattern}")
        
        print(f"Loading {len(parquet_files)} parquet files from {parquet_dir}...")
        
        # Load all parquet files
        all_dfs = []
        for pf in tqdm(parquet_files, desc="Loading parquets"):
            df = pd.read_parquet(pf)
            all_dfs.append(df)
        
        # Concatenate and convert to HF Dataset
        combined_df = pd.concat(all_dfs, ignore_index=True)
        dataset = HFDataset.from_pandas(combined_df)
        
        print(f"Loaded {len(dataset)} samples from local parquet files")
        return dataset
    
    def download_dataset(
        self,
        dataset_key: str,
        parquet_indices: Optional[List[int]] = None,
        streaming: bool = False,
    ) -> Dataset:
        """
        Download a dataset from HuggingFace Hub.
        
        Args:
            dataset_key: Key from DATASET_REGISTRY ('illustration' or 'icon')
            parquet_indices: Optional list of specific parquet file indices to download.
                           If None, downloads all parquet files.
            streaming: Whether to use streaming mode (for large datasets)
        
        Returns:
            HuggingFace Dataset object
        """
        if dataset_key not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_key}. "
                           f"Available: {list(DATASET_REGISTRY.keys())}")
        
        info = DATASET_REGISTRY[dataset_key]
        print(f"Downloading {info.name} from {info.repo_id}...")
        
        if parquet_indices is not None:
            # Download specific parquet files
            data_files = [
                f"data/train-{idx:05d}-of-{info.num_parquets:05d}.parquet"
                for idx in parquet_indices
            ]
            dataset = load_dataset(
                info.repo_id,
                data_files=data_files,
                cache_dir=str(self.cache_dir),
                streaming=streaming,
            )
        else:
            # Download all
            dataset = load_dataset(
                info.repo_id,
                cache_dir=str(self.cache_dir),
                streaming=streaming,
            )
        
        if isinstance(dataset, dict):
            dataset = dataset['train']
        
        print(f"Downloaded {len(dataset) if not streaming else 'streaming'} samples from {info.name}")
        return dataset
    
    def download_multiple_datasets(
        self,
        datasets_config: Dict[str, Optional[List[int]]],
        streaming: bool = False,
    ) -> Dataset:
        """
        Download and combine multiple datasets.
        
        Args:
            datasets_config: Dict mapping dataset keys to parquet indices
                           e.g., {'illustration': None, 'icon': [0, 1, 2]}
                           None means download all parquets
            streaming: Whether to use streaming mode
        
        Returns:
            Combined HuggingFace Dataset
        """
        all_datasets = []
        
        for dataset_key, parquet_indices in datasets_config.items():
            dataset = self.download_dataset(
                dataset_key, 
                parquet_indices=parquet_indices,
                streaming=streaming
            )
            all_datasets.append(dataset)
        
        if len(all_datasets) == 1:
            return all_datasets[0]
        
        # Concatenate datasets
        if streaming:
            from datasets import interleave_datasets
            return interleave_datasets(all_datasets)
        else:
            return concatenate_datasets(all_datasets)
    
    def process_and_save(
        self,
        dataset: Dataset,
        output_dir: str,
        train_ratio: float = 0.95,
        max_token_length: int = 2048,
        min_token_length: int = 1,
        seed: int = 42,
    ) -> Tuple[str, str]:
        """
        Process dataset and save to disk.
        
        Args:
            dataset: HuggingFace Dataset
            output_dir: Output directory
            train_ratio: Ratio of training data
            max_token_length: Maximum token length to include
            min_token_length: Minimum token length to include
            seed: Random seed for splitting
        
        Returns:
            Tuple of (train_path, val_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        svg_dir = output_path / "svg"
        png_dir = output_path / "png"
        svg_dir.mkdir(exist_ok=True)
        png_dir.mkdir(exist_ok=True)
        
        # Filter by token length
        print(f"Filtering by token length: {min_token_length} <= len <= {max_token_length}")
        filtered_dataset = dataset.filter(
            lambda x: min_token_length <= x['token_len'] <= max_token_length
        )
        print(f"Filtered dataset size: {len(filtered_dataset)}")
        
        # Shuffle and split
        shuffled = filtered_dataset.shuffle(seed=seed)
        split_idx = int(len(shuffled) * train_ratio)
        
        train_data = shuffled.select(range(split_idx))
        val_data = shuffled.select(range(split_idx, len(shuffled)))
        
        print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        
        # Process and save
        train_meta = self._process_split(train_data, svg_dir, png_dir, "train")
        val_meta = self._process_split(val_data, svg_dir, png_dir, "val")
        
        # Save metadata CSVs
        train_csv = output_path / "train_meta.csv"
        val_csv = output_path / "val_meta.csv"
        
        train_df = pd.DataFrame(train_meta)
        val_df = pd.DataFrame(val_meta)
        
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        
        print(f"Saved train metadata to {train_csv}")
        print(f"Saved val metadata to {val_csv}")
        
        return str(train_csv), str(val_csv)
    
    def _process_split(
        self,
        dataset: Dataset,
        svg_dir: Path,
        png_dir: Path,
        split_name: str,
    ) -> List[Dict[str, Any]]:
        """Process a data split and save files."""
        meta_records = []
        
        print(f"Processing {split_name} split...")
        for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
            sample_id = sample.get('id', f"{split_name}_{idx}")
            
            # Save SVG
            svg_content = sample.get('svg', '')
            if svg_content:
                svg_path = svg_dir / f"{sample_id}.svg"
                with open(svg_path, 'w', encoding='utf-8') as f:
                    f.write(svg_content)
            
            # Save PNG image
            image = sample.get('image')
            if image is not None:
                if isinstance(image, dict) and 'bytes' in image:
                    # Image stored as bytes
                    img_bytes = image['bytes']
                    img = Image.open(io.BytesIO(img_bytes))
                elif isinstance(image, Image.Image):
                    img = image
                else:
                    img = None
                
                if img is not None:
                    png_path = png_dir / f"{sample_id}.png"
                    img.save(png_path, 'PNG')
            
            # Build metadata record
            meta_records.append({
                'id': sample_id,
                'desc_en': sample.get('description', ''),
                'detail': sample.get('detail', ''),
                'keywords': sample.get('keywords', ''),
                'len_pix': sample.get('token_len', 0),
            })
        
        return meta_records


def download_omnisvg_data(
    output_dir: str = "./data",
    datasets: Optional[List[str]] = None,
    parquet_config: Optional[Dict[str, Optional[List[int]]]] = None,
    train_ratio: float = 0.95,
    max_token_length: int = 2048,
    cache_dir: str = "./data/cache",
    local_data_dirs: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, str, str]:
    """
    Main function to download and prepare OmniSVG datasets.
    
    Args:
        output_dir: Directory to save processed data
        datasets: List of dataset keys to download. Default: ['illustration', 'icon']
        parquet_config: Optional dict mapping dataset keys to specific parquet indices.
                       If None, downloads all parquets for specified datasets.
        train_ratio: Ratio of training data (default 0.95)
        max_token_length: Maximum token length filter
        cache_dir: Directory for HuggingFace cache
        local_data_dirs: Optional dict mapping dataset keys to local parquet directories.
                        If provided, loads from local instead of downloading.
                        Example: {'illustration': '/path/to/local/illustration/parquets'}
    
    Returns:
        Tuple of (train_csv, val_csv, svg_folder, png_folder)
    
    Example:
        # Download all data from both datasets
        train_csv, val_csv, svg_dir, png_dir = download_omnisvg_data()
        
        # Load from local parquet files
        train_csv, val_csv, svg_dir, png_dir = download_omnisvg_data(
            datasets=['illustration'],
            local_data_dirs={
                'illustration': '/mnt/data2/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_process'
            }
        )
    """
    loader = HuggingFaceDataLoader(cache_dir=cache_dir)
    
    # Default: both datasets
    if datasets is None:
        datasets = ['illustration', 'icon']
    
    # Load datasets (from local or HuggingFace)
    all_datasets = []
    
    for dataset_key in datasets:
        if local_data_dirs and dataset_key in local_data_dirs:
            # Load from local parquet files
            local_dir = local_data_dirs[dataset_key]
            print(f"Loading {dataset_key} from local directory: {local_dir}")
            dataset = loader.load_from_local_parquet(local_dir, pattern="*.parquet")
            all_datasets.append(dataset)
        else:
            # Download from HuggingFace
            if parquet_config is None:
                parquet_indices = None
            else:
                parquet_indices = parquet_config.get(dataset_key, None)
            
            dataset = loader.download_dataset(dataset_key, parquet_indices=parquet_indices)
            all_datasets.append(dataset)
    
    # Combine datasets
    if len(all_datasets) == 1:
        combined_dataset = all_datasets[0]
    else:
        combined_dataset = concatenate_datasets(all_datasets)
    
    # Process and save
    train_csv, val_csv = loader.process_and_save(
        combined_dataset,
        output_dir=output_dir,
        train_ratio=train_ratio,
        max_token_length=max_token_length,
    )
    
    svg_folder = os.path.join(output_dir, "svg")
    png_folder = os.path.join(output_dir, "png")
    
    return train_csv, val_csv, svg_folder, png_folder


def list_available_datasets():
    """Print information about available datasets."""
    print("\nAvailable OmniSVG Datasets:")
    print("=" * 60)
    for key, info in DATASET_REGISTRY.items():
        print(f"\n{key}:")
        print(f"  Name: {info.name}")
        print(f"  HuggingFace: {info.repo_id}")
        print(f"  Parquet files: {info.num_parquets}")
        print(f"  Description: {info.description}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download OmniSVG datasets")
    parser.add_argument("--output_dir", type=str, default="./data",
                       help="Output directory for processed data")
    parser.add_argument("--datasets", type=str, nargs="+", 
                       default=["illustration", "icon"],
                       choices=["illustration", "icon"],
                       help="Datasets to download")
    parser.add_argument("--illustration_parquets", type=int, nargs="*", default=None,
                       help="Specific parquet indices for illustration dataset")
    parser.add_argument("--icon_parquets", type=int, nargs="*", default=None,
                       help="Specific parquet indices for icon dataset")
    parser.add_argument("--train_ratio", type=float, default=0.95,
                       help="Train/val split ratio")
    parser.add_argument("--max_token_length", type=int, default=2048,
                       help="Maximum token length filter")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
    else:
        # Build parquet config
        parquet_config = {}
        if "illustration" in args.datasets:
            parquet_config["illustration"] = args.illustration_parquets
        if "icon" in args.datasets:
            parquet_config["icon"] = args.icon_parquets
        
        train_csv, val_csv, svg_dir, png_dir = download_omnisvg_data(
            output_dir=args.output_dir,
            datasets=args.datasets,
            parquet_config=parquet_config,
            train_ratio=args.train_ratio,
            max_token_length=args.max_token_length,
        )
        
        print(f"\nData prepared successfully!")
        print(f"Train CSV: {train_csv}")
        print(f"Val CSV: {val_csv}")
        print(f"SVG folder: {svg_dir}")
        print(f"PNG folder: {png_dir}")
