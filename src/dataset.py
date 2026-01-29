"""
PyTorch Dataset and DataLoader utilities for liver fibrosis images.
"""
import os
from pathlib import Path
from typing import Tuple, Optional, List
import random

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

import sys
sys.path.insert(0, str(__file__).rsplit('src', 1)[0])
from config import DATA_DIR, CLASS_NAMES, TRAIN_SPLIT, RANDOM_SEED, BATCH_SIZE
from src.preprocessing import get_train_transforms, get_val_transforms


class LiverFibrosisDataset(Dataset):
    """
    Dataset for liver fibrosis staging images.
    
    Expects data organized as:
        data_dir/
            F0/
                image1.png
                image2.jpg
                ...
            F1/
            F2/
            F3/
            F4/
    """
    
    def __init__(self, 
                 data_dir: str = DATA_DIR,
                 transform=None,
                 class_names: List[str] = CLASS_NAMES):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing class folders
            transform: Optional transforms to apply
            class_names: List of class names (folder names)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        # Collect all image paths and labels
        self.samples = []
        self.targets = []
        
        for class_name in class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
                
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                    self.targets.append(self.class_to_idx[class_name])
        
        if len(self.samples) == 0:
            print(f"Warning: No images found in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_image_path(self, idx: int) -> str:
        """Get the file path for a specific sample index."""
        return self.samples[idx][0]
    
    def get_class_distribution(self) -> dict:
        """Get the distribution of samples across classes."""
        distribution = {name: 0 for name in self.class_names}
        for _, label in self.samples:
            class_name = self.class_names[label]
            distribution[class_name] += 1
        return distribution


def create_data_loaders(
    data_dir: str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    train_split: float = TRAIN_SPLIT,
    num_workers: int = 4,
    seed: int = RANDOM_SEED
) -> Tuple[DataLoader, DataLoader, LiverFibrosisDataset]:
    """
    Create train and validation data loaders with stratified split.
    
    Args:
        data_dir: Root directory containing class folders
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, full_dataset)
    """
    # Set random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create full dataset (without transforms for splitting)
    full_dataset = LiverFibrosisDataset(data_dir=data_dir, transform=None)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {data_dir}. Please check the data directory structure.")
    
    # Stratified split
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_split,
        random_state=seed
    )
    
    train_indices, val_indices = next(splitter.split(
        range(len(full_dataset)),
        full_dataset.targets
    ))
    
    # Create train and val datasets with appropriate transforms
    train_dataset = LiverFibrosisDataset(data_dir=data_dir, transform=get_train_transforms())
    val_dataset = LiverFibrosisDataset(data_dir=data_dir, transform=get_val_transforms())
    
    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Print dataset info
    print(f"Dataset loaded from: {data_dir}")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Class distribution: {full_dataset.get_class_distribution()}")
    
    return train_loader, val_loader, full_dataset


def get_val_dataset_with_paths(data_dir: str = DATA_DIR) -> LiverFibrosisDataset:
    """
    Get validation dataset that preserves image paths (for Grad-CAM).
    
    Args:
        data_dir: Root directory containing class folders
        
    Returns:
        Dataset with validation transforms
    """
    return LiverFibrosisDataset(data_dir=data_dir, transform=get_val_transforms())
