"""
Dataset Preparation Script for Liver Fibrosis Staging Ensemble Model

This script prepares the liver histopathology dataset by:
1. Applying CLAHE (Contrast Limited Adaptive Histogram Equalization) to all images
2. Creating an 80/20 train/test split using StratifiedShuffleSplit (random_state=42)
3. Generating a dataset_manifest.csv with image paths and assigned splits
4. Ensuring alphabetical sorting for reproducibility

Author: Medical Imaging Engineer
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, 
    CLASS_NAMES, 
    OUTPUT_DIR,
    CLAHE_CLIP_LIMIT, 
    CLAHE_TILE_GRID_SIZE
)


def apply_clahe_to_image(image_path: str, clip_limit: float = CLAHE_CLIP_LIMIT,
                         tile_grid_size: tuple = CLAHE_TILE_GRID_SIZE) -> np.ndarray:
    """
    Apply CLAHE to an image to highlight fibrotic tissue.
    
    CLAHE enhances local contrast in medical imaging, making fibrotic patterns
    more distinguishable for the ensemble model.
    
    Args:
        image_path: Path to the input image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        CLAHE-enhanced image as numpy array
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to LAB color space for CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L (luminance) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    
    # Merge channels and convert back to BGR
    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image


def collect_image_paths(data_dir: Path, class_names: List[str]) -> Tuple[List[str], List[int], List[str]]:
    """
    Collect all image paths from the dataset directory.
    
    Images are collected and sorted alphabetically to ensure reproducibility
    across different agents and runs.
    
    Args:
        data_dir: Root directory containing class folders
        class_names: List of class names (folder names)
        
    Returns:
        Tuple of (image_paths, labels, class_labels_str)
    """
    image_paths = []
    labels = []
    class_labels_str = []
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        # Collect images from this class
        class_images = []
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                class_images.append(str(img_path))
        
        # Sort alphabetically for reproducibility
        class_images.sort()
        
        for img_path in class_images:
            image_paths.append(img_path)
            labels.append(class_idx)
            class_labels_str.append(class_name)
    
    return image_paths, labels, class_labels_str


def create_stratified_split(image_paths: List[str], labels: List[int], 
                            class_labels_str: List[str],
                            train_size: float = 0.8, 
                            random_state: int = 42) -> pd.DataFrame:
    """
    Create a stratified train/test split using StratifiedShuffleSplit.
    
    This ensures balanced class representation in both train and test sets.
    
    Args:
        image_paths: List of image file paths
        labels: List of integer labels
        class_labels_str: List of string class labels
        train_size: Fraction of data for training (default: 0.8)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        DataFrame with image_path and assigned_split columns
    """
    # Create StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state
    )
    
    # Get train and test indices
    train_indices, test_indices = next(splitter.split(image_paths, labels))
    
    # Create split assignments
    splits = [''] * len(image_paths)
    for idx in train_indices:
        splits[idx] = 'Train'
    for idx in test_indices:
        splits[idx] = 'Test'
    
    # Create DataFrame
    manifest_df = pd.DataFrame({
        'image_path': image_paths,
        'assigned_split': splits
    })
    
    return manifest_df


def process_and_save_clahe_images(manifest_df: pd.DataFrame, 
                                   output_dir: Path) -> pd.DataFrame:
    """
    Apply CLAHE to all images and save them to the output directory.
    
    Args:
        manifest_df: DataFrame with image paths and splits
        output_dir: Directory to save CLAHE-processed images
        
    Returns:
        Updated DataFrame with processed image paths
    """
    clahe_output_dir = output_dir / "clahe_processed"
    clahe_output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_paths = []
    
    print("\nApplying CLAHE to all images...")
    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), 
                         desc="Processing images"):
        original_path = row['image_path']
        split = row['assigned_split']
        
        # Create split subdirectory
        split_dir = clahe_output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine class from original path
        original_path_obj = Path(original_path)
        class_name = original_path_obj.parent.name
        
        # Create class subdirectory within split
        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply CLAHE and save
        try:
            enhanced_image = apply_clahe_to_image(original_path)
            output_path = class_dir / original_path_obj.name
            cv2.imwrite(str(output_path), enhanced_image)
            processed_paths.append(str(output_path))
        except Exception as e:
            print(f"\nError processing {original_path}: {e}")
            processed_paths.append("")
    
    # Add processed paths to DataFrame
    manifest_df['clahe_image_path'] = processed_paths
    
    return manifest_df


def print_split_statistics(manifest_df: pd.DataFrame, class_names: List[str]):
    """
    Print statistics about the train/test split.
    
    Args:
        manifest_df: DataFrame with image paths and splits
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("DATASET SPLIT STATISTICS")
    print("="*60)
    
    total_images = len(manifest_df)
    train_count = len(manifest_df[manifest_df['assigned_split'] == 'Train'])
    test_count = len(manifest_df[manifest_df['assigned_split'] == 'Test'])
    
    print(f"\nTotal Images: {total_images}")
    print(f"Training Set: {train_count} ({train_count/total_images*100:.1f}%)")
    print(f"Test Set:     {test_count} ({test_count/total_images*100:.1f}%)")
    
    print("\n" + "-"*60)
    print("CLASS DISTRIBUTION")
    print("-"*60)
    print(f"{'Class':<10} {'Train':<10} {'Test':<10} {'Total':<10}")
    print("-"*60)
    
    for class_name in class_names:
        class_mask = manifest_df['image_path'].str.contains(f"\\{class_name}\\", regex=False)
        class_df = manifest_df[class_mask]
        train_class = len(class_df[class_df['assigned_split'] == 'Train'])
        test_class = len(class_df[class_df['assigned_split'] == 'Test'])
        total_class = len(class_df)
        print(f"{class_name:<10} {train_class:<10} {test_class:<10} {total_class:<10}")
    
    print("="*60)


def main():
    """
    Main function to prepare the dataset.
    
    Steps:
    1. Collect all image paths (alphabetically sorted)
    2. Create stratified 80/20 train/test split
    3. Apply CLAHE to all images
    4. Generate dataset_manifest.csv
    """
    print("="*60)
    print("LIVER HISTOPATHOLOGY DATASET PREPARATION")
    print("For Ensemble Model Training")
    print("="*60)
    
    data_dir = Path(DATA_DIR)
    output_dir = Path(OUTPUT_DIR)
    
    print(f"\nData Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Classes: {CLASS_NAMES}")
    
    # Step 1: Collect image paths (alphabetically sorted)
    print("\n[Step 1/4] Collecting and sorting image paths alphabetically...")
    image_paths, labels, class_labels_str = collect_image_paths(data_dir, CLASS_NAMES)
    print(f"Found {len(image_paths)} images across {len(CLASS_NAMES)} classes")
    
    if len(image_paths) == 0:
        print("ERROR: No images found in the data directory!")
        return
    
    # Step 2: Create stratified split
    print("\n[Step 2/4] Creating stratified 80/20 train/test split...")
    print("Using StratifiedShuffleSplit with random_state=42")
    manifest_df = create_stratified_split(
        image_paths, labels, class_labels_str,
        train_size=0.8, random_state=42
    )
    
    # Step 3: Apply CLAHE to all images
    print("\n[Step 3/4] Applying CLAHE to enhance fibrotic tissue visibility...")
    manifest_df = process_and_save_clahe_images(manifest_df, output_dir)
    
    # Step 4: Save manifest
    print("\n[Step 4/4] Generating dataset_manifest.csv...")
    manifest_path = output_dir / "dataset_manifest.csv"
    
    # Create final manifest with required columns
    final_manifest = manifest_df[['image_path', 'assigned_split']].copy()
    final_manifest.to_csv(manifest_path, index=False)
    print(f"Manifest saved to: {manifest_path}")
    
    # Also save extended manifest with CLAHE paths
    extended_manifest_path = output_dir / "dataset_manifest_extended.csv"
    manifest_df.to_csv(extended_manifest_path, index=False)
    print(f"Extended manifest saved to: {extended_manifest_path}")
    
    # Print statistics
    print_split_statistics(final_manifest, CLASS_NAMES)
    
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - Manifest: {manifest_path}")
    print(f"  - Extended Manifest: {extended_manifest_path}")
    print(f"  - CLAHE Images: {output_dir / 'clahe_processed'}")
    

if __name__ == "__main__":
    main()
