"""
Grad-CAM heatmap generation script for Liver Fibrosis Staging Pipeline.
Generates attention heatmaps for top correctly classified images per class.

Usage:
    python generate_heatmaps.py --checkpoint outputs/checkpoints/best_model.pth
"""
import argparse
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import DEVICE, BATCH_SIZE, CHECKPOINT_DIR, GRADCAM_DIR, TOP_K_HEATMAPS
from src.dataset import create_data_loaders
from src.models import SoftVotingEnsemble
from src.gradcam import generate_top_k_heatmaps


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM Heatmaps for Model Explainability'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default=str(CHECKPOINT_DIR / 'best_model.pth'),
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--top_k', type=int, default=TOP_K_HEATMAPS,
        help=f'Number of heatmaps per class (default: {TOP_K_HEATMAPS})'
    )
    parser.add_argument(
        '--batch_size', type=int, default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers (default: 4)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("LIVER FIBROSIS STAGING - GRAD-CAM HEATMAP GENERATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Top-K per class: {args.top_k}")
    print(f"  Output directory: {GRADCAM_DIR}")
    print("=" * 70 + "\n")
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nPlease train the model first using:")
        print("  python train.py")
        return
    
    # Create data loader
    print("Loading validation data...")
    try:
        _, val_loader, _ = create_data_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    except ValueError as e:
        print(f"\nError: {e}")
        return
    
    # Initialize and load model
    print("\nLoading model...")
    model = SoftVotingEnsemble(pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Generate heatmaps
    print("\nGenerating Grad-CAM heatmaps...")
    print("(This may take a few minutes...)\n")
    
    generate_top_k_heatmaps(
        model=model,
        dataloader=val_loader,
        k=args.top_k,
        device=DEVICE
    )
    
    print("\n" + "=" * 70)
    print("HEATMAP GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nHeatmaps for your paper's Results section are saved to:")
    print(f"  {GRADCAM_DIR}")
    print(f"\nDirectory structure:")
    print(f"  └── F0/ (top {args.top_k} images)")
    print(f"  └── F1/ (top {args.top_k} images)")
    print(f"  └── F2/ (top {args.top_k} images)")
    print(f"  └── F3/ (top {args.top_k} images)")
    print(f"  └── F4/ (top {args.top_k} images)")
    print(f"\nEach folder contains:")
    print(f"  - Combined visualization (all 3 branches side-by-side)")
    print(f"  - Individual heatmaps for ResNet50, EfficientNet, and ViT")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
