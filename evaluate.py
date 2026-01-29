"""
Evaluation script for Liver Fibrosis Staging Pipeline.
Generates confusion matrix, Cohen's Kappa, and classification report.

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/best_model.pth
"""
import argparse
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import DEVICE, BATCH_SIZE, CLASS_NAMES, CHECKPOINT_DIR
from src.dataset import create_data_loaders, get_val_dataset_with_paths
from src.models import SoftVotingEnsemble
from src.training import LabelSmoothingCrossEntropy, validate_epoch
from src.validation import generate_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Liver Fibrosis Staging Model'
    )
    parser.add_argument(
        '--checkpoint', type=str, 
        default=str(CHECKPOINT_DIR / 'best_model.pth'),
        help='Path to model checkpoint'
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
    print("LIVER FIBROSIS STAGING - MODEL EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Batch size: {args.batch_size}")
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
    print(f"  Checkpoint accuracy: {checkpoint.get('accuracy', 'unknown'):.2f}%")
    
    # Run evaluation
    print("\nRunning evaluation...")
    criterion = LabelSmoothingCrossEntropy()
    
    val_loss, val_acc, all_preds, all_labels = validate_epoch(
        model, val_loader, criterion, device=DEVICE, epoch=0
    )
    
    print(f"\nValidation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.2f}%")
    
    # Generate all metrics
    metrics = generate_all_metrics(all_labels, all_preds, CLASS_NAMES)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
    print(f"  Cohen's Kappa: {metrics['cohens_kappa']:.4f}")
    print(f"\nMetrics saved to: outputs/metrics/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
