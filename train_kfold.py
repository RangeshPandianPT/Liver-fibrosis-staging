"""
K-Fold Cross-Validation Training Script for Liver Fibrosis Staging.
Trains the ensemble model using stratified k-fold cross-validation
for robust performance estimation suitable for research papers.

Usage:
    python train_kfold.py --folds 5 --epochs 50
    python train_kfold.py --folds 5 --epochs 1 --dry_run  # Quick test
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, DATA_DIR,
    CHECKPOINT_DIR, METRICS_DIR, CLASS_NAMES, RANDOM_SEED
)
from src.dataset import LiverFibrosisDataset
from src.preprocessing import get_train_transforms, get_val_transforms
from src.models import SoftVotingEnsemble
from src.training import (
    LabelSmoothingCrossEntropy,
    create_optimizer,
    create_scheduler,
    train_epoch,
    validate_epoch
)
from src.cross_validation import (
    CrossValidationResults,
    create_stratified_kfold_splits,
    compute_fold_metrics
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='K-Fold Cross-Validation Training for Liver Fibrosis Staging'
    )
    parser.add_argument(
        '--folds', type=int, default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    parser.add_argument(
        '--epochs', type=int, default=NUM_EPOCHS,
        help=f'Number of training epochs per fold (default: {NUM_EPOCHS})'
    )
    parser.add_argument(
        '--batch_size', type=int, default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Run a quick test with 1 epoch per fold'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers (default: 4)'
    )
    parser.add_argument(
        '--seed', type=int, default=RANDOM_SEED,
        help=f'Random seed (default: {RANDOM_SEED})'
    )
    parser.add_argument(
        '--save_all_folds', action='store_true',
        help='Save model checkpoint for each fold'
    )
    return parser.parse_args()


def train_single_fold(fold: int,
                      train_indices: np.ndarray,
                      val_indices: np.ndarray,
                      full_dataset: LiverFibrosisDataset,
                      args) -> dict:
    """
    Train model for a single fold.
    
    Args:
        fold: Fold number (0-indexed)
        train_indices: Training sample indices
        val_indices: Validation sample indices
        full_dataset: Full dataset
        args: Command line arguments
        
    Returns:
        Dictionary with fold results
    """
    print(f"\n{'='*70}")
    print(f"FOLD {fold + 1}/{args.folds}")
    print(f"{'='*70}")
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")
    
    # Create datasets with appropriate transforms
    train_dataset = LiverFibrosisDataset(
        data_dir=DATA_DIR, 
        transform=get_train_transforms()
    )
    val_dataset = LiverFibrosisDataset(
        data_dir=DATA_DIR,
        transform=get_val_transforms()
    )
    
    # Create subsets
    from torch.utils.data import Subset, DataLoader
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = SoftVotingEnsemble(pretrained=True)
    model = model.to(DEVICE)
    
    # Loss, optimizer, scheduler
    criterion = LabelSmoothingCrossEntropy()
    optimizer = create_optimizer(model, lr=args.lr)
    
    num_epochs = 1 if args.dry_run else args.epochs
    scheduler = create_scheduler(
        optimizer,
        steps_per_epoch=len(train_loader),
        num_epochs=num_epochs
    )
    
    # Training loop
    best_val_acc = 0.0
    best_preds = None
    best_labels = None
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device=DEVICE, epoch=epoch
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device=DEVICE, epoch=epoch
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"  Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds = val_preds
            best_labels = val_labels
            best_val_loss = val_loss
            
            # Save fold checkpoint if requested
            if args.save_all_folds:
                fold_checkpoint = {
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'accuracy': val_acc,
                    'loss': val_loss
                }
                cv_checkpoint_dir = CHECKPOINT_DIR / 'cross_validation'
                cv_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(fold_checkpoint, cv_checkpoint_dir / f'fold_{fold+1}_best.pth')
    
    # Compute metrics for this fold
    metrics = compute_fold_metrics(best_preds, best_labels, CLASS_NAMES)
    
    print(f"\n  Fold {fold+1} Best Results:")
    print(f"    Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"    Cohen's Kappa: {metrics['kappa']:.4f}")
    print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    return {
        'predictions': best_preds,
        'labels': best_labels,
        'metrics': metrics,
        'val_loss': best_val_loss
    }


def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("\n" + "=" * 70)
    print("LIVER FIBROSIS STAGING - K-FOLD CROSS-VALIDATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Number of folds: {args.folds}")
    print(f"  Epochs per fold: {1 if args.dry_run else args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Random seed: {args.seed}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        full_dataset = LiverFibrosisDataset(data_dir=DATA_DIR, transform=None)
        if len(full_dataset) == 0:
            raise ValueError("No images found")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have images in the following structure:")
        print("  data/liver_images/")
        print("    ├── F0/")
        print("    ├── F1/")
        print("    ├── F2/")
        print("    ├── F3/")
        print("    └── F4/")
        return
    
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Class distribution: {full_dataset.get_class_distribution()}")
    
    # Create stratified k-fold splits
    print(f"\nCreating {args.folds}-fold stratified splits...")
    splits = create_stratified_kfold_splits(full_dataset, args.folds, args.seed)
    
    # Initialize results tracker
    cv_results = CrossValidationResults(num_folds=args.folds, class_names=CLASS_NAMES)
    
    # Train each fold
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_result = train_single_fold(
            fold=fold,
            train_indices=train_idx,
            val_indices=val_idx,
            full_dataset=full_dataset,
            args=args
        )
        
        # Add fold result
        cv_results.add_fold_result(
            fold=fold,
            accuracy=fold_result['metrics']['accuracy'],
            kappa=fold_result['metrics']['kappa'],
            f1_weighted=fold_result['metrics']['f1_weighted'],
            f1_macro=fold_result['metrics']['f1_macro'],
            per_class_acc=fold_result['metrics']['per_class_accuracy'],
            predictions=fold_result['predictions'],
            labels=fold_result['labels'],
            val_loss=fold_result['val_loss']
        )
    
    total_time = time.time() - start_time
    
    # Save and display results
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 70)
    
    stats = cv_results.save_results()
    
    print(f"\n" + "-" * 70)
    print("FINAL RESULTS (FOR YOUR RESEARCH PAPER)")
    print("-" * 70)
    
    acc = stats['accuracy']
    kappa = stats['cohens_kappa']
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Accuracy: {acc['mean']*100:.2f}% ± {acc['std']*100:.2f}%")
    print(f"   95% CI: [{acc['ci_95'][0]*100:.2f}%, {acc['ci_95'][1]*100:.2f}%]")
    print(f"\n   Cohen's Kappa: {kappa['mean']:.4f} ± {kappa['std']:.4f}")
    print(f"   95% CI: [{kappa['ci_95'][0]:.4f}, {kappa['ci_95'][1]:.4f}]")
    
    print(f"\n   F1 (weighted): {stats['f1_weighted']['mean']:.4f} ± {stats['f1_weighted']['std']:.4f}")
    print(f"   F1 (macro): {stats['f1_macro']['mean']:.4f} ± {stats['f1_macro']['std']:.4f}")
    
    print(f"\n📈 Per-Class Accuracy:")
    for class_name, class_stats in stats['per_class_accuracy'].items():
        print(f"   {class_name}: {class_stats['mean']*100:.2f}% ± {class_stats['std']*100:.2f}%")
    
    print(f"\n⏱️  Total training time: {total_time/60:.1f} minutes")
    print(f"📁 Results saved to: {METRICS_DIR / 'cross_validation'}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDED PAPER TEXT")
    print("=" * 70)
    print(f"""
We evaluated our ensemble model using {args.folds}-fold stratified 
cross-validation. The model achieved an overall accuracy of 
{acc['mean']*100:.1f}% (95% CI: {acc['ci_95'][0]*100:.1f}%-{acc['ci_95'][1]*100:.1f}%) 
and a quadratic-weighted Cohen's kappa of {kappa['mean']:.3f} 
(95% CI: {kappa['ci_95'][0]:.3f}-{kappa['ci_95'][1]:.3f}), indicating 
{'almost perfect' if kappa['mean'] >= 0.8 else 'substantial' if kappa['mean'] >= 0.6 else 'moderate'} agreement with ground truth labels.
""")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
