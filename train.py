"""
Main training script for Liver Fibrosis Staging Pipeline.

Usage:
    python train.py --epochs 50 --batch_size 16
    python train.py --epochs 1 --batch_size 4 --dry_run  # Quick test
"""
import argparse
import time
from datetime import datetime

import torch

# Import configurations and modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, 
    CHECKPOINT_DIR, CLASS_NAMES
)
from src.dataset import create_data_loaders
from src.models import SoftVotingEnsemble
from src.training import (
    LabelSmoothingCrossEntropy,
    create_optimizer,
    create_scheduler,
    train_epoch,
    validate_epoch,
    save_checkpoint
)
from src.validation import generate_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Liver Fibrosis Staging Ensemble Model'
    )
    parser.add_argument(
        '--epochs', type=int, default=NUM_EPOCHS,
        help=f'Number of training epochs (default: {NUM_EPOCHS})'
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
        help='Run a quick training test with limited data'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers (default: 4)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("LIVER FIBROSIS STAGING - ENSEMBLE TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 70 + "\n")
    
    # Create data loaders
    print("Loading data...")
    try:
        train_loader, val_loader, dataset = create_data_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have images in the following structure:")
        print("  data/liver_images/")
        print("    ├── F0/")
        print("    ├── F1/")
        print("    ├── F2/")
        print("    ├── F3/")
        print("    └── F4/")
        return
    
    if args.dry_run:
        print("\n[DRY RUN MODE] Using limited data for quick testing\n")
    
    # Initialize model
    print("\nInitializing Ensemble Model...")
    model = SoftVotingEnsemble(pretrained=True)
    model = model.to(DEVICE)
    
    print(f"  Total parameters: {model.get_total_params():,}")
    print(f"  Trainable parameters: {model.get_trainable_params():,}")
    
    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy()
    
    # Optimizer
    optimizer = create_optimizer(model, lr=args.lr)
    
    # Scheduler
    scheduler = create_scheduler(
        optimizer,
        steps_per_epoch=len(train_loader),
        num_epochs=args.epochs
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('accuracy', 0.0)
        print(f"  Resuming from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "-" * 70)
    print("TRAINING STARTED")
    print("-" * 70 + "\n")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device=DEVICE, epoch=epoch
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion,
            device=DEVICE, epoch=epoch
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            val_loss, val_acc, is_best,
            filename=f'checkpoint_epoch_{epoch+1}.pth'
        )
        
        # Quick exit for dry run
        if args.dry_run and epoch >= 0:
            print("\n[DRY RUN] Training test completed successfully!")
            break
    
    total_time = time.time() - start_time
    
    print("\n" + "-" * 70)
    print("TRAINING COMPLETED")
    print("-" * 70)
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    
    # Final evaluation with metrics
    print("\nRunning final evaluation...")
    model.load_state_dict(torch.load(CHECKPOINT_DIR / 'best_model.pth')['model_state_dict'])
    
    _, _, final_preds, final_labels = validate_epoch(
        model, val_loader, criterion, device=DEVICE, epoch=0
    )
    
    metrics = generate_all_metrics(final_labels, final_preds, CLASS_NAMES)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Evaluate model: python evaluate.py")
    print(f"  2. Generate heatmaps: python generate_heatmaps.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
