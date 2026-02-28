"""
CNN Model Training Script for Liver Fibrosis Staging.

Trains ResNet50 and EfficientNet-V2 models using transfer learning
with pre-trained ImageNet weights.

Usage:
    python train_cnn_models.py --epochs 20 --batch_size 16
    python train_cnn_models.py --model resnet --epochs 10  # Train only ResNet
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from PIL import Image
from tqdm import tqdm

import sys
# Fix path: src/training -> project_root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
print(f"Added project root to path: {project_root}")

from config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, OUTPUT_DIR, CHECKPOINT_DIR,
    CLASS_NAMES, IMAGE_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MAX_LR, PCT_START, DIV_FACTOR, FINAL_DIV_FACTOR, LABEL_SMOOTHING
)
from src.preprocessing import get_train_transforms, get_val_transforms
from src.models.resnet_branch import ResNet50Branch
from src.models.efficientnet_branch import EfficientNetBranch
from src.training import LabelSmoothingCrossEntropy
try:
    from compute_class_weights import compute_weights
except ImportError:
    # Fallback if running from root with module structure
    from src.training.compute_class_weights import compute_weights


class ManifestDataset(Dataset):
    """Dataset that uses the dataset manifest for Train/Test splits."""
    
    def __init__(self, manifest_path: str, split: str = 'Train', transform=None):
        """
        Initialize dataset from manifest.
        
        Args:
            manifest_path: Path to dataset_manifest.csv
            split: 'Train' or 'Test'
            transform: Image transforms
        """
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        
        # Load manifest and filter by split
        df = pd.read_csv(manifest_path)
        self.data = df[df['assigned_split'] == split].reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} {split} samples from manifest")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        label = self.class_to_idx[row['y_true']]
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total


def train_model(model, model_name, train_loader, val_loader, args, device, start_epoch=0, class_weights=None):
    """
    Train a single model and return training history.
    
    Args:
        model: The model to train
        model_name: Name for saving ('resnet' or 'effnet')
        train_loader: Training data loader
        val_loader: Validation data loader
        args: Command line arguments
        device: Device to use
        start_epoch: Starting epoch (for resume)
        class_weights: Optional tensor of class weights for loss function
        
    Returns:
        dict: Training history with loss/accuracy per epoch
    """
    model = model.to(device)
    
    # Loss function with optional class weights
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print("Using weighted LabelSmoothingCrossEntropy")
    
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING, weight=class_weights)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    
    # Load checkpoint if resuming
    best_val_acc = 0.0
    checkpoint_path = CHECKPOINT_DIR / f'best_{model_name}_model.pth'
    if args.resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_acc = checkpoint.get('val_acc', 0.0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"  Loaded epoch {start_epoch}, best val_acc: {best_val_acc:.2f}%")
    
    # Scheduler - calculate remaining steps
    remaining_epochs = args.epochs - start_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=remaining_epochs,
        pct_start=PCT_START,
        div_factor=DIV_FACTOR,
        final_div_factor=FINAL_DIV_FACTOR
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} (epochs {start_epoch+1} to {args.epochs})")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"  -> Saved best model (acc: {val_acc:.2f}%)")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train CNN Models for Liver Fibrosis Staging')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--model', type=str, default='both', choices=['resnet', 'effnet', 'both'],
                       help='Which model(s) to train')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data workers (0 to avoid deadlocks)')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch (for resume)')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("CNN MODEL TRAINING - LIVER FIBROSIS STAGING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Models: {args.model}")
    print("=" * 70 + "\n")
    
    # Load data from manifest
    manifest_path = OUTPUT_DIR / 'dataset_manifest.csv'
    
    train_dataset = ManifestDataset(manifest_path, split='Train', transform=get_train_transforms())
    val_dataset = ManifestDataset(manifest_path, split='Test', transform=get_val_transforms())
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Compute class weights
    print("\nComputing class weights for balanced training...")
    try:
        weights_dict = compute_weights()
        class_weights = torch.tensor([weights_dict[c] for c in CLASS_NAMES], dtype=torch.float)
        print(f"Class weights tensor: {class_weights}")
    except Exception as e:
        print(f"Error computing class weights: {e}")
        print("Falling back to unweighted loss.")
        class_weights = None
    
    # Training history for all models
    all_history = {}
    
    # Train ResNet50
    if args.model in ['resnet', 'both']:
        resnet = ResNet50Branch(pretrained=True)
        all_history['resnet'] = train_model(
            resnet, 'resnet', train_loader, val_loader, args, DEVICE, class_weights=class_weights
        )
    
    # Train EfficientNet-V2
    if args.model in ['effnet', 'both']:
        effnet = EfficientNetBranch(pretrained=True)
        all_history['effnet'] = train_model(
            effnet, 'effnet', train_loader, val_loader, args, DEVICE, class_weights=class_weights
        )
    
    # Save training history
    history_path = OUTPUT_DIR / 'cnn_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(all_history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    
    # Print summary for research paper
    print("\n" + "=" * 70)
    print("TRAINING HISTORY SUMMARY (for Research Paper)")
    print("=" * 70)
    
    for model_name, history in all_history.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 40)
        print(f"{'Epoch':<8}{'Train Loss':<12}{'Train Acc':<12}{'Val Loss':<12}{'Val Acc':<12}")
        print("-" * 40)
        for i in range(len(history['train_loss'])):
            print(f"{i+1:<8}{history['train_loss'][i]:<12.4f}{history['train_acc'][i]:<12.2f}"
                  f"{history['val_loss'][i]:<12.4f}{history['val_acc'][i]:<12.2f}")
    
    print("\n" + "=" * 70)
    print("Training complete! Run generate_cnn_predictions.py for inference.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
