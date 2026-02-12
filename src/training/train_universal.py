"""
Universal Training Script for ConvNeXt and other models.
Supports WeightedRandomSampler for class balancing.
"""
import argparse
import time
import sys
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models
from tqdm import tqdm
import timm

# Add project root to path
import sys
from pathlib import Path
# Fix path: src/training -> project_root
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    DATA_DIR, OUTPUT_DIR, DEVICE, CLASS_NAMES, NUM_CLASSES,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, LABEL_SMOOTHING
)

# Calculated from compute_class_weights.py & rounded
CLASS_WEIGHTS = [0.5982, 1.4682, 1.5956, 1.4746, 0.7449]

def get_model(model_name, num_classes, pretrained=True):
    """
    Factory function to create models using timm or torchvision.
    """
    print(f"Creating model: {model_name}")
    
    if model_name == 'convnext':
        # ConvNeXt Tiny is a good balance for this task
        model = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'resnet':
        model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'densenet':
        model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    return model

def get_transforms(model_name):
    """
    Get appropriate transforms for the model.
    ConvNeXt typically uses 224x224.
    """
    # Standard ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Image size depends on model, but 224 is standard for most restricted compute
    image_size = 224 

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform

def create_dataloaders(data_dir, batch_size, model_name):
    """
    Create DataLoaders with WeightedRandomSampler for training.
    """
    train_transform, val_transform = get_transforms(model_name)
    
    # Use ImageFolder - Assuming structure: data_dir/class_name/image.jpg
    # Note: We need to manually split train/val or use existing split logic.
    # For this script, we'll rely on the folder structure if it's already split, 
    # OR we'll use SubsetRandomSampler if it's a single directory.
    
    # Since existing scripts (train_deit.py) assume a single DATA_DIR and split dynamically,
    # we will replicate that logic but ADD WeightedRandomSampler for the TRAIN set.
    
    full_dataset = datasets.ImageFolder(data_dir, transform=None) # No transform initially
    
    # Split indices (80/20) - Stratified is ideal for validation set consistency
    from sklearn.model_selection import StratifiedShuffleSplit
    import numpy as np
    
    labels = [s[1] for s in full_dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    # Create concrete datasets with transforms
    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    
    # Subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(val_dataset, val_idx)
    
    # --- WEIGHTED SAMPLER LOGIC ---
    # We need to assign a weight to each sample in the TRAIN SUBSET.
    # get targets for train_subset
    train_targets = [labels[i] for i in train_idx]
    
    # Create sampler weights
    # sample_weight = class_weight[label]
    weights_tensor = torch.DoubleTensor(CLASS_WEIGHTS)
    sample_weights = [weights_tensor[t] for t in train_targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create DataLoaders
    # Note: shuffle=False when using sampler
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"Data Loaders Created:")
    print(f"  Train: {len(train_subset)} samples (Weighted Sampling)")
    print(f"  Val:   {len(val_subset)} samples")
    
    return train_loader, val_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description='Universal Training Script')
    parser.add_argument('--model', type=str, required=True, choices=['convnext', 'resnet', 'efficientnet', 'densenet', 'vit'],
                        help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    
    args = parser.parse_args()
    
    # Setup Output Directory for this specific model
    model_output_dir = OUTPUT_DIR / args.model
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training {args.model.upper()}...")
    print(f"Output Dir: {model_output_dir}")
    print(f"Class Weights: {CLASS_WEIGHTS}")
    
    # Load Data
    train_loader, val_loader = create_dataloaders(DATA_DIR, args.batch_size, args.model)
    
    # Create Model
    model = get_model(args.model, NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training Loop
    best_acc = 0.0
    
    print("\nStarting Training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} [{duration:.1f}s] - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = model_output_dir / f"best_{args.model}_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best Model Saved! ({val_acc:.2f}%)")
            
    print(f"\nTraining Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
