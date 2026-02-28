"""
Lightweight ViT-B-16 Training Script for CPU Training.

Features:
- Single ViT model (not full ensemble) - 3x faster
- 224x224 image size - faster processing
- Early stopping - saves time if model converges
- Optimized for CPU training

Usage:
    python train_vit_light.py --epochs 10 --batch_size 4
    python train_vit_light.py --epochs 5 --batch_size 2 --patience 3
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.models import ViT_B_16_Weights
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# Configuration
IMAGE_SIZE = 224  # Smaller for faster CPU training
NUM_CLASSES = 5
CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "liver_images"
OUTPUT_DIR = BASE_DIR / "outputs" / "vit_light"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  ⚠️ EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
    
    def load_best_model(self, model):
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)


class LightViTModel(nn.Module):
    """Lightweight ViT-B-16 model for liver fibrosis classification."""
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        
        # Load ViT-B/16 with standard weights (224x224)
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.backbone = models.vit_b_16(weights=weights)
        else:
            self.backbone = models.vit_b_16(weights=None)
        
        # Get features dimension and replace head
        num_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleLiverDataset(torch.utils.data.Dataset):
    """Simple dataset for liver fibrosis images."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.targets = []
        
        for idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    self.samples.append((str(img_path), idx))
                    self.targets.append(idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    """Get train and validation transforms for 224x224 images."""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(batch_size=4, num_workers=0):
    """Create train and validation data loaders."""
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    full_dataset = SimpleLiverDataset(DATA_DIR, transform=None)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {DATA_DIR}")
    
    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(splitter.split(range(len(full_dataset)), full_dataset.targets))
    
    # Create train and val datasets
    train_dataset = SimpleLiverDataset(DATA_DIR, transform=train_transform)
    val_dataset = SimpleLiverDataset(DATA_DIR, transform=val_transform)
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    print(f"📊 Dataset: {len(full_dataset)} total | {len(train_idx)} train | {len(val_idx)} val")
    
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Lightweight ViT Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=0, help='Data loader workers')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🔬 LIGHTWEIGHT ViT-B-16 TRAINING")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Early Stopping Patience: {args.patience}")
    print("=" * 60 + "\n")
    
    # Create data loaders
    print("📁 Loading data...")
    train_loader, val_loader = create_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print("\n🧠 Initializing ViT-B-16...")
    model = LightViTModel(pretrained=True)
    model = model.to(DEVICE)
    print(f"  Parameters: {model.get_num_params():,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Training loop
    print("\n" + "-" * 60)
    print("🚀 TRAINING STARTED")
    print("-" * 60 + "\n")
    
    start_time = time.time()
    best_acc = 0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"\n📈 Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"   Train → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, OUTPUT_DIR / 'best_vit_model.pth')
            print(f"   ✅ New best model saved! (Acc: {val_acc:.2f}%)")
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    early_stopping.load_best_model(model)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total Time: {total_time/60:.1f} minutes")
    print(f"  Best Val Accuracy: {best_acc:.2f}%")
    print(f"  Model saved to: {OUTPUT_DIR / 'best_vit_model.pth'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
