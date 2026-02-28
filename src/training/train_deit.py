"""
DeiT Small Training Script (Distilled)
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

import sys
from config import (
    DATA_DIR, OUTPUT_DIR, DEVICE, CLASS_NAMES, NUM_CLASSES,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, LABEL_SMOOTHING
)
from src.models.deit_branch import DeiTBranch
from src.models.vit_branch import ViTBranch # Check if available, or define LightViTModel
# Since train_vit_light.py defined its own model, we might need that exact definition 
# if the state dict keys don't match ViTBranch. 
# Let's try to use ViTBranch, but valid keys might differ slightly if ViTBranch uses different structure?
# Both use self.backbone = models.vit_b_16. Weights should match.


# Configuration
DEIT_OUTPUT_DIR = OUTPUT_DIR / "deit_small"
VIT_CHECKPOINT = OUTPUT_DIR / "vit_light" / "best_vit_model.pth"
DEIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_SIZE = 224 # DeiT default

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

class DistillationLoss(nn.Module):
    """
    Distillation Loss = (1-alpha)*HardLoss + alpha*SoftLoss
    SoftLoss = KLDiv(StudentLogits/T, TeacherLogits/T) * T*T
    """
    def __init__(self, base_criterion, teacher_model, temperature=3.0, alpha=0.5, device=DEVICE):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
    def forward(self, student_outputs, labels, inputs):
        # Hard Loss (Student vs Labels)
        hard_loss = self.base_criterion(student_outputs, labels)
        
        # Soft Loss (Student vs Teacher)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        loss = (1 - self.alpha) * hard_loss + self.alpha * distillation_loss
        return loss


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
    # DeiT expects 224x224 and ImageNet normalization
    # Normalization constants are standard ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), # Resize larger then crop
        transforms.RandomCrop(IMAGE_SIZE), # Random crop to 224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform

def create_data_loaders(batch_size=BATCH_SIZE, num_workers=0):
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
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Dataset: {len(full_dataset)} total | {len(train_idx)} train | {len(val_idx)} val")
    return train_loader, val_loader

import torch.nn.functional as F # Needed for KLDiv

def train_epoch(model, loader, criterion, optimizer, device, mixup_fn=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Check if criterion is DistillationLoss
        if isinstance(criterion, DistillationLoss):
             loss = criterion(outputs, labels, images)
        else:
             loss = criterion(outputs, labels)
             
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        if labels.ndim > 1: # One-hot or soft labels (Mixup)
            acc_labels = labels.argmax(dim=1)
        else:
            acc_labels = labels
            
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(acc_labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
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
    parser = argparse.ArgumentParser(description='DeiT Training')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--resume', action='store_true', help='Resume from best_deit_model.pth')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🔬 DeiT-Small TRAINING")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    
    train_loader, val_loader = create_data_loaders(batch_size=args.batch_size)
    
    print("\n🧠 Initializing DeiT-Small (Pretrained)...")
    model = DeiTBranch(num_classes=NUM_CLASSES, pretrained=True)
    
    if args.resume:
        checkpoint_path = DEIT_OUTPUT_DIR / 'best_deit_model.pth'
        if checkpoint_path.exists():
            print(f"🔄 Resuming from {checkpoint_path}...")
            # Load weights
            state_dict = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print("✅ Weights loaded successfully!")
        else:
            print(f"⚠️ Resume requested but {checkpoint_path} not found. Starting from scratch or random.")
            
    model = model.to(DEVICE)
    
    model = model.to(DEVICE)
    
    # Load Teacher for Distillation
    teacher_model = None
    if VIT_CHECKPOINT.exists():
        print(f"\n🎓 Loading Teacher (ViT) from {VIT_CHECKPOINT}...")
        try:
            # We use ViTBranch as the definition. 
            # Note: train_vit_light.py used LightViTModel. 
            # If ViTBranch structure is identical (it seems so), this works.
            teacher_model = ViTBranch(num_classes=NUM_CLASSES, pretrained=False)
            checkpoint = torch.load(VIT_CHECKPOINT, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                teacher_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                teacher_model.load_state_dict(checkpoint)
            
            teacher_model.to(DEVICE)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            print("✅ Teacher loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load Teacher: {e}")
            print("⚠️ Proceeding without distillation.")
            teacher_model = None
    else:
        print(f"⚠️ Teacher checkpoint not found at {VIT_CHECKPOINT}. Proceeding without distillation.")
    
    # Mixup definition
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=1.0, switch_prob=0.5, mode='batch',
        label_smoothing=LABEL_SMOOTHING, num_classes=NUM_CLASSES
    )
    
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        base_criterion = SoftTargetCrossEntropy()
    else:
        base_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    if teacher_model:
        print("⚗️ Using Knowledge Distillation (Alpha=0.5, T=3.0)")
        criterion = DistillationLoss(base_criterion, teacher_model, temperature=3.0, alpha=0.5, device=DEVICE)
    else:
        criterion = base_criterion

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stopping = EarlyStopping(patience=args.patience)
    
    print("\n" + "-" * 60)
    print("🚀 TRAINING STARTED")
    print("-" * 60 + "\n")
    
    start_time = time.time()
    best_acc = 0
    
    # Validation criterion (always CrossEntropy, no mixup/distillation)
    val_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, mixup_fn=mixup_fn)
        val_loss, val_acc = validate(model, val_loader, val_criterion, DEVICE)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n📈 Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"   Train → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), DEIT_OUTPUT_DIR / 'best_deit_model.pth')
            print(f"   ✅ New best model saved! (Acc: {val_acc:.2f}%)")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
            break
            
    print("\n" + "=" * 60)
    print(f"✅ TRAINING COMPLETE. Best Acc: {best_acc:.2f}%")
    print(f"  Saved to: {DEIT_OUTPUT_DIR / 'best_deit_model.pth'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
