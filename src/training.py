"""
Training utilities for liver fibrosis classification.
Includes Label Smoothing loss, training loop, and optimizer configuration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Tuple, Dict, Optional
from tqdm import tqdm
import os

import sys
sys.path.insert(0, str(__file__).rsplit('src', 1)[0])
from config import (
    LEARNING_RATE, WEIGHT_DECAY, LABEL_SMOOTHING,
    MAX_LR, PCT_START, DIV_FACTOR, FINAL_DIV_FACTOR,
    DEVICE, CHECKPOINT_DIR
)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    
    Label smoothing helps handle the ambiguity between adjacent fibrosis stages
    (e.g., F1 vs F2) by preventing the model from becoming overly confident.
    """
    
    def __init__(self, smoothing: float = LABEL_SMOOTHING, num_classes: int = 5, weight: Optional[torch.Tensor] = None):
        """
        Initialize the loss function.
        
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform)
            num_classes: Number of classes
            weight: Optional class weights tensor of shape (num_classes,)
        """
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the label-smoothed cross entropy loss.
        
        Args:
            pred: Predicted logits of shape (B, num_classes)
            target: Target labels of shape (B,)
            
        Returns:
            Scalar loss value
        """
        # Log softmax for numerical stability
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Create smooth labels
        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth_labels.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Compute loss
        loss = (-smooth_labels * log_probs).sum(dim=-1)
        
        # Apply class weights if provided
        if self.weight is not None:
            # Get weight for each sample based on target class
            # Note: For smoothed labels, we could weight based on target, 
            # or weighted sum. Standard practice with smoothing + weights 
            # is often to just weight by the hard target.
            sample_weights = self.weight.to(pred.device)[target]
            loss = loss * sample_weights
            
        return loss.mean()


def create_optimizer(model: nn.Module, 
                     lr: float = LEARNING_RATE,
                     weight_decay: float = WEIGHT_DECAY) -> AdamW:
    """
    Create AdamW optimizer with proper weight decay handling.
    
    Args:
        model: The model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        
    Returns:
        Configured AdamW optimizer
    """
    # Separate parameters that should/shouldn't have weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'BatchNorm', 'bn']
    
    optimizer_grouped_params = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0
        }
    ]
    
    return AdamW(optimizer_grouped_params, lr=lr)


def create_scheduler(optimizer: AdamW,
                     steps_per_epoch: int,
                     num_epochs: int,
                     max_lr: float = MAX_LR,
                     pct_start: float = PCT_START,
                     div_factor: float = DIV_FACTOR,
                     final_div_factor: float = FINAL_DIV_FACTOR) -> OneCycleLR:
    """
    Create OneCycleLR scheduler.
    
    Args:
        optimizer: The optimizer
        steps_per_epoch: Number of batches per epoch
        num_epochs: Total number of epochs
        max_lr: Maximum learning rate
        pct_start: Percentage of cycle spent increasing LR
        div_factor: Initial LR = max_lr / div_factor
        final_div_factor: Final LR = initial_lr / final_div_factor
        
    Returns:
        Configured OneCycleLR scheduler
    """
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )


def train_epoch(model: nn.Module,
                train_loader,
                criterion: nn.Module,
                optimizer: AdamW,
                scheduler: OneCycleLR,
                device: str = DEVICE,
                epoch: int = 0) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        epoch: Current epoch (for progress bar)
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model: nn.Module,
                   val_loader,
                   criterion: nn.Module,
                   device: str = DEVICE,
                   epoch: int = 0) -> Tuple[float, float, list, list]:
    """
    Validate for one epoch.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        epoch: Current epoch (for progress bar)
        
    Returns:
        Tuple of (average_loss, accuracy, all_predictions, all_labels)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]', leave=False)
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{running_loss/(len(all_preds)//labels.size(0)):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def save_checkpoint(model: nn.Module,
                    optimizer: AdamW,
                    scheduler: OneCycleLR,
                    epoch: int,
                    loss: float,
                    accuracy: float,
                    is_best: bool = False,
                    filename: str = 'checkpoint.pth'):
    """
    Save a training checkpoint.
    
    Args:
        model: The model
        optimizer: The optimizer
        scheduler: The scheduler
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        is_best: Whether this is the best model so far
        filename: Checkpoint filename
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    filepath = CHECKPOINT_DIR / filename
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_path = CHECKPOINT_DIR / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"  ✓ New best model saved (acc: {accuracy:.2f}%)")


def load_checkpoint(model: nn.Module,
                    optimizer: Optional[AdamW] = None,
                    scheduler: Optional[OneCycleLR] = None,
                    filename: str = 'best_model.pth') -> Dict:
    """
    Load a training checkpoint.
    
    Args:
        model: The model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        filename: Checkpoint filename
        
    Returns:
        Checkpoint dictionary with epoch, loss, accuracy info
    """
    filepath = CHECKPOINT_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
    
    checkpoint = torch.load(filepath, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (acc: {checkpoint['accuracy']:.2f}%)")
    
    return checkpoint
