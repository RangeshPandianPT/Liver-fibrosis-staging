"""
ViT Model Evaluation for Liver Fibrosis Staging.

Generates comprehensive evaluation metrics for the trained ViT model:
- Accuracy (overall and per-class F0-F4)
- Precision, Recall, F1-Score
- Confusion Matrix
- Cohen's Kappa
- ROC-AUC curves

Usage:
    python evaluate_vit_model.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ViT_B_16_Weights
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# Configuration
NUM_CLASSES = 5
CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
VIT_CHECKPOINT = OUTPUT_DIR / "vit_light" / "best_vit_model.pth"
METRICS_DIR = OUTPUT_DIR / "metrics" / "vit_evaluation"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


class LightViTModel(nn.Module):
    """Lightweight ViT-B-16 model for liver fibrosis classification."""
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.backbone = models.vit_b_16(weights=weights)
        else:
            self.backbone = models.vit_b_16(weights=None)
        
        num_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class TestDataset(Dataset):
    """Dataset for Test split from manifest."""
    
    def __init__(self, manifest_path, transform=None):
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        
        df = pd.read_csv(manifest_path)
        self.data = df[df['assigned_split'] == 'Test'].reset_index(drop=True)
        print(f"Loaded {len(self.data)} Test samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[row['y_true']]
        return image, label


def get_transforms():
    """Get validation transforms for 224x224 images (ViT uses 224)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def run_inference(model, dataloader, device):
    """Run inference and return predictions, probabilities, and labels."""
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Running inference'):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_all_metrics(y_true, y_pred, y_probs, class_names):
    """Compute comprehensive evaluation metrics."""
    
    metrics = {}
    
    # Basic Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        metrics[f'f1_{name}'] = f1_per_class[i]
    
    # Per-class Precision
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        metrics[f'precision_{name}'] = prec_per_class[i]
    
    # Per-class Recall
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        metrics[f'recall_{name}'] = rec_per_class[i]
    
    # Cohen's Kappa (quadratic for ordinal data)
    metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    metrics['cohens_kappa_linear'] = cohen_kappa_score(y_true, y_pred, weights='linear')
    
    # ROC-AUC (One-vs-Rest)
    try:
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
        metrics['roc_auc_weighted'] = roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr')
    except Exception:
        metrics['roc_auc_macro'] = None
        metrics['roc_auc_weighted'] = None
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    # Per-class accuracy
    for i, name in enumerate(class_names):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            metrics[f'accuracy_{name}'] = (y_pred[class_mask] == i).mean()
        else:
            metrics[f'accuracy_{name}'] = 0.0
    
    return metrics


def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar_kws={'shrink': 0.8})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curves(y_true, y_probs, class_names, save_path, title='ROC Curves'):
    """Plot ROC curves for each class."""
    plt.figure(figsize=(10, 8))
    
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def print_metrics_summary(metrics):
    """Print a formatted summary of metrics."""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: ViT-B-16")
    print(f"{'='*60}")
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"  Accuracy:              {metrics['accuracy']*100:.2f}%")
    print(f"  F1-Score (Macro):      {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted):   {metrics['f1_weighted']:.4f}")
    print(f"  Precision (Macro):     {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):        {metrics['recall_macro']:.4f}")
    print(f"  Cohen's Kappa:         {metrics['cohens_kappa']:.4f}")
    
    if metrics.get('roc_auc_macro'):
        print(f"  ROC-AUC (Macro):       {metrics['roc_auc_macro']:.4f}")
    
    print(f"\n📈 PER-CLASS F1-SCORES:")
    for name in CLASS_NAMES:
        print(f"  {name}: {metrics[f'f1_{name}']:.4f}")
    
    print(f"\n🎯 PER-CLASS ACCURACY:")
    for name in CLASS_NAMES:
        print(f"  {name}: {metrics[f'accuracy_{name}']*100:.2f}%")
    
    print(f"\n📌 PER-CLASS PRECISION:")
    for name in CLASS_NAMES:
        print(f"  {name}: {metrics[f'precision_{name}']:.4f}")
    
    print(f"\n📍 PER-CLASS RECALL:")
    for name in CLASS_NAMES:
        print(f"  {name}: {metrics[f'recall_{name}']:.4f}")


def main():
    print("\n" + "=" * 70)
    print("ViT MODEL EVALUATION - LIVER FIBROSIS STAGING")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Checkpoint: {VIT_CHECKPOINT}")
    print("=" * 70)
    
    # Check checkpoint exists
    if not VIT_CHECKPOINT.exists():
        print(f"\nError: ViT checkpoint not found at {VIT_CHECKPOINT}")
        return
    
    # Load test data from manifest
    manifest_path = OUTPUT_DIR / 'dataset_manifest.csv'
    test_dataset = TestDataset(manifest_path, transform=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Load model
    print("\nLoading ViT model...")
    model = LightViTModel(pretrained=False)
    checkpoint = torch.load(VIT_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    val_acc = checkpoint.get('val_acc', checkpoint.get('accuracy', 'unknown'))
    print(f"  Loaded checkpoint (epoch {epoch}, val_acc: {val_acc:.2f}%)")
    
    # Run inference
    print("\nRunning inference on Test set...")
    y_true, y_pred, y_probs = run_inference(model, test_loader, DEVICE)
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, y_probs, CLASS_NAMES)
    
    # Generate plots
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']), CLASS_NAMES,
        METRICS_DIR / 'vit_confusion_matrix.png',
        'ViT-B-16 Confusion Matrix'
    )
    plot_roc_curves(
        y_true, y_probs, CLASS_NAMES,
        METRICS_DIR / 'vit_roc_curves.png',
        'ViT-B-16 ROC Curves'
    )
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Save results to JSON
    results_path = METRICS_DIR / 'vit_evaluation_results.json'
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(metrics), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {METRICS_DIR}")
    print(f"  - vit_evaluation_results.json")
    print(f"  - vit_confusion_matrix.png")
    print(f"  - vit_roc_curves.png")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
