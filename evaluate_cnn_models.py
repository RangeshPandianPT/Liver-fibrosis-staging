"""
Comprehensive CNN Model Evaluation for Liver Fibrosis Staging.

Generates all evaluation metrics for the trained ResNet50 and EfficientNet-V2 models:
- Accuracy (overall and per-class)
- Precision, Recall, F1-Score
- Confusion Matrix
- Cohen's Kappa (weighted for ordinal data)
- ROC-AUC curves
- Classification Report

Usage:
    python evaluate_cnn_models.py
    python evaluate_cnn_models.py --model resnet  # Evaluate only ResNet
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import DEVICE, OUTPUT_DIR, CHECKPOINT_DIR, CLASS_NAMES, METRICS_DIR
from src.preprocessing import get_val_transforms
from src.models.resnet_branch import ResNet50Branch
from src.models.efficientnet_branch import EfficientNetBranch


class TestDataset(Dataset):
    """Dataset for Test split inference."""
    
    def __init__(self, manifest_path: str, transform=None):
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


def load_model(model_class, checkpoint_name, device):
    """Load a trained model from checkpoint."""
    model = model_class(pretrained=False)
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    
    if not checkpoint_path.exists():
        return None, None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


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


def compute_all_metrics(y_true, y_pred, y_probs, class_names, model_name):
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
    
    # Cohen's Kappa (quadratic for ordinal data)
    metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    metrics['cohens_kappa_linear'] = cohen_kappa_score(y_true, y_pred, weights='linear')
    
    # ROC-AUC (One-vs-Rest)
    try:
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
        metrics['roc_auc_weighted'] = roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr')
    except Exception as e:
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
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
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
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    
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


def print_metrics_summary(metrics, model_name):
    """Print a formatted summary of metrics."""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {model_name.upper()}")
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate CNN Models')
    parser.add_argument('--model', type=str, default='both', choices=['resnet', 'effnet', 'both'])
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("CNN MODEL EVALUATION - LIVER FIBROSIS STAGING")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print("=" * 70)
    
    # Load test data
    manifest_path = OUTPUT_DIR / 'dataset_manifest.csv'
    test_dataset = TestDataset(manifest_path, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create output directory for evaluation results
    eval_dir = METRICS_DIR / 'cnn_evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Evaluate ResNet50
    if args.model in ['resnet', 'both']:
        print("\n" + "-" * 60)
        print("Evaluating ResNet50...")
        resnet, ckpt = load_model(ResNet50Branch, 'best_resnet_model.pth', DEVICE)
        
        if resnet is not None:
            print(f"  Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, val_acc: {ckpt.get('val_acc', '?'):.2f}%)")
            
            y_true, y_pred, y_probs = run_inference(resnet, test_loader, DEVICE)
            metrics = compute_all_metrics(y_true, y_pred, y_probs, CLASS_NAMES, 'resnet')
            
            # Save plots
            plot_confusion_matrix(
                np.array(metrics['confusion_matrix']), CLASS_NAMES,
                eval_dir / 'resnet_confusion_matrix.png',
                'ResNet50 Confusion Matrix'
            )
            plot_roc_curves(y_true, y_probs, CLASS_NAMES, 
                           eval_dir / 'resnet_roc_curves.png',
                           'ResNet50 ROC Curves')
            
            print_metrics_summary(metrics, 'ResNet50')
            all_results['resnet'] = metrics
        else:
            print("  ResNet50 checkpoint not found!")
    
    # Evaluate EfficientNet
    if args.model in ['effnet', 'both']:
        print("\n" + "-" * 60)
        print("Evaluating EfficientNet-V2...")
        effnet, ckpt = load_model(EfficientNetBranch, 'best_effnet_model.pth', DEVICE)
        
        if effnet is not None:
            print(f"  Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, val_acc: {ckpt.get('val_acc', '?'):.2f}%)")
            
            y_true, y_pred, y_probs = run_inference(effnet, test_loader, DEVICE)
            metrics = compute_all_metrics(y_true, y_pred, y_probs, CLASS_NAMES, 'effnet')
            
            # Save plots
            plot_confusion_matrix(
                np.array(metrics['confusion_matrix']), CLASS_NAMES,
                eval_dir / 'effnet_confusion_matrix.png',
                'EfficientNet-V2 Confusion Matrix'
            )
            plot_roc_curves(y_true, y_probs, CLASS_NAMES,
                           eval_dir / 'effnet_roc_curves.png',
                           'EfficientNet-V2 ROC Curves')
            
            print_metrics_summary(metrics, 'EfficientNet-V2')
            all_results['effnet'] = metrics
        else:
            print("  EfficientNet-V2 checkpoint not found!")
    
    # Save all results to JSON
    results_path = eval_dir / 'evaluation_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
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
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {eval_dir}")
    print(f"  - evaluation_results.json")
    print(f"  - *_confusion_matrix.png")
    print(f"  - *_roc_curves.png")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
