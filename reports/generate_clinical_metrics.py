"""
Clinical Metrics Generator for Liver Fibrosis Staging.
Generates clinically relevant metrics for research papers.

Features:
- Sensitivity/Specificity per fibrosis stage
- Binary classification (Early F0-F2 vs Advanced F3-F4)
- Model calibration / reliability diagrams
- Comprehensive clinical report

Usage:
    python generate_clinical_metrics.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    accuracy_score
)
from sklearn.calibration import calibration_curve
from pathlib import Path
from tqdm import tqdm
import json

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import CLASS_NAMES, NUM_CLASSES, DEVICE
from train_vit_light import LightViTModel, SimpleLiverDataset, get_transforms

# Output directories
OUTPUT_DIR = Path(__file__).parent / "outputs" / "clinical_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model checkpoint path
MODEL_PATH = Path(__file__).parent / "outputs" / "vit_light" / "best_vit_model.pth"
DATA_DIR = Path(__file__).parent / "data" / "liver_images"


def load_model():
    """Load the trained ViT model."""
    model = LightViTModel(num_classes=NUM_CLASSES, pretrained=False)
    
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded model from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = model.to(DEVICE)
    model.eval()
    return model


def get_predictions(model, dataloader):
    """Get predictions and probabilities from the model."""
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Getting predictions"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def compute_sensitivity_specificity(y_true, y_pred):
    """
    Compute sensitivity (recall) and specificity for each class.
    
    Sensitivity = TP / (TP + FN) = True Positive Rate
    Specificity = TN / (TN + FP) = True Negative Rate
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    
    results = {}
    for i in range(NUM_CLASSES):
        # True Positives
        tp = cm[i, i]
        # False Negatives
        fn = np.sum(cm[i, :]) - tp
        # False Positives
        fp = np.sum(cm[:, i]) - tp
        # True Negatives
        tn = np.sum(cm) - tp - fn - fp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        results[CLASS_NAMES[i]] = {
            'sensitivity': round(sensitivity, 4),
            'specificity': round(specificity, 4),
            'ppv': round(ppv, 4),
            'npv': round(npv, 4),
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp),
            'tn': int(tn)
        }
    
    return results


def compute_binary_classification(y_true, y_pred, y_probs):
    """
    Compute binary classification metrics: Early (F0-F2) vs Advanced (F3-F4).
    
    This is clinically relevant for treatment decisions.
    """
    # Convert to binary: 0 = Early (F0, F1, F2), 1 = Advanced (F3, F4)
    y_true_binary = np.where(y_true >= 3, 1, 0)
    y_pred_binary = np.where(y_pred >= 3, 1, 0)
    
    # Probability of advanced fibrosis = sum of P(F3) + P(F4)
    prob_advanced = y_probs[:, 3] + y_probs[:, 4]
    
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    
    results = {
        'accuracy': round(accuracy, 4),
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4),
        'ppv': round(ppv, 4),
        'npv': round(npv, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }
    
    return results, y_true_binary, prob_advanced


def plot_calibration_curve(y_true, y_probs, save_path=None):
    """
    Plot reliability/calibration diagram.
    
    Shows whether the model's confidence scores are well-calibrated.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    # Multi-class calibration
    ax1 = axes[0]
    for i in range(NUM_CLASSES):
        y_true_class = (y_true == i).astype(int)
        y_prob_class = y_probs[:, i]
        
        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(
            y_true_class, y_prob_class, n_bins=10, strategy='uniform'
        )
        
        ax1.plot(prob_pred, prob_true, 's-', color=colors[i], 
                 label=f'{CLASS_NAMES[i]}', linewidth=2, markersize=6)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Calibration Curves (Per Class)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Confidence histogram
    ax2 = axes[1]
    max_probs = np.max(y_probs, axis=1)
    correct = (y_true == np.argmax(y_probs, axis=1))
    
    bins = np.linspace(0, 1, 11)
    
    ax2.hist(max_probs[correct], bins=bins, alpha=0.7, color='green', 
             label='Correct', edgecolor='black')
    ax2.hist(max_probs[~correct], bins=bins, alpha=0.7, color='red', 
             label='Incorrect', edgecolor='black')
    
    ax2.set_xlabel('Max Predicted Probability (Confidence)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Model Calibration Analysis - Liver Fibrosis Staging', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Calibration plot saved to {save_path}")
    
    plt.close()


def plot_sensitivity_specificity(metrics, save_path=None):
    """Plot sensitivity and specificity bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(NUM_CLASSES)
    width = 0.35
    
    sensitivities = [metrics[c]['sensitivity'] for c in CLASS_NAMES]
    specificities = [metrics[c]['specificity'] for c in CLASS_NAMES]
    
    bars1 = ax.bar(x - width/2, sensitivities, width, label='Sensitivity', 
                   color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, specificities, width, label='Specificity', 
                   color='#e74c3c', edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars1, sensitivities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, specificities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', fontsize=9)
    
    ax.set_xlabel('Fibrosis Stage', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Sensitivity & Specificity by Fibrosis Stage', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sensitivity/Specificity plot saved to {save_path}")
    
    plt.close()


def plot_binary_confusion_matrix(binary_results, save_path=None):
    """Plot confusion matrix for binary classification."""
    cm = binary_results['confusion_matrix']
    cm_array = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm_array, cmap='Blues')
    
    # Add labels
    labels = ['Early (F0-F2)', 'Advanced (F3-F4)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    
    # Add values in cells
    for i in range(2):
        for j in range(2):
            color = 'white' if cm_array[i, j] > cm_array.max()/2 else 'black'
            ax.text(j, i, str(cm_array[i, j]), ha='center', va='center', 
                    fontsize=16, color=color, fontweight='bold')
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Binary Classification: Early vs Advanced Fibrosis\n' +
                 f'Sensitivity: {binary_results["sensitivity"]:.2%} | Specificity: {binary_results["specificity"]:.2%}',
                 fontsize=13, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Binary confusion matrix saved to {save_path}")
    
    plt.close()


def generate_clinical_report(sens_spec, binary_results, save_path=None):
    """Generate comprehensive clinical metrics report."""
    report = {
        "model": "ViT-B/16",
        "task": "Liver Fibrosis Staging (F0-F4)",
        "per_class_metrics": sens_spec,
        "binary_classification": {
            "description": "Early (F0-F2) vs Advanced (F3-F4) Fibrosis",
            "metrics": binary_results
        },
        "summary": {
            "avg_sensitivity": round(np.mean([sens_spec[c]['sensitivity'] for c in CLASS_NAMES]), 4),
            "avg_specificity": round(np.mean([sens_spec[c]['specificity'] for c in CLASS_NAMES]), 4),
        }
    }
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Clinical report saved to {save_path}")
    
    return report


def main():
    """Main function to generate all clinical metrics."""
    print("=" * 60)
    print("🏥 Generating Clinical Metrics")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading model...")
    model = load_model()
    
    # Create dataset and dataloader
    print("\n📊 Loading data...")
    _, val_transform = get_transforms()
    dataset = SimpleLiverDataset(DATA_DIR, transform=val_transform)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=0
    )
    
    print(f"   Total samples: {len(dataset)}")
    
    # Get predictions
    print("\n🔮 Getting model predictions...")
    y_true, y_probs, y_pred = get_predictions(model, dataloader)
    
    # Compute metrics
    print("\n📈 Computing clinical metrics...")
    
    # 1. Sensitivity/Specificity per class
    sens_spec = compute_sensitivity_specificity(y_true, y_pred)
    
    # 2. Binary classification
    binary_results, y_true_binary, prob_advanced = compute_binary_classification(
        y_true, y_pred, y_probs
    )
    
    # Generate plots
    print("\n📊 Generating visualizations...")
    
    # 3. Calibration curves
    plot_calibration_curve(
        y_true, y_probs,
        save_path=OUTPUT_DIR / "calibration_curves.png"
    )
    
    # 4. Sensitivity/Specificity bar chart
    plot_sensitivity_specificity(
        sens_spec,
        save_path=OUTPUT_DIR / "sensitivity_specificity.png"
    )
    
    # 5. Binary confusion matrix
    plot_binary_confusion_matrix(
        binary_results,
        save_path=OUTPUT_DIR / "binary_confusion_matrix.png"
    )
    
    # 6. Clinical report
    report = generate_clinical_report(
        sens_spec, binary_results,
        save_path=OUTPUT_DIR / "clinical_report.json"
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 CLINICAL METRICS SUMMARY")
    print("=" * 60)
    print(f"\n{'Stage':<10} {'Sensitivity':<15} {'Specificity':<15}")
    print("-" * 40)
    for cls in CLASS_NAMES:
        print(f"{cls:<10} {sens_spec[cls]['sensitivity']:<15.4f} {sens_spec[cls]['specificity']:<15.4f}")
    print("-" * 40)
    print(f"\n📌 BINARY CLASSIFICATION (Early vs Advanced):")
    print(f"   Sensitivity: {binary_results['sensitivity']:.2%}")
    print(f"   Specificity: {binary_results['specificity']:.2%}")
    print(f"   Accuracy: {binary_results['accuracy']:.2%}")
    print(f"   F1-Score: {binary_results['f1_score']:.4f}")
    
    print(f"\n✅ All metrics saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
