"""
Generate MedNeXt Model Evaluation Report (PDF).

This script creates a comprehensive PDF report for the MedNeXt model,
including performance metrics, confusion matrix, ROC curves, and training summary.

It first runs inference on the test set, then generates plots and a PDF.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import timm
import sys

# Paths
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))
from config import DATA_DIR, DEVICE, IMAGE_SIZE

OUTPUT_DIR = BASE_DIR / "outputs"
MEDNEXT_DIR = OUTPUT_DIR / "mednext"
MODEL_PATH = MEDNEXT_DIR / "best_mednext_model.pth"
PREDICTIONS_CSV = MEDNEXT_DIR / "mednext_predictions.csv"
PDF_OUTPUT = MEDNEXT_DIR / "mednext_evaluation_report.pdf"
MANIFEST_CSV = OUTPUT_DIR / "dataset_manifest.csv"

CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']
NUM_CLASSES = 5

# ---- Training Summary ----
# Hardcoded from training log output
TRAINING_SUMMARY = {
    'total_epochs': 50,
    'best_val_accuracy': 98.66,
    'final_train_loss': 0.3897,
    'final_train_accuracy': 100.00,
    'final_val_loss': 0.4397,
    'final_val_accuracy': 98.10,
    'train_samples': 5058,
    'val_samples': 1265,
    'backbone': 'ConvNeXt Tiny (timm)',
    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingLR',
    'batch_size': 16,
    'learning_rate': 1e-4,
    'label_smoothing': 0.1,
    'class_balancing': 'WeightedRandomSampler',
}


def run_inference():
    """Run inference on the test split and save predictions."""
    print("Loading model...")
    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    # Transforms (match training)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load dataset and get test split
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=None)
    labels = [s[1] for s in full_dataset.samples]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    val_subset = torch.utils.data.Subset(val_dataset, val_idx)

    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Running inference on {len(val_subset)} validation samples...")
    all_preds = []
    all_probs = []
    all_true = []
    all_paths = []

    with torch.no_grad():
        batch_start = 0
        for images, batch_labels in tqdm(val_loader, desc="Inference"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_true.extend(batch_labels.numpy().tolist())

            # Get original file paths for this batch
            for i in range(len(batch_labels)):
                idx = val_idx[batch_start + i]
                all_paths.append(full_dataset.samples[idx][0])
            batch_start += len(batch_labels)

    # Build DataFrame
    prob_cols = {f'mednext_{CLASS_NAMES[i]}_prob': [p[i] for p in all_probs] for i in range(NUM_CLASSES)}
    df = pd.DataFrame({
        'image_path': all_paths,
        'y_true': [CLASS_NAMES[t] for t in all_true],
        'y_pred': [CLASS_NAMES[p] for p in all_preds],
        **prob_cols
    })
    df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Predictions saved to {PREDICTIONS_CSV}")
    return df


def load_predictions():
    """Load existing predictions CSV."""
    if not PREDICTIONS_CSV.exists():
        return None
    return pd.read_csv(PREDICTIONS_CSV)


def calculate_metrics(y_true, y_pred, y_pred_probs):
    """Calculate all performance metrics."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    metrics['precision_macro'] = precision
    metrics['recall_macro'] = recall
    metrics['f1_macro'] = f1

    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_NAMES)
    for i, class_name in enumerate(CLASS_NAMES):
        metrics[f'f1_{class_name}'] = f[i]
        metrics[f'precision_{class_name}'] = p[i]
        metrics[f'recall_{class_name}'] = r[i]
        metrics[f'support_{class_name}'] = support[i]

    y_true_bin = label_binarize(y_true, classes=CLASS_NAMES)
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        metrics[f'auc_{class_name}'] = auc(fpr, tpr)

    return metrics


def generate_confusion_matrix(y_true, y_pred):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax1,
                annot_kws={'size': 14})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('True', fontsize=11)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax2,
                annot_kws={'size': 14})
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Predicted', fontsize=11)
    ax2.set_ylabel('True', fontsize=11)

    plt.suptitle('MedNeXt (ConvNeXt-Tiny) — Confusion Matrices', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = MEDNEXT_DIR / 'mednext_confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Confusion matrix saved: {save_path}")


def generate_roc_curves(y_true, y_pred_probs):
    """Generate and save ROC curves."""
    y_true_bin = label_binarize(y_true, classes=CLASS_NAMES)

    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=color, label=f'{class_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('MedNeXt (ConvNeXt-Tiny) — ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = MEDNEXT_DIR / 'mednext_roc_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ ROC curves saved: {save_path}")


def generate_per_class_bar_chart(metrics):
    """Generate a per-class bar chart for precision, recall, F1."""
    classes = CLASS_NAMES
    precision_vals = [metrics[f'precision_{c}'] for c in classes]
    recall_vals = [metrics[f'recall_{c}'] for c in classes]
    f1_vals = [metrics[f'f1_{c}'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, precision_vals, width, label='Precision', color='#42A5F5')
    bars2 = ax.bar(x, recall_vals, width, label='Recall', color='#66BB6A')
    bars3 = ax.bar(x + width, f1_vals, width, label='F1-Score', color='#FFA726')

    ax.set_xlabel('Fibrosis Stage', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('MedNeXt — Per-Class Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    # Add values on top
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    save_path = MEDNEXT_DIR / 'mednext_per_class_metrics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Per-class bar chart saved: {save_path}")


# ========== PDF REPORT PAGES ==========

def create_title_page(pdf, metrics):
    """Page 1: Title and Key Results."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')

    fig.text(0.5, 0.82, 'MedNeXt Model Evaluation Report',
             fontsize=26, ha='center', fontweight='bold', color='#1a237e')
    fig.text(0.5, 0.75, 'Liver Fibrosis Staging (F0–F4)',
             fontsize=18, ha='center', color='#283593')
    fig.text(0.5, 0.70, 'ConvNeXt-Tiny Backbone  ·  5-Class Classification',
             fontsize=12, ha='center', color='#5C6BC0', style='italic')

    # Key Results Box
    rect = matplotlib.patches.FancyBboxPatch(
        (0.2, 0.38), 0.6, 0.25, boxstyle="round,pad=0.02",
        linewidth=3, edgecolor='#2E7D32', facecolor='#E8F5E9')
    fig.add_artist(rect)

    acc = metrics['accuracy'] * 100
    kappa = metrics['cohens_kappa']
    f1 = metrics['f1_macro']

    fig.text(0.5, 0.58, "⚡ Performance Summary", fontsize=16, ha='center', fontweight='bold', color='#1B5E20')
    fig.text(0.5, 0.53, f"Best Validation Accuracy: {TRAINING_SUMMARY['best_val_accuracy']:.2f}%",
             fontsize=16, ha='center', fontweight='bold', color='#2E7D32')
    fig.text(0.5, 0.49, f"Test Accuracy: {acc:.2f}%",
             fontsize=14, ha='center', color='#2E7D32')
    fig.text(0.5, 0.45, f"Cohen's Kappa: {kappa:.4f}   |   F1-Score (Macro): {f1:.4f}",
             fontsize=13, ha='center', color='#2E7D32')

    # Training Info Box
    rect2 = matplotlib.patches.FancyBboxPatch(
        (0.15, 0.12), 0.7, 0.2, boxstyle="round,pad=0.02",
        linewidth=2, edgecolor='#1565C0', facecolor='#E3F2FD')
    fig.add_artist(rect2)

    fig.text(0.5, 0.27, "Training Configuration", fontsize=13, ha='center', fontweight='bold', color='#0D47A1')
    info_lines = [
        f"Backbone: {TRAINING_SUMMARY['backbone']}   |   Epochs: {TRAINING_SUMMARY['total_epochs']}   |   Batch Size: {TRAINING_SUMMARY['batch_size']}",
        f"Optimizer: {TRAINING_SUMMARY['optimizer']}   |   LR: {TRAINING_SUMMARY['learning_rate']}   |   Scheduler: {TRAINING_SUMMARY['scheduler']}",
        f"Label Smoothing: {TRAINING_SUMMARY['label_smoothing']}   |   Class Balancing: {TRAINING_SUMMARY['class_balancing']}",
        f"Train Samples: {TRAINING_SUMMARY['train_samples']}   |   Val Samples: {TRAINING_SUMMARY['val_samples']}"
    ]
    for i, line in enumerate(info_lines):
        fig.text(0.5, 0.23 - i*0.03, line, fontsize=10, ha='center', color='#1565C0')

    fig.text(0.5, 0.06, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             fontsize=9, ha='center', color='gray')
    fig.text(0.5, 0.03, "MedNeXtBranch — Liver Fibrosis Staging Pipeline",
             fontsize=9, ha='center', style='italic', color='gray')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_metrics_page(pdf, metrics):
    """Page 2: Detailed Metrics Table."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.95, "Detailed Performance Metrics",
             fontsize=20, ha='center', fontweight='bold', color='#1a237e')

    ax = fig.add_subplot(111)
    ax.axis('off')

    # Overall
    overall_data = [
        ['Accuracy', f"{metrics['accuracy']*100:.2f}%"],
        ["Cohen's Kappa", f"{metrics['cohens_kappa']:.4f}"],
        ['Precision (Macro)', f"{metrics['precision_macro']:.4f}"],
        ['Recall (Macro)', f"{metrics['recall_macro']:.4f}"],
        ['F1-Score (Macro)', f"{metrics['f1_macro']:.4f}"],
    ]

    table1 = ax.table(cellText=overall_data,
                      colLabels=['Metric', 'Value'],
                      loc='upper center', cellLoc='left',
                      bbox=[0.1, 0.65, 0.8, 0.25])
    table1.auto_set_font_size(False)
    table1.set_fontsize(12)
    table1.scale(1, 2.5)
    for (i, j), cell in table1.get_celld().items():
        if i == 0:
            cell.set_facecolor('#1a237e')
            cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#f5f5f5')

    # Per-class
    fig.text(0.5, 0.55, "Per-Class Performance",
             fontsize=14, ha='center', fontweight='bold', color='#1a237e')

    class_data = []
    for class_name in CLASS_NAMES:
        class_data.append([
            class_name,
            f"{metrics[f'precision_{class_name}']:.4f}",
            f"{metrics[f'recall_{class_name}']:.4f}",
            f"{metrics[f'f1_{class_name}']:.4f}",
            f"{metrics[f'auc_{class_name}']:.4f}",
            f"{int(metrics[f'support_{class_name}'])}"
        ])

    table2 = ax.table(cellText=class_data,
                      colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Support'],
                      loc='center', cellLoc='center',
                      bbox=[0.05, 0.15, 0.9, 0.35])
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1, 2.2)
    for (i, j), cell in table2.get_celld().items():
        if i == 0:
            cell.set_facecolor('#1a237e')
            cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#f5f5f5')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_visualization_page(pdf):
    """Page 3: Confusion Matrix and ROC Curves."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('MedNeXt Visualizations', fontsize=18, fontweight='bold', y=0.98, color='#1a237e')

    cm_path = MEDNEXT_DIR / 'mednext_confusion_matrix.png'
    if cm_path.exists():
        ax1 = fig.add_subplot(2, 1, 1)
        img = plt.imread(str(cm_path))
        ax1.imshow(img)
        ax1.axis('off')

    roc_path = MEDNEXT_DIR / 'mednext_roc_curves.png'
    if roc_path.exists():
        ax2 = fig.add_subplot(2, 1, 2)
        img = plt.imread(str(roc_path))
        ax2.imshow(img)
        ax2.axis('off')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_per_class_page(pdf):
    """Page 4: Per-class bar chart."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Per-Class Performance Breakdown', fontsize=18, fontweight='bold', y=0.98, color='#1a237e')

    chart_path = MEDNEXT_DIR / 'mednext_per_class_metrics.png'
    if chart_path.exists():
        ax = fig.add_subplot(111)
        img = plt.imread(str(chart_path))
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_classification_report_page(pdf, y_true, y_pred):
    """Page 5: Classification Report."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.95, "Classification Report",
             fontsize=20, ha='center', fontweight='bold', color='#1a237e')

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)

    fig.text(0.1, 0.85, report,
             fontsize=11, va='top', ha='left',
             family='monospace', linespacing=1.5)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("MedNeXt Model Evaluation Report Generator")
    print("=" * 60)

    # Step 1: Run inference (or load existing)
    df = load_predictions()
    if df is None:
        print("\n1. Running inference on validation set...")
        df = run_inference()
    else:
        print(f"\n1. Loaded existing predictions: {PREDICTIONS_CSV}")
        print(f"   Found {len(df)} samples")

    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    prob_cols = [f'mednext_{c}_prob' for c in CLASS_NAMES]
    y_pred_probs = df[prob_cols].values

    # Step 2: Calculate metrics
    print("\n2. Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_pred_probs)
    print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   Cohen's Kappa: {metrics['cohens_kappa']:.4f}")
    print(f"   F1-Score (Macro): {metrics['f1_macro']:.4f}")

    # Save metrics to JSON
    metrics_json = MEDNEXT_DIR / "mednext_metrics.json"
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"   ✓ Metrics saved: {metrics_json}")

    # Step 3: Generate visualizations
    print("\n3. Generating visualizations...")
    generate_confusion_matrix(y_true, y_pred)
    generate_roc_curves(y_true, y_pred_probs)
    generate_per_class_bar_chart(metrics)

    # Step 4: Create PDF
    print(f"\n4. Creating PDF report: {PDF_OUTPUT}")
    with PdfPages(PDF_OUTPUT) as pdf:
        create_title_page(pdf, metrics)
        create_metrics_page(pdf, metrics)
        create_visualization_page(pdf)
        create_per_class_page(pdf)
        create_classification_report_page(pdf, y_true, y_pred)

    print("\n" + "=" * 60)
    print("✅ Report generated successfully!")
    print(f"   PDF:              {PDF_OUTPUT}")
    print(f"   Confusion Matrix: {MEDNEXT_DIR / 'mednext_confusion_matrix.png'}")
    print(f"   ROC Curves:       {MEDNEXT_DIR / 'mednext_roc_curves.png'}")
    print(f"   Predictions:      {PREDICTIONS_CSV}")
    print(f"   Metrics JSON:     {metrics_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
