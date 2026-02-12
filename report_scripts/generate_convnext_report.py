"""
Generate ConvNeXt Model Evaluation Report (PDF).

This script creates a comprehensive PDF report for the ConvNeXt model,
including performance metrics, confusion matrix, and ROC curves.
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
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    cohen_kappa_score, confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
CONVNEXT_DIR = OUTPUT_DIR / "convnext"
CONVNEXT_PREDICTIONS = CONVNEXT_DIR / "convnext_predictions.csv"
MANIFEST_CSV = OUTPUT_DIR / "dataset_manifest.csv"
PDF_OUTPUT = CONVNEXT_DIR / "convnext_evaluation_report.pdf"

CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']

def load_and_process_data():
    """Load ConvNeXt predictions and calculate metrics."""
    if not CONVNEXT_PREDICTIONS.exists():
        print(f"Error: ConvNeXt predictions not found at {CONVNEXT_PREDICTIONS}")
        return None
        
    df = pd.read_csv(CONVNEXT_PREDICTIONS)
    
    # Get true labels from manifest
    if MANIFEST_CSV.exists():
        manifest = pd.read_csv(MANIFEST_CSV)
        manifest = manifest[manifest['assigned_split'] == 'Test'].reset_index(drop=True)
        
        # Match by filename
        df['filename'] = df['image_path'].apply(lambda x: Path(x).name)
        manifest['filename'] = manifest['image_path'].apply(lambda x: Path(x).name)
        
        df = pd.merge(df, manifest[['filename', 'y_true']], on='filename', how='inner')
    else:
        print("Warning: Manifest not found, cannot get true labels")
        return None
    
    # Extract predictions
    y_true = df['y_true'].values
    prob_cols = [f'convnext_{c}_prob' for c in CLASS_NAMES]
    y_pred_probs = df[prob_cols].values
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    y_pred = [CLASS_NAMES[i] for i in y_pred_idx]
    
    return y_true, y_pred, y_pred_probs, df

def calculate_metrics(y_true, y_pred, y_pred_probs):
    """Calculate all performance metrics."""
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Macro averages
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    metrics['precision_macro'] = precision
    metrics['recall_macro'] = recall
    metrics['f1_macro'] = f1
    
    # Per-class F1
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_NAMES)
    for i, class_name in enumerate(CLASS_NAMES):
        metrics[f'f1_{class_name}'] = f[i]
        metrics[f'precision_{class_name}'] = p[i]
        metrics[f'recall_{class_name}'] = r[i]
        metrics[f'support_{class_name}'] = support[i]
    
    # ROC AUC per class
    y_true_bin = label_binarize(y_true, classes=CLASS_NAMES)
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        metrics[f'auc_{class_name}'] = auc(fpr, tpr)
    
    return metrics

def generate_confusion_matrix(y_true, y_pred):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(CONVNEXT_DIR / 'convnext_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

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
    plt.title('ConvNeXt ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONVNEXT_DIR / 'convnext_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_title_page(pdf, metrics):
    """Page 1: Title and Key Results."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    
    # Title
    fig.text(0.5, 0.75, 'ConvNeXt Model Evaluation', 
             fontsize=26, ha='center', fontweight='bold', color='#1a237e')
    fig.text(0.5, 0.68, 'Liver Fibrosis Staging', 
             fontsize=20, ha='center', color='#283593')
    
    # Key Results Box
    rect = matplotlib.patches.Rectangle((0.25, 0.35), 0.5, 0.25, 
                                        linewidth=3, edgecolor='#2E7D32', 
                                        facecolor='#E8F5E9')
    fig.add_artist(rect)
    
    acc = metrics['accuracy'] * 100
    kappa = metrics['cohens_kappa']
    f1 = metrics['f1_macro']
    
    fig.text(0.5, 0.55, "Performance Summary", 
             fontsize=16, ha='center', fontweight='bold', color='#1B5E20')
    fig.text(0.5, 0.50, f"Accuracy: {acc:.2f}%", 
             fontsize=18, ha='center', fontweight='bold', color='#2E7D32')
    fig.text(0.5, 0.45, f"Cohen's Kappa: {kappa:.4f}", 
             fontsize=14, ha='center', color='#2E7D32')
    fig.text(0.5, 0.41, f"F1-Score (Macro): {f1:.4f}", 
             fontsize=14, ha='center', color='#2E7D32')
    
    # Footer
    fig.text(0.5, 0.15, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
             fontsize=10, ha='center', color='gray')
    fig.text(0.5, 0.10, "ConvNeXt Tiny with Balanced Training (WeightedRandomSampler)", 
             fontsize=10, ha='center', style='italic', color='gray')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_metrics_page(pdf, metrics):
    """Page 2: Detailed Metrics Table."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.95, "Detailed Performance Metrics", 
             fontsize=20, ha='center', fontweight='bold', color='#1a237e')
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Overall Metrics
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
    
    # Style header
    for (i, j), cell in table1.get_celld().items():
        if i == 0:
            cell.set_facecolor('#1a237e')
            cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#f5f5f5')
    
    # Per-Class Metrics
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
    
    # Style header
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
    fig.suptitle('ConvNeXt Visualizations', fontsize=18, fontweight='bold', y=0.98, color='#1a237e')
    
    # Confusion Matrix
    cm_path = CONVNEXT_DIR / 'convnext_confusion_matrix.png'
    if cm_path.exists():
        ax1 = fig.add_subplot(2, 1, 1)
        img = plt.imread(str(cm_path))
        ax1.imshow(img)
        ax1.axis('off')
    
    # ROC Curves
    roc_path = CONVNEXT_DIR / 'convnext_roc_curves.png'
    if roc_path.exists():
        ax2 = fig.add_subplot(2, 1, 2)
        img = plt.imread(str(roc_path))
        ax2.imshow(img)
        ax2.axis('off')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_classification_report_page(pdf, y_true, y_pred):
    """Page 4: Classification Report."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.95, "Classification Report", 
             fontsize=20, ha='center', fontweight='bold', color='#1a237e')
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    
    fig.text(0.1, 0.85, report, 
             fontsize=11, va='top', ha='left', 
             family='monospace', linespacing=1.5)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 60)
    print("ConvNeXt Model Evaluation Report Generator")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading predictions...")
    result = load_and_process_data()
    if result is None:
        print("Error: Could not load data")
        return
    
    y_true, y_pred, y_pred_probs, df = result
    print(f"   Loaded {len(y_true)} test samples")
    
    # Calculate metrics
    print("\n2. Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_pred_probs)
    print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   Cohen's Kappa: {metrics['cohens_kappa']:.4f}")
    
    # Generate visualizations
    print("\n3. Generating visualizations...")
    generate_confusion_matrix(y_true, y_pred)
    print("   ✓ Confusion matrix saved")
    generate_roc_curves(y_true, y_pred_probs)
    print("   ✓ ROC curves saved")
    
    # Create PDF
    print(f"\n4. Creating PDF report: {PDF_OUTPUT}")
    with PdfPages(PDF_OUTPUT) as pdf:
        create_title_page(pdf, metrics)
        create_metrics_page(pdf, metrics)
        create_visualization_page(pdf)
        create_classification_report_page(pdf, y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("✅ Report generated successfully!")
    print(f"   Location: {PDF_OUTPUT}")
    print("=" * 60)

if __name__ == "__main__":
    main()
