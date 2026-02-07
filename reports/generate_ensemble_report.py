"""
Generate PDF Report for Ensemble Model (ResNet50 + EfficientNet-V2 + ViT).

Creates a professional PDF report with:
- Performance Summary
- Detailed Metrics Table involved
- Confusion Matrix
- ROC Curves

Usage:
    python reports/generate_ensemble_report.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, 
    precision_score, recall_score, roc_auc_score, 
    roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_CSV = OUTPUT_DIR / "ensemble_results.csv"
CONFUSION_MATRIX_IMG = OUTPUT_DIR / "final_analysis" / "ensemble_confusion_matrix.png"
PDF_OUTPUT = OUTPUT_DIR / "ensemble_analysis_report.pdf"

CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']

def load_data():
    """Load ensemble results."""
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_CSV}")
    
    df = pd.read_csv(RESULTS_CSV)
    
    # Parse probabilities
    # Columns are like 'ensemble_f0_prob', 'ensemble_f1_prob', ...
    prob_cols = [c for c in df.columns if 'ensemble_f' in c and '_prob' in c]
    prob_cols.sort() # Ensure F0-F4 order
    
    y_probs = df[prob_cols].values
    y_true = df['true_label'].map({name: i for i, name in enumerate(CLASS_NAMES)}).values
    y_pred = df['ensemble_pred_label'].map({name: i for i, name in enumerate(CLASS_NAMES)}).values
    
    return y_true, y_pred, y_probs

def compute_metrics(y_true, y_pred, y_probs):
    """Compute all necessary metrics."""
    metrics = {}
    
    # Overall
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # ROC-AUC
    try:
        y_true_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))
        metrics['roc_auc'] = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
    except:
        metrics['roc_auc'] = 0.0
        
    # Per-class
    metrics['per_class'] = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    
    return metrics

def create_title_page(pdf, metrics):
    """Create the report title page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    
    # Header
    fig.text(0.5, 0.75, 'Liver Fibrosis Staging', fontsize=28, ha='center', fontweight='bold', color='#1a237e')
    fig.text(0.5, 0.68, 'Ensemble Model Analysis Report', fontsize=22, ha='center', fontweight='bold', color='#333333')
    
    # Subheader
    fig.text(0.5, 0.60, 'Soft-Voting Ensemble: ResNet50 + EfficientNet-V2 + ViT', fontsize=16, ha='center', color='#555555')
    
    # Key Metrics Box
    fig.text(0.5, 0.45, 'Key Performance Indicators', fontsize=18, ha='center', fontweight='bold')
    
    fig.text(0.3, 0.38, f"Accuracy\n{metrics['accuracy']*100:.2f}%", fontsize=16, ha='center', 
             bbox=dict(facecolor='#e8f5e9', edgecolor='#2e7d32', boxstyle='round,pad=1'))
             
    fig.text(0.5, 0.38, f"QWK Score\n{metrics['kappa']:.4f}", fontsize=16, ha='center', 
             bbox=dict(facecolor='#e3f2fd', edgecolor='#1565c0', boxstyle='round,pad=1'))
             
    fig.text(0.7, 0.38, f"F1 Macro\n{metrics['f1_macro']:.4f}", fontsize=16, ha='center', 
             bbox=dict(facecolor='#fff3e0', edgecolor='#ff8f00', boxstyle='round,pad=1'))
    
    # Footer
    fig.text(0.5, 0.10, f"Generated: {datetime.now().strftime('%B %d, %Y')}", fontsize=12, ha='center', color='#777777')
    
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_metrics_page(pdf, metrics):
    """Create detailed metrics page."""
    fig = plt.figure(figsize=(11, 8.5))
    
    # Title
    fig.text(0.5, 0.95, 'Detailed Performance Metrics', fontsize=18, ha='center', fontweight='bold')
    
    # 1. Per-Class Metrics Table
    ax = fig.add_subplot(2, 1, 1)
    ax.axis('off')
    
    cols = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    cell_text = []
    
    for cls in CLASS_NAMES:
        row_data = metrics['per_class'][cls]
        cell_text.append([
            cls,
            f"{row_data['precision']:.4f}",
            f"{row_data['recall']:.4f}",
            f"{row_data['f1-score']:.4f}",
            str(row_data['support'])
        ])
    
    # Add weighted avg row
    w_avg = metrics['per_class']['weighted avg']
    cell_text.append([
        'Weighted Avg',
        f"{w_avg['precision']:.4f}",
        f"{w_avg['recall']:.4f}",
        f"{w_avg['f1-score']:.4f}",
        str(w_avg['support'])
    ])

    table = ax.table(cellText=cell_text, colLabels=cols, loc='center', cellLoc='center', bbox=[0.1, 0.1, 0.8, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    
    # Header styling
    for j in range(len(cols)):
        table[(0, j)].set_facecolor('#3f51b5')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # 2. Overall Metrics Text
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.axis('off')
    
    lines = [
        f"Overall Accuracy: {metrics['accuracy']*100:.2f}%",
        f"Cohen's Kappa (Quadratic): {metrics['kappa']:.4f}",
        f"ROC-AUC (Macro): {metrics['roc_auc']:.4f}",
        f"F1-Score (Macro): {metrics['f1_macro']:.4f}",
        f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}"
    ]
    
    for i, line in enumerate(lines):
        ax2.text(0.1, 0.8 - i*0.15, line, fontsize=14)
        
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_visualizations_page(pdf, y_true, y_probs):
    """Generate ROC curves and embed confusion matrix."""
    fig = plt.figure(figsize=(11, 8.5))
    
    # 1. Confusion Matrix (Load image)
    ax1 = fig.add_subplot(1, 2, 1)
    if CONFUSION_MATRIX_IMG.exists():
        img = plt.imread(str(CONFUSION_MATRIX_IMG))
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title("Confusion Matrix", fontweight='bold')
    else:
        ax1.text(0.5, 0.5, "Confusion Matrix Not Found", ha='center')
        ax1.axis('off')

    # 2. ROC Curves (Generate on the fly)
    ax2 = fig.add_subplot(1, 2, 2)
    
    y_true_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))
    colors = plt.cm.Set2(np.linspace(0, 1, len(CLASS_NAMES)))
    
    for i, color in enumerate(colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=color, lw=2, label=f'{CLASS_NAMES[i]} (area = {roc_auc:.2f})')

    ax2.plot([0, 1], [0, 1], 'k--', lw=2)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves', fontweight='bold')
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def main():
    print("Generating Ensemble Report...")
    
    # Load data
    try:
        y_true, y_pred, y_probs = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_probs)
    
    # Generate PDF
    with PdfPages(PDF_OUTPUT) as pdf:
        create_title_page(pdf, metrics)
        create_metrics_page(pdf, metrics)
        create_visualizations_page(pdf, y_true, y_probs)
        
    print(f"Successfully created report: {PDF_OUTPUT}")

if __name__ == "__main__":
    main()
