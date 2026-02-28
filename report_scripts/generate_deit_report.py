"""
Generate Research Report for DeiT Model (PDF).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, cohen_kappa_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import textwrap

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
PREDICTIONS_CSV = OUTPUT_DIR / "deit_predictions.csv"
MANIFEST_CSV = OUTPUT_DIR / "dataset_manifest.csv"
PDF_OUTPUT = OUTPUT_DIR / "deit_model_report.pdf"

CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']

def load_data():
    """Load predictions and true labels."""
    if not PREDICTIONS_CSV.exists():
        print(f"Error: {PREDICTIONS_CSV} not found.")
        return None
    
    df_pred = pd.read_csv(PREDICTIONS_CSV)
    
    if MANIFEST_CSV.exists():
        print(f"Loading manifest from {MANIFEST_CSV}...")
        df_manifest = pd.read_csv(MANIFEST_CSV)
        # Create a filename column in manifest if it doesn't exist or use image_path
        if 'filename' not in df_manifest.columns:
            df_manifest['filename'] = df_manifest['image_path'].apply(lambda x: Path(x).name)
        
        # Merge on filename
        # We only want the assigned_split column
        # Rename filename in pred if needed, but generate_deit_predictions outputs 'filename'
        
        if 'filename' not in df_pred.columns:
             print("Error: 'filename' column missing in predictions CSV.")
             return df_pred
             
        df = pd.merge(df_pred, df_manifest[['filename', 'assigned_split']], on='filename', how='inner')
        
        # Filter for Test set
        print(f"Total predictions: {len(df_pred)}")
        print(f"Matched manifest: {len(df)}")
        
        df_test = df[df['assigned_split'] == 'Test']
        print(f"Test set predictions: {len(df_test)}")
        
        if len(df_test) > 0:
            return df_test
        else:
            print("Warning: No Test samples found. Returning all matched.")
            return df
    else:
        print("Warning: Manifest not found. Using all predictions.")
        return df_pred

def calculate_metrics(df):
    """Calculate comprehensive metrics."""
    y_true = df['true_label'].values
    
    # Get probabilities
    prob_cols = [c for c in df.columns if 'prob' in c]
    y_prob = df[prob_cols].values
    
    # Get predicted labels (index)
    y_pred_idx = np.argmax(y_prob, axis=1)
    y_pred = [CLASS_NAMES[i] for i in y_pred_idx]
    
    # Map string labels to indices for some metrics
    class_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    y_true_idx = np.array([class_map[l] for l in y_true])
    
    # Basic Metrics
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Per-class and Macro
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_NAMES)
    
    metrics = {
        'accuracy': acc,
        'kappa': kappa,
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        },
        'y_true_idx': y_true_idx,
        'y_pred_idx': y_pred_idx,
        'y_prob': y_prob
    }
    return metrics

def add_header_footer(fig, page_num):
    """Add standard header and footer."""
    fig.text(0.95, 0.02, f"Page {page_num}", ha='right', fontsize=10, color='gray')
    fig.text(0.05, 0.02, f"DeiT-Small Model Evaluation Report", ha='left', fontsize=10, color='gray')
    fig.text(0.95, 0.97, datetime.now().strftime('%Y-%m-%d'), ha='right', fontsize=10, color='gray')

def create_title_page(pdf, metrics):
    """Page 1: Title and Summary."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    
    # Title
    fig.text(0.5, 0.75, 'DeiT-Small Model Evaluation', 
             fontsize=26, ha='center', fontweight='bold', color='#1a237e')
    fig.text(0.5, 0.68, 'Liver Fibrosis Staging', 
             fontsize=20, ha='center', fontweight='normal', color='#283593')
    
    # Stats Box
    acc = metrics['accuracy'] * 100
    kappa = metrics['kappa']
    
    fig.text(0.5, 0.50, f"Overall Accuracy: {acc:.2f}%", fontsize=22, ha='center', fontweight='bold', color='#2E7D32')
    fig.text(0.5, 0.45, f"Cohen's Kappa: {kappa:.4f}", fontsize=18, ha='center', color='#555')
    
    add_header_footer(fig, 1)
    pdf.savefig(fig)
    plt.close()

def create_metrics_table(pdf, metrics):
    """Page 2: Detailed Metrics."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.1, 0.90, "Detailed Performance Metrics", fontsize=20, fontweight='bold', color='#1a237e')
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    col_labels = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    cell_text = []
    
    p = metrics['per_class']['precision']
    r = metrics['per_class']['recall']
    f = metrics['per_class']['f1']
    s = metrics['per_class']['support']
    
    for i, class_name in enumerate(CLASS_NAMES):
        cell_text.append([
            class_name,
            f"{p[i]:.4f}",
            f"{r[i]:.4f}",
            f"{f[i]:.4f}",
            str(s[i])
        ])
    
    # Add valid averages (Macro)
    cell_text.append(['---', '---', '---', '---', '---'])
    cell_text.append([
        'Macro Avg',
        f"{np.mean(p):.4f}",
        f"{np.mean(r):.4f}",
        f"{np.mean(f):.4f}",
        str(np.sum(s))
    ])
    
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0.1, 0.2, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    
    # Styling
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#1a237e')
            cell.set_text_props(color='white', fontweight='bold')
        elif i == len(cell_text): # Macro row
            cell.set_facecolor('#e8eaf6')
            cell.set_text_props(fontweight='bold')
    
    add_header_footer(fig, 2)
    pdf.savefig(fig)
    plt.close()

def create_confusion_matrix(pdf, metrics):
    """Page 3: Confusion Matrix."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Confusion Matrix', fontsize=18, fontweight='bold', y=0.95, color='#1a237e')
    
    y_true = metrics['y_true_idx']
    y_pred = metrics['y_pred_idx']
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    ax = fig.add_subplot(111)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    
    add_header_footer(fig, 3)
    pdf.savefig(fig)
    plt.close()

def create_roc_curves(pdf, metrics):
    """Page 4: ROC Curves."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('ROC Curves (One-vs-Rest)', fontsize=18, fontweight='bold', y=0.95, color='#1a237e')
    
    y_true = metrics['y_true_idx']
    y_prob = metrics['y_prob']
    
    y_true_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))
    
    ax = fig.add_subplot(111)
    
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curves', fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)
    
    add_header_footer(fig, 4)
    pdf.savefig(fig)
    plt.close()

def main():
    print("Loading data...")
    df = load_data()
    if df is None: return
    
    print("Calculating metrics...")
    metrics = calculate_metrics(df)
    
    print(f"Generating PDF report to: {PDF_OUTPUT}")
    with PdfPages(PDF_OUTPUT) as pdf:
        create_title_page(pdf, metrics)
        create_metrics_table(pdf, metrics)
        create_confusion_matrix(pdf, metrics)
        create_roc_curves(pdf, metrics)
        
    print("Done!")

if __name__ == "__main__":
    main()
