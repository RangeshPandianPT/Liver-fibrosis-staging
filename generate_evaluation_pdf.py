"""
Generate PDF Evaluation Report for ViT and ResNet50 Models.

Creates a professional PDF report with:
- Overall metrics comparison
- Per-class accuracy for F0-F4
- Confusion matrices
- ROC curves

Usage:
    python generate_evaluation_pdf.py
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
VIT_METRICS = OUTPUT_DIR / "metrics" / "vit_evaluation" / "vit_evaluation_results.json"
RESNET_METRICS = OUTPUT_DIR / "metrics" / "cnn_evaluation" / "evaluation_results.json"
PDF_OUTPUT = OUTPUT_DIR / "model_evaluation_report.pdf"

CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']


def load_metrics():
    """Load metrics from JSON files."""
    metrics = {}
    
    if VIT_METRICS.exists():
        try:
            with open(VIT_METRICS) as f:
                metrics['vit'] = json.load(f)
            print(f"Loaded ViT metrics from {VIT_METRICS}")
        except Exception as e:
            print(f"Error loading ViT metrics: {e}")
    
    if RESNET_METRICS.exists():
        try:
            with open(RESNET_METRICS) as f:
                data = json.load(f)
                
                # Check for resnet data
                if 'resnet' in data:
                    metrics['resnet'] = data['resnet']
                
                # Check for effnet data
                if 'effnet' in data:
                    metrics['effnet'] = data['effnet']
                    
                # Handle flattened structure (legacy/single model runs)
                if 'accuracy' in data and 'resnet' not in metrics and 'effnet' not in metrics:
                    # Provide a default key if uncertain, or try to infer
                    metrics['resnet'] = data
                    
            print(f"Loaded CNN metrics from {RESNET_METRICS}")
        except Exception as e:
            print(f"Error loading CNN metrics: {e}")
    
    return metrics


def create_title_page(pdf, metrics):
    """Create title page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    
    # Title
    fig.text(0.5, 0.7, 'Liver Fibrosis Staging', fontsize=28, ha='center', fontweight='bold')
    fig.text(0.5, 0.62, 'Model Evaluation Report', fontsize=24, ha='center', fontweight='bold')
    
    # Models
    models_text = []
    if 'vit' in metrics:
        models_text.append('ViT-B-16')
    if 'resnet' in metrics:
        models_text.append('ResNet50')
    
    fig.text(0.5, 0.50, f"Models: {' & '.join(models_text)}", fontsize=16, ha='center')
    
    # Date
    fig.text(0.5, 0.40, f"Generated: {datetime.now().strftime('%B %d, %Y')}", fontsize=14, ha='center')
    
    # Summary box
    if 'effnet' in metrics:
        eff_acc = metrics['effnet']['accuracy'] * 100
        fig.text(0.5, 0.34, f"EfficientNet-V2 Accuracy: {eff_acc:.2f}%", fontsize=16, ha='center',
                 color='#F57C00', fontweight='bold')
    
    if 'vit' in metrics:
        vit_acc = metrics['vit']['accuracy'] * 100
        fig.text(0.5, 0.28, f"ViT-B-16 Best Accuracy: {vit_acc:.2f}%", fontsize=18, ha='center', 
                 color='#2E7D32', fontweight='bold')
    
    if 'resnet' in metrics:
        resnet_acc = metrics['resnet']['accuracy'] * 100
        fig.text(0.5, 0.22, f"ResNet50 Accuracy: {resnet_acc:.2f}%", fontsize=16, ha='center',
                 color='#1565C0')
    
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def create_overall_metrics_page(pdf, metrics):
    """Create overall metrics comparison page."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Overall Metrics Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    models = list(metrics.keys())
    model_names = {'vit': 'ViT-B-16', 'resnet': 'ResNet50', 'effnet': 'EfficientNet-V2'}
    colors = {'vit': '#4CAF50', 'resnet': '#2196F3', 'effnet': '#FF9800'}
    
    # Accuracy comparison
    ax = axes[0, 0]
    accs = [metrics[m]['accuracy'] * 100 for m in models]
    bars = ax.bar([model_names[m] for m in models], accs, color=[colors[m] for m in models])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall Accuracy')
    ax.set_ylim(0, 105)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.2f}%', 
                ha='center', fontweight='bold')
    
    # F1 Score comparison
    ax = axes[0, 1]
    f1s = [metrics[m]['f1_macro'] for m in models]
    bars = ax.bar([model_names[m] for m in models], f1s, color=[colors[m] for m in models])
    ax.set_ylabel('F1-Score (Macro)')
    ax.set_title('F1-Score (Macro)')
    ax.set_ylim(0, 1.1)
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{f1:.4f}', 
                ha='center', fontweight='bold')
    
    # Cohen's Kappa
    ax = axes[1, 0]
    kappas = [metrics[m]['cohens_kappa'] for m in models]
    bars = ax.bar([model_names[m] for m in models], kappas, color=[colors[m] for m in models])
    ax.set_ylabel("Cohen's Kappa")
    ax.set_title("Cohen's Kappa (Quadratic)")
    ax.set_ylim(0, 1.1)
    for bar, k in zip(bars, kappas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{k:.4f}', 
                ha='center', fontweight='bold')
    
    # ROC-AUC
    ax = axes[1, 1]
    aucs = [metrics[m].get('roc_auc_macro', 0) or 0 for m in models]
    bars = ax.bar([model_names[m] for m in models], aucs, color=[colors[m] for m in models])
    ax.set_ylabel('ROC-AUC (Macro)')
    ax.set_title('ROC-AUC (Macro)')
    ax.set_ylim(0, 1.1)
    for bar, a in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{a:.4f}', 
                ha='center', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def create_per_class_metrics_page(pdf, metrics):
    """Create per-class metrics comparison page."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Per-Class Performance (F0-F4)', fontsize=16, fontweight='bold', y=0.98)
    
    models = list(metrics.keys())
    model_names = {'vit': 'ViT-B-16', 'resnet': 'ResNet50', 'effnet': 'EfficientNet-V2'}
    colors = {'vit': '#4CAF50', 'resnet': '#2196F3', 'effnet': '#FF9800'}
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    
    # Per-class Accuracy
    ax = axes[0, 0]
    for i, m in enumerate(models):
        accs = [metrics[m].get(f'accuracy_{c}', 0) * 100 for c in CLASS_NAMES]
        offset = width * (i - 0.5) if len(models) > 1 else 0
        bars = ax.bar(x + offset, accs, width if len(models) > 1 else 0.6, 
                     label=model_names[m], color=colors[m])
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{acc:.1f}', ha='center', fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 115)
    ax.legend()
    
    # Per-class F1-Score
    ax = axes[0, 1]
    for i, m in enumerate(models):
        f1s = [metrics[m].get(f'f1_{c}', 0) for c in CLASS_NAMES]
        offset = width * (i - 0.5) if len(models) > 1 else 0
        bars = ax.bar(x + offset, f1s, width if len(models) > 1 else 0.6,
                     label=model_names[m], color=colors[m])
        for bar, f1 in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{f1:.2f}', ha='center', fontsize=8)
    ax.set_ylabel('F1-Score')
    ax.set_title('Per-Class F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.2)
    ax.legend()
    
    # Per-class Precision
    ax = axes[1, 0]
    for i, m in enumerate(models):
        precs = [metrics[m].get(f'precision_{c}', 0) for c in CLASS_NAMES]
        offset = width * (i - 0.5) if len(models) > 1 else 0
        bars = ax.bar(x + offset, precs, width if len(models) > 1 else 0.6,
                     label=model_names[m], color=colors[m])
        for bar, p in zip(bars, precs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{p:.2f}', ha='center', fontsize=8)
    ax.set_ylabel('Precision')
    ax.set_title('Per-Class Precision')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.2)
    ax.legend()
    
    # Per-class Recall
    ax = axes[1, 1]
    for i, m in enumerate(models):
        recs = [metrics[m].get(f'recall_{c}', 0) for c in CLASS_NAMES]
        offset = width * (i - 0.5) if len(models) > 1 else 0
        bars = ax.bar(x + offset, recs, width if len(models) > 1 else 0.6,
                     label=model_names[m], color=colors[m])
        for bar, r in zip(bars, recs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{r:.2f}', ha='center', fontsize=8)
    ax.set_ylabel('Recall')
    ax.set_title('Per-Class Recall')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.2)
    ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def create_metrics_table_page(pdf, metrics):
    """Create detailed metrics table page."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    fig.text(0.5, 0.95, 'Detailed Metrics Summary', fontsize=16, ha='center', fontweight='bold')
    
    # Build table data
    models = list(metrics.keys())
    model_names = {'vit': 'ViT-B-16', 'resnet': 'ResNet50', 'effnet': 'EfficientNet-V2'}
    
    # Overall metrics table
    overall_data = []
    overall_rows = ['Accuracy', 'F1-Score (Macro)', 'F1-Score (Weighted)', 
                   'Precision (Macro)', 'Recall (Macro)', "Cohen's Kappa", 'ROC-AUC']
    
    for row_name in overall_rows:
        row = [row_name]
        for m in models:
            if row_name == 'Accuracy':
                row.append(f"{metrics[m]['accuracy']*100:.2f}%")
            elif row_name == 'F1-Score (Macro)':
                row.append(f"{metrics[m]['f1_macro']:.4f}")
            elif row_name == 'F1-Score (Weighted)':
                row.append(f"{metrics[m]['f1_weighted']:.4f}")
            elif row_name == 'Precision (Macro)':
                row.append(f"{metrics[m]['precision_macro']:.4f}")
            elif row_name == 'Recall (Macro)':
                row.append(f"{metrics[m]['recall_macro']:.4f}")
            elif row_name == "Cohen's Kappa":
                row.append(f"{metrics[m]['cohens_kappa']:.4f}")
            elif row_name == 'ROC-AUC':
                val = metrics[m].get('roc_auc_macro')
                row.append(f"{val:.4f}" if val else "N/A")
        overall_data.append(row)
    
    cols = ['Metric'] + [model_names[m] for m in models]
    
    # Overall metrics table
    table1 = ax.table(cellText=overall_data, colLabels=cols,
                      cellLoc='center', loc='upper center',
                      bbox=[0.1, 0.55, 0.8, 0.35])
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.5)
    
    # Header styling
    for j in range(len(cols)):
        table1[(0, j)].set_facecolor('#4CAF50')
        table1[(0, j)].set_text_props(fontweight='bold', color='white')
    
    # Per-class accuracy table
    fig.text(0.5, 0.48, 'Per-Class Accuracy', fontsize=14, ha='center', fontweight='bold')
    
    class_data = []
    for c in CLASS_NAMES:
        row = [c]
        for m in models:
            acc = metrics[m].get(f'accuracy_{c}', 0) * 100
            row.append(f"{acc:.2f}%")
        class_data.append(row)
    
    table2 = ax.table(cellText=class_data, colLabels=cols,
                      cellLoc='center', loc='center',
                      bbox=[0.1, 0.08, 0.8, 0.35])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 1.5)
    
    for j in range(len(cols)):
        table2[(0, j)].set_facecolor('#2196F3')
        table2[(0, j)].set_text_props(fontweight='bold', color='white')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def embed_existing_plots(pdf, metrics):
    """Embed existing confusion matrices and ROC curves."""
    # ViT Confusion Matrix
    vit_cm = OUTPUT_DIR / "metrics" / "vit_evaluation" / "vit_confusion_matrix.png"
    if vit_cm.exists():
        fig = plt.figure(figsize=(11, 8.5))
        img = plt.imread(str(vit_cm))
        plt.imshow(img)
        plt.axis('off')
        plt.title('ViT-B-16 Confusion Matrix', fontsize=14, fontweight='bold', pad=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # ViT ROC Curves
    vit_roc = OUTPUT_DIR / "metrics" / "vit_evaluation" / "vit_roc_curves.png"
    if vit_roc.exists():
        fig = plt.figure(figsize=(11, 8.5))
        img = plt.imread(str(vit_roc))
        plt.imshow(img)
        plt.axis('off')
        plt.title('ViT-B-16 ROC Curves', fontsize=14, fontweight='bold', pad=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # ResNet Confusion Matrix
    resnet_cm = OUTPUT_DIR / "metrics" / "cnn_evaluation" / "resnet_confusion_matrix.png"
    if resnet_cm.exists():
        fig = plt.figure(figsize=(11, 8.5))
        img = plt.imread(str(resnet_cm))
        plt.imshow(img)
        plt.axis('off')
        plt.title('ResNet50 Confusion Matrix', fontsize=14, fontweight='bold', pad=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # ResNet ROC Curves
    resnet_roc = OUTPUT_DIR / "metrics" / "cnn_evaluation" / "resnet_roc_curves.png"
    if resnet_roc.exists():
        fig = plt.figure(figsize=(11, 8.5))
        img = plt.imread(str(resnet_roc))
        plt.imshow(img)
        plt.axis('off')
        plt.title('ResNet50 ROC Curves', fontsize=14, fontweight='bold', pad=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # EfficientNet Confusion Matrix
    effnet_cm = OUTPUT_DIR / "metrics" / "cnn_evaluation" / "effnet_confusion_matrix.png"
    if effnet_cm.exists():
        fig = plt.figure(figsize=(11, 8.5))
        img = plt.imread(str(effnet_cm))
        plt.imshow(img)
        plt.axis('off')
        plt.title('EfficientNet-V2 Confusion Matrix', fontsize=14, fontweight='bold', pad=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # EfficientNet ROC Curves
    effnet_roc = OUTPUT_DIR / "metrics" / "cnn_evaluation" / "effnet_roc_curves.png"
    if effnet_roc.exists():
        fig = plt.figure(figsize=(11, 8.5))
        img = plt.imread(str(effnet_roc))
        plt.imshow(img)
        plt.axis('off')
        plt.title('EfficientNet-V2 ROC Curves', fontsize=14, fontweight='bold', pad=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def main():
    print("\n" + "=" * 60)
    print("GENERATING MODEL EVALUATION PDF REPORT")
    print("=" * 60)
    
    # Load metrics
    metrics = load_metrics()
    
    if not metrics:
        print("No metrics found! Run evaluation scripts first.")
        return
    
    # Create PDF
    with PdfPages(PDF_OUTPUT) as pdf:
        print("\nGenerating pages...")
        
        # Title page
        create_title_page(pdf, metrics)
        print("  ✓ Title page")
        
        # Overall metrics comparison
        create_overall_metrics_page(pdf, metrics)
        print("  ✓ Overall metrics page")
        
        # Per-class metrics
        create_per_class_metrics_page(pdf, metrics)
        print("  ✓ Per-class metrics page")
        
        # Detailed table
        create_metrics_table_page(pdf, metrics)
        print("  ✓ Metrics table page")
        
        # Embed existing plots
        embed_existing_plots(pdf, metrics)
        print("  ✓ Confusion matrices & ROC curves")
    
    print(f"\n{'='*60}")
    print(f"PDF Report saved to: {PDF_OUTPUT}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
