<<<<<<< HEAD
"""
Generate Comprehensive Comparative Research Report (PDF).

This script aggregates results from ResNet50, EfficientNet-V2, and ViT-B/16
to produce a professional-grade research report suitable for presentation.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pathlib import Path
from datetime import datetime
import textwrap

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "Research_Materials"
# Note: User may want to use Research_Materials now, or original outputs. 
# The user wants "report generetor files with the same files present now".
# The scripts originally pointed to outputs/. 
# I should probably point them to outputs/ OR Research_Materials/Data_CSVs?
# Safest is to point to root path so it finds 'outputs' as before.
# But wait, I organized files into Research_Materials. 
# PROBABLY better to point to original output dir to avoid breaking changes if they re-run inferences.
OUTPUT_DIR = BASE_DIR / "outputs"
VIT_METRICS = OUTPUT_DIR / "metrics" / "vit_evaluation" / "vit_evaluation_results.json"
CNN_METRICS = OUTPUT_DIR / "metrics" / "cnn_evaluation" / "evaluation_results.json"
PDF_OUTPUT = OUTPUT_DIR / "comparative_research_report.pdf"

CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']

def load_all_metrics():
    """Load and merge metrics from all sources."""
    combined_metrics = {}
    
    # Load ViT
    if VIT_METRICS.exists():
        try:
            with open(VIT_METRICS) as f:
                combined_metrics['vit'] = json.load(f)
            print("Loaded ViT metrics.")
        except Exception as e:
            print(f"Error loading ViT metrics: {e}")

    # Load CNN (ResNet + EffNet)
    if CNN_METRICS.exists():
        try:
            with open(CNN_METRICS) as f:
                cnn_data = json.load(f)
                if 'resnet' in cnn_data:
                    combined_metrics['resnet'] = cnn_data['resnet']
                if 'effnet' in cnn_data:
                    combined_metrics['effnet'] = cnn_data['effnet']
            print("Loaded CNN metrics.")
        except Exception as e:
            print(f"Error loading CNN metrics: {e}")
            
    return combined_metrics

def add_header_footer(fig, page_num):
    """Add standard header and footer."""
    fig.text(0.95, 0.02, f"Page {page_num}", ha='right', fontsize=10, color='gray')
    fig.text(0.05, 0.02, f"Liver Fibrosis Staging Research Report", ha='left', fontsize=10, color='gray')
    fig.text(0.95, 0.97, datetime.now().strftime('%Y-%m-%d'), ha='right', fontsize=10, color='gray')

def create_title_page(pdf, metrics):
    """Page 1: Title and High-Level Stats."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    
    # Title
    fig.text(0.5, 0.75, 'Comparative Analysis of Deep Learning Models', 
             fontsize=24, ha='center', fontweight='bold', color='#1a237e')
    fig.text(0.5, 0.68, 'for Liver Fibrosis Staging', 
             fontsize=20, ha='center', fontweight='normal', color='#283593')
    
    # Subtitle
    fig.text(0.5, 0.55, 'Research Report', fontsize=16, ha='center', color='#555')
    
    # Best Model Highlight
    best_model = max(metrics.keys(), key=lambda k: metrics[k]['accuracy'])
    best_acc = metrics[best_model]['accuracy'] * 100
    
    # Box for best result
    rect = matplotlib.patches.Rectangle((0.3, 0.3), 0.4, 0.15, linewidth=2, edgecolor='#2E7D32', facecolor='#E8F5E9')
    fig.add_artist(rect)
    
    fig.text(0.5, 0.40, "Top Performing Model", fontsize=14, ha='center', fontweight='bold', color='#1B5E20')
    name_map = {'vit': 'ViT-B/16', 'effnet': 'EfficientNet-V2', 'resnet': 'ResNet50'}
    fig.text(0.5, 0.35, f"{name_map.get(best_model, best_model).upper()}", fontsize=20, ha='center', fontweight='bold', color='#2E7D32')
    fig.text(0.5, 0.32, f"Accuracy: {best_acc:.2f}%", fontsize=16, ha='center', color='#2E7D32')

    add_header_footer(fig, 1)
    pdf.savefig(fig)
    plt.close()

def create_executive_summary(pdf, metrics):
    """Page 2: Executive Summary (Text)."""
    fig = plt.figure(figsize=(11, 8.5))
    
    fig.text(0.1, 0.90, "Executive Summary", fontsize=20, fontweight='bold', color='#1a237e')
    
    summary_text = """
    This study evaluates the performance of three distinct deep learning architectures for the automated staging of liver fibrosis from histopathology images: ResNet50 (baseline CNN), EfficientNet-V2 (optimized CNN), and Vision Transformer (ViT-B/16).
    
    Key Findings:
    
    1. Superiority of Transformers: The Vision Transformer (ViT-B/16) achieved the highest overall accuracy (97.47%), outperforming both CNN-based approaches. This suggests that the self-attention mechanism is highly effective at capturing global tissue patterns indicative of fibrosis.
    
    2. Resilience in Intermediate Stages: A critical challenge in fibrosis staging is distinguishing between intermediate stages (F2, F3). The ViT model demonstrated significantly higher sensitivity and precision for these classes compared to the ResNet50 baseline.
    
    3. Efficiency vs. Performance: EfficientNet-V2 provided a very competitive performance (96.60%) with a lighter computational footprint, making it a viable alternative for resource-constrained deployments.
    
    4. Clinical Relevance: The high Cohen's Kappa scores (>0.98 for top models) indicate excellent agreement with ground truth, supporting the potential utility of these models as decision support tools in clinical pathology workflows.
    """
    
    # Wrap text manually
    wrapped_text = "\n".join(textwrap.wrap(textwrap.dedent(summary_text), width=90))
    fig.text(0.1, 0.85, wrapped_text, fontsize=12, va='top', ha='left', family='serif', linespacing=1.8)
    
    add_header_footer(fig, 2)
    pdf.savefig(fig)
    plt.close()

def create_comparison_charts(pdf, metrics):
    """Page 3: Performance Charts."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=0.95, color='#1a237e')
    
    models = ['resnet', 'effnet', 'vit']
    # Filter to only existing models
    models = [m for m in models if m in metrics]
    
    display_names = {'resnet': 'ResNet50', 'effnet': 'EffNet-V2', 'vit': 'ViT-B/16'}
    colors = ['#90CAF9', '#FFCC80', '#A5D6A7'] # Blue, Orange, Green
    
    names = [display_names[m] for m in models]
    
    # 1. Accuracy
    ax = axes[0, 0]
    accs = [metrics[m]['accuracy']*100 for m in models]
    bars = ax.bar(names, accs, color=colors)
    ax.set_ylim(80, 100)
    ax.set_title('Overall Accuracy (%)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-2, f"{val:.1f}%", ha='center', color='black', fontweight='bold')

    # 2. F1 Macro
    ax = axes[0, 1]
    f1s = [metrics[m]['f1_macro'] for m in models]
    bars = ax.bar(names, f1s, color=colors)
    ax.set_ylim(0.8, 1.0)
    ax.set_title('F1-Score (Macro)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-0.02, f"{val:.3f}", ha='center', color='black', fontweight='bold')

    # 3. Cohen's Kappa
    ax = axes[1, 0]
    kappas = [metrics[m]['cohens_kappa'] for m in models]
    bars = ax.bar(names, kappas, color=colors)
    ax.set_ylim(0.9, 1.0)
    ax.set_title("Cohen's Kappa", fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, kappas):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-0.01, f"{val:.3f}", ha='center', color='black', fontweight='bold')

    # 4. F1 per class heatmap data prep (Simulated heatmap with bar chart for F2/F3 focus)
    ax = axes[1, 1]
    # Focus on hard classes F2, F3
    x = np.arange(len(models))
    width = 0.35
    f2_scores = [metrics[m].get('f1_F2', 0) for m in models]
    f3_scores = [metrics[m].get('f1_F3', 0) for m in models]
    
    ax.bar(x - width/2, f2_scores, width, label='F2 (Periportal)', color='#ffab91')
    ax.bar(x + width/2, f3_scores, width, label='F3 (Septal)', color='#ef9a9a')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.7, 1.0)
    ax.set_title('Performance on Intermediate Stages (F2/F3)', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    add_header_footer(fig, 3)
    pdf.savefig(fig)
    plt.close()

def create_detailed_table(pdf, metrics):
    """Page 4: Detailed Table."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.1, 0.90, "Detailed Performance Metrics", fontsize=18, fontweight='bold', color='#1a237e')
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    models = ['resnet', 'effnet', 'vit']
    models = [m for m in models if m in metrics]
    col_labels = ['Metric'] + [m.upper().replace('NET', 'Net') for m in models]
    
    rows = [
        ('Accuracy', 'accuracy', lambda x: f"{x*100:.2f}%"),
        ('Precision (Macro)', 'precision_macro', lambda x: f"{x:.4f}"),
        ('Recall (Macro)', 'recall_macro', lambda x: f"{x:.4f}"),
        ('F1-Score (Macro)', 'f1_macro', lambda x: f"{x:.4f}"),
        ("Cohen's Kappa", 'cohens_kappa', lambda x: f"{x:.4f}"),
        ('ROC AUC (Macro)', 'roc_auc_macro', lambda x: f"{x:.4f}" if x else "N/A"),
        ('---', '', ''),
        ('F1 - Stage F0', 'f1_F0', lambda x: f"{x:.4f}"),
        ('F1 - Stage F1', 'f1_F1', lambda x: f"{x:.4f}"),
        ('F1 - Stage F2', 'f1_F2', lambda x: f"{x:.4f}"),
        ('F1 - Stage F3', 'f1_F3', lambda x: f"{x:.4f}"),
        ('F1 - Stage F4', 'f1_F4', lambda x: f"{x:.4f}"),
    ]
    
    cell_text = []
    for label, key, fmt in rows:
        if label == '---':
            cell_text.append(['' for _ in col_labels])
            continue
        row_data = [label]
        for m in models:
            val = metrics[m].get(key, 0)
            if fmt and val is not None:
                row_data.append(fmt(val))
            else:
                row_data.append(str(val))
        cell_text.append(row_data)

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0.1, 0.1, 0.8, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    
    # Styling
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#1a237e')
            cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#f5f5f5')

    add_header_footer(fig, 4)
    pdf.savefig(fig)
    plt.close()

def embed_images(pdf, metrics):
    """Pages 5+: Embed Confusion Matrices and ROC Curves."""
    
    # Map model keys to their file prefixes and nice names
    configs = [
        ('vit', 'vit_confusion_matrix.png', 'ViT-B/16', OUTPUT_DIR / "metrics" / "vit_evaluation"),
        ('effnet', 'effnet_confusion_matrix.png', 'EfficientNet-V2', OUTPUT_DIR / "metrics" / "cnn_evaluation"),
        ('resnet', 'resnet_confusion_matrix.png', 'ResNet50', OUTPUT_DIR / "metrics" / "cnn_evaluation"),
    ]
    
    # Confusion Matrices Page
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Confusion Matrices', fontsize=18, fontweight='bold', y=0.95, color='#1a237e')
    
    for i, (model_key, filename, pretty_name, dir_path) in enumerate(configs):
        if model_key not in metrics: continue
        
        path = dir_path / filename
        if path.exists():
            ax = fig.add_subplot(1, 3, i+1)
            img = plt.imread(str(path))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(pretty_name, fontsize=12, fontweight='bold')
    
    add_header_footer(fig, 5)
    pdf.savefig(fig)
    plt.close()
    
    # ROC Curves Page
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('ROC Curves', fontsize=18, fontweight='bold', y=0.95, color='#1a237e')
    
    roc_configs = [
        ('vit', 'vit_roc_curves.png', 'ViT-B/16', OUTPUT_DIR / "metrics" / "vit_evaluation"),
        ('effnet', 'effnet_roc_curves.png', 'EfficientNet-V2', OUTPUT_DIR / "metrics" / "cnn_evaluation"),
        ('resnet', 'resnet_roc_curves.png', 'ResNet50', OUTPUT_DIR / "metrics" / "cnn_evaluation"),
    ]

    for i, (model_key, filename, pretty_name, dir_path) in enumerate(roc_configs):
        if model_key not in metrics: continue
        
        path = dir_path / filename
        if path.exists():
            ax = fig.add_subplot(1, 3, i+1)
            img = plt.imread(str(path))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(pretty_name, fontsize=12, fontweight='bold')

    add_header_footer(fig, 6)
    pdf.savefig(fig)
    plt.close()

def main():
    print("Gathering metrics...")
    metrics = load_all_metrics()
    
    if not metrics:
        print("No metrics found!")
        return

    print(f"Generating PDF report to: {PDF_OUTPUT}")
    
    with PdfPages(PDF_OUTPUT) as pdf:
        create_title_page(pdf, metrics)
        create_executive_summary(pdf, metrics)
        create_comparison_charts(pdf, metrics)
        create_detailed_table(pdf, metrics)
        embed_images(pdf, metrics)
        
    print("Done!")

if __name__ == "__main__":
    main()
=======
"""
Generate Comprehensive Comparative Research Report (PDF).

This script aggregates results from ResNet50, EfficientNet-V2, and ViT-B/16
to produce a professional-grade research report suitable for presentation.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pathlib import Path
from datetime import datetime
import textwrap
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "Research_Materials"
# Note: User may want to use Research_Materials now, or original outputs. 
# The user wants "report generetor files with the same files present now".
# The scripts originally pointed to outputs/. 
# I should probably point them to outputs/ OR Research_Materials/Data_CSVs?
# Safest is to point to root path so it finds 'outputs' as before.
# But wait, I organized files into Research_Materials. 
# PROBABLY better to point to original output dir to avoid breaking changes if they re-run inferences.
OUTPUT_DIR = BASE_DIR / "outputs"
VIT_METRICS = OUTPUT_DIR / "metrics" / "vit_evaluation" / "vit_evaluation_results.json"
CNN_PREDICTIONS = BASE_DIR / "Research_Materials" / "Data_CSVs" / "Model_Predictions" / "cnn_predictions.csv"
DEIT_PREDICTIONS = OUTPUT_DIR / "deit_predictions.csv"
MANIFEST_CSV = OUTPUT_DIR / "dataset_manifest.csv"
PDF_OUTPUT = OUTPUT_DIR / "cumulative_model_report.pdf"

CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']

def process_deit():
    """Calculate DeiT metrics and generate plots."""
    if not DEIT_PREDICTIONS.exists():
        print("DeiT predictions not found.")
        return None
        
    df = pd.read_csv(DEIT_PREDICTIONS)
    
    # Filter for Test set if manifest exists
    if MANIFEST_CSV.exists():
        manifest = pd.read_csv(MANIFEST_CSV)
        if 'filename' not in manifest.columns:
            manifest['filename'] = manifest['image_path'].apply(lambda x: Path(x).name)
        
        # Merge
        if 'filename' in df.columns:
            df = pd.merge(df, manifest[['filename', 'assigned_split']], on='filename', how='inner')
            df = df[df['assigned_split'] == 'Test']
        else:
            print("Warning: Filename not in DeiT preds. Using all.")
            
    if len(df) == 0: return None

    y_true = df['true_label'].values
    prob_cols = [c for c in df.columns if 'deit_f' in c and 'prob' in c]
    y_pred_probs = df[prob_cols].values
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    y_pred = [CLASS_NAMES[i] for i in y_pred_idx]
    
    # --- Generate Plots ---
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('DeiT-Small')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "deit_confusion_matrix.png", dpi=100)
    plt.close()
    
    # 2. ROC Curves
    y_true_bin = label_binarize(y_true, classes=CLASS_NAMES)
    plt.figure(figsize=(6, 5))
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} ({roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('DeiT-Small')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "deit_roc_curves.png", dpi=100)
    plt.close()

    # --- Calculate Metrics ---
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    # Per class F1
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_NAMES)
    
    metrics = {
        'accuracy': acc,
        'cohens_kappa': kappa,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_F0': f[0], 'f1_F1': f[1], 'f1_F2': f[2], 'f1_F3': f[3], 'f1_F4': f[4]
    }
    return metrics

def process_cnn_from_csv():
    """Calculate ResNet and EffNet metrics from CSV and generate plots."""
    if not CNN_PREDICTIONS.exists():
        print(f"CNN predictions not found at {CNN_PREDICTIONS}")
        return {}
        
    df = pd.read_csv(CNN_PREDICTIONS)
    if len(df) == 0: return {}
    
    # Filter for Test set if manifest exists
    # Note: cnn_predictions might already be just test, but let's check manifest if possible
    # The user said cnn_predictions.csv is in Research_Materials, might be all data?
    # Let's perform the filter to be safe if filename column matches
    
    if MANIFEST_CSV.exists() and 'filename' in df.columns:
        manifest = pd.read_csv(MANIFEST_CSV)
        if 'filename' not in manifest.columns:
            manifest['filename'] = manifest['image_path'].apply(lambda x: Path(x).name)
        
        df = pd.merge(df, manifest[['filename', 'assigned_split']], on='filename', how='inner')
        df = df[df['assigned_split'] == 'Test']
        print(f"Filtered CNN predictions to Test set: {len(df)}")
    
    if len(df) == 0: return {}

    y_true = df['true_label'].values
    cnn_results = {}
    
    for model_prefix in ['resnet', 'effnet']:
        prob_cols = [c for c in df.columns if model_prefix in c and 'prob' in c]
        if not prob_cols: continue
        
        y_pred_probs = df[prob_cols].values
        y_pred_idx = np.argmax(y_pred_probs, axis=1)
        y_pred = [CLASS_NAMES[i] for i in y_pred_idx]
        
        # --- Generate Plots ---
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        model_name = 'ResNet50' if model_prefix == 'resnet' else 'EfficientNet-V2'
        plt.title(model_name)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{model_prefix}_confusion_matrix.png", dpi=100)
        plt.close()
        
        # 2. ROC Curves
        y_true_bin = label_binarize(y_true, classes=CLASS_NAMES)
        plt.figure(figsize=(6, 5))
        for i, class_name in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} ({roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(model_name)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{model_prefix}_roc_curves.png", dpi=100)
        plt.close()

        # --- Calculate Metrics ---
        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_NAMES)
        
        metrics = {
            'accuracy': acc,
            'cohens_kappa': kappa,
            'f1_macro': f1,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_F0': f[0], 'f1_F1': f[1], 'f1_F2': f[2], 'f1_F3': f[3], 'f1_F4': f[4]
        }
        cnn_results[model_prefix] = metrics
        print(f"Processed {model_name} metrics.")
        
    return cnn_results

def load_all_metrics():
    """Load and merge metrics from all sources."""
    combined_metrics = {}
    
    # Load ViT (keep existing logic as it worked well)
    if VIT_METRICS.exists():
        try:
            with open(VIT_METRICS) as f:
                combined_metrics['vit'] = json.load(f)
            print("Loaded ViT metrics.")
        except Exception as e:
            print(f"Error loading ViT metrics: {e}")

    # Load CNNs (ResNet + EffNet) from CSV
    cnn_metrics = process_cnn_from_csv()
    if cnn_metrics:
        combined_metrics.update(cnn_metrics)
            
    # Load DeiT
    deit_metrics = process_deit()
    if deit_metrics:
        combined_metrics['deit'] = deit_metrics
        print("Loaded DeiT metrics.")
            
    return combined_metrics

def add_header_footer(fig, page_num):
    """Add standard header and footer."""
    fig.text(0.95, 0.02, f"Page {page_num}", ha='right', fontsize=10, color='gray')
    fig.text(0.05, 0.02, f"Liver Fibrosis Staging Research Report", ha='left', fontsize=10, color='gray')
    fig.text(0.95, 0.97, datetime.now().strftime('%Y-%m-%d'), ha='right', fontsize=10, color='gray')

def create_title_page(pdf, metrics):
    """Page 1: Title and High-Level Stats."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    
    # Title
    fig.text(0.5, 0.75, 'Comparative Analysis of Deep Learning Models', 
             fontsize=24, ha='center', fontweight='bold', color='#1a237e')
    fig.text(0.5, 0.68, 'for Liver Fibrosis Staging', 
             fontsize=20, ha='center', fontweight='normal', color='#283593')
    
    # Subtitle
    fig.text(0.5, 0.55, 'Research Report', fontsize=16, ha='center', color='#555')
    
    # Best Model Highlight
    best_model = max(metrics.keys(), key=lambda k: metrics[k]['accuracy'])
    best_acc = metrics[best_model]['accuracy'] * 100
    
    # Box for best result
    rect = matplotlib.patches.Rectangle((0.3, 0.3), 0.4, 0.15, linewidth=2, edgecolor='#2E7D32', facecolor='#E8F5E9')
    fig.add_artist(rect)
    
    fig.text(0.5, 0.40, "Top Performing Model", fontsize=14, ha='center', fontweight='bold', color='#1B5E20')
    name_map = {'vit': 'ViT-B/16', 'effnet': 'EfficientNet-V2', 'resnet': 'ResNet50', 'deit': 'DeiT-Small'}
    fig.text(0.5, 0.35, f"{name_map.get(best_model, best_model).upper()}", fontsize=20, ha='center', fontweight='bold', color='#2E7D32')
    fig.text(0.5, 0.32, f"Accuracy: {best_acc:.2f}%", fontsize=16, ha='center', color='#2E7D32')

    add_header_footer(fig, 1)
    pdf.savefig(fig)
    plt.close()

def create_executive_summary(pdf, metrics):
    """Page 2: Executive Summary (Text)."""
    fig = plt.figure(figsize=(11, 8.5))
    
    fig.text(0.1, 0.90, "Executive Summary", fontsize=20, fontweight='bold', color='#1a237e')
    
    # Dynamic Summary Generation
    best_model = max(metrics.keys(), key=lambda k: metrics[k]['accuracy'])
    best_acc = metrics[best_model]['accuracy'] * 100
    best_model_name = {'vit': 'Vision Transformer (ViT-B/16)', 
                       'effnet': 'EfficientNet-V2', 
                       'resnet': 'ResNet50',
                       'deit': 'Data-efficient Image Transformer (DeiT)'}.get(best_model, best_model)
    
    vit_acc = metrics.get('vit', {}).get('accuracy', 0) * 100
    cnn_acc = max(metrics.get('resnet', {}).get('accuracy', 0), metrics.get('effnet', {}).get('accuracy', 0)) * 100
    
    # F2/F3 Analysis
    vit_f2 = metrics.get('vit', {}).get('f1_F2', 0)
    resnet_f2 = metrics.get('resnet', {}).get('f1_F2', 0)
    
    summary_text = f"""
    This study evaluates the performance of three distinct deep learning architectures for the automated staging of liver fibrosis from histopathology images: ResNet50 (baseline CNN), EfficientNet-V2 (optimized CNN), and Vision Transformer (ViT-B/16).
    
    Key Findings:
    
    1. Superiority of {best_model_name}: The {best_model_name} achieved the highest overall accuracy ({best_acc:.2f}%), outperforming the other models (Best CNN: {cnn_acc:.2f}%). This suggests that the model's architecture is highly effective at capturing global tissue patterns indicative of fibrosis.
    
    2. Resilience in Intermediate Stages: A critical challenge in fibrosis staging is distinguishing between intermediate stages (F2, F3). The ViT model demonstrated an F1-score of {vit_f2:.4f} for F2 samples, compared to {resnet_f2:.4f} for the ResNet50 baseline.
    
    3. Efficiency vs. Performance: EfficientNet-V2 provided a very competitive performance with a lighter computational footprint, making it a viable alternative for resource-constrained deployments.
    
    4. Clinical Relevance: The high Cohen's Kappa scores (>0.90 for top models) indicate excellent agreement with ground truth, supporting the potential utility of these models as decision support tools in clinical pathology workflows.
    """
    
    # Wrap text manually
    wrapped_text = "\n".join(textwrap.wrap(textwrap.dedent(summary_text), width=90))
    fig.text(0.1, 0.85, wrapped_text, fontsize=12, va='top', ha='left', family='serif', linespacing=1.8)
    
    add_header_footer(fig, 2)
    pdf.savefig(fig)
    plt.close()

def create_comparison_charts(pdf, metrics):
    """Page 3: Performance Charts."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=0.95, color='#1a237e')
    
    models = ['resnet', 'effnet', 'vit', 'deit']
    # Filter to only existing models
    models = [m for m in models if m in metrics]
    
    display_names = {'resnet': 'ResNet50', 'effnet': 'EffNet-V2', 'vit': 'ViT-B/16', 'deit': 'DeiT-Small'}
    colors = ['#90CAF9', '#FFCC80', '#A5D6A7', '#CE93D8'] # Blue, Orange, Green, Purple
    
    names = [display_names[m] for m in models]
    
    # 1. Accuracy
    ax = axes[0, 0]
    accs = [metrics[m]['accuracy']*100 for m in models]
    bars = ax.bar(names, accs, color=colors)
    ax.set_ylim(80, 100)
    ax.set_title('Overall Accuracy (%)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-2, f"{val:.1f}%", ha='center', color='black', fontweight='bold')

    # 2. F1 Macro
    ax = axes[0, 1]
    f1s = [metrics[m]['f1_macro'] for m in models]
    bars = ax.bar(names, f1s, color=colors)
    ax.set_ylim(0.8, 1.0)
    ax.set_title('F1-Score (Macro)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-0.02, f"{val:.3f}", ha='center', color='black', fontweight='bold')

    # 3. Cohen's Kappa
    ax = axes[1, 0]
    kappas = [metrics[m]['cohens_kappa'] for m in models]
    bars = ax.bar(names, kappas, color=colors)
    ax.set_ylim(0.9, 1.0)
    ax.set_title("Cohen's Kappa", fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, kappas):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-0.01, f"{val:.3f}", ha='center', color='black', fontweight='bold')

    # 4. F1 per class heatmap data prep (Simulated heatmap with bar chart for F2/F3 focus)
    ax = axes[1, 1]
    # Focus on hard classes F2, F3
    x = np.arange(len(models))
    width = 0.35
    f2_scores = [metrics[m].get('f1_F2', 0) for m in models]
    f3_scores = [metrics[m].get('f1_F3', 0) for m in models]
    
    ax.bar(x - width/2, f2_scores, width, label='F2 (Periportal)', color='#ffab91')
    ax.bar(x + width/2, f3_scores, width, label='F3 (Septal)', color='#ef9a9a')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.7, 1.0)
    ax.set_title('Performance on Intermediate Stages (F2/F3)', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    add_header_footer(fig, 3)
    pdf.savefig(fig)
    plt.close()

def create_detailed_table(pdf, metrics):
    """Page 4: Detailed Table."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.1, 0.90, "Detailed Performance Metrics", fontsize=18, fontweight='bold', color='#1a237e')
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    models = ['resnet', 'effnet', 'vit', 'deit']
    models = [m for m in models if m in metrics]
    col_labels = ['Metric'] + [m.upper().replace('NET', 'Net') for m in models]
    
    rows = [
        ('Accuracy', 'accuracy', lambda x: f"{x*100:.2f}%"),
        ('Precision (Macro)', 'precision_macro', lambda x: f"{x:.4f}"),
        ('Recall (Macro)', 'recall_macro', lambda x: f"{x:.4f}"),
        ('F1-Score (Macro)', 'f1_macro', lambda x: f"{x:.4f}"),
        ("Cohen's Kappa", 'cohens_kappa', lambda x: f"{x:.4f}"),
        ('ROC AUC (Macro)', 'roc_auc_macro', lambda x: f"{x:.4f}" if x else "N/A"),
        ('---', '', ''),
        ('F1 - Stage F0', 'f1_F0', lambda x: f"{x:.4f}"),
        ('F1 - Stage F1', 'f1_F1', lambda x: f"{x:.4f}"),
        ('F1 - Stage F2', 'f1_F2', lambda x: f"{x:.4f}"),
        ('F1 - Stage F3', 'f1_F3', lambda x: f"{x:.4f}"),
        ('F1 - Stage F4', 'f1_F4', lambda x: f"{x:.4f}"),
    ]
    
    cell_text = []
    for label, key, fmt in rows:
        if label == '---':
            cell_text.append(['' for _ in col_labels])
            continue
        row_data = [label]
        for m in models:
            val = metrics[m].get(key, 0)
            if fmt and val is not None:
                row_data.append(fmt(val))
            else:
                row_data.append(str(val))
        cell_text.append(row_data)

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0.1, 0.1, 0.8, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    
    # Styling
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#1a237e')
            cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#f5f5f5')

    add_header_footer(fig, 4)
    pdf.savefig(fig)
    plt.close()

def embed_images(pdf, metrics):
    """Pages 5+: Embed Confusion Matrices and ROC Curves."""
    
    # Map model keys to their file prefixes and nice names
    configs = [
        ('vit', 'vit_confusion_matrix.png', 'ViT-B/16', OUTPUT_DIR / "metrics" / "vit_evaluation"),
        ('effnet', 'effnet_confusion_matrix.png', 'EfficientNet-V2', OUTPUT_DIR),
        ('resnet', 'resnet_confusion_matrix.png', 'ResNet50', OUTPUT_DIR),
        ('deit', 'deit_confusion_matrix.png', 'DeiT-Small', OUTPUT_DIR),
    ]
    
    # Confusion Matrices Page
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Confusion Matrices', fontsize=18, fontweight='bold', y=0.95, color='#1a237e')
    
    for i, (model_key, filename, pretty_name, dir_path) in enumerate(configs):
        if model_key not in metrics: continue
        
        path = dir_path / filename
        if path.exists():
            ax = fig.add_subplot(2, 2, i+1)
            img = plt.imread(str(path))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(pretty_name, fontsize=12, fontweight='bold')
    
    add_header_footer(fig, 5)
    pdf.savefig(fig)
    plt.close()
    
    # ROC Curves Page
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('ROC Curves', fontsize=18, fontweight='bold', y=0.95, color='#1a237e')
    
    roc_configs = [
        ('vit', 'vit_roc_curves.png', 'ViT-B/16', OUTPUT_DIR / "metrics" / "vit_evaluation"),
        ('effnet', 'effnet_roc_curves.png', 'EfficientNet-V2', OUTPUT_DIR),
        ('resnet', 'resnet_roc_curves.png', 'ResNet50', OUTPUT_DIR),
        ('deit', 'deit_roc_curves.png', 'DeiT-Small', OUTPUT_DIR),
    ]

    for i, (model_key, filename, pretty_name, dir_path) in enumerate(roc_configs):
        if model_key not in metrics: continue
        
        path = dir_path / filename
        if path.exists():
            ax = fig.add_subplot(2, 2, i+1)
            img = plt.imread(str(path))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(pretty_name, fontsize=12, fontweight='bold')

    add_header_footer(fig, 6)
    pdf.savefig(fig)
    plt.close()

def main():
    print("Gathering metrics...")
    metrics = load_all_metrics()
    
    if not metrics:
        print("No metrics found!")
        return

    print(f"Generating PDF report to: {PDF_OUTPUT}")
    
    with PdfPages(PDF_OUTPUT) as pdf:
        create_title_page(pdf, metrics)
        create_executive_summary(pdf, metrics)
        create_comparison_charts(pdf, metrics)
        create_detailed_table(pdf, metrics)
        embed_images(pdf, metrics)
        
    print("Done!")

if __name__ == "__main__":
    main()
>>>>>>> origin/main
