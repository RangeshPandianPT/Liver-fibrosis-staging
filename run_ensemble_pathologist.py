"""
THE PATHOLOGIST: Ensemble Analysis & Second Opinion System.

1. Merges Prediction CSVs.
2. Implements Weighted Soft Voting.
3. Computes Quadratic Weighted Kappa (QWK) & Confusion Matrix.
4. Saves final 'ensemble_results.csv'.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
CNN_PREDS_PATH = OUTPUT_DIR / "cnn_predictions.csv"
VIT_PREDS_PATH = OUTPUT_DIR / "vit_predictions.csv"
DEIT_PREDS_PATH = OUTPUT_DIR / "deit_predictions.csv"
RESULTS_PATH = OUTPUT_DIR / "ensemble_results.csv"
ANALYSIS_DIR = OUTPUT_DIR / "final_analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

def load_and_merge_data():
    """Load prediction CSVs and merge on filename."""
    if not CNN_PREDS_PATH.exists():
        raise FileNotFoundError(f"CNN predictions not found at {CNN_PREDS_PATH}")
    if not VIT_PREDS_PATH.exists():
        raise FileNotFoundError(f"ViT predictions not found at {VIT_PREDS_PATH}")
    if not DEIT_PREDS_PATH.exists():
        print(f"Warning: DeiT predictions not found at {DEIT_PREDS_PATH}. Proceeding without DeiT.")
        deit_df = None
    else:
        deit_df = pd.read_csv(DEIT_PREDS_PATH)
        
    cnn_df = pd.read_csv(CNN_PREDS_PATH)
    vit_df = pd.read_csv(VIT_PREDS_PATH)
    
    # Merge (ViT might have duplicates if code wasn't perfect, drop them just in case)
    cnn_df = cnn_df.drop_duplicates(subset=['filename'])
    vit_df = vit_df.drop_duplicates(subset=['filename'])
    if deit_df is not None:
        deit_df = deit_df.drop_duplicates(subset=['filename'])
    
    # Merge
    # CNN DF has: filename, true_label, resnet_*, effnet_*
    # ViT DF has: filename, true_label, vit_*
    # DeiT DF has: filename, true_label, deit_*
    
    merged = pd.merge(cnn_df, vit_df[['filename'] + [c for c in vit_df.columns if 'vit' in c]], 
                      on='filename', how='inner')
                      
    if deit_df is not None:
        merged = pd.merge(merged, deit_df[['filename'] + [c for c in deit_df.columns if 'deit' in c]],
                          on='filename', how='inner')
    
    print(f"Merged {len(merged)} samples.")
    return merged

def weighted_soft_voting(row):
    """
    Apply weighted voting logic.
    Weights derived from performance analysis:
    - ViT (Excellent at F2-F4): Weight 1.2
    - EfficientNet (Strong overall): Weight 1.0
    - ResNet (Baseline): Weight 0.8
    """
    # Weights
    W_VIT = 1.2
    W_EFF = 1.0
    W_RES = 0.8
    W_DEIT = 1.0
    
    probs = np.zeros(5)
    
    for i in range(5):
        p_res = row.get(f'resnet_f{i}_prob', 0)
        p_eff = row.get(f'effnet_f{i}_prob', 0)
        p_vit = row.get(f'vit_f{i}_prob', 0)
        p_deit = row.get(f'deit_f{i}_prob', 0)
        
        # Weighted Sum
        score = (p_res * W_RES) + (p_eff * W_EFF) + (p_vit * W_VIT) + (p_deit * W_DEIT)
        probs[i] = score
        
    return np.argmax(probs) # return class index

def weighted_soft_voting_probs(row):
    """Return aggregated probabilities for calculating metrics like AUC if needed."""
    W_VIT = 1.2
    W_EFF = 1.0
    W_RES = 0.8
    W_DEIT = 1.0
    
    probs = np.zeros(5)
    for i in range(5):
        p_res = row.get(f'resnet_f{i}_prob', 0)
        p_eff = row.get(f'effnet_f{i}_prob', 0)
        p_vit = row.get(f'vit_f{i}_prob', 0)
        p_deit = row.get(f'deit_f{i}_prob', 0)
        probs[i] = (p_res * W_RES) + (p_eff * W_EFF) + (p_vit * W_VIT) + (p_deit * W_DEIT)
    
    # Normalize
    return probs / probs.sum()

def main():
    print("Initializing Pathologist Ensemble System...")
    
    # 1. Load Data
    df = load_and_merge_data()
    
    # 2. Run Ensemble Inference
    tqdm_avail = True
    try:
        from tqdm import tqdm
        tqdm.pandas()
    except ImportError:
        tqdm_avail = False
    
    print("Calculating Ensemble Predictions...")
    # Apply voting
    if tqdm_avail:
        df['ensemble_pred_idx'] = df.progress_apply(weighted_soft_voting, axis=1)
        df['ensemble_probs'] = df.progress_apply(weighted_soft_voting_probs, axis=1)
    else:
        df['ensemble_pred_idx'] = df.apply(weighted_soft_voting, axis=1)
        df['ensemble_probs'] = df.apply(weighted_soft_voting_probs, axis=1)
        
    df['ensemble_pred_label'] = df['ensemble_pred_idx'].apply(lambda x: CLASS_NAMES[x])
    
    # 3. Calculate Metrics
    y_true = df['true_label'].map(CLASS_MAP).values
    y_pred = df['ensemble_pred_idx'].values
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Quadratic Weighted Kappa
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # F1 Score
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Precision & Recall (Macro)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    
    print("\n" + "="*40)
    print("ENSEMBLE PERFORMANCE REPORT")
    print("="*40)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Quadratic Weighted Kappa: {qwk:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print("="*40)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Ensemble Confusion Matrix (QWK={qwk:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Ensemble Prediction')
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'ensemble_confusion_matrix.png')
    print("Saved Confusion Matrix.")
    
    # 4. Save Final CSV
    # Flatten probs for CSV
    final_df = df.copy()
    for i in range(5):
        final_df[f'ensemble_f{i}_prob'] = final_df['ensemble_probs'].apply(lambda x: x[i])
    
    cols_to_keep = ['filename', 'true_label', 'ensemble_pred_label'] + \
                   [c for c in final_df.columns if '_prob' in c]
    
    final_output = final_df[cols_to_keep]
    final_output.to_csv(RESULTS_PATH, index=False)
    print(f"Detailed results saved to {RESULTS_PATH}")
    
    # Save Metrics Summary Text
    with open(ANALYSIS_DIR / 'ensemble_metrics.txt', 'w') as f:
        f.write("Ensemble Performance Report\n")
        f.write("===========================\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"QWK Score: {qwk:.4f}\n")
        f.write(f"F1 Macro: {f1_macro:.4f}\n")
        f.write(f"Precision Macro: {precision_macro:.4f}\n")
        f.write(f"Recall Macro: {recall_macro:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

if __name__ == "__main__":
    main()
