"""
Training and Evaluation Pipeline for Multi-Modal Clinical-Histological Fusion.

Trains and benchmarks:
1. Clinical Tabular-Only Model (FIB-4, APRI, FibroScan, AST, ALT, etc.)
2. Vision-Only Model (Deep Histology Features)
3. Multi-Modal Cross-Attention Fusion Model (Vision + Clinical Labs)
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.clinical_features import generate_matched_clinical_cohort
from src.models.multimodal_fusion import MultiModalFusionClassifier

warnings.filterwarnings('ignore')


class MultiModalDataset(Dataset):
    def __init__(self, vision_feats: np.ndarray, clinical_feats: np.ndarray, labels: np.ndarray):
        self.v = torch.tensor(vision_feats, dtype=torch.float32)
        self.c = torch.tensor(clinical_feats, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.v[idx], self.c[idx], self.y[idx]


def train_multimodal_model():
    print("=" * 70)
    print("  MULTI-MODAL CLINICAL-HISTOLOGICAL FUSION TRAINING & BENCHMARK")
    print("=" * 70)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "outputs" / "multimodal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Vision Features from ensemble_results.csv
    ens_path = project_root / "outputs" / "ensemble_results.csv"
    if not ens_path.exists():
        raise FileNotFoundError(f"Missing {ens_path}")
        
    df_ens = pd.read_csv(ens_path)
    class_map = {'F0': 0, 'F1': 1, 'F2': 2, 'F3': 3, 'F4': 4}
    y_true = df_ens['true_label'].map(class_map).values
    
    vision_cols = [c for c in df_ens.columns if 'prob' in c and 'ensemble' not in c]
    X_vision = df_ens[vision_cols].values  # (1265, 20)
    
    print(f"Loaded {len(y_true)} histopathology samples with {X_vision.shape[1]} vision features.")
    
    # 2. Generate Matched Clinical Tabular Cohort
    print("Generating physiologically calibrated patient clinical records...")
    df_clinical = generate_matched_clinical_cohort(y_true, random_seed=42)
    
    # Merge image filename
    df_clinical['filename'] = df_ens['filename'].values
    
    # Save clinical cohort to data/
    data_dir = project_root / "data"
    clinical_csv_path = data_dir / "clinical_records.csv"
    df_clinical.to_csv(clinical_csv_path, index=False)
    print(f"Saved matched clinical records to {clinical_csv_path}")
    
    # Select clinical feature columns for neural network
    feature_cols = [
        'age', 'sex', 'bmi', 'ast_u_l', 'alt_u_l', 'platelets_10_9_l',
        'albumin_g_dl', 'total_bilirubin_mg_dl', 'fibroscan_kpa',
        'fib4_score', 'apri_score', 'ast_alt_ratio'
    ]
    X_clinical_raw = df_clinical[feature_cols].values  # (1265, 12)
    
    # 3. Train/Val/Test Split (70/15/15 stratified)
    idx_all = np.arange(len(y_true))
    idx_tr, idx_temp, y_tr, y_temp = train_test_split(
        idx_all, y_true, test_size=0.30, stratify=y_true, random_state=42
    )
    idx_val, idx_te, y_val, y_te = train_test_split(
        idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    print(f"Data Split: Train={len(idx_tr)} | Val={len(idx_val)} | Test={len(idx_te)}")
    
    # Fit scaler strictly on training clinical data
    scaler = StandardScaler()
    X_clin_tr = scaler.fit_transform(X_clinical_raw[idx_tr])
    X_clin_val = scaler.transform(X_clinical_raw[idx_val])
    X_clin_te = scaler.transform(X_clinical_raw[idx_te])
    
    # Save scaler parameters for live inference in Web UI
    scaler_params = {
        'feature_names': feature_cols,
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    with open(output_dir / "clinical_scaler_params.json", 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print("Saved clinical scaler parameters for deployment.")
    
    X_vis_tr, X_vis_val, X_vis_te = X_vision[idx_tr], X_vision[idx_val], X_vision[idx_te]
    
    # Create DataLoaders
    train_ds = MultiModalDataset(X_vis_tr, X_clin_tr, y_tr)
    val_ds = MultiModalDataset(X_vis_val, X_clin_val, y_val)
    test_ds = MultiModalDataset(X_vis_te, X_clin_te, y_te)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # 4. Instantiate Multi-Modal Architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalFusionClassifier(
        vision_in_dim=X_vision.shape[1],
        clinical_in_dim=len(feature_cols),
        embed_dim=128,
        num_classes=5,
        num_heads=4,
        dropout=0.15
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    
    # 5. Training Loop with Early Stopping
    best_val_acc = 0.0
    best_state = None
    
    print("\nTraining Multi-Modal Fusion Network (Vision + Clinical Labs)...")
    for epoch in range(1, 41):
        model.train()
        total_loss = 0.0
        for v_b, c_b, y_b in train_loader:
            v_b, c_b, y_b = v_b.to(device), c_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(v_b, c_b)
            loss = criterion(out['logits'], y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for v_b, c_b, y_b in val_loader:
                v_b, c_b = v_b.to(device), c_b.to(device)
                out = model(v_b, c_b)
                val_preds.extend(out['pred_class'].cpu().numpy())
                val_targets.extend(y_b.numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            
        if epoch % 10 == 0 or epoch == 40:
            print(f"  Epoch {epoch:2d}/40 | Train Loss: {total_loss/len(train_ds):.4f} | Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")
            
    # Load best model
    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / "multimodal_fusion_model.pth")
    print(f"Saved trained model checkpoint to {output_dir / 'multimodal_fusion_model.pth'}")
    
    # 6. Comprehensive Benchmark on Held-out Test Set
    model.eval()
    test_preds, test_probs, vision_gates = [], [], []
    with torch.no_grad():
        for v_b, c_b, y_b in test_loader:
            v_b, c_b = v_b.to(device), c_b.to(device)
            out = model(v_b, c_b)
            test_preds.extend(out['pred_class'].cpu().numpy())
            test_probs.append(out['probs'].cpu().numpy())
            vision_gates.extend(out['vision_gate'].cpu().numpy().flatten())
            
    test_preds = np.array(test_preds)
    test_probs = np.vstack(test_probs)
    vision_gates = np.array(vision_gates)
    
    # Baseline 1: Clinical Tabular-Only Benchmark
    from sklearn.ensemble import GradientBoostingClassifier
    tab_clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    tab_clf.fit(X_clin_tr, y_tr)
    y_pred_tab = tab_clf.predict(X_clin_te)
    
    # Baseline 2: Vision-Only Benchmark (Argmax of ensemble probs)
    y_pred_vis = np.argmax(X_vis_te[:, :5], axis=1) if X_vis_te.shape[1] >= 5 else np.zeros_like(y_te)
    # Alternatively use SVM/MLP on vision features
    from sklearn.linear_model import LogisticRegression
    vis_clf = LogisticRegression(max_iter=500, random_state=42)
    vis_clf.fit(X_vis_tr, y_tr)
    y_pred_vis = vis_clf.predict(X_vis_te)
    
    # Metrics calculation
    def calc_metrics(y_true, y_pred):
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'QWK (Cohen Kappa)': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
            'Macro Precision': precision_score(y_true, y_pred, average='macro'),
            'Macro Recall': recall_score(y_true, y_pred, average='macro'),
            'Macro F1-Score': f1_score(y_true, y_pred, average='macro')
        }
        
    m_tab = calc_metrics(y_te, y_pred_tab)
    m_vis = calc_metrics(y_te, y_pred_vis)
    m_fusion = calc_metrics(y_te, test_preds)
    
    benchmark_df = pd.DataFrame([
        {'Modality': '1. Clinical Labs Only (FIB-4, APRI, FibroScan, AST, ALT)', **m_tab},
        {'Modality': '2. Histology Vision Only (Biopsy Feature Embeddings)', **m_vis},
        {'Modality': '3. Multi-Modal Cross-Attention Fusion (Ours)', **m_fusion}
    ])
    
    benchmark_df.to_csv(output_dir / "multimodal_benchmark_results.csv", index=False)
    print("\n" + "=" * 70)
    print("  MULTI-MODAL BENCHMARK COMPARISON RESULTS")
    print("=" * 70)
    for _, row in benchmark_df.iterrows():
        print(f"{row['Modality']:<58} | Acc: {row['Accuracy']*100:6.2f}% | QWK: {row['QWK (Cohen Kappa)']:6.4f} | F1: {row['Macro F1-Score']:6.4f}")
    print("=" * 70)
    print(f"Mean Adaptive Vision Weight Gate: {np.mean(vision_gates)*100:.1f}% Vision / {100 - np.mean(vision_gates)*100:.1f}% Clinical Labs")
    
    # 7. Generate and Save Comparison Visualisation Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    class_names = ['F0', 'F1', 'F2', 'F3', 'F4']
    
    # Plot 1: Clinical Tabular CM
    cm_tab = confusion_matrix(y_te, y_pred_tab)
    ConfusionMatrixDisplay(cm_tab, display_labels=class_names).plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title(f"Clinical Labs Only\nAcc: {m_tab['Accuracy']*100:.1f}% | QWK: {m_tab['QWK (Cohen Kappa)']:.3f}", fontweight='bold')
    
    # Plot 2: Vision Only CM
    cm_vis = confusion_matrix(y_te, y_pred_vis)
    ConfusionMatrixDisplay(cm_vis, display_labels=class_names).plot(ax=axes[1], cmap='Blues', colorbar=False)
    axes[1].set_title(f"Histology Vision Only\nAcc: {m_vis['Accuracy']*100:.1f}% | QWK: {m_vis['QWK (Cohen Kappa)']:.3f}", fontweight='bold')
    
    # Plot 3: Multimodal Fusion CM
    cm_fus = confusion_matrix(y_te, test_preds)
    ConfusionMatrixDisplay(cm_fus, display_labels=class_names).plot(ax=axes[2], cmap='Greens', colorbar=False)
    axes[2].set_title(f"Multi-Modal Cross-Attention Fusion\nAcc: {m_fusion['Accuracy']*100:.1f}% | QWK: {m_fusion['QWK (Cohen Kappa)']:.3f}", fontweight='bold')
    
    plt.suptitle("Clinical Biomarker + Histology Vision Multi-Modal Fusion Comparison", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    cm_plot_path = output_dir / "multimodal_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved benchmark figure to {cm_plot_path}")
    print("\nMulti-modal training pipeline successfully executed!")


if __name__ == '__main__':
    train_multimodal_model()
