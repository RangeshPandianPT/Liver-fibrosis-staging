"""
Generate a combined PNG image with 5 confusion matrices:
  - Top row:    ConvNeXtV2, MedNeXt, DeiT
  - Bottom row: ResNet50, Ensemble
All matrices show raw counts (not normalized), using Blues colormap.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# ── paths ──
BASE_DIR  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_CSV = OUTPUT_DIR / "ensemble_results.csv"
SAVE_PATH   = OUTPUT_DIR / "combined_confusion_matrices.png"

CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']
CLASS_MAP   = {name: i for i, name in enumerate(CLASS_NAMES)}


def pred_from_probs(df, prefix, prob_cols):
    """Return integer predictions by argmax over probability columns."""
    probs = df[prob_cols].values
    return np.argmax(probs, axis=1)


def main():
    df = pd.read_csv(RESULTS_CSV)
    y_true = df['true_label'].map(CLASS_MAP).values

    # ── per-model predicted labels (argmax of probabilities) ──
    models = {
        'ConvNeXtV2': [f'convnext_F{i}_prob' for i in range(5)],
        'MedNeXt':    [f'mednext_F{i}_prob'   for i in range(5)],
        'DeiT':       [f'deit_f{i}_prob'      for i in range(5)],
        'ResNet50':   [f'resnet_f{i}_prob'     for i in range(5)],
    }

    preds = {}
    for name, cols in models.items():
        # Check if columns exist; skip if not
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"Warning: columns {missing} not found for {name}, skipping.")
            continue
        preds[name] = pred_from_probs(df, name, cols)

    # Ensemble prediction
    preds['Ensemble'] = df['ensemble_pred_label'].map(CLASS_MAP).values

    # ── figure layout: 3 on top, 2 on bottom (centred) ──
    fig = plt.figure(figsize=(20, 13))
    gs = gridspec.GridSpec(2, 6, hspace=0.35, wspace=0.45)  # 2 rows, 6 cols for centering

    # Top row: spans cols 0-1, 2-3, 4-5
    ax_positions_top = [
        gs[0, 0:2],
        gs[0, 2:4],
        gs[0, 4:6],
    ]
    # Bottom row: centred → spans cols 1-2, 3-4
    ax_positions_bot = [
        gs[1, 1:3],
        gs[1, 3:5],
    ]

    # Order: top = ConvNeXtV2, MedNeXt, DeiT  |  bottom = ResNet50, Ensemble
    order = ['ConvNeXtV2', 'MedNeXt', 'DeiT', 'ResNet50', 'Ensemble']
    ax_specs = ax_positions_top + ax_positions_bot

    for spec, name in zip(ax_specs, order):
        ax = fig.add_subplot(spec)
        cm = confusion_matrix(y_true, preds[name], labels=list(range(len(CLASS_NAMES))))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, cbar=False,
            annot_kws={'size': 13, 'weight': 'bold'},
            linewidths=0.5, linecolor='white',
        )
        ax.set_title(f'{name} Confusion Matrix', fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.tick_params(labelsize=10)

    fig.suptitle('Individual Model & Ensemble Confusion Matrices',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(SAVE_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined confusion matrices to {SAVE_PATH}")


if __name__ == '__main__':
    main()
