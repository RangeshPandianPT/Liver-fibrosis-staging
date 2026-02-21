"""
Generate PDF Training Report for ConvNeXt V2.

Creates a professional PDF report with:
- Title page with key metrics
- Training & Validation Accuracy curve
- Training & Validation Loss curve
- Epoch-by-epoch metrics table

Usage:
    python report_scripts/generate_convnextv2_training_report.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
PDF_OUTPUT = OUTPUT_DIR / "convnextv2" / "convnextv2_training_report.pdf"

# ── Training data (epochs 23-50) ──────────────────────────────────────
# Parsed from the training run output
EPOCH_DATA = [
    # (epoch, train_loss, train_acc, val_loss, val_acc)
    (23, 0.4094, 99.07, 0.4963, 95.81),
    (24, 0.4083, 99.27, 0.4136, 98.97),
    (25, 0.3910, 99.98, 0.4179, 98.97),
    (26, 0.3902, 100.00, 0.4174, 98.97),
    (27, 0.3901, 100.00, 0.4192, 98.97),
    (28, 0.3950, 99.78, 0.4273, 98.58),
    (29, 0.4019, 99.57, 0.4213, 98.97),
    (30, 0.4013, 99.53, 0.4251, 98.50),
    (31, 0.3944, 99.82, 0.4210, 98.66),
    (32, 0.3904, 100.00, 0.4213, 98.97),
    (33, 0.3907, 99.98, 0.4234, 98.89),
    (34, 0.3904, 99.98, 0.4201, 98.97),
    (35, 0.3917, 99.88, 0.4273, 98.74),
    (36, 0.3916, 99.96, 0.4238, 98.81),
    (37, 0.3900, 100.00, 0.4213, 98.97),
    (38, 0.3924, 99.88, 0.4251, 98.58),
    (39, 0.3917, 99.92, 0.4240, 98.81),
    (40, 0.3917, 99.94, 0.4200, 98.81),
    (41, 0.3900, 100.00, 0.4195, 98.81),
    (42, 0.3899, 100.00, 0.4192, 99.05),
    (43, 0.3898, 100.00, 0.4189, 98.89),
    (44, 0.3898, 100.00, 0.4188, 98.89),
    (45, 0.3898, 100.00, 0.4186, 98.89),
    (46, 0.3898, 100.00, 0.4184, 98.89),
    (47, 0.3898, 100.00, 0.4182, 98.89),
    (48, 0.3899, 100.00, 0.4186, 98.89),
    (49, 0.3898, 100.00, 0.4185, 98.89),
    (50, 0.3898, 100.00, 0.4189, 98.74),
]

# Parse into arrays
epochs     = [d[0] for d in EPOCH_DATA]
train_loss = [d[1] for d in EPOCH_DATA]
train_acc  = [d[2] for d in EPOCH_DATA]
val_loss   = [d[3] for d in EPOCH_DATA]
val_acc    = [d[4] for d in EPOCH_DATA]

BEST_VAL_ACC = 99.05
BEST_EPOCH = 42
TOTAL_EPOCHS = 50
TRAIN_SAMPLES = 5058
VAL_SAMPLES = 1265

# ── Color palette ─────────────────────────────────────────────────────
C_PRIMARY   = '#1565C0'   # Deep blue
C_SECONDARY = '#E53935'   # Red
C_ACCENT    = '#2E7D32'   # Green
C_BG        = '#FAFAFA'
C_HEADER    = '#1B5E20'
C_GOLD      = '#F9A825'


def create_title_page(pdf):
    """Professional title page with key highlights."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')

    # Top bar
    fig.patches.append(plt.Rectangle((0, 0.82), 1, 0.18,
                       transform=fig.transFigure, facecolor=C_PRIMARY, zorder=0))

    fig.text(0.5, 0.90, 'ConvNeXt V2 — Training Report',
             fontsize=28, ha='center', va='center', fontweight='bold', color='white')
    fig.text(0.5, 0.84, 'Liver Fibrosis Staging (F0–F4)',
             fontsize=16, ha='center', va='center', color='#BBDEFB')

    # Key metrics cards
    card_y = 0.60
    cards = [
        ('Best Val Accuracy', f'{BEST_VAL_ACC:.2f}%',  C_ACCENT),
        ('Best Epoch',        f'{BEST_EPOCH}',          C_PRIMARY),
        ('Total Epochs',      f'{TOTAL_EPOCHS}',        '#F57C00'),
    ]
    card_w = 0.22
    gap = 0.06
    start_x = 0.5 - (len(cards) * card_w + (len(cards)-1) * gap) / 2
    for i, (label, value, color) in enumerate(cards):
        x = start_x + i * (card_w + gap)
        fig.patches.append(plt.Rectangle((x, card_y - 0.06), card_w, 0.14,
                           transform=fig.transFigure, facecolor=color,
                           alpha=0.12, edgecolor=color, linewidth=2, zorder=0))
        fig.text(x + card_w/2, card_y + 0.04, value,
                 fontsize=24, ha='center', fontweight='bold', color=color)
        fig.text(x + card_w/2, card_y - 0.03, label,
                 fontsize=11, ha='center', color='#555555')

    # Dataset info
    fig.text(0.5, 0.40,
             f'Dataset: {TRAIN_SAMPLES} train / {VAL_SAMPLES} val  •  '
             f'Architecture: ConvNeXt V2 Tiny  •  Optimizer: AdamW + CosineAnnealing',
             fontsize=11, ha='center', color='#666666')

    # Date
    fig.text(0.5, 0.32,
             f'Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}',
             fontsize=11, ha='center', color='#888888')

    # Footer line
    fig.patches.append(plt.Rectangle((0.1, 0.26), 0.8, 0.002,
                       transform=fig.transFigure, facecolor='#DDDDDD', zorder=0))

    # Training config summary
    config_text = (
        "Training Configuration\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "• Loss: CrossEntropy (label smoothing)\n"
        "• Weighted sampling for class balance\n"
        "• Resumed from epoch 23 checkpoint\n"
        "• Image size: 224 × 224"
    )
    fig.text(0.5, 0.15, config_text, fontsize=10, ha='center', va='center',
             color='#555555', family='monospace', linespacing=1.6)

    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def create_accuracy_curve(pdf):
    """Training & Validation accuracy over epochs."""
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(C_BG)

    ax.plot(epochs, train_acc, color=C_PRIMARY, linewidth=2.2, marker='o',
            markersize=4, label='Train Accuracy', zorder=3)
    ax.plot(epochs, val_acc, color=C_SECONDARY, linewidth=2.2, marker='s',
            markersize=4, label='Val Accuracy', zorder=3)

    # Highlight best epoch
    ax.axvline(x=BEST_EPOCH, color=C_ACCENT, linestyle='--', alpha=0.7, linewidth=1.5)
    ax.annotate(f'Best: {BEST_VAL_ACC}%\n(Epoch {BEST_EPOCH})',
                xy=(BEST_EPOCH, BEST_VAL_ACC),
                xytext=(BEST_EPOCH + 2, BEST_VAL_ACC - 1.5),
                fontsize=10, fontweight='bold', color=C_ACCENT,
                arrowprops=dict(arrowstyle='->', color=C_ACCENT, lw=1.5))

    ax.fill_between(epochs, val_acc, alpha=0.08, color=C_SECONDARY)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Training & Validation Accuracy', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(94, 101)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def create_loss_curve(pdf):
    """Training & Validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(C_BG)

    ax.plot(epochs, train_loss, color=C_PRIMARY, linewidth=2.2, marker='o',
            markersize=4, label='Train Loss', zorder=3)
    ax.plot(epochs, val_loss, color=C_SECONDARY, linewidth=2.2, marker='s',
            markersize=4, label='Val Loss', zorder=3)

    ax.fill_between(epochs, train_loss, val_loss, alpha=0.08, color='#9E9E9E',
                    label='Gap (generalization)')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training & Validation Loss', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def create_metrics_table(pdf):
    """Epoch-by-epoch metrics in a clean table."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.axis('off')

    fig.text(0.5, 0.96, 'Epoch-by-Epoch Training Metrics',
             fontsize=16, ha='center', fontweight='bold')

    # Build table
    col_labels = ['Epoch', 'Train Loss', 'Train Acc (%)', 'Val Loss', 'Val Acc (%)']
    cell_data = []
    for d in EPOCH_DATA:
        ep, tl, ta, vl, va = d
        row = [str(ep), f'{tl:.4f}', f'{ta:.2f}', f'{vl:.4f}', f'{va:.2f}']
        cell_data.append(row)

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     cellLoc='center', loc='upper center',
                     bbox=[0.05, 0.02, 0.90, 0.90])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor(C_PRIMARY)
        table[(0, j)].set_text_props(fontweight='bold', color='white', fontsize=10)

    # Highlight best epoch row & alternate row colors
    best_row_idx = None
    for i, d in enumerate(EPOCH_DATA):
        row_idx = i + 1  # +1 for header
        if d[0] == BEST_EPOCH:
            best_row_idx = row_idx
        # Alternate row shading
        bg = '#F5F5F5' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[(row_idx, j)].set_facecolor(bg)

    # Highlight the best epoch row in green
    if best_row_idx:
        for j in range(len(col_labels)):
            table[(best_row_idx, j)].set_facecolor('#C8E6C9')
            table[(best_row_idx, j)].set_text_props(fontweight='bold')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def create_summary_page(pdf):
    """Final summary page with key observations."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')

    # Header bar
    fig.patches.append(plt.Rectangle((0, 0.88), 1, 0.12,
                       transform=fig.transFigure, facecolor=C_PRIMARY, zorder=0))
    fig.text(0.5, 0.93, 'Training Summary & Observations',
             fontsize=22, ha='center', va='center', fontweight='bold', color='white')

    # Metrics highlight boxes
    metrics_data = [
        ('Final Train Acc',  f'{train_acc[-1]:.2f}%'),
        ('Final Val Acc',    f'{val_acc[-1]:.2f}%'),
        ('Best Val Acc',     f'{BEST_VAL_ACC:.2f}%'),
        ('Final Train Loss', f'{train_loss[-1]:.4f}'),
        ('Final Val Loss',   f'{val_loss[-1]:.4f}'),
        ('Best Epoch',       f'{BEST_EPOCH}'),
    ]

    cols = 3
    rows_m = 2
    card_w = 0.24
    card_h = 0.08
    gap_x = 0.04
    gap_y = 0.03
    start_x = 0.5 - (cols * card_w + (cols-1) * gap_x) / 2
    start_y = 0.72

    for idx, (label, value) in enumerate(metrics_data):
        r = idx // cols
        c = idx % cols
        x = start_x + c * (card_w + gap_x)
        y = start_y - r * (card_h + gap_y)
        fig.patches.append(plt.Rectangle((x, y), card_w, card_h,
                           transform=fig.transFigure, facecolor='#E3F2FD',
                           edgecolor=C_PRIMARY, linewidth=1.5, zorder=0))
        fig.text(x + card_w/2, y + card_h * 0.65, value,
                 fontsize=16, ha='center', fontweight='bold', color=C_PRIMARY)
        fig.text(x + card_w/2, y + card_h * 0.2, label,
                 fontsize=9, ha='center', color='#666666')

    # Observations
    observations = [
        "✓  Model converged to 100% training accuracy by epoch 26, indicating strong learning capacity.",
        "✓  Validation accuracy plateaued around 98.9%, with a peak of 99.05% at epoch 42.",
        "✓  The train–val gap remains very small (≈0.03 loss), suggesting minimal overfitting.",
        "✓  CosineAnnealing LR schedule smoothly reduced the learning rate for stable convergence.",
        "✓  WeightedRandomSampler effectively handled the class imbalance across F0–F4 stages.",
        "✓  Checkpoint resumption from epoch 23 worked seamlessly, preserving optimizer and scheduler state.",
    ]

    obs_y = 0.52
    fig.text(0.08, obs_y + 0.03, 'Key Observations', fontsize=14,
             fontweight='bold', color='#333333')
    for i, obs in enumerate(observations):
        fig.text(0.10, obs_y - i * 0.045, obs, fontsize=10, color='#444444')

    # Footer
    fig.patches.append(plt.Rectangle((0.1, 0.12), 0.8, 0.001,
                       transform=fig.transFigure, facecolor='#CCCCCC', zorder=0))
    fig.text(0.5, 0.08,
             'Liver Fibrosis Staging — Automated Liver Staging (ALS) Pipeline',
             fontsize=10, ha='center', color='#999999')
    fig.text(0.5, 0.05,
             f'Report generated on {datetime.now().strftime("%B %d, %Y")}',
             fontsize=9, ha='center', color='#AAAAAA')

    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def main():
    print("\n" + "=" * 60)
    print("  GENERATING ConvNeXt V2 TRAINING REPORT (PDF)")
    print("=" * 60)

    PDF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(PDF_OUTPUT)) as pdf:
        print("\n  Building pages...")

        create_title_page(pdf)
        print("    ✓ Title page")

        create_accuracy_curve(pdf)
        print("    ✓ Accuracy curve")

        create_loss_curve(pdf)
        print("    ✓ Loss curve")

        create_metrics_table(pdf)
        print("    ✓ Epoch metrics table")

        create_summary_page(pdf)
        print("    ✓ Summary & observations")

    print(f"\n{'=' * 60}")
    print(f"  PDF saved → {PDF_OUTPUT}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
