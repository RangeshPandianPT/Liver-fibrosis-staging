# Medical CV Pipeline for Liver Fibrosis Staging

This project implements a deep learning ensemble pipeline for automated liver fibrosis staging (F0-F4)
using ResNet50, EfficientNet-V2, and Vision Transformer models.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Standard Training (Train/Val Split)
```bash
python train.py --epochs 50 --batch_size 16
```

### K-Fold Cross-Validation (Recommended for Research Papers)
```bash
# Full training with 5-fold cross-validation
python train_kfold.py --folds 5 --epochs 50

# Quick test run
python train_kfold.py --folds 5 --epochs 1 --dry_run

# Save model checkpoints for each fold
python train_kfold.py --folds 5 --epochs 50 --save_all_folds
```

Cross-validation outputs:
- `outputs/metrics/cross_validation/cv_results.json` - Detailed results
- `outputs/metrics/cross_validation/cv_summary.txt` - Paper-ready summary
- `outputs/metrics/cross_validation/fold_comparison.png` - Visualization
- `outputs/metrics/cross_validation/per_class_accuracy.png` - Per-class chart

### Evaluation
```bash
python evaluate.py --checkpoint outputs/checkpoints/best_model.pth
```

### Generate Grad-CAM Heatmaps
```bash
python generate_heatmaps.py --checkpoint outputs/checkpoints/best_model.pth
```

## Project Structure

```
├── train.py              # Standard training script
├── train_kfold.py        # K-fold cross-validation training
├── evaluate.py           # Model evaluation
├── generate_heatmaps.py  # Grad-CAM visualization
├── config.py             # Configuration settings
├── src/
│   ├── preprocessing.py  # CLAHE and image transforms
│   ├── dataset.py        # PyTorch dataset and loaders
│   ├── models/           # ResNet50, EfficientNet, ViT, Ensemble
│   ├── training.py       # Training utilities
│   ├── validation.py     # Metrics and evaluation
│   ├── gradcam.py        # Grad-CAM heatmaps
│   └── cross_validation.py  # K-fold CV utilities
├── outputs/
│   ├── checkpoints/      # Model weights
│   ├── metrics/          # Evaluation results
│   └── gradcam_heatmaps/ # Explainability visualizations
└── data/liver_images/    # Input images (F0-F4 folders)
```

## Research Paper Ready

This pipeline produces paper-ready outputs including:
- **95% confidence intervals** for all metrics
- **Cohen's Kappa** (quadratic-weighted) for ordinal data
- **Per-class accuracy** with standard deviations
- **Visualization charts** suitable for publication
