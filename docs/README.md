# Medical CV Pipeline for Liver Fibrosis Staging

This project implements a state-of-the-art Deep Learning Ensemble Pipeline for automated Liver Fibrosis Staging (F0-F4) using ResNet50, EfficientNet-V2, and Vision Transformer (ViT) models. It features a fully automated workflow from raw images to comprehensive PDF reports.

## 🚀 Key Features

*   **Multi-Model Ensemble**: Combines predictions from ResNet50, EfficientNet-V2, and ViT (Vision Transformer) using a weighted Soft-Voting mechanism.
*   **Automated Pipeline**: A single script (`run_full_pipeline.py`) handles prediction generation, ensemble analysis, and report generation.
*   **Explainable AI (XAI)**: Generates Grad-CAM heatmaps for ViT and CNNs to visualize model attention.
*   **Comprehensive Reporting**:
    *   **PDF Reports**: Auto-generated PDF reports summarizing model performance, confusion matrices, and key metrics.
    *   **Detailed Metrics**: Calculates Accuracy, F1-Score (Macro), and Quadratic Weighted Kappa (QWK) with 95% Confidence Intervals.
    *   **Visualizations**: Confusion matrices, fold comparison charts, and per-class accuracy plots.
*   **Robust Training**: Supports Stratified K-Fold Cross-Validation for reliable performance estimation.
*   **Data Handling**: Automated preprocessing (CLAHE, Resizing) and class imbalance handling using Weighted Loss/SMOTE strategies.

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## ⚡ Usage

### 1. Run the Full Pipeline (Recommended)
The easiest way to run the inference and reporting workflow:
```bash
python run_full_pipeline.py
```
This script will sequentially:
1.  Generate predictions from all models (ResNet50, EfficientNet, ViT).
2.  Run the Ensemble Pathologist (Soft-Voting).
3.  Generate a consolidated PDF Report in `outputs/`.

### 2. Individual Modules
You can also run specific parts of the pipeline independently:

**Ensemble Analysis:**
```bash
python run_ensemble_pathologist.py
```

**Generate ViT Heatmaps:**
```bash
python run_vit_specialist_tasks.py
```

### 3. Training
To train the models from scratch:

**K-Fold Cross-Validation (Full Evaluation):**
```bash
python train_kfold.py --folds 5 --epochs 50
```

**Train Individual Models:**
```bash
python train_cnn_models.py --model resnet50 --epochs 30
python train_vit_light.py --epochs 30
```

## 📂 Project Structure

```
├── run_full_pipeline.py        # 🚀 Main entry point for the full pipeline
├── run_ensemble_pathologist.py # Ensemble voting and analysis logic
├── requirements.txt            # Project dependencies
├── build_ensemble.py           # (Internal) Helper for ensemble logic
├── config.py                   # Configuration settings (paths, params)
├── services/                   # Service modules for core logic
├── data/                       # Dataset directory (F0-F4)
├── src/                        # Source code for models and training
│   ├── models/                 # Model definitions (ResNet, EfficientNet, ViT)
│   ├── training.py             # Training loops and utilities
│   ├── validation.py           # Metrics and validation logic
│   └── gradcam.py              # XAI visualization tools
├── outputs/                    # 📊 All generated outputs
│   ├── checkpoints/            # Saved model weights (.pth)
│   ├── metrics/                # JSON metrics and intermediate files
│   ├── gradcam_heatmaps/       # Generated attention heatmaps
│   └── final_analysis/         # Final ensemble results and charts
└── reports/                    # Scripts for generating PDF reports
```

## 📊 Outputs & Reports
The pipeline generates publication-ready outputs in the `outputs/` directory:
*   `ensemble_analysis_report.pdf`: A complete summary of the pipeline's performance.
*   `ensemble_confusion_matrix.png`: Visual confusion matrix.
*   `ensemble_results.csv`: Detailed CSV with individual and ensemble predictions for every sample.
*   `vit_heatmaps/`: Directory containing Grad-CAM visualizations.

## 🧪 Evaluation Metrics
The pipeline evaluates using:
*   **Accuracy**: Overall correctness.
*   **Quadratic Weighted Kappa (QWK)**: Measures agreement for ordinal staging (F0-F4).
*   **Macro F1-Score**: Balanced metric for multi-class performance.
*   **Confusion Matrix**: Detailed breakdown of misclassifications.
