# Automated Liver Staging (ALS) - Project Structure

## 📁 Directory Organization

```
ALS/
├── src/                          # Source code
│   ├── models/                   # Model architectures (ResNet, ConvNeXt, DeiT)
│   ├── training/                 # Training scripts
│   │   ├── train_universal.py   # Universal trainer (ConvNeXt, etc.)
│   │   ├── compute_class_weights.py
│   │   ├── train_cnn_models.py
│   │   └── train_deit.py
│   ├── inference/                # Prediction generation
│   │   ├── generate_universal_predictions.py
│   │   ├── generate_cnn_predictions.py
│   │   └── generate_deit_predictions.py
│   ├── evaluation/               # Model evaluation
│   │   └── evaluate_cnn_models.py
│   └── utils/                    # Utilities
│       ├── prepare_dataset.py
│       └── download_weights.py
├── scripts/                      # Pipeline orchestration
│   ├── run_full_pipeline.py     # Main pipeline
│   └── run_ensemble_pathologist.py
├── web_app/                      # Streamlit demo
│   └── app.py                    # Live demo application
├── docs/                         # Documentation
├── outputs/                      # Training outputs & checkpoints
├── data/                         # Dataset
├── report_scripts/               # PDF report generators
└── config.py                     # Global configuration

```

## 🚀 Quick Start

### Training a Model
```bash
# Train ConvNeXt
python src/training/train_universal.py --model convnext --epochs 50

# Train DeiT
python src/training/train_deit.py --epochs 100
```

### Generate Predictions
```bash
# ConvNeXt predictions
python src/inference/generate_universal_predictions.py --model convnext

# CNN predictions (ResNet)
python src/inference/generate_cnn_predictions.py
```

### Run Full Pipeline
```bash
python scripts/run_full_pipeline.py
```

### Launch Live Demo
```bash
streamlit run web_app/app.py
```

## 📊 Model Performance

| Model | Accuracy | Cohen's Kappa |
|-------|----------|---------------|
| **Ensemble (All Models)** | **98.26%** | **0.9938** |
| ConvNeXt Tiny | 98.42% | 0.9793 |
| ResNet50 | 91.30% | 0.8900 |
| DeiT-Small | 85.53% | 0.8200 |

## 🔬 Research Highlights

- **Best Individual Model**: ConvNeXt at 98.42%
- **Best Ensemble**: 98.26% with QWK of 0.9938 (near-perfect agreement)
- **Class Balancing**: WeightedRandomSampler for handling imbalanced data
- **5-Stage Classification**: F0, F1, F2, F3, F4 (liver fibrosis stages)
- **Test Set**: 1,265 samples

## 📦 Dependencies

```bash
pip install -r requirements.txt
```

## 🎯 Project Status

✅ All models trained and integrated  
✅ Ensemble pipeline operational  
✅ Live demo functional  
✅ Codebase organized and documented  
✅ Ready for research presentation
