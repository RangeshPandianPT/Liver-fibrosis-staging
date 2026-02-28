# Automated Liver Staging (ALS) Project

## Project Structure

```
ALS/
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── training/          # Training scripts
│   ├── inference/         # Prediction scripts
│   ├── evaluation/        # Evaluation scripts
│   └── utils/             # Utility scripts
├── scripts/               # Pipeline orchestration
├── web_app/              # Streamlit demo
├── docs/                 # Documentation
├── outputs/              # Training outputs & checkpoints
├── data/                 # Dataset
└── requirements.txt      # Dependencies

## Quick Start

### 1. Training
```bash
python src/training/train_universal.py --model convnext --epochs 50
```

### 2. Generate Predictions
```bash
python src/inference/generate_universal_predictions.py --model convnext
```

### 3. Run Full Pipeline
```bash
python scripts/run_full_pipeline.py
```

### 4. Launch Demo
```bash
streamlit run web_app/app.py
```

## Model Performance

| Model | Accuracy | Cohen's Kappa |
|-------|----------|---------------|
| **Ensemble** | **98.26%** | **0.9938** |
| ConvNeXt | 98.42% | 0.9793 |
| ViT-B/16 | 97.47% | 0.9600 |
| EfficientNet-V2 | 96.60% | 0.9500 |
| ResNet50 | 91.30% | 0.8900 |
| DeiT-Small | 85.53% | 0.8200 |
