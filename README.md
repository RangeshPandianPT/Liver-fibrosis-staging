# Automated Liver Staging (ALS) - Enterprise Clinical AI Platform & Open-Source Benchmark

An enterprise-grade, explainable AI platform and research codebase for automated liver fibrosis staging (F0–F4) using histological biopsy slides. Featuring multi-model soft-voting ensembles, ONNX hardware acceleration, FastAPI microservices, and clinical ambiguity alerts.

---

## 📁 Directory Organization

```
ALS/
├── .github/workflows/            # CI/CD pipelines (Pytest, Linting, Docker build)
├── src/                          # Source code
│   ├── api/                      # Production FastAPI REST service
│   │   ├── main.py               # Asynchronous REST endpoints
│   │   └── schemas.py            # Pydantic V2 validation & Swagger UI schemas
│   ├── inference/                # Prediction & hardware optimization
│   │   ├── onnx_engine.py        # ONNX export, INT8 quantization & latency benchmarking
│   │   ├── generate_universal_predictions.py
│   │   ├── generate_cnn_predictions.py
│   │   └── generate_deit_predictions.py
│   ├── models/                   # Model architectures (ResNet50, ConvNeXt V2, MedNeXt, DeiT)
│   ├── training/                 # Training & class balancing scripts
│   ├── evaluation/               # Model evaluation & metrics
│   ├── utils/                    # Dataset helpers
│   ├── cli.py                    # Unified CLI application (`liver-stage`)
│   ├── preprocessing.py          # CLAHE enhancement & flexible normalization
│   ├── xai.py                    # Advanced XAI (EigenCAM) & clinical uncertainty scoring
│   └── gradcam.py                # Legacy Grad-CAM heatmaps
├── tests/                        # Comprehensive automated test suite
│   ├── test_dataset.py           # Preprocessing & denormalization tests
│   ├── test_models.py            # Ensemble forward pass & ONNX export tests
│   ├── test_xai.py               # Shannon entropy & clinical margin tests
│   └── test_api.py               # FastAPI TestClient endpoint integration tests
├── scripts/                      # Pipeline orchestration scripts
├── web_app/                      # Clinical diagnostic Streamlit UI
├── docs/                         # Research paper drafts & documentation
├── outputs/                      # Training outputs, Grad-CAM heatmaps & checkpoints
├── data/                         # Histological slide dataset
├── pyproject.toml                # Python package distribution & CLI registry
├── requirements.txt              # Core dependencies
└── config.py                     # Global configuration
```

---

## 🚀 Quick Start & Enterprise CLI (`liver-stage`)

Install the package in editable mode to unlock the `liver-stage` CLI:
```bash
pip install -e .
```

### 1. Classify Histological Slides via CLI
Analyze a single slide or an entire folder with automated uncertainty analysis and ambiguity alerts:
```bash
liver-stage predict --image data/liver_images/sample.png --output outputs/report.json
```

### 2. Launch FastAPI Microservice Server
Start the production REST API with interactive Swagger documentation:
```bash
liver-stage serve --host 0.0.0.0 --port 8000
# Access interactive Swagger UI at http://localhost:8000/docs
```

### 3. Export & Optimize ONNX Models
Export PyTorch checkpoints to ONNX format with optional INT8 dynamic quantization for edge deployment:
```bash
# Export ensemble to ONNX with INT8 quantization
liver-stage export-onnx --branch ensemble --output outputs/checkpoints/ensemble.onnx --int8
```

### 4. Benchmark Hardware Acceleration
Compare execution latency (ms) and speedup ratio between native PyTorch and ONNX Runtime:
```bash
liver-stage benchmark --onnx-model outputs/checkpoints/ensemble.onnx --runs 50
```

---

## 🧪 Quality Assurance & Test Suite

Run the automated pytest suite with code coverage reporting:
```bash
pytest tests/ -v
```

All pushes and pull requests are automatically validated via **GitHub Actions** (`.github/workflows/ci.yml`), which executes Flake8/Ruff linting, cross-version Python testing (3.9–3.11), and Docker build verification.

---

## 🌐 REST API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | `GET` | System health, GPU acceleration status, and parameter counts |
| `/api/v1/predict` | `POST` | Classify slide image, compute Shannon entropy, and trigger clinical alerts |
| `/api/v1/batch-predict` | `POST` | Evaluate a cohort of slides and aggregate stage distribution statistics |
| `/api/v1/explain` | `POST` | Generate visual attention heatmaps (EigenCAM / Grad-CAM) as Base64 PNGs |

---

## 📊 Diagnostic Model Performance

| Model Architecture | Diagnostic Accuracy | Cohen's Kappa (QWK) |
|---|---|---|
| **Soft-Voting Ensemble (All Branches)** | **98.26%** | **0.9938** (Near-Perfect Agreement) |
| ConvNeXt V2 Tiny | 98.42% | 0.9793 |
| ResNet50 | 91.30% | 0.8900 |
| DeiT-Small | 85.53% | 0.8200 |

---

## 🔬 Explainable AI & Clinical Uncertainty

- **EigenCAM Attention Maps**: Robust, gradient-free visual explainability specifically tailored for Vision Transformers (DeiT/ViT) and CNNs.
- **Predictive Shannon Entropy**: Measures probability distribution diffusion ($-\sum p \log p$). Scores $> 0.65$ flag ambiguous classifications.
- **Confidence Margin Alerts**: Automatically triggers clinical consultation alerts when the probability margin between Top-1 and Top-2 predicted fibrosis stages falls below $0.15$.

---

## 🐳 Docker Deployment

Build and launch the complete containerized clinical platform:
```bash
docker compose up -d --build
```

---

## 🎯 Project Status

✅ Multi-model ensemble (ConvNeXt V2, MedNeXt, DeiT, ResNet50) integrated  
✅ Clinical Streamlit UI with 3-tab diagnostic report workflow  
✅ **Production FastAPI microservice layer with OpenAPI/Swagger UI**  
✅ **Automated test suite (pytest) & GitHub Actions CI/CD pipeline**  
✅ **ONNX Runtime export, INT8 quantization & latency benchmarking**  
✅ **Advanced XAI (EigenCAM) & clinical uncertainty ambiguity alerts**  
✅ **Unified command-line interface (`liver-stage` CLI package)**  
