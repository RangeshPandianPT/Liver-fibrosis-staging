"""
Comprehensive Project Cleanup Script
Organizes all files into a clean, professional structure.
"""
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Define the target structure
MOVES = {
    # Training scripts -> src/training/
    "train_cnn_models.py": "src/training/train_cnn_models.py",
    "train_deit.py": "src/training/train_deit.py",
    "train_vit_light.py": "src/training/train_vit_light.py",
    "train_kfold.py": "src/training/train_kfold.py",
    "train.py": "src/training/train_legacy.py",
    
    # Evaluation scripts -> src/evaluation/
    "evaluate.py": "src/evaluation/evaluate.py",
    "evaluate_cnn_models.py": "src/evaluation/evaluate_cnn_models.py",
    "evaluate_vit_model.py": "src/evaluation/evaluate_vit_model.py",
    "calculate_deit_metrics.py": "src/evaluation/calculate_deit_metrics.py",
    
    # Inference scripts -> src/inference/
    "generate_deit_predictions.py": "src/inference/generate_deit_predictions.py",
    "run_vit_specialist_tasks.py": "src/inference/run_vit_specialist_tasks.py",
    
    # Utility scripts -> src/utils/
    "prepare_dataset.py": "src/utils/prepare_dataset.py",
    "download_weights.py": "src/utils/download_weights.py",
    "test_timm.py": "src/utils/test_timm.py",
    
    # Pipeline scripts -> scripts/
    "run_full_pipeline.py": "scripts/run_full_pipeline.py",
    "run_ensemble_pathologist.py": "scripts/run_ensemble_pathologist.py",
    
    # Documentation -> docs/
    "README.md": "docs/README.md",
    "research_day_abstract.md": "docs/research_day_abstract.md",
    "model_accuracies.txt": "docs/model_accuracies.txt",
    
    # Old organize script -> archive/
    "organize_project_files.py": "archive/organize_project_files.py",
}

def create_directories():
    """Create all necessary directories."""
    dirs = [
        "src/training",
        "src/evaluation", 
        "src/inference",
        "src/utils",
        "scripts",
        "docs",
        "archive"
    ]
    
    for dir_path in dirs:
        (BASE_DIR / dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}/")

def move_files():
    """Move files to their new locations."""
    moved_count = 0
    skipped_count = 0
    
    for src, dest in MOVES.items():
        src_path = BASE_DIR / src
        dest_path = BASE_DIR / dest
        
        if src_path.exists():
            if dest_path.exists():
                print(f"⚠ Skipped {src} (destination exists)")
                skipped_count += 1
            else:
                shutil.move(str(src_path), str(dest_path))
                print(f"✓ Moved {src} -> {dest}")
                moved_count += 1
        else:
            print(f"⚠ Skipped {src} (not found)")
            skipped_count += 1
    
    return moved_count, skipped_count

def create_readme():
    """Create a comprehensive README in docs/."""
    readme_content = """# Automated Liver Staging (ALS) Project

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
"""
    
    readme_path = BASE_DIR / "docs" / "PROJECT_STRUCTURE.md"
    readme_path.write_text(readme_content, encoding='utf-8')
    print(f"✓ Created docs/PROJECT_STRUCTURE.md")

def main():
    print("=" * 60)
    print("PROJECT CLEANUP & ORGANIZATION")
    print("=" * 60)
    
    print("\n1. Creating directory structure...")
    create_directories()
    
    print("\n2. Moving files...")
    moved, skipped = move_files()
    
    print("\n3. Creating documentation...")
    create_readme()
    
    print("\n" + "=" * 60)
    print(f"✅ CLEANUP COMPLETE")
    print(f"   Moved: {moved} files")
    print(f"   Skipped: {skipped} files")
    print("=" * 60)
    
    print("\n📁 New Structure:")
    print("   src/training/    - Training scripts")
    print("   src/inference/   - Prediction scripts")
    print("   src/evaluation/  - Evaluation scripts")
    print("   src/utils/       - Utility scripts")
    print("   scripts/         - Pipeline scripts")
    print("   web_app/         - Streamlit demo")
    print("   docs/            - Documentation")

if __name__ == "__main__":
    main()
