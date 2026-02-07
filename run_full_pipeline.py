"""
Unified Pipeline Script for Liver Fibrosis Staging Ensemble.

This script automates the entire inference and reporting workflow:
1. Generates CNN Predictions (ResNet50 + EfficientNet-V2)
2. Generates ViT Predictions + Heatmaps
3. Runs Ensemble Analysis (Soft-Voting)
4. Generates PDF Report

Usage:
    python run_full_pipeline.py
"""
import subprocess
import sys
import time
from pathlib import Path
import os

# Paths
BASE_DIR = Path(__file__).parent
CNN_SCRIPT = BASE_DIR / "generate_cnn_predictions.py"
VIT_SCRIPT = BASE_DIR / "run_vit_specialist_tasks.py"
ENSEMBLE_SCRIPT = BASE_DIR / "run_ensemble_pathologist.py"
REPORT_SCRIPT = BASE_DIR / "reports" / "generate_ensemble_report.py"

# Checkpoints
CHECKPOINTS = [
    BASE_DIR / "outputs" / "checkpoints" / "best_resnet_model.pth",
    BASE_DIR / "outputs" / "checkpoints" / "best_effnet_model.pth",
    BASE_DIR / "outputs" / "vit_light" / "best_vit_model.pth"
]

def check_prerequisites():
    """Verify all necessary checkpoints exist."""
    print("Checking prerequisites...")
    missing = []
    for cp in CHECKPOINTS:
        if not cp.exists():
            missing.append(str(cp))
    
    if missing:
        print("Error: Missing checkpoints:")
        for m in missing:
            print(f"  - {m}")
        return False
    
    print("All checkpoints found.")
    return True

def run_step(script_path, step_name):
    """Run a single python script."""
    print(f"\n{'-'*60}")
    print(f"STEP: {step_name}")
    print(f"Running {script_path.name}...")
    print(f"{'-'*60}\n")
    
    start_time = time.time()
    
    # Run script
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))
    
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\nError: {step_name} failed with exit code {result.returncode}.")
        return False
    
    print(f"\n{step_name} completed successfully in {duration:.2f}s.")
    return True

def main():
    print("="*60)
    print("LIVER FIBROSIS STAGING - UNIFIED PIPELINE")
    print("="*60)
    
    if not check_prerequisites():
        sys.exit(1)
        
    start_total = time.time()
    
    # Step 1: CNN Predictions
    if not run_step(CNN_SCRIPT, "Generate CNN Predictions"):
        sys.exit(1)
        
    # Step 2: ViT Predictions
    if not run_step(VIT_SCRIPT, "Generate ViT Predictions"):
        sys.exit(1)
        
    # Step 3: Ensemble Analysis
    if not run_step(ENSEMBLE_SCRIPT, "Run Ensemble Analysis"):
        sys.exit(1)
        
    # Step 4: PDF Report
    if not run_step(REPORT_SCRIPT, "Generate PDF Report"):
        sys.exit(1)
        
    total_time = time.time() - start_total
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f}s 🚀")
    print(f"Outputs available in: {BASE_DIR / 'outputs'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
