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
DEIT_SCRIPT = BASE_DIR / "generate_deit_predictions.py"
ENSEMBLE_SCRIPT = BASE_DIR / "run_ensemble_pathologist.py"
REPORT_SCRIPT = BASE_DIR / "reports" / "generate_ensemble_report.py"
COMPARATIVE_REPORT_SCRIPT = BASE_DIR / "report_scripts" / "generate_comparative_report_pdf.py"

# Checkpoints
CHECKPOINTS = [
    BASE_DIR / "outputs" / "checkpoints" / "best_resnet_model.pth",
    BASE_DIR / "outputs" / "checkpoints" / "best_effnet_model.pth",
    BASE_DIR / "outputs" / "checkpoints" / "best_effnet_model.pth",
    BASE_DIR / "outputs" / "vit_light" / "best_vit_model.pth",
    BASE_DIR / "outputs" / "deit_small" / "best_deit_model.pth"
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
       # Step 1: Generate CNN Predictions (ResNet50 & EfficientNet-V2)
    CNN_SCRIPT = BASE_DIR / "src/inference/generate_cnn_predictions.py"
    if not run_step(CNN_SCRIPT, "Generate CNN Predictions"):
        sys.exit(1)

    # Step 2: Generate ViT Predictions (ViT-B/16)
    # Note: Assuming generate_vit_predictions is implemented or covered by universal?
    # Original pipeline had separate script. Let's check file list.
    # Ah, 'generate_deit_predictions.py' exists. 'evaluate_vit_model.py' exists.
    # Let's see what was there before.
    
    # User had:
    # CNN_SCRIPT = BASE_DIR / "generate_cnn_predictions.py"
    # VIT_SCRIPT = BASE_DIR / "run_vit_specialist_tasks.py" (Maybe?)
    # DEIT_SCRIPT = BASE_DIR / "generate_deit_predictions.py"
    
    # Wait, I only moved `generate_cnn` and `generate_universal`.
    # I should check if I missed moving others or if I should point to old ones if not moved.
    # checking file list... `generate_deit_predictions.py` is still in root.
    # `run_vit_specialist_tasks.py` is in root.
    
    # I'll update CNN and Universal paths.
    
    start_total = time.time()
    
    # Step 1: CNN Predictions
    # if not run_step(CNN_SCRIPT, "Generate CNN Predictions"): # Replaced by the above
    #     sys.exit(1)
        
    # Step 2: ViT Predictions
    if not run_step(VIT_SCRIPT, "Generate ViT Predictions"):
        sys.exit(1)
        
    # Step 2.5: DeiT Predictions
    DEIT_SCRIPT = BASE_DIR / "generate_deit_predictions.py"
    if not run_step(DEIT_SCRIPT, "Generate DeiT Predictions"):
        print("Warning: DeiT prediction failed. Continuing without it...")
        # We don't exit here, to allow partial runs if DeiT is not ready yet
        pass
        
    # Step 2.6: ConvNeXt Predictions (Universal Script)
    UNIVERSAL_SCRIPT = BASE_DIR / "src/inference/generate_universal_predictions.py"
    # We call it as a subprocess with arguments
    print(f"\n{'-'*60}")
    print(f"STEP: Generate ConvNeXt Predictions")
    print(f"Running {UNIVERSAL_SCRIPT.name} --model convnext...")
    print(f"{'-'*60}\n")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, str(UNIVERSAL_SCRIPT), "--model", "convnext"], cwd=str(BASE_DIR))
    
    if result.returncode != 0:
        print(f"Warning: ConvNeXt predictions failed. Continuing without it.")
    else:
        print(f"ConvNeXt predictions completed in {time.time() - start_time:.2f}s.")

    # Step 3: Ensemble Analysis
    if not run_step(ENSEMBLE_SCRIPT, "Run Ensemble Analysis"):
        sys.exit(1)
        
    # Step 4: PDF Report
    if not run_step(REPORT_SCRIPT, "Generate PDF Report"):
        sys.exit(1)

    # Step 5: Comparative Research Report
    if not run_step(COMPARATIVE_REPORT_SCRIPT, "Generate Comparative Research Report"):
        sys.exit(1)
        
    total_time = time.time() - start_total
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f}s 🚀")
    print(f"Outputs available in: {BASE_DIR / 'outputs'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
