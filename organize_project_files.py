import os
import shutil
import glob
from pathlib import Path

# Define paths
BASE_DIR = Path(r"d:\ALS")
OUTPUTS_DIR = BASE_DIR / "outputs"
DEST_DIR = BASE_DIR / "Research_Materials"

# Define sub-directories structure
PDF_DIR = DEST_DIR / "PDFs"
CSV_DIR = DEST_DIR / "Data_CSVs"

# Create directories
for d in [PDF_DIR, CSV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Helper function to copy and organize
def organize_files():
    print(f"Organizing files from {OUTPUTS_DIR} into {DEST_DIR}...")
    
    # --- PDF Organization ---
    pdf_files = list(OUTPUTS_DIR.glob("*.pdf")) + list(BASE_DIR.glob("*.pdf"))
    
    for pdf in pdf_files:
        if "confusion" in pdf.name.lower() or "roc" in pdf.name.lower() or "matrix" in pdf.name.lower():
            sub_folder = PDF_DIR / "Plots_and_Figures"
        elif "report" in pdf.name.lower() or "abstract" in pdf.name.lower():
            sub_folder = PDF_DIR / "Reports"
        else:
            sub_folder = PDF_DIR / "Misc"
            
        sub_folder.mkdir(exist_ok=True)
        
        try:
            shutil.copy2(pdf, sub_folder / pdf.name)
            print(f"Computed PDF: {pdf.name} -> {sub_folder}")
        except Exception as e:
            print(f"Error copying {pdf.name}: {e}")

    # --- CSV Organization ---
    csv_files = list(OUTPUTS_DIR.glob("*.csv")) + list(BASE_DIR.glob("*.csv"))
    
    for csv in csv_files:
        if "manifest" in csv.name.lower():
            sub_folder = CSV_DIR / "Dataset_Manifests"
        elif "prediction" in csv.name.lower() or "results" in csv.name.lower():
            sub_folder = CSV_DIR / "Model_Predictions"
        elif "history" in csv.name.lower() or "log" in csv.name.lower():
            sub_folder = CSV_DIR / "Training_Logs"
        else:
            sub_folder = CSV_DIR / "Misc_Data"
            
        sub_folder.mkdir(exist_ok=True)
        
        try:
            shutil.copy2(csv, sub_folder / csv.name)
            print(f"Computed CSV: {csv.name} -> {sub_folder}")
        except Exception as e:
            print(f"Error copying {csv.name}: {e}")

    print("\nOrganization Complete!")
    print(f"Files have been COPIED to {DEST_DIR}")
    print("Original files were left intact to ensure code continuity.")

if __name__ == "__main__":
    organize_files()
