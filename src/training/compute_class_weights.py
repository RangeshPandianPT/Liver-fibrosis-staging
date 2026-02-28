import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
# Fix path: src/training -> project_root
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import OUTPUT_DIR, CLASS_NAMES

def compute_weights():
    manifest_path = OUTPUT_DIR / "dataset_manifest.csv"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        return

    print(f"Loading manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path)
    
    # Filter for training set only, as we only balance training data
    train_df = df[df['assigned_split'] == 'Train']
    print(f"Found {len(train_df)} training samples.")

    # Count classes
    class_counts = train_df['y_true'].value_counts().sort_index()
    
    total_samples = len(train_df)
    num_classes = len(CLASS_NAMES)
    
    print("\nClass Counts (Training):")
    weights = {}
    for class_name in CLASS_NAMES:
        count = class_counts.get(class_name, 0)
        # Weight formula: N_total / (N_classes * N_class)
        weight = total_samples / (num_classes * count) if count > 0 else 0
        weights[class_name] = weight
        print(f"  {class_name}: {count} samples -> Weight: {weight:.4f}")
        
    print("\nCopy these weights for WeightedRandomSampler:")
    weight_list = [weights[c] for c in CLASS_NAMES]
    print(f"Weights List: {weight_list}")
    
    return weights

if __name__ == "__main__":
    compute_weights()
