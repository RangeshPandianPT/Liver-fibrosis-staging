"""
Generate Universal Predictions.
Loads models trained with train_universal.py and generates predictions for the Test split.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from PIL import Image

# Add project root to path
# Fix path: src/inference -> project_root
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import OUTPUT_DIR, DEVICE, CLASS_NAMES, NUM_CLASSES, DATA_DIR
# Import get_model and get_transforms from train_universal (now in src.training)
try:
    from src.training.train_universal import get_model, get_transforms
except ImportError:
    # Fallback if running as script where train_universal is not in path conceptually
    # but we added sys.path so it should be fine.
    # Recopying minimal factory if needed would be safer for standalone, 
    # but let's try direct import first.
    pass

# Move TestDataset to module level to avoid pickling issues with multiprocessing
class TestDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.data = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy tensor or handle error
            image = torch.zeros((3, 224, 224))
        
        return image, row['image_path'] # Return path/filename to ID samples

def get_test_dataset(model_name):
    # Load Manifest
    manifest_path = OUTPUT_DIR / "dataset_manifest.csv"
    df = pd.read_csv(manifest_path)
    test_df = df[df['assigned_split'] == 'Test'].reset_index(drop=True)
    
    # Transforms
    _, val_transform = get_transforms(model_name)
    
    return TestDataset(test_df, val_transform)

def run_inference(model, dataloader):
    probs_list = []
    paths_list = []
    
    model.eval()
    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Inference"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            probs_list.extend(probs)
            paths_list.extend(paths)
            
    return probs_list, paths_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name (convnext, etc)')
    args = parser.parse_args()
    
    print(f"Generating predictions for {args.model}...")
    
    # 1. Load Model
    model = get_model(args.model, NUM_CLASSES, pretrained=False)
    
    # 2. Load Checkpoint
    # Expects: outputs/{model}/best_{model}_model.pth
    ckpt_path = OUTPUT_DIR / args.model / f"best_{args.model}_model.pth"
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)
        
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    
    # 3. Data Loader
    dataset = get_test_dataset(args.model)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 4. Inference
    probs, paths = run_inference(model, dataloader)
    
    # 5. Save Results
    # Format: filename, true_label (optional, but good for verify), class_probs...
    # We will just save filename + probs to merge later or use directly.
    
    df = pd.DataFrame()
    df['image_path'] = paths
    
    probs = np.array(probs)
    for i, class_name in enumerate(CLASS_NAMES):
        df[f'{args.model}_{class_name}_prob'] = probs[:, i]
        
    out_csv = OUTPUT_DIR / args.model / f"{args.model}_predictions.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

if __name__ == "__main__":
    main()
