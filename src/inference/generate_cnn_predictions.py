"""
Generate CNN Predictions CSV.

Loads trained ResNet50 and EfficientNet-V2 models and generates
'cnn_predictions.csv' with class probabilities for the Test split.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path to import model definitions if needed
# Add src to path to import model definitions if needed
# Fix path: src/inference -> project_root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
MANIFEST_PATH = OUTPUT_DIR / "dataset_manifest.csv"
RESNET_CHECKPOINT = OUTPUT_DIR / "checkpoints" / "best_resnet_model.pth"
EFFNET_CHECKPOINT = OUTPUT_DIR / "checkpoints" / "best_effnet_model.pth"
CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']

# --- Model Definitions (Simplified for Inference) ---
# We define them here to avoid complex imports if src structure is tricky, 
# but relying on `src` imports is better if available. 
# let's try to import from src first, if fails, fallback or assume standard structure.
# Based on file list, src/models/resnet_branch.py exists.

from src.models.resnet_branch import ResNet50Branch
from src.models.efficientnet_branch import EfficientNetBranch

class TestDataset(Dataset):
    def __init__(self, manifest_path, transform=None):
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        df = pd.read_csv(manifest_path)
        self.data = df[df['assigned_split'] == 'Test'].reset_index(drop=True)
        print(f"Loaded {len(self.data)} Test samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[row['y_true']]
        filename = os.path.basename(img_path)
        return image, label, filename

def get_transforms():
    # Standard ImageNet normalization for CNNs
    return transforms.Compose([
        transforms.Resize((384, 384)), # Config usually says 384 for these
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def run_inference(model, dataloader):
    probs_list = []
    filenames_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader, desc="Inference"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            probs_list.extend(probs)
            filenames_list.extend(filenames)
            labels_list.extend(labels.numpy())
            
    return np.array(probs_list), filenames_list, np.array(labels_list)

def main():
    print("Generating CNN Predictions...")
    
    # Dataset
    dataset = TestDataset(MANIFEST_PATH, transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Prepare DataFrame
    results_df = pd.DataFrame()
    
    # --- ResNet50 ---
    print("\nRunning ResNet50...")
    resnet = ResNet50Branch(pretrained=False) # Architecture only
    try:
        ckpt = torch.load(RESNET_CHECKPOINT, map_location=DEVICE)
        resnet.load_state_dict(ckpt['model_state_dict'])
        resnet = resnet.to(DEVICE)
        
        probs, filenames, labels = run_inference(resnet, dataloader)
        
        # Populate DF keys from first run
        results_df['filename'] = filenames
        results_df['true_label'] = [CLASS_NAMES[l] for l in labels]
        
        for i in range(5):
            results_df[f'resnet_f{i}_prob'] = probs[:, i]
            
    except Exception as e:
        print(f"Error running ResNet: {e}")
        return

    # --- EfficientNet ---
    print("\nRunning EfficientNet-V2...")
    effnet = EfficientNetBranch(pretrained=False)
    try:
        ckpt = torch.load(EFFNET_CHECKPOINT, map_location=DEVICE)
        effnet.load_state_dict(ckpt['model_state_dict'])
        effnet = effnet.to(DEVICE)
        
        probs, _, _ = run_inference(effnet, dataloader) # filenames same order
        
        for i in range(5):
            results_df[f'effnet_f{i}_prob'] = probs[:, i]
            
    except Exception as e:
        print(f"Error running EfficientNet: {e}")
        return

    # Save
    save_path = OUTPUT_DIR / 'cnn_predictions.csv'
    results_df.to_csv(save_path, index=False)
    print(f"\nSaved CNN predictions to {save_path}")
    print(results_df.head())

if __name__ == "__main__":
    main()
