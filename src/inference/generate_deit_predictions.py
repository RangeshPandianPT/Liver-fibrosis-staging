"""
Generate predictions using DeiT-Small for Ensemble.
"""
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import sys
from config import (
    DATA_DIR, OUTPUT_DIR, DEVICE, CLASS_NAMES, BATCH_SIZE
)
from src.models.deit_branch import DeiTBranch

# Paths
MODEL_PATH = OUTPUT_DIR / "deit_small" / "best_deit_model.pth"
OUTPUT_CSV = OUTPUT_DIR / "deit_predictions.csv"

def get_transforms():
    """Val transforms for DeiT."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

class InferenceDataset(torch.utils.data.Dataset):
    """Dataset for inference."""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.transform = get_transforms()
        self.samples = []
        
        for idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    self.samples.append((str(img_path), idx, img_path.name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, filename = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label, filename

def main():
    print("\n" + "=" * 60)
    print("🔮 GENERATING DeiT PREDICTIONS")
    print("=" * 60)
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train_deit.py first.")
        return

    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = DeiTBranch(num_classes=len(CLASS_NAMES), pretrained=False)
    
    # Checkpoint loading
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # Check if state_dict is directly the dict or nested
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(DEVICE)
    model.eval()
    
    # Data Loader
    dataset = InferenceDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Found {len(dataset)} images.")
    
    results = []
    
    with torch.no_grad():
        for images, labels, filenames in tqdm(loader, desc="Inference"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            probs = probs.cpu().numpy()
            labels = labels.numpy()
            
            for i in range(len(filenames)):
                row = {
                    'filename': filenames[i],
                    'true_label': CLASS_NAMES[labels[i]]
                }
                for c_idx, c_name in enumerate(CLASS_NAMES):
                    row[f'deit_f{c_idx}_prob'] = probs[i][c_idx]
                
                results.append(row)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved DeiT predictions to {OUTPUT_CSV}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
