"""
ViT Specialist Task: Inference CSV and Grad-CAM Visualization.

1. Generates 'vit_predictions.csv' with probabilities for the Test split.
2. Generates Grad-CAM heatmaps for 10 sample images (2 per class).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ViT_B_16_Weights
from PIL import Image
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
VIT_CHECKPOINT = OUTPUT_DIR / "vit_light" / "best_vit_model.pth"
MANIFEST_PATH = OUTPUT_DIR / "dataset_manifest.csv"
HEATMAP_DIR = OUTPUT_DIR / "gradcam_heatmaps" / "vit_specialist"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']

# --- Model Definition (Must match training) ---
class LightViTModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.backbone = models.vit_b_16(weights=weights)
        else:
            self.backbone = models.vit_b_16(weights=None)
        
        num_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# --- Dataset with Filename Support ---
class TestDatasetWithNames(Dataset):
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
        return image, label, filename, img_path

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Reshape Transform for ViT Grad-CAM ---
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)

def main():
    print("Initializing ViT Specialist Task...")
    
    # 1. Load Model
    model = LightViTModel(pretrained=False)
    checkpoint = torch.load(VIT_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded.")

    # 2. Prepare Dataset
    dataset = TestDatasetWithNames(MANIFEST_PATH, transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 3. Validation / CSV Generation Loop
    results = {
        'filename': [],
        'true_label': [],
        'vit_f0_prob': [],
        'vit_f1_prob': [],
        'vit_f2_prob': [],
        'vit_f3_prob': [],
        'vit_f4_prob': []
    }
    
    print("Running Inference...")
    with torch.no_grad():
        for images, labels, filenames, _ in tqdm(dataloader):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            results['filename'].extend(filenames)
            results['true_label'].extend([CLASS_NAMES[l] for l in labels])
            
            for i in range(5):
                results[f'vit_f{i}_prob'].extend(probs[:, i])
    
    # Save CSV
    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / 'vit_predictions.csv'
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")
    
    # 4. Grad-CAM Generation
    print("Generating Grad-CAM Heatmaps...")
    
    # Select 2 images per class
    target_layer = model.backbone.encoder.layers[-1].ln_1
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    
    # Filter dataset for samples (reuse DataFrame to find indices/paths)
    # We'll re-open images to get clean visualizations (not normalized)
    
    samples_found = {c: 0 for c in CLASS_NAMES}
    samples_to_process = []
    
    # Iterate through dataset df to find paths
    df_data = dataset.data
    for idx, row in df_data.iterrows():
        label_name = row['y_true']
        if samples_found[label_name] < 2:
            samples_to_process.append((row['image_path'], label_name))
            samples_found[label_name] += 1
        
        if all(v >= 2 for v in samples_found.values()):
            break
            
    print(f"Selected {len(samples_to_process)} images for visualization.")
    
    for img_path, true_label in samples_to_process:
        # Prepare image
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = get_transforms()(img_pil).unsqueeze(0).to(DEVICE)
        
        # Run Grad-CAM
        grayscale_cam = cam(input_tensor=img_tensor)
        grayscale_cam = grayscale_cam[0, :]
        
        # Prepare for visualization
        img_np = np.array(img_pil.resize((224, 224)))
        img_float = np.float32(img_np) / 255.0
        
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        
        # Save
        filename = os.path.basename(img_path)
        save_path = HEATMAP_DIR / f"{true_label}_{filename}_gradcam.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"Saved heatmap: {save_path}")

    print("ViT Specialist Task Completed.")

if __name__ == "__main__":
    main()
