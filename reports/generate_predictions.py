"""
Generate Model Predictions for Dataset Manifest

This script loads the trained ViT model and runs inference on all images
to add y_pred (predicted labels) to the dataset manifest.

Author: Medical Imaging Engineer
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Configuration
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "outputs" / "vit_light" / "best_vit_model.pth"
MANIFEST_PATH = BASE_DIR / "outputs" / "dataset_manifest.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "dataset_manifest_with_predictions.csv"

# Model settings
NUM_CLASSES = 5
CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']
IMAGE_SIZE = 224  # ViT light uses 224x224

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LightViTModel(nn.Module):
    """Lightweight ViT-B-16 model for liver fibrosis classification."""
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        
        # Load ViT-B-16 base model
        self.backbone = models.vit_b_16(weights=None)
        
        # Replace classification head
        num_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def get_inference_transforms():
    """Get transforms for inference (same as validation transforms)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model(model_path: Path, device: str) -> nn.Module:
    """Load the trained ViT model."""
    print(f"Loading model from: {model_path}")
    
    model = LightViTModel(num_classes=NUM_CLASSES, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model


def predict_single_image(model: nn.Module, image_path: str, 
                         transform, device: str) -> tuple:
    """
    Run inference on a single image.
    
    Returns:
        Tuple of (predicted_class_idx, predicted_class_name, confidence)
    """
    try:
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        return predicted_idx.item(), predicted_class, confidence.item()
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return -1, "ERROR", 0.0


def get_true_label(image_path: str) -> tuple:
    """Extract the true label from the image path."""
    path_obj = Path(image_path)
    class_name = path_obj.parent.name  # Get folder name (F0, F1, etc.)
    
    if class_name in CLASS_NAMES:
        class_idx = CLASS_NAMES.index(class_name)
        return class_idx, class_name
    else:
        return -1, "UNKNOWN"


def run_inference(model: nn.Module, manifest_df: pd.DataFrame, 
                  device: str) -> pd.DataFrame:
    """Run inference on all images in the manifest."""
    transform = get_inference_transforms()
    
    y_true_idx = []
    y_true_label = []
    y_pred_idx = []
    y_pred_label = []
    confidences = []
    
    print(f"\nRunning inference on {len(manifest_df)} images...")
    
    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df),
                         desc="Predicting"):
        image_path = row['image_path']
        
        # Get true label
        true_idx, true_label = get_true_label(image_path)
        y_true_idx.append(true_idx)
        y_true_label.append(true_label)
        
        # Get prediction
        pred_idx, pred_label, conf = predict_single_image(
            model, image_path, transform, device
        )
        y_pred_idx.append(pred_idx)
        y_pred_label.append(pred_label)
        confidences.append(conf)
    
    # Add columns to DataFrame
    manifest_df['y_true'] = y_true_label
    manifest_df['y_true_idx'] = y_true_idx
    manifest_df['y_pred'] = y_pred_label
    manifest_df['y_pred_idx'] = y_pred_idx
    manifest_df['confidence'] = confidences
    manifest_df['correct'] = manifest_df['y_true'] == manifest_df['y_pred']
    
    return manifest_df


def print_statistics(df: pd.DataFrame):
    """Print prediction statistics."""
    print("\n" + "="*60)
    print("PREDICTION STATISTICS")
    print("="*60)
    
    total = len(df)
    correct = df['correct'].sum()
    accuracy = correct / total * 100
    
    print(f"\nOverall Accuracy: {correct}/{total} = {accuracy:.2f}%")
    
    # Per-split accuracy
    print("\n" + "-"*60)
    print("Accuracy by Split:")
    print("-"*60)
    for split in ['Train', 'Test']:
        split_df = df[df['assigned_split'] == split]
        if len(split_df) > 0:
            split_correct = split_df['correct'].sum()
            split_acc = split_correct / len(split_df) * 100
            print(f"{split}: {split_correct}/{len(split_df)} = {split_acc:.2f}%")
    
    # Per-class accuracy
    print("\n" + "-"*60)
    print("Accuracy by Class:")
    print("-"*60)
    for class_name in CLASS_NAMES:
        class_df = df[df['y_true'] == class_name]
        if len(class_df) > 0:
            class_correct = class_df['correct'].sum()
            class_acc = class_correct / len(class_df) * 100
            print(f"{class_name}: {class_correct}/{len(class_df)} = {class_acc:.2f}%")
    
    print("="*60)


def main():
    """Main function to generate predictions."""
    print("="*60)
    print("GENERATING MODEL PREDICTIONS FOR DATASET MANIFEST")
    print("="*60)
    
    print(f"\nDevice: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    
    # Check if manifest exists
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        return
    
    # Load manifest
    print("\nLoading manifest...")
    manifest_df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded {len(manifest_df)} image entries")
    
    # Load model
    model = load_model(MODEL_PATH, DEVICE)
    
    # Run inference
    result_df = run_inference(model, manifest_df, DEVICE)
    
    # Save updated manifest
    print(f"\nSaving predictions to: {OUTPUT_PATH}")
    result_df.to_csv(OUTPUT_PATH, index=False)
    
    # Also update the original manifest
    result_df.to_csv(MANIFEST_PATH, index=False)
    print(f"Updated original manifest: {MANIFEST_PATH}")
    
    # Print statistics
    print_statistics(result_df)
    
    print("\n" + "="*60)
    print("PREDICTION GENERATION COMPLETE!")
    print("="*60)
    print(f"\nOutput file: {OUTPUT_PATH}")
    print("\nColumns in output:")
    print("  - image_path: Path to the image")
    print("  - assigned_split: Train/Test assignment")
    print("  - y_true: Actual fibrosis stage (F0-F4)")
    print("  - y_pred: Predicted fibrosis stage (F0-F4)")
    print("  - confidence: Model confidence (0-1)")
    print("  - correct: Whether prediction matches ground truth")


if __name__ == "__main__":
    main()
