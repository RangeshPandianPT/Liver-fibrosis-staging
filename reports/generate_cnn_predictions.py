"""
Generate CNN Predictions for Liver Fibrosis Staging.

Runs inference on Test split using trained ResNet50 and EfficientNet-V2 models,
and saves predictions to cnn_predictions.csv.

Usage:
    python generate_cnn_predictions.py
"""
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import DEVICE, OUTPUT_DIR, CHECKPOINT_DIR, CLASS_NAMES
from src.preprocessing import get_val_transforms
from src.models.resnet_branch import ResNet50Branch
from src.models.efficientnet_branch import EfficientNetBranch


class TestDataset(Dataset):
    """Dataset for Test split inference."""
    
    def __init__(self, manifest_path: str, transform=None):
        self.transform = transform
        
        # Load manifest and filter Test split
        df = pd.read_csv(manifest_path)
        self.data = df[df['assigned_split'] == 'Test'].reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} Test samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, row['image_path'], row['y_true']


def load_model(model_class, checkpoint_name: str, device: str):
    """Load a trained model from checkpoint."""
    model = model_class(pretrained=False)
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {checkpoint_name} (val_acc: {checkpoint.get('val_acc', 'N/A'):.2f}%)")
    return model


def run_inference(model, dataloader, device):
    """Run inference and return probabilities and predictions."""
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for images, _, _ in tqdm(dataloader, desc='Running inference'):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return all_probs, all_preds


def main():
    print("\n" + "=" * 70)
    print("CNN PREDICTIONS - LIVER FIBROSIS STAGING")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print("=" * 70 + "\n")
    
    # Load test dataset
    manifest_path = OUTPUT_DIR / 'dataset_manifest.csv'
    test_dataset = TestDataset(manifest_path, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Load models
    print("\nLoading trained models...")
    resnet = load_model(ResNet50Branch, 'best_resnet_model.pth', DEVICE)
    effnet = load_model(EfficientNetBranch, 'best_effnet_model.pth', DEVICE)
    
    # Run inference
    print("\nRunning ResNet50 inference...")
    resnet_probs, resnet_preds = run_inference(resnet, test_loader, DEVICE)
    
    print("\nRunning EfficientNet-V2 inference...")
    effnet_probs, effnet_preds = run_inference(effnet, test_loader, DEVICE)
    
    # Build predictions dataframe
    print("\nBuilding predictions CSV...")
    
    # Get filenames and true labels from dataset
    df_manifest = pd.read_csv(manifest_path)
    df_test = df_manifest[df_manifest['assigned_split'] == 'Test'].reset_index(drop=True)
    
    results = {
        'filename': [Path(p).name for p in df_test['image_path']],
        'true_label': df_test['y_true'].values
    }
    
    # Add ResNet probabilities
    for i, class_name in enumerate(CLASS_NAMES):
        results[f'resnet_{class_name.lower()}_prob'] = [p[i] for p in resnet_probs]
    
    # Add EfficientNet probabilities
    for i, class_name in enumerate(CLASS_NAMES):
        results[f'effnet_{class_name.lower()}_prob'] = [p[i] for p in effnet_probs]
    
    # Add predictions
    results['resnet_pred'] = [CLASS_NAMES[p] for p in resnet_preds]
    results['effnet_pred'] = [CLASS_NAMES[p] for p in effnet_preds]
    
    # Create DataFrame
    df_predictions = pd.DataFrame(results)
    
    # Save to CSV
    output_path = OUTPUT_DIR / 'cnn_predictions.csv'
    df_predictions.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to: {output_path}")
    print(f"Total samples: {len(df_predictions)}")
    
    # Calculate and print accuracy summary
    resnet_correct = (df_predictions['resnet_pred'] == df_predictions['true_label']).sum()
    effnet_correct = (df_predictions['effnet_pred'] == df_predictions['true_label']).sum()
    
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"\nResNet50 Accuracy: {100*resnet_correct/len(df_predictions):.2f}%")
    print(f"EfficientNet-V2 Accuracy: {100*effnet_correct/len(df_predictions):.2f}%")
    
    # Preview
    print("\nPreview of cnn_predictions.csv:")
    print(df_predictions.head(5).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
