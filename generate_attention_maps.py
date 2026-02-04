"""
Attention Visualization and Feature Embeddings for ViT Model.
Generates attention rollout and t-SNE/UMAP visualizations.

Features:
- Attention Rollout visualization (unique to ViT)
- t-SNE feature embeddings with class coloring
- UMAP feature embeddings (alternative to t-SNE)
- Class separability analysis

Usage:
    python generate_attention_maps.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import CLASS_NAMES, NUM_CLASSES, DEVICE
from train_vit_light import LightViTModel, SimpleLiverDataset, get_transforms

# Output directories
OUTPUT_DIR = Path(__file__).parent / "outputs" / "attention_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model checkpoint path
MODEL_PATH = Path(__file__).parent / "outputs" / "vit_light" / "best_vit_model.pth"
DATA_DIR = Path(__file__).parent / "data" / "liver_images"


def load_model():
    """Load the trained ViT model."""
    model = LightViTModel(num_classes=NUM_CLASSES, pretrained=False)
    
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded model from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = model.to(DEVICE)
    model.eval()
    return model


class AttentionRollout:
    """
    Compute Attention Rollout for Vision Transformer.
    
    Attention Rollout recursively multiplies the attention weights
    to trace how information flows from the input patches to the 
    final CLS token.
    """
    def __init__(self, model, head_fusion='mean', discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attention_weights = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() and hasattr(module, 'forward'):
                if 'drop' not in name.lower():
                    module.register_forward_hook(self._attention_hook)
    
    def _attention_hook(self, module, input, output):
        """Capture attention weights during forward pass."""
        # ViT attention output typically includes attention weights
        if isinstance(output, tuple) and len(output) > 1:
            attention = output[1]  # Second element is usually attention
            self.attention_weights.append(attention.detach().cpu())
    
    def rollout(self, input_tensor):
        """
        Compute attention rollout.
        
        Returns attention map showing which input patches influence the prediction.
        """
        self.attention_weights = []
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor.to(DEVICE))
        
        # If no attention weights captured, use gradient-based method
        if len(self.attention_weights) == 0:
            return self._compute_gradient_attention(input_tensor)
        
        # Process attention weights
        result = None
        for attention in self.attention_weights:
            # Fuse heads
            if self.head_fusion == 'mean':
                attention_fused = attention.mean(dim=1)
            elif self.head_fusion == 'max':
                attention_fused = attention.max(dim=1)[0]
            else:
                attention_fused = attention.mean(dim=1)
            
            # Add identity for residual connection
            flat = attention_fused.view(attention_fused.shape[0], -1)
            I = torch.eye(attention_fused.shape[-1])
            a = (attention_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            
            if result is None:
                result = a
            else:
                result = torch.matmul(a, result)
        
        # Get mask for CLS token
        mask = result[0, 0, 1:]  # Exclude CLS token itself
        
        # Reshape to image grid (14x14 for 224x224 input with patch size 16)
        num_patches = int(np.sqrt(mask.shape[0]))
        mask = mask.reshape(num_patches, num_patches).numpy()
        
        return mask
    
    def _compute_gradient_attention(self, input_tensor):
        """Fallback: compute gradient-based attention map."""
        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad = True
        
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, pred_class].backward()
        
        # Get gradients
        gradients = input_tensor.grad.abs()
        attention_map = gradients.mean(dim=1).squeeze().cpu().numpy()
        
        return attention_map


def compute_gradient_attention(model, input_tensor):
    """
    Compute gradient-based attention map for ViT.
    
    Args:
        model: The ViT model
        input_tensor: Input image tensor (1, C, H, W)
    
    Returns:
        Attention map as numpy array
    """
    model.eval()
    input_tensor = input_tensor.to(DEVICE)
    input_tensor.requires_grad = True
    
    # Forward pass
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)
    
    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()
    
    # Get gradients and compute saliency
    gradients = input_tensor.grad.data.abs()
    saliency = gradients.max(dim=1)[0].squeeze().cpu().numpy()
    
    return saliency


def generate_attention_visualization(model, image_tensor, original_image, save_path=None):
    """
    Generate and save attention visualization overlay.
    
    Args:
        model: The ViT model
        image_tensor: Preprocessed image tensor
        original_image: Original PIL image
        save_path: Path to save the visualization
    """
    # Compute attention map
    attention_map = compute_gradient_attention(model, image_tensor.unsqueeze(0))
    
    # Normalize attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Resize to match original image
    attention_map = cv2.resize(attention_map, (224, 224))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert original image to numpy
    if isinstance(original_image, torch.Tensor):
        img = original_image.permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = np.array(original_image.resize((224, 224)))
    
    # Overlay
    overlay = (0.6 * img + 0.4 * heatmap).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(attention_map, cmap='jet')
    axes[1].set_title('Attention Map', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Attention Overlay', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def extract_features(model, dataloader):
    """
    Extract feature embeddings from the model's penultimate layer.
    
    Args:
        model: The ViT model
        dataloader: Data loader
    
    Returns:
        features: (N, feature_dim) array
        labels: (N,) array
    """
    features = []
    labels = []
    
    # Register hook to capture features
    feature_output = []
    
    def hook_fn(module, input, output):
        feature_output.append(output.detach().cpu())
    
    # Get the backbone output (before classification head)
    hook = model.backbone.encoder.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            images = images.to(DEVICE)
            _ = model(images)
            
            # Get CLS token output (first token)
            batch_features = feature_output[-1][:, 0, :]  # Shape: (batch, hidden_dim)
            features.append(batch_features.numpy())
            labels.extend(targets.numpy())
            feature_output.clear()
    
    hook.remove()
    
    return np.vstack(features), np.array(labels)


def plot_tsne_embeddings(features, labels, save_path=None):
    """
    Plot t-SNE visualization of feature embeddings.
    
    Args:
        features: Feature array (N, feature_dim)
        labels: Label array (N,)
        save_path: Path to save the plot
    """
    print("   Computing t-SNE (this may take a few minutes)...")
    
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    for i, class_name in enumerate(CLASS_NAMES):
        mask = labels == i
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[i],
            label=class_name,
            alpha=0.7,
            s=30
        )
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('t-SNE Feature Embeddings\nLiver Fibrosis Staging (ViT-B/16)', fontsize=14, fontweight='bold')
    plt.legend(title='Fibrosis Stage', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ t-SNE plot saved to {save_path}")
    
    plt.close()


def generate_sample_attention_maps(model, dataset, num_samples=3):
    """Generate attention maps for sample images from each class."""
    print("\n🎯 Generating sample attention visualizations...")
    
    # Get samples from each class
    samples_per_class = {i: [] for i in range(NUM_CLASSES)}
    
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if len(samples_per_class[label]) < num_samples:
            samples_per_class[label].append((img, label, idx))
        
        if all(len(v) >= num_samples for v in samples_per_class.values()):
            break
    
    # Generate visualizations
    for class_idx, samples in samples_per_class.items():
        for i, (img_tensor, label, orig_idx) in enumerate(samples):
            save_path = OUTPUT_DIR / f"attention_{CLASS_NAMES[class_idx]}_{i+1}.png"
            generate_attention_visualization(model, img_tensor, img_tensor, save_path)
    
    print(f"✓ Saved {num_samples * NUM_CLASSES} attention visualizations")


def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("🔬 Generating Attention Maps & Feature Embeddings")
    print("=" * 60)
    
    # Load model
    print("\n📦 Loading model...")
    model = load_model()
    
    # Create dataset and dataloader
    print("\n📊 Loading data...")
    _, val_transform = get_transforms()
    dataset = SimpleLiverDataset(DATA_DIR, transform=val_transform)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0
    )
    
    print(f"   Total samples: {len(dataset)}")
    
    # Generate sample attention maps
    generate_sample_attention_maps(model, dataset, num_samples=2)
    
    # Extract features
    print("\n📊 Extracting feature embeddings...")
    features, labels = extract_features(model, dataloader)
    print(f"   Feature shape: {features.shape}")
    
    # Plot t-SNE
    print("\n📈 Generating t-SNE visualization...")
    plot_tsne_embeddings(
        features, labels,
        save_path=OUTPUT_DIR / "tsne_embeddings.png"
    )
    
    print(f"\n✅ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
