"""
Grad-CAM implementation for model explainability.
Generates attention heatmaps for liver fibrosis classification.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import sys
sys.path.insert(0, str(__file__).rsplit('src', 1)[0])
from config import CLASS_NAMES, GRADCAM_DIR, DEVICE, TOP_K_HEATMAPS
from src.preprocessing import denormalize


class GradCAMExplainer:
    """
    Grad-CAM explainer for generating attention heatmaps.
    Supports all three model branches in the ensemble.
    """
    
    def __init__(self, model, device: str = DEVICE):
        """
        Initialize the Grad-CAM explainer.
        
        Args:
            model: The ensemble model or individual branch
            device: Device to use
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Create output directories for each class
        for class_name in CLASS_NAMES:
            (GRADCAM_DIR / class_name).mkdir(parents=True, exist_ok=True)
    
    def get_cam_for_branch(self, 
                           branch_name: str,
                           input_tensor: torch.Tensor,
                           target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific model branch.
        
        Args:
            branch_name: One of 'resnet50', 'efficientnet', 'vit'
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for Grad-CAM. If None, uses predicted class.
            
        Returns:
            CAM heatmap as numpy array
        """
        # Get the specific branch
        branch = self.model.get_model_branch(branch_name)
        target_layer = [branch.get_target_layer()]
        
        # Create Grad-CAM
        cam = GradCAM(model=branch, target_layers=target_layer)
        
        # Generate CAM
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]
        else:
            targets = None
        
        grayscale_cam = cam(
            input_tensor=input_tensor.to(self.device),
            targets=targets
        )
        
        return grayscale_cam[0]
    
    def generate_heatmap_overlay(self,
                                  input_tensor: torch.Tensor,
                                  cam_heatmap: np.ndarray,
                                  alpha: float = 0.5) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.
        
        Args:
            input_tensor: Original image tensor (C, H, W)
            cam_heatmap: Grad-CAM heatmap (H, W)
            alpha: Transparency of the heatmap overlay
            
        Returns:
            Image with heatmap overlay as numpy array (H, W, 3)
        """
        # Denormalize image
        img = denormalize(input_tensor)
        img_float = img.astype(np.float32) / 255.0
        
        # Create overlay
        visualization = show_cam_on_image(img_float, cam_heatmap, use_rgb=True)
        
        return visualization
    
    def save_heatmap(self,
                     visualization: np.ndarray,
                     class_name: str,
                     image_idx: int,
                     branch_name: str = 'ensemble',
                     confidence: float = None) -> str:
        """
        Save a heatmap visualization.
        
        Args:
            visualization: The heatmap overlay image
            class_name: True class name
            image_idx: Index of the image
            branch_name: Name of the model branch used
            confidence: Optional prediction confidence
            
        Returns:
            Path to saved image
        """
        filename = f"{class_name}_{image_idx}_{branch_name}"
        if confidence is not None:
            filename += f"_conf{confidence:.2f}"
        filename += ".png"
        
        save_path = GRADCAM_DIR / class_name / filename
        
        # Save image
        Image.fromarray(visualization).save(save_path)
        
        return str(save_path)
    
    def generate_multi_branch_visualization(self,
                                             input_tensor: torch.Tensor,
                                             target_class: int,
                                             figsize: Tuple[int, int] = (16, 4)) -> np.ndarray:
        """
        Generate side-by-side Grad-CAM visualizations from all three branches.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for visualization
            figsize: Figure size
            
        Returns:
            Combined visualization as numpy array
        """
        branch_names = ['resnet50', 'efficientnet', 'vit']
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Original image
        img = denormalize(input_tensor[0])
        axes[0].imshow(img)
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Grad-CAM for each branch
        for idx, branch_name in enumerate(branch_names):
            cam = self.get_cam_for_branch(branch_name, input_tensor, target_class)
            overlay = self.generate_heatmap_overlay(input_tensor[0], cam)
            
            axes[idx + 1].imshow(overlay)
            axes[idx + 1].set_title(branch_name.upper(), fontsize=12, fontweight='bold')
            axes[idx + 1].axis('off')
        
        plt.suptitle(f'Grad-CAM Attention Maps - Class: {CLASS_NAMES[target_class]}',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Convert figure to numpy array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        
        return img_array


def select_top_k_correct_predictions(model,
                                      dataloader,
                                      k: int = TOP_K_HEATMAPS,
                                      device: str = DEVICE) -> Dict[str, List[Tuple]]:
    """
    Select top-k correctly classified images per class with highest confidence.
    
    Args:
        model: The trained model
        dataloader: Validation data loader
        k: Number of images to select per class
        device: Device to use
        
    Returns:
        Dictionary mapping class names to list of (image_tensor, confidence, idx) tuples
    """
    model.eval()
    
    # Storage for candidates per class
    class_candidates = {name: [] for name in CLASS_NAMES}
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            confidences = probs.max(dim=1).values
            
            # Find correct predictions
            correct_mask = preds == labels
            
            for i in range(len(images)):
                if correct_mask[i]:
                    class_idx = labels[i].item()
                    class_name = CLASS_NAMES[class_idx]
                    conf = confidences[i].item()
                    
                    class_candidates[class_name].append({
                        'image': images[i].cpu(),
                        'confidence': conf,
                        'batch_idx': batch_idx,
                        'sample_idx': i
                    })
    
    # Select top-k per class
    top_k_per_class = {}
    for class_name, candidates in class_candidates.items():
        # Sort by confidence
        sorted_candidates = sorted(candidates, key=lambda x: x['confidence'], reverse=True)
        top_k_per_class[class_name] = sorted_candidates[:k]
        print(f"{class_name}: {len(sorted_candidates)} correct, selected top-{min(k, len(sorted_candidates))}")
    
    return top_k_per_class


def generate_top_k_heatmaps(model,
                            dataloader,
                            k: int = TOP_K_HEATMAPS,
                            device: str = DEVICE) -> None:
    """
    Generate Grad-CAM heatmaps for top-k correctly classified images per class.
    
    Args:
        model: The trained ensemble model
        dataloader: Validation data loader
        k: Number of heatmaps per class
        device: Device to use
    """
    print("\n" + "=" * 60)
    print("GENERATING GRAD-CAM HEATMAPS")
    print("=" * 60 + "\n")
    
    # Initialize explainer
    explainer = GradCAMExplainer(model, device)
    
    # Get top-k correct predictions per class
    top_k = select_top_k_correct_predictions(model, dataloader, k, device)
    
    # Generate heatmaps
    for class_name, candidates in top_k.items():
        print(f"\nGenerating heatmaps for {class_name}...")
        class_idx = CLASS_NAMES.index(class_name)
        
        for idx, candidate in enumerate(candidates):
            image = candidate['image'].unsqueeze(0).to(device)
            confidence = candidate['confidence']
            
            # Generate multi-branch visualization
            combined_viz = explainer.generate_multi_branch_visualization(
                image, class_idx
            )
            
            # Save combined visualization
            save_path = GRADCAM_DIR / class_name / f"{class_name}_{idx+1}_combined_conf{confidence:.2f}.png"
            Image.fromarray(combined_viz).save(save_path)
            
            # Also save individual branch heatmaps
            for branch_name in ['resnet50', 'efficientnet', 'vit']:
                cam = explainer.get_cam_for_branch(branch_name, image, class_idx)
                overlay = explainer.generate_heatmap_overlay(image[0], cam)
                explainer.save_heatmap(overlay, class_name, idx+1, branch_name, confidence)
        
        print(f"  Saved {len(candidates)} heatmaps to {GRADCAM_DIR / class_name}")
    
    print("\n" + "=" * 60)
    print(f"Heatmaps saved to: {GRADCAM_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully.")
    print("Use generate_top_k_heatmaps() with a trained model to generate heatmaps.")
