"""
Advanced Explainable AI (XAI) and Clinical Uncertainty Estimation Module.

Provides EigenCAM attention maps tailored for Vision Transformers (DeiT/ViT) and CNNs,
alongside predictive entropy and clinical ambiguity alerts for histological slide grading.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

try:
    from pytorch_grad_cam import GradCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    pass

import sys
sys.path.insert(0, str(__file__).rsplit('src', 1)[0])
from config import CLASS_NAMES, GRADCAM_DIR, DEVICE
from src.preprocessing import denormalize


class AdvancedXAIExplainer:
    """
    Advanced XAI Explainer utilizing EigenCAM and Grad-CAM.
    EigenCAM computes the first principal component of 2D activations, making it
    especially robust and noise-free for Vision Transformers (DeiT, ViT) and ConvNeXt.
    """

    def __init__(self, model: nn.Module, device: str = DEVICE):
        """
        Initialize XAI Explainer.

        Args:
            model: Ensemble model or standalone model branch
            device: Computing device ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.model.eval()

    def get_target_layer(self, branch: nn.Module) -> List[nn.Module]:
        """Extract target layer from branch for CAM computation."""
        if hasattr(branch, "get_target_layer"):
            layer = branch.get_target_layer()
            return [layer] if not isinstance(layer, list) else layer
        # Fallback heuristics for common architectures
        if hasattr(branch, "layer4"):
            return [branch.layer4[-1]]
        if hasattr(branch, "stages"):
            return [branch.stages[-1]]
        if hasattr(branch, "blocks"):
            return [branch.blocks[-1].norm1 if hasattr(branch.blocks[-1], "norm1") else branch.blocks[-1]]
        return [list(branch.children())[-1]]

    def generate_cam(self,
                     input_tensor: torch.Tensor,
                     branch_name: Optional[str] = None,
                     target_class: Optional[int] = None,
                     method: str = "eigencam") -> np.ndarray:
        """
        Generate CAM heatmap using EigenCAM or Grad-CAM.

        Args:
            input_tensor: Image tensor (1, C, H, W)
            branch_name: Name of branch ('convnextv2', 'mednext', 'deit', 'resnet').
                         If None, attempts to use the model directly.
            target_class: Target class index. If None, uses predicted class.
            method: 'eigencam' (default) or 'gradcam'.

        Returns:
            Normalized heatmap as 2D numpy array (H, W) in range [0, 1].
        """
        if not branch_name and hasattr(self.model, "get_model_branch"):
            branch_name = "convnextv2"

        if branch_name and hasattr(self.model, "get_model_branch"):
            branch = self.model.get_model_branch(branch_name)
        else:
            branch = self.model

        target_layers = self.get_target_layer(branch)

        cam_class = EigenCAM if method.lower() == "eigencam" else GradCAM
        cam = cam_class(model=branch, target_layers=target_layers)

        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        
        with torch.no_grad() if method.lower() == "eigencam" else torch.enable_grad():
            grayscale_cam = cam(input_tensor=input_tensor.to(self.device), targets=targets)

        return grayscale_cam[0]

    def create_overlay(self, input_tensor: torch.Tensor, heatmap: np.ndarray) -> np.ndarray:
        """Create RGB image overlay of heatmap over original slide."""
        img = denormalize(input_tensor[0] if input_tensor.ndim == 4 else input_tensor)
        img_float = img.astype(np.float32) / 255.0
        return show_cam_on_image(img_float, heatmap, use_rgb=True)

    def generate_comparative_visualization(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate side-by-side comparison of Grad-CAM vs EigenCAM for available branches.
        """
        branches = ['convnextv2', 'mednext', 'deit', 'resnet']
        valid_branches = [b for b in branches if hasattr(self.model, "get_model_branch")]
        if not valid_branches:
            valid_branches = ['model']

        n_cols = len(valid_branches) + 1
        fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))

        img = denormalize(input_tensor[0])
        
        # Row 0: Grad-CAM
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original Slide", fontweight="bold")
        axes[0, 0].axis("off")

        # Row 1: EigenCAM
        axes[1, 0].imshow(img)
        axes[1, 0].set_title("Original Slide", fontweight="bold")
        axes[1, 0].axis("off")

        for idx, branch_name in enumerate(valid_branches):
            b_name = None if branch_name == 'model' else branch_name
            # GradCAM
            try:
                g_map = self.generate_cam(input_tensor, b_name, target_class, method="gradcam")
                g_ov = self.create_overlay(input_tensor, g_map)
                axes[0, idx + 1].imshow(g_ov)
                axes[0, idx + 1].set_title(f"{branch_name.upper()} (Grad-CAM)", fontsize=10)
            except Exception as e:
                axes[0, idx + 1].text(0.5, 0.5, "Grad-CAM\nUnavailable", ha="center", va="center")
            axes[0, idx + 1].axis("off")

            # EigenCAM
            try:
                e_map = self.generate_cam(input_tensor, b_name, target_class, method="eigencam")
                e_ov = self.create_overlay(input_tensor, e_map)
                axes[1, idx + 1].imshow(e_ov)
                axes[1, idx + 1].set_title(f"{branch_name.upper()} (EigenCAM)", fontsize=10)
            except Exception as e:
                axes[1, idx + 1].text(0.5, 0.5, "EigenCAM\nUnavailable", ha="center", va="center")
            axes[1, idx + 1].axis("off")

        plt.suptitle(f"Advanced XAI Attention Comparison - Class: {CLASS_NAMES[target_class]}", fontweight="bold", y=0.98)
        plt.tight_layout()

        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return img_array


class UncertaintyEstimator:
    """
    Clinical Uncertainty and Ambiguity Estimator for Liver Fibrosis Staging.
    Computes Shannon predictive entropy and confidence margin alerts.
    """

    @staticmethod
    def compute_entropy(probs: np.ndarray) -> float:
        """
        Compute Shannon entropy of prediction probability distribution: -sum(p * log(p)).
        Higher entropy (> 0.8) indicates severe diagnostic ambiguity across fibrosis stages.
        """
        eps = 1e-10
        probs_clipped = np.clip(probs, eps, 1.0)
        entropy = -np.sum(probs_clipped * np.log(probs_clipped))
        # Normalize entropy by max possible entropy (log(num_classes))
        max_entropy = np.log(len(probs))
        return min(1.0, float(entropy / max_entropy))

    @staticmethod
    def compute_margin(probs: np.ndarray) -> float:
        """
        Compute probability margin between Top-1 and Top-2 predicted fibrosis stages.
        Margin < 0.15 indicates borderline fibrosis requiring pathologist review.
        """
        sorted_probs = np.sort(probs)[::-1]
        if len(sorted_probs) < 2:
            return 1.0
        return float(sorted_probs[0] - sorted_probs[1])

    @classmethod
    def analyze_prediction(cls, probs: np.ndarray) -> Dict[str, Any]:
        """
        Perform complete clinical diagnostic uncertainty analysis.

        Args:
            probs: 1D numpy array of class probabilities (e.g. [0.02, 0.45, 0.48, 0.03, 0.02])

        Returns:
            Dictionary containing entropy, margin, clinical alert status, and recommendation.
        """
        entropy = cls.compute_entropy(probs)
        margin = cls.compute_margin(probs)
        top1_idx = int(np.argmax(probs))
        sorted_indices = np.argsort(probs)[::-1]
        top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top1_idx

        alert_triggered = False
        alert_reasons = []

        if entropy > 0.65:
            alert_triggered = True
            alert_reasons.append(f"High predictive entropy ({entropy:.2f}) indicates diffuse probability distribution.")
        if margin < 0.15:
            alert_triggered = True
            alert_reasons.append(f"Low confidence margin ({margin:.2f}) between {CLASS_NAMES[top1_idx]} and {CLASS_NAMES[top2_idx]}.")

        if alert_triggered:
            status = "⚠️ WARNING: Borderline / Ambiguous Fibrosis Stage"
            recommendation = (f"Manual Pathology Consultation Required. The model is uncertain between stage "
                              f"**{CLASS_NAMES[top1_idx]}** ({probs[top1_idx]*100:.1f}%) and stage "
                              f"**{CLASS_NAMES[top2_idx]}** ({probs[top2_idx]*100:.1f}%). "
                              f"Recommend special staining (Masson's Trichrome / Sirius Red) or expert consensus.")
        else:
            status = "✅ Confident Diagnostic Classification"
            recommendation = (f"High confidence prediction for stage **{CLASS_NAMES[top1_idx]}** "
                              f"({probs[top1_idx]*100:.1f}%). Low diagnostic ambiguity (Entropy: {entropy:.2f}).")

        return {
            "predicted_class": CLASS_NAMES[top1_idx],
            "predicted_prob": float(probs[top1_idx]),
            "secondary_class": CLASS_NAMES[top2_idx],
            "secondary_prob": float(probs[top2_idx]),
            "entropy": entropy,
            "margin": margin,
            "alert_triggered": alert_triggered,
            "status": status,
            "recommendation": recommendation,
            "reasons": alert_reasons
        }
