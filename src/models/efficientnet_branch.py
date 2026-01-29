"""
EfficientNet-V2 branch for liver fibrosis classification.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights

import sys
sys.path.insert(0, str(__file__).rsplit('src', 1)[0])
from config import NUM_CLASSES, PRETRAINED, FREEZE_BACKBONE


class EfficientNetBranch(nn.Module):
    """
    EfficientNet-V2-S model branch with pre-trained ImageNet weights.
    Modified classifier head for 5-class liver fibrosis staging.
    """
    
    def __init__(self,
                 num_classes: int = NUM_CLASSES,
                 pretrained: bool = PRETRAINED,
                 freeze_backbone: bool = FREEZE_BACKBONE):
        """
        Initialize EfficientNet-V2 branch.
        
        Args:
            num_classes: Number of output classes (default: 5 for F0-F4)
            pretrained: Whether to use ImageNet pre-trained weights
            freeze_backbone: Whether to freeze backbone layers
        """
        super().__init__()
        
        # Load pre-trained EfficientNet-V2-S
        if pretrained:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_v2_s(weights=weights)
        else:
            self.backbone = models.efficientnet_v2_s(weights=None)
        
        # Get the number of features in classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all layers except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features before the classifier.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return x
    
    def get_target_layer(self):
        """Get the target layer for Grad-CAM (last feature block)."""
        return self.backbone.features[-1]


if __name__ == "__main__":
    # Quick test
    model = EfficientNetBranch()
    x = torch.randn(2, 3, 384, 384)
    out = model(x)
    print(f"EfficientNet-V2 output shape: {out.shape}")  # Expected: (2, 5)
