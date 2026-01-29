"""
ResNet50 branch for liver fibrosis classification.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

import sys
sys.path.insert(0, str(__file__).rsplit('src', 1)[0])
from config import NUM_CLASSES, PRETRAINED, FREEZE_BACKBONE


class ResNet50Branch(nn.Module):
    """
    ResNet50 model branch with pre-trained ImageNet weights.
    Modified final layer for 5-class liver fibrosis staging.
    """
    
    def __init__(self, 
                 num_classes: int = NUM_CLASSES,
                 pretrained: bool = PRETRAINED,
                 freeze_backbone: bool = FREEZE_BACKBONE):
        """
        Initialize ResNet50 branch.
        
        Args:
            num_classes: Number of output classes (default: 5 for F0-F4)
            pretrained: Whether to use ImageNet pre-trained weights
            freeze_backbone: Whether to freeze backbone layers
        """
        super().__init__()
        
        # Load pre-trained ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get the number of features in the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all layers except the final classification head."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
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
        Get features before the final classification layer.
        Useful for Grad-CAM visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        # Forward through all layers except fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        return x
    
    def get_target_layer(self):
        """Get the target layer for Grad-CAM (last conv layer)."""
        return self.backbone.layer4[-1]


if __name__ == "__main__":
    # Quick test
    model = ResNet50Branch()
    x = torch.randn(2, 3, 384, 384)
    out = model(x)
    print(f"ResNet50 output shape: {out.shape}")  # Expected: (2, 5)
