"""
DenseNet121 branch for liver fibrosis classification.

DenseNet121 uses dense connectivity where each layer receives feature maps
from all preceding layers, enabling strong feature reuse and gradient flow.
This makes it well-suited for medical image classification tasks where
subtle textural differences between fibrosis stages matter.
"""
import torch
import torch.nn as nn
from torchvision import models


class DenseNetBranch(nn.Module):
    """
    DenseNet121 branch for liver fibrosis staging.

    Uses torchvision's DenseNet121 pretrained on ImageNet, with the
    classifier head replaced for `num_classes` outputs.

    DenseNet121 architecture highlights:
      - Dense blocks with skip connections to all subsequent layers
      - Growth rate = 32, compression factor = 0.5
      - Final feature map: 1024-dim global average pooled vector
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()

        # Load pretrained DenseNet121
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.densenet121(weights=weights)

        # Replace the classifier head
        in_features = self.model.classifier.in_features  # 1024
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get global-average-pooled features before the classifier head."""
        features = self.model.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    def get_target_layer(self) -> nn.Module:
        """
        Target layer for Grad-CAM.
        Returns the last dense block (denseblock4) for rich spatial features.
        """
        return self.model.features.denseblock4


if __name__ == "__main__":
    model = DenseNetBranch(num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"DenseNetBranch output shape : {out.shape}")          # (2, 5)
    feats = model.get_features(x)
    print(f"DenseNetBranch feature shape: {feats.shape}")        # (2, 1024)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
