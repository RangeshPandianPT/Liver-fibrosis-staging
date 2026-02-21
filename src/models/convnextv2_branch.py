"""
ConvNeXt V2 branch for liver fibrosis classification.
"""
import torch
import torch.nn as nn
import timm


class ConvNeXtV2Branch(nn.Module):
    """
    ConvNeXt V2 Branch using `timm`'s convnextv2_tiny.
    ConvNeXt V2 introduces Global Response Normalization (GRN)
    and uses a fully convolutional masked autoencoder (FCMAE) for pretraining.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.model = timm.create_model('convnextv2_tiny', pretrained=pretrained, num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def get_features(self, x):
        """Get features before the classification head."""
        return self.model.forward_features(x)

    def get_target_layer(self):
        """
        Get target layer for Grad-CAM.
        ConvNeXt V2 structure: model.stages[-1].blocks[-1]
        """
        return self.model.stages[-1].blocks[-1]


if __name__ == "__main__":
    model = ConvNeXtV2Branch(num_classes=5)
    x = torch.randn(2, 3, 384, 384)
    out = model(x)
    print(f"ConvNeXt V2 output shape: {out.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
