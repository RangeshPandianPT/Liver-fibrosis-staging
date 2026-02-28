"""
MedNeXt (ConvNeXt-based) branch for liver fibrosis classification.
"""
import torch
import torch.nn as nn
import timm

class MedNeXtBranch(nn.Module):
    """
    MedNeXt Branch using `timm`'s convnext_tiny as backbone.
    MedNeXt is architecturally based on ConvNeXt blocks.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        
        # Load ConvNeXt Tiny from timm
        # convnext_tiny is a good balance of speed/accuracy for this size
        self.model = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=num_classes)
        
        # timm's convnext has a 'head' (norm + fc)
        # The create_model with num_classes already sets up the head correctly
        # But we can inspect if we need custom head modification.
        # usually: model.head.fc is the final linear layer
        
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def get_features(self, x):
        """
        Get features before the classification head.
        """
        return self.model.forward_features(x)

    def get_target_layer(self):
        """
        Get target layer for Grad-CAM.
        For ConvNeXt, we typically target the last stage's final block.
        timm structure: model.stages[-1].blocks[-1]
        """
        return self.model.stages[-1].blocks[-1]

if __name__ == "__main__":
    # Quick test
    model = MedNeXtBranch(num_classes=5)
    x = torch.randn(2, 3, 384, 384)
    out = model(x)
    print(f"MedNeXt output shape: {out.shape}")
