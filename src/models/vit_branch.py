    """
    Vision Transformer (ViT) branch for liver fibrosis classification.
    """
    import torch
    import torch.nn as nn
    from torchvision import models
    from torchvision.models import ViT_B_16_Weights

    import sys
    sys.path.insert(0, str(__file__).rsplit('src', 1)[0])
    from config import NUM_CLASSES, PRETRAINED, FREEZE_BACKBONE


    class ViTBranch(nn.Module):
        """
        Vision Transformer (ViT-B/16) model branch with pre-trained ImageNet weights.
        Modified head for 5-class liver fibrosis staging.
        """
        
        def __init__(self,
                    num_classes: int = NUM_CLASSES,
                    pretrained: bool = PRETRAINED,
                    freeze_backbone: bool = FREEZE_BACKBONE):
            """
            Initialize ViT branch.
            
            Args:
                num_classes: Number of output classes (default: 5 for F0-F4)
                pretrained: Whether to use ImageNet pre-trained weights
                freeze_backbone: Whether to freeze backbone layers
            """
            super().__init__()
            
            # Load pre-trained ViT-B/16 with SWAG weights (supports 384x384 images)
            if pretrained:
                weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
                self.backbone = models.vit_b_16(weights=weights)
            else:
                self.backbone = models.vit_b_16(weights=None)
            
            # Get the number of features in the head
            num_features = self.backbone.heads.head.in_features
            
            # Replace classification head
            self.backbone.heads = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_features, num_classes)
            )
            
            # Optionally freeze backbone
            if freeze_backbone:
                self._freeze_backbone()
        
        def _freeze_backbone(self):
            """Freeze all layers except the classification head."""
            for name, param in self.backbone.named_parameters():
                if 'heads' not in name:
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
            Get features before the classification head.
            
            Args:
                x: Input tensor
                
            Returns:
                Feature tensor (class token)
            """
            # ViT forward until heads
            x = self.backbone._process_input(x)
            n = x.shape[0]
            
            # Expand class token
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            
            x = self.backbone.encoder(x)
            
            # Return class token features
            return x[:, 0]
        
        def get_target_layer(self):
            """
            Get the target layer for Grad-CAM.
            For ViT, we use the last encoder block's layer norm.
            """
            return self.backbone.encoder.layers[-1].ln_1


    class ViTGradCAMWrapper(nn.Module):
        """
        Wrapper for ViT to make it compatible with Grad-CAM.
        Reshapes attention outputs to spatial format.
        """
        
        def __init__(self, vit_model: ViTBranch):
            super().__init__()
            self.vit = vit_model
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.vit(x)
        
        def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
            """
            Get attention maps from the last transformer block.
            
            Args:
                x: Input tensor
                
            Returns:
                Attention maps reshaped to spatial format
            """
            # This requires hooks - implemented in gradcam.py
            pass


    if __name__ == "__main__":
        # Quick test
        model = ViTBranch()
        x = torch.randn(2, 3, 384, 384)
        out = model(x)
        print(f"ViT output shape: {out.shape}")  # Expected: (2, 5)
