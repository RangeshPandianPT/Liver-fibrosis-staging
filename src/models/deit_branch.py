import torch
import torch.nn as nn
import timm
import sys
sys.path.insert(0, str(__file__).rsplit('src', 1)[0])
from config import IMAGE_SIZE

class DeiTBranch(nn.Module):
    """
    DeiT-Small Branch using `timm`.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        
        # Load from timm
        # 'deit_small_patch16_224' or 'deit_small_distilled_patch16_224'
        # We switch to the distilled version for better performance with a teacher
        self.model = timm.create_model('deit_small_distilled_patch16_224', pretrained=pretrained, img_size=IMAGE_SIZE)
        
        # Replace head
        # timm models usually have a 'head' attribute (or 'fc' or 'classifier')
        # We can check or just reset it.
        # For DeiT in timm, it's usually `head`.
        
        if hasattr(self.model, 'head'):
             in_features = self.model.head.in_features
             self.model.head = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'fc'): # ResNet style
             in_features = self.model.fc.in_features
             self.model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'classifier'): # EfficientNet style
             in_features = self.model.classifier.in_features
             self.model.classifier = nn.Linear(in_features, num_classes)
        else:
            # Fallback or specific check for deit
             # default deit in timm has 'head'
             pass

        # If distilled, handle head_dist
        # This is CRITICAL for the distilled model
        if hasattr(self.model, 'head_dist'):
             in_features = self.model.head_dist.in_features
             self.model.head_dist = nn.Linear(in_features, num_classes)
        
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def get_target_layer(self):
        # For Grad-CAM
        # In timm, blocks are usually in `blocks`
        # We target the last block's norm1 for attention visualization
        return self.model.blocks[-1].norm1
