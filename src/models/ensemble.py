"""
Soft-Voting Ensemble combining ConvNeXtV2 and MedNeXt.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

import sys
sys.path.insert(0, str(__file__).rsplit('src', 1)[0])
from config import NUM_CLASSES, ENSEMBLE_WEIGHTS
from src.models.mednext_branch import MedNeXtBranch
from src.models.convnextv2_branch import ConvNeXtV2Branch
from src.models.deit_branch import DeiTBranch
from src.models.resnet_branch import ResNet50Branch


class SoftVotingEnsemble(nn.Module):
    """
    Soft-Voting Ensemble that merges softmax outputs of ConvNeXtV2, MedNeXt, DeiT, and ResNet50.

    The ensemble computes a weighted average of the softmax probabilities
    from the four branches.
    """

    def __init__(self,
                 num_classes: int = NUM_CLASSES,
                 weights: Optional[Dict[str, float]] = None,
                 pretrained: bool = True):
        """
        Initialize the ensemble.

        Args:
            num_classes: Number of output classes
            weights: Dictionary of model weights for ensemble voting
                     Keys: 'convnextv2', 'mednext', 'deit', 'resnet'
            pretrained: Whether to use pre-trained weights for all branches
        """
        super().__init__()

        # Initialize model branches
        self.mednext = MedNeXtBranch(num_classes=num_classes, pretrained=pretrained)
        self.convnextv2 = ConvNeXtV2Branch(num_classes=num_classes, pretrained=pretrained)
        self.deit = DeiTBranch(num_classes=num_classes, pretrained=pretrained)
        self.resnet = ResNet50Branch(num_classes=num_classes, pretrained=pretrained)

        # Set ensemble weights
        if weights is None:
            weights = ENSEMBLE_WEIGHTS

        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}

        # Store as buffers for save/load
        self.register_buffer('w_mednext',    torch.tensor(self.weights['mednext']))
        self.register_buffer('w_convnextv2', torch.tensor(self.weights['convnextv2']))
        self.register_buffer('w_deit',       torch.tensor(self.weights['deit']))
        self.register_buffer('w_resnet',     torch.tensor(self.weights['resnet']))

    def forward(self, x: torch.Tensor, return_individual: bool = False):
        """
        Forward pass through ensemble.

        Args:
            x: Input tensor of shape (B, C, H, W)
            return_individual: If True, also return individual model outputs

        Returns:
            If return_individual is False:
                Combined logits of shape (B, num_classes)
            If return_individual is True:
                Tuple of (combined_logits, dict of individual logits)
        """
        # Get logits from each branch
        logits_mednext    = self.mednext(x)
        logits_convnextv2 = self.convnextv2(x)
        logits_deit       = self.deit(x)
        logits_resnet     = self.resnet(x)

        # Convert to probabilities (softmax)
        probs_mednext    = F.softmax(logits_mednext, dim=1)
        probs_convnextv2 = F.softmax(logits_convnextv2, dim=1)
        probs_deit       = F.softmax(logits_deit, dim=1)
        probs_resnet     = F.softmax(logits_resnet, dim=1)

        # Weighted average of probabilities (soft voting)
        combined_probs = (
            self.w_mednext    * probs_mednext +
            self.w_convnextv2 * probs_convnextv2 +
            self.w_deit       * probs_deit +
            self.w_resnet     * probs_resnet
        )

        # Convert back to logits for loss computation
        combined_logits = torch.log(combined_probs + 1e-10)

        if return_individual:
            individual = {
                'mednext':    logits_mednext,
                'convnextv2': logits_convnextv2,
                'deit':       logits_deit,
                'resnet':     logits_resnet,
            }
            return combined_logits, individual

        return combined_logits

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get ensemble probabilities (softmax of combined logits).

        Args:
            x: Input tensor

        Returns:
            Probability distribution over classes
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_model_branch(self, name: str) -> nn.Module:
        """
        Get a specific model branch.

        Args:
            name: One of 'convnextv2', 'mednext', 'deit', 'resnet'

        Returns:
            The corresponding model branch
        """
        branches = {
            'mednext':    self.mednext,
            'convnextv2': self.convnextv2,
            'deit':       self.deit,
            'resnet':     self.resnet,
        }
        if name not in branches:
            raise ValueError(f"Unknown branch: {name}. Choose from {list(branches.keys())}")
        return branches[name]

    def get_target_layers(self) -> Dict[str, nn.Module]:
        """
        Get target layers for Grad-CAM from each branch.

        Returns:
            Dictionary mapping branch names to their target layers
        """
        return {
            'mednext':    self.mednext.get_target_layer(),
            'convnextv2': self.convnextv2.get_target_layer(),
            'deit':       self.deit.get_target_layer(),
            'resnet':     self.resnet.get_target_layer(),
        }

    def freeze_branches(self, branch_names: list = None):
        """
        Freeze specific branches.

        Args:
            branch_names: List of branch names to freeze. If None, freeze all.
        """
        if branch_names is None:
            branch_names = ['mednext', 'convnextv2', 'deit', 'resnet']

        for name in branch_names:
            branch = self.get_model_branch(name)
            for param in branch.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Quick test
    model = SoftVotingEnsemble()
    x = torch.randn(2, 3, 384, 384)

    # Test forward
    out = model(x)
    print(f"Ensemble output shape: {out.shape}")  # Expected: (2, 5)

    # Test with individual outputs
    out, individual = model(x, return_individual=True)
    for name, logits in individual.items():
        print(f"{name} output shape: {logits.shape}")

    # Print parameter counts
    print(f"\nTotal parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
