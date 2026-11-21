"""
Multi-Modal Clinical-Histological Fusion Architecture for Liver Fibrosis Staging.

Integrates histological biopsy features (Vision) with patient laboratory biomarkers
and transient elastography (FibroScan kPa, FIB-4, APRI) using Cross-Attention
and Adaptive Gated Multimodal Fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ClinicalTabularEncoder(nn.Module):
    """
    Encodes clinical lab tests (Age, AST, ALT, Platelets, Albumin, Bilirubin,
    FibroScan kPa, FIB-4, APRI, AST/ALT ratio) into a latent embedding space.
    """
    def __init__(self, in_features: int = 12, embed_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (batch_size, embed_dim)
        return self.net(x)


class VisionProjectionEncoder(nn.Module):
    """
    Projects visual embeddings (e.g. from ConvNeXt / ResNet / ensemble probability space)
    into the shared multimodal latent embedding space.
    """
    def __init__(self, in_dim: int = 20, embed_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossAttentionFusionBlock(nn.Module):
    """
    Bidirectional Cross-Attention:
    - Vision queries attend to Clinical biomarkers (histology contextualized by blood labs)
    - Clinical queries attend to Vision features (lab biomarkers guided by tissue architecture)
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha_vis_to_tab = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_tab_to_vis = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm_v1 = nn.LayerNorm(embed_dim)
        self.norm_t1 = nn.LayerNorm(embed_dim)
        self.norm_v2 = nn.LayerNorm(embed_dim)
        self.norm_t2 = nn.LayerNorm(embed_dim)
        
        self.ffn_v = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.ffn_t = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, v_emb: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inputs: (B, D) -> Unsqueeze to sequence length 1: (B, 1, D)
        v_seq = v_emb.unsqueeze(1)
        t_seq = t_emb.unsqueeze(1)
        
        # 1. Vision attends to Tabular
        v_cross, _ = self.mha_vis_to_tab(query=v_seq, key=t_seq, value=t_seq)
        v_seq = self.norm_v1(v_seq + v_cross)
        v_seq = self.norm_v2(v_seq + self.ffn_v(v_seq))
        
        # 2. Tabular attends to Vision
        t_cross, _ = self.mha_tab_to_vis(query=t_seq, key=v_seq, value=v_seq)
        t_seq = self.norm_t1(t_seq + t_cross)
        t_seq = self.norm_t2(t_seq + self.ffn_t(t_seq))
        
        return v_seq.squeeze(1), t_seq.squeeze(1)


class AdaptiveGatedFusion(nn.Module):
    """
    Adaptive Modality Gating:
    Learns dynamic weighting alpha in [0, 1] per sample:
    e_fused = alpha * e_vis + (1 - alpha) * e_tab
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        self.post_fuse = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, v_emb: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cat_feat = torch.cat([v_emb, t_emb], dim=-1)
        gate = self.gate_fc(cat_feat)  # (B, 1) in [0, 1] (weight on vision)
        fused = gate * v_emb + (1.0 - gate) * t_emb
        fused = self.post_fuse(fused)
        return fused, gate


class MultiModalFusionClassifier(nn.Module):
    """
    Full End-to-End Multi-Modal Classifier for METAVIR Liver Fibrosis Staging (F0-F4).
    Combines Vision features + Clinical tabular lab biomarkers.
    """
    def __init__(
        self,
        vision_in_dim: int = 20,
        clinical_in_dim: int = 12,
        embed_dim: int = 128,
        num_classes: int = 5,
        num_heads: int = 4,
        dropout: float = 0.15
    ):
        super().__init__()
        self.vision_encoder = VisionProjectionEncoder(vision_in_dim, embed_dim, dropout)
        self.clinical_encoder = ClinicalTabularEncoder(clinical_in_dim, embed_dim, dropout)
        self.cross_attention = CrossAttentionFusionBlock(embed_dim, num_heads, dropout)
        self.gated_fusion = AdaptiveGatedFusion(embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(
        self,
        vision_feat: torch.Tensor,
        clinical_feat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vision_feat: Visual feature tensor, shape (B, vision_in_dim)
            clinical_feat: Tabular clinical biomarker tensor, shape (B, clinical_in_dim)
            
        Returns:
            Dict containing:
                'logits': Raw classification logits (B, 5)
                'probs': Softmax probability vector (B, 5)
                'pred_class': Predicted METAVIR stage index (B,)
                'entropy': Shannon entropy uncertainty (B,)
                'vision_gate': Adaptive weight assigned to vision modality (B, 1)
        """
        # 1. Project into latent space
        v_emb = self.vision_encoder(vision_feat)
        t_emb = self.clinical_encoder(clinical_feat)
        
        # 2. Bidirectional Cross-Attention
        v_attended, t_attended = self.cross_attention(v_emb, t_emb)
        
        # 3. Gated Fusion
        fused, gate = self.gated_fusion(v_attended, t_attended)
        
        # 4. 5-Class METAVIR classification
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1)
        
        # 5. Shannon entropy uncertainty: -sum(p * log(p + eps))
        eps = 1e-8
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'pred_class': pred_class,
            'entropy': entropy,
            'vision_gate': gate,
            'clinical_gate': 1.0 - gate
        }
