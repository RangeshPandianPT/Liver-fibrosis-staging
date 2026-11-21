"""
Unit tests for Multi-Modal Clinical Fusion:
- Clinical biomarker calculators (FIB-4, APRI, AST/ALT ratio)
- Clinical risk tiering and cutoffs
- Multi-modal neural network forward passes and output shapes
"""

import pytest
import numpy as np
import torch

from src.utils.clinical_features import (
    compute_fib4, compute_apri, compute_ast_alt_ratio,
    interpret_fib4, interpret_apri, interpret_fibroscan,
    generate_matched_clinical_cohort
)
from src.models.multimodal_fusion import (
    ClinicalTabularEncoder, VisionProjectionEncoder,
    CrossAttentionFusionBlock, MultiModalFusionClassifier
)


def test_fib4_calculation():
    # Example: Age 50, AST 60, ALT 40, Platelets 200
    # FIB-4 = (50 * 60) / (200 * sqrt(40)) = 3000 / (200 * 6.3245) = 3000 / 1264.91 = 2.37
    val = compute_fib4(age=50, ast=60, alt=40, platelets=200)
    assert np.isclose(val, 2.37, atol=0.05)
    
    # Boundary / invalid checks
    assert compute_fib4(age=50, ast=60, alt=0, platelets=200) == 0.0
    assert compute_fib4(age=50, ast=60, alt=40, platelets=0) == 0.0


def test_apri_calculation():
    # Example: AST 80, Platelets 150, ULN 40
    # APRI = ((80 / 40) / 150) * 100 = (2 / 150) * 100 = 1.33
    val = compute_apri(ast=80, platelets=150, uln=40.0)
    assert np.isclose(val, 1.33, atol=0.05)


def test_clinical_risk_tiering():
    # FIB-4 interpretations
    low = interpret_fib4(1.20)
    assert low['risk_level'] == 'Low Risk'
    assert 'F0' in low['metavir_equiv']
    
    ind = interpret_fib4(2.45)
    assert ind['risk_level'] == 'Indeterminate Risk'
    
    high = interpret_fib4(4.10)
    assert high['risk_level'] == 'High Risk'
    
    # FibroScan interpretations
    assert interpret_fibroscan(5.2)['stage'] == 'F0-F1'
    assert interpret_fibroscan(8.1)['stage'] == 'F2'
    assert interpret_fibroscan(11.0)['stage'] == 'F3'
    assert interpret_fibroscan(18.5)['stage'] == 'F4'


def test_cohort_generator():
    labels = np.array([0, 1, 2, 3, 4, 2, 0])
    df = generate_matched_clinical_cohort(labels, random_seed=42)
    assert len(df) == 7
    assert 'fib4_score' in df.columns
    assert 'apri_score' in df.columns
    assert 'fibroscan_kpa' in df.columns
    assert (df['fibroscan_kpa'] > 0).all()


def test_multimodal_model_forward_pass():
    batch_size = 4
    vis_dim = 20
    clin_dim = 12
    num_classes = 5
    
    model = MultiModalFusionClassifier(
        vision_in_dim=vis_dim,
        clinical_in_dim=clin_dim,
        embed_dim=64,
        num_classes=num_classes,
        num_heads=2
    )
    model.eval()
    
    dummy_v = torch.randn(batch_size, vis_dim)
    dummy_c = torch.randn(batch_size, clin_dim)
    
    with torch.no_grad():
        out = model(dummy_v, dummy_c)
        
    assert out['logits'].shape == (batch_size, num_classes)
    assert out['probs'].shape == (batch_size, num_classes)
    assert out['pred_class'].shape == (batch_size,)
    assert out['entropy'].shape == (batch_size,)
    assert out['vision_gate'].shape == (batch_size, 1)
    
    # Softmax probabilities should sum to 1.0
    prob_sums = out['probs'].sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5)
    
    # Gates should be in [0, 1]
    assert (out['vision_gate'] >= 0.0).all() and (out['vision_gate'] <= 1.0).all()
