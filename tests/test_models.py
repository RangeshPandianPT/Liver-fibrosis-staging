"""
Unit Tests for Model Architectures, SoftVotingEnsemble, and ONNX Export.
"""
import pytest
import torch
from pathlib import Path

from src.models import SoftVotingEnsemble
from src.inference.onnx_engine import export_to_onnx
from config import NUM_CLASSES, IMAGE_SIZE


@pytest.fixture(scope="module")
def sample_ensemble():
    """Create a non-pretrained ensemble for fast testing."""
    model = SoftVotingEnsemble(pretrained=False)
    model.eval()
    return model


def test_ensemble_forward_pass(sample_ensemble):
    """Verify ensemble forward pass outputs correct logits shape (B, num_classes)."""
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        out = sample_ensemble(x)
    
    assert out.shape == (2, NUM_CLASSES)
    assert not torch.isnan(out).any()


def test_ensemble_individual_outputs(sample_ensemble):
    """Verify returning individual branch predictions outputs all 4 branches."""
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        comb_logits, individual = sample_ensemble(x, return_individual=True)
    
    assert comb_logits.shape == (1, NUM_CLASSES)
    assert set(individual.keys()) == {'mednext', 'convnextv2', 'deit', 'resnet'}
    for branch_name, logits in individual.items():
        assert logits.shape == (1, NUM_CLASSES)


def test_get_model_branch(sample_ensemble):
    """Verify branch retrieval works for valid names and fails for invalid names."""
    for b_name in ['convnextv2', 'mednext', 'deit', 'resnet']:
        branch = sample_ensemble.get_model_branch(b_name)
        assert isinstance(branch, torch.nn.Module)
        
    with pytest.raises(ValueError):
        sample_ensemble.get_model_branch("invalid_branch_name")


def test_parameter_counting(sample_ensemble):
    """Verify total and trainable parameter counts are positive integers."""
    total = sample_ensemble.get_total_params()
    trainable = sample_ensemble.get_trainable_params()
    
    assert total > 1_000_000
    assert trainable == total  # Since none are frozen initially


def test_onnx_export(sample_ensemble, tmp_path):
    """Verify exporting PyTorch ensemble to ONNX format produces a valid .onnx file."""
    out_file = tmp_path / "test_ensemble.onnx"
    export_to_onnx(
        model=sample_ensemble,
        output_path=out_file,
        input_shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        verbose=False
    )
    
    assert out_file.exists()
    assert out_file.stat().st_size > 0
