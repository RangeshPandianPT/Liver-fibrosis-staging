"""
Unit Tests for Data Preprocessing and Image Normalization.
"""
import pytest
import numpy as np
import torch

from src.preprocessing import preprocess_image, denormalize
from config import IMAGE_SIZE


def test_preprocess_image_shapes():
    """Verify image preprocessing outputs correct PyTorch tensor shape (3, H, W)."""
    dummy_img = np.random.randint(0, 256, size=(400, 500, 3), dtype=np.uint8)
    tensor = preprocess_image(dummy_img, is_training=False)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32


def test_preprocess_image_normalization_range():
    """Verify preprocessed tensor values are appropriately normalized around ImageNet stats."""
    dummy_img = np.full((384, 384, 3), 128, dtype=np.uint8)
    tensor = preprocess_image(dummy_img, is_training=False)
    
    # Values should be roughly between -3 and +3
    assert tensor.min() > -5.0
    assert tensor.max() < 5.0


def test_denormalize_reversibility():
    """Verify denormalization returns an 8-bit unsigned RGB image array."""
    dummy_img = np.random.randint(50, 200, size=(384, 384, 3), dtype=np.uint8)
    tensor = preprocess_image(dummy_img, is_training=False)
    
    recovered = denormalize(tensor)
    assert isinstance(recovered, np.ndarray)
    assert recovered.shape == (384, 384, 3)
    assert recovered.dtype == np.uint8
    assert recovered.min() >= 0 and recovered.max() <= 255
