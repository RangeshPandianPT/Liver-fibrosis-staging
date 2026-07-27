"""
Unit Tests for Explainable AI (XAI) and Clinical Uncertainty Estimator.
"""
import pytest
import numpy as np

from src.xai import UncertaintyEstimator


def test_compute_entropy():
    """Verify Shannon entropy calculation for sharp vs uniform distributions."""
    # Sharp, confident distribution -> low entropy
    sharp_probs = np.array([0.96, 0.01, 0.01, 0.01, 0.01])
    low_ent = UncertaintyEstimator.compute_entropy(sharp_probs)
    assert 0.0 <= low_ent < 0.2

    # Uniform, ambiguous distribution -> maximum entropy (~1.0)
    uniform_probs = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    high_ent = UncertaintyEstimator.compute_entropy(uniform_probs)
    assert 0.95 <= high_ent <= 1.0


def test_compute_margin():
    """Verify confidence margin calculation between Top-1 and Top-2."""
    probs = np.array([0.60, 0.35, 0.03, 0.01, 0.01])
    margin = UncertaintyEstimator.compute_margin(probs)
    assert pytest.approx(margin, 0.001) == 0.25


def test_analyze_prediction_confident():
    """Verify confident diagnostic analysis does not trigger clinical alerts."""
    probs = np.array([0.90, 0.05, 0.03, 0.01, 0.01])
    res = UncertaintyEstimator.analyze_prediction(probs)
    
    assert res["predicted_class"] == "F0"
    assert res["secondary_class"] == "F1"
    assert res["alert_triggered"] is False
    assert "Confidential" in res["status"] or "Confident" in res["status"]


def test_analyze_prediction_ambiguous_alert():
    """Verify borderline predictions trigger clinical pathology consultation alert."""
    probs = np.array([0.02, 0.46, 0.48, 0.02, 0.02])
    res = UncertaintyEstimator.analyze_prediction(probs)
    
    assert res["predicted_class"] == "F2"
    assert res["secondary_class"] == "F1"
    assert res["alert_triggered"] is True
    assert "WARNING" in res["status"]
    assert len(res["reasons"]) > 0
