"""
Pydantic Schemas for Liver Fibrosis Staging REST API.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class SlidePredictionResult(BaseModel):
    """Result of single histological slide classification."""
    filename: str = Field(..., description="Name of analyzed slide file")
    predicted_class: str = Field(..., description="Top predicted fibrosis stage (F0-F4)")
    predicted_prob: float = Field(..., description="Top prediction confidence (0.0 to 1.0)")
    secondary_class: str = Field(..., description="Second highest fibrosis stage prediction")
    secondary_prob: float = Field(..., description="Second highest probability")
    probabilities: Dict[str, float] = Field(..., description="Full class probability distribution")
    entropy: float = Field(..., description="Shannon predictive entropy score (0.0 to 1.0)")
    margin: float = Field(..., description="Confidence margin between Top-1 and Top-2")
    alert_triggered: bool = Field(..., description="Whether diagnostic ambiguity alert was triggered")
    status: str = Field(..., description="Clinical diagnostic status banner")
    recommendation: str = Field(..., description="Clinical next-step recommendation")
    reasons: List[str] = Field(default_factory=list, description="Reasons for ambiguity alerts if any")


class BatchPredictionResponse(BaseModel):
    """Response for batch slide prediction endpoint."""
    total_slides: int = Field(..., description="Total number of processed slides")
    successful_slides: int = Field(..., description="Number of successfully classified slides")
    failed_slides: int = Field(default=0, description="Number of failed slide evaluations")
    results: List[SlidePredictionResult] = Field(..., description="List of individual slide results")
    summary: Dict[str, Any] = Field(..., description="Aggregate cohort distribution summary")


class HealthResponse(BaseModel):
    """System and Model Health Check Response."""
    status: str = Field(default="healthy", description="Service operating status")
    gpu_available: bool = Field(..., description="Whether CUDA GPU acceleration is enabled")
    device: str = Field(..., description="Active compute device ('cuda' or 'cpu')")
    model_loaded: bool = Field(..., description="Whether ensemble weights are loaded in memory")
    model_version: str = Field(default="1.0.0", description="Model architecture version")
    total_parameters: int = Field(..., description="Total parameter count in active model")


class XAIResponse(BaseModel):
    """Response for Explainable AI attention heatmap generation."""
    filename: str = Field(..., description="Slide filename")
    target_class: str = Field(..., description="Fibrosis stage visualized in heatmap")
    method_used: str = Field(..., description="XAI method used ('eigencam' or 'gradcam')")
    heatmap_base64: str = Field(..., description="Base64 encoded PNG heatmap overlay image")
    entropy: float = Field(..., description="Diagnostic entropy score")
