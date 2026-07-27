"""
Production FastAPI Backend for Automated Liver Fibrosis Staging Platform.

Provides RESTful endpoints for clinical integration, EHR/PACS communication,
real-time uncertainty evaluation, and visual attention heatmap generation.
"""
import io
import base64
from typing import List, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import CLASS_NAMES, DEVICE
from src.models import SoftVotingEnsemble
from src.preprocessing import preprocess_image
from src.xai import AdvancedXAIExplainer, UncertaintyEstimator
from src.api.schemas import (
    SlidePredictionResult,
    BatchPredictionResponse,
    HealthResponse,
    XAIResponse
)

# Initialize FastAPI App
app = FastAPI(
    title="Automated Liver Staging (ALS) Clinical REST API",
    description="Enterprise API for Histological Liver Fibrosis Staging (F0-F4) with XAI & Uncertainty Scoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model Cache
_model_cache = {}


def get_model() -> SoftVotingEnsemble:
    """Retrieve or initialize the cached ensemble model."""
    if "ensemble" not in _model_cache:
        model = SoftVotingEnsemble(pretrained=False)
        model.eval()
        model.to(DEVICE)
        _model_cache["ensemble"] = model
    return _model_cache["ensemble"]


def load_image_tensor(file_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes into normalized PyTorch tensor."""
    try:
        img_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        tensor = preprocess_image(img_np, is_training=False)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.to(DEVICE)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file or format: {str(e)}"
        )


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check service health and GPU acceleration status."""
    model = get_model()
    return HealthResponse(
        status="healthy",
        gpu_available=(DEVICE == "cuda"),
        device=DEVICE,
        model_loaded=("ensemble" in _model_cache),
        model_version="1.0.0-ensemble",
        total_parameters=model.get_total_params()
    )


@app.post("/api/v1/predict", response_model=SlidePredictionResult, tags=["Inference"])
async def predict_slide(file: UploadFile = File(..., description="Histological slide image (PNG/JPEG/TIFF)")):
    """
    Classify a single histological slide into fibrosis stages F0-F4.
    Returns complete probability distribution, entropy, margin, and clinical recommendations.
    """
    contents = await file.read()
    tensor = load_image_tensor(contents)
    model = get_model()

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    analysis = UncertaintyEstimator.analyze_prediction(probs)
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return SlidePredictionResult(
        filename=file.filename or "unknown_slide.png",
        predicted_class=analysis["predicted_class"],
        predicted_prob=analysis["predicted_prob"],
        secondary_class=analysis["secondary_class"],
        secondary_prob=analysis["secondary_prob"],
        probabilities=prob_dict,
        entropy=analysis["entropy"],
        margin=analysis["margin"],
        alert_triggered=analysis["alert_triggered"],
        status=analysis["status"],
        recommendation=analysis["recommendation"],
        reasons=analysis["reasons"]
    )


@app.post("/api/v1/batch-predict", response_model=BatchPredictionResponse, tags=["Inference"])
async def batch_predict_slides(files: List[UploadFile] = File(..., description="List of slide images to evaluate")):
    """
    Evaluate a batch of histological slides and compute cohort-level summary statistics.
    """
    results = []
    failed_count = 0
    stage_counts = {c: 0 for c in CLASS_NAMES}
    alert_count = 0

    model = get_model()

    for file in files:
        try:
            contents = await file.read()
            tensor = load_image_tensor(contents)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()

            analysis = UncertaintyEstimator.analyze_prediction(probs)
            prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

            res = SlidePredictionResult(
                filename=file.filename or "unknown_slide.png",
                predicted_class=analysis["predicted_class"],
                predicted_prob=analysis["predicted_prob"],
                secondary_class=analysis["secondary_class"],
                secondary_prob=analysis["secondary_prob"],
                probabilities=prob_dict,
                entropy=analysis["entropy"],
                margin=analysis["margin"],
                alert_triggered=analysis["alert_triggered"],
                status=analysis["status"],
                recommendation=analysis["recommendation"],
                reasons=analysis["reasons"]
            )
            results.append(res)
            stage_counts[analysis["predicted_class"]] += 1
            if analysis["alert_triggered"]:
                alert_count += 1
        except Exception:
            failed_count += 1

    total = len(files)
    successful = len(results)

    summary = {
        "cohort_size": successful,
        "stage_distribution": stage_counts,
        "ambiguity_alert_rate": float(alert_count / max(successful, 1)),
        "total_alerts": alert_count
    }

    return BatchPredictionResponse(
        total_slides=total,
        successful_slides=successful,
        failed_slides=failed_count,
        results=results,
        summary=summary
    )


@app.post("/api/v1/explain", response_model=XAIResponse, tags=["Explainability"])
async def explain_slide(
    file: UploadFile = File(..., description="Histological slide image"),
    target_class: Optional[str] = Form(None, description="Target stage (e.g. F2). Defaults to top predicted stage."),
    method: str = Form("eigencam", description="XAI method: 'eigencam' or 'gradcam'")
):
    """
    Generate visual attention heatmap (EigenCAM / Grad-CAM) for clinical validation.
    Returns Base64 encoded PNG overlay image.
    """
    contents = await file.read()
    tensor = load_image_tensor(contents)
    model = get_model()

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    if target_class and target_class.upper() in CLASS_NAMES:
        target_idx = CLASS_NAMES.index(target_class.upper())
    else:
        target_idx = int(np.argmax(probs))

    explainer = AdvancedXAIExplainer(model, device=DEVICE)
    heatmap = explainer.generate_cam(tensor, target_class=target_idx, method=method)
    overlay_rgb = explainer.create_overlay(tensor, heatmap)

    # Encode to Base64 PNG
    pil_img = Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    entropy = UncertaintyEstimator.compute_entropy(probs)

    return XAIResponse(
        filename=file.filename or "unknown_slide.png",
        target_class=CLASS_NAMES[target_idx],
        method_used=method.lower(),
        heatmap_base64=b64_str,
        entropy=entropy
    )
