"""
Integration Tests for FastAPI REST API Endpoints.
"""
import io
import pytest
from PIL import Image
import numpy as np

try:
    from fastapi.testclient import TestClient
    from src.api.main import app
    client = TestClient(app)
except ImportError:
    client = None


@pytest.fixture
def dummy_image_bytes():
    """Generate a synthetic 384x384 RGB slide image as in-memory bytes."""
    img_array = np.random.randint(100, 200, size=(384, 384, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


@pytest.mark.skipif(client is None, reason="FastAPI TestClient not available")
def test_health_endpoint():
    """Test GET /api/v1/health status response."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gpu_available" in data
    assert "total_parameters" in data


@pytest.mark.skipif(client is None, reason="FastAPI TestClient not available")
def test_predict_endpoint(dummy_image_bytes):
    """Test POST /api/v1/predict with uploaded slide image."""
    files = {"file": ("test_slide.png", dummy_image_bytes, "image/png")}
    response = client.post("/api/v1/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_slide.png"
    assert "predicted_class" in data
    assert "entropy" in data
    assert "margin" in data
    assert "recommendation" in data
    assert len(data["probabilities"]) == 5


@pytest.mark.skipif(client is None, reason="FastAPI TestClient not available")
def test_batch_predict_endpoint(dummy_image_bytes):
    """Test POST /api/v1/batch-predict with multiple slides."""
    files = [
        ("files", ("slide_1.png", dummy_image_bytes, "image/png")),
        ("files", ("slide_2.png", dummy_image_bytes, "image/png"))
    ]
    response = client.post("/api/v1/batch-predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["total_slides"] == 2
    assert data["successful_slides"] == 2
    assert len(data["results"]) == 2
    assert "cohort_size" in data["summary"]


@pytest.mark.skipif(client is None, reason="FastAPI TestClient not available")
def test_explain_endpoint(dummy_image_bytes):
    """Test POST /api/v1/explain heatmap generation."""
    files = {"file": ("test_slide.png", dummy_image_bytes, "image/png")}
    data_payload = {"method": "eigencam"}
    response = client.post("/api/v1/explain", files=files, data=data_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["method_used"] == "eigencam"
    assert len(data["heatmap_base64"]) > 100
    assert "entropy" in data
