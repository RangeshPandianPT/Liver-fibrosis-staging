"""
Configuration settings for Liver Fibrosis Staging Pipeline
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "liver_images"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
GRADCAM_DIR = OUTPUT_DIR / "gradcam_heatmaps"

# Create output directories
for dir_path in [CHECKPOINT_DIR, METRICS_DIR, GRADCAM_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data settings
IMAGE_SIZE = 384
NUM_CLASSES = 5
CLASS_NAMES = ['F0', 'F1', 'F2', 'F3', 'F4']
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# CLAHE settings
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Model settings
PRETRAINED = True
FREEZE_BACKBONE = False

# Ensemble weights (equal by default)
ENSEMBLE_WEIGHTS = {
    'resnet50': 1.0,
    'efficientnet': 1.0,
    'vit': 1.0
}

# Training settings
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1

# OneCycleLR settings
MAX_LR = 1e-3
PCT_START = 0.3
DIV_FACTOR = 25
FINAL_DIV_FACTOR = 1000

# Grad-CAM settings
TOP_K_HEATMAPS = 5

# Device
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
except ImportError:
    pass
