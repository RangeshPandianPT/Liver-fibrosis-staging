import os
import sys
from pathlib import Path
import random

# Add project root to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
BASE_DIR = PROJECT_ROOT

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import timm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import DEVICE, NUM_CLASSES, CLASS_NAMES

def get_target_layer(model, model_name):
    if 'convnext' in model_name or 'mednext' in model_name:
        return [model.stages[-1].blocks[-1]]
    elif 'resnet' in model_name:
        return [model.layer4[-1]]
    return None

def load_model(model_name):
    try:
        if model_name == 'mednext':
            model = timm.create_model('convnext_tiny', pretrained=False, num_classes=NUM_CLASSES)
            ckpt_path = BASE_DIR / "outputs" / "mednext" / "best_mednext_model.pth"
        else:
            return None
        
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict, strict=False)
            model.to(DEVICE)
            model.eval()
            return model
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None
    return None

def preprocess_for_model(image_pil):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image_pil).unsqueeze(0).to(DEVICE)

def generate_figure(image_path, model, model_name, true_class_idx):
    image_pil = Image.open(image_path).convert('RGB')
    input_tensor = preprocess_for_model(image_pil)
    
    target_layers = get_target_layer(model, model_name)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
    
    resized_img = image_pil.resize((224, 224))
    rgb_img = np.float32(resized_img) / 255.0
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(resized_img)
    axes[0].set_title(f"Original Image (True: {CLASS_NAMES[true_class_idx]})")
    axes[0].axis('off')
    
    axes[1].imshow(cam_image)
    axes[1].set_title(f"Grad-CAM (Pred: {CLASS_NAMES[pred_idx]}, Conf: {probs[pred_idx]*100:.1f}%)")
    axes[1].axis('off')
    
    plt.tight_layout()
    output_path = BASE_DIR / "outputs" / f"gradcam_{CLASS_NAMES[true_class_idx]}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")

if __name__ == '__main__':
    model_name = 'mednext'
    model = load_model(model_name)
    if model is None:
        print("Model not loaded.")
        sys.exit(1)
        
    data_dir = BASE_DIR / "data" / "liver_images"
    for i, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if images:
                sample_img = images[0]
                generate_figure(sample_img, model, model_name, i)
