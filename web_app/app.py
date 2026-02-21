import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import sys
import timm

# Add project root to path
import os
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
BASE_DIR = PROJECT_ROOT

# Import project modules
from config import DEVICE, NUM_CLASSES, CLASS_NAMES, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE

# --- CONFIGURATION ---
MODEL_ACCURACIES = {
    'Ensemble (All Models)': '99.05%',
    'ConvNeXt V2': '99.05%',
    'MedNeXt (ConvNeXt-Tiny)': '98.66%',
    'ConvNeXt (Best Individual)': '98.42%',
    'DeiT (Vision Transformer)': '97.80%',
    'ResNet-50 (Baseline)': '97.50%',
}

# --- PAGE SETUP ---
st.set_page_config(
    page_title="AI Pathologist - Live Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def apply_clahe(image_pil):
    """Apply CLAHE to a PIL image."""
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced_rgb)

@st.cache_resource
def load_model(model_name):
    """Load individual model."""
    try:
        if model_name == 'convnext':
            model = timm.create_model('convnext_tiny', pretrained=False, num_classes=NUM_CLASSES)
            ckpt_path = BASE_DIR / "outputs" / "convnext" / "best_convnext_model.pth"
        elif model_name == 'vit':
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
            ckpt_path = BASE_DIR / "outputs" / "vit_light" / "best_vit_model.pth"
        elif model_name == 'efficientnet':
            model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=NUM_CLASSES)
            ckpt_path = BASE_DIR / "outputs" / "checkpoints" / "best_effnet_model.pth"
        elif model_name == 'resnet':
            model = timm.create_model('resnet50', pretrained=False, num_classes=NUM_CLASSES)
            ckpt_path = BASE_DIR / "outputs" / "checkpoints" / "best_resnet_model.pth"
        elif model_name == 'deit':
            model = timm.create_model('deit_small_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
            ckpt_path = BASE_DIR / "outputs" / "deit_small" / "best_deit_model.pth"
        elif model_name == 'mednext':
            model = timm.create_model('convnext_tiny', pretrained=False, num_classes=NUM_CLASSES)
            ckpt_path = BASE_DIR / "outputs" / "mednext" / "best_mednext_model.pth"
        elif model_name == 'convnextv2':
            model = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=NUM_CLASSES)
            ckpt_path = BASE_DIR / "outputs" / "convnextv2" / "best_convnextv2_model.pth"
        else:
            return None
        
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # Load with strict=False to handle missing keys
            model.load_state_dict(state_dict, strict=False)
            model.to(DEVICE)
            model.eval()
            return model
    except Exception as e:
        st.warning(f"Failed to load {model_name}: {str(e)[:100]}")
        return None
    
    return None

def get_ensemble_prediction(image_tensor):
    """Get ensemble prediction from all models."""
    weights = {
        'convnextv2': 1.2,
        'mednext': 1.1,
        'deit': 1.0,
        'resnet': 1.0,
    }
    
    all_probs = []
    for model_name, weight in weights.items():
        model = load_model(model_name)
        if model is not None:
            with torch.no_grad():
                logits = model(image_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                all_probs.append(probs * weight)
    
    if all_probs:
        ensemble_probs = np.sum(all_probs, axis=0)
        ensemble_probs = ensemble_probs / ensemble_probs.sum()
        return ensemble_probs
    return None

def get_prediction(image_tensor, model_choice):
    """Get prediction based on selected model."""
    if model_choice == 'Ensemble (All Models)':
        return get_ensemble_prediction(image_tensor)
    
    model_map = {
        'ConvNeXt V2': 'convnextv2',
        'MedNeXt (ConvNeXt-Tiny)': 'mednext',
        'ConvNeXt (Best Individual)': 'convnext',
        'DeiT (Vision Transformer)': 'deit',
        'ResNet-50 (Baseline)': 'resnet',
    }
    
    model_name = model_map.get(model_choice)
    if model_name:
        model = load_model(model_name)
        if model is not None:
            with torch.no_grad():
                logits = model(image_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                return probs
    return None

def preprocess_for_model(image_pil):
    """Standard preprocessing pipeline."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image_pil).unsqueeze(0).to(DEVICE)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("Control Panel")
    
    st.markdown("### 🤖 Model Selection")
    model_choice = st.selectbox(
        "Choose AI Architecture",
        list(MODEL_ACCURACIES.keys())
    )
    
    st.markdown(f"""
    <div class="metric-container">
        <h4 style="margin:0">Model Accuracy</h4>
        <h2 style="color:#1E88E5; margin:0">{MODEL_ACCURACIES[model_choice]}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Visualization")
    show_clahe = st.checkbox("Show Enhancement (CLAHE)", value=True)
    
    st.markdown("---")
    st.info("**Research Day 2026**\\nAutomated Liver Fibrosis Staging Project")

# --- MAIN UI ---
st.markdown('<div class="main-header">ALS: Automated Liver Staging</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Neural Networks for Histopathology Analysis</div>', unsafe_allow_html=True)

# Input Section
col_upload, col_preview = st.columns([1, 1])

with col_upload:
    st.markdown("### 1. Upload Biopsy Slide")
    uploaded_file = st.file_uploader("Drag & Drop Image Here", type=['png', 'jpg', 'jpeg', 'tif'])

image_pil = None
if uploaded_file:
    image_pil = Image.open(uploaded_file).convert('RGB')
    
    with col_preview:
        st.markdown("### 2. Image Preprocessing")
        if show_clahe:
            c1, c2 = st.columns(2)
            with c1:
                st.image(image_pil, caption="Original Raw Input", use_column_width=True)
            with c2:
                with st.spinner("Applying CLAHE..."):
                    processed_img = apply_clahe(image_pil)
                st.image(processed_img, caption="CLAHE Enhanced", use_column_width=True)
        else:
            st.image(image_pil, caption="Input Biopsy Image", width=400)

# Analysis Section
if image_pil:
    st.markdown("---")
    st.markdown("### 3. Diagnostic Results")
    
    start_btn = st.button("Run AI Analysis", type="primary", use_container_width=True)
    
    if start_btn:
        with st.spinner(f"Running Inference with {model_choice}..."):
            # Apply CLAHE for model input
            input_image_pil = apply_clahe(image_pil)
            input_tensor = preprocess_for_model(input_image_pil)
            
            # Predict
            probs = get_prediction(input_tensor, model_choice)
            
            if probs is not None:
                # Parse results
                pred_idx = np.argmax(probs)
                pred_label = CLASS_NAMES[pred_idx]
                confidence = probs[pred_idx]
                
                # Determine color based on stage
                stage_colors = {
                    'F0': '#4CAF50',
                    'F1': '#8BC34A',
                    'F2': '#FFC107',
                    'F3': '#FF9800',
                    'F4': '#F44336'
                }
                res_color = stage_colors.get(pred_label, '#2196F3')
                
                # Result Display
                r_col1, r_col2 = st.columns([1, 2])
                
                with r_col1:
                    st.markdown(f"""
                    <div style="background-color: {res_color}; padding: 20px; border-radius: 15px; color: white; text-align: center;">
                        <h3 style="margin:0">Predicted Stage</h3>
                        <h1 style="font-size: 4rem; margin:0">{pred_label}</h1>
                        <p style="margin:0; opacity: 0.9">Confidence: {confidence*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with r_col2:
                    # Plotly Chart
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(data=[go.Bar(
                        x=CLASS_NAMES,
                        y=probs,
                        marker_color=[stage_colors.get(c, '#ccc') for c in CLASS_NAMES],
                        text=[f"{p*100:.1f}%" for p in probs],
                        textposition='auto',
                    )])
                    
                    fig.update_layout(
                        title="Confidence Distribution Across Stages",
                        yaxis_title="Probability",
                        xaxis_title="Fibrosis Stage",
                        yaxis_range=[0, 1],
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Interpretation
                st.info(f"**Interpretation:** The model has identified features consistent with **Stage {pred_label}** fibrosis with **{confidence*100:.1f}%** certainty.")
            else:
                st.error("Failed to load model. Please check model checkpoints.")

else:
    st.markdown("---")
    st.caption("Upload an image to start the analysis.")
