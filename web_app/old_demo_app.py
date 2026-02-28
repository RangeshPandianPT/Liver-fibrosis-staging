import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import sys
import os

# Add project root to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Import project modules
from src.models.ensemble import SoftVotingEnsemble
from config import DEVICE, NUM_CLASSES, CLASS_NAMES, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE

# --- CONFIGURATION ---
MODEL_ACCURACIES = {
    'Ensemble (Recommended)': '98.26%',
    'ConvNeXt (Best Individual)': '98.42%',
    'Vision Transformer (ViT-B/16)': '97.47%',
    'EfficientNet-V2': '96.60%',
    'ResNet50': '91.30%',
    'DeiT-Small': '85.53%'
}

# --- PAGE SETUP ---
st.set_page_config(
    page_title="AI Pathologist - Live Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
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
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def apply_clahe(image_pil):
    """Apply CLAHE to a PIL image (for visualization only)."""
    # Convert PIL to OpenCV format (RGB -> BGR)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Convert to LAB
    lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    l_enhanced = clahe.apply(l)
    
    # Merge and convert back
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Convert back to RGB for display/processing
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced_rgb)

@st.cache_resource
def load_all_models():
    """Load the ensemble model containing all sub-models."""
    ensemble = SoftVotingEnsemble(num_classes=NUM_CLASSES, pretrained=False)
    
    # Checkpoints map
    checkpoints = {
        'resnet50': BASE_DIR / "outputs" / "checkpoints" / "best_resnet_model.pth",
        'efficientnet': BASE_DIR / "outputs" / "checkpoints" / "best_effnet_model.pth",
        'vit': BASE_DIR / "outputs" / "vit_light" / "best_vit_model.pth",
        'deit': BASE_DIR / "outputs" / "deit_small" / "best_deit_model.pth",
        'convnext': BASE_DIR / "outputs" / "convnext" / "best_convnext_model.pth"
    }
    
    status_log = []
    
    for name, path in checkpoints.items():
        if path.exists():
            try:
                branch = ensemble.get_model_branch(name)
                ckpt = torch.load(path, map_location=DEVICE)
                
                # Handle state dict structure
                state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                branch.load_state_dict(state_dict)
                branch.to(DEVICE)
                branch.eval()
            except Exception as e:
                status_log.append(f"⚠️ Failed to load {name}: {e}")
                ensemble.weights[name] = 0.0 # Disable invalid model
        else:
            status_log.append(f"⚠️ Checkpoint not found: {name}")
            ensemble.weights[name] = 0.0
            
    ensemble.to(DEVICE)
    ensemble.eval()
    return ensemble, status_log

def get_prediction(model, image_tensor, model_choice):
    """Get prediction based on selected model."""
    with torch.no_grad():
        # Get all logits
        ensemble_logits, individual_logits = model(image_tensor, return_individual=True)
        
        if model_choice == 'Ensemble (Recommended)':
            logits = ensemble_logits
        elif model_choice == 'ConvNeXt (Best Individual)':
            logits = individual_logits.get('convnext', ensemble_logits)
        elif model_choice == 'Vision Transformer (ViT-B/16)':
            logits = individual_logits['vit']
        elif model_choice == 'EfficientNet-V2':
            logits = individual_logits['efficientnet']
        elif model_choice == 'ResNet50':
            logits = individual_logits['resnet50']
        elif model_choice == 'DeiT-Small':
            logits = individual_logits.get('deit', ensemble_logits)
        else:
            logits = ensemble_logits # Default
            
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

def preprocess_for_model(image_pil):
    """Standard preprocessing pipeline."""
    from torchvision import transforms
    
    # Use 224x224 as standard for these models
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
    st.info("**Research Day 2026**\nAutomated Liver Fibrosis Staging Project")

# --- MAIN UI ---
st.markdown('<div class="main-header">ALS: Automated Liver Staging</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Neural Networks for Histopathology Analysis</div>', unsafe_allow_html=True)

# 1. Load System
with st.spinner("Initializing Neural Networks..."):
    ensemble_model, logs = load_all_models()

if logs:
    with st.expander("⚠️ System Warnings"):
        for log in logs:
            st.warning(log)

# 2. Input Section
col_upload, col_preview = st.columns([1, 1])

with col_upload:
    st.markdown("### 1. Upload Biopsy Slide")
    uploaded_file = st.file_uploader("Drag & Drop Image Here", type=['png', 'jpg', 'jpeg', 'tif'])
    
    # Add sample button if needed later
    # if st.button("Load Sample Image"):
    #     uploaded_file = "sample.jpg"

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

# 3. Analysis Section
if image_pil:
    st.markdown("---")
    st.markdown("### 3. Diagnostic Results")
    
    start_btn = st.button("Run AI Analysis", type="primary", use_container_width=True)
    
    if start_btn:
        with st.spinner(f"Running Inference with {model_choice}..."):
            # Prepare input (CLAHE is applied within the pipeline effectively if trained on it, 
            # but usually models take normalized tensors. 
            # NOTE: If models were trained on CLAHE images, we MUST pass CLAHE images.
            # prepare_dataset.py saves CLAHE images to disk. Training loads them. 
            # So models EXPECT CLAHE images.
            # We must apply CLAHE before tensor conversion!)
            
            # Apply CLAHE for model input
            input_image_pil = apply_clahe(image_pil)
            input_tensor = preprocess_for_model(input_image_pil)
            
            # Predict
            probs = get_prediction(ensemble_model, input_tensor, model_choice)
            
            # Parse results
            pred_idx = np.argmax(probs)
            pred_label = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx]
            
            # Determine color based on stage
            stage_colors = {
                'F0': '#4CAF50', # Green
                'F1': '#8BC34A',
                'F2': '#FFC107', # Amber
                'F3': '#FF9800', # Orange
                'F4': '#F44336'  # Red
            }
            res_color = stage_colors.get(pred_label, '#2196F3')
            
            # --- Result Display ---
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
                
            # Interpretation (Optional)
            st.info(f"**Interpretation:** The model has identified features consistent with **Stage {pred_label}** fibrosis with **{confidence*100:.1f}%** certainty.")

else:
    st.markdown("---")
    st.caption("Upload an image to start the analysis.")
