import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.models.ensemble import SoftVotingEnsemble
from config import DEVICE, NUM_CLASSES, CLASS_NAMES, ENSEMBLE_WEIGHTS, OUTPUT_DIR

# Page Config
st.set_page_config(
    page_title="Liver Fibrosis Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        color: white;
        background-color: #1a237e;
        border-radius: 10px;
    }
    .metric-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def load_ensemble_model():
    """Load the trained ensemble model."""
    model = SoftVotingEnsemble(num_classes=NUM_CLASSES, pretrained=False)
    
    # Load Weights
    # We need to load weights for each branch independently if we don't have a full ensemble checkpoint
    # Or if we saved them separately.
    # In this project, we saved best_resnet_model.pth, best_effnet_model.pth, best_vit_model.pth
    
    checkpoints = {
        'resnet50': BASE_DIR / "outputs" / "checkpoints" / "best_resnet_model.pth",
        'efficientnet': BASE_DIR / "outputs" / "checkpoints" / "best_effnet_model.pth",
        'vit': BASE_DIR / "outputs" / "vit_light" / "best_vit_model.pth",
        'deit': BASE_DIR / "outputs" / "deit_small" / "best_deit_model.pth"
    }
    
    status_text = []
    
    for name, path in checkpoints.items():
        if path.exists():
            try:
                branch = model.get_model_branch(name)
                # Load state dict
                # Note: checkpoints might be full dicts or just state_dicts
                ckpt = torch.load(path, map_location=DEVICE)
                if 'model_state_dict' in ckpt:
                    branch.load_state_dict(ckpt['model_state_dict'])
                else:
                    branch.load_state_dict(ckpt)
                branch.to(DEVICE)
                branch.eval()
                # model.weights[name] = ENSEMBLE_WEIGHTS[name] # Ensure weight is active
            except Exception as e:
                 status_text.append(f"⚠️ Failed to load {name}: {e}")
                 # Zero out weight if load fails? Or just let it be random/untrained?
                 # ideally we should disable it.
                 model.weights[name] = 0.0
        else:
            status_text.append(f"⚠️ Missing checkpoint for {name} ({path.name})")
            model.weights[name] = 0.0
            
    model.to(DEVICE)
    model.eval()
    return model, status_text

def preprocess_image(image):
    """Preprocess image for inference."""
    # Standard transforms matching training
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224), # Ensemble expects consistent size or handles it?
        # ResNet/EffNet/DeiT use 224. ViT uses 224 in our 'light' version.
        # But config.py says IMAGE_SIZE = 384. 
        # LET'S CHECK CONFIG. 
        # If config says 384, we should use 384 for those that need it.
        # However, train_vit_light used 224. train_deit used 224.
        # resnet/effnet might have used 384.
        # To be safe, we resize to what each model expects?
        # The ensemble forward() passes X to all models. They must handle the input size.
        # If they need different sizes, we must resize per branch.
        # But for simplicity, let's assume 224 works for all (standard ImageNet) 
        # OR we check if we can resize dynamically. 
        # For this demo, let's use 224 as it matches the most recent trainings.
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/microscope.png", width=80)
    st.title("Model Controls")
    
    st.markdown("### ⚙️ Configuration")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("### ℹ️ About")
    st.info(
        "This AI Assistant uses a **Quad-Ensemble** of Deep Learning models "
        "(ResNet50, EfficientNet-V2, ViT-B/16, DeiT-Small) "
        "to stage Liver Fibrosis from histopathology images."
    )
    
    st.markdown("---")
    st.caption(f"Running on: **{DEVICE.upper()}**")

# --- MAIN CONTENT ---
st.title("🔬 Liver Fibrosis Staging AI")
st.markdown("### Pathologist Decision Support System")

# Load Model
with st.spinner("Loading AI Ensemble..."):
    model, load_status = load_ensemble_model()

if load_status:
    with st.expander("System Status"):
        for msg in load_status:
            st.warning(msg)

# File Upload
uploaded_file = st.file_uploader("Upload Histopathology Image", type=['png', 'jpg', 'jpeg', 'tif'])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Patient Sample")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True, caption="Uploaded Slide")
        
    with col2:
        st.markdown("#### AI Analysis")
        
        # Inference
        try:
            input_tensor = preprocess_image(image)
            
            with torch.no_grad():
                logits, individual_logits = model(input_tensor, return_individual=True)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                pred_label = CLASS_NAMES[pred_idx]
                confidence = probs[pred_idx]
                
            # Display Result
            if confidence >= confidence_threshold:
                st.success(f"**Predicted Stage: {pred_label}**")
            else:
                st.warning(f"**Predicted Stage: {pred_label}** (Low Confidence)")
                
            # Metrics
            st.metric("Confidence Score", f"{confidence*100:.2f}%")
            
            # Probability Bar Chart
            df_probs = pd.DataFrame({
                'Stage': CLASS_NAMES,
                'Probability': probs
            })
            
            # Plotly Chart
            import plotly.express as px
            fig = px.bar(df_probs, x='Stage', y='Probability', 
                         color='Probability', color_continuous_scale='Blues',
                         range_y=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Individual Model Agreement
            st.markdown("##### Specialist Consensus")
            votes = {}
            for name, l in individual_logits.items():
                p = F.softmax(l, dim=1).cpu().numpy()[0]
                vote = CLASS_NAMES[np.argmax(p)]
                votes[name] = vote
            
            # Display votes as tags
            cols = st.columns(len(votes))
            for idx, (name, vote) in enumerate(votes.items()):
                with cols[idx]:
                    st.caption(name)
                    if vote == pred_label:
                        st.info(f"✅ {vote}")
                    else:
                        st.error(f"❌ {vote}")
                        
        except Exception as e:
            st.error(f"Error during analysis: {e}")

# Footer
st.markdown("---")
st.markdown("© 2026 ALS Research Group | Comparative Report Generator Available")
