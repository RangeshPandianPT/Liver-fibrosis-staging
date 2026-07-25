import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import sys
import timm
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Add project root to path
import os
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
BASE_DIR = PROJECT_ROOT

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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
        'convnext': 1.1,
        'resnet': 0.8,
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

def get_target_layer(model, model_name):
    """Get target layer for Grad-CAM."""
    if 'convnext' in model_name or 'mednext' in model_name:
        return [model.stages[-1].blocks[-1]]
    elif 'resnet' in model_name:
        return [model.layer4[-1]]
    return None

def generate_clinical_pdf(image_pil, clahe_img, cam_image, pred_label, confidence, probs, model_choice):
    """Generate a 2-page Clinical Diagnostic Report PDF in memory using Matplotlib."""
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # Page 1: Clinical Summary & Diagnoses
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        # Header
        fig.text(0.5, 0.93, "AUTOMATED LIVER STAGING (ALS)", fontsize=20, ha='center', fontweight='bold', color='#1E88E5')
        fig.text(0.5, 0.89, "Clinical Histology Diagnostic Report", fontsize=14, ha='center', color='#424242')
        fig.text(0.5, 0.86, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Model: {model_choice}", fontsize=10, ha='center', color='gray')
        
        # Divider
        fig.add_artist(plt.Line2D((0.1, 0.9), (0.84, 0.84), color='#1E88E5', linewidth=2))
        
        # Diagnostic Box
        stage_colors = {'F0': '#4CAF50', 'F1': '#8BC34A', 'F2': '#FFC107', 'F3': '#FF9800', 'F4': '#F44336'}
        box_color = stage_colors.get(pred_label, '#2196F3')
        
        rect = plt.Rectangle((0.15, 0.67), 0.7, 0.14, facecolor='#F5F5F5', edgecolor=box_color, linewidth=3)
        fig.add_artist(rect)
        fig.text(0.5, 0.76, "PRIMARY DIAGNOSTIC FINDING", fontsize=12, ha='center', fontweight='bold', color='#424242')
        fig.text(0.5, 0.71, f"Stage {pred_label} Fibrosis ({confidence*100:.1f}% Confidence)", fontsize=18, ha='center', fontweight='bold', color=box_color)
        
        # Probabilities Table
        fig.text(0.5, 0.61, "Stage Probability Distribution", fontsize=13, ha='center', fontweight='bold', color='#1E88E5')
        
        table_data = [[c, f"{p*100:.2f}%"] for c, p in zip(CLASS_NAMES, probs)]
        ax = fig.add_axes([0.2, 0.38, 0.6, 0.20])
        ax.axis('off')
        table = ax.table(cellText=table_data, colLabels=['Fibrosis Stage', 'Model Probability'], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#1E88E5')
                cell.set_text_props(color='white', fontweight='bold')
            elif CLASS_NAMES[i-1] == pred_label:
                cell.set_facecolor('#E8F5E9')
                cell.set_text_props(fontweight='bold')
                
        # Clinical Notes & Sign-off
        fig.text(0.1, 0.28, "Clinical Notes & Interpretation:", fontsize=12, fontweight='bold', color='#424242')
        note_text = (
            f"The AI Pathologist system evaluated the submitted biopsy slide using {model_choice}. "
            f"The neural network identified morphological features most consistent with Stage {pred_label}. "
            "This automated screening report is designed to assist pathologists in histological evaluation."
        )
        fig.text(0.1, 0.22, note_text, fontsize=10, color='#616161', wrap=True)
        
        fig.add_artist(plt.Line2D((0.1, 0.45), (0.10, 0.10), color='black', linewidth=1))
        fig.text(0.1, 0.07, "Pathologist Signature & Date", fontsize=10, color='gray')
        
        fig.add_artist(plt.Line2D((0.55, 0.9), (0.10, 0.10), color='black', linewidth=1))
        fig.text(0.55, 0.07, "Laboratory / Institution ID", fontsize=10, color='gray')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Visual Evidence (Images & Grad-CAM)
        fig2 = plt.figure(figsize=(8.5, 11))
        fig2.patch.set_facecolor('white')
        fig2.text(0.5, 0.93, "VISUAL EVIDENCE & XAI HEATMAPS", fontsize=16, ha='center', fontweight='bold', color='#1E88E5')
        fig2.add_artist(plt.Line2D((0.1, 0.9), (0.89, 0.89), color='#1E88E5', linewidth=2))
        
        # Plot 3 images: Original, CLAHE, and Grad-CAM
        ax1 = fig2.add_axes([0.1, 0.55, 0.35, 0.30])
        ax1.imshow(image_pil)
        ax1.set_title("1. Original Biopsy Input", fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig2.add_axes([0.55, 0.55, 0.35, 0.30])
        ax2.imshow(clahe_img)
        ax2.set_title("2. CLAHE Enhanced Input", fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig2.add_axes([0.25, 0.18, 0.5, 0.32])
        if cam_image is not None:
            ax3.imshow(cam_image)
            ax3.set_title("3. Explainable AI (Grad-CAM) Focus", fontsize=11, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, "Grad-CAM Heatmap N/A\nfor selected model architecture", ha='center', va='center', fontsize=12)
        ax3.axis('off')
        
        fig2.text(0.5, 0.08, "Page 2 of 2 | Automated Liver Staging Research Platform", fontsize=9, ha='center', color='gray')
        
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
        
    buffer.seek(0)
    return buffer.getvalue()

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

tab_single, tab_bench, tab_specs = st.tabs([
    "🔬 Single Slide Diagnosis & Report", 
    "⚖️ Multi-Model Benchmarking", 
    "📊 Model Performance & Specs"
])

# --- TAB 1: SINGLE SLIDE DIAGNOSIS ---
with tab_single:
    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        st.markdown("### 1. Upload Biopsy Slide")
        uploaded_file = st.file_uploader("Drag & Drop Image Here", type=['png', 'jpg', 'jpeg', 'tif'], key="single_upload")

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
        
        start_btn = st.button("Run AI Analysis", type="primary", use_container_width=True, key="run_single")
        
        if start_btn or st.session_state.get("single_analyzed", False):
            st.session_state["single_analyzed"] = True
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

                    # Grad-CAM Generation
                    cam_image = None
                    model_map = {
                        'ConvNeXt V2': 'convnextv2',
                        'MedNeXt (ConvNeXt-Tiny)': 'mednext',
                        'ConvNeXt (Best Individual)': 'convnext',
                        'DeiT (Vision Transformer)': 'deit',
                        'ResNet-50 (Baseline)': 'resnet',
                    }
                    # Use ConvNeXtV2 as proxy for Ensemble visualization
                    cam_model_name = 'convnextv2' if model_choice == 'Ensemble (All Models)' else model_map.get(model_choice)
                    
                    if cam_model_name and cam_model_name != 'deit':
                        cam_model = load_model(cam_model_name)
                        target_layers = get_target_layer(cam_model, cam_model_name)
                        if target_layers:
                            try:
                                resized_img = input_image_pil.resize((224, 224))
                                rgb_img = np.float32(resized_img) / 255.0
                                
                                with GradCAM(model=cam_model, target_layers=target_layers) as cam:
                                    targets = [ClassifierOutputTarget(pred_idx)]
                                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                                    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                            except Exception as e:
                                st.warning(f"Grad-CAM generation failed: {e}")

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
                        <div style="background-color: {res_color}; padding: 20px; border-radius: 15px; color: white; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
                            <h3 style="margin:0">Predicted Stage</h3>
                            <h1 style="font-size: 4rem; margin:0">{pred_label}</h1>
                            <p style="margin:0; opacity: 0.9">Confidence: {confidence*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with r_col2:
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
                            height=280,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Clinical Diagnostic Report PDF Download
                    st.markdown("#### 📑 Clinical Documentation")
                    pdf_bytes = generate_clinical_pdf(
                        image_pil=image_pil,
                        clahe_img=input_image_pil,
                        cam_image=cam_image,
                        pred_label=pred_label,
                        confidence=confidence,
                        probs=probs,
                        model_choice=model_choice
                    )
                    st.download_button(
                        label="📥 Download Clinical Diagnostic Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"ALS_Diagnostic_Report_{pred_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    # XAI Section
                    st.markdown("### 🧠 Explainable AI (Grad-CAM)")
                    if cam_image is not None:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(input_image_pil, caption="Standard Biopsy Image", use_column_width=True)
                        with c2:
                            caption = "Grad-CAM Heatmap" if model_choice != 'Ensemble (All Models)' else "Ensemble Highlight (Proxy via ConvNeXt V2)"
                            st.image(cam_image, caption=caption, use_column_width=True)
                        st.info(f"**Interpretation:** The heatmap highlights the critical fibrotic areas the model focused on to make its **Stage {pred_label}** determination.")
                    elif model_choice == 'DeiT (Vision Transformer)':
                        st.info("Grad-CAM visualization is currently tailored for the CNN architectures. To see heatmaps, please select ConvNeXt, MedNeXt, ResNet, or Ensemble.")
                    else:
                        st.info(f"**Interpretation:** The model has identified features consistent with **Stage {pred_label}** fibrosis with **{confidence*100:.1f}%** certainty.")
                else:
                    st.error("Failed to load model. Please check model checkpoints.")

    else:
        st.markdown("---")
        st.caption("Upload an image in Section 1 to start the analysis.")

# --- TAB 2: MULTI-MODEL BENCHMARKING ---
with tab_bench:
    st.markdown("### ⚖️ Side-by-Side Architectural Benchmarking")
    st.write("Compare predictions, agreement rates, and confidence distribution across all trained architectures simultaneously.")
    
    if image_pil is None:
        st.warning("Please upload a biopsy slide in the **Single Slide Diagnosis** tab first to run multi-model benchmarking.")
    else:
        bench_btn = st.button("🚀 Run Full Model Benchmark on Current Slide", type="primary", use_container_width=True)
        if bench_btn:
            with st.spinner("Executing simultaneous inference across 5 models + Ensemble..."):
                input_image_pil = apply_clahe(image_pil)
                input_tensor = preprocess_for_model(input_image_pil)
                
                bench_results = []
                all_probs_dict = {}
                
                for m_name, m_acc in MODEL_ACCURACIES.items():
                    probs = get_prediction(input_tensor, m_name)
                    if probs is not None:
                        idx = np.argmax(probs)
                        pred_stage = CLASS_NAMES[idx]
                        conf = probs[idx]
                        all_probs_dict[m_name] = probs
                        bench_results.append({
                            "AI Architecture": m_name,
                            "Validation Accuracy": m_acc,
                            "Predicted Stage": pred_stage,
                            "Confidence": f"{conf*100:.2f}%",
                            "Raw Confidence": conf
                        })
                
                if bench_results:
                    import pandas as pd
                    df_bench = pd.DataFrame(bench_results)
                    
                    st.markdown("#### 📊 Comparative Diagnosis Summary")
                    st.dataframe(
                        df_bench.drop(columns=["Raw Confidence"]), 
                        use_container_width=True, 
                        hide_index=True
                    )
                    
                    # Agreement check
                    stages_predicted = [r["Predicted Stage"] for r in bench_results]
                    unique_stages = set(stages_predicted)
                    if len(unique_stages) == 1:
                        st.success(f"🤝 **Perfect Consensus:** All {len(bench_results)} AI models agree on **Stage {list(unique_stages)[0]}**!")
                    else:
                        st.warning(f"⚡ **Model Divergence Detected:** Models predicted multiple stages ({', '.join(unique_stages)}). Rely on Ensemble or ConvNeXt V2 as primary arbiters.")
                    
                    # Plot comparative chart
                    import plotly.graph_objects as go
                    fig_bench = go.Figure()
                    
                    stage_colors = {'F0': '#4CAF50', 'F1': '#8BC34A', 'F2': '#FFC107', 'F3': '#FF9800', 'F4': '#F44336'}
                    for stage_idx, stage_name in enumerate(CLASS_NAMES):
                        stage_probs = [all_probs_dict[m][stage_idx] for m in df_bench["AI Architecture"]]
                        fig_bench.add_trace(go.Bar(
                            name=stage_name,
                            x=df_bench["AI Architecture"],
                            y=stage_probs,
                            marker_color=stage_colors[stage_name]
                        ))
                        
                    fig_bench.update_layout(
                        title="Model Comparison - Stage Probability Distribution",
                        xaxis_title="Architecture",
                        yaxis_title="Probability",
                        barmode='stack',
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_bench, use_container_width=True)

# --- TAB 3: MODEL PERFORMANCE & SPECS ---
with tab_specs:
    st.markdown("### 📊 Clinical Validation & Research Highlights")
    st.write("Overview of the test set performance (1,265 samples) across the 5-stage liver fibrosis classification (F0 to F4).")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Best Individual Architecture", "ConvNeXt Tiny", "98.42% Acc")
    with col_m2:
        st.metric("Ensemble Performance", "98.26% / 99.05%", "0.9938 QWK")
    with col_m3:
        st.metric("Test Dataset Size", "1,265 Slides", "5 Fibrosis Stages")
        
    st.markdown("#### 🏆 Benchmark Table")
    perf_data = [
        {"Architecture": "Ensemble (Soft Voting)", "Accuracy": "98.26% - 99.05%", "Cohen's Kappa": "0.9938", "Key Strength": "Maximum clinical consensus & stability"},
        {"Architecture": "ConvNeXt Tiny / V2", "Accuracy": "98.42% - 99.05%", "Cohen's Kappa": "0.9793", "Key Strength": "Best individual spatial feature extractor"},
        {"Architecture": "MedNeXt (Medical Tuned)", "Accuracy": "98.66%", "Cohen's Kappa": "0.9810", "Key Strength": "Optimized for histological staining variances"},
        {"Architecture": "ResNet-50 (Baseline)", "Accuracy": "91.30% - 97.50%", "Cohen's Kappa": "0.8900", "Key Strength": "Standard CNN baseline comparison"},
        {"Architecture": "DeiT-Small (Transformer)", "Accuracy": "85.53% - 97.80%", "Cohen's Kappa": "0.8200", "Key Strength": "Global attention & long-range dependency tracking"}
    ]
    st.table(perf_data)
    
    st.markdown("#### 🔬 Diagnostic Visualizations from Training Outputs")
    # Dynamically check for existing evaluation plots in outputs/
    convnext_dir = BASE_DIR / "outputs" / "convnext"
    cm_img_path = convnext_dir / "convnext_confusion_matrix.png"
    roc_img_path = convnext_dir / "convnext_roc_curves.png"
    
    if cm_img_path.exists() and roc_img_path.exists():
        c_vis1, c_vis2 = st.columns(2)
        with c_vis1:
            st.image(str(cm_img_path), caption="ConvNeXt Confusion Matrix on Test Set", use_column_width=True)
        with c_vis2:
            st.image(str(roc_img_path), caption="ConvNeXt ROC Curves (Multi-Class AUC)", use_column_width=True)
    else:
        st.info("Evaluation plots can be generated by running `python report_scripts/generate_convnext_report.py` in your terminal.")

