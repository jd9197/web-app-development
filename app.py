import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import cv2
import requests
import os
from torchvision import transforms

# --- MEDICAL DASHBOARD CONFIGURATION ---
st.set_page_config(
    page_title="RP Clinical AI Explorer",
    page_icon="👁️",
    layout="wide",
)

# Custom CSS for a professional medical "Clean & Modern" look
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #007bff; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #007bff; color: white; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #0056b3; border: none; }
    .status-box { padding: 20px; border-radius: 10px; margin-top: 20px; border: 1px solid #ddd; }
    .clinical-header { color: #1e3d59; font-weight: 800; border-bottom: 2px solid #1e3d59; padding-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL SETTINGS ---
GITHUB_MODEL_URL = "https://github.com/jd9197/web-app-development/releases/download/v1.0/swin_retina_rp_deploy.pth"
MODEL_PATH = "model_weights.pth"

class SwinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone matches your training (Swin Tiny)
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

@st.cache_resource
def load_medical_model():
    # 1. Verification & Download Logic
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 1000000:
        os.remove(MODEL_PATH) # Delete corrupted file

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Initializing Clinical AI Engine... (Downloading Weights)"):
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(GITHUB_MODEL_URL, headers=headers, stream=True, allow_redirects=True)
            if r.status_code == 200:
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error(f"Network Error: {r.status_code}. Please check the Release link.")
                return None

    # 2. Loading Weights Safely
    try:
        model = SwinClassifier()
        # weights_only=False ensures compatibility with various save formats
        state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        
        # Handle cases where model was saved as a checkpoint dict
        if isinstance(state_dict, dict) and 'model_state' in state_dict:
            model.load_state_dict(state_dict['model_state'])
        elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)
            
        model.eval()
        return model
    except Exception as e:
        st.error(f"Loading Error: {str(e)}")
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        return None

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063067.png", width=80)
    st.markdown("### **Clinical Control Center**")
    st.info("System: Swin Transformer-T\n\nTarget: Retinitis Pigmentosa")
    st.write("---")
    st.markdown("**Device Stats:**")
    st.caption("CPU Processing Enabled")
    st.caption("Precision: Float32")

# --- MAIN DASHBOARD LAYOUT ---
st.markdown("<h1 class='clinical-header'>👁️ Retinal Pathology Intelligence System</h1>", unsafe_allow_html=True)
st.write("Automated screening assistant for Retinitis Pigmentosa identification.")

# Main Dashboard Columns
col_img, col_diag = st.columns([1, 1], gap="large")

with col_img:
    st.subheader("📸 Patient Data Acquisition")
    uploaded_file = st.file_uploader("Upload High-Res Fundus Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Source: Acquired Digital Fundus Photograph', use_column_width=True)
    else:
        # Placeholder image/instructions
        st.info("Please upload a fundus photograph to initiate the diagnostic scan.")

with col_diag:
    st.subheader("🔍 Pathology Analysis")
    if uploaded_file:
        if st.button('🚀 EXECUTE DIAGNOSTIC SCAN'):
            model = load_medical_model()
            
            if model:
                # Pre-processing pipeline
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_tensor = transform(image).unsqueeze(0)

                with st.spinner('Analyzing morphological features...'):
                    with torch.no_grad():
                        logits = model(img_tensor)
                        prob = torch.sigmoid(logits).item()

                    # Result Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("Detection Confidence", f"{prob:.1%}")
                    m2.metric("Pathology Status", "POSITIVE" if prob > 0.5 else "NEGATIVE")

                    # Visual Feedback
                    if prob > 0.5:
                        st.markdown(f"""
                        <div class="status-box" style="background-color: #fff1f0; border-color: #ffa39e;">
                            <h3 style="color: #cf1322;">⚠️ Retinitis Pigmentosa Detected</h3>
                            <p>AI markers identified pattern consistent with RP. Suggested observation for: 
                            <b>Bone-spicule pigmentation, attenuated vessels, and optic disc pallor.</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="status-box" style="background-color: #f6ffed; border-color: #b7eb8f;">
                            <h3 style="color: #389e0d;">✅ Normal Findings</h3>
                            <p>No significant markers for Retinitis Pigmentosa were identified. The retina appears clinically stable according to current model parameters.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Probability Bar
                    st.write("---")
                    st.write("**Risk Spectrum:**")
                    st.progress(prob)
            else:
                st.error("Diagnostic engine failed to initialize. Please check logs.")
    else:
        st.write("---")
        st.caption("Awaiting input data...")

# --- FOOTER ---
st.markdown("---")
st.warning("**LEGAL DISCLAIMER:** This AI platform is for **research and educational assistance only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider.")
