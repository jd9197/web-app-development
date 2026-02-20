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

# --- MEDICAL DASHBOARD THEMING ---
st.set_page_config(
    page_title="RP Clinical AI Explorer",
    page_icon="👁️",
    layout="wide",
)

# Custom CSS to make it look clinical
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATION ---
GITHUB_MODEL_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0/swin_retina_rp_deploy.pth"
MODEL_PATH = "model_weights.pth"

class SwinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features),
            nn.Linear(self.backbone.num_features, 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

@st.cache_resource
def load_medical_model():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(GITHUB_MODEL_URL, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    model = SwinClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

# --- DASHBOARD SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063067.png", width=100)
st.sidebar.title("Clinical Control Panel")
st.sidebar.info("System: Swin-Transformer (T)\nTarget: Retinitis Pigmentosa")

# --- MAIN INTERFACE ---
st.title("👁️ Retinal Pathology Intelligence System")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Image Acquisition")
    uploaded_file = st.file_uploader("Upload Fundus Photograph (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Acquired Fundus Image', use_column_width=True)

with col2:
    st.subheader("🔍 Automated Diagnostics")
    if uploaded_file:
        if st.button('🚀 RUN PATHOLOGY SCAN'):
            model = load_medical_model()
            
            # Processing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image).unsqueeze(0)

            with st.spinner('Calculating pathology probability...'):
                with torch.no_grad():
                    logits = model(img_tensor)
                    prob = torch.sigmoid(logits).item()

                # --- CLINICAL METRICS ---
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("RP Probability", f"{prob:.1%}")
                m_col2.metric("Status", "Diseased" if prob > 0.5 else "Normal")

                if prob > 0.5:
                    st.error("### CRITICAL: RP Pattern Detected")
                    st.write("AI analysis suggests morphological markers consistent with Retinitis Pigmentosa (e.g., peripheral bone-spicules or vascular attenuation).")
                else:
                    st.success("### ANALYSIS: Normal Findings")
                    st.write("No significant markers for Retinitis Pigmentosa were identified in the provided image.")

                # Probability Bar
                st.progress(prob)
                st.caption("Probability threshold set at 50% for clinical screening.")

    else:
        st.info("Waiting for image upload to begin analysis.")

# --- FOOTER ---
st.markdown("---")
st.warning("**Clinical Disclaimer:** This AI tool is for research assistance only and does not replace the diagnosis of a certified Ophthalmologist. Always correlate with clinical findings.")
