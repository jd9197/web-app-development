import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import requests
import os
from torchvision import transforms

# --- CLINICAL THEME SETTINGS ---
st.set_page_config(page_title="RP Pathology AI", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; }
    .status-card { padding: 20px; border-radius: 10px; margin-top: 10px; border: 1px solid #ddd; background-color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL ARCHITECTURE ---
MODEL_URL = "https://github.com/jd9197/web-app-development/releases/download/v1.0/swin_retina_rp_deploy.pth"
MODEL_PATH = "swin_weights.pth"

class SwinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone matches Swin Tiny used in training
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features),
            nn.Linear(self.backbone.num_features, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

@st.cache_resource
def load_engine():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("Downloading Clinical Model Weights..."):
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(MODEL_URL, headers=headers, stream=True, allow_redirects=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    model = SwinClassifier()
    # Loading weights with map_location='cpu' for streamlit servers
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if isinstance(state_dict, dict) and 'model_state' in state_dict:
        model.load_state_dict(state_dict['model_state'])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model

# --- DASHBOARD ---
st.title("👁️ Retinal Intelligence: Clinical AI Dashboard")
st.write("Specialized Identification System for Retinitis Pigmentosa (RP)")
st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📁 Patient Image Upload")
    uploaded_file = st.file_uploader("Upload Fundus Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, use_container_width=True, caption="Source Fundus Image")

with col2:
    st.subheader("🩺 Diagnostic Analysis")
    if uploaded_file:
        if st.button("RUN AI PATHOLOGY SCAN"):
            model = load_engine()
            
            # Clinical Preprocessing (Standard ImageNet Normalization)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0)

            with st.spinner("Analyzing Retinal Layers..."):
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = torch.sigmoid(output).item()

                # Results Display
                st.markdown('<div class="status-card">', unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.metric("RP Risk Score", f"{prob:.1%}")
                m2.metric("Clinical Status", "POSITIVE" if prob > 0.5 else "NEGATIVE")
                
                st.write("**Risk Spectrum:**")
                st.progress(prob)

                if prob > 0.5:
                    st.error("### Pathology Detected")
                    st.write("AI identifies patterns consistent with RP. Correlate with clinical findings.")
                else:
                    st.success("### Normal Findings")
                    st.write("The retina appears stable. No significant RP markers identified.")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload an image to initiate the diagnostic scan.")

st.divider()
st.caption("Legal: For research and academic support only. Not a medical diagnosis.")
