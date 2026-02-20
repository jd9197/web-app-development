import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import requests
import os
from torchvision import transforms

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RetinaAI | Clinical Analysis",
    page_icon="👁️",
    layout="wide"
)

# Custom Medical Theme CSS
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    [data-testid="stMetricValue"] { font-size: 32px; color: #1e3d59; }
    .stButton>button { width: 100%; background-color: #1e3d59; color: white; border-radius: 5px; height: 3em; }
    .stAlert { border-radius: 10px; }
    .report-card { background-color: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL CONFIGURATION ---
MODEL_URL = "https://github.com/jd9197/web-app-development/releases/download/v1.0/swin_retina_rp_deploy.pth"
MODEL_PATH = "swin_weights.pth"

class SwinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone matches your training (Swin Tiny)
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features),
            nn.Linear(self.backbone.num_features, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

@st.cache_resource
def load_clinical_model():
    # Correcting the "Error installing requirements" by handling file downloads properly
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("Synchronizing medical database..."):
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(MODEL_URL, headers=headers, stream=True, allow_redirects=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    model = SwinClassifier()
    # Weights_only=False is used for compatibility with older .pth saves
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Handle different save formats (checkpoint vs state_dict)
    if isinstance(state_dict, dict) and 'model_state' in state_dict:
        model.load_state_dict(state_dict['model_state'])
    else:
        model.load_state_dict(state_dict)
        
    model.eval()
    return model

# --- DASHBOARD SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063067.png", width=80)
    st.title("Clinical Panel")
    st.divider()
    st.info("**Patient Identifier:** RED-9197")
    st.write("**Model:** Swin Transformer-T")
    st.write("**Target:** Retinitis Pigmentosa")
    st.divider()
    sensitivity = st.slider("Diagnostic Threshold", 0.1, 0.9, 0.5, 0.05)

# --- MAIN INTERFACE ---
st.title("👁️ Retinal Pathology Intelligence System")
st.caption("Advanced Screening Assistant for Retinitis Pigmentosa")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📸 Image Acquisition")
    file = st.file_uploader("Upload Fundus Photograph", type=["jpg", "jpeg", "png"])
    
    if file:
        input_image = Image.open(file).convert('RGB')
        st.image(input_image, use_container_width=True, caption="Uploaded Fundus Image")
    else:
        st.info("Awaiting fundus photograph upload...")

with col_right:
    st.subheader("🔍 Diagnostic Analysis")
    if file:
        if st.button("🚀 EXECUTE DIAGNOSTIC SCAN"):
            model = load_clinical_model()
            
            # Preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_t = transform(input_image).unsqueeze(0)

            with st.spinner("Processing retinal layers..."):
                with torch.no_grad():
                    logits = model(img_t)
                    prob = torch.sigmoid(logits).item()

                # --- DASHBOARD METRICS ---
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.metric("Detection Score", f"{prob:.1%}")
                m2.metric("Diagnosis", "POSITIVE" if prob > sensitivity else "NEGATIVE")
                
                # Probability Bar
                st.write("**Pathology Probability Spectrum:**")
                st.progress(prob)
                
                # Clinical Insight
                if prob > sensitivity:
                    st.error("### Alert: High Risk of Retinitis Pigmentosa")
                    st.write("Morphological markers identified. Correlate with peripheral field testing and OCT.")
                else:
                    st.success("### Status: Within Normal Limits")
                    st.write("No major pigmentary changes detected by the AI core.")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("Upload an image on the left to begin.")

# --- FOOTER ---
st.divider()
st.warning("**Disclaimer:** This AI tool is for research support only. Final diagnosis must be performed by a board-certified ophthalmologist.")
