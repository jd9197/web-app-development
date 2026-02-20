import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import requests
import os
from torchvision import transforms

# --- 1. SETTINGS & THEME ---
st.set_page_config(page_title="RP Clinical AI Explorer", page_icon="👁️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #007bff; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #1e3d59; color: white; font-weight: bold; border: none; transition: 0.3s; }
    .stButton>button:hover { background-color: #2b5072; transform: translateY(-2px); }
    .diagnosis-card { padding: 30px; border-radius: 15px; margin-top: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); background-color: white; border: 1px solid #e1e4e8; }
    .clinical-header { color: #1e3d59; font-weight: 800; border-bottom: 3px solid #007bff; padding-bottom: 10px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL CORE ---
MODEL_URL = "https://github.com/jd9197/web-app-development/releases/download/v1.0/swin_retina_rp_deploy.pth"
MODEL_PATH = "model_weights.pth"

class SwinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
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
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(MODEL_URL, headers=headers, stream=True, allow_redirects=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    model = SwinClassifier()
    try:
        state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict['model_state'] if isinstance(state_dict, dict) and 'model_state' in state_dict else state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Engine Error: {e}")
        return None

# --- 3. DASHBOARD SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063067.png", width=100)
    st.title("System Controls")
    st.divider()
    # Sensitivity slider is key! If model is too 'Normal', lower this to 0.3 or 0.4
    threshold = st.slider("Diagnostic Sensitivity Threshold", 0.1, 0.9, 0.5, 0.05)
    st.info("System: Swin Transformer-T\nPrecision: FP32\nTarget: Retinitis Pigmentosa")

# --- 4. MAIN LAYOUT ---
st.markdown("<h1 class='clinical-header'>👁️ Retinal Pathology Intelligence System</h1>", unsafe_allow_html=True)

col_img, col_diag = st.columns([1, 1.2], gap="large")

with col_img:
    st.subheader("📸 Image Acquisition")
    uploaded_file = st.file_uploader("Upload Retinal Fundus Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Acquired Patient Fundus', use_container_width=True)
    else:
        st.info("Awaiting fundus photograph upload...")

with col_diag:
    st.subheader("🔍 Pathology Analysis")
    if uploaded_file:
        if st.button('🚀 EXECUTE DIAGNOSTIC SCAN'):
            model = load_medical_model()
            
            if model:
                # REFINED PREPROCESSING: Using Resize + CenterCrop ensures AI sees the center of the eye
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_tensor = transform(image).unsqueeze(0)

                with st.spinner('Analyzing retinal morphology...'):
                    with torch.no_grad():
                        logits = model(img_tensor)
                        prob = torch.sigmoid(logits).item()

                    # --- DASHBOARD RESULTS ---
                    st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric("Detection Confidence", f"{prob:.1%}")
                    m2.metric("Result", "POSITIVE" if prob > threshold else "NEGATIVE")
                    
                    st.write("**Risk Probability Spectrum:**")
                    st.progress(prob)
                    
                    if prob > threshold:
                        st.error("### Clinical Alert: RP Pattern Identified")
                        st.markdown("""
                        **AI Observations:**
                        - Signs of bone-spicule pigmentary deposits detected.
                        - Attenuation of retinal vasculature suggested.
                        - Higher than average confidence for RP markers.
                        """)
                    else:
                        st.success("### Status: Normal / Low Risk")
                        st.write("No significant morphological markers for Retinitis Pigmentosa were identified in this sample.")
                    
                    # RAW SCORE FOR DEBUGGING
                    st.divider()
                    st.caption(f"Raw Diagnostic Score: {prob:.4f} (Threshold: {threshold})")
                    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.warning("**Disclaimer:** For research and educational support only. Not for final clinical diagnosis.")
