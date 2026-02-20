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

# --- IMPORT GRAD-CAM ---
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    st.error("Grad-CAM library not found. Please ensure 'grad-cam' is in requirements.txt")

# --- 1. SETTINGS & THEME ---
st.set_page_config(page_title="RP Clinical AI", page_icon="👁️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8faff; }
    .stMetric { border-radius: 10px; border: 1px solid #dce4f2; background-color: white; padding: 15px; }
    .status-box { padding: 20px; border-radius: 10px; border: 2px solid #e0e0e0; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

MODEL_URL = "https://github.com/jd9197/web-app-development/releases/download/v1.0/swin_retina_rp_deploy.pth"
MODEL_PATH = "swin_weights.pth"

# --- 2. MODEL ARCHITECTURE ---
class SwinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches Swin-Tiny architecture
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

# REQUIRED FOR SWIN TRANSFORMER GRAD-CAM
def swin_reshape_transform(tensor):
    # Swin-T outputs a sequence that needs to be reshaped to 7x7 for the heatmap
    result = tensor.reshape(tensor.size(0), 7, 7, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(MODEL_URL, headers=headers, stream=True, allow_redirects=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    model = SwinClassifier()
    # Loading weights safely
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if isinstance(state_dict, dict) and 'model_state' in state_dict:
        model.load_state_dict(state_dict['model_state'])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model

# --- 3. DASHBOARD UI ---
st.title("👁️ Retinal Intelligence: Clinical RP Screening")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Retinal Image", type=["jpg", "jpeg", "png"])
# Threshold slider allows the doctor to adjust sensitivity
threshold = st.sidebar.slider("Diagnostic Sensitivity (Threshold)", 0.1, 0.9, 0.5, 0.05)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Input Fundus Photograph", use_column_width=True)
    
    if st.button("🚀 EXECUTE DIAGNOSTIC SCAN"):
        model = load_model()
        
        # Pre-processing
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = tfm(image).unsqueeze(0)

        with st.spinner("Analyzing Morphology & Generating Heatmap..."):
            # 1. Prediction
            with torch.no_grad():
                logits = model(input_tensor)
                prob = torch.sigmoid(logits).item()

            # 2. Grad-CAM (Pathology Visualization)
            # Target the last layer of the Swin backbone
            target_layers = [model.backbone.layers[-1].blocks[-1].norm1]
            
            cam = GradCAM(model=model, 
                         target_layers=target_layers, 
                         reshape_transform=swin_reshape_transform)
            
            # Use index 0 as it's a binary classifier
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0, :]
            
            # Prepare image for overlay
            img_np = np.array(image.resize((224, 224))) / 255.0
            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

            with col2:
                st.image(cam_image, caption="AI Pathological Attention Map (Heatmap)", use_column_width=True)
                
                m1, m2 = st.columns(2)
                m1.metric("RP Confidence", f"{prob:.1%}")
                res_text = "POSITIVE" if prob > threshold else "NEGATIVE"
                m2.metric("Result", res_text)

                if prob > threshold:
                    st.error("### Clinical Alert: High RP Probability")
                    st.markdown(f"""
                    **AI Analysis:** Pattern matches markers for Retinitis Pigmentosa.
                    - **Confidence:** {prob:.2%}
                    - **Recommendation:** Clinical follow-up for bone-spicule visualization.
                    """)
                else:
                    st.success("### Scan Result: Normal Findings")
                    st.write("No significant pathological markers identified.")

st.markdown("---")
st.warning("Disclaimer: This tool is for research assistance only and is not a primary diagnostic device.")
