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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- PAGE CONFIG ---
st.set_page_config(page_title="RP Clinical AI Explorer", page_icon="👁️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #1e3d59; }
    .status-card { padding: 20px; border-radius: 10px; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIG & DOWNLOAD ---
MODEL_URL = "https://github.com/jd9197/web-app-development/releases/download/v1.0/swin_retina_rp_deploy.pth"
MODEL_PATH = "swin_weights.pth"

# --- SWIN ARCHITECTURE ---
class SwinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
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

# --- GRAD-CAM UTILS ---
def swin_reshape_transform(tensor):
    # Required for Swin Transformer Grad-CAM
    result = tensor.reshape(tensor.size(0), 7, 7, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

@st.cache_resource
def load_system():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        r = requests.get(MODEL_URL, stream=True, allow_redirects=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    model = SwinClassifier()
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if isinstance(state_dict, dict) and 'model_state' in state_dict:
        model.load_state_dict(state_dict['model_state'])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model

# --- PROCESSING ENGINE ---
def run_analysis(image):
    model = load_system()
    
    # 1. Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    # 2. Prediction
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()

    # 3. Grad-CAM (Interpretability)
    # Target the last norm layer of the last block
    target_layers = [model.backbone.layers[-1].blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=swin_reshape_transform)
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0, :]
    
    # Retinal Masking (Fixing the "Bleeding" issue)
    img_np = np.array(image.resize((224, 224))) / 255.0
    img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask.astype(np.float32)/255.0, (15, 15), 0)
    
    masked_cam = grayscale_cam * mask
    cam_image = show_cam_on_image(img_np, masked_cam, use_rgb=True)
    
    return prob, cam_image

# --- UI LAYOUT ---
st.title("👁️ RP Clinical Intelligence Dashboard")
st.markdown("---")

with st.sidebar:
    st.header("Settings")
    st.info("Architecture: Swin-T\nInput: 224x224 Fundus")
    threshold = st.slider("Diagnostic Threshold", 0.1, 0.9, 0.5, 0.05)

uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(raw_img, caption="Patient Input Image", use_column_width=True)
    
    if st.button("🚀 EXECUTE FULL PATHOLOGY SCAN"):
        prob, cam_viz = run_analysis(raw_img)
        
        with col2:
            st.image(cam_viz, caption="Grad-CAM: Pathology Attention Map", use_column_width=True)
            
            m1, m2 = st.columns(2)
            m1.metric("RP Confidence", f"{prob:.1%}")
            status = "POSITIVE (Diseased)" if prob > threshold else "NEGATIVE (Normal)"
            m2.metric("Result", status)

            if prob > threshold:
                st.error("### Pathology Detected: Retinitis Pigmentosa")
                st.markdown("""
                **Clinical Observations:**
                - Heatmap indicates attention on peripheral retinal areas.
                - Probability exceeds diagnostic threshold.
                - Review for bone-spicule pigment deposits recommended.
                """)
            else:
                st.success("### No RP Pathology Identified")
                st.write("Model indicates a healthy retinal appearance within standard parameters.")

st.markdown("---")
st.caption("Disclaimer: This tool is for research assistance only and is not a primary diagnostic device.")
