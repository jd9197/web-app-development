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

# --- 1. CONFIGURATION & MODEL SETUP ---
# CHANGE THIS TO YOUR ACTUAL RELEASE LINK!
GITHUB_MODEL_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0/swin_retina_rp_deploy.pth"
MODEL_PATH = "model_weights.pth"

class SwinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches your training backbone
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
    # Download weights if they don't exist
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI Model weights... please wait."):
            r = requests.get(GITHUB_MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    model = SwinClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

# --- 2. WEB UI SETUP ---
st.set_page_config(page_title="RP Diagnosis AI", layout="centered")
st.title("👁️ Retinitis Pigmentosa Diagnostic Assistant")
st.write("Upload a Retinal Fundus image for clinical analysis.")

uploaded_file = st.file_uploader("Choose a fundus image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Prepare Image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Process Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)

    if st.button('Analyze Eye'):
        model = load_medical_model()
        with st.spinner('AI is analyzing retinal patterns...'):
            with torch.no_grad():
                logits = model(img_tensor)
                probability = torch.sigmoid(logits).item()
            
            # Display Results
            st.subheader("Results:")
            if probability > 0.5:
                st.error(f"Prediction: Retinitis Pigmentosa Detected")
                st.write(f"Confidence: **{probability:.2%}**")
                st.warning("Note: Clinical correlation is required. AI suggests presence of bone-spicule pigmentation.")
            else:
                st.success(f"Prediction: Normal / Low Risk")
                st.write(f"Confidence: **{(1-probability):.2%}**")
