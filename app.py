import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
import requests
import os
from torchvision import transforms

# --- CLINICAL UI ---
st.set_page_config(page_title="RP Clinical AI", layout="wide")
st.title("👁️ RP Pathology Intelligence")

# Use YOUR release link
MODEL_URL = "https://github.com/jd9197/web-app-development/releases/download/v1.0/swin_retina_rp_deploy.pth"
MODEL_PATH = "model.pth"

class SwinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features),
            nn.Linear(self.backbone.num_features, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.head(self.backbone(x))

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading Clinical Model..."):
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    model = SwinClassifier()
    # Loading with weights_only=False to ensure the pickle error is bypassed
    d = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(d['model_state'] if 'model_state' in d else d)
    model.eval()
    return model

# --- MAIN APP ---
up_file = st.file_uploader("Upload Fundus Image", type=["jpg", "png"])

if up_file:
    img = Image.open(up_file).convert('RGB')
    st.image(img, width=400)
    
    if st.button("Analyze Pathology"):
        model = get_model()
        tfm = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        batch = tfm(img).unsqueeze(0)
        
        with torch.no_grad():
            score = torch.sigmoid(model(batch)).item()
            
        st.metric("Pathology Confidence", f"{score:.1%}")
        if score > 0.5:
            st.error("Result: High Risk of Retinitis Pigmentosa")
        else:
            st.success("Result: Normal / Low Risk")
