import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
from gradcam import GradCAM

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Dental Cavity Detection", layout="centered")
st.title("🦷 Dental Cavity Detection & Severity Classification")
st.write("Upload a dental OPG X-ray image to detect cavity stage")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 4)
    model.load_state_dict(torch.load("E:/project/src/cavity_stage_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

LABELS = [
    "No Cavity",
    "Early Enamel Decay",
    "Dentin Decay",
    "Severe / Pulp Involvement"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload OPG X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    # Prediction
    output = model(input_tensor)
    pred_class = torch.argmax(output).item()

    st.subheader("🧠 Prediction Result")
    st.success(LABELS[pred_class])

    # Grad-CAM
    cam = GradCAM(model, model.layer4)
    heatmap = cam.generate(input_tensor, pred_class)

    img_np = np.array(image.resize((224,224)))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(
        cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR),
        0.6,
        heatmap,
        0.4,
        0
    )

    st.subheader("📍 Cavity Localization (Grad-CAM)")
    st.image(overlay, use_column_width=True)
