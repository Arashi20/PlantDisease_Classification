import streamlit as st
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model
from config import MODEL_SAVE_PATH

CLASSES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

DISEASE_INFO = {
    'Pepper__bell___Bacterial_spot': 'Caused by Xanthomonas bacteria. Appears as small, dark spots on leaves.',
    'Pepper__bell___healthy': 'No disease detected. Plant appears healthy.',
    'Potato___Early_blight': 'Caused by Alternaria solani fungus. Appears as dark spots with concentric rings.',
    'Potato___Late_blight': 'Caused by Phytophthora infestans. Highly destructive, caused the Irish Potato Famine.',
    'Potato___healthy': 'No disease detected. Plant appears healthy.',
    'Tomato_Bacterial_spot': 'Caused by Xanthomonas bacteria. Spreads in warm, wet conditions.',
    'Tomato_Early_blight': 'Caused by Alternaria solani. One of the most common tomato diseases.',
    'Tomato_Late_blight': 'Caused by Phytophthora infestans. Can destroy an entire crop within days.',
    'Tomato_Leaf_Mold': 'Caused by Passalora fulva fungus. Thrives in humid greenhouse conditions.',
    'Tomato_Septoria_leaf_spot': 'Caused by Septoria lycopersici. Spreads rapidly in wet weather.',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Caused by Tetranychus urticae mites. Damages leaves by feeding on plant cells.',
    'Tomato__Target_Spot': 'Caused by Corynespora cassiicola fungus. Forms concentric ring patterns.',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Viral disease spread by whiteflies. Causes leaf curling and yellowing.',
    'Tomato__Tomato_mosaic_virus': 'Viral disease that causes mosaic patterns on leaves. Spreads through contact.',
    'Tomato_healthy': 'No disease detected. Plant appears healthy.'
}

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    return model, device

def predict_and_explain(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_resized = image.resize((224, 224))
    image_np = np.array(image_resized) / 255.0
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)

    top3_probs, top3_indices = probs.topk(3, dim=1)
    predicted_class = CLASSES[top3_indices[0][0].item()]
    confidence = top3_probs[0][0].item() * 100

    # Grad-CAM
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(top3_indices[0][0].item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    cam_image = show_cam_on_image(image_np.astype(np.float32), grayscale_cam, use_rgb=True)

    top3 = [(CLASSES[top3_indices[0][i].item()], top3_probs[0][i].item() * 100) for i in range(3)]

    return predicted_class, confidence, cam_image, top3


# --- UI ---
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

st.title("🌿 Plant Disease Classifier")
st.markdown("Upload a leaf image to detect plant diseases using EfficientNet-B0 with Grad-CAM visualization.")
st.markdown("---")

model, device = load_model()

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyzing leaf..."):
        predicted_class, confidence, cam_image, top3 = predict_and_explain(image, model, device)

    # Resultaten
    is_healthy = "healthy" in predicted_class.lower()
    status_color = "🟢" if is_healthy else "🔴"

    st.markdown(f"## {status_color} Prediction: `{predicted_class.replace('_', ' ')}`")

    if confidence >= 90:
        st.success(f"Confidence: {confidence:.1f}%")
    elif confidence >= 70:
        st.warning(f"Confidence: {confidence:.1f}%")
    else:
        st.error(f"Confidence: {confidence:.1f}% — Low confidence, consider retaking the photo")

    st.info(f"ℹ️ {DISEASE_INFO.get(predicted_class, '')}")

    st.markdown("---")

    # Afbeeldingen
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Grad-CAM Heatmap")
        st.image(cam_image, use_container_width=True)
        st.caption("Red areas show where the model focused its attention")

    st.markdown("---")

    # Top-3
    st.subheader("Top-3 Predictions")
    for cls, prob in top3:
        st.progress(int(prob), text=f"{cls.replace('_', ' ')}: {prob:.1f}%")

    st.markdown("---")
    st.caption("Model: EfficientNet-B0 pretrained on ImageNet, fine-tuned on PlantVillage dataset (15 classes, 41,000+ images)")