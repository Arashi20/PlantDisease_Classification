import torch
from torchvision import transforms
from PIL import Image
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = probs.max(1)

    predicted_class = CLASSES[pred_idx.item()]
    confidence_pct = confidence.item() * 100

    print(f"\nPrediction:  {predicted_class}")
    print(f"Confidence:  {confidence_pct:.1f}%")

    return predicted_class, confidence_pct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Pad naar de afbeelding")
    args = parser.parse_args()
    predict(args.image)