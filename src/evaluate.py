import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import get_model
from config import DATA_DIR, BATCH_SIZE, MODEL_SAVE_PATH, ARCHITECTURE


def evaluate(model, loader, device, classes):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("models/confusion_matrix2.png", dpi=150)  # Change name when evaluating new models
    print("Confusion matrix saved to models/confusion_matrix2.png")  # Change name when evaluating new models


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)

    model = get_model(num_classes=len(classes), architecture=ARCHITECTURE).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print("Model loaded!")

    print("Evaluating on test set...")
    labels, preds = evaluate(model, test_loader, device, classes)

    print("\n--- Classification Report ---")
    print(classification_report(labels, preds, target_names=classes))

    plot_confusion_matrix(labels, preds, classes)


if __name__ == "__main__":
    main()