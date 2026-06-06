# Using an EfficientNet model with pre-trained weights. 
# It is a family of CNN architectures with good benchmarks for transfer learning. 
# Reference: https://arxiv.org/abs/1905.11946
import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = get_model(num_classes=15)
    print(model.classifier)
    print(f"Model loaded with 15 output classes")