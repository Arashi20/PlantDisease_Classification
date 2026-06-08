# Using an EfficientNet model with pre-trained weights. 
# It is a family of CNN architectures with good benchmarks for transfer learning. 
# Reference: https://arxiv.org/abs/1905.11946

# For comparison I also added the MobilenetV3-Small model, which is a lightweight model designed for edge computing
# See: https://huggingface.co/docs/timm/models/mobilenet-v3

import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes, architecture="efficientnet_b0"): # 4M parameters
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif architecture == "mobilenet_v3_small": #1.5M parameters
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


if __name__ == "__main__":
    for arch in ["efficientnet_b0", "mobilenet_v3_small"]:
        model = get_model(num_classes=15, architecture=arch)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{arch}: {params:.1f}M parameters")