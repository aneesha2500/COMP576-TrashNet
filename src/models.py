# ============================================================================
# OWNER: SÜLEYMAN - the two CNNs (Goals 1 and 4)
# Vishnu uses gradcam_layer() for Grad-CAM.
# ============================================================================
"""The two CNNs we compare (Goals 1 and 4)."""
import torch.nn as nn
from torchvision import models

from config import CLASSES


class SmallCNN(nn.Module):
    """Trained from scratch: 4 blocks of conv-BN-ReLU-maxpool, global average pooling, 6-class layer."""

    def __init__(self, n_classes=len(CLASSES), widths=(32, 64, 128, 256)):
        super().__init__()
        layers, c_in = [], 3
        for c_out in widths:
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_in, n_classes)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


def build_model(name, pretrained=True):
    if name == "smallcnn":
        return SmallCNN()
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, len(CLASSES))   # new 6-class last layer
        return m
    raise ValueError(f"Unknown model {name}")


def gradcam_layer(model, name):
    """The last conv block, used by Grad-CAM."""
    return model.features[-3] if name == "smallcnn" else model.layer4[-1]   # last BatchNorm / last ResNet block
