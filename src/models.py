"""
src/models.py
Deep learning models for CIFAKE image classification.
"""

import torch
import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    """
    Lightweight CNN baseline for CIFAKE.
    Output:
        0 = REAL
        1 = FAKE
    """

    def __init__(self):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_resnet18(num_classes=2, pretrained=True):
    """
    Creates a ResNet18 model for CIFAKE binary classification.

    Classes:
        0 = REAL
        1 = FAKE
    """

    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    # CIFAKE images are only 32x32.
    # A smaller first convolution is more suitable for these images.
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    # Remove the large-image max pooling operation.
    model.maxpool = nn.Identity()

    # Replace final layer for REAL/FAKE classification.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
