import torch
import torch.nn as nn
from torchvision import models


def create_resnet18(num_classes=100, pretrained=True, freeze_backbone=True):
    """
    Create ResNet with changed output layer

    Args:
        num_classes: int - amount of classes for classification
        pretrained: bool - if we use weights of ImageNet
        freeze_backbone: book - if we freeze conv layers for transfer learning
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
