import torch
import torch.nn as nn
from torchvision import models


def create_vgg16(num_classes=100, pretrained=True, freeze_backbone=False):
    """
    Creates VGG16 with changed classifier.

    Args:
        num_classes (int): amount of input classes.
        pretrained (bool): if we use weights of ImageNet.
        freeze_backbone (bool): if we freeze conv layers.
    """

    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model
