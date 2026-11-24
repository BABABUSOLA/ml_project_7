import torch.nn as nn
from torchvision.models import resnet18

def make_resnet18(num_classes=10, dropout_p=0.0):
    """torchvision's ResNet-18 is built for 224x224 ImageNet. For 32x32 CIFAR the
    7x7 stride-2 conv and the maxpool destroy too much resolution, so both are replaced.

    dropout_p > 0 inserts channel-wise dropout after each residual stage, not only
    at the head. MC dropout needs the stochasticity to enter while features are still
    being built; a single head dropout leaves all passes sharing identical features."""
    m = resnet18(weights=None, num_classes=num_classes)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    if dropout_p > 0:
        for stage in ['layer1', 'layer2', 'layer3', 'layer4']:
            setattr(m, stage, nn.Sequential(getattr(m, stage),
                                            nn.Dropout2d(dropout_p)))
        m.fc = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(512, num_classes))
    return m