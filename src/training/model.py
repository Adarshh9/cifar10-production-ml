"""ResNet-18 model for CIFAR-10 classification."""
import torch
import torch.nn as nn
import torchvision.models as models


class CIFAR10ResNet(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 images, 10 classes).
    
    This is a production-ready model that's:
    - Easy to train (10-20 min on GPU)
    - Achieves 85%+ accuracy
    - Perfect for demonstrating MLOps
    """
    
    def __init__(self, num_classes: int = 10, weights: bool = False):
        super(CIFAR10ResNet, self).__init__()
        
        # Load ResNet-18
        self.model = models.resnet18(weights=weights)
        
        # Adapt first conv layer for 32x32 images (CIFAR-10)
        # Original ResNet uses 7x7 kernel for 224x224 ImageNet images
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove max pooling (not needed for small images)
        self.model.maxpool = nn.Identity()
        
        # Adapt final FC layer for 10 classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


def create_model(num_classes: int = 10, weights: bool = False) -> CIFAR10ResNet:
    """
    Factory function to create model.
    
    Args:
        num_classes: Number of output classes (10 for CIFAR-10)
        pretrained: Whether to use ImageNet pretrained weights
    
    Returns:
        Initialized model
    """
    return CIFAR10ResNet(num_classes=num_classes, weights=weights)
