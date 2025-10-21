"""
Helmet Detection Model based on ResNet50
"""
import torch
import torch.nn as nn
from torchvision import models


class HelmetDetector(nn.Module):
    """
    Helmet detection model using pretrained ResNet50
    Binary classification: with_helmet (1) or without_helmet (0)
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super(HelmetDetector, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet50 = models.resnet50(pretrained=pretrained)
        
        # Get the number of features from the last layer
        num_features = self.resnet50.fc.in_features
        
        # Replace the final fully connected layer
        # ResNet50 original output is 1000 classes (ImageNet)
        # We need 2 classes: with helmet, without helmet
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet50(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier"""
        for param in self.resnet50.parameters():
            param.requires_grad = False
        
        # Unfreeze the final classifier
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.resnet50.parameters():
            param.requires_grad = True


def get_model(num_classes=2, pretrained=True, freeze_backbone=False):
    """
    Factory function to create and configure the model
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Use pretrained weights (default: True)
        freeze_backbone (bool): Freeze backbone layers (default: False)
    
    Returns:
        model: Configured HelmetDetector model
    """
    model = HelmetDetector(num_classes=num_classes, pretrained=pretrained)
    
    if freeze_backbone:
        model.freeze_backbone()
    
    return model


if __name__ == "__main__":
    # Test the model
    model = get_model()
    print("Model architecture:")
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

