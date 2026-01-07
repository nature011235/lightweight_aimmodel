import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, pretrained=True, frozen=False):
        super().__init__()

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        
        self.features = backbone.features
        self.out_channels = 576 
        if frozen:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.features(x)