import torch
from torch import nn
from torch.nn import functional as F
from .MobileNetV2 import MobileNetV2


class MobilenetASPoC(nn.Module):
    def __init__(self):
        """
        Mobilenet with Attention Sum-pooling of Convolutions

        """
        super().__init__()
        self.features = MobileNetV2().features
        self.attention = nn.Parameter(torch.torch.randn(1280, 7, 7))

    def forward(self, X):
        out = self.features(X)
        out = out * F.sigmoid(self.attention)
        out = out.sum(2).sum(2)
        out = F.normalize(out)

        return out
