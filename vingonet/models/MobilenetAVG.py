import torch
from torch import nn
from torch.nn import functional as F
from .MobileNetV2 import MobileNetV2


class MobilenetAVG(nn.Module):
    def __init__(self):
        """
        Mobilenet with avg-pooling of Convolutions
        """
        super().__init__()
        self.features = MobileNetV2().features

    def forward(self, X):
        out = self.features(X)
        out = out.mean(2).mean(2)
        out = F.normalize(out)

        return out
