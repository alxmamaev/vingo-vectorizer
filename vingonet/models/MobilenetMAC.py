import torch
from torch import nn
from torch.nn import functional as F
from .MobileNetV2 import MobileNetV2


class MobilenetMAC(nn.Module):
    def __init__(self):
        """
        Mobilenet with Maximum Activations of Convolutions

        https://arxiv.org/abs/1511.05879
        """
        super().__init__()
        self.features = MobileNetV2().features

    def forward(self, X):
        with torch.no_grad():
            out = self.features(X)
            out = out.max(2)[0].max(2)[0]
        out = self.fc(out)
        out = F.normalize(out)

        return out
