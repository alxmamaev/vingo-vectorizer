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

        for p in self.features.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(1280, 512)

    def forward(self, X):
        out = self.features(X)
        out = out.max(2)[0].max(2)[0]
        out = self.fc(out)
        out = F.normalize(out)

        return out
