import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .MobileNetV2 import MobileNetV2


class MobilenetASPoC(nn.Module):
    def __init__(self):
        """
        Mobilenet with Sum-pooling of Convolutions

        https://arxiv.org/abs/1510.07493
        """
        super().__init__()
        self.features = MobileNetV2().features
        u = 0.1

        for p in self.features.parameters():
            p.requires_grad = False

        norm = lambda x: 1/(u*np.sqrt(np.pi*2)) * np.exp(-(x-.4)**2/(2*u**2))
        self.attention_tensor = nn.Parameter(torch.tensor([[(norm(x/7) * norm(y/7)) for x in range(7)] for y in range(7)]))

    def forward(self, X):
        out = self.features(X)
        out = out * self.attention_tensor
        out = out.sum(2).sum(2)
        out = F.normalize(out)

        return out
