import torch
from torch import nn
from torch.nn import functional as F
from .MobileNetV2 import MobileNetV2


class MobilenetSPoC(nn.Module):
    def __init__(self):
        """
        Mobilenet with Sum-pooling of Convolutions

        https://arxiv.org/abs/1510.07493
        """
        super().__init__()
        self.features = MobileNetV2().features

    def forward(self, X):
        out = self.features(X)
        out = out.sum(2).sum(2)
        out = F.normalize(out)

        return out
