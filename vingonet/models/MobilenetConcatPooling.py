import torch
from torch import nn
from torch.nn import functional as F
from .MobileNetV2 import MobileNetV2


class MobilenetConcatPooling(nn.Module):
    def __init__(self):
        """
        Mobilenet with mean, max, min pooling of Convolutions
        """
        super().__init__()
        self.features = MobileNetV2().features

    def forward(self, X):
        out = self.features(X)
        out_mean = out.mean(2).mean(2)
        out_max = out.max(2)[0].max(2)[0]
        out_min = out.min(2)[0].min(2)[0]
        out = torch.cat((out_min, out_max, out_mean), dim=1)
        out = F.normalize(out)

        return out
