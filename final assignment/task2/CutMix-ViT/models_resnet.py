import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

def weight_standardize(w, axis, eps):
    """Subtracts mean and divides by standard deviation."""
    w = w - w.mean(dim=axis, keepdim=True)
    w = w / (w.std(dim=axis, keepdim=True) + eps)
    return w

class StdConv(nn.Conv2d):
    """Convolution with weight standardization."""
    def forward(self, x):
        weight = weight_standardize(self.weight, axis=[0, 1, 2], eps=1e-5)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ResidualUnit(nn.Module):
    """Bottleneck ResNet block."""
    def __init__(self, features, strides=(1, 1)):
        super().__init__()
        self.features = features
        self.strides = strides
        self.needs_projection = True  # 修改为默认需要投影
        self.conv_proj = StdConv(features * 4, kernel_size=1, stride=strides, bias=False) if self.needs_projection else None
        self.gn_proj = nn.GroupNorm(32, features * 4) if self.needs_projection else None

        self.conv1 = StdConv(features, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(32, features)
        self.conv2 = StdConv(features, kernel_size=3, stride=strides, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, features)
        self.conv3 = StdConv(features * 4, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(32, features * 4)

    def forward(self, x):
        residual = x
        if self.needs_projection:
            residual = self.conv_proj(residual)
            residual = self.gn_proj(residual)

        y = self.conv1(x)
        y = self.gn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.gn2(y)
        y = F.relu(y)
        y = self.conv3(y)
        y = self.gn3(y)
        y = F.relu(residual + y)
        return y

class ResNetStage(nn.Module):
    """A ResNet stage."""
    def __init__(self, block_size, nout, first_stride):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualUnit(nout, first_stride)])
        self.blocks.extend([ResidualUnit(nout) for _ in range(1, block_size)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
