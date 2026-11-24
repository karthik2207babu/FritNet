"""V21 Local Feature Augmentation (LFA). V41-A adds an optional pre-split `tap` input.

With tap=None this is the unchanged V21 module; no parameters were added here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LFAModule(nn.Module):
    def __init__(self, in_channels: int = 1792, out_channels: int = 128):
        super().__init__()
        self.bridge = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_bridge = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv1x3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3x1 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def _process_region(self, region: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1x3(region)))
        x = self.relu(self.bn2(self.conv3x3(x)))
        x = self.relu(self.bn3(self.conv3x1(x)))
        return x + region

    def forward(self, x: torch.Tensor, tap: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.bridge.in_channels:
            raise ValueError(f"Expected [B, {self.bridge.in_channels}, H, W] backbone features, got {tuple(x.shape)}")
        x = self.relu(self.bn_bridge(self.bridge(x)))
        x = F.interpolate(x, size=(28, 28), mode="bilinear", align_corners=False)
        if tap is not None:
            x = x + tap

        tl = self._process_region(x[:, :, :14, :14])
        tr = self._process_region(x[:, :, :14, 14:])
        bl = self._process_region(x[:, :, 14:, :14])
        br = self._process_region(x[:, :, 14:, 14:])

        top = torch.cat((tl, tr), dim=3)
        bottom = torch.cat((bl, br), dim=3)
        return torch.cat((top, bottom), dim=2)
