"""V21 spatial SAFM with an optional V38 channel-attention extension.

`use_channel_attention=False` reproduces the original spatial-only SAFM.
`use_channel_attention=True` retains that spatial gate and adds a lightweight,
near-identity channel gate for a separately measurable ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAFM(nn.Module):
    def __init__(
        self,
        channels: int = 128,
        kernel_size: int = 7,
        reduction: int = 8,
        use_channel_attention: bool = False,
    ):
        super().__init__()
        if channels % reduction != 0:
            raise ValueError("channels must be divisible by reduction")
        self.use_channel_attention = bool(use_channel_attention)
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

        if self.use_channel_attention:
            hidden = max(channels // reduction, 8)
            self.channel_mlp = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            )
            # Near-identity channel scaling at initialization; small nonzero
            # weights allow gradients to reach both layers from the start.
            nn.init.normal_(self.channel_mlp[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.channel_mlp[-1].bias)
        else:
            self.channel_mlp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 128:
            raise ValueError(f"Expected [B, 128, H, W], got {tuple(x.shape)}")

        avg_channel = torch.mean(x, dim=1, keepdim=True)
        max_channel = torch.amax(x, dim=1, keepdim=True)
        spatial_gate = self.sigmoid(
            self.spatial_conv(torch.cat((avg_channel, max_channel), dim=1))
        )
        out = x * spatial_gate

        if self.channel_mlp is not None:
            avg_pool = F.adaptive_avg_pool2d(x, 1)
            max_pool = F.adaptive_max_pool2d(x, 1)
            channel_logits = self.channel_mlp(avg_pool) + self.channel_mlp(max_pool)
            channel_scale = 2.0 * self.sigmoid(channel_logits)  # [0, 2], starts near 1
            out = out * channel_scale
        return out
