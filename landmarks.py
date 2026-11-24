"""Frozen MobileFaceNet landmark features and landmark-to-image window cross-attention.

MobileFaceNet follows models/mobilefacenet.py in github.com/Talented-Q/POSTER_V2
(MIT License, Copyright (c) 2023 Talented-Q). Layer names match that file, so its
landmark checkpoint (mobilefacenet_model_best.pth.tar) loads with strict=True.
The network is frozen and always kept in eval mode, as in POSTER and POSTER++.

The fusion takes queries from landmark features and keys/values from FRITNet's
28x28 map, inside 7x7 windows, and adds the result through a gate that starts at
zero. At initialization the model is therefore identical to one without landmarks.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        self.conv = ConvBlock(in_c, groups)
        self.conv_dw = ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = LinearBlock(groups, out_c)
        self.residual = residual

    def forward(self, x):
        out = self.project(self.conv_dw(self.conv(x)))
        return x + out if self.residual else out


class Residual(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.model = nn.Sequential(*[DepthWise(c, c, True, kernel, stride, padding, groups) for _ in range(num_block)])

    def forward(self, x):
        return self.model(x)


class GDC(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.conv_6_dw = LinearBlock(512, 512, groups=512, kernel=(7, 7))
        self.conv_6_flatten = nn.Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        return self.bn(self.linear(self.conv_6_flatten(self.conv_6_dw(x))))


class MobileFaceNet(nn.Module):
    """Landmark network. Output head is kept only so the checkpoint loads strictly."""

    def __init__(self, embedding_size=136):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = DepthWise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128)
        self.conv_34 = DepthWise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256)
        self.conv_45 = DepthWise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256)
        self.conv_6_sep = ConvBlock(128, 512)
        self.output_layer = GDC(embedding_size)

    def forward(self, x):
        """Returns the 64x28x28 landmark map for a 112x112 input."""
        return self.conv_3(self.conv_23(self.conv2_dw(self.conv1(x))))


def load_landmark_weights(model: MobileFaceNet, path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return {"loaded_tensors": len(state)}


class FrozenLandmarks(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MobileFaceNet()
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()

    def train(self, mode=True):  # the landmark network never leaves eval mode
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            x = F.interpolate(images, size=(112, 112), mode="bilinear", align_corners=False)
            return self.net(x)


class LandmarkCrossAttention(nn.Module):
    """Landmark queries attend to image features inside 7x7 windows of a 28x28 map."""

    def __init__(self, dim=128, landmark_dim=64, heads=4, window=7):
        super().__init__()
        self.heads, self.window = heads, window
        self.landmark_proj = nn.Sequential(nn.Conv2d(landmark_dim, dim, 1, bias=False), nn.BatchNorm2d(dim))
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def _windows(self, x):
        b, c, h, w = x.shape
        s = self.window
        x = x.view(b, c, h // s, s, w // s, s).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(-1, s * s, c)

    def _merge(self, x, b, c, h, w):
        s = self.window
        x = x.view(b, h // s, w // s, s, s, c).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(b, c, h, w)

    def forward(self, image_map, landmark_map):
        b, c, h, w = image_map.shape
        with torch.autocast(device_type=image_map.device.type, enabled=False):
            image = image_map.float()
            q_in = self._windows(self.landmark_proj(landmark_map.float()))
            kv_in = self._windows(image)
            q = self.q(self.norm_q(q_in))
            k, v = self.kv(self.norm_kv(kv_in)).chunk(2, dim=-1)
            n, t, _ = q.shape
            split = lambda z: z.view(n, t, self.heads, c // self.heads).transpose(1, 2)
            attn = (split(q) @ split(k).transpose(-2, -1)) / math.sqrt(c // self.heads)
            out = (attn.softmax(-1) @ split(v)).transpose(1, 2).reshape(n, t, c)
            out = self._merge(self.proj(out), b, c, h, w)
            return image + self.gate * out
