"""IR-50 face backbone, first three stages, as used by POSTER and POSTER++.

Architecture follows models/ir50.py in github.com/Talented-Q/POSTER_V2 (MIT License,
Copyright (c) 2023 Talented-Q). Layer names match that file, so its MS-Celeb-1M
weights (ir50.pth) load directly.

At 112x112 input: body1 -> 64x56x56, body2 -> 128x28x28, body3 -> 256x14x14.
"""
from __future__ import annotations

import collections
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

STAGE_UNITS = (3, 4, 14)  # IR-50 stages 1-3; stage 4 (512 channels) is not used
STAGE_DEPTHS = (64, 128, 256)


class BottleneckIR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                                                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), nn.BatchNorm2d(depth))

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


def _stage(in_channel, depth, units):
    return nn.Sequential(BottleneckIR(in_channel, depth, 2), *[BottleneckIR(depth, depth, 1) for _ in range(units - 1)])


class IR50(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
        self.body1 = _stage(64, 64, STAGE_UNITS[0])
        self.body2 = _stage(64, 128, STAGE_UNITS[1])
        self.body3 = _stage(128, 256, STAGE_UNITS[2])

    def forward(self, x):
        # The IR-50 training pipeline now resizes images directly to 112x112.
        # Keep compatibility with older callers/checkpoints that pass 224x224.
        if x.shape[-2:] != (112, 112):
            x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        x1 = self.body1(self.input_layer(x))
        x2 = self.body2(x1)
        return x1, x2, self.body3(x2)


def _remap_key(key):
    """Accept POSTER keys (body1/body2/body3) and original single `body.N` keys."""
    key = key[7:] if key.startswith("module.") else key
    match = re.match(r"body\.(\d+)\.(.*)", key)
    if not match:
        return key
    index, rest = int(match.group(1)), match.group(2)
    for stage, units in enumerate(STAGE_UNITS, start=1):
        if index < units:
            return f"body{stage}.{index}.{rest}"
        index -= units
    return None  # stage 4, not used


def load_ir50_weights(model: IR50, path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    target = model.state_dict()
    loaded = collections.OrderedDict()
    for key, value in state.items():
        mapped = _remap_key(key)
        if mapped in target and target[mapped].shape == value.shape:
            loaded[mapped] = value
    missing = sorted(set(target) - set(loaded))
    if missing:
        raise RuntimeError(f"IR-50 weights incomplete: {len(missing)} of {len(target)} tensors missing, "
                           f"first {missing[:3]}. Check the file: {path}")
    model.load_state_dict(loaded, strict=True)
    return {"loaded_tensors": len(loaded), "file_tensors": len(state)}
