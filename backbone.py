"""FRITNet backbone wrapper.

The V21 feature-extraction topology is retained. `pretrained` is configurable:
- None: fully random InceptionResNetV1 initialization.
- 'vggface2': FaceNet weights pretrained on VGGFace2.

No V21 FERPlus checkpoint is loaded by this module.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TruncatedFaceNet(nn.Module):
    def __init__(self, pretrained: str | None = None, freeze_early_layers: bool = False):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError as exc:
            raise ImportError(
                "facenet-pytorch is required when constructing the real FaceNet backbone. "
                "The architecture unit test can run without it by injecting a dummy backbone."
            ) from exc
        facenet = InceptionResnetV1(pretrained=pretrained)
        self.features = nn.Sequential(
            facenet.conv2d_1a,
            facenet.conv2d_2a,
            facenet.conv2d_2b,
            facenet.maxpool_3a,
            facenet.conv2d_3b,
            facenet.conv2d_4a,
            facenet.conv2d_4b,
            facenet.repeat_1,
            facenet.mixed_6a,
            facenet.repeat_2,
            facenet.mixed_7a,
            facenet.repeat_3,
            facenet.block8,
        )
        if freeze_early_layers:
            for param in self.features[:-4].parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input BCHW with 3 channels; got {tuple(x.shape)}")
        features = self.features(x)
        if features.ndim != 4 or features.shape[1] != 1792:
            raise RuntimeError(
                "Unexpected FaceNet feature shape. Expected [B, 1792, H, W], "
                f"got {tuple(features.shape)}"
            )
        return features
