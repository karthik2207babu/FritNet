"""FRITNet V42: V21 graph with switchable changes.

  use_tap         FaceNet repeat_1 or IR-50 stage-2 features added after LFA upsampling (gate starts at 0)
  use_silu_gate   optional zero-initialized residual SiLU gate before the legacy classifier
  num_heads       transformer heads (V21 = 8)
  dropout         transformer dropout (V21 = 0.5)
  parallel_gates  global and local SAFM in parallel, one gate per quadrant
  use_relation    annotation-aware sample relations, used in training only

All switches off gives the V21_CONTROL graph. Extra modules are built in forked RNG
streams, so shared tensors match the control at the same seed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import TruncatedFaceNet
from ir50 import IR50, load_ir50_weights
from landmarks import FrozenLandmarks, LandmarkCrossAttention, load_landmark_weights
from lfa import LFAModule
from safm import SAFM
from transformer_v42 import FRITTransformerV42
from transformer_v44 import MultiScaleEmotionQueryHead

TAP_STAGE_INDEX = 7  # features[7] is repeat_1: [B, 256, 25, 25] at 224 input


@dataclass
class V42Config:
    num_classes: int = 8
    use_tap: bool = True
    num_heads: int = 2
    dropout: float = 0.1
    parallel_gates: bool = True
    use_relation: bool = True
    relation_heads: int = 8
    relation_beta: float = 1.0
    backbone: str = "facenet"   # V43: "facenet" (InceptionResNetV1, VGGFace2) or "ir50" (MS-Celeb-1M)
    landmarks: bool = False     # V43: frozen MobileFaceNet landmark stream with cross-attention
    head: str = "legacy"        # V44: "legacy" or compact multi-scale "queries" head
    use_silu_gate: bool = False # V46 Upgrade: residual SiLU feature gate before classifier

    @classmethod
    def control(cls, num_classes=8, backbone="facenet", landmarks=False, head="legacy"):
        return cls(num_classes=num_classes, use_tap=False, num_heads=8, dropout=0.5,
                   parallel_gates=False, use_relation=False, relation_beta=0.0,
                    backbone=backbone, landmarks=landmarks, head=head)

    def as_dict(self):
        return asdict(self)


def vote_similarity(targets):
    """1 - normalized Jensen-Shannon divergence between every pair of label distributions."""
    p = targets.float().clamp_min(1e-8)
    p = p / p.sum(1, keepdim=True)
    a, b = p[:, None, :], p[None, :, :]
    m = 0.5 * (a + b)
    js = 0.5 * (a * (a / m).log()).sum(-1) + 0.5 * (b * (b / m).log()).sum(-1)
    return 1.0 - (js / math.log(2.0)).clamp(0.0, 1.0)


class SampleRelation(nn.Module):
    """Batch-level attention, biased by similarity of human label distributions.

    Each sample attends to the others in its batch, never to itself. Training only:
    the module is not called at inference.
    """

    def __init__(self, dim=128, heads=8, beta=1.0):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.beta = heads, float(beta)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, features, targets):
        batch, dim = features.shape
        if batch < 2:
            return features
        with torch.autocast(device_type=features.device.type, enabled=False):
            z = features.float()
            q, k, v = self.qkv(z).chunk(3, dim=-1)
            shape = (batch, self.heads, dim // self.heads)
            q, k, v = (t.view(shape).transpose(0, 1) for t in (q, k, v))
            scores = q @ k.transpose(1, 2) / math.sqrt(dim // self.heads)
            if self.beta:
                scores = scores + self.beta * vote_similarity(targets)[None]
            eye = torch.eye(batch, dtype=torch.bool, device=z.device)
            attention = scores.masked_fill(eye[None], float("-inf")).softmax(dim=-1)
            mixed = (attention @ v).transpose(0, 1).reshape(batch, dim)
            z = self.norm1(z + self.proj(mixed))
            return self.norm2(z + self.mlp(z))


class FRITNetV42(nn.Module):
    def __init__(self, config: V42Config | None = None, pretrained_backbone: str | None = "vggface2",
                 ir50_weights: str | None = None, landmark_weights: str | None = None):
        super().__init__()
        self.config = config or V42Config()
        c = self.config
        if c.backbone not in ("facenet", "ir50"):
            raise ValueError(f"Unknown backbone {c.backbone}")
        if c.head not in ("legacy", "queries"):
            raise ValueError(f"Unknown head {c.head}")
        if c.head == "queries" and c.backbone != "ir50":
            raise ValueError("The V44 query head requires IR-50 stage-2 features.")
        if c.head == "queries" and c.use_relation:
            raise ValueError("The V44 query head does not support the training-only relation block.")
        if c.use_silu_gate and c.head != "legacy":
            raise ValueError("The residual SiLU feature gate requires the legacy classifier head.")
        self.load_report = {}
        if c.backbone == "ir50":
            self.backbone = IR50()
            if pretrained_backbone is not None:
                if not ir50_weights:
                    raise ValueError("IR-50 needs its MS-Celeb-1M weights file (ir50_weights).")
                self.load_report["ir50"] = load_ir50_weights(self.backbone, ir50_weights)
        else:
            self.backbone = TruncatedFaceNet(pretrained=pretrained_backbone, freeze_early_layers=False)
        final_channels, tap_channels = (256, 128) if c.backbone == "ir50" else (1792, 256)
        self.lfa = LFAModule(in_channels=final_channels, out_channels=128)
        self.safm = SAFM(channels=128, kernel_size=7, use_channel_attention=False)
        if c.head == "queries":
            self.transformer = MultiScaleEmotionQueryHead(
                embed_dim=128, num_heads=c.num_heads, num_layers=2, num_classes=c.num_classes, dropout=c.dropout,
            )
            self.classifier = None
        else:
            self.transformer = FRITTransformerV42(
                embed_dim=128, num_heads=c.num_heads, num_local_layers=2, num_classes=c.num_classes,
                dropout=c.dropout, region_gates=c.parallel_gates,
            )
            self.classifier = nn.Linear(128, c.num_classes)
            if c.use_silu_gate:
                # Independent RNG stream: adding this optional branch must not
                # perturb initialization of any shared V46 tensor.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(44044)
                    self.silu_gate_proj = nn.Linear(128, 128)
                self.silu_gate_gamma = nn.Parameter(torch.zeros(()))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41041)
            if c.use_tap:
                self.tap_conv = nn.Conv2d(tap_channels, 128, kernel_size=1, bias=False)
                self.tap_bn = nn.BatchNorm2d(128)
                self.tap_gate = nn.Parameter(torch.zeros(()))
            if c.use_relation:
                self.relation = SampleRelation(128, c.relation_heads, c.relation_beta)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43043)
            if c.landmarks:
                self.landmarks = FrozenLandmarks()
                self.landmark_fusion = LandmarkCrossAttention(dim=128, landmark_dim=64)
                if pretrained_backbone is not None:
                    if not landmark_weights:
                        raise ValueError("Landmarks need the MobileFaceNet weights file (landmark_weights).")
                    self.load_report["landmarks"] = load_landmark_weights(self.landmarks.net, landmark_weights)

    def extract(self, images):
        if self.config.backbone == "ir50":
            _, x2, x3 = self.backbone(images)
            return x2, x3  # real 128x28x28 map is the tap source; 256x14x14 feeds LFA
        x, tap_source = images, None
        for index, stage in enumerate(self.backbone.features):
            x = stage(x)
            if index == TAP_STAGE_INDEX:
                tap_source = x
        return tap_source, x

    def forward(self, images):
        tap_source, final_map = self.extract(images)
        tap = None
        if self.config.use_tap:
            tap = F.relu(self.tap_bn(self.tap_conv(tap_source)))
            if tap.shape[-2:] != (28, 28):
                tap = F.interpolate(tap, size=(28, 28), mode="bilinear", align_corners=False)
            tap = self.tap_gate * tap
        x = self.lfa(final_map, tap=tap)
        if self.config.landmarks:
            x = self.landmark_fusion(x, self.landmarks(images))
        gated = self.safm(x)
        local_input = x if self.config.parallel_gates else gated
        if self.config.head == "queries":
            logits, features, aux_global, aux_local = self.transformer(gated, local_input, tap_source)
            return logits, features, aux_global, aux_local
        features, aux_global, aux_local = self.transformer(gated, local_input)
        classifier_features = features
        if self.config.use_silu_gate:
            classifier_features = features + self.silu_gate_gamma * (
                features * F.silu(self.silu_gate_proj(features))
            )
        return self.classifier(classifier_features), features, aux_global, aux_local

    def inference_parameter_count(self):
        skipped = sum(p.numel() for p in self.relation.parameters()) if self.config.use_relation else 0
        return sum(p.numel() for p in self.parameters()) - skipped

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
