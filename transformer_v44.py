"""V44 compact multi-scale, class-query transformer head.

It replaces V21's single shared global token with one learnable query per
emotion.  The head retains FRITNet's global SAFM map and four local LFA
regions, then adds four pooled tokens from the real IR-50 stage-2 map.
No external landmark model or image-level auxiliary input is required.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from safm import SAFM


class MultiScaleEmotionQueryHead(nn.Module):
    """Class-conditioned decoding over compact IR-50/LFA feature tokens."""

    def __init__(self, embed_dim=128, num_heads=8, num_layers=2, num_classes=8, dropout=0.5):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_classes = int(num_classes)
        self.local_safm = SAFM(channels=embed_dim, kernel_size=7, use_channel_attention=False)
        self.global_pool_proj = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim))
        self.local_pool_proj = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim))
        self.stage2_pool_proj = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim))
        self.memory_pos = nn.Parameter(torch.randn(1, 9, embed_dim) * 0.02)
        self.emotion_queries = nn.Parameter(torch.randn(1, num_classes, embed_dim) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(1, num_classes, embed_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.query_classifier = nn.Linear(embed_dim, 1)
        self.aux_global_head = nn.Linear(embed_dim, num_classes)
        self.aux_local_head = nn.Linear(embed_dim, num_classes)

    @staticmethod
    def _avg_max_pool(region):
        return torch.cat((F.adaptive_avg_pool2d(region, 1).flatten(1),
                          F.adaptive_max_pool2d(region, 1).flatten(1)), dim=1)

    @staticmethod
    def _quadrants(x):
        return (x[:, :, :14, :14], x[:, :, :14, 14:], x[:, :, 14:, :14], x[:, :, 14:, 14:])

    def forward(self, global_map, local_map, stage2_map):
        expected = (128, 28, 28)
        if tuple(global_map.shape[1:]) != expected or tuple(local_map.shape[1:]) != expected:
            raise ValueError("V44 query head expects global and local [B,128,28,28] maps")
        if tuple(stage2_map.shape[1:]) != expected:
            raise ValueError("V44 query head expects IR-50 stage-2 [B,128,28,28] map")

        global_feat = self.global_pool_proj(self._avg_max_pool(global_map))
        local_tokens = [self.local_pool_proj(self._avg_max_pool(self.local_safm(region)))
                        for region in self._quadrants(local_map)]
        stage2_tokens = [self.stage2_pool_proj(self._avg_max_pool(region))
                         for region in self._quadrants(stage2_map)]
        memory = torch.cat((global_feat.unsqueeze(1), torch.stack(local_tokens, 1),
                            torch.stack(stage2_tokens, 1)), dim=1)
        memory = memory + self.memory_pos
        queries = self.emotion_queries.expand(memory.shape[0], -1, -1) + self.query_pos
        decoded = self.decoder(tgt=queries, memory=memory)
        logits = self.query_classifier(decoded).squeeze(-1)
        aux_global = self.aux_global_head(global_feat)
        aux_local = self.aux_local_head(torch.stack(local_tokens, 1).mean(1))
        return logits, decoded.mean(1), aux_global, aux_local
