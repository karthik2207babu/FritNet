"""V42 five-token transformer: V21 topology with switchable heads, dropout and region gates.

With num_heads=8, dropout=0.5 and region_gates=False this is the V21 transformer.
Construction order matches transformer_v21.py so shared tensors match at the same seed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from safm import SAFM


class FRITTransformerV42(nn.Module):
    def __init__(self, embed_dim=128, num_heads=2, num_local_layers=2, num_classes=8,
                 dropout=0.1, region_gates=True):
        super().__init__()
        self.region_gates = bool(region_gates)
        self.local_safm = SAFM(channels=embed_dim, kernel_size=7, use_channel_attention=False)
        self.local_pool_proj = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim))
        self.global_pool_proj = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 5, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.local_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_local_layers)
        self.aux_global_head = nn.Linear(embed_dim, num_classes)
        self.aux_local_head = nn.Linear(embed_dim, num_classes)
        if self.region_gates:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42042)
                self.extra_gates = nn.ModuleList(
                    SAFM(channels=embed_dim, kernel_size=7, use_channel_attention=False) for _ in range(3)
                )

    @staticmethod
    def _avg_max_pool(region):
        return torch.cat((F.adaptive_avg_pool2d(region, 1).flatten(1),
                          F.adaptive_max_pool2d(region, 1).flatten(1)), dim=1)

    def forward(self, global_map, local_map):
        if tuple(global_map.shape[1:]) != (128, 28, 28) or tuple(local_map.shape[1:]) != (128, 28, 28):
            raise ValueError(f"Expected [B,128,28,28], got {tuple(global_map.shape)}, {tuple(local_map.shape)}")
        global_feat = self.global_pool_proj(self._avg_max_pool(global_map))
        aux_global = self.aux_global_head(global_feat)

        regions = (local_map[:, :, :14, :14], local_map[:, :, :14, 14:],
                   local_map[:, :, 14:, :14], local_map[:, :, 14:, 14:])
        gates = [self.local_safm, *self.extra_gates] if self.region_gates else [self.local_safm] * 4
        tokens = [self.local_pool_proj(self._avg_max_pool(gate(region))) for gate, region in zip(gates, regions)]

        sequence = torch.cat((global_feat.unsqueeze(1), torch.stack(tokens, dim=1)), dim=1) + self.pos_embed
        relation = self.local_transformer(sequence)
        aux_local = self.aux_local_head(relation[:, 1:, :].mean(dim=1))
        return relation[:, 0, :], aux_global, aux_local
