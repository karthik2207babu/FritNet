#!/usr/bin/env python3
"""Focused unit tests for FRITNet V46 Upgrade V2 Tier-3 features."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sam import SAM
from train_v42 import (
    EPOCH_LOG_FIELDS,
    ModelEMA,
    batchnorm_no_running_update,
    make_ce_targets,
    make_epoch_log_row,
    make_train_sampler,
    optimizer_step_optimizer,
)
from evaluation_utils import infer_logits


def test_sampler_none_uses_shuffle_protocol():
    sampler, shuffle = make_train_sampler(np.array([0, 0, 0, 1, 1]), "none", 42)
    assert sampler is None and shuffle is True


def test_sampler_balanced_uses_weighted_sampler():
    from torch.utils.data import WeightedRandomSampler
    sampler, shuffle = make_train_sampler(np.array([0, 0, 0, 1, 1]), "balanced", 42)
    assert isinstance(sampler, WeightedRandomSampler) and shuffle is False
    assert len(sampler) == 5


def test_raf_label_smoothing_formula_and_ferplus_unchanged():
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    smoothed = make_ce_targets(target, "rafdb", 0.1, 3)
    expected = (1.0 - 0.1) * target + 0.1 / 3
    assert torch.allclose(smoothed, expected)
    assert make_ce_targets(target, "ferplus", 0.1, 3) is target
    assert torch.allclose(make_ce_targets(target, "rafdb", 0.0, 3), target)


def test_sam_wraps_adamw_and_updates_weights():
    layer = nn.Linear(3, 2)
    before = layer.weight.detach().clone()
    sam = SAM(layer.parameters(), torch.optim.AdamW, rho=0.05, lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(sam.base_optimizer, lr_lambda=lambda _: 0.5)
    x = torch.randn(4, 3)
    loss1 = layer(x).square().mean()
    loss1.backward()
    sam.first_step(zero_grad=True)
    loss2 = layer(x).square().mean()
    loss2.backward()
    sam.second_step(zero_grad=True)
    scheduler.step()
    assert not torch.equal(before, layer.weight.detach())
    assert abs(optimizer_step_optimizer(sam).param_groups[0]["lr"] - 0.005) < 1e-8
    assert not sam._perturbations


def test_sam_cancel_restores_perturbed_parameters():
    layer = nn.Linear(2, 2)
    before = layer.weight.detach().clone()
    sam = SAM(layer.parameters(), torch.optim.AdamW, rho=0.05, lr=0.01)
    layer(torch.ones(1, 2)).sum().backward()
    sam.first_step(zero_grad=True)
    assert not torch.equal(before, layer.weight.detach())
    sam.cancel_step()
    assert torch.allclose(before, layer.weight.detach(), atol=1e-7)


def test_batchnorm_momentum_zero_and_restored():
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2)).train()
    bn = model[0]
    original = bn.momentum
    model(torch.randn(8, 3))
    before_mean = bn.running_mean.clone()
    with batchnorm_no_running_update(model):
        assert bn.momentum == 0.0
        model(torch.randn(8, 3) + 10.0)
        assert torch.equal(before_mean, bn.running_mean)
    assert bn.momentum == original


def test_ema_averages_parameters_and_copies_integer_buffers():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    ema = ModelEMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.0)
        model[1].num_batches_tracked.fill_(7)
    old = [p.detach().clone() for p in ema.ema_model.parameters()]
    ema.update(model)
    for old_p, ema_p, current in zip(old, ema.ema_model.parameters(), model.parameters()):
        assert torch.allclose(ema_p, old_p + 1.0)
    assert int(ema.ema_model[1].num_batches_tracked) == 7
    assert not any(p.requires_grad for p in ema.ema_model.parameters())


class _FirstColumnLogit(nn.Module):
    def forward(self, x):
        # Logit depends on leftmost column; horizontal flip changes it.
        left = x[:, 0, :, 0].mean(dim=1)
        right = x[:, 0, :, -1].mean(dim=1)
        logits = torch.stack((left, right), dim=1)
        return (logits,)


def test_horizontal_flip_tta_averages_logits():
    images = torch.tensor([
        [[[1.0, 3.0], [1.0, 3.0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        [[[2.0, 6.0], [2.0, 6.0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
    ])
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    logits, target = infer_logits(_FirstColumnLogit(), loader, torch.device("cpu"), horizontal_flip=True)
    expected = np.array([[2.0, 2.0], [4.0, 4.0]])
    assert np.allclose(logits, expected)
    assert np.array_equal(target, labels.numpy())


def test_epoch_csv_header_and_row_are_aligned():
    metrics = {
        "accuracy": 0.8, "balanced_accuracy": 0.7, "macro_f1": 0.72,
        "soft_cross_entropy": 0.9, "twin_free": {"accuracy": 0.81},
    }
    row = make_epoch_log_row(1, 1.2, 0.75, 0.0, 0.0, 2.0, -0.1, -0.2,
                             metrics, [1e-5, 2e-4, 3e-4], 10.0)
    assert len(row) == len(EPOCH_LOG_FIELDS)
    assert EPOCH_LOG_FIELDS[7] == "silu_gate_gamma"
    assert row[7] == -0.2 and row[8] == 0.8
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "epoch_log.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(EPOCH_LOG_FIELDS)
            writer.writerow(row)
        with path.open(newline="") as f:
            records = list(csv.reader(f))
        assert len(records[0]) == len(records[1])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__, flush=True)
    print(f"ALL {len(tests)} TESTS PASSED")
