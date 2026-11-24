#!/usr/bin/env python3
"""CPU unit tests for the additive V46 Upgrade features."""
from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import torch

import data_v40
import ir50 as ir50_module
from ir50 import IR50
from model_v42 import FRITNetV42, V42Config
from train_v42 import parse_args, resolve_settings
from torchvision.transforms import InterpolationMode, RandomErasing
from evaluation_utils import classification_metrics


def _build(config):
    return FRITNetV42(config, pretrained_backbone=None)


def test_ir50_skips_resize_for_112_input():
    model = IR50().eval()
    x = torch.randn(1, 3, 112, 112)
    with patch.object(ir50_module.F, "interpolate", side_effect=AssertionError("112 input should not be resized")):
        with torch.inference_mode():
            x1, x2, x3 = model(x)
    assert tuple(x1.shape[-2:]) == (56, 56)
    assert tuple(x2.shape[-2:]) == (28, 28)
    assert tuple(x3.shape[-2:]) == (14, 14)


def test_ir50_keeps_224_compatibility():
    model = IR50().eval()
    with torch.inference_mode():
        _, x2, x3 = model(torch.randn(1, 3, 224, 224))
    assert tuple(x2.shape[-2:]) == (28, 28)
    assert tuple(x3.shape[-2:]) == (14, 14)


def test_ir50_transform_is_direct_112_lanczos_and_random_erasing_is_training_only():
    settings = data_v40.configure_transforms("ir50", random_erasing=True)
    assert settings == {
        "image_size": 112, "interpolation": "lanczos", "random_erasing": True,
        "re_prob": 0.5, "re_scale": [0.02, 0.10],
    }
    train_ops = data_v40.TRAIN_TRANSFORM.transforms
    eval_ops = data_v40.EVAL_TRANSFORM.transforms
    assert train_ops[0].size == (112, 112)
    assert train_ops[0].interpolation == InterpolationMode.LANCZOS
    assert any(isinstance(op, RandomErasing) and op.p == 0.5 and op.scale == (0.02, 0.10) for op in train_ops)
    custom = data_v40.configure_transforms("ir50", random_erasing=True, re_prob=0.2, re_max_scale=0.06)
    custom_erasing = [op for op in data_v40.TRAIN_TRANSFORM.transforms if isinstance(op, RandomErasing)]
    assert custom["re_prob"] == 0.2 and custom["re_scale"] == [0.02, 0.06]
    assert len(custom_erasing) == 1 and custom_erasing[0].p == 0.2 and custom_erasing[0].scale == (0.02, 0.06)
    data_v40.configure_transforms("ir50", random_erasing=True)
    assert not any(isinstance(op, RandomErasing) for op in eval_ops)
    x = data_v40.TRAIN_TRANSFORM(data_v40._read_image(" ".join(["127"] * (48 * 48)), 0))
    assert tuple(x.shape) == (3, 112, 112)
    data_v40.configure_transforms("facenet", random_erasing=False)


def test_tap_ir50_112_identity_and_parameter_delta():
    base_cfg = V42Config.control(num_classes=7, backbone="ir50")
    tap_cfg = deepcopy(base_cfg)
    tap_cfg.use_tap = True
    torch.manual_seed(2301)
    base = _build(base_cfg).eval()
    torch.manual_seed(2301)
    tap = _build(tap_cfg).eval()
    base_keys, tap_state = set(base.state_dict()), tap.state_dict()
    assert all(torch.equal(base.state_dict()[key], tap_state[key]) for key in base_keys)
    assert sum(p.numel() for p in tap.parameters()) - sum(p.numel() for p in base.parameters()) == 16641
    assert float(tap.tap_gate.detach()) == 0.0
    images = torch.randn(2, 3, 112, 112)
    with torch.inference_mode():
        base_logits = base(images)[0]
        tap_logits = tap(images)[0]
    assert torch.allclose(base_logits, tap_logits, atol=1e-6, rtol=0)
    assert tap.extract(images)[0].shape == (2, 128, 28, 28)


def test_silu_gate_zero_identity_parameter_count_and_gradients():
    base_cfg = V42Config.control(num_classes=7, backbone="ir50")
    gate_cfg = deepcopy(base_cfg)
    gate_cfg.use_silu_gate = True
    torch.manual_seed(7654)
    base = _build(base_cfg).eval()
    torch.manual_seed(7654)
    gated = _build(gate_cfg).eval()
    base_state, gated_state = base.state_dict(), gated.state_dict()
    assert all(torch.equal(base_state[key], gated_state[key]) for key in base_state)
    assert sum(p.numel() for p in gated.parameters()) - sum(p.numel() for p in base.parameters()) == 16513
    assert gated.silu_gate_gamma.item() == 0.0
    images = torch.randn(2, 3, 112, 112)
    with torch.inference_mode():
        assert torch.equal(base(images)[0], gated(images)[0])

    gated.train()
    gated.zero_grad(set_to_none=True)
    logits = gated(images)[0]
    logits.square().mean().backward()
    assert gated.silu_gate_gamma.grad is not None
    assert torch.isfinite(gated.silu_gate_gamma.grad)
    assert gated.silu_gate_gamma.grad.abs().item() > 0

    # At gamma=0 the projection is correctly dormant; once gamma moves,
    # projection parameters receive gradients and the gate is learnable.
    with torch.no_grad():
        gated.silu_gate_gamma.fill_(0.1)
    gated.zero_grad(set_to_none=True)
    gated(images)[0].square().mean().backward()
    assert gated.silu_gate_proj.weight.grad is not None
    assert gated.silu_gate_proj.weight.grad.abs().sum().item() > 0


def test_official_evaluation_metric_summary():
    logits = torch.tensor([[4., 0., 0.], [0., 4., 0.], [0., 4., 0.]]).numpy()
    targets = torch.tensor([0, 1, 2]).numpy()
    result = classification_metrics(logits, targets, ["a", "b", "c"])
    assert result["n"] == 3
    assert abs(result["accuracy"] - 2 / 3) < 1e-9
    assert len(result["confusion_rows_true_cols_pred"]) == 3


def test_combined_upgrade_parameter_delta_and_cli_flags():
    base = V42Config.control(num_classes=7, backbone="ir50")
    upgraded = deepcopy(base)
    upgraded.use_tap = True
    upgraded.use_silu_gate = True
    torch.manual_seed(123)
    model = _build(base)
    torch.manual_seed(123)
    upgraded_model = _build(upgraded)
    delta = sum(p.numel() for p in upgraded_model.parameters()) - sum(p.numel() for p in model.parameters())
    assert delta == 33154
    assert upgraded_model.tap_gate.item() == 0.0
    assert upgraded_model.silu_gate_gamma.item() == 0.0
    sample = torch.randn(1, 3, 112, 112)
    model.eval(); upgraded_model.eval()
    with torch.inference_mode():
        assert torch.equal(model(sample)[0], upgraded_model(sample)[0])
    args = parse_args([
        "--dataset", "rafdb", "--preset", "control", "--backbone", "ir50",
        "--tap", "on", "--silu-gate", "--random-erasing",
        "--soft-contrast-weight", "0.05", "--soft-contrast-temperature", "0.10",
    ])
    cfg, settings = resolve_settings(args)
    assert cfg.use_tap and cfg.use_silu_gate
    assert args.random_erasing
    assert settings["soft_contrast_weight"] == 0.05
    assert settings["soft_contrast_temperature"] == 0.10
    tier3_args = parse_args([
        "--dataset", "ferplus", "--sampler", "none", "--hard-weight", "0.0",
        "--use-sam", "--sam-rho", "0.05", "--use-ema", "--ema-decay", "0.999",
        "--random-erasing", "--re-prob", "0.2", "--re-max-scale", "0.06",
    ])
    tier3_cfg, tier3_settings = resolve_settings(tier3_args)
    assert tier3_args.sampler == "none" and tier3_args.use_sam and tier3_args.use_ema
    assert tier3_settings["hard_weight"] == 0.0
    assert tier3_args.re_prob == 0.2 and tier3_args.re_max_scale == 0.06
    raf_args = parse_args(["--dataset", "rafdb", "--label-smoothing", "0.1"])
    _, _ = resolve_settings(raf_args)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__, flush=True)
    print(f"ALL {len(tests)} V46 UPGRADE TESTS PASSED", flush=True)
