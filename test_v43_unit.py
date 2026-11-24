#!/usr/bin/env python3
"""CPU tests for V43 IR-50 support and the V44 multi-scale query head."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from ir50 import IR50, load_ir50_weights
from landmarks import LandmarkCrossAttention, MobileFaceNet, load_landmark_weights
from model_v42 import FRITNetV42, V42Config
from train_v42 import optimizer_groups, parse_args, resolve_settings

TMP = Path(tempfile.mkdtemp())


def build(**kw):
    torch.manual_seed(7)
    return FRITNetV42(V42Config.control(**kw), pretrained_backbone=None)


def test_ir50_shapes():
    with torch.no_grad():
        x1, x2, x3 = IR50().eval()(torch.randn(2, 3, 224, 224))
    assert tuple(x1.shape[1:]) == (64, 56, 56) and tuple(x2.shape[1:]) == (128, 28, 28) and tuple(x3.shape[1:]) == (256, 14, 14)


def test_ir50_stage2_tap_on_with_direct_112_input():
    base_cfg = V42Config.control(num_classes=7, backbone="ir50")
    tap_cfg = V42Config.control(num_classes=7, backbone="ir50")
    tap_cfg.use_tap = True
    torch.manual_seed(1907)
    base = FRITNetV42(base_cfg, pretrained_backbone=None).eval()
    torch.manual_seed(1907)
    tapped = FRITNetV42(tap_cfg, pretrained_backbone=None).eval()
    shared = base.state_dict()
    assert all(torch.equal(shared[key], tapped.state_dict()[key]) for key in shared)
    assert sum(p.numel() for p in tapped.parameters()) - sum(p.numel() for p in base.parameters()) == 16641
    assert tapped.tap_gate.item() == 0.0
    images = torch.randn(2, 3, 112, 112)
    with torch.inference_mode():
        base_logits = base(images)[0]
        tap_logits = tapped(images)[0]
    assert torch.allclose(base_logits, tap_logits, atol=1e-6, rtol=0)
    tap_source, final_map = tapped.extract(images)
    assert tuple(tap_source.shape) == (2, 128, 28, 28)
    assert tuple(final_map.shape) == (2, 256, 14, 14)


def test_ir50_loads_poster_and_original_key_formats():
    torch.manual_seed(1)
    source = IR50().state_dict()
    poster = {"module." + k: v for k, v in source.items()}
    original, offset = {}, {"body1": 0, "body2": 3, "body3": 7}
    for k, v in source.items():
        stage = k.split(".")[0]
        if stage in offset:
            index, rest = k.split(".", 2)[1:]
            original[f"body.{offset[stage] + int(index)}.{rest}"] = v
        else:
            original[k] = v
    original["body.21.res_layer.0.weight"] = torch.zeros(256)  # stage 4 is ignored
    original["output_layer.1.weight"] = torch.zeros(512)
    for name, state in (("poster.pth", poster), ("original.pth", {"state_dict": original})):
        torch.save(state, TMP / name)
        torch.manual_seed(2)
        model = IR50()
        report = load_ir50_weights(model, TMP / name)
        assert report["loaded_tensors"] == len(source)
        assert all(torch.equal(model.state_dict()[k], source[k]) for k in source), name


def test_ir50_incomplete_weights_refused():
    state = IR50().state_dict()
    state.pop("body3.13.res_layer.4.weight")
    torch.save(state, TMP / "broken.pth")
    try:
        load_ir50_weights(IR50(), TMP / "broken.pth")
    except RuntimeError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("incomplete IR-50 weights were accepted")


def test_landmark_weights_strict_and_shape():
    net = MobileFaceNet()
    torch.save({"state_dict": net.state_dict(), "epoch": 1}, TMP / "lm.pth.tar")
    fresh = MobileFaceNet()
    load_landmark_weights(fresh, TMP / "lm.pth.tar")
    assert all(torch.equal(fresh.state_dict()[k], v) for k, v in net.state_dict().items())
    with torch.no_grad():
        assert tuple(fresh.eval()(torch.randn(1, 3, 112, 112)).shape) == (1, 64, 28, 28)


def test_window_split_and_merge_are_inverse():
    block = LandmarkCrossAttention()
    x = torch.randn(2, 128, 28, 28)
    assert torch.equal(block._merge(block._windows(x), 2, 128, 28, 28), x)


def test_landmarks_start_as_identity():
    images = torch.randn(2, 3, 224, 224)
    plain, with_landmarks = build(backbone="ir50"), build(backbone="ir50", landmarks=True)
    shared = plain.state_dict()
    assert all(torch.equal(shared[k], with_landmarks.state_dict()[k]) for k in shared)
    plain.eval(); with_landmarks.eval()
    with torch.no_grad():
        assert torch.allclose(plain(images)[0], with_landmarks(images)[0], atol=1e-6)


def test_landmark_net_frozen_and_gradients_flow():
    model = build(backbone="ir50", landmarks=True).train()
    assert not model.landmarks.net.training
    assert not any(p.requires_grad for p in model.landmarks.parameters())
    model(torch.randn(3, 3, 224, 224))[0].sum().backward()
    assert model.landmark_fusion.gate.grad is not None and model.landmark_fusion.gate.grad.abs() > 0
    assert model.backbone.input_layer[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.landmarks.parameters())


def test_full_variants_shapes_and_optimizer():
    for classes in (7, 8):
        for landmarks in (False, True):
            torch.manual_seed(3)
            model = FRITNetV42(V42Config(num_classes=classes, backbone="ir50", landmarks=landmarks), pretrained_backbone=None)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(2, 3, 224, 224))
            assert tuple(out[0].shape) == (2, classes)
            groups = optimizer_groups(model)
            ids = [id(p) for g in groups for p in g["params"]]
            assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}


def test_v44_query_head_shapes_and_gradients():
    model = FRITNetV42(V42Config.control(num_classes=8, backbone="ir50", head="queries"), pretrained_backbone=None)
    logits, features, aux_global, aux_local = model(torch.randn(2, 3, 224, 224))
    assert tuple(logits.shape) == (2, 8) and tuple(features.shape) == (2, 128)
    assert tuple(aux_global.shape) == (2, 8) and tuple(aux_local.shape) == (2, 8)
    logits.sum().backward()
    assert model.transformer.emotion_queries.grad is not None
    groups = optimizer_groups(model)
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len({id(p) for p in model.parameters()})


def test_v44_query_head_requires_ir50_and_no_relation():
    try:
        FRITNetV42(V42Config.control(head="queries"), pretrained_backbone=None)
    except ValueError as error:
        assert "IR-50" in str(error)
    else:
        raise AssertionError("V44 query head accepted non-IR50 backbone")


def test_settings_switches():
    config, _ = resolve_settings(parse_args(["--preset", "control", "--backbone", "ir50", "--landmarks", "on"]))
    assert config.backbone == "ir50" and config.landmarks and config.num_heads == 8 and not config.use_tap
    config, _ = resolve_settings(parse_args(["--preset", "control"]))
    assert config.backbone == "facenet" and not config.landmarks


def test_parameter_counts():
    for kw in ({"backbone": "ir50"}, {"backbone": "ir50", "landmarks": True}):
        model = build(**kw)
        print(f"   {kw}: total {sum(p.numel() for p in model.parameters()):,} trainable {model.trainable_parameter_count():,}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__, flush=True)
    print(f"ALL {len(tests)} V43/V44 TESTS PASSED")
