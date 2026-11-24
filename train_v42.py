#!/usr/bin/env python3
"""FRITNet V42--V46 training on FERPlus or RAF-DB.

`--preset all` turns on all six changes. `--preset control` is the V21_CONTROL recipe.
Any single change can be overridden, so each one can be ablated later.

FERPlus supports either the internal 90/10 split (`--ferplus-split internal90`) or
the official 3-way split (`--ferplus-split official`: Training / PublicTest / untouched
PrivateTest). RAF-DB uses train_labels.csv only, and its test images are never extracted.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

import data_raf_v42 as raf
import data_v40 as fer_data
from data_v40 import (
    FERPLUS_ORDER,
    FERPlusV40ValidationDataset,
    _read_image,
    prepare_training_frame,
    stratified_train_val_indices,
)
from model_v42 import FRITNetV42, V42Config
from sam import SAM
from protocol_v41 import ValidationContext, evaluate_logits, majority_keep_all_rows, pixel_digests

OUTPUT_ROOT = "/content/drive/MyDrive/karthik_nitish"
PRETRAIN_DIR = f"{OUTPUT_ROOT}/pretrain"
CONTROLS = {
    42: "/content/drive/MyDrive/FRITNet_V21_CONTROL_VGGFACE2_seed42_20261106_044018",
    43: "/content/drive/MyDrive/FRITNet_V21_CONTROL_VGGFACE2_seed43_20261106_060849",
}
EXPECTED_TWINS = {42: 186, 43: 215}
EXPECTED_PUBLIC_MAJORITY = 3199
BACKBONE_LR, TASK_LR, CLASSIFIER_LR = 1.5e-5, 3.0e-4, 4.0e-4
WARMUP_EPOCHS = 3


class FerPlusTrainDataset(Dataset):
    def __init__(self, frame, indices):
        self.pixels = frame["pixels"].to_numpy(dtype=object)[indices]
        self.soft = torch.from_numpy(np.stack(frame["soft_label"].to_numpy()).astype(np.float32))[indices]
        self.hard = torch.from_numpy(frame["hard_label"].to_numpy(dtype=np.int64))[indices]

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        return fer_data.TRAIN_TRANSFORM(_read_image(self.pixels[index], index)), self.soft[index], self.hard[index]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("ferplus", "rafdb"), default="ferplus")
    p.add_argument("--ferplus-split", choices=("internal90", "official"), default="internal90",
                   help="FERPlus only: internal90 or official Training/PublicTest/PrivateTest split.")
    p.add_argument("--preset", choices=("all", "control"), default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-dir", default="")
    p.add_argument("--raf-zip", default=raf.DEFAULT_ZIP)
    p.add_argument("--raf-root", default="", help="Existing RAF-DB folder, instead of extracting the zip.")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Two tiny epochs to prove the pipeline; never a result.")
    p.add_argument("--skip-control-check", action="store_true", help="Smoke tests away from Drive only.")
    # Single-change overrides, for later ablation
    p.add_argument("--tap", choices=("on", "off"))
    p.add_argument("--silu-gate", action="store_true",
                   help="V46 Upgrade: zero-initialized residual SiLU gate on the 128-d feature before the classifier.")
    p.add_argument("--random-erasing", action="store_true",
                   help="V46 Upgrade: enable training-only Random Erasing.")
    p.add_argument("--re-prob", type=float, default=0.5,
                   help="Random Erasing probability when --random-erasing is enabled (default: 0.5).")
    p.add_argument("--re-max-scale", type=float, default=0.10,
                   help="Maximum Random Erasing area fraction; minimum is fixed at 0.02 (default: 0.10).")
    p.add_argument("--heads", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--parallel-gates", choices=("on", "off"))
    p.add_argument("--relation", choices=("on", "off"))
    p.add_argument("--relation-beta", type=float)
    p.add_argument("--relation-weight", type=float)
    p.add_argument("--aux-weight", type=float)
    p.add_argument("--mixup-prob", type=float)
    # V45 loss-only ablations.  They leave the network graph unchanged.
    p.add_argument("--pair-margin-pairs", choices=("none", "sadness-neutral", "sadness-neutral+surprise-fear"),
                   default="none", help="FERPlus soft-label pair(s) to receive a directional ranking margin.")
    p.add_argument("--pair-margin", type=float, default=0.35,
                   help="Required logit gap for each active V45 pair.")
    p.add_argument("--pair-margin-weight", type=float, default=0.0,
                   help="Multiplier for the V45 pair-margin loss; 0 disables it.")
    p.add_argument("--pair-margin-min-gap", type=float, default=0.20,
                   help="Only apply a pair margin when the two soft-label masses differ by at least this amount.")
    p.add_argument("--class-weighting", choices=("off", "invsqrt"), default="off",
                   help="Optional V45B primary-soft-CE weighting.  Keep off for V45A.")
    p.add_argument("--soft-contrast-weight", type=float, default=0.0,
                   help="V46: coefficient for vote-distribution-aware contrastive feature loss.")
    p.add_argument("--soft-contrast-temperature", type=float, default=0.10,
                   help="V46 cosine-similarity temperature.")
    # Tier 3 optimization, sampling and target-shape ablations.
    p.add_argument("--sampler", choices=("balanced", "none"), default="balanced",
                   help="balanced uses inverse hard-class frequency WeightedRandomSampler; none uses shuffled natural sampling.")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="RAF-DB only: epsilon for smoothing one-hot targets before soft cross entropy.")
    p.add_argument("--hard-weight", type=float, default=None,
                   help="Override preset hard-label CE coefficient; e.g. 0.0 disables it on FERPlus.")
    p.add_argument("--use-sam", action="store_true", help="Use SAM wrapping AdamW (two forward/backward passes per batch).")
    p.add_argument("--sam-rho", type=float, default=0.05, help="SAM neighborhood radius (default: 0.05).")
    p.add_argument("--use-ema", action="store_true", help="Maintain EMA model weights and validate/save the EMA model.")
    p.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay (default: 0.999).")
    # V43 changes
    p.add_argument("--backbone", choices=("facenet", "ir50"), default="facenet")
    p.add_argument("--landmarks", choices=("on", "off"), default="off")
    p.add_argument("--head", choices=("legacy", "queries"), default="legacy",
                   help="V44: legacy five-token transformer or multi-scale emotion-query head.")
    p.add_argument("--ir50-weights", default=f"{PRETRAIN_DIR}/ir50.pth")
    p.add_argument("--landmark-weights", default=f"{PRETRAIN_DIR}/mobilefacenet_model_best.pth.tar")
    return p.parse_args(argv)


def resolve_settings(args):
    classes = 8 if args.dataset == "ferplus" else 7
    config = V42Config(num_classes=classes) if args.preset == "all" else V42Config.control(classes)
    settings = ({"aux_weight": 0.50, "mixup_prob": 0.50, "relation_weight": 0.50} if args.preset == "all"
                else {"aux_weight": 0.10, "mixup_prob": 0.0, "relation_weight": 0.0})
    settings.update(mixup_alpha=0.20, mixup_stop_fraction=0.80,
                    hard_weight=0.15 if args.dataset == "ferplus" else 0.0,
                    pair_margin_pairs=args.pair_margin_pairs, pair_margin=args.pair_margin,
                    pair_margin_weight=args.pair_margin_weight, pair_margin_min_gap=args.pair_margin_min_gap,
                    class_weighting=args.class_weighting, class_weights=None,
                    soft_contrast_weight=args.soft_contrast_weight,
                    soft_contrast_temperature=args.soft_contrast_temperature)
    if args.hard_weight is not None:
        settings["hard_weight"] = float(args.hard_weight)
    config.backbone = getattr(args, "backbone", "facenet")
    config.landmarks = getattr(args, "landmarks", "off") == "on"
    config.head = args.head
    config.use_silu_gate = bool(args.silu_gate)
    if args.tap:
        config.use_tap = args.tap == "on"
    if args.heads is not None:
        config.num_heads = args.heads
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.parallel_gates:
        config.parallel_gates = args.parallel_gates == "on"
    if args.relation:
        config.use_relation = args.relation == "on"
    if args.relation_beta is not None:
        config.relation_beta = args.relation_beta
    if args.relation_weight is not None:
        settings["relation_weight"] = args.relation_weight
    if args.aux_weight is not None:
        settings["aux_weight"] = args.aux_weight
    if args.mixup_prob is not None:
        settings["mixup_prob"] = args.mixup_prob
    if config.use_silu_gate and config.head != "legacy":
        raise ValueError("--silu-gate is supported only with --head legacy.")
    if config.use_relation and not settings["relation_weight"]:
        settings["relation_weight"] = 0.50
    if not config.use_relation:
        settings["relation_weight"] = 0.0
    if args.dataset != "ferplus" and (settings["pair_margin_weight"] or settings["class_weighting"] != "off"):
        raise ValueError("V45 pair-margin and class-weight ablations are defined only for FERPlus.")
    if settings["pair_margin_weight"] and settings["pair_margin_pairs"] == "none":
        raise ValueError("--pair-margin-weight requires --pair-margin-pairs.")
    if settings["pair_margin"] <= 0 or not 0 <= settings["pair_margin_min_gap"] <= 1:
        raise ValueError("Pair margin must be positive and --pair-margin-min-gap must be in [0, 1].")
    if settings["soft_contrast_weight"] < 0 or settings["soft_contrast_temperature"] <= 0:
        raise ValueError("V46 contrast weight must be nonnegative and temperature positive.")
    if settings["hard_weight"] < 0:
        raise ValueError("--hard-weight must be nonnegative.")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1).")
    if args.dataset != "rafdb" and args.label_smoothing > 0:
        raise ValueError("--label-smoothing is currently supported only for RAF-DB.")
    if not 0.0 <= args.re_prob <= 1.0 or not 0.02 <= args.re_max_scale <= 1.0:
        raise ValueError("Random Erasing requires --re-prob in [0,1] and --re-max-scale in [0.02,1].")
    if args.use_sam and args.sam_rho <= 0:
        raise ValueError("--sam-rho must be positive when --use-sam is enabled.")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in (0, 1).")
    return config, settings


def build_model(config, args):
    """Pretrained model for training. IR-50 and landmark weights are read from Drive."""
    missing = [path for needed, path in ((config.backbone == "ir50", args.ir50_weights),
                                         (config.landmarks, args.landmark_weights)) if needed and not Path(path).is_file()]
    if missing and args.smoke:
        print(f"[SMOKE] weights not found {missing}; random weights, pipeline check only", flush=True)
        return FRITNetV42(config, pretrained_backbone=None)
    if missing:
        raise FileNotFoundError(f"Weights file not found: {missing[0]}. Run the download cell first.")
    return FRITNetV42(config, pretrained_backbone="vggface2", ir50_weights=args.ir50_weights,
                      landmark_weights=args.landmark_weights)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def atomic_save(payload, destination: Path):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def optimizer_groups(model):
    backbone = list(model.backbone.parameters())
    # V44's query head owns its class-specific projection inside the transformer.
    classifier = list(model.classifier.parameters()) if model.classifier is not None else []
    skip = {id(p) for p in backbone + classifier}
    task = [p for p in model.parameters() if id(p) not in skip]
    return [{"params": backbone, "lr": BACKBONE_LR, "name": "backbone"},
            {"params": task, "lr": TASK_LR, "name": "task"},
            {"params": classifier, "lr": CLASSIFIER_LR, "name": "classifier"}]


def lr_schedule(epochs):
    def factor(epoch_index):
        if epoch_index < WARMUP_EPOCHS:
            return float(epoch_index + 1) / WARMUP_EPOCHS
        progress = (epoch_index - WARMUP_EPOCHS) / max(1, epochs - WARMUP_EPOCHS)
        return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
    return factor


def make_ce_targets(soft_targets, dataset: str, epsilon: float, num_classes: int):
    """Apply conventional uniform label smoothing to RAF-DB CE targets only.

    The original targets are retained for V46 vote-aware contrastive and any
    agreement-aware auxiliary losses; only primary/auxiliary soft CE sees the
    smoothed distribution.
    """
    if dataset != "rafdb" or epsilon <= 0:
        return soft_targets
    return (1.0 - epsilon) * soft_targets + epsilon / float(num_classes)


def make_train_sampler(labels, sampler_name: str, seed: int):
    """Return (sampler, shuffle) for the selected training sampling protocol."""
    if sampler_name == "none":
        return None, True
    if sampler_name != "balanced":
        raise ValueError(f"Unknown sampler: {sampler_name}")
    counts = np.bincount(np.asarray(labels, dtype=np.int64))
    weights = 1.0 / np.maximum(counts, 1)[np.asarray(labels, dtype=np.int64)]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels), replacement=True,
        generator=torch.Generator().manual_seed(int(seed) + 101),
    )
    return sampler, False


@contextmanager
def batchnorm_no_running_update(model):
    """Temporarily set BatchNorm momentum to zero for SAM's second pass."""
    modules = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    previous = [(module, module.momentum) for module in modules]
    try:
        for module, _ in previous:
            module.momentum = 0.0
        yield
    finally:
        for module, momentum in previous:
            module.momentum = momentum


class ModelEMA:
    """EMA shadow model, including floating buffers and integer BN counters."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
            if torch.is_floating_point(ema_buffer):
                ema_buffer.mul_(self.decay).add_(buffer.detach(), alpha=1.0 - self.decay)
            else:
                ema_buffer.copy_(buffer)
        self.ema_model.eval()


def optimizer_step_optimizer(optimizer):
    """Return the optimizer whose parameter groups are stepped by LR scheduler."""
    return optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer


EPOCH_LOG_FIELDS = [
    "epoch", "train_loss", "train_accuracy", "relation_ce", "pair_margin", "soft_contrast",
    "tap_gate", "silu_gate_gamma", "val_accuracy", "val_balanced_accuracy", "val_macro_f1",
    "val_soft_ce", "val_twin_free_accuracy", "lr_backbone", "lr_task", "lr_classifier", "seconds",
]


def make_epoch_log_row(epoch, train_loss, train_accuracy, relation_mean, pair_mean, contrast_mean,
                       tap_gate, silu_gamma, metrics, lrs, seconds):
    """Build a CSV row in exactly the same order as EPOCH_LOG_FIELDS."""
    row = [epoch, train_loss, train_accuracy, relation_mean, pair_mean, contrast_mean,
           tap_gate, silu_gamma, metrics["accuracy"], metrics["balanced_accuracy"],
           metrics["macro_f1"], metrics["soft_cross_entropy"],
           metrics["twin_free"]["accuracy"], *lrs, seconds]
    if len(row) != len(EPOCH_LOG_FIELDS):
        raise RuntimeError(
            f"epoch_log.csv row/header mismatch: {len(row)} values vs {len(EPOCH_LOG_FIELDS)} columns."
        )
    return row


def soft_ce(logits, targets, class_weights=None):
    """Soft cross entropy, optionally with normalised per-class weights.

    Normalising each row retains a unit loss scale and prevents the optional
    inverse-square-root weights from silently multiplying the whole objective.
    """
    targets = targets.float()
    per_row = -(targets * F.log_softmax(logits.float(), dim=1)).sum(dim=1)
    if class_weights is not None:
        weights = logits.new_tensor(class_weights, dtype=torch.float32)
        weighted_targets = targets * weights
        per_row = -(weighted_targets * F.log_softmax(logits.float(), dim=1)).sum(dim=1)
        per_row = per_row / weighted_targets.sum(dim=1).clamp_min(1e-8)
    return per_row.mean()


PAIR_SETS = {
    "none": (),
    "sadness-neutral": ((4, 6),),
    "sadness-neutral+surprise-fear": ((4, 6), (0, 1)),
}


def soft_pair_margin(logits, targets, pair_set, margin, min_gap):
    """Agreement-gated directional logit margins for FERPlus soft targets.

    A pair contributes only when annotators gave one member at least
    ``min_gap`` more probability than the other.  Thus ambiguous sadness /
    neutral faces are deliberately *not* forced apart.  Images whose votes
    favour neither member also receive no pair loss.
    """
    losses = []
    for left, right in PAIR_SETS[pair_set]:
        vote_gap = targets[:, left].float() - targets[:, right].float()
        active = vote_gap.abs() >= min_gap
        if active.any():
            direction = vote_gap[active].sign()
            logit_gap = logits.float()[active, left] - logits.float()[active, right]
            # Larger vote separation is trusted more, without changing the
            # relative scale of a batch with a different active count.
            strength = vote_gap[active].abs()
            losses.append((F.relu(margin - direction * logit_gap) * strength).sum() / strength.sum().clamp_min(1e-8))
    return torch.stack(losses).mean() if losses else logits.float().sum() * 0.0


def soft_label_contrastive(features, targets, temperature=0.10):
    """Vote-aware supervised contrastive loss on the existing feature vectors.

    Human vote distributions define a soft target over other examples in the
    minibatch (dot product of distributions). Entropy reduces the influence of
    uncertain anchors. No projection head or inference-time parameters are added.
    """
    z = F.normalize(features.float(), p=2, dim=1, eps=1e-8)
    y = targets.float().clamp_min(0)
    y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-8)
    n = z.shape[0]
    if n < 2:
        return features.float().sum() * 0.0
    logits = (z @ z.T) / temperature
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    affinity = (y @ y.T).clamp_min(0)
    affinity = affinity.masked_fill(eye, 0.0)
    row_mass = affinity.sum(dim=1, keepdim=True)
    target_dist = affinity / row_mass.clamp_min(1e-8)
    valid = row_mass.squeeze(1) > 1e-8

    entropy = -(y * y.clamp_min(1e-8).log()).sum(dim=1) / np.log(y.shape[1])
    reliability = (1.0 - entropy).clamp(0.0, 1.0)
    anchor_weight = reliability * valid.float()
    per_anchor = -(target_dist * torch.nan_to_num(log_prob, neginf=0.0)).sum(dim=1)
    return (per_anchor * anchor_weight).sum() / anchor_weight.sum().clamp_min(1e-8)


def mix_batch(images, soft, hard, alpha, rng):
    """Soft-label Mixup. Returns mixed images and targets, plus both hard labels and lambda."""
    lam = float(rng.beta(alpha, alpha))
    order = torch.from_numpy(rng.permutation(images.shape[0])).to(images.device)
    mixed = lam * images + (1.0 - lam) * images[order]
    return mixed, lam * soft + (1.0 - lam) * soft[order], hard[order], lam


def total_loss(model, outputs, soft, hard, hard_b, lam, settings, soft_ce_targets=None):
    logits, features, aux_global, aux_local = outputs
    ce_targets = soft if soft_ce_targets is None else soft_ce_targets
    primary = soft_ce(logits, ce_targets, settings["class_weights"])
    loss = primary + settings["aux_weight"] * (soft_ce(aux_global, ce_targets) + soft_ce(aux_local, ce_targets))
    if settings["hard_weight"]:
        hard_ce = lam * F.cross_entropy(logits.float(), hard)
        if lam < 1.0:
            hard_ce = hard_ce + (1.0 - lam) * F.cross_entropy(logits.float(), hard_b)
        loss = loss + settings["hard_weight"] * hard_ce
    relation_ce = None
    if settings["relation_weight"] and model.config.use_relation:
        relation_ce = soft_ce(model.classifier(model.relation(features, soft)), ce_targets)
        loss = loss + settings["relation_weight"] * relation_ce
    pair_loss = soft_pair_margin(logits, soft, settings["pair_margin_pairs"], settings["pair_margin"],
                                 settings["pair_margin_min_gap"])
    loss = loss + settings["pair_margin_weight"] * pair_loss
    contrast_loss = soft_label_contrastive(features, soft, settings["soft_contrast_temperature"])
    loss = loss + settings["soft_contrast_weight"] * contrast_loss
    return loss, relation_ce, pair_loss, contrast_loss


@torch.inference_mode()
def validation_logits(model, loader, device, max_batches=0):
    model.eval()
    collected = []
    for step, batch in enumerate(tqdm(loader, desc="Validation (FP32)", leave=False, dynamic_ncols=True)):
        if max_batches and step >= max_batches:
            break
        images = batch[0].to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        collected.append(model(images)[0].float().cpu().numpy())
    return np.concatenate(collected)


def raf_metrics(logits, hard, twin):
    pred = logits.argmax(1)

    def block(t, p):
        confusion = np.zeros((7, 7), dtype=np.int64)
        np.add.at(confusion, (t, p), 1)
        tp = np.diag(confusion).astype(float)
        recall = np.divide(tp, confusion.sum(1), out=np.zeros(7), where=confusion.sum(1) > 0)
        precision = np.divide(tp, confusion.sum(0), out=np.zeros(7), where=confusion.sum(0) > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(7), where=(precision + recall) > 0)
        return {"n": int(len(t)), "accuracy": float((t == p).mean()), "balanced_accuracy": float(recall.mean()),
                "macro_f1": float(f1.mean()),
                "per_class": {raf.RAF_CLASSES[i]: {"support": int(confusion[i].sum()), "precision": float(precision[i]),
                                                   "recall": float(recall[i]), "f1": float(f1[i])} for i in range(7)},
                "confusion_rows_true_cols_pred": confusion.tolist()}

    result = block(hard, pred)
    result["twin_free"] = block(hard[~twin], pred[~twin])
    z = logits - logits.max(1, keepdims=True)
    result["soft_cross_entropy"] = float(-(z - np.log(np.exp(z).sum(1, keepdims=True)))[np.arange(len(hard)), hard].mean())
    return result


def check_ferplus_control(seed, csv_rows, train_idx, val_idx):
    folder = CONTROLS.get(seed)
    if folder is None:
        raise ValueError(f"No saved FERPlus control for seed {seed}.")
    path = Path(folder) / "config.json"
    if not path.is_file():
        print(f"[CONTROL] {path} not found; the split is checked by its twin count only", flush=True)
        return
    config = json.loads(path.read_text(encoding="utf-8"))
    for key, indices in (("train_original_csv_rows", train_idx), ("validation_original_csv_rows", val_idx)):
        saved = config.get(key)
        if saved is None or sorted(saved) != sorted(csv_rows[indices].tolist()):
            raise RuntimeError(f"{key} do not match the control. The comparison would be unpaired.")
    print(f"[CONTROL] train and validation row IDs match {Path(folder).name}", flush=True)


def prepare_ferplus_official_frames():
    """Load Training and PublicTest pixels; never load PrivateTest votes or pixels."""
    pd = fer_data.pd

    pixels_path = fer_data._resolve(
        "FERPLUS_PIXELS_CSV", fer_data.PIXELS_CANDIDATES, "FER2013 pixel CSV"
    )
    labels_path = fer_data._resolve(
        "FERPLUS_LABELS_CSV", fer_data.LABELS_CANDIDATES, "FERPlus annotation CSV"
    )

    header = pd.read_csv(labels_path, nrows=0)
    columns = {str(c).strip().lower(): c for c in header.columns}

    required = set(fer_data.ALL_VOTES + fer_data.FERPLUS_ORDER + ["usage"])
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"FERPlus columns missing: {missing}")

    # Read usage metadata only to identify allowed rows.
    usage_col = columns["usage"]
    usage_all = (
        pd.read_csv(labels_path, usecols=[usage_col])[usage_col]
        .astype(str).str.strip().str.lower().to_numpy()
    )

    if len(usage_all) != fer_data.EXPECTED_RAW_ROWS:
        raise RuntimeError(f"Unexpected FERPlus row count: {len(usage_all)}")

    allowed_rows = np.flatnonzero(
        np.isin(usage_all, ["training", "publictest"])
    )

    allowed_mask = np.zeros(len(usage_all), dtype=bool)
    allowed_mask[allowed_rows] = True

    def skip_private_annotation(line_number):
        return (
            line_number > 0
            and (
                line_number - 1 >= len(allowed_mask)
                or not allowed_mask[line_number - 1]
            )
        )

    selected_cols = [
        columns[c] for c in fer_data.ALL_VOTES + fer_data.FERPLUS_ORDER
    ]

    # PrivateTest vote annotations are skipped by the CSV parser.
    selected_labels = pd.read_csv(
        labels_path,
        usecols=selected_cols,
        skiprows=skip_private_annotation
    )

    if len(selected_labels) != len(allowed_rows):
        raise RuntimeError("Training/PublicTest annotation alignment failed.")

    all_votes = selected_labels[
        [columns[c] for c in fer_data.ALL_VOTES]
    ].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)

    raw_emotion = selected_labels[
        [columns[c] for c in fer_data.FERPLUS_ORDER]
    ].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)

    rounded = np.rint(raw_emotion)
    if not np.allclose(raw_emotion, rounded, atol=1e-6):
        raise ValueError("Non-integer FERPlus emotion votes detected.")

    emotion_votes = rounded.astype(np.int64)

    total_all = all_votes.sum(axis=1)
    unknown_nf = all_votes[
        :, [
            fer_data.ALL_VOTES.index("unknown"),
            fer_data.ALL_VOTES.index("nf")
        ]
    ].sum(axis=1)

    unknown_fraction = np.divide(
        unknown_nf, total_all,
        out=np.ones_like(unknown_nf),
        where=total_all > 0
    )

    total_emotion = emotion_votes.sum(axis=1)

    valid = (
        (total_all > 0)
        & (unknown_fraction <= 0.50)
        & (total_emotion > 0)
    )

    usage_selected = usage_all[allowed_rows]

    train_valid = valid & (usage_selected == "training")
    public_valid = valid & (usage_selected == "publictest")

    retained = {
        "training": int(train_valid.sum()),
        "publictest": int(public_valid.sum())
    }

    expected = {
        "training": fer_data.EXPECTED_RETAINED["training"],
        "publictest": fer_data.EXPECTED_RETAINED["publictest"]
    }

    print("[V21 FILTER / OFFICIAL 3-WAY]", retained, flush=True)

    if retained != expected:
        raise RuntimeError(
            f"Official FERPlus count mismatch. Expected {expected}, got {retained}."
        )

    valid_local = np.flatnonzero(valid)
    original_rows = allowed_rows[valid_local]

    permitted_pixels = np.zeros(len(usage_all), dtype=bool)
    permitted_pixels[original_rows] = True

    def skip_unselected_pixels(line_number):
        return (
            line_number > 0
            and (
                line_number - 1 >= len(permitted_pixels)
                or not permitted_pixels[line_number - 1]
            )
        )

    # PrivateTest pixel records are never loaded.
    pixels_df = pd.read_csv(
        pixels_path,
        usecols=["pixels"],
        skiprows=skip_unselected_pixels
    )

    pixels = pixels_df["pixels"].astype(str).to_numpy()

    if len(pixels) != len(original_rows):
        raise RuntimeError("Selected FERPlus pixel/label alignment failed.")

    lengths = np.fromiter(
        (len(x.split()) for x in pixels),
        dtype=np.int32,
        count=len(pixels)
    )

    bad = np.flatnonzero(lengths != 48 * 48)
    if len(bad):
        raise ValueError(
            f"Invalid FERPlus image: {lengths[int(bad[0])]} pixels."
        )

    votes = emotion_votes[valid_local]
    soft = (votes / votes.sum(axis=1, keepdims=True)).astype(np.float32)
    hard = soft.argmax(axis=1).astype(np.int64)

    majority_votes = all_votes[valid_local].astype(np.float64, copy=True)
    majority_votes[majority_votes <= 1] = 0

    majority = (
        (majority_votes.max(axis=1) > 0.5 * majority_votes.sum(axis=1))
        & (majority_votes.argmax(axis=1) < 8)
    )

    frame = pd.DataFrame({
        "pixels": pixels,
        "soft_label": list(soft),
        "hard_label": hard,
        "vote_counts": list(votes),
        "original_csv_row": original_rows.astype(np.int64),
        "usage": usage_selected[valid_local],
        "majority_keep": majority
    })

    train_class_counts = np.bincount(
        frame.loc[frame["usage"] == "training", "hard_label"],
        minlength=8
    )

    summary = {
        "raw_rows": fer_data.EXPECTED_RAW_ROWS,
        "retained_split_counts": retained,
        "training_class_counts_hard_argmax": {
            fer_data.FERPLUS_ORDER[i]: int(train_class_counts[i])
            for i in range(8)
        },
        "pixels_path": str(pixels_path),
        "labels_path": str(labels_path),
        "training_rows_pixel_count_checked": retained["training"],
        "public_validation_rows_pixel_count_checked": retained["publictest"],
        "private_test_expected_rows": fer_data.EXPECTED_RETAINED["privatetest"],
        "private_test_images_used": False,
        "private_test_pixel_rows_loaded": 0,
        "private_test_vote_rows_loaded": 0
    }

    return frame, summary


def prepare_ferplus(args):
    if args.ferplus_split == "official":
        frame, summary = prepare_ferplus_official_frames()

        usage = frame["usage"].to_numpy()
        train_idx = np.flatnonzero(usage == "training")
        val_idx = np.flatnonzero(usage == "publictest")

        hard = frame["hard_label"].to_numpy(np.int64)
        votes = np.stack(frame["vote_counts"].to_numpy()).astype(np.float64)
        rows = frame["original_csv_row"].to_numpy()

        if len(train_idx) != fer_data.EXPECTED_RETAINED["training"]:
            raise RuntimeError("Incorrect official Training count.")
        if len(val_idx) != fer_data.EXPECTED_RETAINED["publictest"]:
            raise RuntimeError("Incorrect official PublicTest count.")

        if np.intersect1d(rows[train_idx], rows[val_idx]).size:
            raise RuntimeError("Training and PublicTest rows overlap.")

        train_hashes = pixel_digests(frame.iloc[train_idx]["pixels"])
        val_hashes = pixel_digests(frame.iloc[val_idx]["pixels"])
        twin = np.isin(val_hashes, train_hashes)

        majority = frame.iloc[val_idx]["majority_keep"].to_numpy(dtype=bool)

        if int(majority.sum()) != EXPECTED_PUBLIC_MAJORITY:
            raise RuntimeError(
                f"PublicTest majority-label count mismatch: {int(majority.sum())}"
            )

        context_all = ValidationContext(
            votes[val_idx], hard[val_idx], twin, majority
        )

        print(
            f"[FERPLUS OFFICIAL 3-WAY] "
            f"train={len(train_idx):,} "
            f"public_validation={len(val_idx):,} "
            f"private_test=3,569 untouched "
            f"| public/train twins={int(twin.sum())}",
            flush=True
        )

        print(
            "[LEAKAGE POLICY] PrivateTest vote annotations and pixel rows "
            "were not loaded. PrivateTest is excluded from training and "
            "checkpoint selection.",
            flush=True
        )

        def evaluate(logits):
            n = len(logits)
            context = (
                context_all if n == len(val_idx)
                else ValidationContext(
                    votes[val_idx][:n],
                    hard[val_idx][:n],
                    twin[:n],
                    majority[:n]
                )
            )
            return evaluate_logits(logits, context), context.top_share

        summary.update({
            "ferplus_split": "official_3way_public_val_private_test",
            "private_test_images_used": False,
            "publictest_used_for_validation": True
        })

        return {
            "train_set": FerPlusTrainDataset(frame, train_idx),
            "val_set": FERPlusV40ValidationDataset(frame, val_idx),
            "y_train": hard[train_idx],
            "hard_val": hard[val_idx],
            "twin": twin,
            "rows": rows,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "evaluate": evaluate,
            "summary": summary
        }

    frame, summary = prepare_training_frame()
    hard = frame["hard_label"].to_numpy(np.int64)
    votes = np.stack(frame["vote_counts"].to_numpy()).astype(np.float64)
    rows = frame["original_csv_row"].to_numpy()
    train_idx, val_idx = stratified_train_val_indices(hard, 0.10, args.seed)
    digests = pixel_digests(frame["pixels"])
    twin = np.isin(digests[val_idx], digests[train_idx])
    keep, usage = majority_keep_all_rows(summary["labels_path"])
    public_majority = int((keep & (usage == "publictest")).sum())  # label rows only, no images
    print(f"[PROTOCOL] twin rows={int(twin.sum())} | majority rule on PublicTest labels={public_majority}", flush=True)
    if public_majority != EXPECTED_PUBLIC_MAJORITY:
        raise RuntimeError(f"Majority rule gave {public_majority}, expected {EXPECTED_PUBLIC_MAJORITY}.")
    if args.seed in EXPECTED_TWINS and int(twin.sum()) != EXPECTED_TWINS[args.seed]:
        raise RuntimeError(f"Twin count {int(twin.sum())} differs from {EXPECTED_TWINS[args.seed]}.")
    if not args.skip_control_check:
        check_ferplus_control(args.seed, rows, train_idx, val_idx)
    context_all = ValidationContext(votes[val_idx], hard[val_idx], twin, keep[rows[val_idx]])

    def evaluate(logits):
        n = len(logits)
        context = context_all if n == len(val_idx) else ValidationContext(
            votes[val_idx][:n], hard[val_idx][:n], twin[:n], keep[rows[val_idx]][:n])
        return evaluate_logits(logits, context), context.top_share

    return {
        "train_set": FerPlusTrainDataset(frame, train_idx), "val_set": FERPlusV40ValidationDataset(frame, val_idx),
        "y_train": hard[train_idx], "hard_val": hard[val_idx], "twin": twin, "rows": rows, "train_idx": train_idx,
        "val_idx": val_idx, "evaluate": evaluate, "summary": summary,
    }


def prepare_rafdb(args):
    root = Path(args.raf_root) if args.raf_root else raf.extract_train_only(args.raf_zip)
    paths, labels = raf.load_train_table(root)
    train_idx, val_idx = raf.stratified_split(labels, args.seed)
    digests = raf.file_digests(paths)
    twin = np.isin(digests[val_idx], digests[train_idx])
    print(f"[RAF-DB] train={len(train_idx):,} validation={len(val_idx):,} exact twin rows={int(twin.sum())}", flush=True)
    hard_val = labels[val_idx]

    def evaluate(logits):
        n = len(logits)
        return raf_metrics(logits, hard_val[:n], twin[:n]), np.ones(n)

    return {
        "train_set": raf.RafTrainDataset(paths[train_idx], labels[train_idx]),
        "val_set": raf.RafValidationDataset(paths[val_idx], labels[val_idx]),
        "y_train": labels[train_idx], "hard_val": hard_val, "twin": twin, "rows": np.arange(len(labels)),
        "train_idx": train_idx, "val_idx": val_idx, "evaluate": evaluate,
        "summary": {"root": str(root), "train_partition_rows": len(labels), "test_images_used": False},
    }


def main():
    args = parse_args()
    if args.dataset != "ferplus" and args.ferplus_split != "internal90":
        raise ValueError("--ferplus-split is only valid with --dataset ferplus.")
    if args.smoke:
        args.epochs, args.patience = 2, 2
    max_batches = 3 if args.smoke else 0
    config, settings = resolve_settings(args)
    transform_config = fer_data.configure_transforms(
        config.backbone, random_erasing=args.random_erasing,
        re_prob=args.re_prob, re_max_scale=args.re_max_scale,
    )
    # RAF datasets resolve these shared transforms dynamically through data_v40.
    seed_everything(args.seed)
    data = prepare_ferplus(args) if args.dataset == "ferplus" else prepare_rafdb(args)
    if settings["class_weighting"] == "invsqrt":
        counts = np.bincount(data["y_train"], minlength=config.num_classes).astype(np.float64)
        # The sampler is already inverse-frequency balanced.  This optional,
        # gentler correction is deliberately retained only as V45B and is
        # recorded in config.json for a reproducible comparison.
        settings["class_weights"] = np.sqrt(counts.max() / np.maximum(counts, 1)).round(8).tolist()
    print(f"[SPLIT] train={len(data['train_idx']):,} validation={len(data['val_idx']):,}", flush=True)
    print("[INPUT]", json.dumps(transform_config), flush=True)
    print("[SETTINGS]", json.dumps({**config.as_dict(), **settings,
                                      "random_erasing": args.random_erasing}), flush=True)

    if args.preflight_only:
        if config.backbone == "ir50" or config.landmarks:
            check = build_model(config, args)
            print("[WEIGHTS]", json.dumps(check.load_report), f"trainable={check.trainable_parameter_count():,}", flush=True)
        print(json.dumps({"status": "PREFLIGHT_PASS_NO_TRAINING", "dataset": args.dataset, "preset": args.preset,
                          "seed": args.seed, "train": len(data["train_idx"]), "validation": len(data["val_idx"]),
                          "twin_rows": int(data["twin"].sum()), "cuda": torch.cuda.is_available()}, indent=2), flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke:
        raise RuntimeError("CUDA GPU required. In Colab select an L4 runtime.")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This recipe expects BF16. Select an L4 runtime.")
    amp = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if device.type == "cuda" else nullcontext

    y_train = data["y_train"]
    sampler, sampler_shuffle = make_train_sampler(y_train, args.sampler, args.seed)
    common = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda",
              "generator": torch.Generator().manual_seed(args.seed + 202)}
    if args.num_workers > 0:
        common.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(
        data["train_set"], batch_size=args.batch_size, sampler=sampler,
        shuffle=sampler_shuffle, drop_last=False, **common
    )
    val_loader = DataLoader(data["val_set"], batch_size=args.batch_size, shuffle=False, drop_last=False, **common)

    seed_everything(args.seed)
    model = build_model(config, args)
    if model.load_report:
        print("[WEIGHTS]", json.dumps(model.load_report), flush=True)
    model = model.to(device=device, memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format)
    if args.use_sam:
        optimizer = SAM(
            optimizer_groups(model), torch.optim.AdamW, rho=args.sam_rho,
            weight_decay=1e-4, betas=(0.9, 0.999),
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_groups(model), weight_decay=1e-4, betas=(0.9, 0.999))
    # The scheduler is attached directly to SAM's wrapped AdamW optimizer.
    schedule_optimizer = optimizer_step_optimizer(optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(schedule_optimizer, lr_lambda=lr_schedule(args.epochs))
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    eval_model = ema.ema_model if ema is not None else model
    parameters = list(model.parameters())
    total_params = sum(p.numel() for p in parameters)
    inference_params = model.inference_parameter_count()
    trainable_params = model.trainable_parameter_count()
    print(f"[MODEL] training params={total_params:,} inference params={inference_params:,} "
          f"trainable={trainable_params:,}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_tag = "IR50_MSCELEB" if config.backbone == "ir50" else "VGGFACE2"
    landmark_tag = "_LANDMARKS" if config.landmarks else ""
    loss_ablation = settings["pair_margin_weight"] > 0 or settings["class_weighting"] != "off"
    upgrade_active = config.use_tap or config.use_silu_gate or args.random_erasing
    tier3_active = (args.sampler != "balanced" or args.label_smoothing > 0 or args.hard_weight is not None
                    or args.use_sam or args.use_ema
                    or (args.random_erasing and (args.re_prob != 0.5 or args.re_max_scale != 0.10)))
    version_tag = ("V46_UPGRADE_V2" if tier3_active else ("V46_UPGRADE" if upgrade_active else
                   ("V46" if settings["soft_contrast_weight"] > 0 else
                    ("V45" if loss_ablation else ("V44" if config.head == "queries" else "V43")))))
    name = f"FRITNet_{version_tag}_{args.preset.upper()}_{args.dataset.upper()}_{weights_tag}{landmark_tag}_seed{args.seed}_{stamp}"
    output_dir = Path(args.output_dir or f"{OUTPUT_ROOT}/{name}")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = data["rows"]
    run_config = {
        "variant": f"{version_tag}_{args.preset.upper()}", "dataset": args.dataset, "smoke": args.smoke, "seed": args.seed,
        "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
        "model": config.as_dict(), "settings": settings, "training_params": total_params,
        "input_pipeline": transform_config,
        "upgrades": {"tap": config.use_tap, "silu_gate": config.use_silu_gate,
                     "random_erasing": args.random_erasing,
                     "re_prob": args.re_prob, "re_max_scale": args.re_max_scale},
        "tier3": {"sampler": args.sampler, "label_smoothing": args.label_smoothing,
                  "hard_weight": settings["hard_weight"], "use_sam": args.use_sam,
                  "sam_rho": args.sam_rho if args.use_sam else None,
                  "use_ema": args.use_ema, "ema_decay": args.ema_decay if args.use_ema else None},
        "trainable_params": trainable_params, "weights_loaded": model.load_report,
        "ir50_weights": args.ir50_weights if config.backbone == "ir50" else None,
        "landmark_weights": args.landmark_weights if config.landmarks else None,
        "inference_params": inference_params,
        "optimizer": "SAM(AdamW)" if args.use_sam else "AdamW",
        "weight_decay": 1e-4,
        "backbone_lr": BACKBONE_LR, "task_lr": TASK_LR, "classifier_lr": CLASSIFIER_LR,
        "scheduler": "3-epoch linear warm-up then cosine",
        "sampling": ("inverse hard-class frequency WeightedRandomSampler" if args.sampler == "balanced"
                     else "natural frequency DataLoader shuffle=True"),
        "selection": "validation accuracy; soft CE breaks exact ties", "data_summary": data["summary"],
        "test_images_used": bool(args.dataset == "ferplus" and args.ferplus_split == "official"),
        "ferplus_split": ("official_3way_public_val_private_test" if args.dataset == "ferplus" and args.ferplus_split == "official" else args.ferplus_split),
        "private_test_images_used": False,
        "publictest_used_for_validation": bool(args.dataset == "ferplus" and args.ferplus_split == "official"),
        "train_original_csv_rows": rows[data["train_idx"]].tolist(),
        "validation_original_csv_rows": rows[data["val_idx"]].tolist(),
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    log_path = output_dir / "epoch_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(EPOCH_LOG_FIELDS)

    mix_rng = np.random.default_rng(args.seed + 303)
    mix_until = int(args.epochs * settings["mixup_stop_fraction"])
    best_accuracy, best_ce, best_epoch, stale = -1.0, float("inf"), 0, 0
    stop_reason, started, epoch = "reached epoch limit", time.time(), 0
    gate_value = lambda: float(model.tap_gate.detach()) if config.use_tap else 0.0
    silu_gamma_value = lambda: float(model.silu_gate_gamma.detach()) if config.use_silu_gate else 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tick = time.time()
        loss_sum, relation_sum, relation_n, pair_sum, pair_n, contrast_sum, contrast_n, correct, seen = 0.0, 0.0, 0, 0.0, 0, 0.0, 0, 0, 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}", dynamic_ncols=True)
        for step, (images, soft, hard) in enumerate(bar):
            if max_batches and step >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            soft = soft.to(device, non_blocking=True)
            hard = hard.to(device, non_blocking=True)
            hard_b, lam = hard, 1.0
            if epoch <= mix_until and mix_rng.random() < settings["mixup_prob"]:
                images, soft, hard_b, lam = mix_batch(images, soft, hard, settings["mixup_alpha"], mix_rng)
            # RAF-DB label smoothing affects CE targets only. Original one-hot or
            # mixup targets remain untouched for contrastive/pair-based losses.
            ce_targets = make_ce_targets(soft, args.dataset, args.label_smoothing, config.num_classes)
            optimizer.zero_grad(set_to_none=True)
            with amp():
                outputs = model(images)
                loss, relation_ce, pair_loss, contrast_loss = total_loss(
                    model, outputs, soft, hard, hard_b, lam, settings,
                    soft_ce_targets=ce_targets,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Non-finite gradient at epoch {epoch}, step {step}.")
            if args.use_sam:
                optimizer.first_step(zero_grad=True)
                try:
                    # SAM's second forward sees perturbed weights but must not
                    # update BN running statistics a second time.
                    with batchnorm_no_running_update(model):
                        with amp():
                            second_outputs = model(images)
                            second_loss, _, _, _ = total_loss(
                                model, second_outputs, soft, hard, hard_b, lam, settings,
                                soft_ce_targets=ce_targets,
                            )
                        if not torch.isfinite(second_loss):
                            raise FloatingPointError(f"Non-finite SAM second-pass loss at epoch {epoch}.")
                        second_loss.backward()
                        second_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
                        if not torch.isfinite(second_norm):
                            raise FloatingPointError(f"Non-finite SAM second-pass gradient at epoch {epoch}, step {step}.")
                    optimizer.second_step(zero_grad=True)
                except Exception:
                    optimizer.cancel_step(zero_grad=True)
                    raise
            else:
                optimizer.step()
            if ema is not None:
                ema.update(model)
            batch = int(hard.numel())
            seen += batch
            loss_sum += float(loss.detach()) * batch
            if relation_ce is not None:
                relation_sum += float(relation_ce.detach()) * batch
                relation_n += batch
            pair_sum += float(pair_loss.detach()) * batch
            pair_n += batch
            contrast_sum += float(contrast_loss.detach()) * batch
            contrast_n += batch
            correct += int((outputs[0].detach().float().argmax(1) == hard).sum())
            bar.set_postfix(loss=f"{float(loss.detach()):.4f}", acc=f"{100 * correct / seen:.1f}%")

        logits = validation_logits(eval_model, val_loader, device, max_batches)
        metrics, top_share = data["evaluate"](logits)
        lrs = [g["lr"] for g in optimizer.param_groups]
        seconds = time.time() - tick
        relation_mean = relation_sum / relation_n if relation_n else 0.0
        pair_mean = pair_sum / pair_n if pair_n else 0.0
        contrast_mean = contrast_sum / contrast_n if contrast_n else 0.0
        print(f"[EPOCH {epoch:02d}] loss={loss_sum / seen:.4f} train_acc={100 * correct / seen:.2f}% tap_gate={gate_value():+.4f} silu_gamma={silu_gamma_value():+.4f} "
              f"relationCE={relation_mean:.4f} pairMargin={pair_mean:.4f} softContrast={contrast_mean:.4f} | VAL acc={100 * metrics['accuracy']:.3f}% "
              f"twin_free={100 * metrics['twin_free']['accuracy']:.3f}% BA={100 * metrics['balanced_accuracy']:.3f}% "
              f"F1={100 * metrics['macro_f1']:.3f}% softCE={metrics['soft_cross_entropy']:.5f} | {seconds:.1f}s", flush=True)
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(make_epoch_log_row(
                epoch, loss_sum / seen, correct / seen, relation_mean, pair_mean, contrast_mean,
                gate_value(), silu_gamma_value(), metrics, lrs, seconds,
            ))

        improved = (metrics["accuracy"] > best_accuracy + 1e-12 or (
            abs(metrics["accuracy"] - best_accuracy) <= 1e-12 and metrics["soft_cross_entropy"] < best_ce - 1e-8))
        if improved:
            best_accuracy, best_ce, best_epoch, stale = metrics["accuracy"], metrics["soft_cross_entropy"], epoch, 0
            atomic_save({"epoch": epoch, "variant": run_config["variant"], "dataset": args.dataset, "seed": args.seed,
                         "model_config": config.as_dict(), "model_state_dict": eval_model.state_dict(),
                         "weights_source": "ema" if ema is not None else "student",
                         "ema_decay": args.ema_decay if ema is not None else None}, output_dir / "best_model.pth")
            best_tap_gate = float(eval_model.tap_gate.detach()) if config.use_tap else 0.0
            best_silu_gamma = float(eval_model.silu_gate_gamma.detach()) if config.use_silu_gate else 0.0
            (output_dir / "best_validation_metrics.json").write_text(
                json.dumps({"epoch": epoch, "tap_gate": best_tap_gate,
                            "silu_gate_gamma": best_silu_gamma,
                            "weights_source": "ema" if ema is not None else "student",
                            "metrics": metrics}, indent=2), encoding="utf-8")
            n = len(logits)
            np.savez_compressed(output_dir / "best_val_logits.npz", logits=logits, original_csv_row=rows[data["val_idx"]][:n],
                                hard=data["hard_val"][:n], twin=data["twin"][:n], top_share=top_share)
            print(f"[CHECKPOINT] new best at epoch {epoch}", flush=True)
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            stop_reason = f"no improvement for {args.patience} epochs"
            print(f"[EARLY STOP] {stop_reason}", flush=True)
            break

    summary = {"status": "SMOKE_ONLY_NOT_A_RESULT" if args.smoke else "COMPLETED", "variant": run_config["variant"],
               "dataset": args.dataset, "seed": args.seed, "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
               "best_validation_soft_ce": best_ce, "epochs_completed": epoch, "stop_reason": stop_reason,
               "final_tap_gate": gate_value(), "final_silu_gate_gamma": silu_gamma_value(),
               "sam_enabled": args.use_sam, "sam_rho": args.sam_rho if args.use_sam else None,
               "ema_enabled": args.use_ema, "ema_decay": args.ema_decay if args.use_ema else None,
               "sampler": args.sampler, "label_smoothing": args.label_smoothing,
               "hard_weight": settings["hard_weight"],
               "ferplus_split": ("official_3way_public_val_private_test" if args.dataset == "ferplus" and args.ferplus_split == "official" else args.ferplus_split),
               "private_test_images_used": False,
               "publictest_used_for_validation": bool(args.dataset == "ferplus" and args.ferplus_split == "official"),
               "training_params": total_params, "inference_params": inference_params,
               "elapsed_minutes": (time.time() - started) / 60.0, "output_dir": str(output_dir)}
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[COMPLETE]", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()