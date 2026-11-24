#!/usr/bin/env python3
"""Evaluate a trained FRITNet checkpoint on FERPlus PublicTest and/or PrivateTest.

Uses the established V21 FERPlus filtering rule. This script is intended to be
run only after training/checkpoint selection is complete; test partitions are
never used by the training entry point.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from data_v40 import (ALL_VOTES, FERPLUS_ORDER, EXPECTED_RAW_ROWS, EXPECTED_RETAINED,
                      PIXELS_CANDIDATES, LABELS_CANDIDATES, _resolve)
from evaluation_utils import classification_metrics, infer_logits, load_checkpoint_model, make_eval_transform, write_json


class FERPlusOfficialDataset(Dataset):
    def __init__(self, pixels, soft, hard, transform):
        self.pixels = np.asarray(pixels, dtype=object)
        self.soft = np.asarray(soft, dtype=np.float32)
        self.hard = np.asarray(hard, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        values = np.fromstring(str(self.pixels[index]), sep=" ", dtype=np.uint8)
        if values.size != 48 * 48:
            raise RuntimeError(f"FERPlus official image row {index} has {values.size} pixels, expected 2304.")
        image = Image.fromarray(values.reshape(48, 48), mode="L").convert("RGB")
        return self.transform(image), self.hard[index]


def parse_official_splits():
    pixels_path = _resolve("FERPLUS_PIXELS_CSV", PIXELS_CANDIDATES, "FER2013 pixel CSV")
    labels_path = _resolve("FERPLUS_LABELS_CSV", LABELS_CANDIDATES, "FERPlus annotation CSV")
    labels = pd.read_csv(labels_path)
    pixels = pd.read_csv(pixels_path, usecols=["pixels"])
    if len(labels) != EXPECTED_RAW_ROWS or len(pixels) != len(labels):
        raise RuntimeError(f"Expected {EXPECTED_RAW_ROWS} aligned FERPlus rows; got labels={len(labels)}, pixels={len(pixels)}.")
    columns = {str(c).strip().lower(): c for c in labels.columns}
    required = set(ALL_VOTES + FERPLUS_ORDER + ["usage"])
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"FERPlus annotation CSV missing columns: {missing}")
    all_votes = labels[[columns[c] for c in ALL_VOTES]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    emotion = labels[[columns[c] for c in FERPLUS_ORDER]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    if not np.allclose(emotion, np.rint(emotion), atol=1e-6):
        raise ValueError("FERPlus emotion vote columns contain non-integer counts.")
    emotion = np.rint(emotion).astype(np.int64)
    total_all = all_votes.sum(axis=1)
    unknown_nf = all_votes[:, [ALL_VOTES.index("unknown"), ALL_VOTES.index("nf")]].sum(axis=1)
    unknown_fraction = np.divide(unknown_nf, total_all, out=np.ones_like(unknown_nf), where=total_all > 0)
    total_emotion = emotion.sum(axis=1)
    valid = (total_all > 0) & (unknown_fraction <= 0.50) & (total_emotion > 0)
    usage = labels[columns["usage"]].astype(str).str.strip().str.lower().to_numpy()
    retained = {split: int(np.sum(valid & (usage == split))) for split in EXPECTED_RETAINED}
    if retained != EXPECTED_RETAINED:
        raise RuntimeError(f"FERPlus filter counts mismatch. Expected {EXPECTED_RETAINED}, got {retained}.")
    result = {}
    pixel_values = pixels["pixels"].astype(str).to_numpy()
    for split in ("publictest", "privatetest"):
        mask = valid & (usage == split)
        votes = emotion[mask]
        soft = votes / np.maximum(votes.sum(axis=1, keepdims=True), 1)
        result[split] = {
            "pixels": pixel_values[mask],
            "soft": soft.astype(np.float32),
            "hard": soft.argmax(axis=1).astype(np.int64),
            "rows": np.flatnonzero(mask),
        }
    return result, {"pixels_csv": str(pixels_path), "labels_csv": str(labels_path), "retained_counts": retained}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("publictest", "privatetest", "both"), default="both")
    p.add_argument("--output", default="")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--tta", action="store_true",
                   help="Also evaluate horizontal-flip TTA by averaging original and flipped logits.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, checkpoint = load_checkpoint_model(args.checkpoint, device)
    if config.num_classes != 8:
        raise ValueError(f"FERPlus checkpoint must have 8 classes; found {config.num_classes}.")
    splits, source_info = parse_official_splits()
    selected = ("publictest", "privatetest") if args.split == "both" else (args.split,)
    results = {}
    tta_results = {}
    for split in selected:
        block = splits[split]
        dataset = FERPlusOfficialDataset(block["pixels"], block["soft"], block["hard"], make_eval_transform(config))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        logits, target = infer_logits(model, loader, device)
        standard_metrics = classification_metrics(logits, target, FERPLUS_ORDER, block["soft"])
        results[split] = standard_metrics
        if args.tta:
            tta_logits, tta_target = infer_logits(model, loader, device, horizontal_flip=True)
            if not (target == tta_target).all():
                raise RuntimeError(f"FERPlus {split} standard/TTA labels are misaligned.")
            tta_results[split] = classification_metrics(
                tta_logits, tta_target, FERPLUS_ORDER, block["soft"]
            )
    result = {
        "evaluation": "FERPlus official test partitions",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_weights_source": checkpoint.get("weights_source", "student"),
        "backbone": config.backbone,
        "input_size": 112 if config.backbone == "ir50" else 224,
        "filter_protocol": "V21: total_votes>0, (unknown+nf)/total<=0.50, at least one emotion vote",
        "test_images_used_during_training": False,
        "tta_enabled": bool(args.tta),
        "source": source_info,
        "metrics": results,
        "tta_metrics": tta_results if args.tta else None,
    }
    print(json.dumps(result, indent=2))
    if args.tta:
        for split in selected:
            print(f"\n[{split.upper()} STANDARD OFFICIAL METRICS]", json.dumps(results[split], indent=2))
            print(f"\n[{split.upper()} HORIZONTAL-FLIP TTA OFFICIAL METRICS]",
                  json.dumps(tta_results[split], indent=2))
    default_name = "official_ferplus_test_metrics_tta.json" if args.tta else "official_ferplus_test_metrics.json"
    output = args.output or str(Path(args.checkpoint).parent / default_name)
    write_json(result, output)


if __name__ == "__main__":
    main()
