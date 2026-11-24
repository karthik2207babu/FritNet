#!/usr/bin/env python3
"""Evaluate a trained FRITNet checkpoint on the official RAF-DB test partition.

This script alone extracts/opens DATASET/test. Training continues to use only
train_labels.csv and DATASET/train.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import torch
from torch.utils.data import DataLoader

from data_raf_v42 import RAF_CLASSES, DEFAULT_ZIP
from evaluation_utils import ImagePathDataset, classification_metrics, infer_logits, load_checkpoint_model, make_eval_transform, write_json


def extract_official_test(zip_path: Path, cache_dir: Path) -> Path:
    out = cache_dir / "RAF-DB"
    manifest = out / "test_labels.csv"
    test_dir = out / "DATASET" / "test"
    has_test_images = test_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")
        for path in test_dir.rglob("*")
    )
    if manifest.is_file() and has_test_images:
        return out
    if not zip_path.is_file():
        raise FileNotFoundError(f"RAF-DB archive not found: {zip_path}")
    written = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename.replace("\\", "/")).parts
            low = [part.lower() for part in parts]
            if "raf-db" not in low:
                continue
            rel = parts[low.index("raf-db") + 1:]
            is_manifest = len(rel) >= 1 and rel[-1].lower() == "test_labels.csv"
            is_test_image = (len(rel) >= 4 and rel[0].lower() == "dataset"
                             and rel[1].lower() == "test"
                             and Path(rel[-1]).suffix.lower() in (".jpg", ".jpeg", ".png"))
            if not (is_manifest or is_test_image):
                continue
            target = (out / "test_labels.csv") if is_manifest else out.joinpath(*rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            written += 1
    if not manifest.is_file() or not test_dir.is_dir():
        raise RuntimeError(
            "Could not locate RAF-DB test_labels.csv and DATASET/test in the archive. "
            "Pass --raf-root pointing to the extracted official RAF-DB directory."
        )
    print(f"[RAF-DB OFFICIAL] Extracted {written:,} test-only files to {out}")
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--raf-zip", default=DEFAULT_ZIP)
    p.add_argument("--raf-root", default="", help="Extracted official RAF-DB folder containing test_labels.csv and DATASET/test.")
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
    if config.num_classes != 7:
        raise ValueError(f"RAF-DB checkpoint must have 7 classes; found {config.num_classes}.")
    root = Path(args.raf_root) if args.raf_root else extract_official_test(Path(args.raf_zip), Path("/content/rafdb_official_eval"))
    import pandas as pd
    table = pd.read_csv(root / "test_labels.csv")
    names = table.iloc[:, 0].astype(str).to_numpy()
    folders = table.iloc[:, 1].astype(int).to_numpy()
    if not set(folders.tolist()).issubset(set(range(1, 8))):
        raise ValueError("RAF-DB test labels must use folder/class IDs 1 through 7.")
    labels = folders - 1
    image_root = root / "DATASET" / "test"
    paths = [image_root / str(folder) / name for folder, name in zip(folders, names)]
    dataset = ImagePathDataset(paths, labels, make_eval_transform(config))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    logits, target = infer_logits(model, loader, device)
    metrics = classification_metrics(logits, target, RAF_CLASSES)
    tta_metrics = None
    if args.tta:
        tta_logits, tta_target = infer_logits(model, loader, device, horizontal_flip=True)
        if not (target == tta_target).all():
            raise RuntimeError("RAF-DB standard/TTA labels are misaligned.")
        tta_metrics = classification_metrics(tta_logits, tta_target, RAF_CLASSES)
    result = {
        "evaluation": "RAF-DB official test set",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_weights_source": checkpoint.get("weights_source", "student"),
        "backbone": config.backbone,
        "input_size": 112 if config.backbone == "ir50" else 224,
        "test_images_used_during_training": False,
        "metrics": metrics,
        "tta_enabled": bool(args.tta),
        "tta_metrics": tta_metrics,
    }
    print(json.dumps(result, indent=2))
    if args.tta:
        print("\n[STANDARD OFFICIAL METRICS]", json.dumps(metrics, indent=2))
        print("\n[HORIZONTAL-FLIP TTA OFFICIAL METRICS]", json.dumps(tta_metrics, indent=2))
    default_name = "official_raf_test_metrics_tta.json" if args.tta else "official_raf_test_metrics.json"
    output = args.output or str(Path(args.checkpoint).parent / default_name)
    write_json(result, output)


if __name__ == "__main__":
    main()
