"""FERPlus filtering and leakage-safe data views for FRITNet V40."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

FERPLUS_ORDER = [
    "surprise", "fear", "disgust", "happiness",
    "sadness", "anger", "neutral", "contempt",
]
ALL_VOTES = [
    "neutral", "happiness", "surprise", "sadness", "anger",
    "disgust", "fear", "contempt", "unknown", "nf",
]
EXPECTED_RAW_ROWS = 35887
EXPECTED_RETAINED = {"training": 28540, "publictest": 3574, "privatetest": 3569}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

PIXELS_CANDIDATES = [
    "/content/drive/MyDrive/Anjimoon/DATA SETS/fer2013.csv",
    "/content/drive/MyDrive/fer2013.csv",
]
LABELS_CANDIDATES = [
    "/content/drive/MyDrive/fer2013new.csv",
    "/content/drive/MyDrive/Anjimoon/DATA SETS/FERPLUS_fer2013new.csv",
    "/content/drive/MyDrive/Anjimoon/DATA SETS/fer2013new.csv",
    "/content/drive/MyDrive/FERPLUS_fer2013new.csv",
]

def build_transforms(image_size: int = 224, random_erasing: bool = False,
                     re_prob: float = 0.5, re_max_scale: float = 0.10):
    """Build train/eval transforms. IR-50 uses one direct Lanczos resize to 112px."""
    if not 0.0 <= float(re_prob) <= 1.0:
        raise ValueError("Random Erasing probability must be in [0, 1].")
    if not 0.02 <= float(re_max_scale) <= 1.0:
        raise ValueError("Random Erasing max scale must be in [0.02, 1.0].")
    interpolation = InterpolationMode.LANCZOS if int(image_size) == 112 else InterpolationMode.BILINEAR
    resize = transforms.Resize(
        (int(image_size), int(image_size)),
        interpolation=interpolation,
    )
    train_ops = [
        resize,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=8, translate=(0.04, 0.04), scale=(0.95, 1.05), fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
    if random_erasing:
        train_ops.append(transforms.RandomErasing(p=float(re_prob), scale=(0.02, float(re_max_scale))))
    eval_ops = [resize, transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


# Defaults preserve the original V46/FaceNet 224px behavior until the training
# entry point selects the IR-50-specific path.
TRAIN_TRANSFORM, WEAK_TRANSFORM = build_transforms(224, random_erasing=False)
EVAL_TRANSFORM = WEAK_TRANSFORM


def configure_transforms(backbone: str = "facenet", random_erasing: bool = False,
                          re_prob: float = 0.5, re_max_scale: float = 0.10):
    """Configure shared train/eval transforms before constructing DataLoaders."""
    global TRAIN_TRANSFORM, WEAK_TRANSFORM, EVAL_TRANSFORM
    if backbone not in ("facenet", "ir50"):
        raise ValueError(f"Unknown backbone for image transforms: {backbone}")
    image_size = 112 if backbone == "ir50" else 224
    TRAIN_TRANSFORM, WEAK_TRANSFORM = build_transforms(
        image_size, random_erasing, re_prob=re_prob, re_max_scale=re_max_scale
    )
    EVAL_TRANSFORM = WEAK_TRANSFORM
    interpolation = "lanczos" if image_size == 112 else "bilinear"
    return {
        "image_size": image_size,
        "interpolation": interpolation,
        "random_erasing": bool(random_erasing),
        "re_prob": float(re_prob) if random_erasing else 0.0,
        "re_scale": [0.02, float(re_max_scale)] if random_erasing else None,
    }


def _resolve(env_name: str, candidates: list[str], label: str) -> Path:
    explicit = os.environ.get(env_name, "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{env_name} points to a missing file: {path}")
        return path
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Cannot locate {label}. Set {env_name} to its exact path. Checked: {candidates}"
    )


def prepare_training_frame() -> tuple[pd.DataFrame, dict]:
    """Apply the existing V21 filter; only Training pixels enter the frame."""
    pixels_path = _resolve("FERPLUS_PIXELS_CSV", PIXELS_CANDIDATES, "FER2013 pixel CSV")
    labels_path = _resolve("FERPLUS_LABELS_CSV", LABELS_CANDIDATES, "FERPlus annotation CSV")
    print(f"[DATA] Pixels CSV: {pixels_path}", flush=True)
    print(f"[DATA] Labels CSV: {labels_path}", flush=True)

    labels = pd.read_csv(labels_path)
    if len(labels) != EXPECTED_RAW_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_RAW_ROWS:,} annotation rows; found {len(labels):,}.")
    columns = {str(c).strip().lower(): c for c in labels.columns}
    if len(columns) != len(labels.columns):
        raise ValueError("Duplicate annotation columns after normalization.")
    required = set(ALL_VOTES + FERPLUS_ORDER + ["usage"])
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"FERPlus annotation CSV missing columns: {missing}")

    pixels = pd.read_csv(pixels_path, usecols=["pixels"])
    if len(pixels) != len(labels):
        raise RuntimeError(f"Pixel/label row mismatch: {len(pixels)} vs {len(labels)}.")

    all_cols = [columns[c] for c in ALL_VOTES]
    emotion_cols = [columns[c] for c in FERPLUS_ORDER]
    all_votes = labels[all_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    emotion_votes_raw = labels[emotion_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    rounded = np.rint(emotion_votes_raw)
    if not np.allclose(emotion_votes_raw, rounded, atol=1e-6):
        raise ValueError("FERPlus emotion vote columns contain non-integer counts.")
    emotion_votes = rounded.astype(np.int64)

    total_all = all_votes.sum(axis=1)
    unknown_nf = all_votes[:, [ALL_VOTES.index("unknown"), ALL_VOTES.index("nf")]].sum(axis=1)
    unknown_fraction = np.divide(unknown_nf, total_all, out=np.ones_like(unknown_nf), where=total_all > 0)
    total_emotion = emotion_votes.sum(axis=1)
    valid = (total_all > 0) & (unknown_fraction <= 0.50) & (total_emotion > 0)
    usage = labels[columns["usage"]].astype(str).str.strip().str.lower().to_numpy()
    retained = {split: int(np.sum(valid & (usage == split))) for split in EXPECTED_RETAINED}
    print(f"[V21 FILTER] retained split counts: {retained}", flush=True)
    if retained != EXPECTED_RETAINED:
        raise RuntimeError(f"V21 filter mismatch. Expected {EXPECTED_RETAINED}, got {retained}.")

    train_mask = valid & (usage == "training")
    row_ids = np.flatnonzero(train_mask)
    counts = emotion_votes[train_mask]
    totals = counts.sum(axis=1)
    soft = (counts / totals[:, None]).astype(np.float32)
    hard = soft.argmax(axis=1).astype(np.int64)
    train_pixels = pixels.loc[train_mask, "pixels"].astype(str).to_numpy()
    lengths = np.fromiter((len(s.split()) for s in train_pixels), dtype=np.int32, count=len(train_pixels))
    invalid = np.flatnonzero(lengths != 48 * 48)
    if len(invalid):
        raise ValueError(f"Training pixel row has {lengths[int(invalid[0])]} values; expected 2304.")
    if not np.isfinite(soft).all() or not np.allclose(soft.sum(1), 1.0, atol=1e-6):
        raise RuntimeError("Invalid normalized FERPlus soft labels.")

    frame = pd.DataFrame({
        "pixels": train_pixels,
        "soft_label": list(soft),
        "hard_label": hard,
        "vote_counts": list(counts),
        "original_csv_row": row_ids.astype(np.int64),
    })
    class_counts = np.bincount(hard, minlength=8)
    summary = {
        "raw_rows": len(labels),
        "retained_split_counts": retained,
        "training_class_counts_hard_argmax": {FERPLUS_ORDER[i]: int(class_counts[i]) for i in range(8)},
        "pixels_path": str(pixels_path),
        "labels_path": str(labels_path),
        "training_rows_pixel_count_checked": len(train_pixels),
    }
    return frame, summary


def stratified_train_val_indices(labels: Iterable[int], val_fraction: float, seed: int):
    labels = np.asarray(list(labels), dtype=np.int64)
    if not 0.05 <= val_fraction <= 0.20:
        raise ValueError("val_fraction must be between 0.05 and 0.20.")
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for cls in range(8):
        idx = np.flatnonzero(labels == cls)
        if len(idx) < 5:
            raise RuntimeError(f"Class {FERPLUS_ORDER[cls]} has too few samples for stratification.")
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx, val_idx = np.concatenate(train_parts), np.concatenate(val_parts)
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    if len(np.intersect1d(train_idx, val_idx)) or len(train_idx) + len(val_idx) != len(labels):
        raise RuntimeError("Invalid train/validation split (overlap or incomplete coverage).")
    return train_idx, val_idx


def _read_image(pixel_string: str, index: int) -> Image.Image:
    values = np.fromstring(pixel_string, sep=" ", dtype=np.uint8)
    if values.size != 48 * 48:
        raise RuntimeError(f"Invalid FERPlus image at local index {index}: {values.size} pixels.")
    return Image.fromarray(values.reshape(48, 48), mode="L").convert("RGB")


class FERPlusV40TrainDataset(Dataset):
    """Returns a strong student view and a weak deterministic EMA-teacher view."""
    def __init__(self, frame: pd.DataFrame, indices: np.ndarray):
        self.pixels = frame["pixels"].to_numpy(dtype=object)[indices]
        self.soft = torch.from_numpy(np.stack(frame["soft_label"].to_numpy()).astype(np.float32))[indices]
        self.hard = torch.from_numpy(frame["hard_label"].to_numpy(dtype=np.int64))[indices]
        self.votes = torch.from_numpy(np.stack(frame["vote_counts"].to_numpy()).astype(np.float32))[indices]

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index: int):
        image = _read_image(self.pixels[index], index)
        student_view = TRAIN_TRANSFORM(image)
        teacher_view = WEAK_TRANSFORM(image)
        return student_view, teacher_view, self.soft[index], self.hard[index], self.votes[index]


class FERPlusV40ValidationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, indices: np.ndarray):
        self.pixels = frame["pixels"].to_numpy(dtype=object)[indices]
        self.soft = torch.from_numpy(np.stack(frame["soft_label"].to_numpy()).astype(np.float32))[indices]
        self.hard = torch.from_numpy(frame["hard_label"].to_numpy(dtype=np.int64))[indices]
        self.votes = torch.from_numpy(np.stack(frame["vote_counts"].to_numpy()).astype(np.float32))[indices]

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index: int):
        image = _read_image(self.pixels[index], index)
        return WEAK_TRANSFORM(image), self.soft[index], self.hard[index], self.votes[index]
