"""RAF-DB training partition only. The test images are never extracted or opened.

Validation is a stratified split of train_labels.csv. RAF folders 1..7 map to
surprise, fear, disgust, happiness, sadness, anger, neutral: the first seven FERPlus
classes in FRITNet order. Contempt is not present and is not mapped.
"""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import data_v40

RAF_CLASSES = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]
EXPECTED_TRAIN_COUNTS = [1290, 281, 717, 4772, 1982, 705, 2524]
DEFAULT_ZIP = "/content/drive/MyDrive/Datasets.zip"
DEFAULT_CACHE = "/content/rafdb_v42_train_only"


def extract_train_only(zip_path, cache_dir=DEFAULT_CACHE) -> Path:
    out = Path(cache_dir) / "RAF-DB"
    if (out / "train_labels.csv").is_file() and (out / "DATASET" / "train").is_dir():
        return out
    if not Path(zip_path).is_file():
        raise FileNotFoundError(f"RAF-DB archive not found: {zip_path}")
    written = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename.replace("\\", "/")).parts
            low = [p.lower() for p in parts]
            if "raf-db" not in low:
                continue
            rel = parts[low.index("raf-db") + 1:]
            is_image = len(rel) >= 3 and rel[0].lower() == "dataset" and rel[1].lower() == "train"
            is_manifest = len(rel) == 1 and rel[0].lower() == "train_labels.csv"
            if not (is_image or is_manifest):
                continue
            target = out.joinpath(*rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)
            written += 1
    if not written:
        raise RuntimeError(f"No RAF-DB training files found in {zip_path}")
    print(f"[RAF-DB] extracted {written:,} training files to {out}; test images not extracted", flush=True)
    return out


def load_train_table(root):
    root = Path(root)
    table = pd.read_csv(root / "train_labels.csv")
    names = table.iloc[:, 0].astype(str).to_numpy()
    folders = table.iloc[:, 1].astype(int).to_numpy()
    labels = folders - 1
    counts = np.bincount(labels, minlength=7).tolist()
    if counts != EXPECTED_TRAIN_COUNTS:
        raise RuntimeError(f"RAF-DB train counts {counts} differ from {EXPECTED_TRAIN_COUNTS}.")
    paths = np.array([str(root / "DATASET" / "train" / str(f) / n) for n, f in zip(names, folders)], dtype=object)
    missing = [p for p in paths[:50] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RAF-DB image: {missing[0]}")
    return paths, labels


def stratified_split(labels, seed, fraction=0.10):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for cls in range(7):
        index = np.flatnonzero(labels == cls)
        rng.shuffle(index)
        n = max(1, int(round(len(index) * fraction)))
        val.append(index[:n])
        train.append(index[n:])
    train, val = np.concatenate(train), np.concatenate(val)
    rng.shuffle(train)
    rng.shuffle(val)
    if len(np.intersect1d(train, val)) or len(train) + len(val) != len(labels):
        raise RuntimeError("Invalid RAF-DB split.")
    return train, val


def file_digests(paths):
    return np.array([hashlib.md5(Path(p).read_bytes()).hexdigest() for p in paths])


class RafTrainDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = data_v40.TRAIN_TRANSFORM(Image.open(self.paths[index]).convert("RGB"))
        hard = int(self.labels[index])
        return image, torch.nn.functional.one_hot(torch.tensor(hard), 7).float(), torch.tensor(hard)


class RafValidationDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return data_v40.EVAL_TRANSFORM(Image.open(self.paths[index]).convert("RGB")), int(self.labels[index])
