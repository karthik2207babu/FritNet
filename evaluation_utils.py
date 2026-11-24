"""Shared checkpoint reconstruction, image preprocessing and classification metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from model_v42 import FRITNetV42, V42Config
from data_v40 import MEAN, STD


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_config" not in payload or "model_state_dict" not in payload:
        raise ValueError("Checkpoint must contain model_config and model_state_dict.")
    config = V42Config(**payload["model_config"])
    model = FRITNetV42(config, pretrained_backbone=None)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, config, payload


def make_eval_transform(config: V42Config):
    image_size = 112 if config.backbone == "ir50" else 224
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def classification_metrics(logits: np.ndarray, target: np.ndarray, class_names: list[str], soft_targets=None):
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    pred = logits.argmax(axis=1)
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (target, pred), 1)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)
    correct = (pred == target).astype(np.float64)
    ece = 0.0
    for lower in np.linspace(0, 1, 16)[:-1]:
        upper = lower + 1 / 15
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    result = {
        "n": int(len(target)),
        "accuracy": float(correct.mean()) if len(target) else 0.0,
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "ece_15_bins": float(ece),
        "per_class": {
            class_names[i]: {
                "support": int(support[i]), "precision": float(precision[i]),
                "recall": float(recall[i]), "f1": float(f1[i]),
            } for i in range(n_classes)
        },
        "confusion_rows_true_cols_pred": cm.tolist(),
    }
    if soft_targets is not None:
        soft = np.asarray(soft_targets, dtype=np.float64)
        soft = soft / np.maximum(soft.sum(axis=1, keepdims=True), 1e-12)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        result["soft_cross_entropy"] = float(-(soft * log_probs).sum(axis=1).mean())
        result["brier_score"] = float(((probs - soft) ** 2).sum(axis=1).mean())
    return result


class ImagePathDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = [Path(p) for p in paths]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform
        if len(self.paths) != len(self.labels):
            raise ValueError("Image and label counts differ.")
        missing = [str(p) for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Official evaluation image missing: {missing[0]}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, int(self.labels[index])


def infer_logits(model, loader, device, horizontal_flip: bool = False):
    """Infer standard logits or average original/horizontal-flip logits.

    TTA averages *logits* as requested, not probabilities. The standard path
    remains byte-for-byte equivalent in its batching and prediction behavior.
    """
    outputs, targets = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            if horizontal_flip:
                flipped = torch.flip(images, dims=[3])
                joined = torch.cat((images, flipped), dim=0)
                if device.type == "cuda":
                    joined = joined.contiguous(memory_format=torch.channels_last)
                both_logits = model(joined)[0].float()
                batch_size = images.shape[0]
                logits = 0.5 * (both_logits[:batch_size] + both_logits[batch_size:])
            else:
                logits = model(images)[0].float()
            outputs.append(logits.cpu().numpy())
            targets.append(labels.numpy())
    if not outputs:
        raise RuntimeError("Official evaluation dataset is empty.")
    return np.concatenate(outputs), np.concatenate(targets)


def write_json(result: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[SAVED] {output_path}")
