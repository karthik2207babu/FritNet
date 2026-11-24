"""Evaluation protocol for V41: metrics, twin rows, majority subset, agreement groups."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from data_v40 import FERPLUS_ORDER

VOTE_COLUMNS = ["neutral", "happiness", "surprise", "sadness", "anger",
                "disgust", "fear", "contempt", "unknown", "nf"]


def scores(target, pred):
    confusion = np.zeros((8, 8), dtype=np.int64)
    np.add.at(confusion, (target, pred), 1)
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(1).astype(float)
    predicted = confusion.sum(0).astype(float)
    recall = np.divide(tp, support, out=np.zeros(8), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(8), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(8), where=(precision + recall) > 0)
    return {
        "n": int(len(target)),
        "accuracy": float((target == pred).mean()) if len(target) else None,
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            FERPLUS_ORDER[i]: {"support": int(support[i]), "precision": float(precision[i]),
                               "recall": float(recall[i]), "f1": float(f1[i])}
            for i in range(8)
        },
        "confusion_rows_true_cols_pred": confusion.tolist(),
    }


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def expected_calibration_error(confidence, correct, bins=15):
    """Equal-width bins; the target is the hard top-vote label."""
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    return float(sum(
        (index == b).mean() * abs(confidence[index == b].mean() - correct[index == b].mean())
        for b in range(bins) if (index == b).any()
    ))


def pixel_digests(pixel_strings):
    return np.array([hashlib.md5(p.encode()).hexdigest() for p in pixel_strings])


def majority_keep_all_rows(labels_path):
    """Microsoft majority rule over all CSV rows: remove single votes, then the top
    class must be an emotion holding strictly more than half of the remaining votes."""
    labels = pd.read_csv(labels_path)
    columns = {str(c).strip().lower(): c for c in labels.columns}
    raw = labels[[columns[c] for c in VOTE_COLUMNS]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    raw[raw <= 1] = 0
    keep = (raw.max(1) > 0.5 * raw.sum(1)) & (raw.argmax(1) < 8)
    usage = labels[columns["usage"]].astype(str).str.strip().str.lower().to_numpy()
    return keep, usage


class ValidationContext:
    """Annotation facts for one fixed set of validation rows. Independent of any model."""

    def __init__(self, votes, hard, twin, majority):
        self.votes = np.asarray(votes, dtype=np.float64)
        self.hard = np.asarray(hard, dtype=np.int64)
        self.twin = np.asarray(twin, dtype=bool)
        self.majority = np.asarray(majority, dtype=bool)
        total = self.votes.sum(1)
        self.soft = self.votes / total[:, None]
        top2 = np.sort(self.votes, 1)[:, ::-1][:, :2]
        self.tie = top2[:, 0] == top2[:, 1]
        self.top_share = top2[:, 0] / total
        self.margin = (top2[:, 0] - top2[:, 1]) / total
        self.entropy = -(self.soft * np.log(np.clip(self.soft, 1e-12, 1))).sum(1) / np.log(8)
        self.panel = total
        self.tied_set = self.votes == self.votes.max(1, keepdims=True)

    def groups(self):
        return {
            "top_share_lt_0.50": self.top_share < 0.50,
            "top_share_0.50_to_0.75": (self.top_share >= 0.50) & (self.top_share < 0.75),
            "top_share_ge_0.75": self.top_share >= 0.75,
            "entropy_lt_0.20": self.entropy < 0.20,
            "entropy_0.20_to_0.50": (self.entropy >= 0.20) & (self.entropy < 0.50),
            "entropy_ge_0.50": self.entropy >= 0.50,
            "margin_0_tie": self.margin == 0,
            "margin_gt0_to_0.30": (self.margin > 0) & (self.margin <= 0.30),
            "margin_gt0.30_to_0.60": (self.margin > 0.30) & (self.margin <= 0.60),
            "margin_gt_0.60": self.margin > 0.60,
            "panel_10": self.panel == 10,
            "panel_9": self.panel == 9,
            "panel_le_8": self.panel <= 8,
        }


def evaluate_logits(logits, context: ValidationContext):
    pred = logits.argmax(1)
    target = context.hard
    logp = log_softmax(logits)
    probs = np.exp(logp)
    correct = (pred == target).astype(float)
    free = ~context.twin
    result = scores(target, pred)
    result.update({
        "twin_free": scores(target[free], pred[free]),
        "majority_subset": scores(target[context.majority], pred[context.majority]),
        "majority_subset_twin_free_accuracy": float(correct[context.majority & free].mean()),
        "tie_aware_accuracy": float(context.tied_set[np.arange(len(pred)), pred].mean()),
        "tie_rows_accuracy": float(correct[context.tie].mean()) if context.tie.any() else None,
        "twin_rows_accuracy": float(correct[context.twin].mean()) if context.twin.any() else None,
        "soft_cross_entropy": float(-(context.soft * logp).sum(1).mean()),
        "soft_brier_score": float(((probs - context.soft) ** 2).sum(1).mean()),
        "hard_label_ece_15bin": expected_calibration_error(probs.max(1), correct),
        "groups": {
            name: {"n": int(mask.sum()), "accuracy": float(correct[mask].mean()) if mask.any() else None}
            for name, mask in context.groups().items()
        },
    })
    return result
