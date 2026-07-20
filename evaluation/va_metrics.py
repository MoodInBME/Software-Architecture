"""Metrik evaluasi cabang ekspresi: CCC (VA), akurasi kuadran mood, macro-F1."""
from __future__ import annotations

import numpy as np

from models.st_gcn_graph_tran.labels import va_to_mood_id, MOOD_CLASSES


def ccc(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    mp, mt = pred.mean(), target.mean()
    vp, vt = pred.var(), target.var()
    cov = ((pred - mp) * (target - mt)).mean()
    return float((2 * cov) / (vp + vt + (mp - mt) ** 2 + eps))


def mood_from_va(va: np.ndarray, neutral_radius: float = 0.2) -> np.ndarray:
    va = np.asarray(va, dtype=np.float64).reshape(-1, 2)
    return np.array([va_to_mood_id(v, a, neutral_radius) for v, a in va], dtype=np.int64)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_class: int = 5) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    f1s = []
    for c in range(num_class):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def evaluate_va(pred_va: np.ndarray, target_va: np.ndarray, target_mood=None,
                neutral_radius: float = 0.2) -> dict:
    pred_va = np.asarray(pred_va).reshape(-1, 2)
    target_va = np.asarray(target_va).reshape(-1, 2)
    pred_mood = mood_from_va(pred_va, neutral_radius)
    if target_mood is None:
        target_mood = mood_from_va(target_va, neutral_radius)
    target_mood = np.asarray(target_mood).ravel()
    return {
        "ccc_valence": ccc(pred_va[:, 0], target_va[:, 0]),
        "ccc_arousal": ccc(pred_va[:, 1], target_va[:, 1]),
        "mood_accuracy": float(np.mean(pred_mood == target_mood)),
        "mood_macro_f1": macro_f1(target_mood, pred_mood, len(MOOD_CLASSES)),
    }
