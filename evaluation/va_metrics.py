"""Metrik evaluasi cabang ST-GCN (regresi Valence-Arousal + mood kuadran).

Dipakai oleh models/st_gcn_graph_tran/train.py:
    from evaluation.va_metrics import evaluate_va
    metrics = evaluate_va(preds, targets_va, targets_mood)

Semua fungsi berbasis numpy (tanpa torch/sklearn) supaya ringan dan bisa
dipakai baik saat training maupun inferensi edge.
"""
from __future__ import annotations

import numpy as np

try:
    from models.st_gcn_graph_tran.labels import (
        va_to_mood_id, MOOD_CLASSES, NEUTRAL_RADIUS,
    )
except Exception:  # fallback agar modul tetap mandiri
    MOOD_CLASSES = ["Ceria", "Cemas", "Tertekan", "Rileks", "Netral"]
    NEUTRAL_RADIUS = 0.2

    def va_to_mood_id(v, a, neutral_radius=NEUTRAL_RADIUS):
        if (v * v + a * a) ** 0.5 <= neutral_radius:
            return 4
        if v >= 0 and a >= 0:
            return 0
        if v < 0 and a >= 0:
            return 1
        if v < 0 and a < 0:
            return 2
        return 3


def concordance_cc(pred, target, eps=1e-8):
    """Concordance Correlation Coefficient (skalar) untuk 1 dimensi."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if pred.size == 0:
        return 0.0
    mp_, mt = pred.mean(), target.mean()
    vp, vt = pred.var(), target.var()
    cov = ((pred - mp_) * (target - mt)).mean()
    return float((2.0 * cov) / (vp + vt + (mp_ - mt) ** 2 + eps))


def _mae(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.mean(np.abs(a - b))) if a.size else 0.0


def _rmse(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.sqrt(np.mean(d ** 2))) if d.size else 0.0


def macro_f1(y_true, y_pred, num_classes=None):
    """Macro-F1 tanpa sklearn. Kelas yang tak muncul di ground-truth diabaikan."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_true.size == 0:
        return 0.0
    if num_classes is None:
        num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    f1s = []
    for c in range(num_classes):
        if np.sum(y_true == c) == 0:
            continue  # kelas absen di GT -> tak dihitung ke macro
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate_va(preds, targets_va, targets_mood=None, neutral_radius=NEUTRAL_RADIUS):
    """Hitung metrik VA + mood.

    preds, targets_va : array (N, 2) -> kolom 0 valence, kolom 1 arousal.
    targets_mood      : array (N,) id mood 0..4 (opsional; kalau None dihitung
                        dari targets_va via kuadran).
    Return dict lengkap; kunci yang dipakai train.py: ccc_valence, ccc_arousal,
    mood_macro_f1.
    """
    preds = np.asarray(preds, dtype=np.float64).reshape(-1, 2)
    targets_va = np.asarray(targets_va, dtype=np.float64).reshape(-1, 2)

    ccc_v = concordance_cc(preds[:, 0], targets_va[:, 0])
    ccc_a = concordance_cc(preds[:, 1], targets_va[:, 1])

    pred_mood = np.array([va_to_mood_id(v, a, neutral_radius) for v, a in preds], dtype=int)
    if targets_mood is None:
        gt_mood = np.array([va_to_mood_id(v, a, neutral_radius) for v, a in targets_va], dtype=int)
    else:
        gt_mood = np.asarray(targets_mood).astype(int).ravel()

    acc = float(np.mean(pred_mood == gt_mood)) if pred_mood.size else 0.0
    return {
        "ccc_valence": ccc_v,
        "ccc_arousal": ccc_a,
        "ccc_mean": 0.5 * (ccc_v + ccc_a),
        "mae_valence": _mae(preds[:, 0], targets_va[:, 0]),
        "mae_arousal": _mae(preds[:, 1], targets_va[:, 1]),
        "rmse_va": _rmse(preds, targets_va),
        "mood_accuracy": acc,
        "mood_macro_f1": macro_f1(gt_mood, pred_mood, num_classes=len(MOOD_CLASSES)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gt = rng.uniform(-1, 1, size=(64, 2))
    pr = gt + rng.normal(0, 0.1, size=(64, 2))
    from pprint import pprint
    pprint(evaluate_va(pr, gt))
