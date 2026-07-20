"""Fungsi loss untuk cabang ST-GCN.

- Regresi VA: Concordance Correlation Coefficient (CCC) loss -- metrik/loss
  standar pada challenge Aff-Wild2 (ABAW) untuk valence-arousal.
- Klasifikasi 5 kelas: CrossEntropy.
- Dual: kombinasi berbobot.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def concordance_cc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """CCC untuk satu dimensi (dihitung sepanjang dimensi batch). Skalar."""
    pred = pred.view(-1)
    target = target.view(-1)
    mean_p = pred.mean()
    mean_t = target.mean()
    var_p = pred.var(unbiased=False)
    var_t = target.var(unbiased=False)
    cov = ((pred - mean_p) * (target - mean_t)).mean()
    ccc = (2.0 * cov) / (var_p + var_t + (mean_p - mean_t) ** 2 + eps)
    return ccc


class CCCLoss(nn.Module):
    """1 - CCC dirata-rata untuk valence & arousal. Input (B, 2)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ccc_v = concordance_cc(pred[:, 0], target[:, 0])
        ccc_a = concordance_cc(pred[:, 1], target[:, 1])
        return 1.0 - 0.5 * (ccc_v + ccc_a)


class DualLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ccc = CCCLoss()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, out: dict, target_va: torch.Tensor, target_cls: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ccc(out["va"], target_va) + self.beta * self.ce(out["logits"], target_cls)


def build_loss(head_type: str, class_weights: torch.Tensor | None = None,
               alpha: float = 1.0, beta: float = 0.5):
    head_type = head_type.lower()
    if head_type == "va":
        return CCCLoss()
    if head_type in ("class", "cls", "categorical"):
        return nn.CrossEntropyLoss(weight=class_weights)
    if head_type == "dual":
        return DualLoss(alpha, beta, class_weights)
    raise ValueError(f"head_type tak dikenal: {head_type!r}")
