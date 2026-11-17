"""Dua kepala keluaran sejajar (intensitas AU dan valensi-arousal) dengan rugi bermask.

Usulan untuk ISU-071 (menunggu persetujuan Jeremia sebelum `heads.build_head` diubah):
  * Kepala AU: K keluaran sigmoid dalam [0, 1] (intensitas 0..5 dibagi 5), K = 12 AU DISFA.
  * Kepala VA: 2 keluaran tanh dalam [-1, 1], sama dengan `heads.VAHead`.
  * Rugi: MSE AU hanya pada AU bermask 1, rugi VA = 1 - CCC per sumbu (selaras metrik
    CCC naskah) hanya pada sampel ber-VA. Batch campuran AFEW-VA dan DISFA aman.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .heads import VAHead


class AUHead(nn.Module):
    def __init__(
        self, in_dim: int, n_au: int = 12, hidden: int = 128, dropout: float = 0.3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_au),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class AUVAHead(nn.Module):
    def __init__(
        self, in_dim: int, n_au: int = 12, hidden: int = 128, dropout: float = 0.3
    ):
        super().__init__()
        self.au = AUHead(in_dim, n_au, hidden, dropout)
        self.va = VAHead(in_dim, hidden, dropout)

    def forward(self, x):
        return {"au": self.au(x), "va": self.va(x)}


def _ccc_loss(pred, target, eps: float = 1e-8):
    pm, tm = pred.mean(0), target.mean(0)
    pv, tv = pred.var(0, unbiased=False), target.var(0, unbiased=False)
    cov = ((pred - pm) * (target - tm)).mean(0)
    return (1 - 2 * cov / (pv + tv + (pm - tm) ** 2 + eps)).mean()


def masked_multitask_loss(
    out: dict, va, va_mask, au, au_mask, w_au: float = 1.0, w_va: float = 1.0
):
    """Return (rugi total, rincian). Bagian tanpa sampel valid bernilai 0 tanpa gradien NaN."""
    zero = out["va"].sum() * 0.0
    m = va_mask > 0
    loss_va = _ccc_loss(out["va"][m], va[m]) if m.sum() >= 2 else zero
    n_au = au_mask.sum()
    loss_au = (((out["au"] - au) ** 2) * au_mask).sum() / n_au if n_au > 0 else zero
    total = w_au * loss_au + w_va * loss_va
    return total, {"au": float(loss_au.detach()), "va": float(loss_va.detach())}
