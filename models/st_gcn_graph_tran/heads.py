"""Head keluaran fleksibel untuk cabang ST-GCN.

Sesuai keputusan VA-bridge, default = regresi Valence-Arousal (2 neuron). Head
5-kelas dan dual-head disediakan untuk eksperimen/ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VAHead(nn.Module):
    """Regresi (valence, arousal) dalam rentang [-1, 1] via tanh."""

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 2)


class ClassHead(nn.Module):
    """Logits 5 kelas mood (dipakai bila ingin klasifikasi langsung)."""

    def __init__(self, in_dim: int, num_class: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, num_class)


class DualHead(nn.Module):
    """Multi-task: regresi VA + logits 5 kelas (berbagi backbone)."""

    def __init__(self, in_dim: int, num_class: int = 5, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.va = VAHead(in_dim, hidden, dropout)
        self.cls = ClassHead(in_dim, num_class, dropout)

    def forward(self, x: torch.Tensor) -> dict:
        return {"va": self.va(x), "logits": self.cls(x)}


def build_head(head_type: str, in_dim: int, num_class: int = 5,
               hidden: int = 128, dropout: float = 0.3) -> nn.Module:
    head_type = head_type.lower()
    if head_type == "va":
        return VAHead(in_dim, hidden, dropout)
    if head_type in ("class", "cls", "categorical"):
        return ClassHead(in_dim, num_class, dropout)
    if head_type == "dual":
        return DualHead(in_dim, num_class, hidden, dropout)
    raise ValueError(f"head_type tak dikenal: {head_type!r} (pilih: va|class|dual)")
