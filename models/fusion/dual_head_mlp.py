"""Fusi multimodal TERLATIH (alternatif dari VA-bridge berbasis aturan).

Menggabungkan fitur wajah (embedding ST-GCN / VA prediksi) dengan fitur
fisiologis (HR, HRV, arousal-proxy dari rPPG) melalui MLP + dropout, lalu
menghasilkan dua kepala:
  - va     : regresi (valence, arousal) dengan aktivasi tanh -> [-1, 1]
  - logits : klasifikasi 5 mood (Ceria, Cemas, Tertekan, Rileks, Netral)

Gunakan ini bila nanti ada data berlabel cukup untuk MELATIH bobot fusi
(w1, w2 tidak lagi manual). Untuk demo/real-time awal, LateFusionVABridge
sudah memadai.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DualHeadFusionMLP(nn.Module):
    def __init__(self, face_dim=2, physio_dim=3, hidden=(128, 64),
                 num_class=5, dropout=0.3):
        """face_dim  : dimensi fitur wajah (default 2 = valence,arousal ST-GCN;
                     bisa dinaikkan bila memakai embedding).
        physio_dim : dimensi fitur fisiologis (default 3 = hr, hrv, arousal_proxy).
        """
        super().__init__()
        in_dim = face_dim + physio_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.va_head = nn.Sequential(nn.Linear(prev, 2), nn.Tanh())
        self.cls_head = nn.Linear(prev, num_class)

    def forward(self, face_feat, physio_feat):
        """face_feat: (B, face_dim); physio_feat: (B, physio_dim).
        Return dict {"va": (B,2), "logits": (B,num_class)}."""
        z = torch.cat([face_feat, physio_feat], dim=-1)
        z = self.backbone(z)
        return {"va": self.va_head(z), "logits": self.cls_head(z)}


def build_physio_features(hr_bpm, hrv_ms, arousal_proxy):
    """Bantu susun tensor fitur fisiologis ternormalisasi (B,3)."""
    hr = (torch.as_tensor(hr_bpm, dtype=torch.float32) - 70.0) / 30.0
    hrv = (torch.as_tensor(hrv_ms, dtype=torch.float32) - 50.0) / 40.0
    ar = torch.as_tensor(arousal_proxy, dtype=torch.float32)
    return torch.stack([hr, hrv, ar], dim=-1)


if __name__ == "__main__":
    m = DualHeadFusionMLP()
    m.eval()
    f = torch.randn(4, 2)
    p = torch.randn(4, 3)
    out = m(f, p)
    print({k: tuple(v.shape) for k, v in out.items()})
