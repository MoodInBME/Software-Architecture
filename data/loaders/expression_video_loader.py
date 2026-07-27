"""Dataset landmark ekspresi untuk cabang ST-GCN (regresi VA).

Membaca manifest.json hasil data/preprocessing/extract_landmarks_video.py:
    [
      {"file": "clips/S005_001_000.pt", "valence": 0.8, "arousal": 0.5,
       "subject": "S005", "source": "ckplus"},
      ...
    ]

Setiap file .pt berisi dict:
    {"landmarks": FloatTensor (C, T, V), "valence": float, "arousal": float,
     "subject": str, "source": str}

Mengembalikan per item: (x (C,T,V) float32, target_va (2,) float32, mood int64)
-- cocok dengan iterasi di models/st_gcn_graph_tran/train.py:
    for x, target_va, mood in loader: ...
"""
from __future__ import annotations

import json
import os

import torch
from torch.utils.data import Dataset

try:
    from models.st_gcn_graph_tran.labels import va_to_mood_id, NEUTRAL_RADIUS
except Exception:  # fallback mandiri
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


class ExpressionLandmarkDataset(Dataset):
    """Dataset klip landmark wajah -> (landmarks, target_va, mood_id)."""

    def __init__(self, manifest, clip_len=150, neutral_radius=NEUTRAL_RADIUS, root=None):
        with open(manifest, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        if not self.items:
            raise ValueError(f"Manifest kosong: {manifest}")
        self.clip_len = int(clip_len)
        self.neutral_radius = float(neutral_radius)
        # path .pt di manifest boleh relatif terhadap folder manifest
        self.root = root or os.path.dirname(os.path.abspath(manifest))

    def __len__(self):
        return len(self.items)

    def _resolve(self, path):
        return path if os.path.isabs(path) else os.path.join(self.root, path)

    @staticmethod
    def _fix_len(x, length):
        """x: (C, T, V) -> pad (edge-repeat) atau center-crop di dimensi T."""
        t = x.shape[1]
        if t == length:
            return x
        if t > length:
            start = (t - length) // 2
            return x[:, start:start + length, :].contiguous()
        pad = length - t
        last = x[:, -1:, :].repeat(1, pad, 1)
        return torch.cat([x, last], dim=1).contiguous()

    def __getitem__(self, idx):
        it = self.items[idx]
        data = torch.load(self._resolve(it["file"]), map_location="cpu", weights_only=True)
        x = data["landmarks"].float()
        # Toleransi tata-letak: kalau tersimpan sebagai (T, V, C) -> jadikan (C, T, V)
        if x.dim() == 3 and x.shape[0] not in (2, 3) and x.shape[-1] in (2, 3):
            x = x.permute(2, 0, 1).contiguous()
        x = self._fix_len(x, self.clip_len)
        v = float(it.get("valence", data.get("valence", 0.0)))
        a = float(it.get("arousal", data.get("arousal", 0.0)))
        target_va = torch.tensor([v, a], dtype=torch.float32)
        mood = torch.tensor(va_to_mood_id(v, a, self.neutral_radius), dtype=torch.long)
        return x, target_va, mood
