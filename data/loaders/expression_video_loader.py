"""Dataset landmark ekspresi (ter-cache) -> tensor (C, T, V) + target VA.

Membaca manifest JSON hasil `data/preprocessing/extract_landmarks_video.py`.
Setiap entri manifest:
    {
      "path": "landmarks/xxx.npy",   # array (C, T, V) float32
      "label_type": "va" | "emotion",
      "value": [v, a]  ATAU  "happy"   # sesuai label_type
    }

Target selalu dikembalikan dalam ruang VA (2,) + indeks mood (untuk metrik).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.loaders.label_adapters import (
    emotion_name_to_va, affwild2_va, va_to_mood_index,
)


class ExpressionLandmarkDataset(Dataset):
    def __init__(self, manifest_path: str, clip_len: int = 150,
                 neutral_radius: float = 0.2, root: str | None = None):
        self.clip_len = clip_len
        self.neutral_radius = neutral_radius
        self.root = Path(root) if root else Path(manifest_path).parent
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        if not self.items:
            raise ValueError(f"Manifest kosong: {manifest_path}")

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_va(self, item):
        if item["label_type"] == "va":
            v, a = item["value"]
            return affwild2_va(v, a)
        return emotion_name_to_va(str(item["value"]))

    def _fit_clip(self, arr: np.ndarray) -> np.ndarray:
        c, t, v = arr.shape
        if t == self.clip_len:
            return arr
        if t > self.clip_len:
            idx = np.linspace(0, t - 1, self.clip_len).round().astype(np.int64)
            return arr[:, idx, :]
        last = arr[:, -1:, :].repeat(self.clip_len - t, axis=1)
        return np.concatenate([arr, last], axis=1)

    def __getitem__(self, i: int):
        item = self.items[i]
        arr = np.load(self.root / item["path"]).astype(np.float32)  # (C, T, V)
        arr = self._fit_clip(arr)
        x = torch.from_numpy(arr)
        v, a = self._resolve_va(item)
        target_va = torch.tensor([v, a], dtype=torch.float32)
        mood = va_to_mood_index(v, a, self.neutral_radius)
        return x, target_va, mood
