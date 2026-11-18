"""Pilihan aturan pemetaan valensi-arousal ke lima kategori mood.

* ``tabel33`` (bawaan): aturan rancangan Pra-TA di ``labels_tabel33.py``.
* ``lingkaran``: aturan era GEMASTIK di ``labels.py`` (zona Netral berbentuk
  lingkaran), dipertahankan untuk arsip dan perbandingan.

Kombinasi yang belum tercakup aturan ``tabel33`` diberi ``MOOD_GAP_ID`` (-1).
Nilai ini diabaikan rugi klasifikasi dan dikeluarkan dari metrik kategori.
"""

from __future__ import annotations

from torch.utils.data import Dataset

from .labels import va_to_mood_id
from .labels_tabel33 import va_to_mood_id_tabel33

MOOD_RULES = ("tabel33", "lingkaran")
MOOD_GAP_ID = -1


def mood_id_function(rule: str = "tabel33", neutral_radius: float = 0.2):
    """Return fungsi (v, a) -> id mood 0..4, atau MOOD_GAP_ID untuk celah aturan."""
    rule = str(rule).lower()
    if rule == "tabel33":

        def fn(v, a):
            m = va_to_mood_id_tabel33(float(v), float(a))
            return MOOD_GAP_ID if m is None else m

        return fn
    if rule == "lingkaran":

        def fn(v, a):
            return va_to_mood_id(float(v), float(a), neutral_radius)

        return fn
    raise ValueError(f"mood_rule harus salah satu dari {MOOD_RULES}, bukan {rule!r}")


class MoodRuleDataset(Dataset):
    """Bungkus dataset (x, va, mood) dan hitung ulang mood dari va dengan aturan pilihan.

    Dipakai agar ``ExpressionLandmarkDataset`` (aturan lingkaran bawaan) tidak perlu
    diubah. Label VA tidak disentuh.
    """

    def __init__(self, base, mood_fn):
        self.base = base
        self.mood_fn = mood_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        import torch

        x, va, _ = self.base[idx]
        mood = torch.tensor(self.mood_fn(va[0], va[1]), dtype=torch.long)
        return x, va, mood
