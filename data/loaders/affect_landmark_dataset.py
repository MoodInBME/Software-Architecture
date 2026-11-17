"""Dataset klip landmark dengan label VA dan intensitas AU bermask (AFEW-VA, DISFA, gabungan).

Membaca satu atau beberapa manifest dari `afew_va.py` dan `disfa.py`. Setiap item:
    x (C, T, V) float32, va (2,) float32, va_mask () float32, au (K,) float32, au_mask (K,) float32
Label yang tidak tersedia bernilai 0 dengan mask 0, sehingga rugi bermask
(`models/st_gcn_graph_tran/au_va_heads.masked_multitask_loss`) mengabaikannya.
Contoh: klip AFEW-VA punya VA tetapi tidak punya AU; klip DISFA sebaliknya.

Pembagian data harus berbasis subjek (`subject_split.py`) agar wajah yang sama tidak
muncul di data latih dan uji.
"""

from __future__ import annotations

import json
import os

import torch
from torch.utils.data import Dataset


class AffectLandmarkDataset(Dataset):
    def __init__(
        self,
        manifests,
        au_list,
        clip_len: int = 90,
        subjects=None,
        max_filled: float = 0.2,
    ):
        if isinstance(manifests, str):
            manifests = [manifests]
        self.au_list = list(au_list)
        self.clip_len = int(clip_len)
        self.items = []
        for m in manifests:
            root = os.path.dirname(os.path.abspath(m))
            meta_path = os.path.join(root, "meta.json")
            src_aus = None
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    src_aus = json.load(f).get("au_list")
            with open(m, encoding="utf-8") as f:
                for it in json.load(f):
                    if subjects is not None and it["subject"] not in subjects:
                        continue
                    if it.get("frac_filled", 0.0) > max_filled:
                        continue
                    self.items.append((root, it, src_aus))
        if not self.items:
            raise ValueError(
                "Tidak ada klip setelah penyaringan subjek dan frac_filled"
            )

    def __len__(self):
        return len(self.items)

    def subjects(self):
        return sorted({it["subject"] for _, it, _ in self.items})

    @staticmethod
    def _fix_len(x, length):
        t = x.shape[1]
        if t == length:
            return x
        if t > length:
            s = (t - length) // 2
            return x[:, s : s + length].contiguous()
        return torch.cat([x, x[:, -1:].repeat(1, length - t, 1)], dim=1).contiguous()

    def __getitem__(self, idx):
        root, it, src_aus = self.items[idx]
        data = torch.load(
            os.path.join(root, it["file"]), map_location="cpu", weights_only=True
        )
        x = self._fix_len(data["landmarks"].float(), self.clip_len)
        has_va = float(it.get("va_mask", 0.0)) > 0
        va = torch.tensor(
            [it["valence"], it["arousal"]] if has_va else [0.0, 0.0],
            dtype=torch.float32,
        )
        au = torch.zeros(len(self.au_list))
        au_mask = torch.zeros(len(self.au_list))
        if src_aus and it.get("au"):
            for j, a in enumerate(self.au_list):
                if a in src_aus:
                    k = src_aus.index(a)
                    if it["au_mask"][k] > 0 and it["au"][k] is not None:
                        au[j] = float(it["au"][k])
                        au_mask[j] = 1.0
        return x, va, torch.tensor(float(has_va)), au, au_mask
