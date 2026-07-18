"""Loader UBFC-rPPG untuk melatih STC-rPPG.

Membaca sampel .pt hasil prepare_data_ubfc.py yang berisi pasangan tersinkron
(video wajah + gelombang PPG ground-truth), lalu memotongnya menjadi klip
temporal (jendela geser) siap-latih.

Format ground_truth.txt UBFC-rPPG DATASET_2 (VERIFIKASI dgn unduhan aktual):
    baris 0: gelombang PPG (trace)
    baris 1: detak jantung (BPM) per sampel
    baris 2: timestamp (detik)  [opsional]
"""
import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset


def load_ubfc_ground_truth(gt_path):
    """Kembalikan (ppg, hr, ts) sebagai np.float32; ts bisa None."""
    with open(gt_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"Format ground_truth.txt tak dikenali: {gt_path}")
    ppg = np.asarray(lines[0].split(), dtype=np.float32)
    hr = np.asarray(lines[1].split(), dtype=np.float32)
    ts = np.asarray(lines[2].split(), dtype=np.float32) if len(lines) >= 3 else None
    return ppg, hr, ts


class UBFCrPPGDataset(Dataset):
    """Menghasilkan (klip video (C,T,H,W), target PPG (T,)) tersinkron.

    Args:
        prepared_dir: folder .pt hasil prepare_data_ubfc.py.
        clip_len: panjang jendela temporal (frame).
        stride: langkah antar-jendela (frame).
        normalize_target: standarisasi PPG per-klip (mean 0, std 1) -> stabil
            untuk Negative-Pearson loss.
    """

    def __init__(self, prepared_dir, clip_len=128, stride=64, normalize_target=True):
        self.paths = sorted(glob.glob(os.path.join(prepared_dir, "*.pt")))
        if not self.paths:
            raise FileNotFoundError(f"Tidak ada .pt di {prepared_dir}")
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.normalize_target = normalize_target
        self.index = self._build_index()

    def _build_index(self):
        index = []
        for pi, p in enumerate(self.paths):
            meta = torch.load(p, weights_only=True)
            n_frames = int(meta["video"].shape[1])
            last = max(0, n_frames - self.clip_len)
            starts = list(range(0, last + 1, self.stride)) or [0]
            for s in starts:
                index.append((pi, s))
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pi, start = self.index[idx]
        sample = torch.load(self.paths[pi], weights_only=True)
        video = sample["video"].float()   # (C,T,H,W)
        ppg = sample["ppg"].float()       # (T,)

        vclip = self._fix_len(video[:, start:start + self.clip_len], self.clip_len, dim=1)
        pclip = self._fix_len(ppg[start:start + self.clip_len], self.clip_len, dim=0)

        if self.normalize_target:
            pclip = (pclip - pclip.mean()) / (pclip.std() + 1e-6)
        return vclip, pclip

    @staticmethod
    def _fix_len(x, length, dim):
        cur = x.shape[dim]
        if cur == length:
            return x
        if cur > length:
            return x.narrow(dim, 0, length).contiguous()
        # ulangi indeks terakhir (edge-pad) alih-alih nol -> hindari artefak spektral
        idx = torch.cat([
            torch.arange(cur),
            torch.full((length - cur,), cur - 1, dtype=torch.long),
        ])
        return x.index_select(dim, idx).contiguous()
