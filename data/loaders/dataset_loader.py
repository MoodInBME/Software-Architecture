"""Base dataset untuk MoodIn_Software.

Memuat sampel terpreproses (.pt) hasil `prepare_data_*.py`. Setiap file .pt
berisi dict minimal: {"video": Tensor(C,T,H,W), "target": Tensor(...)}.

Gunakan kelas spesifik (mis. UBFCrPPGDataset) bila butuh windowing + sinkronisasi
label; kelas di sini untuk kasus generik/sederhana.
"""
import os
import glob

import torch
from torch.utils.data import Dataset


class TensorClipDataset(Dataset):
    """Dataset generik untuk sampel .pt {"video": (C,T,H,W), "target": ...}."""

    def __init__(self, root_dir, transform=None, video_key="video", target_key="target"):
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        if not self.paths:
            raise FileNotFoundError(f"Tidak ada file .pt di {root_dir}")
        self.transform = transform
        self.video_key = video_key
        self.target_key = target_key

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample = torch.load(self.paths[idx], weights_only=True)
        video = sample[self.video_key].float()
        target = sample[self.target_key]
        if self.transform is not None:
            video = self.transform(video)
        return video, target
