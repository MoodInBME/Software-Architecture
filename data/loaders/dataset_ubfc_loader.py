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
import argparse
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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
        subject_ids: opsional, daftar nama subjek (tanpa .pt) untuk membatasi
            dataset ini pada subset tertentu -- dipakai untuk split train/test
            per-subjek (bukan per-klip), supaya tidak ada kebocoran data.
    """

    def __init__(self, prepared_dir, clip_len=128, stride=64,
                 normalize_target=True, subject_ids=None):
        all_paths = sorted(glob.glob(os.path.join(prepared_dir, "*.pt")))
        if not all_paths:
            raise FileNotFoundError(f"Tidak ada .pt di {prepared_dir}")

        if subject_ids is not None:
            wanted = set(subject_ids)
            all_paths = [p for p in all_paths
                         if os.path.splitext(os.path.basename(p))[0] in wanted]
            if not all_paths:
                raise ValueError("subject_ids diberikan tapi tidak ada yang cocok "
                                  f"dengan file .pt di {prepared_dir}")

        self.paths = all_paths
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.normalize_target = normalize_target
        self.index = self._build_index()

    def _build_index(self):
        index = []
        for pi, p in enumerate(self.paths):
            meta = torch.load(p, weights_only=True)
            n_frames_video = int(meta["video"].shape[1])
            n_frames_ppg = int(meta["ppg"].shape[0])
            if n_frames_video != n_frames_ppg:
                raise ValueError(
                    f"Panjang video ({n_frames_video}) != panjang ppg "
                    f"({n_frames_ppg}) pada file {p}"
                )
            last = max(0, n_frames_video - self.clip_len)
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


def list_subjects(prepared_dir):
    """Kembalikan daftar nama subjek (tanpa ekstensi .pt) di sebuah folder."""
    paths = sorted(glob.glob(os.path.join(prepared_dir, "*.pt")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def split_subjects(prepared_dir, train_ratio=0.8, seed=42):
    """Split subjek (bukan klip) menjadi train/test sesuai rasio paper (8:2)."""
    subjects = list_subjects(prepared_dir)
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_ratio))
    return shuffled[:n_train], shuffled[n_train:]


def _sanity_check(prepared_dir, clip_len, stride, batch_size, num_workers, max_batches):
    print(f"[1/4] Menyusun daftar subjek di: {prepared_dir}")
    subjects = list_subjects(prepared_dir)
    print(f"      Ditemukan {len(subjects)} subjek: {subjects[:5]}"
          f"{' ...' if len(subjects) > 5 else ''}")
    if not subjects:
        return

    print(f"[2/4] Split subjek 8:2 (train/test)...")
    train_ids, test_ids = split_subjects(prepared_dir, train_ratio=0.8)
    print(f"      Train: {len(train_ids)} subjek -> {train_ids}")
    print(f"      Test : {len(test_ids)} subjek -> {test_ids}")

    print(f"[3/4] Membangun UBFCrPPGDataset (clip_len={clip_len}, stride={stride})...")
    t0 = time.time()
    train_set = UBFCrPPGDataset(prepared_dir, clip_len=clip_len, stride=stride,
                                 subject_ids=train_ids)
    print(f"      Selesai membangun index dalam {time.time() - t0:.2f}s")
    print(f"      Total klip train: {len(train_set)}")

    if test_ids:
        test_set = UBFCrPPGDataset(prepared_dir, clip_len=clip_len, stride=stride,
                                    subject_ids=test_ids)
        print(f"      Total klip test : {len(test_set)}")

    print(f"[4/4] Mengambil {max_batches} batch lewat DataLoader "
          f"(batch_size={batch_size}, num_workers={num_workers})...")
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                         num_workers=num_workers)

    t0 = time.time()
    for i, (vclip, pclip) in enumerate(loader):
        elapsed = time.time() - t0
        print(f"      Batch {i}: video={tuple(vclip.shape)} "
              f"dtype={vclip.dtype} range=[{vclip.min():.3f}, {vclip.max():.3f}] | "
              f"ppg={tuple(pclip.shape)} mean={pclip.mean():.3f} std={pclip.std():.3f} | "
              f"{elapsed:.2f}s")
        t0 = time.time()
        if i + 1 >= max_batches:
            break

    print("\nSanity check selesai. Jika shape dan range di atas sesuai ekspektasi "
          "(video ~[0,1], ppg mean~0 std~1), loader siap dipakai untuk training.")


def main():
    ap = argparse.ArgumentParser(
        description="Verifikasi UBFCrPPGDataset: cek index, split subjek, dan "
                     "ambil beberapa batch contoh."
    )
    ap.add_argument("--prepared_dir", required=True,
                     help="folder berisi file .pt hasil prepare_data_ubfc.py")
    ap.add_argument("--clip_len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=3,
                     help="jumlah batch contoh yang diambil untuk verifikasi")
    args = ap.parse_args()

    _sanity_check(
        prepared_dir=args.prepared_dir,
        clip_len=args.clip_len,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()