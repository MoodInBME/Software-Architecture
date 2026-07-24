"""Training script untuk STC-rPPG.

Menjalankan training end-to-end: memuat data lewat UBFCrPPGDataset (split
per-subjek 8:2), melatih model STCrPPG dengan STCHybridLoss, mengevaluasi
error HR (MAE/RMSE) tiap epoch lewat rPPGProcessor, dan menyimpan checkpoint.

Cara pakai (dijalankan sebagai modul dari root project, bukan langsung
`python train.py`, supaya relative import `.fsfe`, `.stfe`, dst berfungsi):

    python -m stc_rppg.train --config stc_rppg/config.yaml

Kalau `stc_rppg` sudah jadi working directory Anda, bisa juga:

    python -m train --config config.yaml   (dijalankan dari luar folder stc_rppg)
"""
import os
import sys
import time
import argparse
import importlib.util

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

from .model import STCrPPG
from .loss import STCHybridLoss
from .postprocessing import rPPGProcessor


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_module_from_path(module_name, file_path):
    """Import modul Python dari path file secara langsung, tanpa perlu
    modul tsb berada di sys.path atau jadi bagian dari package manapun.
    Dipakai untuk mengimpor dataset_ubfc_loader.py yang letaknya di luar
    package stc_rppg (mis. di data/loaders/)."""
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Tidak menemukan loader module di: {file_path}\n"
            f"Cek field 'data.loader_module_path' di config.yaml -- path "
            f"tsb dihitung relatif terhadap file train.py ini."
        )
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg):
    name = cfg["training"].get("optimizer", "adadelta").lower()
    lr = cfg["training"]["lr"]
    if name == "adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=lr)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer '{name}' tidak dikenali (pilih: adadelta/adam/sgd)")


def make_dataloader(loader_mod, cfg, subject_ids, shuffle):
    dcfg = cfg["data"]
    dataset = loader_mod.UBFCrPPGDataset(
        prepared_dir=dcfg["prepared_dir"],
        clip_len=dcfg["clip_len"],
        stride=dcfg["stride"],
        normalize_target=True,
        subject_ids=subject_ids,
    )
    num_workers = dcfg.get("num_workers", 0)
    persistent = dcfg.get("persistent_workers", False) and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=dcfg["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent,
        pin_memory=dcfg.get("pin_memory", False),
    ), dataset


@torch.no_grad()
def evaluate(model, loader, criterion, processor, device):
    """Kembalikan dict berisi loss rata-rata dan MAE/RMSE HR (bpm).

    HR dihitung dari sinyal prediksi vs sinyal target (keduanya sudah
    ternormalisasi per-klip) lewat FFT band-pass. Karena estimasi frekuensi
    dominan tidak terpengaruh skala amplitudo, ini metrik yang valid untuk
    membandingkan HR walau sinyal tidak dalam satuan fisik aslinya.
    """
    model.eval()
    total_loss, total_time, total_freq, n_batches = 0.0, 0.0, 0.0, 0
    abs_errors = []

    for vclip, pclip in loader:
        vclip = vclip.to(device, non_blocking=True)
        pclip = pclip.to(device, non_blocking=True)

        pred = model(vclip)
        loss, loss_time, loss_freq = criterion(pred, pclip)

        total_loss += loss.item()
        total_time += loss_time.item()
        total_freq += loss_freq.item()
        n_batches += 1

        pred_np = pred.detach().cpu().numpy()
        gt_np = pclip.detach().cpu().numpy()
        for b in range(pred_np.shape[0]):
            pred_filt = processor.filter_signal(pred_np[b])
            gt_filt = processor.filter_signal(gt_np[b])
            pred_bpm = processor.calculate_bpm(pred_filt)
            gt_bpm = processor.calculate_bpm(gt_filt)
            abs_errors.append(abs(pred_bpm - gt_bpm))

    abs_errors = np.asarray(abs_errors) if abs_errors else np.array([0.0])
    return {
        "loss": total_loss / max(n_batches, 1),
        "loss_time": total_time / max(n_batches, 1),
        "loss_freq": total_freq / max(n_batches, 1),
        "hr_mae": float(np.mean(abs_errors)),
        "hr_rmse": float(np.sqrt(np.mean(abs_errors ** 2))),
    }


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip_norm, log_every, epoch):
    model.train()
    running_loss, running_time, running_freq = 0.0, 0.0, 0.0
    t0 = time.time()

    for step, (vclip, pclip) in enumerate(loader):
        vclip = vclip.to(device, non_blocking=True)
        pclip = pclip.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(vclip)
        loss, loss_time, loss_freq = criterion(pred, pclip)
        loss.backward()

        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        running_loss += loss.item()
        running_time += loss_time.item()
        running_freq += loss_freq.item()

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"  [epoch {epoch}] step {step + 1}/{len(loader)} "
                  f"loss={loss.item():.4f} (time={loss_time.item():.4f}, "
                  f"freq={loss_freq.item():.4f}) | {elapsed:.1f}s")
            t0 = time.time()

    n = len(loader)
    return {
        "loss": running_loss / n,
        "loss_time": running_time / n,
        "loss_freq": running_freq / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"].get("seed", 42))

    device_name = cfg["training"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and device.type == "cpu":
        print("Peringatan: CUDA diminta di config tapi tidak tersedia -> jatuh ke CPU.")
    print(f"Menggunakan device: {device}")

    # --- Muat loader dari luar package ---
    loader_rel_path = cfg["data"].get(
        "loader_module_path", "../data/loaders/dataset_ubfc_loader.py"
    )
    loader_abs_path = os.path.join(os.path.dirname(__file__), loader_rel_path)
    loader_mod = load_module_from_path("dataset_ubfc_loader", loader_abs_path)

    # --- Split subjek & dataset ---
    prepared_dir = cfg["data"]["prepared_dir"]
    train_ratio = cfg["data"].get("train_ratio", 0.8)
    seed = cfg["data"].get("seed", 42)
    train_ids, test_ids = loader_mod.split_subjects(prepared_dir, train_ratio=train_ratio, seed=seed)
    print(f"Subjek train: {len(train_ids)} | Subjek test: {len(test_ids)}")

    train_loader, train_set = make_dataloader(loader_mod, cfg, train_ids, shuffle=True)
    test_loader, test_set = make_dataloader(loader_mod, cfg, test_ids, shuffle=False)
    print(f"Klip train: {len(train_set)} | Klip test: {len(test_set)}")

    # --- Model, loss, optimizer ---
    mcfg = cfg["model"]
    model = STCrPPG().to(device)
    # NB: STCrPPG saat ini belum menerima parameter cdc_theta/channels dari
    # config secara langsung (constructor-nya masih hardcode default). Kalau
    # nanti mau dibuat konfigurable, cukup ubah signature STCrPPG.__init__.

    criterion = STCHybridLoss(
        lambda_weight=cfg["training"].get("lambda_weight", 0.5),
        fps=cfg["loss"].get("fps", 30.0),
        low=cfg["loss"].get("freq_low", 0.7),
        high=cfg["loss"].get("freq_high", 3.5),
    )

    optimizer = build_optimizer(model, cfg)

    processor = rPPGProcessor(
        fps=cfg["postprocessing"].get("fps", 30.0),
        lowcut=cfg["postprocessing"].get("lowcut", 0.7),
        highcut=cfg["postprocessing"].get("highcut", 3.5),
        order=cfg["postprocessing"].get("order", 4),
    )

    # --- Checkpoint dir ---
    ckpt_dir = cfg["training"].get("checkpoint_dir", "checkpoints")
    ckpt_dir = os.path.join(os.path.dirname(__file__), ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    epochs = cfg["training"]["epochs"]
    save_every = cfg["training"].get("save_every", 5)
    log_every = cfg["training"].get("log_every", 20)
    grad_clip_norm = cfg["training"].get("grad_clip_norm")

    best_hr_mae = float("inf")
    history = []

    print(f"\nMulai training: {epochs} epoch, optimizer={cfg['training'].get('optimizer')}, "
          f"lr={cfg['training']['lr']}, lambda={cfg['training'].get('lambda_weight')}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip_norm, log_every, epoch,
        )
        val_stats = evaluate(model, test_loader, criterion, processor, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
              f"train_loss={train_stats['loss']:.4f} | "
              f"val_loss={val_stats['loss']:.4f} | "
              f"val_HR_MAE={val_stats['hr_mae']:.2f} bpm | "
              f"val_HR_RMSE={val_stats['hr_rmse']:.2f} bpm")

        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()},
                         **{f"val_{k}": v for k, v in val_stats.items()}})

        # simpan checkpoint berkala
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "val_stats": val_stats,
            }, ckpt_path)
            print(f"  Checkpoint disimpan: {ckpt_path}")

        # simpan model terbaik berdasar HR MAE di test set
        if val_stats["hr_mae"] < best_hr_mae:
            best_hr_mae = val_stats["hr_mae"]
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": cfg,
                "val_stats": val_stats,
            }, best_path)
            print(f"  Model terbaik baru (HR MAE={best_hr_mae:.2f} bpm) -> {best_path}")

    print(f"\nTraining selesai. Best HR MAE (test): {best_hr_mae:.2f} bpm")
    return history


if __name__ == "__main__":
    main()