"""V2: latih ST-GCN kepala valensi-arousal pada AFEW-VA, GroupKFold per aktor.

Masukan: folder hasil ``python data/loaders/afew_va.py prepare --landmarks gt`` (manifest
.json dan klip .pt landmark acuan, di luar repo). Setiap nilai T butuh folder prepare sendiri
(``--clip_len 90`` atau ``48``, ``--stride 15``).

Per lipatan: model baru dengan seed tetap, jumlah epoch tetap (tanpa pemilihan epoch pada
lipatan uji), rugi MSE + (1 - CCC) per sumbu, mask frame padding dipakai. Metrik pada data
uji lipatan: CCC valensi dan arousal, akurasi lima kelas Tabel 3.3 (prediksi memakai mode
evaluasi tanpa keadaan; label di celah aturan dikeluarkan dan dihitung), F1 makro dan
akurasi seimbang atas kelas yang muncul di label uji, pembanding kelas mayoritas (kelas
terbanyak di data latih lipatan, ditebak untuk semua jendela uji), sebaran label dan
prediksi dengan nama kelas, waktu latih, dan VRAM puncak. Bobot disimpan hanya di folder
keluaran (di luar repo).

Arsitektur dibaca dari ``models/st_gcn_graph_tran/config.yaml`` (rancangan B.3: empat unit,
tanpa Transformer). ``--arsitektur lama`` memakai susunan enam unit + Transformer untuk
dibandingkan dengan hasil V2 sebelumnya.

Catatan AFEW-VA: kelas Rileks (v > 0, a < 0 di luar pita Netral) kosong karena datanya,
bukan karena pemetaan. Pada manifest T90 tidak ada satu pun jendela dengan v > 0 dan
a < 0. F1 makro dan akurasi seimbang karena itu hanya merata-ratakan kelas yang muncul di
label uji, dan daftar kelas yang dipakai ikut dilaporkan.

    python -m scripts.validasi.v2_afew_stgcn --manifest D:/moodin_runs/prepared/afew-va-T90/manifest.json --T 90 --keluaran D:/moodin_runs/v2/T90
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from scripts.validasi._umum import (
    REPO,
    info_lingkungan,
    pastikan_di_luar_repo,
    tulis_json,
)


def rugi_va(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    mse = ((pred - target) ** 2).mean()
    pm, tm = pred.mean(0), target.mean(0)
    pv, tv = pred.var(0, unbiased=False), target.var(0, unbiased=False)
    cov = ((pred - pm) * (target - tm)).mean(0)
    ccc = 2 * cov / (pv + tv + (pm - tm) ** 2 + eps)
    return mse + (1 - ccc).mean()


def metrik_kelas(lab: np.ndarray, pre: np.ndarray, n_kelas: int = 5) -> dict:
    """F1 makro dan akurasi seimbang atas kelas yang muncul di ``lab``.

    Kelas yang tidak pernah muncul di label tidak punya recall, sehingga tidak ikut dirata-rata;
    prediksi ke kelas itu tetap terhitung salah.
    """
    from sklearn.metrics import f1_score

    if lab.size == 0:
        return {"f1_makro": None, "akurasi_seimbang": None, "kelas_dinilai": []}
    hadir = sorted(int(k) for k in np.unique(lab))
    return {
        "f1_makro": float(f1_score(lab, pre, labels=hadir, average="macro", zero_division=0)),
        # Rerata recall per kelas yang hadir; dihitung langsung karena sklearn memberi
        # peringatan bila prediksi memuat kelas yang tidak ada di label.
        "akurasi_seimbang": float(np.mean([(pre[lab == k] == k).mean() for k in hadir])),
        "kelas_dinilai": hadir,
    }


def sebaran(ids: np.ndarray, nama) -> dict:
    return {nama[k]: int((ids == k).sum()) for k in range(len(nama))}


def label_mood(va: np.ndarray) -> np.ndarray:
    from models.st_gcn_graph_tran.mood_rules import mood_id_function

    fn = mood_id_function("tabel33", kind="label")
    return np.array([fn(v, a) for v, a in va], dtype=np.int64)


def evaluasi(model, loader, device, kelas_mayoritas: int | None = None) -> dict:
    from evaluation.affect_metrics import ccc
    from models.st_gcn_graph_tran.labels import MOOD_CLASSES
    from models.st_gcn_graph_tran.mood_rules import MOOD_GAP_ID, mood_id_function

    pred_fn = mood_id_function("tabel33", kind="prediction")
    label_fn = mood_id_function("tabel33", kind="label")
    model.eval()
    p_all, t_all = [], []
    with torch.no_grad():
        for x, va, has_va, _au, _aum, m in loader:
            keep = has_va > 0
            if not keep.any():
                continue
            out = (
                model(x[keep].to(device), frame_mask=m[keep].to(device))["va"]
                .float()
                .cpu()
            )
            p_all.append(out.clamp(-1, 1).numpy())
            t_all.append(va[keep].numpy())
    p = np.concatenate(p_all)
    t = np.concatenate(t_all)
    lab = np.array([label_fn(v, a) for v, a in t])
    pre = np.array([pred_fn(v, a) for v, a in p])
    ok = lab != MOOD_GAP_ID
    lab, pre = lab[ok], pre[ok]
    hasil = {
        "n_jendela": int(len(t)),
        "ccc_valensi": ccc(t[:, 0], p[:, 0]),
        "ccc_arousal": ccc(t[:, 1], p[:, 1]),
        "akurasi_5_kelas": float((pre == lab).mean()) if lab.size else None,
        **metrik_kelas(lab, pre),
        "n_celah_dikeluarkan": int((~ok).sum()),
        "sebaran_label": sebaran(lab, MOOD_CLASSES),
        "sebaran_prediksi": sebaran(pre, MOOD_CLASSES),
    }
    hasil["kelas_dinilai"] = [MOOD_CLASSES[k] for k in hasil["kelas_dinilai"]]
    if kelas_mayoritas is not None:
        dasar = np.full_like(lab, kelas_mayoritas)
        m = metrik_kelas(lab, dasar)
        hasil["mayoritas"] = {
            "kelas": MOOD_CLASSES[kelas_mayoritas],
            "akurasi_5_kelas": float((dasar == lab).mean()) if lab.size else None,
            "f1_makro": m["f1_makro"],
            "akurasi_seimbang": m["akurasi_seimbang"],
        }
    return hasil


def satu_epoch(model, loader, opt, device) -> None:
    model.train()
    for x, va, has_va, _au, _aum, m in loader:
        keep = has_va > 0
        if keep.sum() < 2:
            continue
        out = model(x[keep].to(device), frame_mask=m[keep].to(device))["va"]
        rugi = rugi_va(out, va[keep].to(device))
        opt.zero_grad(set_to_none=True)
        rugi.backward()
        opt.step()


def jalankan(args) -> dict:
    import yaml
    from sklearn.model_selection import GroupKFold
    from torch.utils.data import DataLoader, Subset

    from data.loaders.affect_landmark_dataset import AffectLandmarkDataset
    from models.st_gcn_graph_tran.model import CHANNELS_LAMA, STRIDES_LAMA
    from models.st_gcn_graph_tran.mood_rules import MOOD_GAP_ID
    from models.st_gcn_graph_tran.train import build_model, uraian_arsitektur

    folder = pastikan_di_luar_repo(args.keluaran)
    with open(REPO / "models/st_gcn_graph_tran/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["head_type"] = "va"
    if getattr(args, "arsitektur", "b3") == "lama":
        cfg["model"].update(
            channels=list(CHANNELS_LAMA), strides=list(STRIDES_LAMA),
            use_transformer=True, block_dropout=0.0,
        )
    cfg["data"]["window_frames"] = args.T
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    ds = AffectLandmarkDataset(
        args.manifest,
        au_list=[],
        clip_len=args.T,
        return_frame_mask=True,
        max_filled=1.0,
    )
    grup = np.array([it["subject"] for _, it, _ in ds.items])
    # Label VA per jendela dibaca dari manifest, bukan dari klip, agar kelas mayoritas
    # lipatan latih didapat tanpa memuat ulang semua tensor.
    va_semua = np.array(
        [
            [it["valence"], it["arousal"]] if float(it.get("va_mask", 0)) > 0 else [np.nan, np.nan]
            for _, it, _ in ds.items
        ],
        dtype=np.float64,
    )
    hasil = []
    for k, (tr, te) in enumerate(
        GroupKFold(n_splits=args.lipatan).split(np.zeros(len(ds)), groups=grup)
    ):
        torch.manual_seed(args.seed + k)
        np.random.seed(args.seed + k)
        model = build_model(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        g = torch.Generator().manual_seed(args.seed + k)
        dl_tr = DataLoader(
            Subset(ds, tr),
            batch_size=args.batch,
            shuffle=True,
            generator=g,
            num_workers=0,
        )
        dl_te = DataLoader(
            Subset(ds, te), batch_size=args.batch, shuffle=False, num_workers=0
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(args.epoch):
            satu_epoch(model, dl_tr, opt, device)
        detik = time.perf_counter() - t0
        va_tr = va_semua[tr]
        lab_tr = label_mood(va_tr[~np.isnan(va_tr).any(axis=1)])
        lab_tr = lab_tr[lab_tr != MOOD_GAP_ID]
        mayoritas = int(np.bincount(lab_tr, minlength=5).argmax()) if lab_tr.size else None
        r = evaluasi(model, dl_te, device, mayoritas)
        r.update(
            {
                "lipatan": k,
                "aktor_uji": int(len(set(grup[te]))),
                "detik_latih": round(detik, 1),
                "vram_puncak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if device.type == "cuda"
                else None,
            }
        )
        torch.save(
            {"model": model.state_dict(), "T": args.T, "lipatan": k},
            folder / f"stgcn_va_T{args.T}_{getattr(args, 'arsitektur', 'b3')}_lipatan{k}.pt",
        )
        hasil.append(r)
        print(r)
    ringkas = {
        m: {
            "rerata": float(np.mean([h[m] for h in hasil])),
            "sd": float(np.std([h[m] for h in hasil], ddof=1)),
        }
        for m in ("ccc_valensi", "ccc_arousal", "akurasi_5_kelas", "f1_makro", "akurasi_seimbang")
        if len(hasil) > 1 and all(h[m] is not None for h in hasil)
    }
    for m in ("akurasi_5_kelas", "f1_makro", "akurasi_seimbang"):
        nilai = [h["mayoritas"][m] for h in hasil if h.get("mayoritas") and h["mayoritas"][m] is not None]
        if len(nilai) > 1:
            ringkas[f"mayoritas_{m}"] = {"rerata": float(np.mean(nilai)), "sd": float(np.std(nilai, ddof=1))}
    lap = {
        "lingkungan": info_lingkungan(),
        "T": args.T,
        "epoch": args.epoch,
        "seed": args.seed,
        "jumlah_jendela": len(ds),
        "jumlah_aktor": int(len(set(grup))),
        "arsitektur": uraian_arsitektur(cfg),
        "parameter": int(sum(p.numel() for p in build_model(cfg).parameters())),
        "per_lipatan": hasil,
        "ringkasan": ringkas,
    }
    tulis_json(folder / f"v2_T{args.T}_{getattr(args, 'arsitektur', 'b3')}_ringkasan.json", lap)
    return lap


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--keluaran", required=True)
    ap.add_argument("--lipatan", type=int, default=5)
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--arsitektur", choices=("b3", "lama"), default="b3",
        help="b3 = config.yaml (4 unit, tanpa Transformer); lama = 6 unit + Transformer",
    )
    jalankan(ap.parse_args())


if __name__ == "__main__":
    main()
