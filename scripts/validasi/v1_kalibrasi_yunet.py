"""Kalibrasi kotak YuNet ke gaya kotak detektor HOG dlib, dipasang pada bagian latih 300-W.

shape_predictor_68 dlib dilatih dengan kotak dari detektor HOG dlib. Kotak YuNet lebih
tinggi dan bergeser ke atas, sehingga prediktor menerima wajah di posisi yang tidak
pernah ia lihat saat dilatih; hasil V1b memperlihatkan galat rahang yunet_dlib jauh
lebih besar daripada dlib_hog. Skrip ini mencari empat parameter yang memetakan kotak
YuNet ke kotak HOG pada gambar yang sama:

    cx_dlib = cx_yunet + dx * w_yunet      w_dlib = sw * w_yunet
    cy_dlib = cy_yunet + dy * h_yunet      h_dlib = sh * h_yunet

Tiap parameter adalah median atas pasangan kotak, agar satu deteksi meleset tidak
menggeser hasil. Hanya pasangan yang kedua kotaknya beririsan dengan kotak anotasi
(IoU >= 0,3) yang dipakai.

Data: bagian latih 300-W (lfpw/trainset, helen/trainset, afw). Folder uji V1
(01_Indoor, 02_Outdoor) dan folder ``testset`` ditolak supaya parameter tidak pernah
melihat gambar uji.

    python -m scripts.validasi.v1_kalibrasi_yunet --latih-root D:/300w_latih ^
        --keluaran D:/moodin_runs/v1_kalibrasi ^
        --dlib-model D:/models/shape_predictor_68_face_landmarks.dat ^
        --yunet-model D:/models/face_detection_yunet_2023mar.onnx --maks 1000

Hasil ``kalibrasi_yunet_dlib.json`` lalu dipakai V1 lewat ``--kalibrasi``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from scripts.validasi._umum import info_lingkungan, pastikan_di_luar_repo, tulis_json
from scripts.validasi.v1_nme_300w import IOU_MIN, baca_pts, iou, kotak

FOLDER_UJI = {"01_indoor", "02_outdoor", "testset"}
KUNCI = ("dx", "dy", "sw", "sh")


def daftar_latih(root) -> list[tuple[Path, Path]]:
    """Pasangan (gambar, .pts) di bawah root, tanpa folder uji."""
    root = Path(root)
    out = []
    for pts in sorted(root.rglob("*.pts")):
        bagian = {p.lower() for p in pts.relative_to(root).parts[:-1]}
        if bagian & FOLDER_UJI:
            continue
        for ext in (".png", ".jpg", ".jpeg"):
            img = pts.with_suffix(ext)
            if img.exists():
                out.append((img, pts))
                break
    return out


def _ke_ltrb(x, y, w, h) -> tuple[float, float, float, float]:
    return (float(x), float(y), float(x + w), float(y + h))


def _pusat_ukuran(b) -> tuple[float, float, float, float]:
    kiri, atas, kanan, bawah = b
    return ((kiri + kanan) / 2, (atas + bawah) / 2, kanan - kiri, bawah - atas)


def kotak_cocok(kandidat: list, gt_kotak) -> tuple | None:
    """Kotak dengan IoU tertinggi terhadap kotak anotasi, bila IoU >= IOU_MIN."""
    terbaik, nilai = None, IOU_MIN
    for b in kandidat:
        v = iou(b, gt_kotak)
        if v >= nilai:
            terbaik, nilai = b, v
    return terbaik


def pasang(pasangan: list[tuple[tuple, tuple]]) -> dict:
    """Median parameter dari pasangan (kotak_yunet, kotak_dlib) dalam format (l, t, r, b)."""
    if len(pasangan) < 20:
        raise ValueError(
            f"Hanya {len(pasangan)} pasangan kotak; butuh minimal 20. "
            "Tambah gambar latih atau periksa detektor."
        )
    nilai = {k: [] for k in KUNCI}
    for yu, dl in pasangan:
        ycx, ycy, yw, yh = _pusat_ukuran(yu)
        dcx, dcy, dw, dh = _pusat_ukuran(dl)
        if yw <= 0 or yh <= 0:
            continue
        nilai["dx"].append((dcx - ycx) / yw)
        nilai["dy"].append((dcy - ycy) / yh)
        nilai["sw"].append(dw / yw)
        nilai["sh"].append(dh / yh)
    hasil = {"n_pasangan": len(nilai["dx"])}
    for k, v in nilai.items():
        a = np.asarray(v)
        q1, q3 = np.percentile(a, [25, 75])
        hasil[k] = float(np.median(a))
        hasil[f"{k}_iqr"] = [float(q1), float(q3)]
    return hasil


def terapkan(kotak_yunet, param: dict) -> tuple[float, float, float, float]:
    """Kotak YuNet (l, t, r, b) -> kotak bergaya HOG dlib (l, t, r, b)."""
    cx, cy, w, h = _pusat_ukuran(kotak_yunet)
    cx, cy = cx + param["dx"] * w, cy + param["dy"] * h
    w, h = param["sw"] * w, param["sh"] * h
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def baca_kalibrasi(path) -> dict:
    param = json.loads(Path(path).read_text(encoding="utf-8"))
    hilang = [k for k in KUNCI if k not in param]
    if hilang:
        raise ValueError(
            f"{path}: kunci {hilang} tidak ada. Jalankan ulang v1_kalibrasi_yunet."
        )
    return param


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--latih-root", required=True)
    ap.add_argument("--keluaran", required=True)
    ap.add_argument("--dlib-model", required=True)
    ap.add_argument("--yunet-model", required=True)
    ap.add_argument(
        "--maks", type=int, default=0, help="sampel acak gambar latih (0 = semua)"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cv2
    import dlib

    folder = pastikan_di_luar_repo(args.keluaran)
    daftar = daftar_latih(args.latih_root)
    if not daftar:
        raise SystemExit(
            f"Tidak ada pasangan gambar dan .pts di {args.latih_root} di luar folder uji."
        )
    if args.maks and args.maks < len(daftar):
        daftar = random.Random(args.seed).sample(daftar, args.maks)

    det = dlib.get_frontal_face_detector()
    yunet = cv2.FaceDetectorYN.create(args.yunet_model, "", (320, 320), 0.6, 0.3, 5000)
    pasangan, n_tanpa = [], 0
    for img, pts in daftar:
        bgr = cv2.imread(str(img))
        if bgr is None:
            raise SystemExit(f"Gambar tidak terbaca: {img}")
        gt = kotak(baca_pts(pts))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dl = [(r.left(), r.top(), r.right(), r.bottom()) for r in det(rgb, 1)]
        h, w = bgr.shape[:2]
        yunet.setInputSize((w, h))
        _, faces = yunet.detect(bgr)
        yu = [] if faces is None else [_ke_ltrb(*f[:4]) for f in faces]
        b_dl, b_yu = kotak_cocok(dl, gt), kotak_cocok(yu, gt)
        if b_dl is None or b_yu is None:
            n_tanpa += 1
            continue
        pasangan.append((b_yu, b_dl))
        del bgr, rgb

    param = pasang(pasangan)
    param.update(
        {
            "sumber": "300-W bagian latih (tanpa 01_Indoor, 02_Outdoor, testset)",
            "n_gambar": len(daftar),
            "n_tanpa_pasangan": n_tanpa,
            "seed": args.seed,
            "lingkungan": info_lingkungan(),
        }
    )
    tulis_json(folder / "kalibrasi_yunet_dlib.json", param)
    print({k: param[k] for k in ("n_pasangan", *KUNCI)})


if __name__ == "__main__":
    main()
