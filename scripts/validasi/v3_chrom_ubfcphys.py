"""V3: CHROM pada UBFC-Phys (vid_sN_T1..T3) terhadap BVP kontak 64 Hz.

Video dibaca frame demi frame (streaming); dari setiap frame hanya rerata RGB ROI kulit
yang disimpan di memori, lalu frame dibuang. HR rPPG dan HR acuan dihitung dengan
estimator yang sama (``rPPGProcessor.estimate_bpm``) per jendela 60 detik tanpa
tumpang tindih. Keluaran: CSV per jendela dan ringkasan JSON di luar repo.

    python -m scripts.validasi.v3_chrom_ubfcphys --subjek-dir D:/s1/s1 --subjek s1 --keluaran D:/moodin_runs/v3

Laju frame diambil dari cap waktu video (CAP_PROP_POS_MSEC) bila tersedia; laju
nominal berkas (35,14 fps pada s1) dipakai bila cap waktu tidak ada.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from data.preprocessing.chrom import chrom, rerata_rgb_roi
from models.stc_rppg.postprocessing import rPPGProcessor
from scripts.validasi._umum import info_lingkungan, pastikan_di_luar_repo, tulis_json

FS_BVP = 64.0


def baca_bvp(path) -> np.ndarray:
    """CSV BVP UBFC-Phys: satu nilai per baris (64 Hz)."""
    v = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=1)
    return v.ravel()


def rgb_streaming(video: str, lebar_deteksi: int = 512):
    """Return (rgb (T, 3), fps). Landmark FaceMesh dari frame yang diperkecil."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Video tidak bisa dibuka: {video}")
    fps_nom = cap.get(cv2.CAP_PROP_FPS)
    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False
    )
    rgb, ts = [], []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            h, w = frame.shape[:2]
            s = lebar_deteksi / max(h, w)
            kecil = cv2.resize(
                frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA
            )
            res = fm.process(cv2.cvtColor(kecil, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                px = np.array(
                    [[p.x * kecil.shape[1], p.y * kecil.shape[0]] for p in lm]
                )
                rgb.append(rerata_rgb_roi(kecil, px))
            else:
                rgb.append(np.full(3, np.nan))
            del frame, kecil
    finally:
        cap.release()
        fm.close()
    ts = np.asarray(ts)
    dts = np.diff(ts)
    fps = (
        1.0 / np.median(dts)
        if dts.size and np.all(np.isfinite(dts)) and np.median(dts) > 0
        else fps_nom
    )
    return np.asarray(rgb), float(fps)


def hr_per_jendela(
    rppg: np.ndarray, fs_rppg: float, bvp: np.ndarray, jendela_s: float = 60.0
) -> list[dict]:
    pr = rPPGProcessor(fps=fs_rppg)
    pb = rPPGProcessor(fps=FS_BVP)
    durasi = min(len(rppg) / fs_rppg, len(bvp) / FS_BVP)
    out = []
    for k in range(int(durasi // jendela_s)):
        a, b = k * jendela_s, (k + 1) * jendela_s
        hr_r = pr.estimate_bpm(rppg[int(a * fs_rppg) : int(b * fs_rppg)])
        hr_b = pb.estimate_bpm(bvp[int(a * FS_BVP) : int(b * FS_BVP)])
        out.append(
            {
                "jendela": k,
                "mulai_s": a,
                "hr_rppg": hr_r,
                "hr_bvp": hr_b,
                "galat": hr_r - hr_b,
            }
        )
    return out


def ringkas(baris: list[dict]) -> dict:
    e = np.array([r["galat"] for r in baris], dtype=np.float64)
    return {
        "n_jendela": int(e.size),
        "mae_bpm": float(np.abs(e).mean()) if e.size else None,
        "rmse_bpm": float(np.sqrt((e**2).mean())) if e.size else None,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--subjek-dir", required=True)
    ap.add_argument("--subjek", default="s1")
    ap.add_argument("--keluaran", required=True)
    ap.add_argument("--tugas", nargs="+", default=["T1", "T2", "T3"])
    ap.add_argument("--jendela-s", type=float, default=60.0)
    args = ap.parse_args()
    folder = pastikan_di_luar_repo(args.keluaran)
    d = Path(args.subjek_dir)
    semua, lap = (
        [],
        {"lingkungan": info_lingkungan(), "subjek": args.subjek, "per_tugas": {}},
    )
    for tg in args.tugas:
        vid = next(iter(sorted(d.glob(f"vid_{args.subjek}_{tg}.*"))), None)
        bvp_p = d / f"bvp_{args.subjek}_{tg}.csv"
        if vid is None or not bvp_p.exists():
            raise SystemExit(f"Berkas {tg} tidak lengkap di {d}")
        t0 = time.perf_counter()
        rgb, fps = rgb_streaming(str(vid))
        s = chrom(rgb, fps)
        baris = hr_per_jendela(s, fps, baca_bvp(bvp_p), args.jendela_s)
        for r in baris:
            r.update({"subjek": args.subjek, "tugas": tg})
        semua += baris
        lap["per_tugas"][tg] = {
            **ringkas(baris),
            "frame": int(len(rgb)),
            "fps": round(fps, 3),
            "persen_frame_tanpa_wajah": round(
                100.0 * float(np.isnan(rgb).any(axis=1).mean()), 2
            ),
            "detik_proses": round(time.perf_counter() - t0, 1),
        }
        print(tg, lap["per_tugas"][tg])
    lap["gabungan"] = ringkas(semua)
    tulis_json(folder / f"v3_{args.subjek}_ringkasan.json", lap)
    with open(
        folder / f"v3_{args.subjek}_per_jendela.csv", "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(semua[0]))
        w.writeheader()
        w.writerows(semua)


if __name__ == "__main__":
    main()
