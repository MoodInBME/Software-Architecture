"""V1: NME 68 landmark pada 300-W (01_Indoor dan 02_Outdoor, 600 gambar dengan .pts).

Kandidat:
  * ``mediapipe_A`` dan ``mediapipe_B``: FaceMesh 468 titik lalu subset iBUG-68 varian A/B
    (``data/preprocessing/landmark_map.py``);
  * ``openface``: CSV FaceLandmarkImg (satu CSV per gambar, nama sama dengan gambar);
  * ``dlib``: detektor HOG + shape_predictor_68 (bila paket dlib dan berkas model ada).

NME = rerata jarak titik ke titik dibagi jarak sudut luar mata acuan (titik 36 dan 45).
Gagal = tidak ada deteksi atau NME > 0,08. AUC dihitung sampai 0,08 dari kurva CED
(gagal dihitung tak hingga). Waktu per gambar diukur di CPU untuk kandidat yang
dijalankan di skrip ini (OpenFace diukur terpisah oleh FaceLandmarkImg).

Bila gambar memuat beberapa wajah, deteksi yang dipakai adalah yang pusat kotaknya
paling dekat dengan pusat kotak landmark acuan (praktik umum pada 300-W).

    python -m scripts.validasi.v1_nme_300w --root D:/300w --keluaran D:/moodin_runs/v1
    python -m scripts.validasi.v1_nme_300w --root D:/300w --keluaran D:/moodin_runs/v1 --openface-dir D:/moodin_runs/v1/openface_csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from evaluation.affect_metrics import nme
from scripts.validasi._umum import info_lingkungan, pastikan_di_luar_repo, tulis_json

AMBANG = 0.08


def baca_pts(path) -> np.ndarray:
    """Berkas .pts 300-W -> (68, 2) dalam piksel (indeks 1-based di berkas diubah ke 0-based)."""
    baris = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        i0 = baris.index("{") + 1
        i1 = baris.index("}")
    except ValueError:
        raise ValueError(f"{path}: format .pts tanpa kurung kurawal") from None
    pts = np.array(
        [[float(v) for v in b.split()[:2]] for b in baris[i0:i1]], dtype=np.float64
    )
    if pts.shape != (68, 2):
        raise ValueError(f"{path}: {pts.shape} titik, diharapkan (68, 2)")
    return pts - 1.0


def daftar_gambar(root) -> list[tuple[Path, Path]]:
    out = []
    for sub in ("01_Indoor", "02_Outdoor"):
        for pts in sorted(Path(root).rglob(f"{sub}/*.pts")):
            for ext in (".png", ".jpg", ".jpeg"):
                img = pts.with_suffix(ext)
                if img.exists():
                    out.append((img, pts))
                    break
    return out


def ringkas(nilai: list[float], waktu_ms: list[float]) -> dict:
    e = np.asarray(nilai, dtype=np.float64)
    gagal = ~np.isfinite(e) | (e > AMBANG)
    xs = np.linspace(0, AMBANG, 801)
    ced = [(np.nan_to_num(e, nan=np.inf) <= x).mean() for x in xs]
    ok = e[np.isfinite(e)]
    return {
        "n": int(e.size),
        "nme_rerata": float(ok.mean()) if ok.size else None,
        "nme_median": float(np.median(ok)) if ok.size else None,
        "auc_0_08": float(np.trapz(ced, xs) / AMBANG),
        "persen_gagal": round(100.0 * gagal.mean(), 2),
        "tanpa_deteksi": int((~np.isfinite(e)).sum()),
        "ms_per_gambar_cpu": round(float(np.mean(waktu_ms)), 1) if waktu_ms else None,
    }


def _pilih_terdekat(kandidat: list[np.ndarray], gt: np.ndarray):
    if not kandidat:
        return None
    pusat = (gt.min(0) + gt.max(0)) / 2
    return min(
        kandidat, key=lambda p: np.linalg.norm((p.min(0) + p.max(0)) / 2 - pusat)
    )


class KandidatMediaPipe:
    def __init__(self):
        import mediapipe as mp

        self.fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=5, refine_landmarks=False
        )

    def __call__(self, bgr, gt):
        import cv2

        from data.preprocessing.landmark_map import peta_ibug68

        h, w = bgr.shape[:2]
        res = self.fm.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return {"A": None, "B": None}
        semua = [
            np.array([[p.x * w, p.y * h] for p in f.landmark])
            for f in res.multi_face_landmarks
        ]
        out = {}
        for v in ("A", "B"):
            idx = peta_ibug68(v)
            out[v] = _pilih_terdekat([s[idx] for s in semua], gt)
        return out


class KandidatDlib:
    def __init__(self, model_path: str):
        import dlib

        self.det = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(model_path)

    def __call__(self, bgr, gt):
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hasil = []
        for r in self.det(rgb, 1):
            s = self.sp(rgb, r)
            hasil.append(
                np.array(
                    [[s.part(i).x, s.part(i).y] for i in range(68)], dtype=np.float64
                )
            )
        return _pilih_terdekat(hasil, gt)


def baca_openface(folder: Path, img: Path, gt):
    from data.loaders.openface_csv import read_openface_image_csv

    p = folder / (img.stem + ".csv")
    if not p.exists():
        return None
    rec = read_openface_image_csv(str(p))
    return _pilih_terdekat(list(rec["landmarks"].astype(np.float64)), gt)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True)
    ap.add_argument("--keluaran", required=True)
    ap.add_argument("--openface-dir")
    ap.add_argument(
        "--dlib-model", help="shape_predictor_68_face_landmarks.dat (opsional)"
    )
    ap.add_argument(
        "--maks", type=int, default=0, help="batasi jumlah gambar (0 = semua)"
    )
    args = ap.parse_args()

    import cv2

    folder = pastikan_di_luar_repo(args.keluaran)
    pasangan = daftar_gambar(args.root)
    if args.maks:
        pasangan = pasangan[: args.maks]
    if not pasangan:
        raise SystemExit(
            f"Tidak ada pasangan gambar dan .pts di {args.root}/**/01_Indoor atau 02_Outdoor"
        )

    mp_c = KandidatMediaPipe()
    dl_c = None
    if args.dlib_model:
        try:
            dl_c = KandidatDlib(args.dlib_model)
        except (ImportError, RuntimeError) as e:
            print(f"dlib dilewati: {e}")
    nilai = {k: [] for k in ("mediapipe_A", "mediapipe_B", "openface", "dlib")}
    waktu = {k: [] for k in nilai}
    baris = []
    for img, pts in pasangan:
        gt = baca_pts(pts)
        bgr = cv2.imread(str(img))
        if bgr is None:
            raise SystemExit(f"Gambar tidak terbaca: {img}")
        t = time.perf_counter()
        mp_out = mp_c(bgr, gt)
        dt = 1000 * (time.perf_counter() - t)
        prediksi = {"mediapipe_A": mp_out["A"], "mediapipe_B": mp_out["B"]}
        waktu["mediapipe_A"].append(dt)
        waktu["mediapipe_B"].append(dt)
        if args.openface_dir:
            prediksi["openface"] = baca_openface(Path(args.openface_dir), img, gt)
        if dl_c is not None:
            t = time.perf_counter()
            prediksi["dlib"] = dl_c(bgr, gt)
            waktu["dlib"].append(1000 * (time.perf_counter() - t))
        del bgr
        rec = {"gambar": img.name, "subset": img.parent.name}
        for k, p in prediksi.items():
            e = float(nme(p[None], gt[None])[0]) if p is not None else float("nan")
            nilai[k].append(e)
            rec[k] = e
        baris.append(rec)

    hasil = {k: ringkas(v, waktu[k]) for k, v in nilai.items() if v}
    lap = {
        "lingkungan": info_lingkungan(),
        "jumlah_gambar": len(pasangan),
        "ambang_gagal": AMBANG,
        "kandidat": hasil,
    }
    tulis_json(folder / "v1_ringkasan.json", lap)
    with open(folder / "v1_per_gambar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in baris for k in r}))
        w.writeheader()
        w.writerows(baris)
    for k, v in hasil.items():
        print(k, v)


if __name__ == "__main__":
    main()
