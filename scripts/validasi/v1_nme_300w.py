"""V1: NME 68 landmark pada 300-W (01_Indoor dan 02_Outdoor, 600 gambar dengan .pts).

Kandidat (semua memilih wajah yang kotaknya paling dekat dengan kotak anotasi .pts):
  * ``mediapipe_A`` dan ``mediapipe_B``: FaceMesh 468 titik lalu subset iBUG-68 varian A/B
    (``data/preprocessing/landmark_map.py``). Catatan: topologi FaceMesh bukan iBUG-68,
    sehingga titik rahang dan alis secara definisi tidak berimpit dengan anotasi 300-W.
  * ``openface``: CSV FaceLandmarkImg (satu CSV per gambar, nama sama dengan gambar).
  * ``dlib_hog``: detektor HOG dlib + shape_predictor_68 (``--dlib-model``).
  * ``yunet_dlib``: kotak wajah YuNet (OpenCV ``FaceDetectorYN``, ``--yunet-model``)
    diteruskan ke shape_predictor_68 dlib.
  * ``yunet_dlib_kalibrasi``: seperti ``yunet_dlib``, tetapi kotak YuNet dipetakan dulu ke
    gaya kotak HOG dlib dengan parameter dari ``v1_kalibrasi_yunet.py`` (``--kalibrasi``).
    Parameter dipasang pada bagian latih 300-W, bukan pada 600 gambar uji ini.

Metrik:
  * ``persen_terdeteksi``: ada prediksi wajah; ``persen_cocok``: kotak landmark prediksi
    beririsan dengan kotak anotasi (IoU >= 0,3), yaitu wajah yang benar.
  * NME rerata dan median hanya pada gambar yang cocok; NME = rerata jarak titik ke titik
    dibagi jarak sudut luar mata acuan (titik 36 dan 45).
  * ``auc_0_08`` dan ``persen_gagal`` atas SEMUA gambar (gagal = tidak cocok atau NME > 0,08).
  * NME per wilayah (rahang 0-16, alis 17-26, hidung 27-35, mata 36-47, mulut 48-67) pada
    gambar yang cocok.
  * Waktu per gambar di CPU untuk kandidat yang dijalankan skrip ini (OpenFace diukur
    oleh FaceLandmarkImg sendiri).

    python -m scripts.validasi.v1_nme_300w --root D:/300w --keluaran D:/moodin_runs/v1b ^
        --openface-dir D:/moodin_runs/v1/openface_csv ^
        --dlib-model D:/models/shape_predictor_68_face_landmarks.dat ^
        --yunet-model D:/models/face_detection_yunet_2023mar.onnx ^
        --kalibrasi D:/moodin_runs/v1_kalibrasi/kalibrasi_yunet_dlib.json
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from scripts.validasi._umum import info_lingkungan, pastikan_di_luar_repo, tulis_json

AMBANG = 0.08
IOU_MIN = 0.3
MATA_LUAR = (36, 45)
WILAYAH = {
    "rahang": range(0, 17),
    "alis": range(17, 27),
    "hidung": range(27, 36),
    "mata": range(36, 48),
    "mulut": range(48, 68),
}


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


def kotak(p: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(p[:, 0].min()),
        float(p[:, 1].min()),
        float(p[:, 0].max()),
        float(p[:, 1].max()),
    )


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    u = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / u if u > 0 else 0.0


def _pilih_terdekat(kandidat: list[np.ndarray], gt: np.ndarray):
    """Wajah yang pusat kotaknya paling dekat dengan pusat kotak anotasi."""
    if not kandidat:
        return None
    pusat = (gt.min(0) + gt.max(0)) / 2
    return min(
        kandidat, key=lambda p: np.linalg.norm((p.min(0) + p.max(0)) / 2 - pusat)
    )


def nilai_gambar(pred, gt) -> dict:
    """NME total dan per wilayah, serta status deteksi untuk satu gambar."""
    if pred is None:
        return {
            "terdeteksi": False,
            "cocok": False,
            "nme": float("nan"),
            **{f"nme_{w}": float("nan") for w in WILAYAH},
        }
    d = float(np.linalg.norm(gt[MATA_LUAR[0]] - gt[MATA_LUAR[1]]))
    err = np.linalg.norm(pred - gt, axis=1) / d
    out = {
        "terdeteksi": True,
        "cocok": iou(kotak(pred), kotak(gt)) >= IOU_MIN,
        "nme": float(err.mean()),
    }
    for w, idx in WILAYAH.items():
        out[f"nme_{w}"] = float(err[list(idx)].mean())
    return out


def ringkas(baris: list[dict], waktu_ms: list[float]) -> dict:
    n = len(baris)
    det = np.array([b["terdeteksi"] for b in baris])
    cocok = np.array([b["cocok"] for b in baris])
    e = np.array([b["nme"] for b in baris], dtype=np.float64)
    e_all = np.where(cocok, e, np.inf)
    xs = np.linspace(0, AMBANG, 801)
    ced = [(e_all <= x).mean() for x in xs]
    ok = e[cocok]
    out = {
        "n": n,
        "persen_terdeteksi": round(100.0 * det.mean(), 2),
        "persen_cocok": round(100.0 * cocok.mean(), 2),
        "nme_rerata_cocok": float(ok.mean()) if ok.size else None,
        "nme_median_cocok": float(np.median(ok)) if ok.size else None,
        "auc_0_08": float(np.trapz(ced, xs) / AMBANG),
        "persen_gagal": round(100.0 * (~(e_all <= AMBANG)).mean(), 2),
        "ms_per_gambar_cpu": round(float(np.mean(waktu_ms)), 1) if waktu_ms else None,
        "per_wilayah_cocok": {},
    }
    for w in WILAYAH:
        v = np.array([b[f"nme_{w}"] for b in baris], dtype=np.float64)[cocok]
        out["per_wilayah_cocok"][w] = {
            "rerata": float(v.mean()) if v.size else None,
            "median": float(np.median(v)) if v.size else None,
        }
    return out


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
        return {
            v: _pilih_terdekat([s[peta_ibug68(v)] for s in semua], gt)
            for v in ("A", "B")
        }


class KandidatDlib:
    """shape_predictor_68 dlib dengan detektor HOG dlib atau kotak YuNet."""

    def __init__(
        self, model_path: str, yunet_path: str | None = None, kalibrasi: dict | None = None
    ):
        import dlib

        self.kalibrasi = kalibrasi
        self.dlib = dlib
        self.det = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(model_path)
        self.yunet = None
        if yunet_path:
            import cv2

            self.yunet = cv2.FaceDetectorYN.create(
                yunet_path, "", (320, 320), 0.6, 0.3, 5000
            )

    def _bentuk(self, rgb, rect) -> np.ndarray:
        s = self.sp(rgb, rect)
        return np.array(
            [[s.part(i).x, s.part(i).y] for i in range(68)], dtype=np.float64
        )

    def hog(self, bgr, gt):
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return _pilih_terdekat([self._bentuk(rgb, r) for r in self.det(rgb, 1)], gt)

    def yunet_dlib(self, bgr, gt, kalibrasi: dict | None = None):
        import cv2

        h, w = bgr.shape[:2]
        self.yunet.setInputSize((w, h))
        _, faces = self.yunet.detect(bgr)
        if faces is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hasil = []
        for f in faces:
            x, y, fw, fh = (float(v) for v in f[:4])
            b = (x, y, x + fw, y + fh)
            if kalibrasi is not None:
                from scripts.validasi.v1_kalibrasi_yunet import terapkan

                b = terapkan(b, kalibrasi)
            kiri, atas, kanan, bawah = (int(round(v)) for v in b)
            hasil.append(self._bentuk(rgb, self.dlib.rectangle(kiri, atas, kanan, bawah)))
        return _pilih_terdekat(hasil, gt)

    def yunet_dlib_kalibrasi(self, bgr, gt):
        return self.yunet_dlib(bgr, gt, self.kalibrasi)


def baca_openface(folder: Path, img: Path, gt):
    from data.loaders.openface_csv import read_openface_image_csv

    p = folder / (img.stem + ".csv")
    if not p.exists():
        return None
    rec = read_openface_image_csv(str(p))
    return _pilih_terdekat(list(rec["landmarks"].astype(np.float64)), gt)


def _ukur(fungsi, *a):
    t = time.perf_counter()
    hasil = fungsi(*a)
    return hasil, 1000 * (time.perf_counter() - t)


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
        "--yunet-model",
        help="face_detection_yunet_2023mar.onnx (opsional, butuh --dlib-model)",
    )
    ap.add_argument(
        "--kalibrasi",
        help="kalibrasi_yunet_dlib.json dari v1_kalibrasi_yunet (opsional, butuh --yunet-model)",
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
            kal = None
            if args.kalibrasi:
                from scripts.validasi.v1_kalibrasi_yunet import baca_kalibrasi

                kal = baca_kalibrasi(args.kalibrasi)
            dl_c = KandidatDlib(args.dlib_model, args.yunet_model, kal)
        except (ImportError, RuntimeError, AttributeError) as e:
            print(f"dlib dilewati: {e}")
    hasil = {}
    waktu = {}
    baris_csv = []
    for img, pts in pasangan:
        gt = baca_pts(pts)
        bgr = cv2.imread(str(img))
        if bgr is None:
            raise SystemExit(f"Gambar tidak terbaca: {img}")
        prediksi = {}
        mp_out, ms = _ukur(mp_c, bgr, gt)
        prediksi["mediapipe_A"], prediksi["mediapipe_B"] = mp_out["A"], mp_out["B"]
        waktu.setdefault("mediapipe_A", []).append(ms)
        waktu.setdefault("mediapipe_B", []).append(ms)
        if args.openface_dir:
            prediksi["openface"] = baca_openface(Path(args.openface_dir), img, gt)
        if dl_c is not None:
            prediksi["dlib_hog"], ms = _ukur(dl_c.hog, bgr, gt)
            waktu.setdefault("dlib_hog", []).append(ms)
            if dl_c.yunet is not None:
                prediksi["yunet_dlib"], ms = _ukur(dl_c.yunet_dlib, bgr, gt)
                waktu.setdefault("yunet_dlib", []).append(ms)
                if dl_c.kalibrasi is not None:
                    prediksi["yunet_dlib_kalibrasi"], ms = _ukur(
                        dl_c.yunet_dlib_kalibrasi, bgr, gt
                    )
                    waktu.setdefault("yunet_dlib_kalibrasi", []).append(ms)
        del bgr
        for k, p in prediksi.items():
            r = nilai_gambar(p, gt)
            hasil.setdefault(k, []).append(r)
            baris_csv.append(
                {"gambar": img.name, "subset": img.parent.name, "kandidat": k, **r}
            )

    ringkasan = {k: ringkas(v, waktu.get(k, [])) for k, v in hasil.items()}
    lap = {
        "lingkungan": info_lingkungan(),
        "jumlah_gambar": len(pasangan),
        "ambang_gagal": AMBANG,
        "iou_min_cocok": IOU_MIN,
        "kandidat": ringkasan,
    }
    tulis_json(folder / "v1_ringkasan.json", lap)
    with open(folder / "v1_per_gambar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris_csv[0]))
        w.writeheader()
        w.writerows(baris_csv)
    for k, v in ringkasan.items():
        print(k, {x: y for x, y in v.items() if x != "per_wilayah_cocok"})


if __name__ == "__main__":
    main()
