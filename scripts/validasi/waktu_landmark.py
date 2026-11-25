"""Uji waktu jalur landmark 68 titik pada video 640x480: laptop dan Jetson.

Tiga mode, semuanya memakai shape_predictor_68 dlib untuk titik:
  * ``hog``: detektor HOG dlib di setiap frame.
  * ``yunet_dlib``: YuNet di setiap frame; kotaknya dikalibrasi ke gaya HOG bila
    ``--kalibrasi`` diberikan.
  * ``lacak``: detektor (YuNet, atau HOG bila YuNet tidak diberikan) hanya tiap
    ``--tiap`` frame atau saat wajah hilang. Di antaranya kotak frame sebelumnya digeser
    mengikuti pusat landmark dan diskalakan dengan jarak sudut mata, lalu prediktor
    dijalankan pada kotak itu. Wajah dianggap hilang bila jarak sudut mata berubah lebih
    dari 30 persen antarframe.

Frame tidak pernah disimpan (ADR-14); hanya waktu per tahap yang ditulis ke JSON.

    python -m scripts.validasi.waktu_landmark --video D:/moodin_runs/contoh_640.mp4 ^
        --keluaran D:/moodin_runs/waktu_landmark/laptop ^
        --dlib-model D:/models/shape_predictor_68_face_landmarks.dat ^
        --yunet-model D:/models/face_detection_yunet_2023mar.onnx ^
        --kalibrasi D:/moodin_runs/v1_kalibrasi/kalibrasi_yunet_dlib.json

Di Jetson ganti path; ``--video 0`` membaca kamera langsung.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np

from scripts.validasi._umum import info_lingkungan, pastikan_di_luar_repo, tulis_json

LEBAR, TINGGI = 640, 480
MATA_LUAR = (36, 45)
BATAS_UBAH_SKALA = 0.30


def jarak_mata(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[MATA_LUAR[0]] - lm[MATA_LUAR[1]]))


def geser_kotak(kotak, lm_lama: np.ndarray, lm_baru: np.ndarray):
    """Kotak (l, t, r, b) ikut bergeser dengan pusat landmark dan berskala dengan jarak mata."""
    kiri, atas, kanan, bawah = kotak
    s = jarak_mata(lm_baru) / max(jarak_mata(lm_lama), 1e-6)
    d = lm_baru.mean(0) - lm_lama.mean(0)
    cx, cy = (kiri + kanan) / 2 + d[0], (atas + bawah) / 2 + d[1]
    w, h = (kanan - kiri) * s, (bawah - atas) * s
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class Pelacak:
    """Deteksi tiap ``tiap`` frame; di antaranya kotak dilacak dari landmark sebelumnya.

    ``deteksi(frame) -> kotak | None`` dan ``bentuk(frame, kotak) -> (68, 2)`` disuntikkan
    agar logika ini bisa diuji tanpa dlib.
    """

    def __init__(self, deteksi, bentuk, tiap: int = 10):
        if tiap < 1:
            raise ValueError("tiap harus >= 1")
        self.deteksi, self.bentuk, self.tiap = deteksi, bentuk, tiap
        self.kotak = None
        self.lm = None
        self.sejak = 0
        self.n_deteksi = 0

    def langkah(self, frame):
        """Kembalikan (landmark | None, dideteksi_ulang: bool)."""
        ulang = self.kotak is None or self.sejak >= self.tiap
        if ulang:
            self.kotak = self.deteksi(frame)
            self.n_deteksi += 1
            self.sejak = 0
            if self.kotak is None:
                self.lm = None
                return None, True
        lm = self.bentuk(frame, self.kotak)
        if self.lm is not None and not ulang:
            ubah = abs(jarak_mata(lm) / max(jarak_mata(self.lm), 1e-6) - 1.0)
            if ubah > BATAS_UBAH_SKALA:
                # Kotak sudah lepas dari wajah; paksa deteksi pada frame berikutnya.
                self.kotak, self.lm = None, None
                return None, False
            self.kotak = geser_kotak(self.kotak, self.lm, lm)
        self.lm = lm
        self.sejak += 1
        return lm, ulang


def ringkas_ms(ms: list[float]) -> dict:
    if not ms:
        return {"n": 0}
    a = np.asarray(ms, dtype=np.float64)
    return {
        "n": int(a.size),
        "rerata": round(float(a.mean()), 2),
        "median": round(float(np.median(a)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "fps_setara": round(1000.0 / float(a.mean()), 1) if a.mean() > 0 else None,
    }


def _perangkat() -> str:
    tegra = Path("/etc/nv_tegra_release")
    if tegra.exists():
        return (
            "jetson: "
            + tegra.read_text(encoding="utf-8", errors="replace").splitlines()[0][:80]
        )
    return platform.processor() or platform.machine()


class Jalur:
    def __init__(self, dlib_model, yunet_model=None, kalibrasi=None, hog_upsample=0):
        import cv2
        import dlib

        self.cv2, self.dlib = cv2, dlib
        self.hog = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(dlib_model)
        self.up = hog_upsample
        self.kalibrasi = kalibrasi
        self.yunet = None
        if yunet_model:
            self.yunet = cv2.FaceDetectorYN.create(
                yunet_model, "", (LEBAR, TINGGI), 0.6, 0.3, 5000
            )

    @staticmethod
    def _terbesar(kotak_list):
        if not kotak_list:
            return None
        return max(kotak_list, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    def kotak_hog(self, rgb):
        r = self.hog(rgb, self.up)
        return self._terbesar([(d.left(), d.top(), d.right(), d.bottom()) for d in r])

    def kotak_yunet(self, bgr):
        _, faces = self.yunet.detect(bgr)
        if faces is None:
            return None
        kotak = [(f[0], f[1], f[0] + f[2], f[1] + f[3]) for f in faces]
        b = self._terbesar(kotak)
        if self.kalibrasi is not None:
            from scripts.validasi.v1_kalibrasi_yunet import terapkan

            b = terapkan(b, self.kalibrasi)
        return b

    def bentuk(self, rgb, kotak):
        kiri, atas, kanan, bawah = (int(round(v)) for v in kotak)
        s = self.sp(rgb, self.dlib.rectangle(kiri, atas, kanan, bawah))
        return np.array(
            [[s.part(i).x, s.part(i).y] for i in range(68)], dtype=np.float64
        )


def _buka(sumber):
    import cv2

    cap = cv2.VideoCapture(int(sumber) if str(sumber).isdigit() else str(sumber))
    if not cap.isOpened():
        raise SystemExit(f"Video atau kamera {sumber} tidak bisa dibuka.")
    return cap


def jalankan_mode(mode: str, jalur: Jalur, sumber, n_frame: int, tiap: int) -> dict:
    cv2 = jalur.cv2
    cap = _buka(sumber)
    t_det, t_bentuk, t_total = [], [], []
    ada = 0
    pelacak = None
    if mode == "lacak":
        det = jalur.kotak_yunet if jalur.yunet is not None else None
        pelacak = Pelacak(lambda f: None, lambda f, k: None, tiap)
    try:
        for _ in range(n_frame):
            ok, bgr = cap.read()
            if not ok:
                break
            bgr = cv2.resize(bgr, (LEBAR, TINGGI))
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if mode == "lacak":
                waktu = {}

                def deteksi(_f):
                    a = time.perf_counter()
                    b = det(bgr) if det is not None else jalur.kotak_hog(rgb)
                    waktu["det"] = 1000 * (time.perf_counter() - a)
                    return b

                def bentuk(_f, k):
                    a = time.perf_counter()
                    lm = jalur.bentuk(rgb, k)
                    waktu["bentuk"] = 1000 * (time.perf_counter() - a)
                    return lm

                pelacak.deteksi, pelacak.bentuk = deteksi, bentuk
                lm, _ = pelacak.langkah(bgr)
                if "det" in waktu:
                    t_det.append(waktu["det"])
                if "bentuk" in waktu:
                    t_bentuk.append(waktu["bentuk"])
            else:
                a = time.perf_counter()
                k = jalur.kotak_hog(rgb) if mode == "hog" else jalur.kotak_yunet(bgr)
                t_det.append(1000 * (time.perf_counter() - a))
                lm = None
                if k is not None:
                    a = time.perf_counter()
                    lm = jalur.bentuk(rgb, k)
                    t_bentuk.append(1000 * (time.perf_counter() - a))
            t_total.append(1000 * (time.perf_counter() - t0))
            ada += lm is not None
            del bgr, rgb
    finally:
        cap.release()
    n = len(t_total)
    hasil = {
        "frame": n,
        "persen_wajah": round(100.0 * ada / n, 1) if n else None,
        "deteksi_ms": ringkas_ms(t_det),
        "prediktor_ms": ringkas_ms(t_bentuk),
        "total_ms": ringkas_ms(t_total),
    }
    if pelacak is not None:
        hasil["n_deteksi"] = pelacak.n_deteksi
        hasil["tiap"] = tiap
    return hasil


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--video", required=True, help="berkas video atau indeks kamera")
    ap.add_argument("--keluaran", required=True)
    ap.add_argument("--dlib-model", required=True)
    ap.add_argument("--yunet-model")
    ap.add_argument("--kalibrasi")
    ap.add_argument("--frame", type=int, default=300)
    ap.add_argument(
        "--tiap", type=int, default=10, help="mode lacak: deteksi tiap N frame"
    )
    ap.add_argument("--hog-upsample", type=int, default=0)
    ap.add_argument("--mode", nargs="+", default=["hog", "yunet_dlib", "lacak"])
    args = ap.parse_args()

    folder = pastikan_di_luar_repo(args.keluaran)
    kal = None
    if args.kalibrasi:
        from scripts.validasi.v1_kalibrasi_yunet import baca_kalibrasi

        kal = baca_kalibrasi(args.kalibrasi)
    jalur = Jalur(args.dlib_model, args.yunet_model, kal, args.hog_upsample)
    if "yunet_dlib" in args.mode and jalur.yunet is None:
        raise SystemExit("Mode yunet_dlib butuh --yunet-model.")
    lap = {
        "lingkungan": info_lingkungan(),
        "perangkat": _perangkat(),
        "resolusi": [LEBAR, TINGGI],
        "hog_upsample": args.hog_upsample,
        "kalibrasi": bool(kal),
        "mode": {},
    }
    for m in args.mode:
        if m not in ("hog", "yunet_dlib", "lacak"):
            raise SystemExit(
                f"Mode {m!r} tidak dikenal. Pilihan: hog, yunet_dlib, lacak."
            )
        lap["mode"][m] = jalankan_mode(m, jalur, args.video, args.frame, args.tiap)
        r = lap["mode"][m]
        print(m, r["persen_wajah"], r["total_ms"])
    tulis_json(folder / "waktu_landmark.json", lap)


if __name__ == "__main__":
    main()
