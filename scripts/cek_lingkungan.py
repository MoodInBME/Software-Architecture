"""Cek lingkungan Python MoodIn tanpa membuka kamera dan tanpa membaca data.

Mencetak versi Python, torch (build CUDA atau CPU), CUDA, nama GPU, numpy, OpenCV,
mediapipe, onnx, onnxruntime, dan backend kamera OpenCV yang tersedia. Baris
``MASALAH:`` menandai jebakan yang dikenal; kode keluar 1 bila ada masalah.

    python scripts/cek_lingkungan.py
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import platform
import sys

PAKET_OPENCV = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)


def _versi_dist(nama: str) -> str | None:
    try:
        return md.version(nama)
    except md.PackageNotFoundError:
        return None


def _impor(nama: str):
    try:
        return importlib.import_module(nama), None
    except ImportError as e:
        return None, str(e)


def periksa() -> list[str]:
    masalah: list[str] = []
    print(f"Python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform    : {platform.platform()}")
    if sys.version_info[:2] != (3, 12):
        masalah.append("Python bukan 3.12 (target repo, ISU-061).")

    torch, err = _impor("torch")
    if torch is None:
        print(f"torch       : tidak terpasang ({err})")
        masalah.append("torch belum terpasang.")
    else:
        cuda_build = torch.version.cuda
        print(f"torch       : {torch.__version__} (build CUDA: {cuda_build})")
        if cuda_build is None:
            masalah.append("torch build CPU; pasang ulang dari indeks cu124.")
        elif torch.cuda.is_available():
            print(f"CUDA        : tersedia, {torch.cuda.device_count()} GPU")
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(
                    f"GPU {i}       : {p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}"
                )
            print(f"cuDNN       : {torch.backends.cudnn.version()}")
            print(f"bfloat16    : {torch.cuda.is_bf16_supported()}")
        else:
            masalah.append("torch build CUDA tetapi GPU tidak terlihat (driver?).")
        tv = _versi_dist("torchvision")
        print(f"torchvision : {tv}")

    np_mod, err = _impor("numpy")
    print(f"numpy       : {np_mod.__version__ if np_mod else 'tidak terpasang'}")
    if np_mod is not None and not np_mod.__version__.startswith("1.26."):
        masalah.append(
            f"numpy {np_mod.__version__}; repo mengunci 1.26.4 (mediapipe 0.10.14)."
        )

    ocv = {n: v for n in PAKET_OPENCV if (v := _versi_dist(n))}
    print(f"paket OpenCV: {ocv or 'tidak ada'}")
    if len(ocv) > 1:
        masalah.append(f"Lebih dari satu paket OpenCV terpasang: {sorted(ocv)}.")
    cv2, err = _impor("cv2")
    if cv2 is None:
        print(f"cv2         : gagal diimpor ({err})")
        masalah.append("cv2 tidak bisa diimpor.")
    else:
        print(f"cv2         : {cv2.__version__} ({cv2.__file__})")
        vr = getattr(cv2, "videoio_registry", None)
        if vr is not None:
            kamera = [vr.getBackendName(b) for b in vr.getCameraBackends()]
            print(f"backend kamera OpenCV: {kamera}")
            if sys.platform == "win32" and "DSHOW" not in kamera:
                masalah.append(
                    "Backend DSHOW tidak tersedia; demo webcam memakai cv2.CAP_DSHOW."
                )
        info = cv2.getBuildInformation()
        gst = [ln.strip() for ln in info.splitlines() if "GStreamer" in ln]
        print(f"GStreamer   : {gst[0] if gst else 'tidak disebut'}")

    mp_mod, err = _impor("mediapipe")
    if mp_mod is None:
        print(f"mediapipe   : tidak terpasang ({err})")
        masalah.append("mediapipe belum terpasang.")
    else:
        ada_solutions = hasattr(mp_mod, "solutions")
        print(f"mediapipe   : {mp_mod.__version__} (solutions: {ada_solutions})")
        if not ada_solutions:
            masalah.append("mediapipe tanpa mp.solutions (versi 1.x); pakai 0.10.14.")

    for nama in ("onnx", "onnxruntime"):
        m, err = _impor(nama)
        print(f"{nama:<12}: {m.__version__ if m else 'tidak terpasang'}")
        if nama == "onnxruntime" and m is not None:
            print(f"ORT provider: {m.get_available_providers()}")
    return masalah


def main() -> int:
    masalah = periksa()
    print()
    if masalah:
        for m in masalah:
            print("MASALAH: " + m)
        return 1
    print("OK: tidak ada masalah yang dikenal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
