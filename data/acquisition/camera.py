"""Pemilihan backend kamera: ``csi`` (nvarguscamerasrc, bawaan) atau ``uvc`` (V4L2).

Backend ``uvc`` disiapkan untuk usulan Logitech C920 sebagai kamera utama (status
usulan, menunggu keputusan bimbingan). Setelan bawaan sama dengan UBFC-rPPG:
YUYV (tanpa kompresi) 640 x 480 pada 30 fps lewat OpenCV ``CAP_V4L2``.

Kontrol kamera (eksposur, white balance, fokus, frekuensi jala-jala) dikunci di luar
program dengan ``scripts/kunci_kontrol_uvc.sh`` sebelum program dijalankan.
"""

from __future__ import annotations

from dataclasses import dataclass

CAMERA_BACKENDS = ("csi", "uvc")


@dataclass
class UvcSetting:
    device: str = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = "YUYV"

    @classmethod
    def from_config(cls, cfg: dict) -> "UvcSetting":
        return cls(
            device=str(cfg.get("uvc_device", cls.device)),
            width=int(cfg.get("uvc_width", cls.width)),
            height=int(cfg.get("uvc_height", cls.height)),
            fps=int(cfg.get("uvc_fps", cls.fps)),
            fourcc=str(cfg.get("uvc_fourcc", cls.fourcc)),
        )


def camera_backend(cfg: dict) -> str:
    backend = str(cfg.get("camera_backend", "csi")).lower()
    if backend not in CAMERA_BACKENDS:
        raise ValueError(
            f"camera_backend harus salah satu dari {CAMERA_BACKENDS}, bukan {backend!r}"
        )
    return backend


def _device_arg(device: str):
    """'/dev/video2' atau '2' -> indeks int bila berupa angka, selain itu path apa adanya."""
    d = str(device)
    if d.isdigit():
        return int(d)
    return d


def open_uvc_capture(setting: UvcSetting, cv2_module=None):
    """Buka kamera UVC lewat V4L2 dan minta FOURCC, ukuran, dan fps yang diminta.

    Return (cap, laporan). ``laporan`` berisi nilai yang benar-benar dipakai driver
    menurut OpenCV; pemanggil wajib memeriksa ``cocok`` karena driver boleh menolak
    permintaan tanpa galat.
    """
    if cv2_module is None:
        import cv2 as cv2_module
    cv2 = cv2_module
    if len(setting.fourcc) != 4:
        raise ValueError(f"FOURCC harus 4 karakter, dapat {setting.fourcc!r}")
    cap = cv2.VideoCapture(_device_arg(setting.device), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*setting.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, setting.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, setting.height)
    cap.set(cv2.CAP_PROP_FPS, setting.fps)
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    laporan = {
        "fourcc": fourcc,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    laporan["cocok"] = (
        laporan["fourcc"] == setting.fourcc
        and laporan["width"] == setting.width
        and laporan["height"] == setting.height
        and abs(laporan["fps"] - setting.fps) < 0.5
    )
    return cap, laporan
