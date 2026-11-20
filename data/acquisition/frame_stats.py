"""Statistik akuisisi kamera tanpa menyimpan citra: selang antar-frame dan kecerahan ROI.

Dipakai alat uji banding kamera (scripts/uji_banding_kamera.py). Hanya angka yang
disimpan; frame dibuang setelah dihitung (ADR-14).

Kecerahan memakai luma Y BT.601 (0,299 R + 0,587 G + 0,114 B) pada frame BGR 8-bit.
Nilai ini indikator relatif antar-sesi dengan kamera dan setelan yang sama, bukan
ukuran iluminasi fisik.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

import numpy as np

LUMA_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float64)


def roi_luma(frame_bgr: np.ndarray, roi: tuple[float, float, float, float]) -> float:
    """Rerata luma Y pada ROI (x, y, w, h) dalam pecahan lebar/tinggi frame [0, 1].

    frame_bgr: (H, W, 3) uint8. Galat bila ROI kosong setelah dipotong ke frame.
    """
    f = np.asarray(frame_bgr)
    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError(f"Frame harus (H, W, 3), dapat {f.shape}")
    h, w = f.shape[:2]
    x, y, rw, rh = roi
    x0, y0 = max(0, int(round(x * w))), max(0, int(round(y * h)))
    x1, y1 = min(w, int(round((x + rw) * w))), min(h, int(round((y + rh) * h)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {roi} kosong untuk frame {w}x{h}")
    patch = f[y0:y1, x0:x1].astype(np.float64)
    return float((patch @ LUMA_BGR).mean())


@dataclass
class FrameStats:
    """Akumulator per sesi: cap waktu (detik) dan luma ROI per frame.

    expected_fps menentukan ambang frame hilang: selang > 1,5 periode dihitung
    sebagai round(selang / periode) - 1 frame hilang.
    """

    expected_fps: float
    window_s: float = 10.0
    t: list = field(default_factory=list)
    luma: list = field(default_factory=list)

    def add(self, timestamp_s: float, luma_value: float) -> None:
        if self.t and timestamp_s <= self.t[-1]:
            raise ValueError("Cap waktu harus naik monoton")
        self.t.append(float(timestamp_s))
        self.luma.append(float(luma_value))

    def intervals(self) -> np.ndarray:
        return np.diff(np.asarray(self.t, dtype=np.float64))

    def dropped_frames(self) -> int:
        period = 1.0 / self.expected_fps
        dt = self.intervals()
        big = dt[dt > 1.5 * period]
        return int(np.sum(np.rint(big / period) - 1))

    def summary(self) -> dict:
        dt = self.intervals()
        n = len(self.t)
        dur = self.t[-1] - self.t[0] if n > 1 else 0.0
        return {
            "frames": n,
            "durasi_s": round(dur, 3),
            "fps_terukur": round((n - 1) / dur, 3) if dur > 0 else math.nan,
            "selang_rerata_ms": round(1000 * float(dt.mean()), 3)
            if dt.size
            else math.nan,
            "selang_sd_ms": round(1000 * float(dt.std(ddof=1)), 3)
            if dt.size > 1
            else math.nan,
            "frame_hilang": self.dropped_frames(),
            "luma_median": round(float(np.median(self.luma)), 3) if n else math.nan,
            "luma_iqr": (
                round(float(np.subtract(*np.percentile(self.luma, [75, 25]))), 3)
                if n
                else math.nan
            ),
        }

    def windows(self) -> list[dict]:
        """Rerata luma dan jumlah frame per jendela window_s sejak frame pertama."""
        if not self.t:
            return []
        t = np.asarray(self.t) - self.t[0]
        y = np.asarray(self.luma)
        k = np.floor(t / self.window_s).astype(int)
        out = []
        for j in range(int(k.max()) + 1):
            m = k == j
            out.append(
                {
                    "jendela": j,
                    "mulai_s": j * self.window_s,
                    "frames": int(m.sum()),
                    "luma_rerata": round(float(y[m].mean()), 3)
                    if m.any()
                    else math.nan,
                }
            )
        return out

    def write_csv(
        self, path_ringkasan: str, path_jendela: str, meta: dict | None = None
    ) -> None:
        """Tulis dua CSV: satu baris ringkasan sesi dan satu baris per jendela."""
        meta = dict(meta or {})
        row = {**meta, **self.summary()}
        with open(path_ringkasan, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
        rows = [{**meta, **r} for r in self.windows()]
        fields = list(meta) + ["jendela", "mulai_s", "frames", "luma_rerata"]
        with open(path_jendela, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
