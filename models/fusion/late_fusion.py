"""VA-bridge: fusi late berbasis aturan antara cabang wajah (ST-GCN) dan
cabang fisiologis (STC-rPPG).

Kontrak (sesuai desain proposal Bab III & docs/ipc-contract.md):
  - Valence akhir  = valence dari wajah (rPPG tak membawa informasi valence).
  - Arousal akhir  = kombinasi berbobot arousal-wajah + arousal-fisiologis.
  - Mood akhir     = SATU kali pemetaan kuadran va_to_mood(valence, arousal).

Modul ini SENGAJA bebas-torch (murni numpy/python) supaya:
  - ringan untuk dijalankan real-time di edge (Jetson Orin Nano Super),
  - mudah diuji tanpa GPU,
  - TIDAK bergantung pada kode STC-rPPG teman Anda (hanya memakai keluaran
    HR/HRV yang sudah menjadi kontrak antar-modul).

Menangani laju sampel asinkron: wajah (cepat, ~15-30 fps) vs fisiologis
(lambat, HR/HRV diperbarui tiap beberapa detik). Nilai fisiologis di-buffer
dan dianggap kedaluwarsa (stale) bila melewati physio_timeout_s.
"""
from __future__ import annotations

import math
import time

try:
    from models.st_gcn_graph_tran.labels import (
        va_to_mood_id, MOOD_CLASSES, NEUTRAL_RADIUS,
    )
except Exception:  # fallback mandiri
    MOOD_CLASSES = ["Ceria", "Cemas", "Tertekan", "Rileks", "Netral"]
    NEUTRAL_RADIUS = 0.2

    def va_to_mood_id(v, a, neutral_radius=NEUTRAL_RADIUS):
        if math.hypot(v, a) <= neutral_radius:
            return 4
        if v >= 0 and a >= 0:
            return 0
        if v < 0 and a >= 0:
            return 1
        if v < 0 and a < 0:
            return 2
        return 3


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def physio_to_arousal(hr_bpm, hrv_ms, hr_base=70.0, hr_span=30.0,
                      hrv_base=50.0, hrv_span=40.0, w_hr=0.6, w_hrv=0.4):
    """Peta HR/HRV -> proxy arousal di [-1, 1].

    Prinsip fisiologis: arousal naik -> HR naik & HRV (RMSSD) turun.
      - komponen HR : (hr - hr_base) / hr_span  (positif = arousal tinggi)
      - komponen HRV: (hrv_base - hrv) / hrv_span (HRV rendah = arousal tinggi)
    Dihaluskan dengan tanh agar robust terhadap outlier. hrv_ms boleh None.
    """
    hr_term = (float(hr_bpm) - hr_base) / max(hr_span, 1e-6)
    if hrv_ms is None:
        arousal = hr_term
    else:
        hrv_term = (hrv_base - float(hrv_ms)) / max(hrv_span, 1e-6)
        arousal = w_hr * hr_term + w_hrv * hrv_term
    return _clamp(math.tanh(arousal))


class LateFusionVABridge:
    """Menggabungkan keluaran wajah & fisiologis menjadi satu estimasi mood.

    Parameter
    ---------
    w_face_arousal, w_physio_arousal : bobot arousal (dinormalisasi otomatis).
    neutral_radius : radius kuadran netral.
    ema_alpha : faktor pemulusan EMA untuk valence/arousal akhir (0=off).
    physio_timeout_s : ambang kedaluwarsa data fisiologis.
    """

    def __init__(self, w_face_arousal=0.5, w_physio_arousal=0.5,
                 neutral_radius=NEUTRAL_RADIUS, ema_alpha=0.3,
                 physio_timeout_s=5.0):
        s = w_face_arousal + w_physio_arousal
        if s <= 0:
            raise ValueError("Total bobot arousal harus > 0")
        self.w_face = w_face_arousal / s
        self.w_physio = w_physio_arousal / s
        self.neutral_radius = float(neutral_radius)
        self.ema_alpha = float(ema_alpha)
        self.physio_timeout_s = float(physio_timeout_s)
        self._physio = None  # dict {arousal, hr_bpm, hrv_ms, ts}
        self._ema_v = None
        self._ema_a = None

    def update_physio(self, hr_bpm, hrv_ms=None, timestamp=None):
        """Simpan sampel fisiologis terbaru (dipanggil saat rPPG menghasilkan HR)."""
        ts = time.monotonic() if timestamp is None else float(timestamp)
        self._physio = {
            "arousal": physio_to_arousal(hr_bpm, hrv_ms),
            "hr_bpm": float(hr_bpm),
            "hrv_ms": None if hrv_ms is None else float(hrv_ms),
            "ts": ts,
        }

    def _physio_fresh(self, now):
        return (self._physio is not None
                and (now - self._physio["ts"]) <= self.physio_timeout_s)

    def _ema(self, prev, new):
        if self.ema_alpha <= 0 or prev is None:
            return new
        return self.ema_alpha * new + (1 - self.ema_alpha) * prev

    def fuse(self, face_valence, face_arousal, face_confidence=1.0,
             timestamp=None, hr_bpm=None, hrv_ms=None):
        """Hasilkan estimasi mood tergabung.

        Bila hr_bpm diberikan, buffer fisiologis diperbarui lebih dulu. Bila
        tidak, dipakai buffer terakhir selama belum kedaluwarsa.
        Return dict siap-pakai sesuai payload inference/result ipc-contract.
        """
        now = time.monotonic() if timestamp is None else float(timestamp)
        if hr_bpm is not None:
            self.update_physio(hr_bpm, hrv_ms, now)

        valence = _clamp(float(face_valence))
        fresh = self._physio_fresh(now)
        if fresh:
            arousal = self.w_face * float(face_arousal) + self.w_physio * self._physio["arousal"]
            physio_used = True
        else:
            arousal = float(face_arousal)  # fallback: wajah saja
            physio_used = False
        arousal = _clamp(arousal)

        # pemulusan temporal
        self._ema_v = self._ema(self._ema_v, valence)
        self._ema_a = self._ema(self._ema_a, arousal)
        valence, arousal = _clamp(self._ema_v), _clamp(self._ema_a)

        mood_id = va_to_mood_id(valence, arousal, self.neutral_radius)
        # confidence: gabungan keyakinan wajah & jarak dari batas netral
        dist = math.hypot(valence, arousal)
        margin = min(1.0, abs(dist - self.neutral_radius) / max(self.neutral_radius, 1e-6))
        confidence = _clamp(float(face_confidence) * (0.5 + 0.5 * margin), 0.0, 1.0)

        return {
            "mood_id": int(mood_id),
            "mood_label": MOOD_CLASSES[mood_id],
            "confidence": round(confidence, 4),
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "heart_rate_bpm": (round(self._physio["hr_bpm"], 1)
                               if (self._physio and physio_used) else None),
            "hrv_ms": (round(self._physio["hrv_ms"], 1)
                       if (self._physio and physio_used and self._physio["hrv_ms"] is not None)
                       else None),
            "physio_used": physio_used,
        }

    def reset(self):
        self._physio = None
        self._ema_v = None
        self._ema_a = None


if __name__ == "__main__":
    br = LateFusionVABridge(ema_alpha=0.0)
    br.update_physio(hr_bpm=95, hrv_ms=25)
    print(br.fuse(face_valence=-0.6, face_arousal=0.3, face_confidence=0.9))
    print(br.fuse(face_valence=0.7, face_arousal=0.4, hr_bpm=68, hrv_ms=60))
