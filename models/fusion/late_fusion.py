"""Penggabung keluaran runtime cermin: mood dari jalur wajah, data fisiologis diteruskan.

Kontrak:
  - Valence dan arousal = keluaran jalur wajah (ST-GCN), dihaluskan EMA.
  - Mood = pemetaan valence-arousal menurut aturan rancangan Pra-TA (``mood_rules``,
    bawaan ``tabel33``) dengan histeresis tampilan.
  - HR dan HRV dari jalur fisiologis TIDAK ikut menentukan mood (dua jalur paralel,
    ISU-077). Nilainya hanya diteruskan ke payload selama belum kedaluwarsa.
    Pemakaian HRV sebagai fitur pendukung arousal adalah eksperimen terpisah (ISU-078).
  - ``decision_margin`` adalah heuristik jarak ke batas kategori, bukan peluang
    terkalibrasi, sehingga tidak boleh ditampilkan sebagai persen.

Modul ini bebas torch (murni Python) agar ringan di Jetson dan mudah diuji.
"""

from __future__ import annotations

import logging
import math
import time

from models.st_gcn_graph_tran.labels import MOOD_CLASSES, MOOD_TO_ID, va_to_mood_id
from models.st_gcn_graph_tran.labels_tabel33 import BAND
from models.st_gcn_graph_tran.mood_rules import (
    MOOD_RULES,
    MoodDisplayHysteresis,
    decision_margin,
    is_rule_gap,
)

log = logging.getLogger(__name__)

NEUTRAL_RADIUS_GEMASTIK = 0.2


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def _margin_lingkaran(v, a, face_conf, r=NEUTRAL_RADIUS_GEMASTIK):
    """Skor lama era GEMASTIK (jarak radial ke lingkaran Netral), hanya untuk arsip."""
    margin = min(1.0, abs(math.hypot(v, a) - r) / r)
    return _clamp(face_conf * (0.5 + 0.5 * margin), 0.0, 1.0)


class LateFusionVABridge:
    """Menghasilkan satu estimasi mood dari jalur wajah, ditambah data fisiologis terakhir.

    Parameter
    ---------
    mood_rule : "tabel33" (bawaan) atau "lingkaran" (arsip GEMASTIK).
    margin_scale : skala jarak untuk decision_margin (bawaan 0,25).
    display_hysteresis : lebar histeresis tampilan; 0 mematikan histeresis.
    ema_alpha : faktor pemulusan EMA untuk valence/arousal (0 = mati).
    physio_timeout_s : ambang kedaluwarsa data fisiologis yang diteruskan.
    """

    def __init__(
        self,
        mood_rule="tabel33",
        margin_scale=BAND,
        display_hysteresis=0.05,
        ema_alpha=0.3,
        physio_timeout_s=5.0,
    ):
        if mood_rule not in MOOD_RULES:
            raise ValueError(
                f"mood_rule harus salah satu dari {MOOD_RULES}, bukan {mood_rule!r}"
            )
        if margin_scale <= 0:
            raise ValueError("margin_scale harus positif")
        self.mood_rule = mood_rule
        self.margin_scale = float(margin_scale)
        self.display = MoodDisplayHysteresis(display_hysteresis)
        self.ema_alpha = float(ema_alpha)
        self.physio_timeout_s = float(physio_timeout_s)
        self._physio = None  # dict {hr_bpm, hrv_ms, ts}
        self._ema_v = None
        self._ema_a = None

    def update_physio(self, hr_bpm, hrv_ms=None, timestamp=None):
        """Simpan sampel fisiologis terbaru untuk diteruskan ke payload."""
        ts = time.monotonic() if timestamp is None else float(timestamp)
        self._physio = {
            "hr_bpm": float(hr_bpm),
            "hrv_ms": None if hrv_ms is None else float(hrv_ms),
            "ts": ts,
        }

    def _physio_fresh(self, now):
        return (
            self._physio is not None
            and (now - self._physio["ts"]) <= self.physio_timeout_s
        )

    def _ema(self, prev, new):
        if self.ema_alpha <= 0 or prev is None:
            return new
        return self.ema_alpha * new + (1 - self.ema_alpha) * prev

    def fuse(
        self,
        face_valence,
        face_arousal,
        face_confidence=1.0,
        timestamp=None,
        hr_bpm=None,
        hrv_ms=None,
    ):
        """Hasilkan estimasi mood. HR/HRV hanya diteruskan, tidak memengaruhi mood."""
        now = time.monotonic() if timestamp is None else float(timestamp)
        if hr_bpm is not None:
            self.update_physio(hr_bpm, hrv_ms, now)
        fresh = self._physio_fresh(now)

        self._ema_v = self._ema(self._ema_v, _clamp(float(face_valence)))
        self._ema_a = self._ema(self._ema_a, _clamp(float(face_arousal)))
        valence, arousal = _clamp(self._ema_v), _clamp(self._ema_a)

        gap = False
        if self.mood_rule == "tabel33":
            label = self.display.update(valence, arousal)
            gap = is_rule_gap(valence, arousal)
            if gap:
                log.info(
                    "mood_gap: v tepat 0 di luar pita Netral (a=%.3f), tampil %s",
                    arousal,
                    label,
                )
            mood_id = MOOD_TO_ID[label]
            margin = decision_margin(
                valence, arousal, float(face_confidence), self.margin_scale
            )
        else:
            mood_id = va_to_mood_id(valence, arousal, NEUTRAL_RADIUS_GEMASTIK)
            margin = _margin_lingkaran(valence, arousal, float(face_confidence))

        return {
            "mood_id": int(mood_id),
            "mood_label": MOOD_CLASSES[mood_id],
            "decision_margin": round(margin, 4),
            "mood_gap": gap,
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "heart_rate_bpm": round(self._physio["hr_bpm"], 1) if fresh else None,
            "hrv_ms": (
                round(self._physio["hrv_ms"], 1)
                if fresh and self._physio["hrv_ms"] is not None
                else None
            ),
            "physio_used": fresh,
        }

    def reset(self):
        self._physio = None
        self._ema_v = None
        self._ema_a = None
        self.display.reset()


if __name__ == "__main__":
    br = LateFusionVABridge(ema_alpha=0.0)
    br.update_physio(hr_bpm=95, hrv_ms=25)
    print(br.fuse(face_valence=-0.6, face_arousal=0.3, face_confidence=0.9))
    print(br.fuse(face_valence=0.7, face_arousal=0.4, hr_bpm=68, hrv_ms=60))
