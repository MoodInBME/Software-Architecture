"""Pemetaan rating SAM skala 9 ke lima kategori mood sesuai Tabel 3.3 naskah Pra-TA.

Berbeda dengan `labels.va_to_mood` era GEMASTIK (lingkaran radius 0,2 di sekitar titik asal):
  * Netral   : 4 <= V <= 6 dan 4 <= A <= 6 (pita persegi)
  * Ceria    : V > 5, A > 5, di luar pita Netral
  * Rileks   : V > 5, A <= 5, di luar pita Netral
  * Cemas    : V < 5, A > 5, di luar pita Netral
  * Tertekan : V < 5, A <= 5, di luar pita Netral
  * V tepat 5 di luar pita Netral: belum tercakup Tabel 3.3 (keputusan bersama dosen
    pembimbing). Fungsi mengembalikan None agar sampel tidak masuk metrik lima kategori
    tetapi tetap dipakai untuk CCC.
Skala SAM 1..9 dipetakan ke [-1, 1] dengan (s - 5) / 4, sehingga pita Netral = |v| <= 0,25.
"""

from __future__ import annotations

try:
    from .labels import MOOD_TO_ID
except ImportError:  # dipakai mandiri
    MOOD_TO_ID = {"Ceria": 0, "Cemas": 1, "Tertekan": 2, "Rileks": 3, "Netral": 4}

BAND = 0.25


def sam_to_unit(score: float) -> float:
    if not 1 <= score <= 9:
        raise ValueError(f"Rating SAM {score} di luar 1..9")
    return (float(score) - 5.0) / 4.0


def va_to_mood_tabel33(v: float, a: float, band: float = BAND):
    """v, a dalam [-1, 1]. Return nama kategori atau None bila jatuh di celah Tabel 3.3."""
    if abs(v) <= band and abs(a) <= band:
        return "Netral"
    if v == 0:
        return None
    if v > 0:
        return "Ceria" if a > 0 else "Rileks"
    return "Cemas" if a > 0 else "Tertekan"


def sam_to_mood(valence_sam: float, arousal_sam: float):
    return va_to_mood_tabel33(sam_to_unit(valence_sam), sam_to_unit(arousal_sam))


def va_to_mood_id_tabel33(v: float, a: float, band: float = BAND):
    m = va_to_mood_tabel33(v, a, band)
    return None if m is None else MOOD_TO_ID[m]
