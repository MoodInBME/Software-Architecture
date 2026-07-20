"""Adapter label per-dataset ekspresi -> (valence, arousal).

Semua dataset dinormalkan ke titik VA memakai `emotion_to_va` (untuk dataset
kategorikal) atau diteruskan apa adanya (untuk dataset ber-VA seperti Aff-Wild2).
"""
from __future__ import annotations

from models.st_gcn_graph_tran.labels import emotion_to_va, va_to_mood_id

# --- CK+ ---------------------------------------------------------------------
# Kode emosi CK+ (Emotion_labels): 0=neutral,1=anger,2=contempt,3=disgust,
# 4=fear,5=happy,6=sadness,7=surprise
CKPLUS_CODE_TO_EMOTION = {
    0: "neutral", 1: "anger", 2: "contempt", 3: "disgust",
    4: "fear", 5: "happy", 6: "sadness", 7: "surprise",
}

OULU_EMOTIONS = ["anger", "disgust", "fear", "happy", "sadness", "surprise"]
CAER_EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]


def ckplus_code_to_va(code) -> tuple[float, float]:
    return emotion_to_va(CKPLUS_CODE_TO_EMOTION[int(float(code))])


def emotion_name_to_va(name: str) -> tuple[float, float]:
    return emotion_to_va(name)


def affwild2_va(valence: float, arousal: float) -> tuple[float, float]:
    """Aff-Wild2 sudah beranotasi VA di [-1, 1]; clamp untuk keamanan."""
    v = max(-1.0, min(1.0, float(valence)))
    a = max(-1.0, min(1.0, float(arousal)))
    return v, a


def va_to_mood_index(valence: float, arousal: float, neutral_radius: float = 0.2) -> int:
    return va_to_mood_id(valence, arousal, neutral_radius)
