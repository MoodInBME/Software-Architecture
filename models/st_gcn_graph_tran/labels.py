"""
Harmonisasi label emosi lintas-dataset ke 5 kelas mood MoodIn,
berbasis kuadran Valence-Arousal (VA). V dan A diasumsikan ternormalisasi
ke rentang [-1, 1].

Pemetaan kuadran:
  Q1 (V>=0, A>=0) -> Ceria     (positif-aktif)
  Q2 (V<0,  A>=0) -> Cemas     (negatif-aktif)
  Q3 (V<0,  A<0)  -> Tertekan  (negatif-pasif)
  Q4 (V>=0, A<0)  -> Rileks    (positif-pasif)
  Zona radius kecil di sekitar origin -> Netral
"""
from __future__ import annotations

MOOD_CLASSES = ["Ceria", "Cemas", "Tertekan", "Rileks", "Netral"]
MOOD_TO_ID = {m: i for i, m in enumerate(MOOD_CLASSES)}
ID_TO_MOOD = {i: m for m, i in MOOD_TO_ID.items()}

NEUTRAL_RADIUS = 0.2


def va_to_mood(valence: float, arousal: float,
               neutral_radius: float = NEUTRAL_RADIUS) -> str:
    if (valence ** 2 + arousal ** 2) ** 0.5 <= neutral_radius:
        return "Netral"
    if valence >= 0 and arousal >= 0:
        return "Ceria"
    if valence < 0 and arousal >= 0:
        return "Cemas"
    if valence < 0 and arousal < 0:
        return "Tertekan"
    return "Rileks"


# Pemetaan label kategorikal dataset (AffectNet/CK+/DFEW dsb.) -> 5 kelas MoodIn.
# CATATAN: 'surprise' ambigu (arousal tinggi, valence netral); default ke Ceria
# tetapi sebaiknya ditinjau per-dataset. Lihat docs/ST-GCN-PLAN.md.
CATEGORICAL_MAP = {
    "happy": "Ceria", "happiness": "Ceria", "joy": "Ceria", "surprise": "Ceria",
    "fear": "Cemas", "anger": "Cemas", "angry": "Cemas",
    "sad": "Tertekan", "sadness": "Tertekan", "disgust": "Tertekan", "contempt": "Tertekan",
    "calm": "Rileks", "relaxed": "Rileks", "content": "Rileks",
    "neutral": "Netral",
}


def categorical_to_mood(label: str) -> str:
    return CATEGORICAL_MAP.get(str(label).strip().lower(), "Netral")


def mood_to_id(mood: str) -> int:
    return MOOD_TO_ID[mood]
