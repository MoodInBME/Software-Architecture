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

Revisi (VA-bridge): ditambahkan pemetaan emosi->VA (EMOTION_TO_VA) + helper
numerik (va_to_mood_id) untuk melatih regresi VA. API lama dipertahankan penuh.
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
# tetapi sebaiknya ditinjau per-dataset. Lihat docs/arsip-gemastik/ST-GCN-PLAN.md.
CATEGORICAL_MAP = {
    "happy": "Ceria", "happiness": "Ceria", "joy": "Ceria", "surprise": "Ceria",
    "fear": "Cemas", "anger": "Cemas", "angry": "Cemas", "disgust": "Cemas", "contempt": "Cemas",
    "sad": "Tertekan", "sadness": "Tertekan",
    "calm": "Rileks", "relaxed": "Rileks", "content": "Rileks",
    "neutral": "Netral",
}


def categorical_to_mood(label: str) -> str:
    return CATEGORICAL_MAP.get(str(label).strip().lower(), "Netral")


def mood_to_id(mood: str) -> int:
    return MOOD_TO_ID[mood]


# ---------------------------------------------------------------------------
# Tambahan VA-bridge: emosi kategorikal -> titik (valence, arousal)
# Nilai mengacu norma Russell/AffectNet; sesuaikan bila punya anotasi VA sendiri.
# ---------------------------------------------------------------------------
EMOTION_TO_VA = {
    "neutral": (0.0, 0.0),
    "happy": (0.80, 0.50), "happiness": (0.80, 0.50), "joy": (0.80, 0.50),
    "sad": (-0.70, -0.40), "sadness": (-0.70, -0.40),
    "angry": (-0.70, 0.70), "anger": (-0.70, 0.70),
    "fear": (-0.60, 0.80), "fearful": (-0.60, 0.80),
    "disgust": (-0.60, 0.30), "disgusted": (-0.60, 0.30),
    "surprise": (0.30, 0.80), "surprised": (0.30, 0.80), "surprising": (0.30, 0.80),
    "contempt": (-0.50, 0.20),
    "calm": (0.50, -0.55), "relaxed": (0.55, -0.60), "content": (0.60, -0.30),
}


def emotion_to_va(name: str) -> tuple[float, float]:
    key = str(name).strip().lower()
    if key not in EMOTION_TO_VA:
        raise KeyError(f"Emosi tak dikenal: {name!r}. Tambahkan ke EMOTION_TO_VA.")
    return EMOTION_TO_VA[key]


def va_to_mood_id(valence: float, arousal: float,
                  neutral_radius: float = NEUTRAL_RADIUS) -> int:
    """Versi numerik dari va_to_mood (mengembalikan indeks 0..4)."""
    return MOOD_TO_ID[va_to_mood(valence, arousal, neutral_radius)]
