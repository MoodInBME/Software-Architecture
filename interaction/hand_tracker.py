"""Pembungkus MediaPipe Hands 0.10.14 (``mp.solutions.hands``) untuk satu tangan.

Frame hanya diproses di memori dan tidak disimpan. Impor mediapipe ditunda agar
modul lain di paket ini tetap bisa diuji tanpa mediapipe.
"""

from __future__ import annotations

import numpy as np


class PelacakTangan:
    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise RuntimeError("mediapipe tanpa mp.solutions; pasang mediapipe==0.10.14")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def proses(self, frame_rgb: np.ndarray):
        """Return (titik (21, 2) dalam [0, 1], skor) atau (None, 0.0)."""
        res = self._hands.process(frame_rgb)
        if not res.multi_hand_landmarks:
            return None, 0.0
        lm = res.multi_hand_landmarks[0].landmark
        pts = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
        skor = 1.0
        if res.multi_handedness:
            skor = float(res.multi_handedness[0].classification[0].score)
        return pts, skor

    def tutup(self) -> None:
        self._hands.close()
