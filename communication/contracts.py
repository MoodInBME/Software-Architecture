"""Skema payload MQTT MoodIn (kontrak IPC v1.0.0) + validasi.

Memakai dataclass murni (tanpa dependensi eksternal) supaya bisa dipakai di
sisi backend ML maupun UI (PySide6/QML). Setiap payload punya to_json/from_json
dan validasi rentang nilai.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from . import topics


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class InferenceResult:
    """Payload moodin/inference/result."""
    mood_id: int
    mood_label: str
    confidence: float
    valence: float
    arousal: float
    heart_rate_bpm: Optional[float] = None
    hrv_ms: Optional[float] = None
    physio_used: bool = False
    version: str = topics.CONTRACT_VERSION
    ts: float = field(default_factory=time.time)

    def validate(self):
        if not (0 <= self.mood_id <= 4):
            raise ValueError(f"mood_id di luar 0..4: {self.mood_id}")
        self.confidence = _clamp(float(self.confidence), 0.0, 1.0)
        self.valence = _clamp(float(self.valence), -1.0, 1.0)
        self.arousal = _clamp(float(self.arousal), -1.0, 1.0)
        if self.heart_rate_bpm is not None and self.heart_rate_bpm <= 0:
            raise ValueError("heart_rate_bpm harus > 0 atau None")
        if self.hrv_ms is not None and self.hrv_ms < 0:
            raise ValueError("hrv_ms harus >= 0 atau None")
        return self

    def to_json(self):
        return json.dumps(asdict(self.validate()), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload):
        d = json.loads(payload)
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known).validate()


@dataclass
class SignalQuality:
    """Payload moodin/signal/quality."""
    face_detected: bool
    snr_db: Optional[float] = None      # SNR sinyal rPPG
    illumination: Optional[float] = None  # 0..1 kecukupan cahaya
    roi_stability: Optional[float] = None  # 0..1 kestabilan ROI wajah
    version: str = topics.CONTRACT_VERSION
    ts: float = field(default_factory=time.time)

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload):
        d = json.loads(payload)
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class Presence:
    """Payload moodin/presence (retained + LWT)."""
    service: str
    status: str = topics.PRESENCE_ONLINE
    version: str = topics.CONTRACT_VERSION
    ts: float = field(default_factory=time.time)

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload):
        d = json.loads(payload)
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class UiCommand:
    """Payload moodin/ui/command. action mis.: 'start', 'stop', 'calibrate', 'set_profile'."""
    action: str
    params: dict = field(default_factory=dict)
    version: str = topics.CONTRACT_VERSION
    ts: float = field(default_factory=time.time)

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload):
        d = json.loads(payload)
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)
