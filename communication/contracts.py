"""Skema payload MQTT MoodIn (kontrak IPC 2.0.0) + validasi.

Memakai dataclass murni (tanpa dependensi eksternal). Setiap payload memuat field
wajib kontrak: ``schema_version``, ``session_id``, dan ``ts`` (epoch milidetik UTC,
integer). Semua field diisi lewat kata kunci.

Catatan selaras (lihat kontrak bagian 3.1 dan 3.2): ``Presence`` dan
``SignalQuality`` masih memakai isi versi mesin inferensi, belum isi kontrak.
"""

from __future__ import annotations

import json
import math
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from . import topics


def now_ms() -> int:
    """Waktu sekarang dalam epoch milidetik UTC (integer)."""
    return int(time.time() * 1000)


def new_session_id() -> str:
    """``<ISO8601-detik>_<4 hex acak>``, misalnya ``2026-11-22T13-40-00_a1b2``."""
    t = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{t}_{secrets.token_hex(2)}"


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _check_common(obj) -> None:
    if not isinstance(obj.session_id, str) or not obj.session_id:
        raise ValueError("session_id wajib berupa string tidak kosong")
    if isinstance(obj.ts, bool) or not isinstance(obj.ts, int) or obj.ts < 0:
        raise ValueError(f"ts harus integer epoch milidetik, dapat {obj.ts!r}")


def _from_json(cls, payload):
    d = json.loads(payload)
    known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
    return cls(**known).validate()


@dataclass(kw_only=True)
class InferenceResult:
    """Payload moodin/inference/result. Mood dari jalur wajah saja."""

    session_id: str
    mood_id: int
    mood_label: str
    decision_margin: float
    valence: float
    arousal: float
    heart_rate_bpm: Optional[float] = None
    hrv_ms: Optional[float] = None
    physio_used: bool = False
    simulated: bool = False
    schema_version: str = topics.CONTRACT_VERSION
    ts: int = field(default_factory=now_ms)

    def validate(self):
        _check_common(self)
        if not (0 <= int(self.mood_id) <= 4):
            raise ValueError(f"mood_id di luar 0..4: {self.mood_id}")
        m = float(self.decision_margin)
        if not math.isfinite(m) or not (0.0 <= m <= 1.0):
            raise ValueError(f"decision_margin harus di [0, 1], dapat {m}")
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
        return _from_json(cls, payload)


@dataclass(kw_only=True)
class GestureEvent:
    """Payload moodin/gesture/event. Tidak memuat landmark tangan."""

    session_id: str
    seq: int
    gesture: str
    x: Optional[float]
    y: Optional[float]
    score: float
    schema_version: str = topics.CONTRACT_VERSION
    ts: int = field(default_factory=now_ms)

    def validate(self):
        _check_common(self)
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 0:
            raise ValueError(f"seq harus integer >= 0, dapat {self.seq!r}")
        if self.gesture not in topics.GESTURES:
            raise ValueError(f"gesture tidak dikenal: {self.gesture!r}")
        tanpa_posisi = self.gesture in (topics.GESTURE_BACK, topics.GESTURE_HAND_LOST)
        if tanpa_posisi:
            if self.x is not None or self.y is not None:
                raise ValueError(f"x dan y harus null untuk {self.gesture}")
        else:
            for nama, v in (("x", self.x), ("y", self.y)):
                if v is None or not math.isfinite(float(v)) or not (0.0 <= v <= 1.0):
                    raise ValueError(
                        f"{nama} harus di [0, 1] untuk {self.gesture}, dapat {v!r}"
                    )
        if not math.isfinite(float(self.score)) or not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score harus di [0, 1], dapat {self.score!r}")
        return self

    @property
    def qos(self) -> int:
        return topics.gesture_qos(self.gesture)

    def to_json(self):
        return json.dumps(asdict(self.validate()), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload):
        return _from_json(cls, payload)


@dataclass(kw_only=True)
class SignalQuality:
    """Payload moodin/signal/quality (isi belum selaras kontrak bagian 3.2)."""

    session_id: str
    face_detected: bool
    snr_db: Optional[float] = None  # SNR sinyal rPPG
    illumination: Optional[float] = None  # 0..1 kecukupan cahaya
    roi_stability: Optional[float] = None  # 0..1 kestabilan ROI wajah
    schema_version: str = topics.CONTRACT_VERSION
    ts: int = field(default_factory=now_ms)

    def validate(self):
        _check_common(self)
        return self

    def to_json(self):
        return json.dumps(asdict(self.validate()), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload):
        return _from_json(cls, payload)


@dataclass(kw_only=True)
class Presence:
    """Payload moodin/presence (retained + LWT; isi belum selaras kontrak bagian 3.1)."""

    session_id: str
    service: str
    status: str = topics.PRESENCE_ONLINE
    schema_version: str = topics.CONTRACT_VERSION
    ts: int = field(default_factory=now_ms)

    def validate(self):
        _check_common(self)
        return self

    def to_json(self):
        return json.dumps(asdict(self.validate()), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload):
        return _from_json(cls, payload)


@dataclass(kw_only=True)
class UiCommand:
    """Payload moodin/ui/command, dari antarmuka ke mesin inferensi."""

    session_id: str
    command: str
    params: dict = field(default_factory=dict)
    schema_version: str = topics.CONTRACT_VERSION
    ts: int = field(default_factory=now_ms)

    def validate(self):
        _check_common(self)
        if not isinstance(self.command, str) or not self.command:
            raise ValueError("command wajib berupa string tidak kosong")
        return self

    def to_json(self):
        return json.dumps(asdict(self.validate()), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload):
        return _from_json(cls, payload)
