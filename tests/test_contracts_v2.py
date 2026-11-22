"""Kontrak IPC 2.0.0: field wajib, decision_margin, topik gestur, dan QoS.

Tidak butuh broker; klien paho diganti objek tiruan.
"""

import json
import re

import pytest

from communication import topics
from communication.contracts import (
    GestureEvent,
    InferenceResult,
    Presence,
    SignalQuality,
    UiCommand,
    new_session_id,
    now_ms,
)

SID = "2026-11-22T13-40-00_a1b2"


def _hasil(**ubah):
    d = dict(
        session_id=SID,
        mood_id=3,
        mood_label="Rileks",
        decision_margin=0.62,
        valence=0.42,
        arousal=-0.35,
    )
    d.update(ubah)
    return InferenceResult(**d)


def test_versi_dan_field_wajib():
    assert topics.CONTRACT_VERSION == "2.0.0"
    d = json.loads(_hasil().to_json())
    assert d["schema_version"] == "2.0.0" and d["session_id"] == SID
    assert isinstance(d["ts"], int) and d["ts"] > 1_700_000_000_000
    assert "confidence" not in d and "version" not in d
    assert d["decision_margin"] == 0.62 and d["simulated"] is False


def test_session_id_dan_ts():
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d_[0-9a-f]{4}", new_session_id())
    assert isinstance(now_ms(), int)
    with pytest.raises(ValueError):
        _hasil(session_id="").to_json()
    with pytest.raises(ValueError):
        _hasil(ts=1753189230.5).to_json()  # detik float versi lama ditolak
    with pytest.raises(TypeError):
        InferenceResult(
            mood_id=0, mood_label="Ceria", decision_margin=0.5, valence=0, arousal=0
        )


@pytest.mark.parametrize("m", [-0.01, 1.01, float("nan")])
def test_margin_di_luar_rentang_ditolak(m):
    with pytest.raises(ValueError):
        _hasil(decision_margin=m).to_json()


def test_from_json_toleran_field_tak_dikenal():
    d = json.loads(_hasil().to_json())
    d["field_baru"] = 1
    r = InferenceResult.from_json(json.dumps(d))
    assert r.mood_id == 3


def test_gestur_payload_dan_qos():
    ev = GestureEvent(session_id=SID, seq=7, gesture="pointer", x=0.4, y=0.6, score=0.9)
    d = json.loads(ev.to_json())
    assert set(d) == {
        "schema_version",
        "session_id",
        "seq",
        "gesture",
        "x",
        "y",
        "score",
        "ts",
    }
    assert ev.qos == 0
    for g in ("select", "back", "hand_lost"):
        assert topics.gesture_qos(g) == 1
    GestureEvent(
        session_id=SID, seq=8, gesture="back", x=None, y=None, score=0.8
    ).to_json()


@pytest.mark.parametrize(
    "ubah",
    [
        dict(gesture="swipe"),
        dict(x=1.2),
        dict(y=None),
        dict(seq=-1),
        dict(seq=True),
        dict(score=1.5),
        dict(gesture="back"),  # back dengan posisi
    ],
)
def test_gestur_tidak_valid(ubah):
    d = dict(session_id=SID, seq=1, gesture="select", x=0.5, y=0.5, score=0.9)
    d.update(ubah)
    with pytest.raises(ValueError):
        GestureEvent(**d).to_json()


def test_payload_lain_memuat_field_wajib():
    for obj in (
        SignalQuality(session_id=SID, face_detected=True),
        Presence(session_id=SID, service="moodin-inference"),
        UiCommand(session_id=SID, command="start_session"),
    ):
        d = json.loads(obj.to_json())
        assert d["schema_version"] == "2.0.0" and isinstance(d["ts"], int)


def test_tidak_ada_tls_atau_port_8883():
    assert not hasattr(topics, "PROD_PORT_TLS") and topics.DEV_PORT == 1883


class _PahoPalsu:
    def __init__(self):
        self.pub = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.pub.append((topic, json.loads(payload), qos, retain))


def test_klien_menerbitkan_dengan_qos_kontrak():
    pytest.importorskip("paho.mqtt")
    from communication.mqtt_client import MoodInMqttClient

    c = MoodInMqttClient(session_id=SID)
    c.client = _PahoPalsu()
    c.publish_result(_hasil())
    c.publish_gesture(
        GestureEvent(session_id=SID, seq=1, gesture="pointer", x=0.1, y=0.2, score=0.9)
    )
    c.publish_gesture(
        GestureEvent(
            session_id=SID, seq=2, gesture="hand_lost", x=None, y=None, score=0.0
        )
    )
    (t1, p1, q1, r1), (t2, _, q2, r2), (t3, _, q3, _) = c.client.pub
    assert (t1, q1, r1) == ("moodin/inference/result", 1, False)
    assert (t2, q2, r2) == ("moodin/gesture/event", 0, False) and q3 == 1
    assert p1["session_id"] == SID
