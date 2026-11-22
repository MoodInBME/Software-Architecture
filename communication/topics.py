"""Topik & QoS MQTT MoodIn (kontrak IPC 2.0.0, MoodIn-master/docs/ipc-contract.md).

Semua proses (inference, UI, sensor) HARUS memakai konstanta ini agar konsisten.
Transport mengikuti ADR-10: broker lokal port 1883, tanpa TLS.
"""

from __future__ import annotations

CONTRACT_VERSION = "2.0.0"
BASE = "moodin"

# Topik
TOPIC_PRESENCE = f"{BASE}/presence"  # status hidup/mati proses (retained + LWT)
TOPIC_SIGNAL_QUALITY = f"{BASE}/signal/quality"  # kualitas sinyal rPPG/wajah
TOPIC_INFERENCE_RESULT = f"{BASE}/inference/result"  # hasil kategori mood
TOPIC_UI_COMMAND = f"{BASE}/ui/command"  # perintah dari UI ke mesin inferensi
TOPIC_GESTURE_EVENT = f"{BASE}/gesture/event"  # kursor dan pemilihan gestur tangan

# QoS
QOS_PRESENCE = 1
QOS_SIGNAL_QUALITY = 0
QOS_INFERENCE_RESULT = 1
QOS_UI_COMMAND = 1
QOS_GESTURE_POINTER = 0
QOS_GESTURE_DISCRETE = 1  # select, back, hand_lost

# Retain
RETAIN_PRESENCE = True
RETAIN_INFERENCE_RESULT = False
RETAIN_GESTURE_EVENT = False

# Broker (ADR-10: lokal, tanpa TLS)
DEV_HOST, DEV_PORT = "localhost", 1883

# Payload presence
PRESENCE_ONLINE = "online"
PRESENCE_OFFLINE = "offline"

# Gestur
GESTURE_POINTER = "pointer"
GESTURE_SELECT = "select"
GESTURE_BACK = "back"
GESTURE_HAND_LOST = "hand_lost"
GESTURES = (GESTURE_POINTER, GESTURE_SELECT, GESTURE_BACK, GESTURE_HAND_LOST)
GESTURE_POINTER_MAX_HZ = 30.0


def gesture_qos(gesture: str) -> int:
    """QoS per jenis peristiwa gestur sesuai kontrak 2.0.0 bagian 3.5."""
    if gesture not in GESTURES:
        raise ValueError(f"gesture harus salah satu dari {GESTURES}, bukan {gesture!r}")
    return QOS_GESTURE_POINTER if gesture == GESTURE_POINTER else QOS_GESTURE_DISCRETE
