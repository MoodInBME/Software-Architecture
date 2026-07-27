"""Topik & QoS MQTT MoodIn (kontrak IPC v1.0.0).

Semua proses (inference, UI, sensor) HARUS memakai konstanta ini agar konsisten.
"""
from __future__ import annotations

CONTRACT_VERSION = "1.0.0"
BASE = "moodin"

# Topik
TOPIC_PRESENCE = f"{BASE}/presence"            # status hidup/mati proses (retained + LWT)
TOPIC_SIGNAL_QUALITY = f"{BASE}/signal/quality"  # kualitas sinyal rPPG/wajah
TOPIC_INFERENCE_RESULT = f"{BASE}/inference/result"  # hasil mood tergabung
TOPIC_UI_COMMAND = f"{BASE}/ui/command"          # perintah dari UI ke backend

# QoS
QOS_PRESENCE = 1
QOS_SIGNAL_QUALITY = 0
QOS_INFERENCE_RESULT = 1
QOS_UI_COMMAND = 1

# Retain
RETAIN_PRESENCE = True
RETAIN_INFERENCE_RESULT = False

# Broker default
DEV_HOST, DEV_PORT = "localhost", 1883
PROD_PORT_TLS = 8883

# Payload presence
PRESENCE_ONLINE = "online"
PRESENCE_OFFLINE = "offline"
