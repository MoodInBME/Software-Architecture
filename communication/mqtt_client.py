"""Klien MQTT MoodIn (pembungkus paho-mqtt 2.x) sesuai kontrak IPC 2.0.0.

Fitur:
  - Last Will & Testament (LWT) otomatis di moodin/presence (retained) ->
    UI langsung tahu bila proses backend mati mendadak.
  - Helper publish untuk InferenceResult, SignalQuality, dan GestureEvent dengan
    QoS/retain sesuai kontrak.
  - Subscribe perintah UI (moodin/ui/command) via callback.
  - Transport sesuai ADR-10: broker lokal port 1883, tanpa TLS, username/password.
  - ``session_id`` dibuat saat klien dibuat (kontrak bagian 4 butir 1).

Butuh paho-mqtt==2.1.0. API callback memakai CallbackAPIVersion.VERSION2.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

import paho.mqtt.client as mqtt

from . import topics
from .contracts import (
    GestureEvent,
    InferenceResult,
    Presence,
    SignalQuality,
    UiCommand,
    new_session_id,
)


class MoodInMqttClient:
    def __init__(
        self,
        service_name="moodin-backend",
        host=topics.DEV_HOST,
        port=topics.DEV_PORT,
        username=None,
        password=None,
        client_id=None,
        session_id=None,
    ):
        self.service_name = service_name
        self.host = host
        self.port = port
        self.session_id = session_id or new_session_id()
        self._command_cb: Optional[Callable[[UiCommand], None]] = None
        self.connected = False

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id or service_name,
        )
        if username:
            self.client.username_pw_set(username, password)

        # LWT: bila koneksi putus mendadak, broker publish 'offline' (retained).
        offline = Presence(
            session_id=self.session_id,
            service=service_name,
            status=topics.PRESENCE_OFFLINE,
        ).to_json()
        self.client.will_set(
            topics.TOPIC_PRESENCE,
            offline,
            qos=topics.QOS_PRESENCE,
            retain=topics.RETAIN_PRESENCE,
        )

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    # ---- lifecycle ----
    def connect(self, keepalive=30):
        self.client.connect(self.host, self.port, keepalive)
        self.client.loop_start()

    def disconnect(self):
        # umumkan offline secara sopan sebelum putus
        self.publish_presence(topics.PRESENCE_OFFLINE)
        self.client.loop_stop()
        self.client.disconnect()

    # ---- callbacks ----
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = getattr(reason_code, "value", reason_code) == 0
        # umumkan online (retained) + langganan perintah UI
        self.publish_presence(topics.PRESENCE_ONLINE)
        client.subscribe(topics.TOPIC_UI_COMMAND, qos=topics.QOS_UI_COMMAND)

    def _on_disconnect(self, client, userdata, *args):
        self.connected = False

    def _on_message(self, client, userdata, msg):
        if msg.topic == topics.TOPIC_UI_COMMAND and self._command_cb:
            try:
                cmd = UiCommand.from_json(msg.payload.decode("utf-8"))
            except (ValueError, TypeError, KeyError, UnicodeDecodeError) as e:
                print(f"[mqtt] gagal parse UiCommand: {e}")
                return
            self._command_cb(cmd)

    # ---- publish ----
    def publish_presence(self, status):
        payload = Presence(
            session_id=self.session_id, service=self.service_name, status=status
        ).to_json()
        self.client.publish(
            topics.TOPIC_PRESENCE,
            payload,
            qos=topics.QOS_PRESENCE,
            retain=topics.RETAIN_PRESENCE,
        )

    def publish_result(self, result: InferenceResult):
        self.client.publish(
            topics.TOPIC_INFERENCE_RESULT,
            result.to_json(),
            qos=topics.QOS_INFERENCE_RESULT,
            retain=topics.RETAIN_INFERENCE_RESULT,
        )

    def publish_signal_quality(self, sq: SignalQuality):
        self.client.publish(
            topics.TOPIC_SIGNAL_QUALITY,
            sq.to_json(),
            qos=topics.QOS_SIGNAL_QUALITY,
            retain=False,
        )

    def publish_gesture(self, ev: GestureEvent):
        self.client.publish(
            topics.TOPIC_GESTURE_EVENT,
            ev.to_json(),
            qos=ev.qos,
            retain=topics.RETAIN_GESTURE_EVENT,
        )

    def publish_command(self, cmd: UiCommand):
        """Dipakai sisi UI untuk mengirim perintah ke mesin inferensi."""
        self.client.publish(
            topics.TOPIC_UI_COMMAND,
            cmd.to_json(),
            qos=topics.QOS_UI_COMMAND,
            retain=False,
        )

    # ---- subscribe ----
    def on_command(self, callback: Callable[[UiCommand], None]):
        self._command_cb = callback


def payload_dict(obj) -> dict:
    """Payload sebagai dict (untuk log atau uji), lewat to_json agar tervalidasi."""
    return json.loads(obj.to_json())
