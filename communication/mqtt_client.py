"""Klien MQTT MoodIn (pembungkus paho-mqtt 2.x) sesuai kontrak IPC v1.0.0.

Fitur:
  - Last Will & Testament (LWT) otomatis di moodin/presence (retained) ->
    UI langsung tahu bila proses backend mati mendadak.
  - Helper publish untuk InferenceResult & SignalQuality dengan QoS/retain benar.
  - Subscribe perintah UI (moodin/ui/command) via callback.
  - Mode dev (localhost:1883) & prod (TLS:8883).

Butuh paho-mqtt==2.1.0. API callback memakai CallbackAPIVersion.VERSION2.
"""
from __future__ import annotations

from typing import Callable, Optional

import paho.mqtt.client as mqtt

from . import topics
from .contracts import InferenceResult, SignalQuality, Presence, UiCommand


class MoodInMqttClient:
    def __init__(self, service_name="moodin-backend", host=topics.DEV_HOST,
                 port=topics.DEV_PORT, use_tls=False, username=None, password=None,
                 ca_certs=None, client_id=None):
        self.service_name = service_name
        self.host = host
        self.port = topics.PROD_PORT_TLS if (use_tls and port == topics.DEV_PORT) else port
        self._command_cb: Optional[Callable[[UiCommand], None]] = None
        self.connected = False

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id or service_name,
        )
        if username:
            self.client.username_pw_set(username, password)
        if use_tls:
            self.client.tls_set(ca_certs=ca_certs)

        # LWT: bila koneksi putus mendadak, broker publish 'offline' (retained).
        offline = Presence(service=service_name, status=topics.PRESENCE_OFFLINE).to_json()
        self.client.will_set(topics.TOPIC_PRESENCE, offline,
                             qos=topics.QOS_PRESENCE, retain=topics.RETAIN_PRESENCE)

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
        self.connected = (getattr(reason_code, "value", reason_code) == 0)
        # umumkan online (retained) + langganan perintah UI
        self.publish_presence(topics.PRESENCE_ONLINE)
        client.subscribe(topics.TOPIC_UI_COMMAND, qos=topics.QOS_UI_COMMAND)

    def _on_disconnect(self, client, userdata, *args):
        self.connected = False

    def _on_message(self, client, userdata, msg):
        if msg.topic == topics.TOPIC_UI_COMMAND and self._command_cb:
            try:
                self._command_cb(UiCommand.from_json(msg.payload.decode("utf-8")))
            except Exception as e:  # jangan sampai callback merusak loop
                print(f"[mqtt] gagal parse UiCommand: {e}")

    # ---- publish ----
    def publish_presence(self, status):
        payload = Presence(service=self.service_name, status=status).to_json()
        self.client.publish(topics.TOPIC_PRESENCE, payload,
                            qos=topics.QOS_PRESENCE, retain=topics.RETAIN_PRESENCE)

    def publish_result(self, result: InferenceResult):
        self.client.publish(topics.TOPIC_INFERENCE_RESULT, result.to_json(),
                            qos=topics.QOS_INFERENCE_RESULT,
                            retain=topics.RETAIN_INFERENCE_RESULT)

    def publish_signal_quality(self, sq: SignalQuality):
        self.client.publish(topics.TOPIC_SIGNAL_QUALITY, sq.to_json(),
                            qos=topics.QOS_SIGNAL_QUALITY, retain=False)

    def publish_command(self, cmd: UiCommand):
        """Dipakai sisi UI untuk mengirim perintah ke backend."""
        self.client.publish(topics.TOPIC_UI_COMMAND, cmd.to_json(),
                            qos=topics.QOS_UI_COMMAND, retain=False)

    # ---- subscribe ----
    def on_command(self, callback: Callable[[UiCommand], None]):
        self._command_cb = callback


if __name__ == "__main__":
    # Contoh pemakaian (butuh broker MQTT berjalan di localhost:1883).
    import time
    from .contracts import InferenceResult

    cli = MoodInMqttClient(service_name="moodin-demo")
    cli.on_command(lambda c: print("perintah UI:", c))
    cli.connect()
    try:
        cli.publish_result(InferenceResult(
            mood_id=0, mood_label="Ceria", confidence=0.82,
            valence=0.6, arousal=0.4, heart_rate_bpm=72, hrv_ms=55, physio_used=True))
        time.sleep(1)
    finally:
        cli.disconnect()
