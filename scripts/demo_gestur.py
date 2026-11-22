"""Demo gestur tangan dengan webcam laptop: kursor, sembilan posisi SAM, cubit, dan kembali.

Tidak ada frame yang disimpan; frame hanya ditampilkan di jendela OpenCV lalu dibuang.

Kontrol:
  * gerakkan tangan: kursor mengikuti pangkal telunjuk;
  * cubit (ibu jari ke telunjuk): pilih kotak SAM di bawah kursor;
  * telapak terbuka ditahan 1 detik: kembali (pilihan dihapus);
  * tombol m: simulasi fase ukur (gestur mati); tombol q atau Esc: keluar.

Contoh (PowerShell, dari folder repo):
    .\\.venv-gpu\\Scripts\\python.exe scripts\\demo_gestur.py
    .\\.venv-gpu\\Scripts\\python.exe scripts\\demo_gestur.py --mqtt --mqtt-host localhost
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from interaction.gesture_state import MesinGestur, PenomorPeristiwa, SetelanGestur  # noqa: E402

N_SAM = 9
WARNA = {
    "teks": (240, 240, 240),
    "kotak": (90, 90, 90),
    "arah": (0, 200, 255),
    "pilih": (0, 200, 0),
}


def kotak_sam(
    lebar: int, tinggi: int, n: int = N_SAM
) -> list[tuple[int, int, int, int]]:
    """Kotak (x0, y0, x1, y1) piksel untuk n posisi SAM dalam satu baris di bawah layar."""
    tepi = int(0.04 * lebar)
    w = (lebar - 2 * tepi) / n
    y0, y1 = int(0.78 * tinggi), int(0.92 * tinggi)
    return [
        (int(tepi + k * w) + 3, y0, int(tepi + (k + 1) * w) - 3, y1) for k in range(n)
    ]


def kotak_di(x: float, y: float, kotak, lebar: int, tinggi: int) -> int | None:
    """Indeks kotak di bawah kursor (koordinat layar 0..1), atau None."""
    px, py = x * lebar, y * tinggi
    for k, (x0, y0, x1, y1) in enumerate(kotak):
        if x0 <= px <= x1 and y0 <= py <= y1:
            return k
    return None


def _gambar(cv2, frame, kotak, arah, pilihan, mesin, fps, fase_ukur):
    h, w = frame.shape[:2]
    for k, (x0, y0, x1, y1) in enumerate(kotak):
        warna = (
            WARNA["pilih"]
            if k == pilihan
            else WARNA["arah"]
            if k == arah
            else WARNA["kotak"]
        )
        cv2.rectangle(frame, (x0, y0), (x1, y1), warna, -1 if k == pilihan else 2)
        cv2.putText(
            frame,
            str(k + 1),
            ((x0 + x1) // 2 - 6, (y0 + y1) // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WARNA["teks"],
            2,
        )
    if mesin.kursor is not None and not fase_ukur:
        cx, cy = int(mesin.kursor[0] * w), int(mesin.kursor[1] * h)
        cv2.circle(
            frame, (cx, cy), 12, WARNA["pilih"] if mesin.cubit else WARNA["arah"], -1
        )
    rasio = "-" if mesin.rasio is None else f"{mesin.rasio:.2f}"
    baris = [
        f"FPS {fps:4.1f}   keadaan: {mesin.keadaan}   rasio cubit: {rasio}",
        f"SAM: {'-' if pilihan is None else pilihan + 1}   (m: fase ukur, q: keluar)",
    ]
    if fase_ukur:
        baris.append("FASE UKUR: gestur nonaktif")
    for i, s in enumerate(baris):
        cv2.putText(
            frame, s, (12, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WARNA["teks"], 2
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--kamera", type=int, default=0)
    ap.add_argument("--lebar", type=int, default=1280)
    ap.add_argument("--tinggi", type=int, default=720)
    ap.add_argument(
        "--mqtt",
        action="store_true",
        help="kirim peristiwa ke broker lokal (bawaan mati)",
    )
    ap.add_argument("--mqtt-host", default="localhost")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    args = ap.parse_args()

    import cv2

    from interaction.hand_tracker import PelacakTangan

    api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.kamera, api)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.lebar)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.tinggi)
    if not cap.isOpened():
        raise SystemExit(f"Kamera {args.kamera} tidak bisa dibuka")

    mq = None
    if args.mqtt:
        from communication.mqtt_client import MoodInMqttClient

        mq = MoodInMqttClient(
            service_name="moodin-demo-gestur", host=args.mqtt_host, port=args.mqtt_port
        )
        mq.connect()
        print(f"[mqtt] {args.mqtt_host}:{args.mqtt_port} session_id={mq.session_id}")
    nomor = PenomorPeristiwa(mq.session_id) if mq else None

    # Frame dicerminkan sebelum diproses, jadi koordinat tidak perlu dicerminkan lagi.
    mesin = MesinGestur(SetelanGestur(cermin=False))
    pelacak = PelacakTangan(model_complexity=0)
    pilihan = None
    fase_ukur = False
    t_prev, fps = time.monotonic(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            t = time.monotonic()
            lm, skor = pelacak.proses(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            h, w = frame.shape[:2]
            kotak = kotak_sam(w, h)
            for ev in mesin.langkah(lm, skor, t):
                if ev.gesture == "select":
                    k = kotak_di(ev.x, ev.y, kotak, w, h)
                    if k is not None:
                        pilihan = k
                elif ev.gesture == "back":
                    pilihan = None
                if mq is not None:
                    mq.publish_gesture(nomor(ev))
            arah = None
            if mesin.kursor is not None:
                arah = kotak_di(mesin.kursor[0], mesin.kursor[1], kotak, w, h)
            dt = t - t_prev
            t_prev = t
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
            _gambar(cv2, frame, kotak, arah, pilihan, mesin, fps, fase_ukur)
            cv2.imshow("MoodIn - demo gestur", frame)
            del frame
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                fase_ukur = not fase_ukur
                mesin.set_fase_ukur(fase_ukur)
    finally:
        pelacak.tutup()
        cap.release()
        cv2.destroyAllWindows()
        if mq is not None:
            mq.disconnect()


if __name__ == "__main__":
    main()
