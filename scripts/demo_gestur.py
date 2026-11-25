"""Demo gestur tangan dengan webcam laptop: kursor, sembilan posisi SAM, cubit, dan kembali.

Tidak ada frame yang disimpan; frame hanya ditampilkan di jendela OpenCV lalu dibuang.

Panel kanan layar menjelaskan setiap gestur dan menyorot gestur yang sedang terbaca:
  * menunjuk: kursor mengikuti pangkal telunjuk;
  * cubit (ibu jari ke telunjuk): pilih atau klik kotak SAM di bawah kursor;
  * telapak terbuka ditahan 1 detik: kembali (pilihan dihapus);
  * tangan hilang: kursor hilang;
  * tombol m: simulasi fase ukur 60 detik (gestur nonaktif, hitung mundur);
  * tombol q atau Esc: keluar.

Zoom (cubit lalu geser vertikal) adalah rancangan yang menunggu keputusan Calon Dosen
Pembimbing 1; hanya aktif dengan --zoom dan tidak pernah dikirim lewat MQTT.

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

from interaction.gesture_state import (  # noqa: E402
    GESTURE_ZOOM,
    MEMILIH,
    MENUNJUK,
    MesinGestur,
    PenomorPeristiwa,
    SetelanGestur,
)

N_SAM = 9
DURASI_UKUR_S = 60.0
LEBAR_PANEL = 360
WARNA = {
    "teks": (240, 240, 240),
    "redup": (150, 150, 150),
    "kotak": (90, 90, 90),
    "arah": (0, 200, 255),
    "pilih": (0, 200, 0),
    "panel": (30, 30, 30),
    "aktif": (0, 140, 255),
}
LEGENDA = [
    ("tunjuk", "Menunjuk", "kursor mengikuti pangkal telunjuk"),
    ("cubit", "Cubit", "pilih / klik kotak di bawah kursor"),
    ("telapak", "Telapak ditahan 1 dtk", "kembali (hapus pilihan)"),
    ("hilang", "Tangan hilang", "kursor hilang"),
    ("ukur", "Tombol m", "simulasi fase ukur 60 dtk: gestur mati"),
]


JUDUL_JENDELA = "MoodIn - demo gestur"
PETUNJUK = "Demo gestur berjalan. Keluar: q atau Esc di jendela, atau Ctrl+C di terminal. Fase ukur: m."
MAKS_GAGAL_BACA = 30


def aksi_tombol(kode: int) -> str | None:
    """Kode dari cv2.waitKey -> "keluar", "ukur", atau None."""
    if kode < 0:
        return None
    k = kode & 0xFF
    if k in (ord("q"), ord("Q"), 27):
        return "keluar"
    if k in (ord("m"), ord("M")):
        return "ukur"
    return None


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


def gestur_aktif(mesin: MesinGestur, fase_ukur: bool, t: float) -> str | None:
    """Kunci baris legenda yang sedang terbaca (untuk disorot)."""
    if fase_ukur:
        return "ukur"
    if mesin.kursor is None:
        return "hilang"
    if mesin._t_palm is not None and t - mesin._t_palm > 0.1:
        return "telapak"
    if mesin.keadaan == MEMILIH:
        return "cubit"
    if mesin.keadaan == MENUNJUK:
        return "tunjuk"
    return "hilang"


def progres_telapak(mesin: MesinGestur, t: float) -> float:
    if mesin._t_palm is None or mesin._palm_terkirim:
        return 0.0
    return min(1.0, (t - mesin._t_palm) / mesin.s.tahan_kembali_s)


def _panel(cv2, kanvas, x0, aktif, progres, sisa_ukur, info, pesan, zoom):
    h = kanvas.shape[0]
    cv2.rectangle(kanvas, (x0, 0), (kanvas.shape[1], h), WARNA["panel"], -1)
    y = 34
    cv2.putText(
        kanvas,
        "Gestur MoodIn",
        (x0 + 16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        WARNA["teks"],
        2,
    )
    y += 20
    for kunci, nama, arti in LEGENDA:
        y += 56
        on = kunci == aktif
        if on:
            cv2.rectangle(
                kanvas,
                (x0 + 8, y - 26),
                (kanvas.shape[1] - 8, y + 22),
                WARNA["aktif"],
                2,
            )
        cv2.putText(
            kanvas,
            nama,
            (x0 + 16, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WARNA["teks"] if on else WARNA["redup"],
            2,
        )
        cv2.putText(
            kanvas,
            arti,
            (x0 + 16, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WARNA["redup"],
            1,
        )
        if kunci == "telapak" and progres > 0:
            cv2.rectangle(
                kanvas,
                (x0 + 16, y + 18),
                (x0 + 16 + int(progres * (LEBAR_PANEL - 40)), y + 22),
                WARNA["pilih"],
                -1,
            )
    if zoom:
        y += 56
        cv2.putText(
            kanvas,
            "Zoom (rancangan)",
            (x0 + 16, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WARNA["teks"],
            2,
        )
        cv2.putText(
            kanvas,
            f"cubit + geser vertikal: {zoom:.2f}x",
            (x0 + 16, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WARNA["redup"],
            1,
        )
    y += 60
    if sisa_ukur is not None:
        cv2.putText(
            kanvas,
            f"FASE UKUR: {sisa_ukur:4.1f} s",
            (x0 + 16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WARNA["arah"],
            2,
        )
        y += 30
    for s in info:
        cv2.putText(
            kanvas, s, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WARNA["redup"], 1
        )
        y += 24
    if pesan:
        cv2.putText(
            kanvas,
            pesan,
            (x0 + 16, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WARNA["pilih"],
            2,
        )


def _gambar_kamera(cv2, frame, kotak, arah, pilihan, mesin, fase_ukur):
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
    cv2.putText(
        frame,
        "SAM (1-9)",
        (kotak[0][0], kotak[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        WARNA["teks"],
        2,
    )
    h, w = frame.shape[:2]
    if mesin.kursor is not None and not fase_ukur:
        cx, cy = int(mesin.kursor[0] * w), int(mesin.kursor[1] * h)
        cv2.circle(
            frame, (cx, cy), 14, WARNA["pilih"] if mesin.cubit else WARNA["arah"], -1
        )
        cv2.circle(frame, (cx, cy), 16, WARNA["teks"], 2)
    if fase_ukur:
        lapis = frame.copy()
        cv2.rectangle(lapis, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(lapis, 0.45, frame, 0.55, 0, frame)
        cv2.putText(
            frame,
            "Fase ukur: gestur nonaktif",
            (int(0.25 * w), int(0.45 * h)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            WARNA["teks"],
            2,
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--kamera", type=int, default=0)
    ap.add_argument("--lebar", type=int, default=1280)
    ap.add_argument("--tinggi", type=int, default=720)
    ap.add_argument("--zoom", action="store_true", help="rancangan zoom (bawaan mati)")
    ap.add_argument(
        "--mqtt",
        action="store_true",
        help="kirim peristiwa ke broker lokal (bawaan mati)",
    )
    ap.add_argument("--mqtt-host", default="localhost")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    args = ap.parse_args()

    import cv2
    import numpy as np

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
    mesin = MesinGestur(SetelanGestur(cermin=False, zoom_aktif=args.zoom))
    pelacak = PelacakTangan(model_complexity=0)
    pilihan = None
    t_ukur = None
    pesan, t_pesan = "", 0.0
    t_prev, fps = time.perf_counter(), 0.0
    gagal_baca = 0
    print(PETUNJUK)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                gagal_baca += 1
                if gagal_baca >= MAKS_GAGAL_BACA:
                    print(f"Kamera berhenti mengirim frame ({MAKS_GAGAL_BACA} kali gagal). Periksa kabel atau aplikasi lain yang memakai kamera.")
                    break
                continue
            gagal_baca = 0
            frame = cv2.flip(frame, 1)
            t = max(time.perf_counter(), t_prev + 1e-6)
            if t_ukur is not None and t - t_ukur >= DURASI_UKUR_S:
                t_ukur = None
                mesin.set_fase_ukur(False)
            fase_ukur = t_ukur is not None
            lm, skor = pelacak.proses(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            h, w = frame.shape[:2]
            kotak = kotak_sam(w, h)
            for ev in mesin.langkah(lm, skor, t):
                if ev.gesture == "select":
                    k = kotak_di(ev.x, ev.y, kotak, w, h)
                    if k is not None:
                        pilihan = k
                        pesan, t_pesan = f"Dipilih: SAM {k + 1}", t
                elif ev.gesture == "back":
                    pilihan = None
                    pesan, t_pesan = "Kembali", t
                if mq is not None and ev.gesture != GESTURE_ZOOM:
                    mq.publish_gesture(nomor(ev))
            arah = None
            if mesin.kursor is not None:
                arah = kotak_di(mesin.kursor[0], mesin.kursor[1], kotak, w, h)
            dt = t - t_prev
            t_prev = t
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
            _gambar_kamera(cv2, frame, kotak, arah, pilihan, mesin, fase_ukur)
            kanvas = np.zeros((h, w + LEBAR_PANEL, 3), np.uint8)
            kanvas[:, :w] = frame
            del frame
            rasio = "-" if mesin.rasio is None else f"{mesin.rasio:.2f}"
            info = [
                f"FPS {fps:4.1f}",
                f"keadaan: {mesin.keadaan}",
                f"rasio cubit: {rasio} (tutup<{mesin.s.ambang_tutup}, buka>{mesin.s.ambang_buka})",
                f"SAM terpilih: {'-' if pilihan is None else pilihan + 1}",
                "frame tidak disimpan" + ("; MQTT aktif" if mq else ""),
            ]
            _panel(
                cv2,
                kanvas,
                w,
                gestur_aktif(mesin, fase_ukur, t),
                progres_telapak(mesin, t),
                None if t_ukur is None else DURASI_UKUR_S - (t - t_ukur),
                info,
                pesan if t - t_pesan < 1.5 else "",
                mesin.faktor_zoom if args.zoom else None,
            )
            cv2.imshow(JUDUL_JENDELA, kanvas)
            aksi = aksi_tombol(cv2.waitKey(1))
            # Tombol silang jendela tidak mengirim kode tombol; cek visibilitas jendela.
            if aksi == "keluar" or cv2.getWindowProperty(JUDUL_JENDELA, cv2.WND_PROP_VISIBLE) < 1:
                break
            if aksi == "ukur":
                t_ukur = None if t_ukur is not None else t
                mesin.set_fase_ukur(t_ukur is not None)
    except KeyboardInterrupt:
        print("Dihentikan dengan Ctrl+C.")
    finally:
        pelacak.tutup()
        cap.release()
        cv2.destroyAllWindows()
        if mq is not None:
            mq.disconnect()


if __name__ == "__main__":
    main()
