# PATCH NOTES - MoodIn Software-Architecture (rev-b, 6 Agu 2026)

Cakupan patch ini: **S1 (DEMO_MODE jujur), S2 (ROI wajah untuk rPPG), S3+S4+S7 (MQTT lewat lapisan communication), dan rename frame Figma.**

Tema cermin **tidak diubah**. Palet `#3A3A3A` dipertahankan sesuai keputusan desain tim. Catatan risikonya tetap saya simpan di bagian E supaya ada jejaknya kalau nanti hasil rekaman video ternyata terlihat abu-abu menyala di balik sticker rubben.

---

## Cara pakai paling cepat

1. Salin `inference.py` dari ZIP ini menimpa `inference.py` di root repo.
2. Salin `default.yaml` ke `config/default.yaml` (sebelumnya file itu kosong 0 baris).
3. Tambahkan satu field ke `communication/contracts.py` (bagian B di bawah).
4. Tambahkan satu blok ke UI (bagian C).
5. Rename 25 frame Figma memakai tabel di `FIGMA_RENAME.md`.

Semua sudah lolos `ast.parse` (688 baris, 0 karakter non-ASCII). Belum bisa saya jalankan karena torch/mediapipe/paho tidak terpasang di sandbox saya.

---

## A. Apa yang berubah di `inference.py`

| Tanda | Masalah lama | Perlakuan baru |
|---|---|---|
| S1 | Sinusoid waktu menimpa keluaran model tanpa penanda apa pun | Tetap ada, tapi kini berlabel: banner CLI saat start, pita oranye permanen di layar, flag `simulated: true` di payload MQTT, suffix `(sim)` di angka HR |
| S2 | `cv2.resize(frame_rgb, (128,128))` meresize seluruh frame padahal komentarnya bilang crop wajah | `face_bbox_from_landmarks()` + `crop_face_for_rppg()`; margin 10% horizontal, 15% vertikal dengan bobot ke arah dahi |
| S3 | `mqtt.Client(client_id=...)` invalid di paho 2.x, melempar `TypeError` yang lolos dari `except ImportError` | `MoodInMqttClient` dari `communication/mqtt_client.py`, semua kegagalan ditangkap di `connect_mqtt()` |
| S4 | Payload dict manual: key `"timestamp"` (kontrak memakai `"ts"`), tanpa `"version"`, topic hardcode | `InferenceResult(**fields)` + `mq.publish_result()`; topic diambil dari `communication/topics.py` |
| S7 | `SignalQuality` dan `Presence` tidak pernah terbit sehingga aturan degradasi UI mustahil dijalankan | `publish_signal_quality()` tiap 15 frame (~2 Hz); `Presence` otomatis lewat LWT retained di `MoodInMqttClient` |
| S6 | `rr = bpm / 4.5` tampil seolah hasil ukur | Nilainya belum diubah, tapi label di layar jadi `RR: 17 /min (estimasi turunan)` |
| S8 | `gstreamer_pipeline` meminta 1280x960, bukan mode sensor IMX477 lazim | Default 1920x1080, dan bisa diubah dari YAML tanpa sentuh kode |
| S10 | Semua konstanta hardcode | `load_config()` membaca `config/default.yaml`, jatuh ke `_defaults()` bila gagal |

Catatan implementasi yang perlu Anda tahu:

- Ketika wajah tidak terdeteksi, kedua buffer sekarang **di-clear**, bukan dibiarkan. Ini mencegah sinyal rPPG tersambung dari dua sesi wajah yang berbeda.
- `bpm_from_signal()` saya angkat jadi fungsi modul-level dengan Hanning window dan band 0,7-3,5 Hz (42-210 BPM), supaya bisa diuji unit tanpa menyalakan kamera.
- Crop portrait sekarang punya cabang `else` untuk kasus frame lebih lebar dari rasio target; versi lama bisa menghasilkan indeks negatif pada resolusi tertentu.

---

## B. Patch `communication/contracts.py` (1 baris)

Tambahkan satu field ke dataclass `InferenceResult`. Taruh **sebelum** `version` dan `ts` supaya urutan default argument tetap valid.

GANTI:

    physio_used: bool = False

MENJADI:

    physio_used: bool = False
    simulated: bool = False

Lalu di dalam `validate()`, tambahkan baris ini di akhir:

    assert isinstance(self.simulated, bool), "simulated harus bool"

Kenapa perlu: `publish_result()` menyerialisasi dataclass, jadi atribut yang ditempel lewat `setattr` akan hilang saat `asdict()`. Field harus benar-benar dideklarasikan. `inference.py` sudah saya bungkus `try/except TypeError`, jadi kalau Anda lupa menambahkan field ini, program tetap jalan; hanya flag-nya yang tidak sampai ke UI.

Naikkan juga `CONTRACT_VERSION` di `communication/topics.py`:

GANTI:

    CONTRACT_VERSION = "1.0.0"

MENJADI:

    CONTRACT_VERSION = "1.1.0"

---

## C. Sisi UI (repo Interface-Display)

Dua hal, keduanya kecil.

**C1. Perbaiki `docs/ipc-contract.md`.** File itu menulis `"mood_id": "rileks"` sebagai string, padahal `labels.py` dan `late_fusion.py` memakai integer. `MOOD_CLASSES = ["Ceria", "Cemas", "Tertekan", "Rileks", "Netral"]` sehingga Rileks = 3. Yang salah dokumennya, bukan kodenya. Ganti contoh payload jadi:

    "mood_id": 3,
    "mood_label": "Rileks",

**C2. Tampilkan pita simulasi.** Di `core/bridge.py`, pada `_on_result`, teruskan field baru:

    self._simulated = bool(payload.get("simulated", False))
    self.simulatedChanged.emit()

Lalu di QML, di layer paling atas `Main.qml`, tanpa menyentuh tema warna yang sudah ada:

    Rectangle {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 44
        z: 999
        visible: bridge.simulated
        color: "#C86400"
        Text {
            anchors.centerIn: parent
            text: "MODE DEMO - NILAI DISIMULASIKAN"
            color: "#FFFFFF"
            font.pixelSize: 16
            font.bold: true
        }
    }

Saya taruh di `bottom` bukan `top` karena pada layar portrait 600x1024 bagian atas biasanya dipakai kartu mood, dan pita bawah lebih kecil kemungkinannya terpotong saat frame video di-crop.

---

## D. Mengapa label jujur ini penting untuk GEMASTIK

Saya sarankan Anda juga menyebutkannya lisan di narasi video, satu kalimat saja, misalnya: "Nilai yang tampil pada demo ini masih disimulasikan karena pelatihan model pada dataset CK+ sedang berjalan; arsitektur dan alur data yang ditampilkan sudah final."

Alasannya praktis, bukan moralistis. Juri Divisi 9 hampir pasti akan meminta dua orang berbeda berdiri di depan cermin. Dengan `DEMO_MODE`, keduanya mendapat mood identik pada detik yang sama karena nilainya fungsi `time.time()`. Kalau ketahuan tanpa disclaimer, kerusakan penilaiannya jauh lebih besar daripada kalau Anda menyatakannya lebih dulu. Dinyatakan di depan, itu jadi "prototipe dengan pipeline lengkap"; ketahuan belakangan, itu jadi masalah integritas. [High]

---

## E. Tema cermin: keputusan tim dihormati, risikonya dicatat

Palet tetap seperti sekarang (`#3A3A3A` dominan, teks sekunder `#A0A0A0`). Tidak ada file tema di ZIP ini.

Dua hal yang tetap perlu Anda ukur sebelum syuting, bukan untuk mengubah desain, tapi supaya tidak kaget di lokasi:

1. **Uji nyala sebelum figura ditutup.** Nyalakan layar berisi latar `#3A3A3A` penuh di balik kaca + sticker rubben, matikan lampu ruangan, lalu foto dari jarak 1 meter. Yang Anda cari: apakah area layar terbaca sebagai persegi panjang abu-abu yang menyala, atau menyatu dengan bidang cermin. Kalau menyatu, tidak ada yang perlu dilakukan.
2. **Kontras teks.** `#A0A0A0` di atas `#3A3A3A` bernilai 4,35:1, sedikit di bawah ambang WCAG AA 4,5:1, dan itu sebelum rugi transmitansi film 10-20%. Kalau saat uji nyala teks sekunder sulit dibaca dari 1 meter, opsi paling murah yang **tidak** mengubah identitas desain adalah menaikkan teks sekunder saja dari `#A0A0A0` ke `#C4C4C4` (jadi ~6,3:1) sambil latar tetap `#3A3A3A`. Itu perubahan satu token, bukan perubahan tema. [Medium]

Saya tidak menyiapkan patch untuk ini karena keputusannya ada di tangan tim setelah uji nyala.

---

## F. Yang belum tersentuh (sengaja, agar scope hari ini tetap kecil)

- `generate_dummy.py` masih melatih 50 iterasi MSE terhadap target acak. Selama `DEMO_MODE` menyala hal ini tidak berdampak, karena keluarannya toh ditimpa. Tapi jangan sampai file itu dipakai sebagai bukti "model sudah dilatih".
- POS/CHROM sebagai jalur BPM tanpa pelatihan belum saya sertakan. Ini kandidat terkuat begitu error notebook CK+ selesai: sekitar 40 baris NumPy, tanpa bobot, dan bisa membuat angka HR jadi nyata walau mood masih dari model.
- Konsolidasi modul preprocessing yang duplikat, dan `__pycache__/*.pyc` yang masih ikut ter-commit.

---

## G. Soal error notebook CK+

Kalau nanti mau ditangani, kirimkan traceback lengkapnya saja. Tiga penyebab yang paling sering pada notebook itu, berdasarkan konstanta yang kita pakai: path `/kaggle/input/ckplus` yang struktur subfoldernya beda dari asumsi, regex subjek `S\d{3}` tidak menemukan match sehingga split subject-independent menghasilkan fold kosong, dan panjang klip yang lebih pendek dari `CLIP_LEN = 150` sehingga tensor gagal di-stack. Traceback akan langsung membedakan ketiganya.
