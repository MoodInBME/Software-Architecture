# Rename 25 Frame Figma - MoodIn (ready for development)

Saat ini frame di section ready-for-development bernama `Frame 56` sampai `Frame 80`. Nama ini bermasalah bukan karena estetika, tapi karena tiga hal konkret:

1. Figma MCP mengembalikan nama frame apa adanya di `get_metadata`. Kalau namanya `Frame 63`, Antigravity CLI tidak punya informasi apa pun untuk menebak file QML tujuan, dan Anda harus menjelaskan setiap layar secara manual di prompt.
2. Seat Starter / View / Collab dibatasi **6 tool call MCP per bulan**. Setiap prompt yang gagal karena agent salah menebak layar membakar jatah yang tidak bisa dikembalikan. Verifikasi tier seat tim Anda dulu sebelum menjalankan MCP secara serius. [High]
3. Nama frame biasanya jadi nama file hasil generate. `Frame 63.qml` tidak bisa di-review.

## Konvensi yang saya sarankan

    <Path>/<NN>-<NamaLayar>

- `Path` = `Mirror` atau `App`, karena satu desain dipakai untuk dua jalur.
- `NN` = urutan dua digit dalam alur, supaya frame tersusun rapi di panel layer.
- `NamaLayar` = PascalCase, sama persis dengan nama komponen QML tujuan.

Contoh: `Mirror/03-CaptureCountdown` akan menghasilkan `CaptureCountdown.qml` di `src/moodin_interface/ui/mirror/`.

## Tabel isian

Isi kolom "Nama baru" bersama rekan Anda yang membuat desain, karena hanya kalian yang tahu isi tiap frame. Kolom ketiga saya isi tebakan berdasarkan struktur `bridge.py` yang sudah ada (`Screen.CAPTURE`, `Screen.MAIN_MENU`, `Screen.RESULT`) supaya ada titik awal.

| Nama sekarang | Nama baru | Dugaan peran |
|---|---|---|
| Frame 56 | `Mirror/01-Idle` | layar tunggu, tidak ada wajah |
| Frame 57 | `Mirror/02-FaceDetected` | wajah terdeteksi, ajakan mulai |
| Frame 58 | `Mirror/03-CaptureCountdown` | progress ring pengisian buffer |
| Frame 59 | `Mirror/04-Result` | kartu mood + VA |
| Frame 60 | `Mirror/05-ResultTTV` | kartu BPM / RR |
| Frame 61 | ... | |
| Frame 62 | ... | |
| Frame 63 | ... | |
| Frame 64 | ... | |
| Frame 65 | ... | |
| Frame 66 | ... | |
| Frame 67 | ... | |
| Frame 68 | ... | |
| Frame 69 | ... | |
| Frame 70 | ... | |
| Frame 71 | `App/01-Splash` | dugaan: mulai jalur aplikasi |
| Frame 72 | `App/02-Login` | |
| Frame 73 | `App/03-Consent` | persetujuan UU PDP 27/2022 |
| Frame 74 | `App/04-Home` | |
| Frame 75 | ... | |
| Frame 76 | ... | |
| Frame 77 | ... | |
| Frame 78 | ... | |
| Frame 79 | ... | |
| Frame 80 | ... | |

## Dua permintaan lain ke rekan desain Anda

**Ukuran frame.** Frame sekarang 473x832 (rasio 0,5685), layar target 600x1024 (rasio 0,5859). Selisihnya 3%, cukup untuk membuat teks tampak sedikit gepeng kalau di-stretch, atau memunculkan bilah kosong kalau di-letterbox. Minta frame diubah tepat ke **600x1024**. Di Figma ini satu operasi: pilih frame, set W=600 H=1024, dan pastikan constraint anak-anaknya `Scale` bukan `Left/Top`.

**Ekspor SVG.** 25 SVG di ZIP kemarin tidak memuat satu pun elemen `<text>`, `font-family`, atau `font-size`; semua teks sudah di-outline jadi path. Artinya SVG itu hanya berguna sebagai gambar acuan, tidak bisa dijadikan sumber ukuran font atau warna teks. Dua pilihan: aktifkan opsi ekspor tanpa outline text, atau lewati SVG sama sekali dan ambil token langsung dari MCP `get_variable_defs`. Pilihan kedua lebih hemat waktu dan lebih akurat, asal jatah tool call mencukupi.

## Prompt siap pakai untuk Antigravity CLI (setelah rename selesai)

Jalankan satu per layar, jangan sekaligus, supaya konsumsi tool call terukur:

    Ambil token desain dan struktur dari frame Figma bernama "Mirror/04-Result"
    (gunakan get_variable_defs lalu get_metadata, jangan get_design_context).
    Jangan hasilkan React atau Tailwind. Target proyek ini PySide6 + QML.
    Keluarkan satu file Result.qml memakai Item sebagai root, ukuran 600x1024,
    dan rujuk warna lewat Theme.qml yang sudah ada, bukan literal hex.
    Tampilkan dulu rencana pemetaan node Figma ke elemen QML sebelum menulis file.
