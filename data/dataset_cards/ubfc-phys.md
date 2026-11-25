# Kartu Dataset — UBFC-Phys

## Ringkasan

- **Nama:** UBFC-Phys, basis data multimodal untuk studi psikofisiologis stres sosial [1].
- **Isi:** video wajah 56 partisipan selama tiga tugas (istirahat, berbicara, dan aritmetika), disertai sinyal fisiologis dari gelang di pergelangan tangan (BVP dan EDA).
- **Peran di MoodIn:** dataset utama jalur fisiologis pada tahap Pra-TA (milestone M1).
- **Status:** tersedia bagi mahasiswa pengusul.

Struktur berkas, laju _frame_ video, dan laju sampel sinyal **dibaca langsung dari berkas dataset**, bukan diasumsikan. Kode lama di repo ini mengasumsikan 30 fps (UBFC-rPPG); asumsi itu tidak boleh dibawa ke UBFC-Phys tanpa pengecekan.

## Akses dan lisensi

- Akses diperoleh dari pengelola dataset sesuai syarat mereka. Pemakaian dan sitasi mengikuti syarat tersebut.
- **Data mentah maupun turunannya** (video, sinyal, _landmark_, fitur, _cache_) **tidak boleh di-commit** ke repo mana pun di MoodInBME.

## Cara pakai di repo

`data/loaders/ubfc_phys.py` membaca BVP 64 Hz dan kelompok `test`/`ctrl` dari `info_sN.txt`, memeriksa kelengkapan subjek, dan memilih subjek seimbang per kelompok dengan seed tetap. Pemetaan tugas ke tiga kelas stres (T1 Normal; T2/T3 ctrl Stres Ringan; T2/T3 test Stres Akut) adalah asumsi penelitian, bukan label dataset. Skrip `scripts/validasi/v4_unduh_ubfcphys.py` mengambil berkas sinyal, memilih 10 test + 10 ctrl, lalu memproses video satu per satu dan menghapusnya; `v4_fitur_ubfcphys.py` membuat tabel fitur HRV per jendela. Semua keluaran di luar repo.

## Sitasi

[1] R. Meziati Sabour, Y. Benezeth, P. De Oliveira, J. Chappé, dan F. Yang, "UBFC-Phys: A multimodal database for psychophysiological studies of social stress," _IEEE Transactions on Affective Computing_, vol. 14, no. 1, hal. 622–636, 2023, doi: [10.1109/TAFFC.2021.3056960](https://doi.org/10.1109/TAFFC.2021.3056960).
