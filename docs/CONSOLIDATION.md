# Paket Perbaikan Struktur (poin 1-5)

Status tiap poin:

| Poin | Isu | Cara perbaikan | Status di paket ini |
|---|---|---|---|
| 1 | `models/stc_rppg/` punya `model.py` DAN `stc_rppg.py` | Pilih satu kanonik, hapus duplikat | Langkah git (butuh isi file Anda) |
| 2 | `__init__.py` hilang di banyak paket | Tambah file kosong/aditif | **File jadi disertakan** |
| 3 | MediaPipe 468 vs graf 68 | Peta subset eksplisit | **File jadi: `landmark_map.py`** |
| 4 | `tests/main.py` & `tests/test_models.py` duplikat | Hapus/rename `main.py` | Langkah git (butuh isi file Anda) |
| 5 | 2 file preprocessing tumpang tindih | Gabung jadi satu | Langkah git (butuh isi file Anda) |

> Poin 1, 4, 5 = penghapusan/penggabungan file yang sudah ada. Saya TIDAK menimpanya
> secara buta agar kode teman Anda aman. Ikuti langkah git di bawah (aman & reversibel).

---

## Cara pasang file aditif (poin 2 & 3)

Salin isi ZIP ini ke root repo (mempertahankan struktur folder). File yang ditambahkan:

    models/__init__.py
    models/fusion/__init__.py
    data/__init__.py
    data/loaders/__init__.py
    data/preprocessing/__init__.py          # mengekspor helper landmark
    data/preprocessing/landmark_map.py      # peta 468 -> 68
    tests/__init__.py

> JANGAN menimpa `models/stc_rppg/__init__.py` dan
> `models/st_gcn_graph_tran/__init__.py` yang SUDAH ada. Paket ini sengaja tidak
> menyertakan keduanya.

Setelah disalin, uji dari root repo:

    python -c "from data.preprocessing import subset_to_ibug68; print('import OK')"
    python data/preprocessing/landmark_map.py    # cetak bentuk (150,68,2) + OK

---

## Poin 3 - integrasi peta landmark

Di `landmark_extraction_mediapipe.py`, setelah mengambil 468 landmark FaceMesh,
konversikan ke 68 sebelum menyusun tensor `(N, C, T, V)` untuk ST-GCN:

    from data.preprocessing.landmark_map import subset_to_ibug68

    # coords_468: np.ndarray bentuk (T, 468, 2)  -> (x, y) ternormalisasi
    coords_68 = subset_to_ibug68(coords_468)      # -> (T, 68, 2)
    # lalu transpose ke (C, T, V) sesuai kebutuhan model:
    x = coords_68.transpose(2, 0, 1)              # (2, T, 68)

**Verifikasi visual sekali** (wajib, karena pemetaan Medium confidence): overlay 68
titik hasil subset ke wajah sampel; pastikan urutannya cocok dengan `graph.py`.

---

## Poin 1 - duplikat `model.py` vs `stc_rppg.py`

1. Temukan file mana yang mendefinisikan kelas rakitan:

        grep -n "class STCrPPG" models/stc_rppg/model.py models/stc_rppg/stc_rppg.py

2. **Pertahankan** file yang berisi `class STCrPPG` (rekomendasi: `model.py`).
3. Pastikan `models/stc_rppg/__init__.py` mengekspor dari file kanonik:

        from .model import STCrPPG          # sesuaikan bila kanoniknya stc_rppg.py

4. Perbarui import yang menunjuk ke file yang akan dihapus:

        grep -rn "stc_rppg.stc_rppg\|from .stc_rppg import" .
        # ganti menjadi  models.stc_rppg.model  /  from .model import

5. Hapus duplikatnya:

        git rm models/stc_rppg/stc_rppg.py    # atau model.py, sesuai keputusan

6. Uji: `python main.py` (atau `pytest tests/`).

> Jika KEDUANYA memuat isi berbeda (mis. `model.py` = rakitan, `stc_rppg.py` =
> definisi lama), gabungkan manual dulu sebelum `git rm`. Kirim kedua file kalau
> mau saya yang menggabungkan.

---

## Poin 4 - duplikat `tests/main.py` vs `tests/test_models.py`

1. Bandingkan cakupannya:

        diff <(grep -o "STCrPPG\|assert\|shape" tests/main.py | sort -u) \
             <(grep -o "STCrPPG\|assert\|shape" tests/test_models.py | sort -u)

2. Jika `test_models.py` sudah mencakup smoke test `main.py`:

        git rm tests/main.py

   Jika ingin menyimpan `main.py` sebagai skrip manual (bukan koleksi pytest),
   ganti nama agar tak berprefiks `test_`:

        git mv tests/main.py tests/smoke_stc_rppg.py

> Catatan: file bernama `main.py` di dalam `tests/` membingungkan (bukan entry
> point, bukan gaya pytest). Salah satu dari dua opsi di atas menghilangkan ambiguitas.

---

## Poin 5 - duplikat preprocessing MediaPipe

Dua file kemungkinan tumpang tindih:
`data/preprocessing/landmark_extraction_mediapipe.py` dan
`data/preprocessing/preprocessing_mediapipe.py`.

1. Bandingkan fungsi/kelas yang didefinisikan masing-masing:

        grep -nE "^(def|class) " data/preprocessing/landmark_extraction_mediapipe.py
        grep -nE "^(def|class) " data/preprocessing/preprocessing_mediapipe.py

2. **Pertahankan** `landmark_extraction_mediapipe.py` (nama sesuai README).
   Pindahkan fungsi unik dari `preprocessing_mediapipe.py` ke situ (mis. helper
   crop ROI untuk rPPG), lalu integrasikan `subset_to_ibug68` (poin 3).
3. Perbarui import yang memakai file lama:

        grep -rn "preprocessing_mediapipe" .

4. Hapus duplikatnya:

        git rm data/preprocessing/preprocessing_mediapipe.py

> Sisakan `clahe_roi.py` (normalisasi pencahayaan ROI) sebagai file terpisah -
> tanggung jawabnya beda dan tidak duplikat.

---

## Ringkas urutan eksekusi

    # 1. salin file aditif (poin 2 & 3) ke root repo
    # 2. python -c "from data.preprocessing import subset_to_ibug68; print('OK')"
    # 3. selesaikan poin 1, 4, 5 dengan langkah git di atas
    # 4. pytest tests/   (validasi akhir; butuh torch terpasang)
