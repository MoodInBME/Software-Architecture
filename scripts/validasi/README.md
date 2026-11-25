# `scripts/validasi/` — validasi lokal

Semua skrip menulis keluaran ke luar repo (bawaan `D:/moodin_runs/`). Peta blok diagram ke skrip ada di `docs/jejak-diagram.md`.

| Skrip                                                           | Isi                                                                                                                                               |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v1_nme_300w.py`                                                | NME 68 landmark pada 300-W: MediaPipe varian A/B, OpenFace (CSV FaceLandmarkImg), dlib HOG + shape_predictor_68, kotak YuNet + shape_predictor_68 |
| `v1_kalibrasi_yunet.py` | parameter pemetaan kotak YuNet ke gaya kotak HOG dlib, dipasang pada bagian latih 300-W; dipakai V1 lewat `--kalibrasi` |
| `waktu_landmark.py` | waktu per frame pada video 640x480 untuk dlib HOG, YuNet + dlib, dan pelacakan antarframe; dijalankan di laptop dan Jetson |
| `v2_afew_stgcn.py`                                              | ST-GCN kepala valensi-arousal pada AFEW-VA, GroupKFold per aktor                                                                                  |
| `v3_chrom_ubfcphys.py`                                          | CHROM terhadap BVP kontak UBFC-Phys, HR per jendela 60 detik                                                                                      |
| `v4_unduh_ubfcphys.py`, `v4_fitur_ubfcphys.py` | UBFC-Phys: berkas sinyal 56 subjek, pilih 10 test + 10 ctrl, video bertahap (CHROM lalu video dihapus), tabel fitur HRV jendela 60 dan 120 detik dengan baseline T1 awal |
| `v5_ekspor_onnx.py`, `v5_jetson_trt.sh`, `v5_kalibrasi_int8.py` | ekspor ONNX dan TensorRT di Jetson                                                                                                                |
| `laporan_padding_afew.py`                                       | jumlah jendela dan persen frame padding AFEW-VA                                                                                                   |

## Kandidat landmark pihak ketiga untuk V1

Tidak ada yang dipasang otomatis; semuanya opsional dan hanya untuk pembanding.

- **dlib di Windows.** PyPI tidak punya wheel `dlib` resmi untuk Windows (hanya kode sumber). Paket `dlib-bin` 20.0.1.post1 menyediakan wheel `cp312-win_amd64`; paket itu dibangun ulang oleh pihak ketiga ([alesanfra/dlib-wheels](https://github.com/alesanfra/dlib-wheels)), bukan oleh pengembang dlib. Lisensi dlib: Boost Software License. Alternatif tanpa biner pihak ketiga: memasang `dlib` dari kode sumber di WSL2 atau Jetson (butuh CMake dan kompiler C++).
- **Model `shape_predictor_68_face_landmarks.dat`** (dlib.net/files). Model ini dilatih dengan dataset iBUG 300-W. Menurut catatan pengembang dlib, lisensi dataset itu tidak mengizinkan pemakaian komersial, sehingga model ini hanya dipakai untuk riset dan perbandingan. Model tidak disimpan di repo.
- **YuNet** (`face_detection_yunet_2023mar.onnx`, repositori opencv_zoo), dimuat dengan `cv2.FaceDetectorYN`. Lisensi model perlu dicek di halaman opencv_zoo sebelum dipakai di luar perbandingan. Model tidak disimpan di repo.
- **OpenFace 2.2.0** (FaceLandmarkImg): lisensi akademik nonkomersial; dijalankan terpisah dan hanya CSV keluarannya yang dibaca.

## Kalibrasi kotak YuNet (T1.2)

Kotak YuNet lebih tinggi dan lebih ke atas daripada kotak HOG dlib, padahal shape_predictor_68 dilatih dengan kotak HOG. `v1_kalibrasi_yunet.py` mencari geser pusat (`dx`, `dy`, relatif ukuran kotak) dan skala (`sw`, `sh`) sebagai median atas pasangan kotak YuNet dan HOG pada gambar yang sama.

- Data pemasangan: bagian latih 300-W (lfpw/trainset, helen/trainset, afw). Folder `01_Indoor`, `02_Outdoor`, dan `testset` ditolak skrip supaya 600 gambar uji V1 tidak ikut dipakai.
- Urutan kerja: jalankan `v1_kalibrasi_yunet.py`, lalu V1 dengan `--kalibrasi <folder>/kalibrasi_yunet_dlib.json`. V1 lalu melaporkan `yunet_dlib` dan `yunet_dlib_kalibrasi` berdampingan.
- `waktu_landmark.py` menerima `--kalibrasi` yang sama, sehingga waktu dan akurasi diukur pada konfigurasi yang sama.

## Catatan keadilan V1

- Setiap kandidat memilih wajah yang kotaknya paling dekat dengan kotak anotasi `.pts`, lalu dihitung "cocok" bila IoU kotak landmark prediksi dan anotasi >= 0,3. NME rerata dan median hanya dihitung pada gambar yang cocok; AUC dan persen gagal dihitung atas semua gambar.
- Topologi MediaPipe FaceMesh bukan iBUG-68. Titik rahang dan alis hasil subset tidak berimpit secara definisi dengan anotasi 300-W, sehingga NME wilayah rahang dan alis MediaPipe mencampur galat deteksi dengan selisih definisi titik.
- dlib dan OpenFace dilatih dengan data latih keluarga 300-W (anotasi dan konvensi 68 titik yang sama), sedangkan 600 gambar Indoor/Outdoor adalah set uji terpisah [Sedang, perlu dicek di makalah masing-masing]. Kesamaan konvensi anotasi menguntungkan keduanya dibanding MediaPipe.
