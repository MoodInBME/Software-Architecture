# Rencana Arsitektur & Skrip ST-GCN (MoodIn)

Dokumen perencanaan cabang klasifikasi mood berbasis graf landmark wajah
(ST-GCN + Graph-Transformer), dan bagaimana ia terhubung ke cabang STC-rPPG
dan modul fusion.

## 1. Peran dalam sistem

MoodIn memakai tiga model:

1. **STC-rPPG** -> sinyal fisiologis (detak jantung / rPPG) dari video wajah.
2. **ST-GCN + Graph-Transformer** (dokumen ini) -> pola ekspresi wajah dari
   urutan landmark -> 5 kelas mood.
3. **Fusion (dual-head/late fusion)** -> menggabungkan (1) & (2) menjadi
   keputusan mood akhir + skor kepercayaan.

Kontrak keluaran ke UI (repo Interface-Display) memakai 5 kelas:
**Ceria (Q1), Cemas (Q2), Tertekan (Q3), Rileks (Q4), Netral**.

## 2. Representasi input

- Ekstraksi landmark: MediaPipe FaceMesh (468 titik) per frame.
- Reduksi ke subset **68 titik** (konvensi iBUG/dlib) sebagai node graf, agar
  ringan untuk edge inference. `graph.py` memakai topologi ini secara default.
- Tensor model: `(N, C, T, V)` dengan `C=2` (x,y) atau `3` (+confidence),
  `T=150` frame (~5 dtk @ 30 FPS), `V=68`.
- Normalisasi: pusatkan ke titik hidung (node 33), skala terhadap jarak
  antar-mata agar invarian terhadap jarak wajah ke kamera (penting untuk
  skenario cermin, terkait W11 di proposal).

## 3. Arsitektur

    input (N, C, T, V)
      -> data BatchNorm
      -> STGCNBlock x6 : C->64->64->128(stride2)->128->256(stride2)->256
      -> pooling spasial (rata-rata node)  => (N, 256, T')
      -> Temporal Transformer (2 layer, 8 head)
      -> global temporal pooling            => (N, 256)
      -> Linear                             => (N, 5)

- **Spatial graph conv** memakai partisi `spatial` (root/centripetal/
  centrifugal) seperti Yan et al. (AAAI 2018).
- **Temporal Transformer** menambah dependensi temporal jangka panjang di atas
  fitur ST-GCN (selaras referensi "ST-GCN + Graph-Tran", Zhao 2023 di README).
- Aktivasi default ReLU; bisa ditukar DiTAC agar konsisten dengan STC-rPPG
  (eksperimen opsional).

## 4. Dataset & harmonisasi label

Tidak ada satu dataset yang menyediakan semua sinyal. Rencana peran:

| Peran | Kandidat dataset | Catatan |
|---|---|---|
| FER temporal (video) | DFEW, FERV39k, Oulu-CASIA, CK+ | untuk dinamika ekspresi |
| Dimensional VA in-the-wild | Aff-Wild2, AffectNet/AffectNet+ | satu-satunya dengan expr+VA(+AU) in-the-wild = Aff-Wild2 |
| Action Unit (opsional) | DISFA, BP4D/BP4D+ | intensitas AU 0-5 |
| Validasi wajah+fisiologi+VA | MAHNOB-HCI, DEAP | untuk uji lintas-modal |

**Strategi harmonisasi (lihat `labels.py`):**

- Dataset dimensional: `va_to_mood(valence, arousal)` -> kuadran VA + zona
  Netral (radius kecil di sekitar origin).
- Dataset kategorikal: `categorical_to_mood(label)` (peta emosi -> mood).
- **Isu terbuka**: 'surprise' ambigu (arousal tinggi, valence netral).
  Perlu keputusan tim: masuk Ceria, Cemas, atau dibuang.
- **Bias**: AffectNet/RAF-DB punya label-noise ~10-20% dan skew demografi
  (dewasa muda, kulit terang). Rencanakan validasi lintas-dataset & pelaporan
  per-subkelompok bila memungkinkan (terkait W2).

Akses dataset (MAHNOB-HCI, DISFA, BP4D, Aff-Wild2, DFEW) umumnya butuh
pengajuan akademik/EULA -> ajukan lebih awal karena bisa memakan waktu.

## 5. Pelatihan

- Loss: CrossEntropy (+ class weight bila tak seimbang).
- Optimizer: AdamW, cosine schedule, `lr=1e-3`, `weight_decay=1e-4`.
- Augmentasi landmark: jitter posisi, rotasi kecil, temporal crop/scaling,
  horizontal flip (dengan penukaran indeks kiri-kanan yang benar).
- Validasi: split lintas-subjek (subject-independent) agar tidak bocor.
- Metrik: akurasi, F1 makro, confusion matrix (lihat `evaluation/metrics.py`).

## 6. Deployment (target Jetson Orin Nano)

- Ekspor ONNX -> TensorRT (FP16) untuk inferensi edge.
- Precedent: NVIDIA TAO PoseClassificationNet berbasis ST-GCN -> jalur ST-GCN
  di TensorRT sudah terbukti.
- Target anggaran latensi bersama STC-rPPG + fusion @ 30 FPS; ukur dengan
  `deployment/benchmark_latency.md`.

## 7. Integrasi lintas-repo

- **MoodIn-master**: hub/dokumentasi; kontrak IPC pusat.
- **Hardware-Connections**: kamera + board (Jetson) + pencahayaan.
- **Interface-Display**: menerima hasil mood via MQTT
  (`moodin/inference/result`: mood_id, mood_label, confidence, valence,
  arousal, heart_rate_bpm, hrv_ms, session_id).
- Cabang ini menghasilkan `mood_id`/`mood_label` + logit; modul fusion
  menggabungkannya dengan HR/HRV dari STC-rPPG.

## 8. Milestone

1. [x] Kerangka model ST-GCN + Transformer (repo ini).
2. [ ] Implementasi `data/loaders/dataset_loader.py` + augmentasi.
3. [ ] Pipeline ekstraksi landmark MediaPipe -> subset 68 titik.
4. [ ] Pelatihan baseline pada 1 dataset (mis. DFEW atau CK+).
5. [ ] Harmonisasi multi-dataset + validasi lintas-subjek.
6. [ ] Integrasi fusion dengan STC-rPPG.
7. [ ] Ekspor ONNX/TensorRT + benchmark di Jetson.

## 9. Keputusan terbuka (perlu konfirmasi tim)

- 5 kelas via **kuadran VA** (dipakai default) vs pemetaan kategorikal murni.
- Jumlah node graf: 68 (default) vs subset MediaPipe lain.
- Sertakan Temporal Transformer (default) atau ST-GCN murni sebagai baseline
  pembanding.
- Penanganan kelas 'surprise' dan definisi zona 'Netral' (radius).
