# `models/st_gcn_graph_tran`

> **Catatan fase TA:** folder ini berisi jalur wajah versi GEMASTIK (ST-GCN + Graph Transformer, jendela 150 _frame_, 68 titik yang dipetakan dari MediaPipe 468). Rancangan jalur wajah TA berbeda dan menunggu keputusan bimbingan (milestone M4). Rencana kerja di bawah adalah rencana era GEMASTIK dan tidak dilanjutkan.

Cabang **ekspresi** MoodIn. Memetakan urutan 150 frame landmark wajah iBUG-68 ke sepasang skalar valence dan arousal.

Dokumen ini adalah rencana kerja subfolder, bukan sekadar deskripsi. Setiap berkas punya status dan pekerjaan yang tersisa.

---

## 1. Berkas dan status

| Berkas | Isi | Status | Pekerjaan tersisa |
|---|---|---|---|
| `graph.py` | `FaceGraph(num_node=68, center=33)` -> adjacency `(K, V, V)` | stabil | dokumentasikan definisi tepi; hasilkan Gambar 2.3 topologi |
| `model.py` | `STGCN_GraphTransformer` | stabil | tidak ada |
| `train.py` | loop pelatihan, `build_model(config)`, `--sanity` | **perlu diubah** | tambah `--folds` dan `--fold` (lihat bagian 4) |
| `config.yaml` | hiperparameter | stabil | pisahkan path lokal ke `config.local.yaml` |
| `losses.py` | `CCCLoss`, `DualLoss(alpha=1.0, beta=0.5)` | stabil | uji sensitivitas beta terhadap penciutan prediksi |
| `__init__.py` | ekspor publik | stabil | tidak ada |

---

## 2. Arsitektur

```
masukan (N, 2, 150, 68)
   |
   v
BatchNorm atas simpul
   |
   v
9x blok ST-GCN            konvolusi spasial berbobot adjacency partisi
   |                      + konvolusi temporal kernel 9
   v
pooling spasial -> (N, T', C)
   |
   v
2x layer Transformer      d_ff 512, 8 head, dropout 0.3
   |
   v
pooling temporal
   |
   v
head VA -> (N, 2), tanh -> [-1, 1]
```

**Kenapa hibrida.** ST-GCN unggul menangkap ko-aktivasi lokal antar-landmark yang bertetangga secara anatomis, tetapi konvolusi temporalnya berjangkauan terbatas. Transformer menambahkan atensi jangka panjang atas rangkaian waktu setelah dimensi spasial diringkas - misalnya untuk membedakan senyum yang muncul perlahan dari senyum yang tersentak.

| Konfigurasi kunci | Nilai |
|---|---|
| `in_channels` | 2 (x, y ternormalisasi) |
| `num_node` / `center` | 68 / 33 (ujung hidung) |
| `graph_strategy` / `max_hop` | `spatial` / 1 |
| `t_kernel` | 9 |
| `transformer_layers` / `heads` / `ff` | 2 / 8 / 512 |
| `head_type` | `va` |
| Parameter | ~0,82 M -> 3,3 MB FP32, 0,82 MB INT8 |

---

## 3. Kontrak data

**Masukan.** Tensor `(2, 150, 68)` float32. Landmark dipusatkan di simpul 33 dan diskalakan jarak antar-mata, sehingga invarian terhadap translasi dan skala tetapi **tidak** invarian terhadap rotasi. Rotasi kepala ditangani di preprocessing.

**Peta indeks iBUG-68.** jaw 0-16, alis kanan 17-21, alis kiri 22-26, batang hidung 27-30, dasar hidung 31-35, mata kanan 36-41, mata kiri 42-47, bibir luar 48-59, bibir dalam 60-67. Peta dari MediaPipe ada di `MEDIAPIPE_TO_IBUG68`.

> **Belum diverifikasi.** Peta itu disusun manual dan belum pernah dicek numerik. Rencana: ambil LCK+ dari IEEE DataPort, cocokkan landmark hasil MediaPipe dengan landmark LCK+ pada wajah yang sama lewat Procrustes. RMSE ternormalisasi di atas 0,05 berarti peta salah. Ini prasyarat untuk setiap angka yang dilaporkan.

**Keluaran.** `dict(va=(N, 2))`, urutan valence lalu arousal, rentang `[-1, 1]`. Kuantisasi ke mood dilakukan **di hilir** oleh `common.labels.va_to_mood` dengan `NEUTRAL_RADIUS = 0.2`. Model tidak pernah mengeluarkan label kelas.

---

## 4. Perubahan wajib pada `train.py`

CV subject-independent tidak bisa dijalankan sampai dua argumen ini ada:

```python
parser.add_argument('--folds', type=str, default=None,
                    help='JSON berisi [{"fold": k, "train": [...], "test": [...]}]')
parser.add_argument('--fold', type=int, default=None)
parser.add_argument('--out_dir', type=str, default='checkpoints')

# setelah dataset penuh dibangun
if args.folds is not None and args.fold is not None:
    with open(args.folds) as fh:
        records = json.load(fh)
    rec = next(r for r in records if r['fold'] == args.fold)
    train_ds = Subset(full_ds, rec['train'])
    val_ds = Subset(full_ds, rec['test'])
else:
    train_ds, val_ds = random_split_by_ratio(full_ds, args.val_split)
```

Sekitar 15 baris. Simpan checkpoint terbaik ke `{out_dir}/stgcn_va_best.pth` beserta `config` di dalamnya, karena notebook memanggil `build_model(ckpt['config'])`.

---

## 5. Rencana kerja berurutan

| # | Tugas | Prasyarat | Keluaran |
|---|---|---|---|
| 1 | Tambah `--folds`, `--fold`, `--out_dir` | - | `train.py` siap CV |
| 2 | Verifikasi Procrustes `MEDIAPIPE_TO_IBUG68` | LCK+ terunduh | laporan RMSE |
| 3 | Pastikan manifest memuat field `subject` | - | manifest v2 |
| 4 | Latih fold 0, periksa kurva kerugian | 1, 3 | checkpoint fold 0 |
| 5 | CV 10 fold penuh | 4 | 10 checkpoint |
| 6 | Inferensi OOF dan seluruh metrik | 5 | `oof_predictions.npz` |
| 7 | Uji falsifikasi baseline | 6 | `baseline_comparison.json` |
| 8 | Ekspor ONNX dan asersi paritas | 5 | `stgcn_va.onnx` |
| 9 | TensorRT INT8 di Jetson, `tegrastats` | 8, perangkat | Tabel 3 proposal |
| 10 | Kalibrasi bobot fusi | 6 | `w1`, `w2` |

Langkah 1-8 seluruhnya dijalankan oleh `notebooks/arsip-gemastik/ck_plus_stgcn_kaggle.ipynb`.

---

## 6. Catatan tentang bobot pra-terlatih

Tidak ada checkpoint publik yang bisa dipakai langsung. Semua checkpoint ST-GCN yang beredar dilatih pada **skeleton tubuh** (NTU-RGB+D 25 sendi, Kinetics 18 sendi) dan bentuk adjacency-nya tidak kompatibel dengan 68 simpul wajah.

| Sumber | Kegunaan | Peringatan |
|---|---|---|
| `PanosAntoniadis/emotion-gcn` | rujukan arsitektur | **GPL-3.0** - jangan menyalin kode ke basis kode yang akan dikomersialkan |
| `xjtupanda/AUW-GCN` | rujukan graf AU | - |
| `face-analysis/emonet` | guru untuk pseudo-label VA | lisensi riset saja |
| LCK+ (IEEE DataPort) | verifikasi landmark, jalur cepat | tampaknya statis, bukan urutan |

**Plan A:** latih dari nol pada urutan CK+. Kurang dari 1 jam di T4. Lebih cepat daripada mencari bobot yang tidak ada.

**Plan B:** kalau CK+ terlalu kecil, jalankan EmoNet pada video tak berlabel untuk menghasilkan pseudo-label VA, lalu distilasi ke ST-GCN. Ini menaikkan jumlah data tetapi mewarisi bias EmoNet - sebutkan di batasan.

---

## 7. Mode kegagalan yang harus diawasi

| Gejala | Kemungkinan sebab | Tindakan |
|---|---|---|
| `accuracy - macro_f1 > 0.15` | runtuh ke kelas mayoritas | class weighting, periksa confusion matrix |
| Std prediksi jauh di bawah std label | CCCLoss didominasi suku varians | naikkan `beta` di `DualLoss` |
| Akurasi sangat tinggi di fold 0 | subjek bocor | periksa asersi grup di notebook bagian 7 |
| Cabang temporal tidak belajar apa pun | mirror CK+ hanya berisi frame puncak | ganti sumber dataset |
| ONNX menyimpang lebih dari 1e-4 | operator tidak didukung opset 17 | jangan pakai untuk TensorRT |
