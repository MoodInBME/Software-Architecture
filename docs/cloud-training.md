# Pelatihan di cloud (cadangan, bukan jalur utama)

Jalur utama pelatihan dan validasi adalah laptop lokal (`docs/setup-windows.md`). Cloud (Kaggle atau Google Colab) hanya cadangan bila GPU laptop tidak cukup.

## Kapan cloud boleh dipakai

Cloud berarti data disalin ke server pihak ketiga. Karena itu cloud **hanya** boleh dipakai untuk:

1. data sintetis (misalnya uji `--sanity`), atau
2. turunan dataset publik yang lisensinya jelas mengizinkan penyimpanan di layanan pihak ketiga, setelah lisensinya dibaca dan dicatat di kartu dataset (`data/dataset_cards/`).

Bila lisensi tidak menyebutnya, anggap **tidak boleh**.

## Larangan

| Data                                    | Alasan                                                                                                        | Tempat yang boleh                               |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Aff-Wild2                               | Perjanjian lisensi (EULA) melarang memindahkan data ke pihak lain; bobot hasil pelatihan tidak dipublikasikan | laptop lokal                                    |
| DISFA                                   | Ketentuan penggunaan melarang memberikan video ke pihak kedua                                                 | laptop lokal                                    |
| Video UBFC-rPPG dan UBFC-Phys           | Video wajah; tetap diproses lokal walaupun turunan numeriknya (BVP, fitur) mungkin boleh                      | laptop lokal                                    |
| Data partisipan (video, ECG, kuesioner) | ADR-14 dan protokol KEP ITS                                                                                   | laptop atau Jetson lokal, tidak pernah ke cloud |
| AFEW-VA                                 | Lisensi riset; izin penyimpanan di pihak ketiga **belum dicek**                                               | laptop lokal sampai dicek                       |

Kredensial Kaggle hanya disimpan di `%USERPROFILE%\.kaggle\kaggle.json`; tidak pernah di repo, notebook, atau log.

## Google Colab lewat WSL2

Colab CLI resmi hanya untuk Linux dan macOS, jadi di Windows dijalankan dari WSL2. Langkah berikut dijalankan sendiri oleh mahasiswa pengusul; login dilakukan di WSL2, bukan oleh agen.

```powershell
wsl --install -d Ubuntu-24.04
wsl
```

Di dalam WSL2, pasang CLI mengikuti dokumentasi resmi Colab CLI (nama paket dan perintah login **perlu dicek di dokumentasi terkini**), lalu unggah hanya data yang lolos bagian "Kapan cloud boleh dipakai".

## Kaggle lewat web

1. Buat dataset **privat** di kaggle.com berisi tensor landmark turunan (bukan video atau gambar), hanya bila lisensi sumbernya mengizinkan.
2. Buka notebook `notebooks/kaggle_latih_stgcn.ipynb`, tambahkan dataset itu lewat **Add Input**, pilih GPU, lalu jalankan.
3. Unduh bobot dari tab **Output** ke `D:/moodin_runs/`, bukan ke repo.

Notebook itu saat ini **DIBLOKIR**: sel pertama berhenti dengan galat sampai lisensi AFEW-VA dicek dan bendera di sel itu diubah.
