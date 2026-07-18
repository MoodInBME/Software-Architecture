import os
import zipfile
import torch
import shutil
# Asumsi: Anda memiliki fungsi atau kelas untuk memproses video
# from models.face_processor import extract_face_tensor 

def ekstrak_dan_proses_dataset(zip_path, output_dir, temp_dir="/content/temp_video"):
    """
    Fungsi untuk mengekstrak video satu per satu, memprosesnya, lalu menghapusnya.
    """
    # Buat direktori output dan temporary jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Membuka arsip: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Cari semua file video di dalam arsip zip
        video_files = [f for f in zip_ref.namelist() if f.endswith(('.mp4', '.avi'))]
        
        print(f"Ditemukan {len(video_files)} video. Memulai pemrosesan...")
        
        for file_path_in_zip in video_files:
            try:
                # 1. Ekstrak 1 video ke folder temporary
                zip_ref.extract(file_path_in_zip, path=temp_dir)
                extracted_video_path = os.path.join(temp_dir, file_path_in_zip)
                
                # Dapatkan nama dasar file (misal: 'vid_subject1')
                base_name = os.path.basename(file_path_in_zip).split('.')[0]
                print(f"Memproses: {base_name}...")
                
                # 2. Proses video menjadi Tensor (PANGGIL FUNGSI ANDA DI SINI)
                # Contoh: tensor_wajah = extract_face_tensor(extracted_video_path)
                
                # (Bagian ini hanya dummy/contoh tensor agar kode bisa jalan saat dites)
                tensor_wajah = torch.randn(3, 128, 128, 128) 
                
                # 3. Simpan Tensor ke format .pt di folder output
                save_path = os.path.join(output_dir, f"{base_name}.pt")
                torch.save(tensor_wajah, save_path)
                
                # 4. Hapus video mentah untuk menghemat ruang disk
                os.remove(extracted_video_path)
                
            except Exception as e:
                print(f"Terjadi kesalahan pada {file_path_in_zip}: {e}")
                
    # Bersihkan folder temporary setelah semua selesai
    shutil.rmtree(temp_dir)
    print("Semua data berhasil diproses dan disk tetap aman!")

# ==========================================
# Cara Menjalankan
# ==========================================
if __name__ == "__main__":
    # Ganti dengan path file ZIP dataset Anda yang ada di Colab
    FILE_ZIP_DATASET = "/content/ubfc_dataset.zip" 
    FOLDER_HASIL_TENSOR = "/content/dataset_tensor_ringan"
    
    ekstrak_dan_proses_dataset(FILE_ZIP_DATASET, FOLDER_HASIL_TENSOR)
