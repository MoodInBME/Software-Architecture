import cv2
import dlib
import torch
import numpy as np

class FaceVideoProcessor:
    """
    Pipeline pra-pemrosesan video wajah untuk rPPG
    """
    def __init__(self, target_size=(128, 128), seq_length=128):
        # Inisialisasi detektor wajah Dlib
        self.detector = dlib.get_frontal_face_detector()
        self.target_size = target_size
        self.seq_length = seq_length

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened() and len(frames) < self.seq_length:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Konversi warna dari BGR (standar OpenCV) ke RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Deteksi wajah pada frame
            faces = self.detector(frame_rgb, 1)
            
            if len(faces) > 0:
                # Ambil kotak pembatas (bounding box) wajah pertama yang terdeteksi
                face = faces[0]
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                
                # Pastikan koordinat tidak bernilai negatif
                x, y = max(0, x), max(0, y)
                
                # Crop area wajah
                face_crop = frame_rgb[y:y+h, x:x+w]
                
                # Menghindari error jika wajah terdeteksi di ujung frame (lebar/tinggi 0)
                if face_crop.size == 0:
                    continue
                
                # Resize menjadi 128x128
                face_resized = cv2.resize(face_crop, self.target_size)
                
                # Normalisasi nilai piksel ke rentang [0, 1]
                face_normalized = face_resized.astype(np.float32) / 255.0
                frames.append(face_normalized)
            else:
                # Lewati frame jika tidak ada wajah yang terdeteksi
                continue
                
        cap.release()
        
        # Jika video terlalu pendek, lakukan padding (mengisi sisa frame dengan nol)
        while len(frames) < self.seq_length:
            frames.append(np.zeros((*self.target_size, 3), dtype=np.float32))
            
        # Susun array numpy dengan bentuk: (Time, Height, Width, Channel)
        video_array = np.array(frames)
        
        # Mengubah bentuk array menjadi format PyTorch: (Channel, Time, Height, Width)
        # Sumbu berpindah dari (T, H, W, C) -> (C, T, H, W)
        tensor_video = torch.tensor(np.transpose(video_array, (3, 0, 1, 2)))
        
        # Tambahkan dimensi Batch di indeks 0 sehingga menjadi (1, C, T, H, W)
        return tensor_video.unsqueeze(0)

# ==========================================
# PENGUJIAN PIPELINE DATA
# ==========================================
if __name__ == "__main__":
    # Ganti 'sample.mp4' dengan nama file video wajah Anda
    video_file = "sample.mp4" 
    
    print(f"Memproses video: {video_file}...")
    processor = FaceVideoProcessor(target_size=(128, 128), seq_length=128)
    
    try:
        # Eksekusi proses cropping dan konversi tensor
        model_input = processor.process_video(video_file)
        
        print(f"Pemrosesan selesai!")
        print(f"Bentuk tensor siap latih: {model_input.shape}")
        
        # Verifikasi bahwa dimensi sesuai dengan target (Batch=1, C=3, T=128, H=128, W=128)
        assert model_input.shape == (1, 3, 128, 128, 128), "Dimensi tensor tidak sesuai!"
        
    except Exception as e:
        print(f"Terjadi kesalahan saat memproses video: {e}")
        print("Pastikan file 'sample.mp4' ada di direktori yang sama.")