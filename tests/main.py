import torch
from models import STCrPPG

def test_model():
    print("Mempersiapkan tensor simulasi video wajah...")
    # Parameter: Batch=2, Channel=3, Frame(T)=128, Height=128, Width=128
    B, C, T, H, W = 2, 3, 128, 128, 128
    dummy_video_batch = torch.randn(B, C, T, H, W)
    
    print(f"Bentuk Input  : {dummy_video_batch.shape}")
    
    # Inisialisasi model dari package
    print("Menginisialisasi arsitektur STC-rPPG...")
    model = STCrPPG()
    
    # Menghitung parameter model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameter: {total_params / 1e6:.2f} Juta (Sesuai paper ~0.82M)")
    
    # Melakukan forward pass
    print("Melakukan forward pass (mengalirkan data ke jaringan)...")
    predicted_signal = model(dummy_video_batch)
    
    # Sinyal output harus berupa (Batch, Time)
    print(f"Bentuk Output : {predicted_signal.shape}")
    
    assert predicted_signal.shape == (B, T), "Error: Dimensi output tidak sesuai dengan target (Batch, Time)!"
    print("Pengujian Berhasil! Dimensi arsitektur sudah saling terkalibrasi.")

if __name__ == "__main__":
    test_model()