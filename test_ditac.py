import torch
from models.ditac import DiTAC

def run_ditac_test():
    print("--- Memulai Pengujian DiTAC ---")
    
    # 1. Inisialisasi Modul
    ditac = DiTAC(num_segments=10, a=-3.0, b=3.0)
    print("Modul DiTAC berhasil diinisialisasi.\n")
    
    # 2. Persiapan Data Uji
    x = torch.tensor([[-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0]], requires_grad=True)
    
    # 3. Uji Forward Pass
    print("[1] Menguji Forward Pass & Kondisi Batas...")
    out = ditac(x)
    
    print(f"  Input  : {x.data.numpy().round(4)}")
    print(f"  Output : {out.data.numpy().round(4)}")
    
    mask_outside = (x < -3.0) | (x > 3.0)
    if torch.allclose(x[mask_outside], out[mask_outside]):
        print("  [SUKSES] Nilai di luar batas [-3, 3] tidak berubah (Fungsi Identitas bekerja).")
    else:
        print("  [GAGAL] Nilai di luar batas ikut berubah!")

    # 4. Uji Backward Pass (Aliran Gradien)
    print("\n[2] Menguji Backward Pass (Gradient Flow)...")
    
    dummy_loss = out.sum()
    dummy_loss.backward()
    
    if ditac.theta.grad is not None:
        non_zero_grads = torch.sum(ditac.theta.grad != 0).item()
        print(f"  [SUKSES] Parameter 'theta' menerima gradien! (DiTAC bersifat trainable).")
        print(f"  -> Ada {non_zero_grads} titik theta yang mendapatkan update dari data input.")
    else:
        print("  [GAGAL] Parameter 'theta' tidak menerima gradien. Aktivasi tidak bisa dilatih.")
        
    if x.grad is not None:
        print("  [SUKSES] Input menerima gradien! Jembatan backpropagation ke layer sebelumnya aman.")
    else:
        print("  [GAGAL] Gradien terputus di dalam DiTAC. Layer sebelumnya tidak akan bisa belajar.")

if __name__ == "__main__":
    run_ditac_test()