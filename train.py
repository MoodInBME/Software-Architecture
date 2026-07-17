import torch
from torch.utils.data import Dataset, DataLoader
import os
# import FaceVideoProcessor dan STCHybridLoss di sini...

class rPPGDataset(Dataset):
    def __init__(self, data_dir):
        # 1. Daftarkan semua path file video dan file ground-truth di sini
        self.video_paths = [...] # List lokasi file mp4
        self.target_paths = [...] # List lokasi file ground-truth
        self.processor = FaceVideoProcessor()

    def __len__(self):
        # Mengembalikan total jumlah sampel data
        return len(self.video_paths)

    def __getitem__(self, idx):
        # 2. Memuat hanya 1 video dan 1 target saat fungsi ini dipanggil
        video_tensor = self.processor.process_video(self.video_paths[idx])
        target_tensor = ... # Logika untuk memuat sinyal BPM dari file CSV/TXT
        
        return video_tensor, target_tensor

def train_model():
    # ... (Setup model, device, dan optimizer) ...
    
    # 3. Gunakan DataLoader untuk mengatur antrean dan Batch Size
    # Ingat, memori GPU terbatas, atur batch_size ke 2 atau 4 pada awalnya
    dataset = rPPGDataset(data_dir='/content/dataset_aktif/')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    for epoch in range(num_epochs):
        for batch_videos, batch_targets in dataloader:
            batch_videos = batch_videos.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass -> Hitung Loss -> Backward pass -> Optimizer Step