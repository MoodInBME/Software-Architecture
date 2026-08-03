"""
MoodIn - Edge Inference Orchestrator (Jetson Orin Nano Super)

Menjalankan pipeline end-to-end secara real-time:
Kamera -> MediaPipe FaceMesh -> Buffer Temporal ->
[ST-GCN (Wajah) & STC-rPPG (Fisik)] -> LateFusionVABridge -> MQTT (UI/Firebase)
"""
import cv2
import json
import time
import torch
import numpy as np
import collections
import paho.mqtt.client as mqtt
import mediapipe as mp
import os
import sys

# Tambahkan root directory ke sys.path agar import lokal berjalan lancar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.st_gcn_graph_tran.model import STGCN_GraphTransformer
from models.stc_rppg.model import STCrPPG
from models.fusion.late_fusion import LateFusionVABridge

# Konfigurasi Hardware (Jetson Orin Nano Super - IMX477)
CAMERA_SOURCE = 0  
MQTT_BROKER = "localhost" 
MQTT_TOPIC = "moodin/inference/result"

# Konfigurasi Path Bobot
WEIGHT_STGCN = "weights/dummy_stgcn.pth"
WEIGHT_STCRPPG = "weights/dummy_stcrppg.pth"

# Setup MediaPipe untuk visualisasi sci-fi UI
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def gstreamer_pipeline(capture_width=1280, capture_height=720, framerate=30):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

def main():
    print("=== Menginisialisasi MoodIn Edge Inference ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Menggunakan: {device}")

    # 1. Inisialisasi Arsitektur Deep Learning
    print("[Model] Memuat ST-GCN dan STC-rPPG...")
    stgcn = STGCN_GraphTransformer(
        in_channels=2, num_class=5, num_node=68, graph_strategy="spatial", 
        max_hop=1, t_kernel=9, transformer_layers=2, transformer_heads=8, 
        transformer_ff=512, dropout=0.3, head_type="va"
    ).to(device)
    stgcn.eval()
    
    stc_rppg = STCrPPG().to(device)
    stc_rppg.eval()
    
    fusion_bridge = LateFusionVABridge(w_face_arousal=0.7, w_physio_arousal=0.3)

    # 2. Muat Bobot (Gunakan dummy jika bobot asli belum dilatih)
    if os.path.exists(WEIGHT_STGCN):
        stgcn.load_state_dict(torch.load(WEIGHT_STGCN, map_location=device, weights_only=True))
        print(f"[Bobot] ST-GCN dimuat dari {WEIGHT_STGCN}")
    else:
        print("[Bobot] WARNING: Bobot ST-GCN tidak ditemukan! Menggunakan bobot inisialisasi acak.")

    if os.path.exists(WEIGHT_STCRPPG):
        stc_rppg.load_state_dict(torch.load(WEIGHT_STCRPPG, map_location=device, weights_only=True))
        print(f"[Bobot] STC-rPPG dimuat dari {WEIGHT_STCRPPG}")
    else:
        print("[Bobot] WARNING: Bobot STC-rPPG tidak ditemukan! Menggunakan bobot inisialisasi acak.")

    # 3. Setup MQTT Publisher
    client = mqtt.Client(client_id="Jetson_Inference")
    try:
        client.connect(MQTT_BROKER, 1883, 60)
        client.loop_start()
        print(f"[MQTT] Terhubung ke broker {MQTT_BROKER}")
    except Exception as e:
        print(f"[MQTT] Gagal terhubung ke broker: {e}. Melanjutkan tanpa MQTT.")

    # 4. Inisialisasi Kamera
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[Kamera] Pipeline CSI gagal. Fallback ke USB Webcam lokal...")
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        
    if not cap.isOpened():
        print("[Kamera] ERROR: Tidak ada kamera yang terdeteksi!")
        return

    # Buffer Temporal
    buffer_stgcn = collections.deque(maxlen=150)
    buffer_rppg = collections.deque(maxlen=128)
    
    print("[Inference] Mulai membaca frame. Tekan 'q' pada jendela untuk keluar.")
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            display_frame = frame.copy()
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=display_frame, landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    
                    # 468 landmarks -> 68 iBUG (Mocking the mapping logic for inference speed)
                    # Di produksi, gunakan MEDIAPIPE_TO_IBUG68 dari labels.py
                    coords = [[lm.x, lm.y] for lm in face_landmarks.landmark[:68]] 
                    buffer_stgcn.append(coords)
                    
                    # RPPG butuh RGB image (resized 128x128)
                    resized_frame = cv2.resize(frame_rgb, (128, 128))
                    buffer_rppg.append(resized_frame)
            else:
                buffer_stgcn.clear()
                buffer_rppg.clear()
            
            # --- MODEL FORWARD PASS ---
            if len(buffer_stgcn) == 150 and len(buffer_rppg) >= 128:
                try:
                    # Siapkan Tensor ST-GCN: (N, C, T, V) = (1, 2, 150, 68)
                    stgcn_input = torch.tensor(buffer_stgcn, dtype=torch.float32).unsqueeze(0)
                    stgcn_input = stgcn_input.permute(0, 3, 1, 2).to(device) # N, C, T, V
                    
                    # Siapkan Tensor STC-rPPG: (N, C, T, H, W) = (1, 3, 128, 128, 128)
                    # Format yang benar untuk STCrPPG: (N, C, T, H, W)
                    rppg_input = np.array(list(buffer_rppg)[-128:]) # Ambil 128 terakhir
                    rppg_input = torch.tensor(rppg_input, dtype=torch.float32).unsqueeze(0)
                    rppg_input = rppg_input.permute(0, 4, 1, 2, 3).to(device) # N, C, T, H, W
                    
                    with torch.no_grad():
                        # Inference Cabang Ekspresi (Wajah)
                        mood_logits, (val_face, aro_face) = stgcn(stgcn_input)
                        val_face = val_face.item()
                        aro_face = aro_face.item()
                        
                        # Inference Cabang Fisiologis (rPPG)
                        # Model mengembalikan sinyal 1D sepanjang waktu (T)
                        rppg_signal = stc_rppg(rppg_input)
                        
                        # Ekstraksi BPM sederhana dari sinyal FFT (Dummy logic for inference mapping)
                        # Dalam produksi gunakan postprocessing.rPPGProcessor
                        bpm = float(torch.mean(rppg_signal).item() * 100 + 80) 
                        aro_physio = (bpm - 70) / 30.0 # Normalisasi kasar BPM -> Arousal
                        
                        # Fusi V-A Bridge
                        fusion_result = fusion_bridge.fuse(val_face, aro_face, aro_physio)
                        final_mood = fusion_result['mood_label']
                    
                    # Kirim hasil inferensi via MQTT
                    payload = {
                        "mood_label": final_mood,
                        "valence": val_face,
                        "arousal": fusion_result['arousal'],
                        "heart_rate_bpm": bpm,
                        "physio_used": True,
                        "timestamp": time.time()
                    }
                    client.publish(MQTT_TOPIC, json.dumps(payload))
                    
                    cv2.putText(display_frame, f"Mood: {final_mood} | BPM: {bpm:.1f}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Stride 15 frame agar tidak freeze
                    for _ in range(15): 
                        buffer_stgcn.popleft()
                        
                except Exception as e:
                    print(f"[Error Inference]: {e}")
                    buffer_stgcn.clear()
            
            cv2.imshow('MoodIn - Real-time Landmark (50% Progress Demo)', display_frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()

if __name__ == "__main__":
    main()
