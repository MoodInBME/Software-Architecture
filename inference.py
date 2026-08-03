"""
MoodIn - Edge Inference Orchestrator (Jetson Orin Nano Super)

Menjalankan pipeline end-to-end secara real-time:
Kamera -> MediaPipe FaceMesh (Visualisasi UI) -> Buffer Temporal ->
[ST-GCN (Wajah) & STC-rPPG (Fisiologis)] -> LateFusionVABridge -> MQTT (UI/Firebase)

Desain ini dibuat agar saat presentasi progres 50%, Anda bisa mendemonstrasikan
garis-garis landmark futuristik (ala salvion.ai) secara real-time di layar, 
bahkan jika model Deep Learning masih menggunakan bobot acak (dummy).
"""
import cv2
import json
import time
import torch
import numpy as np
import collections
import paho.mqtt.client as mqtt
import mediapipe as mp

# Impor model Deep Learning (Gunakan dummy jika belum dilatih)
# from models.st_gcn_graph_tran.model import STGCN_GraphTransformer
# from models.stc_rppg.model import STCrPPG
# from models.fusion.late_fusion import LateFusionVABridge

# Konfigurasi Hardware (Jetson Orin Nano Super - IMX477)
# Untuk webcam USB lokal (laptop), gunakan 0. Untuk Jetson CSI, gunakan string GStreamer.
CAMERA_SOURCE = 0  
MQTT_BROKER = "localhost" # Ganti jika broker berjalan di perangkat lain
MQTT_TOPIC = "moodin/inference/result"

# Setup MediaPipe untuk visualisasi sci-fi
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def gstreamer_pipeline(capture_width=1280, capture_height=720, framerate=30):
    """Pipeline GStreamer khusus untuk Jetson + nvarguscamerasrc (IMX477)."""
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

def main():
    print("=== Menginisialisasi MoodIn Edge Inference ===")
    
    # 1. Setup MQTT Publisher (untuk komunikasi ke UI lokal/Firebase)
    client = mqtt.Client(client_id="Jetson_Inference")
    try:
        client.connect(MQTT_BROKER, 1883, 60)
        client.loop_start()
        print(f"[MQTT] Terhubung ke broker {MQTT_BROKER}")
    except Exception as e:
        print(f"[MQTT] Gagal terhubung ke broker: {e}. Melanjutkan tanpa MQTT.")

    # 2. Inisialisasi Kamera (Fallback otomatis dari Jetson CSI ke USB Webcam)
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[Kamera] Pipeline CSI gagal. Mencoba USB Webcam lokal (Laptop)...")
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        
    if not cap.isOpened():
        print("[Kamera] ERROR: Tidak ada kamera yang terdeteksi!")
        return

    # Buffer untuk Model Temporal (Dummy sizes)
    # ST-GCN butuh (C=2, T=150, V=68), STC-rPPG butuh (C=3, T=128, H, W)
    buffer_stgcn = collections.deque(maxlen=150)
    buffer_rppg = collections.deque(maxlen=128)
    
    print("[Inference] Mulai membaca frame. Tekan 'q' pada jendela untuk keluar.")
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Konversi warna untuk MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            # Buat salinan untuk digambar visualisasi UI
            display_frame = frame.copy()
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Gambar jaring wajah futuristik ala Salvion.ai
                    mp_drawing.draw_landmarks(
                        image=display_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=display_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    
                    # Ekstraksi koordinat mentah untuk dimasukkan ke buffer ST-GCN (dummy logic)
                    coords = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                    buffer_stgcn.append(coords)
                    
                    # Ekstraksi crop wajah untuk STC-rPPG (dummy logic)
                    # (Dalam implementasi penuh, gunakan bounding box wajah)
                    buffer_rppg.append(frame)
            else:
                # Wajah hilang (degradasi terencana)
                buffer_stgcn.clear()
                buffer_rppg.clear()
            
            # --- MOCKUP MODEL INFERENCE ---
            # Jika buffer ST-GCN penuh (150 frame = 5 detik @30fps)
            if len(buffer_stgcn) == 150:
                # 1. Forward pass ST-GCN -> Valence & Arousal (Wajah)
                val_face, aro_face = np.random.uniform(-1, 1, 2) # DUMMY
                
                # 2. Forward pass STC-rPPG -> BPM -> Arousal (Fisik)
                aro_physio = np.random.uniform(-1, 1) # DUMMY
                bpm = np.random.uniform(60, 100) # DUMMY
                
                # 3. Late Fusion (Fusi V-A) -> Label Kelas
                # final_mood = fusion_bridge.fuse(val_face, aro_face, aro_physio)
                final_mood = "Ceria" if val_face > 0 else "Rileks" # DUMMY
                
                # 4. Kirim hasil inferensi via MQTT ke UI Cermin
                payload = {
                    "mood_label": final_mood,
                    "valence": float(val_face),
                    "arousal": float((aro_face + aro_physio)/2),
                    "heart_rate_bpm": float(bpm),
                    "physio_used": True,
                    "timestamp": time.time()
                }
                client.publish(MQTT_TOPIC, json.dumps(payload))
                
                # Tampilkan hasil di layar komputer untuk debugging
                cv2.putText(display_frame, f"Mood: {final_mood} | BPM: {bpm:.1f}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Kosongkan sebagian buffer (stride) agar tidak memproses ulang 100% frame lama
                for _ in range(15): # Stride 15 frame
                    buffer_stgcn.popleft()
            
            # Tampilkan ke layar lokal (Laptop/Jetson GUI)
            cv2.imshow('MoodIn - Real-time Landmark (50% Progress Demo)', display_frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()

if __name__ == "__main__":
    main()
