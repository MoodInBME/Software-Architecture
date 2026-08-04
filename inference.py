"""
MoodIn - Edge Inference Orchestrator (Jetson Orin Nano Super)

Pipeline real-time:
  Kamera (IMX477 CSI / USB) -> MediaPipe FaceMesh (468 titik, visualisasi UI)
  -> Pemetaan iBUG-68 -> Buffer Temporal 150 frame
  -> ST-GCN (valence, arousal wajah) + STC-rPPG (BPM, arousal fisik)
  -> LateFusionVABridge -> MQTT publish -> UI Cermin

Dirancang agar dapat langsung dijalankan sebagai demo progres 50%
bahkan jika model masih menggunakan bobot acak (dummy weights).

Fallback: jika MediaPipe tidak tersedia (umum di Jetson ARM64),
script akan menggunakan dlib sebagai alternatif.
"""
import cv2
import json
import time
import torch
import numpy as np
import collections
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.st_gcn_graph_tran.model import STGCN_GraphTransformer
from models.stc_rppg.model import STCrPPG
from models.fusion.late_fusion import LateFusionVABridge
from data.preprocessing.landmark_map import MEDIAPIPE_FACEMESH_TO_IBUG68

# ---- Konfigurasi ----
CAMERA_SOURCE = 0
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_RESULT = "moodin/inference/result"
MQTT_TOPIC_QUALITY = "moodin/signal/quality"
WEIGHT_STGCN = os.path.join(os.path.dirname(__file__), "weights", "dummy_stgcn.pth")
WEIGHT_STCRPPG = os.path.join(os.path.dirname(__file__), "weights", "dummy_stcrppg.pth")
CLIP_LEN_STGCN = 150   # frame buffer untuk ST-GCN
CLIP_LEN_RPPG = 128    # frame buffer untuk STC-rPPG
STRIDE = 15             # stride frame setelah satu kali inferensi
RPPG_H, RPPG_W = 128, 128  # resolusi input STC-rPPG
FPS_TARGET = 30

# ---- Inisialisasi Deteksi Wajah ----
# Strategi: coba MediaPipe dulu. Jika gagal (ARM64 tanpa wheel),
# fallback ke dlib. Jika dlib juga tidak ada, gunakan OpenCV Haar.
FACE_BACKEND = None
face_mesh = None
face_detector_dlib = None
landmark_predictor_dlib = None
haar_cascade = None

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    FACE_BACKEND = "mediapipe"
    print("[FaceMesh] Backend: MediaPipe (468 landmark)")
except ImportError:
    print("[FaceMesh] MediaPipe tidak tersedia, mencoba dlib...")
    try:
        import dlib
        face_detector_dlib = dlib.get_frontal_face_detector()
        predictor_path = os.path.join(
            os.path.dirname(__file__), "weights", "shape_predictor_68_face_landmarks.dat")
        if os.path.exists(predictor_path):
            landmark_predictor_dlib = dlib.shape_predictor(predictor_path)
            FACE_BACKEND = "dlib"
            print(f"[FaceMesh] Backend: dlib (68 landmark langsung)")
        else:
            print(f"[FaceMesh] dlib tersedia tapi prediktor tidak ditemukan di {predictor_path}")
            raise FileNotFoundError(predictor_path)
    except (ImportError, FileNotFoundError):
        print("[FaceMesh] dlib juga tidak tersedia. Fallback ke OpenCV Haar (tanpa landmark).")
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        haar_cascade = cv2.CascadeClassifier(haar_path)
        FACE_BACKEND = "haar"

# ---- MQTT (opsional) ----
mqtt_client = None
try:
    import paho.mqtt.client as mqtt
    mqtt_client = mqtt.Client(client_id="Jetson_Inference")
except ImportError:
    print("[MQTT] paho-mqtt tidak tersedia. Melanjutkan tanpa MQTT.")


def gstreamer_pipeline(width=1280, height=720, fps=30, flip_method=2):
    """Pipeline GStreamer untuk nvarguscamerasrc (Jetson CSI, IMX477)."""
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


def extract_landmarks_mediapipe(frame_rgb):
    """Ekstrak 68 iBUG landmarks dari MediaPipe FaceMesh 468."""
    results = face_mesh.process(frame_rgb)
    if not results.multi_face_landmarks:
        return None, None
    face_lm = results.multi_face_landmarks[0]
    all_468 = np.array([[lm.x, lm.y] for lm in face_lm.landmark])  # (468, 2)
    ibug68 = all_468[MEDIAPIPE_FACEMESH_TO_IBUG68]  # (68, 2) -- pemetaan yang benar
    return ibug68, face_lm


def extract_landmarks_dlib(frame_gray, frame_rgb):
    """Ekstrak 68 iBUG landmarks langsung dari dlib."""
    faces = face_detector_dlib(frame_gray, 0)
    if len(faces) == 0:
        return None
    shape = landmark_predictor_dlib(frame_gray, faces[0])
    h, w = frame_rgb.shape[:2]
    coords = np.array([[shape.part(i).x / w, shape.part(i).y / h]
                        for i in range(68)])  # (68, 2) normalized
    return coords


def draw_landmarks_on_frame(display_frame, face_lm_mediapipe=None, ibug68=None):
    """Gambar landmark futuristik pada frame untuk UI demo."""
    if FACE_BACKEND == "mediapipe" and face_lm_mediapipe is not None:
        mp_drawing.draw_landmarks(
            image=display_frame, landmark_list=face_lm_mediapipe,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=display_frame, landmark_list=face_lm_mediapipe,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
    elif ibug68 is not None:
        h, w = display_frame.shape[:2]
        for (nx, ny) in ibug68:
            px, py = int(nx * w), int(ny * h)
            cv2.circle(display_frame, (px, py), 2, (0, 255, 0), -1)


def main():
    print("=" * 65)
    print(" MoodIn Edge Inference - Demo 50% ".center(65, "="))
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ---- Model Init ----
    print("[Model] Memuat ST-GCN (head_type=va)...")
    stgcn = STGCN_GraphTransformer(
        in_channels=2, num_class=5, num_node=68, graph_strategy="spatial",
        max_hop=1, t_kernel=9, transformer_layers=2, transformer_heads=8,
        transformer_ff=512, dropout=0.3, head_type="va"
    ).to(device).eval()

    print("[Model] Memuat STC-rPPG...")
    stc_rppg = STCrPPG().to(device).eval()

    fusion = LateFusionVABridge(
        w_face_arousal=0.5, w_physio_arousal=0.5,
        ema_alpha=0.3, physio_timeout_s=5.0)

    # ---- Muat Bobot ----
    for name, model, path in [("ST-GCN", stgcn, WEIGHT_STGCN),
                                ("STC-rPPG", stc_rppg, WEIGHT_STCRPPG)]:
        if os.path.exists(path):
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=True))
            print(f"[Bobot] {name} dimuat dari {os.path.basename(path)}")
        else:
            print(f"[Bobot] WARNING: {path} tidak ditemukan. "
                  f"{name} menggunakan bobot inisialisasi acak.")

    # ---- MQTT Connect ----
    if mqtt_client is not None:
        try:
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            mqtt_client.loop_start()
            print(f"[MQTT] Terhubung ke {MQTT_BROKER}:{MQTT_PORT}")
        except Exception as e:
            print(f"[MQTT] Gagal terhubung: {e}. Melanjutkan tanpa MQTT.")

    # ---- Kamera Init ----
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[Kamera] GStreamer CSI gagal. Fallback ke USB webcam...")
        cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("[Kamera] FATAL: Tidak ada kamera terdeteksi.")
        return

    # ---- Buffers ----
    buf_stgcn = collections.deque(maxlen=CLIP_LEN_STGCN)
    buf_rppg = collections.deque(maxlen=CLIP_LEN_RPPG)
    
    # Variabel status untuk overlay UI
    last_mood = "Menunggu..."
    last_bpm = 0.0
    last_confidence = 0.0
    frame_count = 0
    inference_running = False

    def run_inference_thread(stgcn_t_in, rppg_t_in):
        nonlocal last_mood, last_bpm, last_confidence, inference_running
        try:
            with torch.no_grad():
                stgcn_out = stgcn(stgcn_t_in)
                va = stgcn_out["va"].squeeze(0).cpu().numpy()
                face_val, face_aro = float(va[0]), float(va[1])

                rppg_signal = stc_rppg(rppg_t_in).squeeze().cpu().numpy()
                n = len(rppg_signal)
                if n > 1:
                    fft_mag = np.abs(np.fft.rfft(rppg_signal))
                    freqs = np.fft.rfftfreq(n, d=1.0 / FPS_TARGET)
                    valid = (freqs >= 0.7) & (freqs <= 3.5)
                    if np.any(valid):
                        peak_idx = np.argmax(fft_mag[valid])
                        peak_freq = freqs[valid][peak_idx]
                        bpm = peak_freq * 60.0
                    else:
                        bpm = 0.0
                else:
                    bpm = 0.0

            result = fusion.fuse(
                face_valence=face_val, face_arousal=face_aro,
                face_confidence=1.0, hr_bpm=bpm if bpm > 0 else None, hrv_ms=None
            )

            last_mood = result["mood_label"]
            last_bpm = result["heart_rate_bpm"] if result["heart_rate_bpm"] else bpm
            last_confidence = result["confidence"]

            if mqtt_client is not None:
                payload = {
                    "mood_id": result["mood_id"], "mood_label": result["mood_label"],
                    "confidence": result["confidence"], "valence": result["valence"],
                    "arousal": result["arousal"], "heart_rate_bpm": result["heart_rate_bpm"],
                    "hrv_ms": result["hrv_ms"], "physio_used": result["physio_used"],
                    "timestamp": time.time()
                }
                mqtt_client.publish(MQTT_TOPIC_RESULT, json.dumps(payload))
        except Exception as e:
            print(f"[Error] Forward pass gagal: {e}")
        finally:
            inference_running = False

    print(f"[Inference] Backend wajah: {FACE_BACKEND}")
    print(f"[Inference] Tekan 'q' untuk keluar.\n")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display = frame.copy()

        # ---- Ekstraksi Landmark ----
        ibug68 = None
        face_lm_mp = None
        face_detected = False

        if FACE_BACKEND == "mediapipe":
            ibug68, face_lm_mp = extract_landmarks_mediapipe(frame_rgb)
            face_detected = ibug68 is not None

        elif FACE_BACKEND == "dlib":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ibug68 = extract_landmarks_dlib(gray, frame_rgb)
            face_detected = ibug68 is not None

        elif FACE_BACKEND == "haar":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                face_detected = True
                # Tidak ada landmark, tapi STC-rPPG tetap bisa jalan dengan ROI wajah

        # ---- Gambar landmark di frame ----
        draw_landmarks_on_frame(display, face_lm_mp, ibug68)

        # ---- Isi buffer ----
        if face_detected:
            if ibug68 is not None:
                buf_stgcn.append(ibug68)  # (68, 2)
            
            # Crop wajah untuk rPPG (resize ke 128x128)
            resized = cv2.resize(frame_rgb, (RPPG_W, RPPG_H))
            resized = resized.astype(np.float32) / 255.0  # normalize [0, 1]
            buf_rppg.append(resized)
        else:
            buf_stgcn.clear()
            buf_rppg.clear()

        # ---- Forward Pass Model ----
        if len(buf_stgcn) >= CLIP_LEN_STGCN and len(buf_rppg) >= CLIP_LEN_RPPG:
            if not inference_running:
                try:
                    inference_running = True
                    
                    # Salin data agar aman digunakan oleh thread lain
                    stgcn_np = np.array(list(buf_stgcn))       
                    stgcn_np = stgcn_np.transpose(2, 0, 1)     
                    stgcn_t = torch.tensor(stgcn_np, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    rppg_np = np.array(list(buf_rppg)[-CLIP_LEN_RPPG:]) 
                    rppg_np = rppg_np.transpose(3, 0, 1, 2)
                    rppg_t = torch.tensor(rppg_np, dtype=torch.float32).unsqueeze(0).to(device)

                    # Luncurkan thread inferensi agar UI tidak nge-lag/nge-frame
                    threading.Thread(target=run_inference_thread, args=(stgcn_t, rppg_t), daemon=True).start()

                    # Stride: buang frame lama setelah mengirim data
                    for _ in range(STRIDE):
                        if buf_stgcn:
                            buf_stgcn.popleft()

                except Exception as e:
                    print(f"[Error] Gagal meluncurkan thread inferensi: {e}")
                    inference_running = False
                    buf_stgcn.clear()
                    buf_rppg.clear()

        # ---- Overlay UI ----
        bar_h = 80
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        cv2.putText(display, f"Mood: {last_mood}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)
        cv2.putText(display, f"BPM: {last_bpm:.0f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        buf_pct = min(len(buf_stgcn) / CLIP_LEN_STGCN, 1.0)
        bar_w = int(200 * buf_pct)
        cv2.rectangle(display, (display.shape[1] - 220, 15),
                      (display.shape[1] - 220 + bar_w, 35), (0, 255, 0), -1)
        cv2.putText(display, f"Buffer: {len(buf_stgcn)}/{CLIP_LEN_STGCN}",
                    (display.shape[1] - 220, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("MoodIn - Real-time Inference", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if mqtt_client is not None:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    print("[Inference] Selesai.")


if __name__ == "__main__":
    main()
