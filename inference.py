"""
MoodIn - Edge Inference Orchestrator (Jetson Orin Nano Super)

Pipeline real-time:
  Kamera (IMX477 CSI / USB) -> deteksi wajah -> iBUG-68
  -> buffer temporal -> ST-GCN (valence, arousal) + STC-rPPG (BPM)
  -> LateFusionVABridge -> MQTT -> UI Cermin

PATCH 2026-08-06 (rev-b):
  [S1]  DEMO_MODE berlabel jujur: badge permanen di layar, flag `simulated`
        ikut terkirim lewat MQTT, dan log awal menyatakannya eksplisit.
  [S2]  rPPG memakai crop ROI wajah, bukan seluruh frame.
  [S3]  MQTT memakai communication/mqtt_client.py (aman untuk paho 2.x).
  [S4]  Payload dirakit lewat contracts.InferenceResult (key `ts`, `version`).
  [S6]  RR ditandai sebagai estimasi turunan, bukan hasil ukur.
  [S7]  SignalQuality dan Presence ikut dipublikasikan.
  [S8]  Mode sensor CSI default diubah ke 1920x1080.
  [S10] Konfigurasi dibaca dari config/default.yaml.
"""
from __future__ import annotations

import collections
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.st_gcn_graph_tran.model import STGCN_GraphTransformer
from models.stc_rppg.model import STCrPPG
from models.fusion.late_fusion import LateFusionVABridge
from data.preprocessing.landmark_map import MEDIAPIPE_FACEMESH_TO_IBUG68

CONFIG_PATH = os.path.join(ROOT, "config", "default.yaml")


def _defaults():
    c = dict()
    c["camera_source"] = 0
    c["csi_width"] = 1920
    c["csi_height"] = 1080
    c["csi_fps"] = 30
    c["csi_flip_method"] = 2
    c["csi_sensor_mode"] = 1
    c["proc_width"] = 1280
    c["proc_height"] = 720
    c["mqtt_enabled"] = True
    c["mqtt_host"] = "localhost"
    c["mqtt_port"] = 1883
    c["clip_len_stgcn"] = 150
    c["clip_len_rppg"] = 128
    c["stride"] = 15
    c["rppg_size"] = 128
    c["fps_target"] = 30
    c["screen_width"] = 600
    c["screen_height"] = 1024
    c["display_portrait"] = True
    c["fullscreen"] = True
    c["demo_mode"] = True
    c["demo_banner"] = "MODE DEMO - NILAI DISIMULASIKAN"
    c["weight_stgcn"] = "weights/dummy_stgcn.pth"
    c["weight_stcrppg"] = "weights/dummy_stcrppg.pth"
    return c


def load_config():
    cfg = _defaults()
    try:
        import yaml
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if isinstance(loaded, dict):
            for k in loaded:
                cfg[str(k)] = loaded[k]
            print("[Config] Dimuat dari config/default.yaml")
        else:
            print("[Config] default.yaml kosong, memakai nilai bawaan.")
    except FileNotFoundError:
        print("[Config] default.yaml tidak ditemukan, memakai nilai bawaan.")
    except Exception as e:
        print("[Config] Gagal membaca YAML (" + str(e) + "), memakai bawaan.")
    return cfg


CFG = load_config()

CLIP_LEN_STGCN = int(CFG["clip_len_stgcn"])
CLIP_LEN_RPPG = int(CFG["clip_len_rppg"])
STRIDE = int(CFG["stride"])
RPPG_H = RPPG_W = int(CFG["rppg_size"])
FPS_TARGET = float(CFG["fps_target"])
SCREEN_WIDTH = int(CFG["screen_width"])
SCREEN_HEIGHT = int(CFG["screen_height"])
DISPLAY_PORTRAIT = bool(CFG["display_portrait"])
DEMO_MODE = bool(CFG["demo_mode"])
DEMO_BANNER = str(CFG["demo_banner"])
WEIGHT_STGCN = os.path.join(ROOT, str(CFG["weight_stgcn"]))
WEIGHT_STCRPPG = os.path.join(ROOT, str(CFG["weight_stcrppg"]))


# --------------------------------------------------------------------------
# Inisialisasi deteksi wajah: MediaPipe -> dlib -> Haar
# --------------------------------------------------------------------------
FACE_BACKEND = None
face_mesh = None
mp_drawing = None
mp_drawing_styles = None
mp_face_mesh = None
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
            ROOT, "weights", "shape_predictor_68_face_landmarks.dat")
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(predictor_path)
        landmark_predictor_dlib = dlib.shape_predictor(predictor_path)
        FACE_BACKEND = "dlib"
        print("[FaceMesh] Backend: dlib (68 landmark langsung)")
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        print("[FaceMesh] dlib tidak dapat dipakai (" + str(e) + ").")
        print("[FaceMesh] Fallback ke OpenCV Haar (tanpa landmark).")
        haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        FACE_BACKEND = "haar"


# --------------------------------------------------------------------------
# [S3] MQTT lewat lapisan communication/ (bukan paho mentah)
# --------------------------------------------------------------------------
MoodInMqttClient = None
InferenceResult = None
SignalQuality = None
_MQTT_IMPORT_ERROR = None

if bool(CFG["mqtt_enabled"]):
    try:
        from communication.mqtt_client import MoodInMqttClient
        from communication.contracts import InferenceResult, SignalQuality
    except Exception as e:
        _MQTT_IMPORT_ERROR = e
        print("[MQTT] Lapisan communication tidak dapat dimuat (" + str(e) + ").")
        print("[MQTT] Melanjutkan tanpa MQTT.")


def connect_mqtt():
    """Kembalikan klien MQTT siap pakai, atau None bila gagal.

    Semua kegagalan ditangkap di sini supaya pipeline inferensi tetap jalan
    walau broker mati. Ini yang dulu bocor: mqtt.Client(client_id=...) pada
    paho 2.x melempar TypeError, bukan ImportError, sehingga proses mati.
    """
    if MoodInMqttClient is None:
        return None
    try:
        client = MoodInMqttClient(
            service_name="moodin-inference",
            host=str(CFG["mqtt_host"]),
            port=int(CFG["mqtt_port"]),
        )
        client.connect()
        print("[MQTT] Terhubung ke " + str(CFG["mqtt_host"]) + ":"
              + str(CFG["mqtt_port"]) + " (presence=online, LWT aktif)")
        return client
    except Exception as e:
        print("[MQTT] Gagal terhubung: " + str(e) + ". Melanjutkan tanpa MQTT.")
        return None


# --------------------------------------------------------------------------
# Helper kamera & landmark
# --------------------------------------------------------------------------
def gstreamer_pipeline(width=None, height=None, fps=None, flip_method=None):
    """Pipeline nvarguscamerasrc untuk IMX477 di Jetson.

    [S8] Default diubah ke 1920x1080. 1280x960 bukan mode sensor IMX477 yang
    lazim sehingga negosiasi caps sering gagal dan diam-diam jatuh ke USB.
    Verifikasi mode yang benar-benar didukung dengan:
        v4l2-ctl --list-formats-ext -d /dev/video0
    """
    width = int(CFG["csi_width"]) if width is None else int(width)
    height = int(CFG["csi_height"]) if height is None else int(height)
    fps = int(CFG["csi_fps"]) if fps is None else int(fps)
    flip_method = (int(CFG["csi_flip_method"])
                   if flip_method is None else int(flip_method))
    mode = int(CFG["csi_sensor_mode"])
    pw = int(CFG["proc_width"])
    ph = int(CFG["proc_height"])
    return (
        "nvarguscamerasrc sensor-id=0 sensor-mode=" + str(mode) + " ! "
        "video/x-raw(memory:NVMM), width=(int)" + str(width) + ", "
        "height=(int)" + str(height) + ", format=(string)NV12, "
        "framerate=(fraction)" + str(fps) + "/1 ! "
        "nvvidconv flip-method=" + str(flip_method) + " ! "
        "video/x-raw, width=(int)" + str(pw) + ", "
        "height=(int)" + str(ph) + ", format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=2 sync=false"
    )


def extract_landmarks_mediapipe(frame_rgb):
    results = face_mesh.process(frame_rgb)
    if not results.multi_face_landmarks:
        return None, None
    face_lm = results.multi_face_landmarks[0]
    all_468 = np.array([[lm.x, lm.y] for lm in face_lm.landmark])
    return all_468[MEDIAPIPE_FACEMESH_TO_IBUG68], face_lm


def extract_landmarks_dlib(frame_gray, frame_shape):
    faces = face_detector_dlib(frame_gray, 0)
    if len(faces) == 0:
        return None
    shape = landmark_predictor_dlib(frame_gray, faces[0])
    h, w = frame_shape[:2]
    return np.array([[shape.part(i).x / w, shape.part(i).y / h]
                     for i in range(68)])


def face_bbox_from_landmarks(ibug68, frame_w, frame_h,
                             margin_x=0.10, margin_y=0.15):
    """[S2] Kotak batas wajah dari 68 landmark ternormalisasi.

    Mengembalikan (x0, y0, x1, y1) dalam piksel, atau None bila tidak masuk
    akal. Margin vertikal lebih besar supaya dahi ikut terambil: dahi adalah
    ROI rPPG paling stabil karena tidak bergerak saat orang bicara.
    """
    if ibug68 is None:
        return None
    xs = ibug68[:, 0] * frame_w
    ys = ibug68[:, 1] * frame_h
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    bw, bh = x1 - x0, y1 - y0
    if bw < 40 or bh < 40:
        return None
    mx, my = margin_x * bw, margin_y * bh
    x0 = int(max(0, x0 - mx))
    y0 = int(max(0, y0 - my * 1.6))
    x1 = int(min(frame_w, x1 + mx))
    y1 = int(min(frame_h, y1 + my * 0.4))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    return (x0, y0, x1, y1)


def face_bbox_from_haar(frame_gray):
    """[S2] Kotak batas wajah ketika hanya Haar yang tersedia."""
    faces = haar_cascade.detectMultiScale(frame_gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (int(x), int(y), int(x + w), int(y + h))


def crop_face_for_rppg(frame_rgb, bbox):
    """[S2] Potong ROI wajah lalu skalakan ke input STC-rPPG.

    Sebelum patch ini seluruh frame diperas ke 128x128, sehingga piksel kulit
    hanya menempati sebagian kecil citra dan sinyal rPPG (orde 0,1% perubahan
    warna) tenggelam oleh latar statis.
    """
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    patch = frame_rgb[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    patch = cv2.resize(patch, (RPPG_W, RPPG_H), interpolation=cv2.INTER_AREA)
    return patch.astype(np.float32) / 255.0


def bpm_from_signal(rppg_signal, fps=None, f_lo=0.7, f_hi=3.5):
    """Ambil BPM dari puncak spektrum sinyal rPPG pada pita 42-210 BPM."""
    fps = FPS_TARGET if fps is None else float(fps)
    sig = np.asarray(rppg_signal, dtype=np.float64).ravel()
    n = sig.size
    if n < 8:
        return 0.0
    sig = sig - sig.mean()
    if not np.isfinite(sig).all() or sig.std() < 1e-9:
        return 0.0
    sig = sig * np.hanning(n)
    mag = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return 0.0
    return float(freqs[band][int(np.argmax(mag[band]))] * 60.0)


# --------------------------------------------------------------------------
# [S1] Overlay: badge MODE DEMO permanen
# --------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX

RUNTIME = dict()
RUNTIME["fps"] = 0.0
RUNTIME["frames"] = 0
RUNTIME["t0"] = time.time()


def apply_fullscreen(win):
    """Terapkan fullscreen SETELAH imshow pertama.

    Backend highgui baru benar-benar membuat jendela pada imshow pertama.
    Properti yang diset sebelum itu sering diabaikan diam-diam, dan itulah
    sebabnya jendela tetap berbingkai pada percobaan sebelumnya.
    """
    try:
        cv2.setWindowProperty(
            win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception as e:
        print("[UI] setWindowProperty fullscreen gagal: " + str(e))
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    if os.system("command -v wmctrl > /dev/null 2>&1") == 0:
        os.system('wmctrl -r "' + win + '" -b add,fullscreen,above'
                  " > /dev/null 2>&1")


def tick_fps():
    """Perbarui FPS nyata tiap detik.

    Ini bukan hiasan. bpm_from_signal() mengasumsikan laju frame sama
    dengan fps_target; kalau FPS nyata melenceng, BPM ikut melenceng
    dengan faktor yang sama.
    """
    RUNTIME["frames"] += 1
    dt = time.time() - RUNTIME["t0"]
    if dt >= 1.0:
        RUNTIME["fps"] = RUNTIME["frames"] / dt
        RUNTIME["frames"] = 0
        RUNTIME["t0"] = time.time()


def draw_landmarks_on_frame(display_frame, face_lm_mediapipe=None, ibug68=None):
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
            cv2.circle(display_frame, (int(nx * w), int(ny * h)), 2,
                       (0, 255, 0), -1)


def draw_demo_badge(display_frame):
    """[S1] Pita oranye permanen di bawah layar.

    Ditaruh di bawah, bukan di atas, supaya tidak tertimpa bilah status dan
    tetap terbaca ketika frame di-crop portrait. Warnanya sengaja mencolok
    supaya tetap terlihat pada rekaman video yang dikompresi.
    """
    if not DEMO_MODE:
        return
    h, w = display_frame.shape[:2]
    band = 42
    y0 = h - band
    strip = display_frame[y0:h, 0:w].copy()
    cv2.rectangle(strip, (0, 0), (w, band), (0, 90, 200), -1)
    cv2.addWeighted(strip, 0.85, display_frame[y0:h, 0:w], 0.15, 0,
                    display_frame[y0:h, 0:w])
    cv2.rectangle(display_frame, (0, y0), (w - 1, h - 1), (0, 140, 255), 2)

    text = DEMO_BANNER
    scale = 0.52
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    while tw > w - 20 and scale > 0.28:
        scale -= 0.02
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(display_frame, text, ((w - tw) // 2, y0 + (band + th) // 2),
                FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_status_overlay(display_frame, mood, bpm, rr, conf, buf_fill,
                        physio_used, face_detected):
    h, w = display_frame.shape[:2]
    bar_h = 118
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

    cv2.putText(display_frame, "Mood: " + str(mood), (18, 28),
                FONT, 0.75, (0, 255, 180), 2, cv2.LINE_AA)

    hr_txt = ("HR: " + format(bpm, ".0f") + " BPM") if bpm else "HR: --"
    if DEMO_MODE:
        hr_txt += "  (sim)"
    elif not physio_used:
        hr_txt += "  (stale)"
    cv2.putText(display_frame, hr_txt, (18, 56),
                FONT, 0.62, (0, 200, 255), 2, cv2.LINE_AA)

    # [S6] RR bukan hasil ukur: rasio tetap terhadap BPM. Label harus jujur.
    rr_txt = ("RR: " + format(rr, ".0f") + " /min (estimasi turunan)"
              if rr else "RR: --")
    cv2.putText(display_frame, rr_txt, (18, 82),
                FONT, 0.52, (255, 200, 0), 1, cv2.LINE_AA)

    fps_now = RUNTIME["fps"]
    fps_col = (200, 200, 200) if fps_now >= FPS_TARGET * 0.8 else (0, 80, 255)
    cv2.putText(display_frame,
                "Conf: " + format(conf, ".2f")
                + "   FPS: " + format(fps_now, ".1f"),
                (18, 106), FONT, 0.5, fps_col, 1, cv2.LINE_AA)

    bw = 150
    bx = w - bw - 18
    cv2.rectangle(display_frame, (bx, 16), (bx + bw, 32), (70, 70, 70), 1)
    cv2.rectangle(display_frame, (bx, 16), (bx + int(bw * buf_fill), 32),
                  (0, 255, 0), -1)
    cv2.putText(display_frame, "Buffer " + format(buf_fill * 100, ".0f") + "%",
                (bx, 50), FONT, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(display_frame, "Wajah: " + ("ya" if face_detected else "tidak"),
                (bx, 70), FONT, 0.44,
                (0, 255, 0) if face_detected else (0, 80, 255), 1, cv2.LINE_AA)

    draw_demo_badge(display_frame)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("=" * 66)
    print(" MoodIn Edge Inference ".center(66, "="))
    print("=" * 66)
    if DEMO_MODE:
        print("")
        print("  !! MODE DEMO AKTIF !!")
        print("  Valence, arousal, BPM, dan RR yang ditampilkan adalah nilai")
        print("  SIMULASI berbasis waktu, bukan hasil pengukuran wajah.")
        print("  Matikan lewat config/default.yaml -> demo_mode: false")
        print("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_txt = "[Device] " + str(device)
    if device.type == "cuda":
        dev_txt += " (" + torch.cuda.get_device_name(0) + ")"
    print(dev_txt)

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

    weights_real = True
    for name, model, path in (("ST-GCN", stgcn, WEIGHT_STGCN),
                              ("STC-rPPG", stc_rppg, WEIGHT_STCRPPG)):
        if os.path.exists(path):
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=True))
            print("[Bobot] " + name + " dimuat dari " + os.path.basename(path))
            if "dummy" in os.path.basename(path).lower():
                weights_real = False
        else:
            print("[Bobot] PERINGATAN: " + path + " tidak ada. "
                  + name + " memakai inisialisasi acak.")
            weights_real = False
    if not weights_real:
        print("[Bobot] Bobot bukan hasil pelatihan pada data nyata. "
              "Keluaran model tidak bermakna klinis.")

    mq = connect_mqtt()

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[Kamera] GStreamer CSI gagal. Fallback ke USB webcam...")
        cap = cv2.VideoCapture(int(CFG["camera_source"]))
    if not cap.isOpened():
        print("[Kamera] FATAL: tidak ada kamera terdeteksi.")
        if mq is not None:
            mq.disconnect()
        return 1

    buf_stgcn = collections.deque(maxlen=CLIP_LEN_STGCN)
    buf_rppg = collections.deque(maxlen=CLIP_LEN_RPPG)

    state_lock = threading.Lock()
    state = dict()
    state["mood"] = "Menunggu..."
    state["bpm"] = 0.0
    state["rr"] = 0.0
    state["confidence"] = 0.0
    state["physio_used"] = False
    state["running"] = False

    def run_inference_thread(stgcn_t_in, rppg_t_in):
        """Inferensi di thread terpisah supaya loop kamera tidak tersendat."""
        bpm = 0.0
        rr = 0.0
        try:
            with torch.no_grad():
                va = stgcn(stgcn_t_in)["va"].squeeze(0).cpu().numpy()
                face_val, face_aro = float(va[0]), float(va[1])

                rppg_signal = stc_rppg(rppg_t_in).squeeze().cpu().numpy()
                bpm = bpm_from_signal(rppg_signal)

            # [S6] RR adalah rasio tetap terhadap BPM, bukan pengukuran
            # independen. Nilai sebenarnya perlu respiratory sinus arrhythmia
            # (band-pass deret interval antar-denyut pada 0,15-0,4 Hz).
            rr = bpm / 4.5 if bpm > 0 else 0.0

            # ------------------------------------------------------------
            # [S1] MODE DEMO
            # Nilai di bawah ini adalah fungsi waktu, BUKAN fungsi wajah.
            # Dua orang berbeda pada detik yang sama akan mendapat mood yang
            # sama persis. Wajib disertai badge di layar dan flag `simulated`
            # pada payload MQTT. Matikan setelah model CK+ selesai dilatih.
            # ------------------------------------------------------------
            if DEMO_MODE:
                t = time.time()
                face_val = float(np.clip(face_val + np.sin(t * 0.4) * 0.7,
                                         -1.0, 1.0))
                face_aro = float(np.clip(face_aro + np.cos(t * 0.25) * 0.7,
                                         -1.0, 1.0))
                bpm = 77.0 + np.sin(t * 0.2) * 8.0 + np.random.uniform(-1.5, 1.5)
                rr = 17.0 + np.cos(t * 0.15) * 3.0 + np.random.uniform(-0.5, 0.5)

            result = fusion.fuse(
                face_valence=face_val, face_arousal=face_aro,
                face_confidence=1.0,
                hr_bpm=bpm if bpm > 0 else None, hrv_ms=None)

            with state_lock:
                state["mood"] = result["mood_label"]
                state["bpm"] = result["heart_rate_bpm"] or bpm
                state["rr"] = rr
                state["confidence"] = result["confidence"]
                state["physio_used"] = result["physio_used"]

            # [S4] Payload dirakit lewat kontrak resmi, bukan dict manual.
            # Versi lama mengirim key "timestamp" padahal contracts.py memakai
            # "ts", dan tidak menyertakan "version" sama sekali.
            if mq is not None:
                fields = dict()
                fields["mood_id"] = result["mood_id"]
                fields["mood_label"] = result["mood_label"]
                fields["confidence"] = result["confidence"]
                fields["valence"] = result["valence"]
                fields["arousal"] = result["arousal"]
                fields["heart_rate_bpm"] = result["heart_rate_bpm"]
                fields["hrv_ms"] = result["hrv_ms"]
                fields["physio_used"] = result["physio_used"]
                # [S1] Penanda simulasi ikut ke UI. Interface-Display wajib
                # menampilkan pita peringatan bila field ini true. Butuh field
                # `simulated` di contracts.py (lihat PATCH_NOTES.md bagian B).
                fields["simulated"] = bool(DEMO_MODE)
                try:
                    payload = InferenceResult(**fields)
                except TypeError:
                    fields.pop("simulated", None)
                    payload = InferenceResult(**fields)
                mq.publish_result(payload)
        except Exception as e:
            print("[Error] Forward pass gagal: " + str(e))
        finally:
            with state_lock:
                state["running"] = False

    print("[Inference] Backend wajah: " + str(FACE_BACKEND))
    print("[Inference] Tekan 'q' untuk keluar.")
    print("")

    frame_count = 0
    win = "MoodIn - Real-time Inference"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, SCREEN_WIDTH, SCREEN_HEIGHT)
    fs_pending = bool(CFG["fullscreen"])
    if os.environ.get("XDG_SESSION_TYPE", "") == "wayland":
        print("[UI] Sesi Wayland terdeteksi. Fullscreen highgui sering "
              "gagal di Wayland; pasang wmctrl atau login memakai sesi "
              "Xorg bila jendela tetap berbingkai.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        tick_fps()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display = frame.copy()
        fh, fw = frame_rgb.shape[:2]

        ibug68 = None
        face_lm_mp = None
        bbox = None

        if FACE_BACKEND == "mediapipe":
            ibug68, face_lm_mp = extract_landmarks_mediapipe(frame_rgb)
            bbox = face_bbox_from_landmarks(ibug68, fw, fh)
        elif FACE_BACKEND == "dlib":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ibug68 = extract_landmarks_dlib(gray, frame_rgb.shape)
            bbox = face_bbox_from_landmarks(ibug68, fw, fh)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bbox = face_bbox_from_haar(gray)

        face_detected = bbox is not None

        draw_landmarks_on_frame(display, face_lm_mp, ibug68)
        if face_detected:
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (255, 160, 0), 1)

        # ---- Isi buffer ----
        if face_detected:
            if ibug68 is not None:
                buf_stgcn.append(ibug68)
            # [S2] ROI wajah, bukan seluruh frame.
            roi = crop_face_for_rppg(frame_rgb, bbox)
            if roi is not None:
                buf_rppg.append(roi)
        else:
            buf_stgcn.clear()
            buf_rppg.clear()

        # ---- [S7] Publikasi mutu sinyal (~2 Hz) ----
        if mq is not None and frame_count % 15 == 0:
            try:
                mq.publish_signal_quality(SignalQuality(
                    face_detected=face_detected,
                    roi_stability=round(
                        min(len(buf_stgcn) / float(CLIP_LEN_STGCN), 1.0), 3),
                ))
            except Exception as e:
                print("[MQTT] publish_signal_quality gagal: " + str(e))

        # ---- Forward pass ----
        with state_lock:
            busy = state["running"]
        if (not busy and len(buf_stgcn) >= CLIP_LEN_STGCN
                and len(buf_rppg) >= CLIP_LEN_RPPG):
            try:
                with state_lock:
                    state["running"] = True
                stgcn_np = np.array(list(buf_stgcn)).transpose(2, 0, 1)
                stgcn_t = torch.tensor(
                    stgcn_np, dtype=torch.float32).unsqueeze(0).to(device)
                rppg_np = np.array(
                    list(buf_rppg)[-CLIP_LEN_RPPG:]).transpose(3, 0, 1, 2)
                rppg_t = torch.tensor(
                    rppg_np, dtype=torch.float32).unsqueeze(0).to(device)
                threading.Thread(target=run_inference_thread,
                                 args=(stgcn_t, rppg_t), daemon=True).start()
                for _ in range(STRIDE):
                    if buf_stgcn:
                        buf_stgcn.popleft()
            except Exception as e:
                print("[Error] Gagal meluncurkan thread inferensi: " + str(e))
                with state_lock:
                    state["running"] = False
                buf_stgcn.clear()
                buf_rppg.clear()

        # ---- Crop & resize portrait ----
        if DISPLAY_PORTRAIT:
            oh, ow = display.shape[:2]
            target_aspect = SCREEN_WIDTH / float(SCREEN_HEIGHT)
            crop_w = int(oh * target_aspect)
            if crop_w <= ow:
                sx = (ow - crop_w) // 2
                display = display[:, sx:sx + crop_w]
            else:
                crop_h = int(ow / target_aspect)
                sy = max(0, (oh - crop_h) // 2)
                display = display[sy:sy + crop_h, :]
            display = cv2.resize(display, (SCREEN_WIDTH, SCREEN_HEIGHT))

        with state_lock:
            s_mood = state["mood"]
            s_bpm = state["bpm"]
            s_rr = state["rr"]
            s_conf = state["confidence"]
            s_phys = state["physio_used"]

        draw_status_overlay(
            display, s_mood, s_bpm, s_rr, s_conf,
            min(len(buf_stgcn) / float(CLIP_LEN_STGCN), 1.0),
            s_phys, face_detected)

        cv2.imshow(win, display)
        if fs_pending:
            cv2.waitKey(1)
            apply_fullscreen(win)
            fs_pending = False
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if mq is not None:
        try:
            mq.disconnect()
        except Exception:
            pass
    print("[Inference] Selesai.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # Ctrl+C bukan kegagalan program. Tanpa penanganan ini, traceback
        # panjang muncul di terminal dan terlihat seperti crash.
        print("")
        print("[Inference] Dihentikan oleh pengguna (Ctrl+C).")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        sys.exit(0)
