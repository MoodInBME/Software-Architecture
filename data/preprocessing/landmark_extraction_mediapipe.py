"""Ekstraksi landmark wajah MediaPipe FaceMesh (468) -> subset iBUG-68.

Dipakai oleh:
  - data/preprocessing/extract_landmarks_video.py (ekstraksi + cache)
  - models/st_gcn_graph_tran/graph.py (topologi 68 titik)

PENTING: pemetaan MEDIAPIPE_TO_IBUG68 di bawah adalah pemetaan yang UMUM
dipakai komunitas (FaceMesh -> dlib/iBUG 68), BUKAN angka pasti dari paper.
Indeks WAJIB diverifikasi secara visual (overlay pada wajah nyata) sebelum
training sungguhan -- gunakan verify_mapping(). Lihat backlog docs/arsip-gemastik/ST-GCN-PLAN.md.
"""
from __future__ import annotations

import numpy as np

NUM_MEDIAPIPE_LANDMARKS = 468

# 68 indeks FaceMesh (0-based) yang dipetakan ke konvensi iBUG-68.
MEDIAPIPE_TO_IBUG68 = [
    # Rahang / jaw (0-16): 17 titik mengikuti kontur wajah
    127, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    # Alis kanan (17-21)
    70, 63, 105, 66, 107,
    # Alis kiri (22-26)
    336, 296, 334, 293, 300,
    # Batang hidung (27-30)
    168, 197, 5, 4,
    # Dasar hidung (31-35)
    75, 97, 2, 326, 305,
    # Mata kanan (36-41)
    33, 160, 158, 133, 153, 144,
    # Mata kiri (42-47)
    362, 385, 387, 263, 373, 380,
    # Mulut luar (48-59)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # Mulut dalam (60-67)
    78, 82, 13, 312, 308, 317, 14, 87,
]

assert len(MEDIAPIPE_TO_IBUG68) == 68, "Pemetaan harus tepat 68 titik"


def mediapipe_to_ibug68(landmarks_468):
    """Ambil subset 68 titik dari array (>=468, D). D=2 (x,y) atau 3 (x,y,z)."""
    arr = np.asarray(landmarks_468)
    if arr.shape[0] < NUM_MEDIAPIPE_LANDMARKS:
        raise ValueError(
            f"Diharapkan >= {NUM_MEDIAPIPE_LANDMARKS} landmark, dapat {arr.shape[0]}"
        )
    return arr[MEDIAPIPE_TO_IBUG68]


def normalize_landmarks(pts68, eps=1e-6):
    """Normalisasi invarian translasi + skala.

    - Translasi: pusatkan ke centroid seluruh titik.
    - Skala: bagi dengan jarak antar sudut mata luar (inter-ocular) agar
      invarian terhadap ukuran/jarak wajah ke kamera.
    pts68: (68, >=2) -> return (68, 2) float32.
    """
    pts = np.asarray(pts68, dtype=np.float32)[:, :2].copy()
    pts -= pts.mean(axis=0, keepdims=True)
    inter_ocular = np.linalg.norm(pts[36] - pts[45]) + eps  # 36 & 45 = sudut mata luar
    pts /= inter_ocular
    return pts.astype(np.float32)


class MediaPipeLandmarkExtractor:
    """Pembungkus MediaPipe FaceMesh.

    Impor mediapipe ditunda (lazy) supaya modul ini tetap bisa di-import di
    lingkungan tanpa mediapipe (mis. hanya untuk memakai MEDIAPIPE_TO_IBUG68
    atau untuk py_compile).
    """

    def __init__(self, static_image_mode=False, max_num_faces=1,
                 refine_landmarks=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        import mediapipe as mp  # lazy import
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_rgb):
        """frame_rgb: (H,W,3) uint8 RGB. Return (68,2) ternormalisasi atau None."""
        h, w = frame_rgb.shape[:2]
        res = self.face_mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)  # (468,2) px
        return normalize_landmarks(mediapipe_to_ibug68(pts))

    def close(self):
        self.face_mesh.close()


def verify_mapping(image_path, out_path="mapping_check.png"):
    """Overlay 68 titik hasil pemetaan ke satu gambar untuk verifikasi visual.

    Jalankan ini pada beberapa wajah sebelum training untuk memastikan urutan
    & posisi titik benar-benar mengikuti konvensi iBUG-68.
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ext = MediaPipeLandmarkExtractor(static_image_mode=True)
    h, w = img.shape[:2]
    res = ext.face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        print("Tidak ada wajah terdeteksi.")
        ext.close()
        return
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    pts68 = pts[MEDIAPIPE_TO_IBUG68]
    for i, (x, y) in enumerate(pts68):
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.putText(img, str(i), (int(x) + 1, int(y) - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)
    cv2.imwrite(out_path, img)
    ext.close()
    print(f"Tersimpan: {out_path} (periksa urutan & posisi 68 titik).")


if __name__ == "__main__":
    print("Jumlah titik peta:", len(MEDIAPIPE_TO_IBUG68))
    dummy = np.random.rand(468, 2).astype(np.float32)
    sub = mediapipe_to_ibug68(dummy)
    print("Subset shape:", sub.shape)
    print("Normalisasi shape:", normalize_landmarks(sub).shape)
