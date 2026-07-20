"""Pra-pemrosesan video wajah berbasis MediaPipe FaceMesh (pengganti dlib - S2).

Satu detektor menghasilkan:
- ROI wajah (crop + resize) untuk cabang STC-rPPG, bentuk keluaran (C, T, H, W)
  TANPA dimensi batch (memperbaiki bug bentuk Dataset - S5).
- (opsional) koordinat landmark ternormalisasi (C=2, T, V) untuk cabang ST-GCN.

Padding memakai 'edge'/'loop' (bukan nol) agar tidak menyuntik artefak spektral (S6).
"""
import cv2
import numpy as np
import torch

try:
    import mediapipe as mp
except ImportError:  # biar file tetap bisa di-import saat mediapipe belum ada
    mp = None


class FaceMeshVideoProcessor:
    def __init__(self, target_size=(128, 128), seq_length=128,
                 min_det_conf=0.5, margin=0.15, pad_mode="edge"):
        if mp is None:
            raise ImportError(
                "mediapipe belum terpasang. Jalankan: pip install mediapipe==0.10.14"
            )
        self.target_size = target_size
        self.seq_length = int(seq_length)
        self.margin = float(margin)
        self.pad_mode = pad_mode
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_det_conf,
        )

    def _bbox(self, lms, w, h):
        xs = np.array([p.x for p in lms]) * w
        ys = np.array([p.y for p in lms]) * h
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        mw, mh = (x1 - x0) * self.margin, (y1 - y0) * self.margin
        return (int(max(0, x0 - mw)), int(max(0, y0 - mh)),
                int(min(w, x1 + mw)), int(min(h, y1 + mh)))

    def process_video(self, video_path, return_landmarks=False):
        cap = cv2.VideoCapture(video_path)
        frames, landmarks = [], []
        try:
            while cap.isOpened() and len(frames) < self.seq_length:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self._mesh.process(rgb)
                if not res.multi_face_landmarks:
                    continue
                lms = res.multi_face_landmarks[0].landmark
                h, w = rgb.shape[:2]
                x0, y0, x1, y1 = self._bbox(lms, w, h)
                crop = rgb[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, self.target_size).astype(np.float32) / 255.0
                frames.append(crop)
                if return_landmarks:
                    landmarks.append(
                        np.array([[p.x, p.y] for p in lms], dtype=np.float32)
                    )
        finally:
            cap.release()

        if len(frames) == 0:
            return (None, None) if return_landmarks else None

        frames = self._pad(frames)
        video = np.transpose(np.asarray(frames), (3, 0, 1, 2))  # (C,T,H,W)
        video_t = torch.from_numpy(np.ascontiguousarray(video))

        if return_landmarks:
            landmarks = self._pad(landmarks)
            lm = np.transpose(np.asarray(landmarks), (2, 0, 1))  # (C=2,T,V)
            return video_t, torch.from_numpy(np.ascontiguousarray(lm))
        return video_t

    def _pad(self, seq):
        n = len(seq)
        if n >= self.seq_length:
            return seq[:self.seq_length]
        if self.pad_mode == "loop":
            idx = np.arange(self.seq_length) % n
        else:  # 'edge': ulangi frame terakhir
            idx = np.clip(np.arange(self.seq_length), 0, n - 1)
        return [seq[i] for i in idx]

    def close(self):
        self._mesh.close()


if __name__ == "__main__":
    proc = FaceMeshVideoProcessor(seq_length=128)
    out = proc.process_video("sample.mp4")
    print("Bentuk tensor:", None if out is None else tuple(out.shape))
    proc.close()
