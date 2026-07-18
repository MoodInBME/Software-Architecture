"""
Facial-landmark graph untuk MoodIn ST-GCN + Graph-Transformer.

Membangun matriks adjacency ternormalisasi dengan strategi partisi spasial,
mengikuti Yan, Xiong, Lin (AAAI 2018) "Spatial Temporal Graph Convolutional
Networks for Skeleton-Based Action Recognition", diadaptasi dari graf skeleton
tubuh ke graf landmark wajah.

Catatan:
- Topologi default memakai konvensi 68 titik (iBUG/dlib). Jika tim memakai
  MediaPipe FaceMesh (468 titik), petakan dulu ke subset 68 titik ini
  (lihat data/preprocessing/landmark_extraction_mediapipe.py), atau ganti
  `num_node` dan daftar edge sesuai subset Anda.
"""
from __future__ import annotations

import numpy as np

# --- Region wajah pada konvensi 68 titik (iBUG/dlib) ---
_JAW = list(range(0, 17))
_EYEBROW_R = list(range(17, 22))
_EYEBROW_L = list(range(22, 27))
_NOSE_BRIDGE = list(range(27, 31))
_NOSE_BOTTOM = list(range(31, 36))
_EYE_R = list(range(36, 42))
_EYE_L = list(range(42, 48))
_MOUTH_OUT = list(range(48, 60))
_MOUTH_IN = list(range(60, 68))


def _chain(indices, closed=False):
    edges = [(indices[i], indices[i + 1]) for i in range(len(indices) - 1)]
    if closed and len(indices) > 2:
        edges.append((indices[-1], indices[0]))
    return edges


def default_face_edges():
    """Daftar edge antar-tetangga untuk graf wajah 68 titik."""
    edges = []
    edges += _chain(_JAW)
    edges += _chain(_EYEBROW_R)
    edges += _chain(_EYEBROW_L)
    edges += _chain(_NOSE_BRIDGE)
    edges += _chain(_NOSE_BOTTOM)
    edges += _chain(_EYE_R, closed=True)
    edges += _chain(_EYE_L, closed=True)
    edges += _chain(_MOUTH_OUT, closed=True)
    edges += _chain(_MOUTH_IN, closed=True)
    # Link antar-region agar graf terhubung penuh
    edges += [
        (27, 28),
        (21, 27), (22, 27),
        (39, 27), (42, 27),
        (30, 33),
        (33, 51),
        (48, 3), (54, 13),
        (36, 17), (45, 26),
    ]
    return edges


class FaceGraph:
    """
    Membentuk adjacency `A` (K, V, V) untuk graf landmark wajah.

    strategy:
      - 'uniform'  : 1 subset
      - 'distance' : (max_hop + 1) subset
      - 'spatial'  : 3 subset (root/centripetal/centrifugal) -- default ST-GCN
    """

    def __init__(self, num_node=68, edges=None, strategy="spatial", max_hop=1, center=33):
        self.num_node = num_node
        self.max_hop = max_hop
        self.strategy = strategy
        self.center = center
        self_link = [(i, i) for i in range(num_node)]
        neighbor = list(edges) if edges is not None else default_face_edges()
        neighbor = [(j, i) for (i, j) in neighbor] + neighbor
        self.edge = self_link + neighbor
        self._build_adjacency()

    @staticmethod
    def _normalize_digraph(A):
        Dl = np.sum(A, 0)
        n = A.shape[0]
        Dn = np.zeros((n, n))
        for i in range(n):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        return A @ Dn

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive = (np.stack(transfer) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive[d]] = d
        return hop_dis

    def _build_adjacency(self):
        valid_hop = range(0, self.max_hop + 1)
        hop_dis = self._get_hop_distance()
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        norm = self._normalize_digraph(adjacency)

        if self.strategy == "uniform":
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = norm
        elif self.strategy == "distance":
            A = np.zeros((self.max_hop + 1, self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = norm[hop_dis == hop]
        elif self.strategy == "spatial":
            c = self.center
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_far = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if hop_dis[j, i] == hop:
                            if hop_dis[j, c] == hop_dis[i, c]:
                                a_root[j, i] = norm[j, i]
                            elif hop_dis[j, c] > hop_dis[i, c]:
                                a_close[j, i] = norm[j, i]
                            else:
                                a_far[j, i] = norm[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_far)
            A = np.stack(A)
        else:
            raise ValueError(f"Strategy tidak dikenal: {self.strategy}")
        self.A = A.astype(np.float32)


if __name__ == "__main__":
    g = FaceGraph(strategy="spatial", max_hop=1)
    print("Bentuk adjacency A:", g.A.shape)
