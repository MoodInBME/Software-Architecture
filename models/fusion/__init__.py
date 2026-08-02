"""Modul fusi multimodal: VA-bridge berbasis aturan + MLP dual-head terlatih."""
from .late_fusion import LateFusionVABridge, physio_to_arousal

__all__ = ["LateFusionVABridge", "physio_to_arousal"]
