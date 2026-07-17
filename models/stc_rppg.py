import torch.nn as nn

from .fsfe import FSFE
from .stfe import STFE
from .predictor import rPPGPredictor

class STCrPPG(nn.Module):
    """
    Arsitektur Lengkap STC-rPPG Network
    """
    def __init__(self):
        super(STCrPPG, self).__init__()
        self.fsfe = FSFE(in_channels=3, out_channels=32)
        self.stfe = STFE(in_channels=32, out_channels=32)
        self.predictor = rPPGPredictor(in_channels=32)
        
    def forward(self, x):
        x = self.fsfe(x) 
        x = self.stfe(x) 
        rppg_signal = self.predictor(x) 
        return rppg_signal