import torch.nn as nn

from .fsfe import FSFE
from .stfe import STFE
from .predictor import rPPGPredictor

import yaml
import os

class STCrPPG(nn.Module):
    """
    Arsitektur Lengkap STC-rPPG Network
    """
    def __init__(self, config_path=None, **kwargs):
        super(STCrPPG, self).__init__()
        
        in_channels = kwargs.get('in_channels', 3)
        fsfe_out = kwargs.get('fsfe_out_channels', 32)
        stfe_out = kwargs.get('stfe_out_channels', 32)
        cdc_theta = kwargs.get('cdc_theta', 0.6)

        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                if 'model' in cfg:
                    in_channels = cfg['model'].get('in_channels', in_channels)
                    fsfe_out = cfg['model'].get('fsfe_out_channels', fsfe_out)
                    stfe_out = cfg['model'].get('stfe_out_channels', stfe_out)
                    cdc_theta = cfg['model'].get('cdc_theta', cdc_theta)

        self.fsfe = FSFE(in_channels=in_channels, out_channels=fsfe_out)
        self.stfe = STFE(in_channels=fsfe_out, out_channels=stfe_out, cdc_theta=cdc_theta)
        self.predictor = rPPGPredictor(in_channels=stfe_out)
        
    def forward(self, x):
        x = self.fsfe(x) 
        x = self.stfe(x) 
        rppg_signal = self.predictor(x) 
        return rppg_signal