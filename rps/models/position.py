# Adapted from MTL-TabNet, TableMASTER, and OpenMMLab components.
# Changes: rps package imports and framework-independent integration.
# Upstream notices: THIRD_PARTY_NOTICES.md and licenses/.
import math
import torch
from torch import nn
from .base import BaseEncoder

class PositionalEncoding(BaseEncoder):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.
            log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, feat, **kwargs):
        if len(feat.shape) > 3:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h * w)
            feat = feat.permute((0, 2, 1))
        feat = feat + self.pe[:, :feat.size(1)]
        return self.dropout(feat)

    def init_weights(self):
        pass



class my_PositionalEncoding(BaseEncoder):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(my_PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.
            log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, feat, **kwargs):
        if len(feat.shape) > 3:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h * w)
            feat = feat.permute((0, 2, 1))
        feat = feat + self.pe[:, :feat.size(1)]
        return self.dropout(feat)

    def init_weights(self):
        pass
