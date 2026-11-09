from torch import nn


class BaseEncoder(nn.Module):
    def init_weights(self):
        pass

    def forward(self, feat, **kwargs):
        return feat


class BaseDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def init_weights(self):
        pass
