import torch
from torch import nn
import torch.nn.functional as F

class FeatureJitter(nn.Module):

    def __init__(self, sigma=0.05, scale=0.1, dropblock_p=0.0, block_size=5):
        super().__init__()
        self.sigma = sigma
        self.scale = scale
        self.dropblock_p = dropblock_p
        self.block_size = block_size

    def forward(self, x, training=True):
        if not self.training or not training:
            return x
        B, C, H, W = x.shape
        if self.sigma > 0:
            s = x.detach().flatten(2).std(dim=2).clamp_min(1e-06).view(B, C,
                1, 1)
            x = x + torch.randn_like(x) * (self.sigma * s)
        if self.scale > 0:
            g = torch.randn(B, C, 1, 1, device=x.device) * self.scale + 1.0
            x = x * g
        if self.dropblock_p > 0:
            gamma = self.dropblock_p * (H * W) / self.block_size ** 2 / ((H -
                self.block_size + 1) * (W - self.block_size + 1))
            mask = (torch.rand(B, 1, H, W, device=x.device) < gamma).float()
            mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1,
                padding=self.block_size // 2)
            x = x * (1 - mask) * (mask.numel() / (mask.numel() - mask.sum()
                .clamp_min(1)))
        return x


def adain_jitter(x, m_scale=0.05, s_scale=0.05):
    B, C, H, W = x.shape
    mean = x.mean(dim=[2, 3], keepdim=True)
    std = x.std(dim=[2, 3], keepdim=True).clamp_min(1e-06)
    mean_j = mean + torch.randn_like(mean) * (m_scale * std)
    std_j = std * (1.0 + torch.randn_like(std) * s_scale)
    x_norm = (x - mean) / std
    return x_norm * std_j + mean_j
