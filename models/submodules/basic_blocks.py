# ./models/submodules/basic_blocks.py

import torch
import torch.nn as nn
from einops import rearrange


class Downsample1d(nn.Module):
    """
    Downsampling layer for 1D data.
    一维数据的下采样层。
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    """
    Upsampling layer for 1D data.
    一维数据的上采样层。
    """
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim_in
        self.conv = nn.ConvTranspose1d(dim_in, dim_out, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)