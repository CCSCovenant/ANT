# ./models/submodules/basic_blocks.py

import torch
import torch.nn as nn
from einops import rearrange

def ada_shift_scale(x, shift, scale):
    """
    Helper function for adaptive shifting and scaling.
    自适应位移和缩放的辅助函数。
    """
    return x * (1 + scale) + shift

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

class Conv1dBlock(nn.Module):
    '''
    Standard 1D convolution block: Conv1d -> GroupNorm -> Mish.
    标准一维卷积块：Conv1d -> GroupNorm -> Mish。
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=4, zero=False):
        super().__init__()
        self.block = nn.Conv1d(
            inp_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.activation = nn.Mish()

        if zero:
            nn.init.zeros_(self.block.weight)
            nn.init.zeros_(self.block.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [bs, nfeat, nframes].
                              输入张量 [bs, nfeat, nframes]。
        """
        x = self.block(x)
        # GroupNorm requires shape [N, C, *]
        # To normalize over feature dimension, we rearrange
        batch_size, channels, horizon = x.size()
        x_reshaped = x.view(batch_size * horizon, channels)
        x_normed = self.norm(x_reshaped)
        x = x_normed.view(batch_size, channels, horizon)
        
        return self.activation(x)

class Conv1dAdaGNBlock(nn.Module):
    '''
    1D convolution block with adaptive GroupNorm: Conv1d -> GroupNorm -> AdaGN (scale, shift) -> Mish.
    带自适应GroupNorm的一维卷积块：Conv1d -> GroupNorm -> AdaGN (缩放, 位移) -> Mish。
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=4):
        super().__init__()
        self.block = nn.Conv1d(
            inp_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        self.activation = nn.Mish()

    def forward(self, x, scale, shift):
        """
        Args:
            x (torch.Tensor): Input tensor [bs, nfeat, nframes].
                              输入张量 [bs, nfeat, nframes]。
            scale (torch.Tensor): Scale tensor [bs, out_feat, 1].
                                  缩放张量 [bs, out_feat, 1]。
            shift (torch.Tensor): Shift tensor [bs, out_feat, 1].
                                  位移张量 [bs, out_feat, 1]。
        """
        x = self.block(x)
        
        batch_size, channels, horizon = x.size()
        x_reshaped = x.view(batch_size * horizon, channels)
        x_normed = self.group_norm(x_reshaped)
        x = x_normed.view(batch_size, channels, horizon)
        
        x = ada_shift_scale(x, shift, scale)

        return self.activation(x)