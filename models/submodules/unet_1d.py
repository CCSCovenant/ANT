# ./models/submodules/unet_1d.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_blocks import Downsample1d, Upsample1d
from .cond_conv_block import CondConv1DBlock
from .time_embedding import TimestepEmbedder

class CondUnet1D(nn.Module):
    """
    Conditional 1D U-Net for motion sequence denoising.
    用于运动序列去噪的条件一维U-Net。
    """
    def __init__(self, input_dim, cond_dim, dim=128, dim_mults=(1, 2, 4, 8), dims=None,
                 time_dim=512, adagn=True, zero=True, dropout=0.1, no_eff=False):
        super().__init__()
        if not dims:
            dims = [input_dim, *map(lambda m: int(dim * m), dim_mults)]
        print(f'[CondUnet1D] Dims: {dims}, Multipliers: {dim_mults}')
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_mlp = nn.Sequential(
            TimestepEmbedder(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                CondConv1DBlock(dim_in, dim_out, cond_dim, time_dim, adagn, zero, no_eff, dropout),
                CondConv1DBlock(dim_out, dim_out, cond_dim, time_dim, adagn, zero, no_eff, dropout),
                Downsample1d(dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = CondConv1DBlock(mid_dim, mid_dim, cond_dim, time_dim, adagn, zero, no_eff, dropout)
        self.mid_block2 = CondConv1DBlock(mid_dim, mid_dim, cond_dim, time_dim, adagn, zero, no_eff, dropout)

        last_dim = mid_dim
        for ind, dim_out in enumerate(reversed(dims[1:])):
            self.ups.append(nn.ModuleList([
                Upsample1d(last_dim, dim_out),
                CondConv1DBlock(dim_out * 2, dim_out, cond_dim, time_dim, adagn, zero, no_eff, dropout),
                CondConv1DBlock(dim_out, dim_out, cond_dim, time_dim, adagn, zero, no_eff, dropout),
            ]))
            last_dim = dim_out
            
        self.final_conv = nn.Conv1d(last_dim, input_dim, 1)

        if zero:
            nn.init.zeros_(self.final_conv.weight)
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, t, cond, cond_indices, return_attn_weights=False):
        temb = self.time_mlp(t)
        h = []
        attention_maps = {'down': [], 'mid': [], 'up': []} if return_attn_weights else None

        for i, (block1, block2, downsample) in enumerate(self.downs):
            if return_attn_weights:
                x, w1 = block1(x, temb, cond, cond_indices, return_attn_weights=True)
                x, w2 = block2(x, temb, cond, cond_indices, return_attn_weights=True)
                attention_maps['down'].append((w1, w2))
            else:
                x = block1(x, temb, cond, cond_indices)
                x = block2(x, temb, cond, cond_indices)
            h.append(x)
            x = downsample(x)

        if return_attn_weights:
            x, w_mid1 = self.mid_block1(x, temb, cond, cond_indices, return_attn_weights=True)
            x, w_mid2 = self.mid_block2(x, temb, cond, cond_indices, return_attn_weights=True)
            attention_maps['mid'].append((w_mid1, w_mid2))
        else:
            x = self.mid_block1(x, temb, cond, cond_indices)
            x = self.mid_block2(x, temb, cond, cond_indices)

        for i, (upsample, block1, block2) in enumerate(self.ups):
            x = upsample(x)
            x = torch.cat((x, h.pop()), dim=1)
            if return_attn_weights:
                x, w1 = block1(x, temb, cond, cond_indices, return_attn_weights=True)
                x, w2 = block2(x, temb, cond, cond_indices, return_attn_weights=True)
                attention_maps['up'].append((w1, w2))
            else:
                x = block1(x, temb, cond, cond_indices)
                x = block2(x, temb, cond, cond_indices)

        x = self.final_conv(x)
        
        if return_attn_weights:
            return x, attention_maps
        return x