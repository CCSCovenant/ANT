# ./models/submodules/residual_block.py

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .basic_blocks import Conv1dBlock, Conv1dAdaGNBlock

class ResidualTemporalBlock(nn.Module):
    """
    Residual block with temporal convolutions and time embedding injection.
    包含时间卷积和时间编码注入的残差块。
    """
    def __init__(
        self,
        inp_channels,
        out_channels,
        embed_dim,
        kernel_size=5,
        zero=True,
        n_groups=8,
        dropout: float = 0.1,
        adagn=True
    ):
        super().__init__()
        self.adagn = adagn
        
        # The first conv block can be adaptive
        first_conv = (
            Conv1dAdaGNBlock(inp_channels, out_channels, kernel_size, n_groups) 
            if adagn 
            else Conv1dBlock(inp_channels, out_channels, kernel_size, n_groups)
        )
        
        self.blocks = nn.ModuleList([
            first_conv,
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups, zero=zero),
        ])

        # MLP for time embedding
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            # adagn needs scale and shift, so double the output channels
            nn.Linear(embed_dim, out_channels * 2 if adagn else out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        if zero:
            nn.init.zeros_(self.time_mlp[1].weight)
            nn.init.zeros_(self.time_mlp[1].bias)

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_embeds=None):
        '''
        Args:
            x (torch.Tensor): [batch_size, inp_channels, nframes]
                              输入张量
            t (torch.Tensor): [batch_size, embed_dim]
                              时间编码
        Returns:
            torch.Tensor: [batch_size, out_channels, nframes]
                          输出张量
        '''
        out = x
        
        time_cond = self.time_mlp(time_embeds)

        if self.adagn:
            scale, shift = time_cond.chunk(2, dim=1)
            out = self.blocks[0](out, scale, shift)
        else:
            out = self.blocks[0](out) + time_cond
            
        out = self.blocks[1](out)
        out = self.dropout(out)
        
        return out + self.residual_conv(x)