# ./models/submodules/cond_conv_block.py

import torch.nn as nn

from .residual_block import ResidualTemporalBlock
from .attention import ResidualCrossAttentionLayer

class CondConv1DBlock(nn.Module):
    """
    A block combining a ResidualTemporalBlock with a ResidualCrossAttentionLayer.
    一个结合了残差时间块和残差跨注意力层的模块。
    """
    def __init__(
        self,
        dim_in,
        dim_out,
        cond_dim,
        time_dim,
        adagn=True,
        zero=True,
        no_eff=False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1d = ResidualTemporalBlock(
            dim_in, dim_out, embed_dim=time_dim, adagn=adagn, zero=zero, dropout=dropout
        )
        self.cross_attn = ResidualCrossAttentionLayer(
            dim1=dim_out, dim2=cond_dim, no_eff=no_eff, dropout=dropout
        )

    def forward(self, x, t, cond, cond_indices=None, return_attn_weights=False):
        """
        Forward pass with optional attention weight return.
        前向传播，可选择性返回注意力权重。
        """
        x = self.conv1d(x, t)
        
        if return_attn_weights:
            x, attn_weights = self.cross_attn(x, cond, cond_indices, return_weights=True)
            return x, attn_weights
        else:
            x = self.cross_attn(x, cond, cond_indices, return_weights=False)
            return x