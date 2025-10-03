# ./models/submodules/cond_conv_block.py

# ./models/cond_conv_block.py

import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.nn.functional as F
import torch

class CondConv1DBlock(nn.Module):

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
    ) -> None:
        super().__init__()
        self.conv1d = ResidualTemporalBlock(
            dim_in,
            dim_out,
            embed_dim=time_dim,
            adagn=adagn,
            zero=zero,
            dropout=dropout,
        )
        self.cross_attn = ResidualCrossAttentionLayer(
            dim1=dim_out,
            dim2=cond_dim,
            no_eff=no_eff,
            dropout=dropout,
        )

    def forward(self, x, t, cond, cond_indices=None):
        x = self.conv1d(x, t)
        x = self.cross_attn(x, cond, cond_indices)
        return x

class ResidualTemporalBlock(nn.Module):
    def __init__(self,
                 inp_channels,
                 out_channels,
                 embed_dim,
                 kernel_size=5,
                 zero=True,
                 n_groups=8,
                 dropout: float = 0.1,
                 adagn=True):
        super().__init__()
        self.adagn = adagn
        
        self.blocks = nn.ModuleList([
            # adagn only the first conv (following guided-diffusion)
            (Conv1dAdaGNBlock(inp_channels, out_channels, kernel_size, n_groups) if adagn
            else Conv1dBlock(inp_channels, out_channels, kernel_size)),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups, zero=zero),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            # adagn = scale and shift
            nn.Linear(embed_dim, out_channels * 2 if adagn else out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        self.dropout = nn.Dropout(dropout)    
        if zero:
            nn.init.zeros_(self.time_mlp[1].weight)
            nn.init.zeros_(self.time_mlp[1].bias)

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, time_embeds=None):
        '''
            x : [ batch_size x inp_channels x nframes ]
            t : [ batch_size x embed_dim ]
            returns: [ batch_size x out_channels x nframes ]
        '''
        if self.adagn:
            scale, shift = self.time_mlp(time_embeds).chunk(2, dim=1)
            out = self.blocks[0](x, scale, shift)
        else:
            out = self.blocks[0](x) + self.time_mlp(time_embeds)
        out = self.blocks[1](out)
        out = self.dropout(out)
        return out + self.residual_conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''
    def __init__(self,
                 inp_channels,
                 out_channels,
                 kernel_size,
                 n_groups=4,
                 zero=False):
        super().__init__()
        self.out_channels = out_channels
        self.block =nn.Conv1d(inp_channels,
                      out_channels,
                      kernel_size,
                      padding=kernel_size // 2)
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.activation =  nn.Mish()

        if zero:
            # zero init the convolution
            nn.init.zeros_(self.block.weight)
            nn.init.zeros_(self.block.bias)

    def forward(self, x):
        """
        Args:
            x: [bs, nfeat, nframes]
        """
        x = self.block(x)

        batch_size, channels, horizon = x.size()
        x = rearrange(x,'batch channels horizon -> (batch horizon) channels') # [bs*seq, nfeats]
        x = self.norm(x)
        x = rearrange(x.reshape(batch_size,horizon,channels),'batch horizon channels -> batch channels horizon')

        return self.activation(x)

class Conv1dAdaGNBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> scale,shift --> Mish
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=4):
        super().__init__()
        self.out_channels = out_channels
        self.block = nn.Conv1d(inp_channels,
                      out_channels,
                      kernel_size,
                      padding=kernel_size // 2)
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        self.avtication = nn.Mish()

    def forward(self, x, scale, shift):
        """
        Args:
            x: [bs, nfeat, nframes]
            scale: [bs, out_feat, 1]
            shift: [bs, out_feat, 1]
        """
        x = self.block(x)

        batch_size, channels, horizon = x.size()
        x = rearrange(x,'batch channels horizon -> (batch horizon) channels') # [bs*seq, nfeats]
        x = self.group_norm(x)
        x = rearrange(x.reshape(batch_size,horizon,channels),'batch horizon channels -> batch channels horizon')

        x = ada_shift_scale(x, shift, scale)

        return self.avtication(x)
def ada_shift_scale(x, shift, scale):
    return x * (1 + scale) + shift

class LinearCrossAttention(nn.Module):
    def __init__(
        self, 
        latent_dim, 
        text_latent_dim, 
        num_heads:int = 8,
        dropout: float = 0.0
        ):
        super().__init__()
        self.num_head = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_tensor, condition_tensor):
        """
        input_tensor: B, T, D  
        condition_tensor: B, N, L 
        """
        B, T, D = input_tensor.shape
        N = condition_tensor.shape[1]    
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(input_tensor))
        # B, N, D
        key = self.key(self.text_norm(condition_tensor))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(condition_tensor)).view(B, N, H, -1)
        # B, H, HD, HD
        attention = self.dropout(torch.einsum('bnhd,bnhl->bhdl', key, value))
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        return y
    
    def forward_w_weight(self, input_tensor, condition_tensor):
        """
        与 forward 类似，但额外返回计算得到的注意力权重
        input_tensor: [B, T, D]
        condition_tensor: [B, N, L]
        返回:
            y: [B, T, D] 最终输出
            attention: [B, H, HD, HD] 计算后的注意力权重（经过 dropout）
        """
        B, T, D = input_tensor.shape
        N = condition_tensor.shape[1]
        H = self.num_head
        # 计算 query、key、value
        query = self.query(self.norm(input_tensor))
        key = self.key(self.text_norm(condition_tensor))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        value = self.value(self.text_norm(condition_tensor)).view(B, N, H, -1)
        # 计算注意力权重
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        attention = self.dropout(attention)
        # 根据 attention 计算最终输出
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        return y, attention


class ResidualCrossAttentionLayer(nn.Module):
    def __init__(
        self, 
        dim1, 
        dim2, 
        num_heads:int = 8,
        dropout: float = 0.1,
        no_eff: bool = False
    ):
        super(ResidualCrossAttentionLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        
        # Multi-Head Attention Layer
        if no_eff:
            self.cross_attention = CrossAttention(
                latent_dim=dim1, 
                text_latent_dim = dim2,
                num_heads=num_heads,
                dropout=dropout
            )  
        else:
             self.cross_attention = LinearCrossAttention(
                latent_dim=dim1, 
                text_latent_dim = dim2,
                num_heads=num_heads,
                dropout=dropout
            )  
        
    def forward(self, input_tensor, condition_tensor, cond_indices):
        '''
        input_tensor :B, D, L
        condition_tensor: B, L, D
        '''
        if cond_indices.numel() == 0:
            return input_tensor
        
        x = input_tensor

        # Ensure that the dimensions match for the MultiheadAttention
        x = x[cond_indices].permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
        
        # Compute cross-attention
        x = self.cross_attention(x, condition_tensor[cond_indices])
        
        # Rearrange output tensor
        x = x.permute(0, 2, 1)  # (batch_size, feat_dim, seq_length)
        
        input_tensor[cond_indices] = input_tensor[cond_indices] + x
        return  input_tensor