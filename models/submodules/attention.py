# ./models/submodules/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    """
    Standard full cross-attention mechanism.
    标准的全连接跨注意力机制。
    """
    def __init__(self, latent_dim, text_latent_dim, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_head = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, xf, return_weights=False):
        """
        Args:
            x (torch.Tensor): Motion features, shape [B, T, D]
                              动态特征
            xf (torch.Tensor): Text condition features, shape [B, N, L]
                               文本条件特征
            return_weights (bool): If True, returns attention weights along with output.
                                   如果为True，则同时返回注意力权重。
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        
        query = self.query(self.norm(x)).view(B, T, H, -1)
        key = self.key(self.text_norm(xf)).view(B, N, H, -1)
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        
        # B, H, T, N
        attention_scores = torch.einsum('bthd,bnhd->bhtn', query, key) / math.sqrt(D // H)
        attention_weights = self.dropout(F.softmax(attention_scores, dim=-1))
        
        # B, H, T, D_h
        y = torch.einsum('bhtn,bnhd->bthd', attention_weights, value).reshape(B, T, D)

        if return_weights:
            return y, attention_weights
        return y

class LinearCrossAttention(nn.Module):
    """
    Efficient linear cross-attention.
    高效的线性跨注意力机制。
    """
    def __init__(self, latent_dim, text_latent_dim, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_head = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor, condition_tensor, return_weights=False):
        """
        Args:
            input_tensor (torch.Tensor): Motion features, shape [B, T, D]
                                         动态特征
            condition_tensor (torch.Tensor): Text condition features, shape [B, N, L]
                                             文本条件特征
            return_weights (bool): If True, returns attention context matrix.
                                   如果为True，返回注意力上下文矩阵。
        """
        B, T, D = input_tensor.shape
        N = condition_tensor.shape[1]
        H = self.num_head

        query = self.query(self.norm(input_tensor))
        key = self.key(self.text_norm(condition_tensor))
        value = self.value(self.text_norm(condition_tensor))
        
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        value = value.view(B, N, H, -1)

        # B, H, HD_k, HD_v
        context = torch.einsum('bnhk,bnhv->bhkv', key, value)
        attention = self.dropout(context)
        
        # B, T, D
        y = torch.einsum('bthk,bhkv->bthv', query, attention).reshape(B, T, D)
        
        if return_weights:
            return y, attention  # Returning the context matrix as 'weights'
        return y

class ResidualCrossAttentionLayer(nn.Module):
    """
    A residual layer that applies cross-attention.
    一个应用了跨注意力的残差层。
    """
    def __init__(self, dim1, dim2, num_heads: int = 8, dropout: float = 0.1, no_eff: bool = False):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        
        AttentionModule = CrossAttention if no_eff else LinearCrossAttention
        self.cross_attention = AttentionModule(
            latent_dim=dim1,
            text_latent_dim=dim2,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, input_tensor, condition_tensor, cond_indices, return_weights=False):
        """
        Args:
            input_tensor (torch.Tensor): Shape [B, D, L]
            condition_tensor (torch.Tensor): Shape [B, L, D]
            cond_indices (torch.Tensor): Indices for applying conditioning.
                                          应用条件的索引。
            return_weights (bool): Whether to return attention weights.
                                   是否返回注意力权重。
        """
        if cond_indices.numel() == 0:
            if return_weights:
                return input_tensor, None
            return input_tensor
            
        x_cond = input_tensor[cond_indices].permute(0, 2, 1)  # (B_cond, L, D)
        cond = condition_tensor[cond_indices]

        attn_output, weights = self.cross_attention(x_cond, cond, return_weights=True)
        
        attn_output = attn_output.permute(0, 2, 1)  # (B_cond, D, L)
        
        output_tensor = input_tensor.clone()
        output_tensor[cond_indices] = output_tensor[cond_indices] + attn_output
        
        if return_weights:
            return output_tensor, weights
        return output_tensor