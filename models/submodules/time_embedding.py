# ./models/submodules/time_embedding.py

import torch
import torch.nn as nn
import numpy as np
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

class TimestepEmbedder(nn.Module):
    """
    Sinusoidal Positional Embedding for Timesteps.
    用于时间步的正弦位置编码。
    """
    def __init__(self, d_model, max_len=5000):
        super(TimestepEmbedder, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor of timesteps, shape [batch_size].
                              时间步张量，形状为 [batch_size]。
        Returns:
            torch.Tensor: Embedded timesteps, shape [batch_size, d_model].
                          编码后的时间步，形状为 [batch_size, d_model]。
        """
        return self.pe[x]