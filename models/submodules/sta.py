# ./models/submodules/sta.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.square(torch.relu(x))

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with time embedding.
    带时间编码的自适应层归一化。
    """
    def __init__(self, embedding_dim: int, time_embedding_dim: int = None):
        super().__init__()
        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, timestep_embedding):
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x

class PerceiverAttentionBlock(nn.Module):
    """
    Attention block for the Perceiver model.
    Perceiver模型的注意力块。
    """
    def __init__(self, d_model, n_heads, time_embedding_dim=None, is_abstractor=True, enable_noise=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.is_abstractor = is_abstractor
        self.enable_noise = enable_noise
        
        if not is_abstractor:
            self.mlp = nn.Sequential(
                OrderedDict([
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ])
            )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        # 改动（中文注释）：仅在非抽象器模式下创建 ln_ff；抽象器模式(is_abstractor=True)不会用到该层，避免注册未使用参数导致DDP报错。
        if not is_abstractor:
            self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)
        else:
            self.ln_ff = None  # 中文注释：抽象器模式下不创建 ln_ff，从根源移除未使用的可训练参数

    def attention(self, q, kv, return_weights=False):
        if return_weights:
            return self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        attn_output, _ = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(self, x, latents, timestep_embedding=None, return_weights=False):
        normed_latents = self.ln_1(latents, timestep_embedding)
        
        k = torch.randn_like(normed_latents) if self.enable_noise else normed_latents
        kv = torch.cat([k, self.ln_2(x, timestep_embedding)], dim=1)
        
        if return_weights:
            attn_output, weights = self.attention(q=normed_latents, kv=kv, return_weights=True)
            latents = latents + attn_output
        else:
            latents = latents + self.attention(q=normed_latents, kv=kv)

        if not self.is_abstractor:
            latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))  # 中文注释：仅非抽象器模式下使用 ln_ff + MLP

        if return_weights:
            return latents, weights
        return latents

class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler to compress sequence length.
    用于压缩序列长度的Perceiver重采样器。
    """
    def __init__(self, width: int=768, layers: int=6, heads: int=8, num_latents: int=64,
                 output_dim=None, input_dim=None, time_embedding_dim=None,
                 is_abstractor=True, enable_noise=False):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
        self.time_aware_linear = nn.Linear(time_embedding_dim or width, width, bias=True)

        if self.input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)
            print(f"proj_in shape:hyper:{input_dim},{width}")

        self.perceiver_blocks = nn.ModuleList([
            PerceiverAttentionBlock(width, heads, time_embedding_dim=time_embedding_dim,
                                    is_abstractor=is_abstractor, enable_noise=enable_noise)
            for _ in range(layers)
        ])

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(nn.Linear(width, output_dim), nn.LayerNorm(output_dim))

    def forward(self, x, timestep_embedding=None, return_attn_weights=False):
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
        latents = learnable_latents + self.time_aware_linear(F.silu(timestep_embedding))
        
        if self.input_dim is not None:
            x = self.proj_in(x)
        
        all_weights = []
        for p_block in self.perceiver_blocks:
            if return_attn_weights:
                latents, weights = p_block(x, latents, timestep_embedding=timestep_embedding, return_weights=True)
                all_weights.append(weights)
            else:
                latents = p_block(x, latents, timestep_embedding=timestep_embedding)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        if return_attn_weights:
            return latents, all_weights
        return latents

class STA(nn.Module):
    """
    STA connector module.
    STA连接器模块。
    """
    def __init__(self, time_channel=320, time_embed_dim=512, act_fn="silu", out_dim=None,
                 width=256, layers=6, heads=8, num_latents=77, input_dim=2048,
                 is_abstractor=True, enable_noise=False):
        super().__init__()
        self.position = Timesteps(time_channel, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel, time_embed_dim=time_embed_dim, act_fn=act_fn, out_dim=out_dim
        )
        self.connector = PerceiverResampler(
            width=width, layers=layers, heads=heads, num_latents=num_latents,
            input_dim=input_dim, time_embedding_dim=time_embed_dim,
            is_abstractor=is_abstractor, enable_noise=enable_noise, output_dim=out_dim
        )

    def forward(self, text_encode_features, timesteps, return_attn_weights=False):
        device = text_encode_features.device
        dtype = text_encode_features.dtype

        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        if ori_time_feature.ndim == 2:
            ori_time_feature = ori_time_feature.unsqueeze(dim=1)
        ori_time_feature = ori_time_feature.expand(len(text_encode_features), -1, -1)
        
        time_embedding = self.time_embedding(ori_time_feature)
        
        if return_attn_weights:
            encoder_hidden_states, attn_weights = self.connector(
                text_encode_features, timestep_embedding=time_embedding, return_attn_weights=True
            )
            return encoder_hidden_states, attn_weights
        
        encoder_hidden_states = self.connector(
            text_encode_features, timestep_embedding=time_embedding
        )
        return encoder_hidden_states