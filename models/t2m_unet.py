import torch
import torch.nn as nn
import torch.nn.functional as F
# 可选依赖：clip。在非 CLIP 文本编码器场景下无需导入。
try:
    import clip  # noqa: F401
except Exception:
    clip = None
#import nvtx
from .submodules.unet_1d import CondUnet1D
from .submodules.sta import STA
from .submodules.text_processors import create_text_processor      # NEW

class T2MUnet(nn.Module):
    """
    Top-level Text-to-Motion U-Net model.
    """

    # --------------------------------------------------------------------- #
    # 1. 初始化
    # --------------------------------------------------------------------- #
    def __init__(self, config, text_encoder, text_encoder_dim):
        super().__init__()
        self.config   = config
        self.device   = config.device
        self.input_feats     = config.dim_pose
        self.text_encoder_tp = config.text_encoder_type     # ↓ 旧名字保留兼容
        self.cond_mask_prob  = config.cond_mask_prob

        # ---------------- 1.1 冻结大模型 ---------------- #
        self.text_encoder = text_encoder    # 已在外部冻结

        # ---------------- 1.2 可训练投影层 ---------------- #
        text_latent_dim = config.text_latent_dim
        self.text_proj  = nn.Linear(text_encoder_dim, text_latent_dim)
        self.text_ln    = nn.LayerNorm(text_latent_dim)

        # 仅 CLIP 需要额外 Transformer
        self.text_transformer = None
        if self.text_encoder_tp.lower() == 'clip':
            trans_layer = nn.TransformerEncoderLayer(
                d_model=text_latent_dim,
                nhead=config.text_num_heads,
                dim_feedforward=config.text_ff_size,
                dropout=config.dropout,
                activation=config.activation,
            )
            self.text_transformer = nn.TransformerEncoder(
                trans_layer, num_layers=config.num_text_layers
            )

        # ---------------- 1.3 文本策略 ---------------- #
        self.text_processor = create_text_processor(
            config,
            text_encoder       = self.text_encoder,
            text_proj          = self.text_proj,
            text_ln            = self.text_ln,
            text_transformer   = self.text_transformer,
        )

        # ---------------- 1.4 STA ---------------- #
        self.use_sta = not getattr(config, "disable_sta", False)
        if self.use_sta:
            print(f"Building STA connector,text_encoder_dim:{text_encoder_dim}")
            self.sta_model = STA(
                # 改动1（中文注释）：当开启 STA 时，输入维度应当为"原始文本编码器维度"，以复现旧逻辑"STA 内部自己做输入映射（proj_in）"。
                # 之前是传入 text_latent_dim（例如256），会造成与旧权重不一致（例如T5应为2048）。
                input_dim=text_encoder_dim,
                is_abstractor=getattr(config, "is_abstractor", True),
                enable_noise=getattr(config, "enable_noise", False),
                num_latents=getattr(config, "laten_size", 77),
            )
            # 改动2（中文注释）：启用 STA 时禁用 text_proj/text_ln 的梯度，避免 DDP 报告未使用参数
            for p in self.text_proj.parameters():
                p.requires_grad = False  # 中文注释：text_proj 在 STA 路径中不使用，关闭梯度
            for p in self.text_ln.parameters():
                p.requires_grad = False  # 中文注释：text_ln 在 STA 路径中不使用，关闭梯度

        # ---------------- 1.5 U-Net ---------------- #
        self.unet = CondUnet1D(
            input_dim=self.input_feats,
            cond_dim=text_latent_dim,
            dim=config.base_dim,
            dim_mults=config.dim_mults,
            adagn=not config.no_adagn,
            zero=True,
            dropout=config.dropout,
            no_eff=config.no_eff,
            dims = None,
            time_dim=config.time_dim,
        )

    # --------------------------------------------------------------------- #
    # 2. 条件 Mask
    # --------------------------------------------------------------------- #
   
    def mask_cond(self, bs, force_mask=False):
        if force_mask:
            return torch.empty(0, device=self.device, dtype=torch.long)
        if self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=self.device) * self.cond_mask_prob)
            return torch.nonzero(1. - mask).squeeze(-1)
        return torch.arange(bs, device=self.device)
    '''
    def mask_cond(self, bs, force_mask=False):
        if force_mask:
            return torch.ones(bs, device=self.device, dtype=torch.bool)
        if self.training and self.cond_mask_prob > 0.:
            mask_float = torch.bernoulli(torch.ones(bs, device=self.device) * self.cond_mask_prob)
            # ✅ 返回固定形状的布尔掩码，True 的位置代表需要屏蔽
            return mask_float == 0
        return torch.zeros(bs, device=self.device, dtype=torch.bool)
    '''
    # --------------------------------------------------------------------- #
    # 3. 获取文本条件（统一入口）
    # --------------------------------------------------------------------- #
    def encode_text(self, *, text=None, raw_embeds=None, proj_embeds=None):
        """
        优先级: proj_embeds > raw_embeds > text
        1) 若直接给 proj_embeds, 原样返回;  
        2) 若给 raw_embeds, 仅在未启用 STA 时进行可训练投影;  
        3) 否则根据 text 获取 raw，并在未启用 STA 时进行可训练投影。
        开启 STA 时将直接返回"原始编码器输出"（仅做必要的形状变换），以复现旧逻辑：
        "如果启动 STA 模块，则不走线性层（由 STA 内部的 proj_in 接收原始维度）"。
        """
        if proj_embeds is not None:
            return proj_embeds                                            # 已是最终形式
        if raw_embeds is not None:
            # 若外部直接给出 raw_embeds
            if self.use_sta:
                return raw_embeds              # 开启 STA：跳过投影，直通给 STA
            return self.text_processor.project_embeds(raw_embeds)         # 未开启 STA：按原逻辑投影
        # 注意 encode_text的形状适配在text_processor里执行.
        # 否则需完整流程：先拿 raw，再根据是否启用 STA 决定是否投影
        assert text is not None, "Either text / raw_embeds / proj_embeds must be provided."
        raw = self.text_processor.get_raw_embeds(text)                    # no-grad
        if self.use_sta:
            return raw                                # 开启 STA：返回原始编码器输出（形状适配）
        return self.text_processor.project_embeds(raw)                    # 未开启 STA：可训练投影

    # --------------------------------------------------------------------- #
    # 4. forward
    # --------------------------------------------------------------------- #
    def forward(
        self,
        x, timesteps,
        *,
        text=None,
        raw_embeds=None,
        proj_embeds=None,
        uncond=False,
        return_intermediates=False
    ):
        """
        Args:
            x            : [B, T, dim]
            timesteps    : [B]
            text         : list[str]            – 原始字符串（可选）
            raw_embeds   : frozen encoder 输出  – shape 取决于编码器（可选）
            proj_embeds  : 投影后的条件         – shape [B, S, C] （可选）
        Notes:
            三者只需提供其一, 按照优先级 proj_embeds>raw_embeds>text.
        """
        B, T, _ = x.shape
        x = x.transpose(1, 2)                                    # [B, nfeats, nframes]
        #nvtx.push_range("T2MUnet.forward", color="blue")
        # ---------- 4.1 文本条件 ---------- #
        enc_text = self.encode_text(
            text=text,
            raw_embeds=raw_embeds,
            proj_embeds=proj_embeds
        )
        #nvtx.pop_range()
        #nvtx.push_range("T2MUnet.forward.STA", color="green")
        # STA
        intermediates = {}
        if self.use_sta:
            if return_intermediates:
                enc_text, sta_w = self.sta_model(enc_text, timesteps, return_attn_weights=True)
                intermediates['sta_attn'] = sta_w
            else:
                enc_text = self.sta_model(enc_text, timesteps)
        #nvtx.pop_range()
        #nvtx.push_range("T2MUnet.forward.mask and padding", color="orange")
        # ---------- 4.2 条件 Mask ---------- #
        cond_indices = self.mask_cond(B, force_mask=uncond)
        
        # ---------- 4.3 Padding ---------- #
        pad = (16 - (T % 16)) % 16
        x_pad = F.pad(x, (0, pad), value=0)
        #nvtx.pop_range()
        #nvtx.push_range("T2MUnet.forward.unet", color="purple")
        # ---------- 4.4 U-Net ---------- #
        if return_intermediates:
            out_pad, unet_w = self.unet(
                x_pad, timesteps, enc_text, cond_indices, return_attn_weights=True
            )
            intermediates['unet_attn'] = unet_w
        else:
            #nvtx.push_range("T2MUnet.forward.unet.core", color="purple")
            out_pad = self.unet(
                x_pad, t=timesteps, cond=enc_text, cond_indices=cond_indices, return_attn_weights=False
            )
            #nvtx.pop_range()
        #nvtx.pop_range()
        out = out_pad[:, :, :T].transpose(1, 2)                  # [B, T, dim]
        return (out, intermediates) if return_intermediates else out

    # --------------------------------------------------------------------- #
    # 5. forward_with_cfg (classifier-free guidance)
    # --------------------------------------------------------------------- #
    def forward_with_cfg(
        self,
        x, timesteps,
        *,
        text=None,
        raw_embeds=None,
        proj_embeds=None,
        return_intermediates=False,
        opt=None
    ):
        """
        与 forward 基本一致，只是把 条件/无条件 拼接后一次过送入 U-Net。
        """
        B, T, _ = x.shape
        x = x.transpose(1, 2)

        enc_text = self.encode_text(
            text=text, raw_embeds=raw_embeds, proj_embeds=proj_embeds
        )
        intermediates = {}
        if self.use_sta:
            if return_intermediates:
                enc_text, sta_w = self.sta_model(enc_text, timesteps, return_attn_weights=True)
                intermediates['sta_attn'] = sta_w
            else:
                enc_text = self.sta_model(enc_text, timesteps)

        cond_indices = self.mask_cond(B, force_mask=False)       # 条件端
        pad = (16 - (T % 16)) % 16
        x_pad = F.pad(x, (0, pad), value=0)

        # 拼接条件/无条件
        x_cmb   = torch.cat([x_pad, x_pad], dim=0)
        t_cmb   = torch.cat([timesteps, timesteps], dim=0)

        if return_intermediates:
            out_pad, unet_w = self.unet(
                x_cmb, t_cmb, enc_text, cond_indices, return_attn_weights=True
            )
            intermediates['unet_attn'] = unet_w
        else:
            out_pad = self.unet(
                x_cmb, t_cmb, enc_text, cond_indices, return_attn_weights=False
            )

        out = out_pad[:, :, :T].transpose(1, 2)
        out_cond, out_uncond = torch.split(out, len(out) // 2, dim=0)
        cfg_scale = opt.cfg_scale
        out = out_uncond + (cfg_scale * (out_cond - out_uncond))
        return (out, intermediates) if return_intermediates else out
