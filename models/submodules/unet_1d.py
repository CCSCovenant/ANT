# ./models/submodules/unet_1d.py

import os
import sys
import importlib
import importlib.util
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util

from .basic_blocks import Downsample1d, Upsample1d
from .time_embedding import TimestepEmbedder

# 动态解析 CondConv1DBlock：当开启 OPENANT_ABLATION_USE_BACKUP_UNETBLOCK 时，使用备份实现
def _resolve_CondConv1DBlock():
    use_backup = os.environ.get("OPENANT_ABLATION_USE_BACKUP_UNETBLOCK", "0") == "1"
    if use_backup:
        backup_path = "/data/kuimou/backup/models/unet.py"
        try:
            spec = importlib.util.spec_from_file_location("backup_unet", backup_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load spec for {backup_path}")
            backup_unet = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(backup_unet)
            print("[CondUnet1D] 使用备份 CondConv1DBlock (ablation=ON)")
            return backup_unet.CondConv1DBlock
        except Exception as e:
            print(f"[CondUnet1D] 备份 CondConv1DBlock 加载失败，改用本地实现。错误: {e}")
    # 默认使用本地实现
    from .cond_conv_block import CondConv1DBlock as LocalCondConv1DBlock
    return LocalCondConv1DBlock

# 导出解析后的类供后续构建网络使用
CondConv1DBlock = _resolve_CondConv1DBlock()

# Ablation: when OPENANT_ABLATION_USE_BACKUP_UNETBLOCK=1, use backup CondConv1DBlock
USE_BACKUP_UNETBLOCK = os.environ.get("OPENANT_ABLATION_USE_BACKUP_UNETBLOCK", "0") == "1"
CondBlockClass = CondConv1DBlock
if USE_BACKUP_UNETBLOCK:
    BackupCondConv1DBlock = None
    try:
        # Prefer package import from backup repo
        sys.path.insert(0, "/data/kuimou/backup")
        backup_mod = importlib.import_module("models.unet")
        BackupCondConv1DBlock = getattr(backup_mod, "CondConv1DBlock", None)
    except Exception:
        # Fallback: direct file import
        try:
            spec = importlib.util.spec_from_file_location(
                "backup_unet", "/data/kuimou/backup/models/unet.py"
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                BackupCondConv1DBlock = getattr(mod, "CondConv1DBlock", None)
        except Exception:
            BackupCondConv1DBlock = None
    finally:
        if "/data/kuimou/backup" in sys.path:
            try:
                sys.path.remove("/data/kuimou/backup")
            except ValueError:
                pass
    if BackupCondConv1DBlock is not None:
        CondBlockClass = BackupCondConv1DBlock

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
                CondBlockClass(dim_in, dim_out, cond_dim, time_dim, adagn, zero, no_eff, dropout),
                CondBlockClass(dim_out, dim_out, cond_dim, time_dim, adagn, zero, no_eff, dropout),
                Downsample1d(dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = CondBlockClass(mid_dim, mid_dim, cond_dim, time_dim, adagn, zero, no_eff, dropout)
        self.mid_block2 = CondBlockClass(mid_dim, mid_dim, cond_dim, time_dim, adagn, zero, no_eff, dropout)

        last_dim = mid_dim
        for ind, dim_out in enumerate(reversed(dims[1:])):
            self.ups.append(nn.ModuleList([
                Upsample1d(last_dim, dim_out),
                CondBlockClass(dim_out * 2, dim_out, cond_dim, time_dim, adagn, zero, no_eff, dropout),
                CondBlockClass(dim_out, dim_out, cond_dim, time_dim, adagn, zero, no_eff, dropout),
            ]))
            last_dim = dim_out
            
        self.final_conv = nn.Conv1d(last_dim, input_dim, 1)

        if zero:
            nn.init.zeros_(self.final_conv.weight)
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, t, cond, cond_indices):
        """
        与备份版 CondUnet1D 对齐的前向实现：不支持返回注意力权重。
        """
        temb = self.time_mlp(t)
        h = []

        for i, (block1, block2, downsample) in enumerate(self.downs):
            x = block1(x, temb, cond, cond_indices)
            x = block2(x, temb, cond, cond_indices)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, temb, cond, cond_indices)
        x = self.mid_block2(x, temb, cond, cond_indices)

        for i, (upsample, block1, block2) in enumerate(self.ups):
            x = upsample(x)
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, temb, cond, cond_indices)
            x = block2(x, temb, cond, cond_indices)

        x = self.final_conv(x)
        return x