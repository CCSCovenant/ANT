import torch
from typing import Dict, List, Tuple, Optional

def load_model_weights(model, ckpt_path, use_ema=True, device=None):
    """
    Load weights of a model from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        ckpt_path (str): Path to the checkpoint file.
        use_ema (bool): Whether to use Exponential Moving Average (EMA) weights if available.
        device (torch.device or str, optional): target device for loading checkpoint.
    Returns:
        int: total training iterations stored in checkpoint
    """
    # 改动1（中文注释）：尊重调用方传入的 device；若未传入，则保持兼容默认使用当前 CUDA（如可用）或 CPU
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    # 严格按目标 device 加载权重
    checkpoint = torch.load(ckpt_path, map_location=device)

    total_iter = checkpoint.get('total_it', 0)

    # 严格加载：仅接受指定键，必须完全匹配
    if use_ema:
        if 'model_ema' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_ema' while use_ema=True")
        ema_state = checkpoint['model_ema']
        if isinstance(ema_state, dict) and 'state_dict' in ema_state:
            ema_sd = ema_state['state_dict']
        elif isinstance(ema_state, dict):
            ema_sd = ema_state
        else:
            raise TypeError("'model_ema' must be a dict or contain a 'state_dict' dict")
        missing, unexpected = model.load_state_dict(ema_sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Strict EMA load failed: missing={len(missing)} unexpected={len(unexpected)}")
        print(f"\nLoaded EMA weights strictly from {ckpt_path} with {total_iter} iterations")
    else:
        if 'encoder' not in checkpoint:
            raise KeyError("Checkpoint missing 'encoder' while use_ema=False")
        encoder_state = checkpoint['encoder']
        missing, unexpected = model.load_state_dict(encoder_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Strict RAW load failed: missing={len(missing)} unexpected={len(unexpected)}")
        print(f"\nLoaded RAW weights strictly from {ckpt_path} with {total_iter} iterations")

    return total_iter

def load_weights_strict_components(model: torch.nn.Module, ckpt_path: str, use_ema: bool, device: torch.device) -> int:
    """
    仅严格加载组件级权重：sta_model 和 unet。
    - 读取 checkpoint 的 state_dict（优先 model_ema，其次 encoder，再次 state_dict）。
    - 过滤键前缀 "sta_model." 与 "unet."，分别加载到 model.sta_model 与 model.unet（strict=True）。
    - 返回 niter（若存在则返回，否则 0）。
    """
    # 读 ckpt
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError('Checkpoint format error: not a dict')

    # 选择分支
    branch = None
    if use_ema and ('model_ema' in ckpt):
        branch = ckpt['model_ema']
    elif 'encoder' in ckpt:
        branch = ckpt['encoder']
    elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        branch = ckpt['state_dict']
    else:
        # 直接把 ckpt 当作 state_dict
        branch = ckpt

    # 统一取出映射
    if isinstance(branch, dict) and 'state_dict' in branch and isinstance(branch['state_dict'], dict):
        sd = branch['state_dict']
    elif isinstance(branch, dict):
        sd = branch
    else:
        raise RuntimeError('Selected branch is not a mapping with state_dict')

    sd = _normalize_sd_keys(sd)

    # 过滤并去掉组件前缀
    sd_sta = { _strip_prefix(k, 'sta_model.'): v for k, v in sd.items() if k.startswith('sta_model.') }
    sd_unet = { _strip_prefix(k, 'unet.'): v for k, v in sd.items() if k.startswith('unet.') }

    # 严格加载到组件
    missing_all = []
    unexpected_all = []
    try:
        missing, unexpected = model.sta_model.load_state_dict(sd_sta, strict=True)
        missing_all += list(missing)
        unexpected_all += list(unexpected)
    except Exception as e:
        raise RuntimeError(f"Strict load failed for sta_model: {e}")

    try:
        missing, unexpected = model.unet.load_state_dict(sd_unet, strict=True)
        missing_all += list(missing)
        unexpected_all += list(unexpected)
    except Exception as e:
        raise RuntimeError(f"Strict load failed for unet: {e}")

    if missing_all or unexpected_all:
        raise RuntimeError(f"Strict component loading reported issues. missing={missing_all}, unexpected={unexpected_all}")

    # niter/total_it
    niter = ckpt.get('niter', ckpt.get('total_it', 0))
    try:
        return int(niter or 0)
    except Exception:
        return 0

def load_lora_weight(model, lora_path, use_ema, device='cuda'):
    # 严格按目标 device 加载权重
    if isinstance(device, str):
        device = torch.device(device)
    checkpoint = torch.load(lora_path, map_location=device)

    total_iter = checkpoint.get('total_it', 0)

    if use_ema:
        if 'model_ema' not in checkpoint:
            raise KeyError("LoRA checkpoint missing 'model_ema' while use_ema=True")
        ema_state = checkpoint['model_ema']
        if isinstance(ema_state, dict) and 'state_dict' in ema_state:
            ema_sd = ema_state['state_dict']
        elif isinstance(ema_state, dict):
            ema_sd = ema_state
        else:
            raise TypeError("LoRA 'model_ema' must be a dict or contain a 'state_dict' dict")
        missing, unexpected = model.load_state_dict(ema_sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Strict LoRA-EMA load failed: missing={len(missing)} unexpected={len(unexpected)}")
        print(f"\nLoaded LoRA EMA weights strictly from {lora_path} with {total_iter} iterations")
    else:
        if 'lora' not in checkpoint:
            raise KeyError("LoRA checkpoint missing 'lora' while use_ema=False")
        lora_sd = checkpoint['lora']
        # 一些 PEFT 实现提供专用入口；保持严格加载要求
        if hasattr(model, 'set_peft_model_state_dict'):
            model.set_peft_model_state_dict(lora_sd, strict=True)
            print(f"\nLoaded LoRA weights strictly via PEFT from {lora_path} with {total_iter} iterations")
        else:
            missing, unexpected = model.load_state_dict(lora_sd, strict=True)
            if missing or unexpected:
                raise RuntimeError(f"Strict LoRA load failed: missing={len(missing)} unexpected={len(unexpected)}")
            print(f"\nLoaded LoRA weights strictly from {lora_path} with {total_iter} iterations")

    return total_iter

def _normalize_sd_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # 统一去除 "module." 前缀，便于严格加载
    out = {}
    for k, v in sd.items():
        if k.startswith('module.'):
            out[k[7:]] = v
        else:
            out[k] = v
    return out

def _strip_prefix(k: str, prefix: str) -> str:
    return k[len(prefix):] if k.startswith(prefix) else k
