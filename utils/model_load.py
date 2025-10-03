import torch
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
