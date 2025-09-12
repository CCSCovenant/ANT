import torch
from .ema import ExponentialMovingAverage

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

    # 改动2（中文注释）：严格按目标 device 加载权重，避免跨 GPU id 导致的 map_location 不一致
    checkpoint = torch.load(ckpt_path, map_location=device)

    total_iter = checkpoint.get('total_it', 0)

    # 改动3（中文注释）：修复原逻辑在加载 EMA 权重时对 model 的本地重新赋值，导致调用方模型未真正载入参数的问题。
    # 现在无论 EMA 的 state_dict 形态如何，均“提取并对齐”权重后，直接 load 到传入的原始 model 上。
    if use_ema and ('model_ema' in checkpoint):
        ema_state = checkpoint['model_ema']

        # 处理 timm/swa_utils 风格的 EMA：包含 'n_averaged' 与 'module.' 前缀
        # 情况A：字典里既有参数（带 module. 前缀），也可能含有 n_averaged 等统计量
        if any(k.startswith('module.') for k in ema_state.keys()):
            # 去掉 'module.' 前缀，仅保留与模型参数名匹配的键值
            ema_weights = {k[len('module.'):] : v for k, v in ema_state.items() if k.startswith('module.')}
            # 中文注释：严格加载到原 model；strict=False 以兼容轻微的键名差异（比如未用到参数）
            missing, unexpected = model.load_state_dict(ema_weights, strict=False)
            print(f"\nLoading EMA(model.module) weights from {ckpt_path} with {total_iter} iterations")
        else:
            # 情况B：直接是与模型同名的参数字典（无 module. 前缀），或其他简单形式
            # 同时过滤可能存在的统计项如 n_averaged
            ema_weights = {k: v for k, v in ema_state.items() if k in model.state_dict()}
            missing, unexpected = model.load_state_dict(ema_weights, strict=False)
            print(f"\nLoading EMA model from {ckpt_path} with {total_iter} iterations")

        if missing:
            print(f"[load_model_weights][EMA] missing keys: {len(missing)} (showing first 10) -> {missing[:10]}")
        if unexpected:
            print(f"[load_model_weights][EMA] unexpected keys: {len(unexpected)} (showing first 10) -> {unexpected[:10]}")
    else:
        # 回退：加载常规 non-EMA 权重（多数工程保存为 'encoder' 或类似键）
        encoder_state = checkpoint.get('encoder', None)
        if encoder_state is None:
            raise KeyError("Checkpoint does not contain 'encoder' or 'model_ema' keys.")
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        print(f"\nLoading model from {ckpt_path} with {total_iter} iterations")
        if missing:
            print(f"[load_model_weights][RAW] missing keys: {len(missing)} (showing first 10) -> {missing[:10]}")
        if unexpected:
            print(f"[load_model_weights][RAW] unexpected keys: {len(unexpected)} (showing first 10) -> {unexpected[:10]}")

    return total_iter


def load_lora_weight(model, lora_path, use_ema, device='cuda'):
    # 改动4（中文注释）：同样规范 map_location 逻辑，且不在此处引入 ExponentialMovingAverage 包装，直接对齐并加载。
    if isinstance(device, str):
        device = torch.device(device)
    checkpoint = torch.load(lora_path, map_location=device)

    total_iter = checkpoint.get('total_it', 0)

    if use_ema and ('model_ema' in checkpoint):
        ema_state = checkpoint['model_ema']
        if any(k.startswith('module.') for k in ema_state.keys()):
            ema_weights = {k[len('module.'):] : v for k, v in ema_state.items() if k.startswith('module.')}
        else:
            ema_weights = {k: v for k, v in ema_state.items() if k in model.state_dict()}
        missing, unexpected = model.load_state_dict(ema_weights, strict=False)
        print(f"\nLoading EMA LoRA model from {lora_path} with {total_iter} iterations")
        if missing:
            print(f"[load_lora_weight][EMA] missing keys: {len(missing)} (showing first 10) -> {missing[:10]}")
        if unexpected:
            print(f"[load_lora_weight][EMA] unexpected keys: {len(unexpected)} (showing first 10) -> {unexpected[:10]}")
    else:
        if hasattr(model, 'set_peft_model_state_dict'):
            model.set_peft_model_state_dict(checkpoint['lora'], strict=False)
        else:
            model.load_state_dict(checkpoint['lora'], strict=False)
        print(f"\nLoading model from {lora_path} with {total_iter} iterations")

    return total_iter
