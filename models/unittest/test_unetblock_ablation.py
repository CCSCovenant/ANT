import os
import sys
import importlib.util
from types import SimpleNamespace

import torch


# 固定路径（与用户命令一致）
NEW_OPT = "/data/kuimou/openANT/checkpoints/t2m/ant_t2m/opt.txt"
NEW_CKPT = "/data/kuimou/openANT/checkpoints/t2m/ant_t2m/model/old_best_clean.tar"
OLD_OPT = "/data/kuimou/backup/checkpoints/t2m/t2m_t5/opt.txt"
OLD_CKPT = "/data/kuimou/backup/checkpoints/t2m/t2m_t5/model/latest_120000.tar"

THRESH_MATCH = 1e-5  # ablation=1 时，UNet 前向最终 L2 差异应近乎为 0
THRESH_DIFF = 1.0    # ablation=0 时，UNet 前向最终 L2 差异应显著大于该阈值


def _skip_if_missing():
    missing = []
    for p in [NEW_OPT, NEW_CKPT, OLD_OPT, OLD_CKPT]:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        # 若环境未安装 pytest，则抛出可识别的跳过异常
        try:
            import pytest  # type: ignore
            pytest.skip(f"缺失必要文件: {missing}")
        except Exception:
            raise RuntimeError(f"SKIP: 缺失必要文件: {missing}")


def _import_model_load_compare():
    path = os.path.join("/data/kuimou/openANT", "tmp_scripts", "model_comparison_debug", "model_load_compare.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"model_load_compare.py not found: {path}")
    spec = importlib.util.spec_from_file_location("model_load_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clear_openant_modules():
    # 确保根据环境变量重新解析 CondConv1DBlock
    for name in list(sys.modules.keys()):
        if name.startswith("models.submodules.unet_1d") or \
           name.startswith("models.submodules.cond_conv_block") or \
           name.startswith("models.t2m_unet") or \
           name.startswith("models.unet_factory"):
            sys.modules.pop(name, None)
    # 关键：移除顶层 models 包，避免 residual 的 backup/models 覆盖 openANT 的 models
    if 'models' in sys.modules:
        sys.modules.pop('models', None)
    importlib.invalidate_caches()


def _build_models_with_env(env_val: str):
    # 设置 ablation 环境变量并清理相关模块缓存
    os.environ["OPENANT_ABLATION_USE_BACKUP_UNETBLOCK"] = env_val
    _clear_openant_modules()

    mlc = _import_model_load_compare()

    # 设备均使用 CPU
    dev_new = torch.device("cpu")
    dev_old = torch.device("cpu")

    # 加载 opts
    opt_new = mlc.load_opt_openant(NEW_OPT)
    opt_old = mlc.load_opt_backup(OLD_OPT)
    opt_new.device = str(dev_new)
    opt_old.device = str(dev_old)

    # 构建模型
    model_new = mlc.build_model_new(opt_new).to(dev_new).eval()
    model_old = mlc.build_model_old(opt_old).to(dev_old).eval()

    # 自动选择是否使用 EMA
    use_ema_new = mlc.detect_has_ema(NEW_CKPT)
    use_ema_old = mlc.detect_has_ema(OLD_CKPT)

    # 加载权重（严格）
    _ = mlc.load_weights_strict_components(model_new, NEW_CKPT, use_ema=use_ema_new, device=dev_new)
    _ = mlc.load_weights_strict_on_model(model_old, OLD_CKPT, use_ema=use_ema_old, device=dev_old)

    return mlc, model_new, model_old, dev_new, dev_old


def _extract_unet_l2_from_logs(logs):
    prefix = "UNet forward final L2 diff: "
    for line in logs:
        if line.startswith(prefix):
            try:
                return float(line[len(prefix):])
            except Exception:
                continue
    raise AssertionError(f"未在日志中找到前向 L2 差异: {logs}")


def test_forward_consistency_with_ablation_on():
    _skip_if_missing()
    mlc, model_new, model_old, dev_new, dev_old = _build_models_with_env("1")

    logs = mlc.compare_forward_outputs(
        model_new, model_old, dev_new, dev_old,
        B=2, T=64, S=None,
        seed_sta=123, seed_unet_x=456, seed_unet_cond=789,
        eps=1e-6,
    )
    l2 = _extract_unet_l2_from_logs(logs)
    assert l2 <= THRESH_MATCH, f"ablation=1 时 UNet 前向 L2 差异过大: {l2}"


def test_forward_divergence_with_ablation_off():
    _skip_if_missing()
    mlc, model_new, model_old, dev_new, dev_old = _build_models_with_env("0")

    logs = mlc.compare_forward_outputs(
        model_new, model_old, dev_new, dev_old,
        B=2, T=64, S=None,
        seed_sta=123, seed_unet_x=456, seed_unet_cond=789,
        eps=1e-6,
    )
    l2 = _extract_unet_l2_from_logs(logs)
    assert l2 >= THRESH_DIFF, f"ablation=0 时 UNet 前向 L2 差异不显著: {l2}"