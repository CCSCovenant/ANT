#!/usr/bin/env python3
"""
随机逐层对比脚本（不改动源文件）

目标：
- 对 OLD 与 NEW 两套 T2M-Unet 模型的关键中间层进行“相同随机输入”的输出对比，定位差异来源。
- 专门检查：
  1) text_encoder 的 embedding 一致性（使用真实文本对比原始嵌入）；
  2) STA / ELLA 模块的一致性（随机输入 + 时间嵌入）；
  3) U-Net 里的 ResidualCrossAttentionLayer 等关键层的一致性（逐层随机输入）。

说明：
- 不改动任何源文件，仅在脚本内构造输入和注册必要的逻辑。
- 支持将 OLD/NEW 放在不同设备上（如 cuda:0 / cuda:1）。跨设备比较时统一搬到 CPU 上计算差异。

用法示例：
  python openANT/tmp_scripts/model_comparison_debug/compare_random_per_layer.py \
    --old_opt /data/kuimou/backup/checkpoints/t2m/t2m_t5/opt.txt \
    --old_ckpt /data/kuimou/backup/checkpoints/t2m/t2m_t5/model/latest_120000.tar \
    --new_opt /data/kuimou/openANT/checkpoints/t2m/ant_t2m/opt.txt \
    --new_ckpt /data/kuimou/openANT/checkpoints/t2m/ant_t2m/model/old_best.tar \
    --device_old cuda:0 --device_new cuda:1 --batch_size 1 --nframes 64 \
    --text "a person walks forward slowly" \
    --out_json openANT/tmp_scripts/model_comparison_debug/output/random_per_layer_compare.json

"""

import os
import sys
import json
import ast
import argparse
import types
import inspect
from types import SimpleNamespace

import torch
import torch.nn.functional as F


OPENANT_ROOT = "/data/kuimou/openANT"
BACKUP_ROOT = "/data/kuimou/backup"


# ------------------------- 通用工具 ------------------------- #
def add_paths():
    if OPENANT_ROOT not in sys.path:
        sys.path.insert(0, OPENANT_ROOT)
    if BACKUP_ROOT not in sys.path:
        sys.path.insert(0, BACKUP_ROOT)


def _parse_opt_txt(opt_txt_path: str):
    def _parse_val(v):
        v = v.strip()
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    cfg = {}
    with open(opt_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                k, v = line.split(':', 1)
                cfg[k.strip()] = _parse_val(v)
    opt = SimpleNamespace(**cfg)
    if not hasattr(opt, 'dim_pose'):
        setattr(opt, 'dim_pose', 263)
    if not hasattr(opt, 'device'):
        setattr(opt, 'device', 'cpu')
    return opt


def load_opt_via_get_opt(repo_root: str, opt_txt_path: str):
    repo_root = os.path.abspath(repo_root)
    sys.path.insert(0, repo_root)
    try:
        opt = None
        if repo_root == os.path.abspath(OPENANT_ROOT):
            try:
                from options.get_opt import get_opt as get_opt_openant
                opt = get_opt_openant(opt_txt_path)
            except Exception:
                opt = None
        if opt is None:
            try:
                from scripts.get_opt import get_opt as get_opt_backup
                opt = get_opt_backup(opt_txt_path)
            except Exception:
                opt = None
        if opt is None:
            opt = _parse_opt_txt(opt_txt_path)
        if not hasattr(opt, 'dim_pose'):
            setattr(opt, 'dim_pose', 263)
        if not hasattr(opt, 'device'):
            setattr(opt, 'device', 'cpu')
        return opt
    finally:
        if repo_root in sys.path:
            sys.path.remove(repo_root)


def build_model_old(opt):
    sys.path.insert(0, BACKUP_ROOT)
    try:
        # 兼容旧库可能强行 import clip
        if 'clip' not in sys.modules:
            clip_stub = types.ModuleType('clip')
            def _noop(*args, **kwargs):
                return None
            clip_stub.load = _noop
            sys.modules['clip'] = clip_stub
        from models import build_models as build_old
        model = build_old(opt)
    finally:
        if 'models' in sys.modules:
            try:
                mod = sys.modules['models']
                mod_file = getattr(mod, '__file__', '') or ''
                if BACKUP_ROOT in mod_file:
                    del sys.modules['models']
            except Exception:
                del sys.modules['models']
        if BACKUP_ROOT in sys.path:
            sys.path.remove(BACKUP_ROOT)
    return model


def build_model_new(opt):
    import importlib.util, pathlib
    root = pathlib.Path(OPENANT_ROOT)
    models_dir = root / 'models'
    uf_path = models_dir / 'unet_factory.py'
    if not uf_path.exists():
        raise FileNotFoundError(f"unet_factory.py not found at {uf_path}")
    pkg = types.ModuleType('models')
    pkg.__file__ = str((models_dir / '__init__.py').resolve())
    pkg.__path__ = [str(models_dir.resolve())]
    sys.modules['models'] = pkg
    spec = importlib.util.spec_from_file_location('models.unet_factory', str(uf_path))
    uf_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(uf_module)
    model = uf_module.UnetFactory.create_unet(opt)
    return model


def load_weights(model, ckpt_path, use_ema=True, device='cpu'):
    sys.path.insert(0, OPENANT_ROOT)
    try:
        from utils.model_load import load_model_weights
        niter = load_model_weights(model, ckpt_path, use_ema=use_ema, device=device)
        return niter
    finally:
        if OPENANT_ROOT in sys.path:
            sys.path.remove(OPENANT_ROOT)


def tensor_digest(t: torch.Tensor):
    try:
        t_cpu = t.detach().to('cpu')
        return {
            'shape': list(t_cpu.shape),
            'dtype': str(t_cpu.dtype),
            'mean': float(t_cpu.mean().item()),
            'std': float(t_cpu.std().item()),
            'min': float(t_cpu.min().item()),
            'max': float(t_cpu.max().item()),
        }
    except Exception as e:
        return {'error': f'digest_failed: {repr(e)}'}


def compare_tensors(a: torch.Tensor, b: torch.Tensor):
    """统一搬到 CPU 比较，避免跨设备错误。"""
    a_cpu = a.detach().to('cpu')
    b_cpu = b.detach().to('cpu')
    diff = a_cpu - b_cpu
    l2 = float(torch.norm(diff).item())
    cosine = float(F.cosine_similarity(a_cpu.flatten(), b_cpu.flatten(), dim=0).item())
    mean_abs = float(diff.abs().mean().item())
    return {
        'l2': l2,
        'cosine': cosine,
        'mean_abs': mean_abs,
        'a_digest': tensor_digest(a_cpu),
        'b_digest': tensor_digest(b_cpu),
    }


# ------------------------- 随机输入构造 ------------------------- #
def rand_int_timesteps(batch_size: int, device: torch.device, max_t: int = 1000):
    return torch.randint(0, max_t, (batch_size,), device=device, dtype=torch.long)


def synth_time_emb(module: torch.nn.Module, batch_size: int, device: torch.device):
    """为 PerceiverResampler 合成 timestep embedding（根据 time_aware_linear 的 in_features）。"""
    if hasattr(module, 'time_aware_linear') and hasattr(module.time_aware_linear, 'in_features'):
        d = module.time_aware_linear.in_features
        return torch.randn(batch_size, d, device=device)
    # 兜底：给一个 width 尺寸的向量
    width = getattr(module, 'width', 256)
    return torch.randn(batch_size, width, device=device)


def collect_modules(model: torch.nn.Module):
    """收集需要逐层对比的模块列表（按出现顺序）。"""
    per_resamplers = []
    res_cross_attn = []
    downs_1d = []
    ups_1d = []
    residual_blocks = []

    for name, m in model.named_modules():
        cls = m.__class__.__name__
        if cls == 'PerceiverResampler':
            per_resamplers.append((name, m))
        elif cls == 'ResidualCrossAttentionLayer':
            res_cross_attn.append((name, m))
        elif cls == 'Downsample1d':
            downs_1d.append((name, m))
        elif cls == 'Upsample1d':
            ups_1d.append((name, m))
        elif cls == 'ResidualBlock':
            residual_blocks.append((name, m))
    return {
        'PerceiverResampler': per_resamplers,
        'ResidualCrossAttentionLayer': res_cross_attn,
        'Downsample1d': downs_1d,
        'Upsample1d': ups_1d,
        'ResidualBlock': residual_blocks,
    }


def run_module_pair_random(cls_name: str, old_pair, new_pair, batch_size: int, nframes: int, num_tokens_raw: int,
                           num_latents_cond: int):
    """针对某一类模块，生成相同形状的随机输入，分别运行 OLD / NEW 并给出比较结果。"""
    name_old, mod_old = old_pair
    name_new, mod_new = new_pair
    dev_old = next(mod_old.parameters(), torch.empty(0)).device if any(True for _ in mod_old.parameters()) else torch.device('cpu')
    dev_new = next(mod_new.parameters(), torch.empty(0)).device if any(True for _ in mod_new.parameters()) else torch.device('cpu')

    mod_old.eval(); mod_new.eval()

    out_old = None
    out_new = None
    err = None

    try:
        if cls_name == 'PerceiverResampler':
            # 需要 [B, N_raw, input_dim] + timestep_embedding
            in_dim_old = getattr(mod_old, 'input_dim', None)
            in_dim_new = getattr(mod_new, 'input_dim', None)
            # 若两边不一致，以较小者为准
            cand = [d for d in [in_dim_old, in_dim_new] if d is not None]
            input_dim = min(cand) if cand else 256
            x_old = torch.randn(batch_size, num_tokens_raw, input_dim, device=dev_old)
            x_new = x_old.detach().to(dev_new)
            te_old = synth_time_emb(mod_old, batch_size, dev_old)
            te_new = te_old.detach().to(dev_new)
            with torch.no_grad():
                out_old = mod_old(x_old, timestep_embedding=te_old)
                out_new = mod_new(x_new, timestep_embedding=te_new)

        elif cls_name == 'ResidualCrossAttentionLayer':
            # 输入 [B, D, L]，条件 [B, N, D2]，全部参与 cond_indices
            D_old = getattr(mod_old, 'dim1', None)
            D_new = getattr(mod_new, 'dim1', None)
            D = min([d for d in [D_old, D_new] if d is not None]) if any([D_old, D_new]) else 256
            D2_old = getattr(mod_old, 'dim2', None)
            D2_new = getattr(mod_new, 'dim2', None)
            D2 = min([d for d in [D2_old, D2_new] if d is not None]) if any([D2_old, D2_new]) else 256
            x_old = torch.randn(batch_size, D, nframes, device=dev_old)
            x_new = x_old.detach().to(dev_new)
            cond_old = torch.randn(batch_size, num_latents_cond, D2, device=dev_old)
            cond_new = cond_old.detach().to(dev_new)
            idx_old = torch.arange(batch_size, device=dev_old, dtype=torch.long)
            idx_new = idx_old.detach().to(dev_new)
            with torch.no_grad():
                out_old = mod_old(x_old, cond_old, idx_old)
                out_new = mod_new(x_new, cond_new, idx_new)

        elif cls_name == 'Downsample1d':
            C_in = getattr(mod_old.conv, 'in_channels', None) or getattr(mod_new.conv, 'in_channels', 256)
            x_old = torch.randn(batch_size, C_in, nframes, device=dev_old)
            x_new = x_old.detach().to(dev_new)
            with torch.no_grad():
                out_old = mod_old(x_old)
                out_new = mod_new(x_new)

        elif cls_name == 'Upsample1d':
            C_in = getattr(mod_old.conv, 'in_channels', None) or getattr(mod_new.conv, 'in_channels', 256)
            x_old = torch.randn(batch_size, C_in, nframes // 2, device=dev_old)
            x_new = x_old.detach().to(dev_new)
            with torch.no_grad():
                out_old = mod_old(x_old)
                out_new = mod_new(x_new)

        elif cls_name == 'ResidualBlock':
            # 需要 time_embeds；用 time_mlp 的第一个 Linear 的 in_features 推断
            # channel 以 blocks[0] 的 conv1.in_channels 作为输入通道
            try:
                conv1 = getattr(mod_old.blocks[0], 'conv1', None) or getattr(mod_new.blocks[0], 'conv1', None)
                C_in = getattr(conv1, 'in_channels', 256)
            except Exception:
                C_in = 256
            # time embedding 维度
            def _infer_time_in(m):
                if hasattr(m, 'time_mlp'):
                    for l in m.time_mlp.modules():
                        if isinstance(l, torch.nn.Linear):
                            return l.in_features
                return 512
            T_in = min(_infer_time_in(mod_old), _infer_time_in(mod_new))
            x_old = torch.randn(batch_size, C_in, nframes, device=dev_old)
            x_new = x_old.detach().to(dev_new)
            te_old = torch.randn(batch_size, T_in, device=dev_old)
            te_new = te_old.detach().to(dev_new)
            with torch.no_grad():
                out_old = mod_old(x_old, time_embeds=te_old)
                out_new = mod_new(x_new, time_embeds=te_new)
        else:
            err = f'Unsupported module class: {cls_name}'

        if out_old is None or out_new is None:
            err = err or 'module forward returned None'

        comp = None
        if err is None:
            comp = compare_tensors(out_old, out_new)
        return {
            'cls': cls_name,
            'name_old': name_old,
            'name_new': name_new,
            'error': err,
            'old_out': tensor_digest(out_old) if out_old is not None else None,
            'new_out': tensor_digest(out_new) if out_new is not None else None,
            'compare': comp,
        }
    except Exception as e:
        return {
            'cls': cls_name,
            'name_old': name_old,
            'name_new': name_new,
            'error': f'exception: {repr(e)}',
        }


# ------------------------- 顶层流程 ------------------------- #
def main():
    parser = argparse.ArgumentParser('Random per-layer IO comparison: OLD vs NEW (no source changes)')
    parser.add_argument('--old_opt', required=True)
    parser.add_argument('--old_ckpt', required=True)
    parser.add_argument('--new_opt', required=True)
    parser.add_argument('--new_ckpt', required=True)
    parser.add_argument('--device_old', default=None)
    parser.add_argument('--device_new', default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nframes', type=int, default=64)
    parser.add_argument('--num_tokens_raw', type=int, default=77)
    parser.add_argument('--num_latents_cond', type=int, default=77)
    parser.add_argument('--text', type=str, default='a person walks forward slowly')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--skip_load', action='store_true', help='跳过加载权重，仅结构层面对比')
    parser.add_argument('--out_json', default='openANT/tmp_scripts/model_comparison_debug/output/random_per_layer_compare.json')

    args = parser.parse_args()

    add_paths()

    # 设备解析（默认双卡分配）
    def _resolve_dev(arg, default):
        if arg is None:
            return default
        return torch.device(arg)

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        default_old = torch.device('cuda:0')
        default_new = torch.device('cuda:1')
    elif torch.cuda.is_available():
        default_old = torch.device('cuda:0')
        default_new = torch.device('cuda:0')
    else:
        default_old = torch.device('cpu')
        default_new = torch.device('cpu')

    dev_old = _resolve_dev(args.device_old, default_old)
    dev_new = _resolve_dev(args.device_new, default_new)

    # 加载配置与模型
    opt_old = load_opt_via_get_opt(BACKUP_ROOT, args.old_opt)
    opt_new = load_opt_via_get_opt(OPENANT_ROOT, args.new_opt)
    opt_old.device = str(dev_old)
    opt_new.device = str(dev_new)

    print('[INFO] Building models...')
    model_old = build_model_old(opt_old).to(dev_old).eval()
    model_new = build_model_new(opt_new).to(dev_new).eval()

    if not args.skip_load:
        print('[INFO] Loading checkpoints...')
        try:
            _ = load_weights(model_old, args.old_ckpt, use_ema=args.use_ema, device=dev_old)
        except Exception as e:
            print('[WARN] load old weights failed:', repr(e))
        try:
            _ = load_weights(model_new, args.new_ckpt, use_ema=args.use_ema, device=dev_new)
        except Exception as e:
            print('[WARN] load new weights failed:', repr(e))

    report = {
        'device_old': str(dev_old),
        'device_new': str(dev_new),
        'batch_size': args.batch_size,
        'nframes': args.nframes,
        'num_tokens_raw': args.num_tokens_raw,
        'num_latents_cond': args.num_latents_cond,
        'text': args.text,
        'sections': {}
    }

    # 1) text_encoder embedding 一致性（尽量取原始嵌入）
    print('[INFO] Comparing text encoder raw embeddings with real text...')
    text_list = [args.text] * args.batch_size
    raw_old = None
    raw_new = None
    te_err = None
    try:
        # NEW 优先走 text_processor.get_raw_embeds
        if hasattr(model_new, 'text_processor') and hasattr(model_new.text_processor, 'get_raw_embeds'):
            with torch.no_grad():
                raw_new = model_new.text_processor.get_raw_embeds(text_list)
        else:
            with torch.no_grad():
                raw_new = model_new.encode_text(text=text_list)
        # OLD 类似处理
        if hasattr(model_old, 'text_processor') and hasattr(model_old.text_processor, 'get_raw_embeds'):
            with torch.no_grad():
                raw_old = model_old.text_processor.get_raw_embeds(text_list)
        else:
            with torch.no_grad():
                raw_old = model_old.encode_text(text_list)
        # 搬到 CPU 对比
        report['sections']['text_encoder_raw'] = compare_tensors(raw_old, raw_new)
    except Exception as e:
        te_err = repr(e)
        report['sections']['text_encoder_raw'] = {'error': te_err}

    # 2) STA / ELLA 随机输入一致性
    print('[INFO] Comparing STA/ELLA modules with random inputs...')
    sta_old = getattr(model_old, 'sta_model', None)
    sta_new = getattr(model_new, 'sta_model', None)
    ella_old = getattr(model_old, 'ella_model', None)
    ella_new = getattr(model_new, 'ella_model', None)

    def _compare_connector(conn_old, conn_new, tag: str):
        sec_key = f'connector_{tag}'
        if conn_old is None or conn_new is None:
            report['sections'][sec_key] = {'error': 'one side missing'}
            return
        # 推断 input_dim（PerceiverResampler 里保存）。
        in_dim_old = getattr(conn_old.connector, 'input_dim', None) if hasattr(conn_old, 'connector') else getattr(conn_old, 'input_dim', None)
        in_dim_new = getattr(conn_new.connector, 'input_dim', None) if hasattr(conn_new, 'connector') else getattr(conn_new, 'input_dim', None)
        cand = [d for d in [in_dim_old, in_dim_new] if d is not None]
        input_dim = min(cand) if cand else 256

        # 合成原始文本特征 + timesteps
        x_old = torch.randn(args.batch_size, args.num_tokens_raw, input_dim, device=dev_old)
        x_new = x_old.detach().to(dev_new)
        t_old = rand_int_timesteps(args.batch_size, dev_old)
        t_new = t_old.detach().to(dev_new)

        try:
            with torch.no_grad():
                out_old = conn_old(x_old, t_old)
                out_new = conn_new(x_new, t_new)
            report['sections'][sec_key] = compare_tensors(out_old, out_new)
        except Exception as e:
            report['sections'][sec_key] = {'error': repr(e)}

    # 优先比较存在的组合
    if sta_old and sta_new:
        _compare_connector(sta_old, sta_new, 'sta_vs_sta')
    elif ella_old and sta_new:
        _compare_connector(ella_old, sta_new, 'ella_vs_sta')
    elif sta_old and ella_new:
        _compare_connector(sta_old, ella_new, 'sta_vs_ella')
    elif ella_old and ella_new:
        _compare_connector(ella_old, ella_new, 'ella_vs_ella')
    else:
        report['sections']['connector'] = {'error': 'no sta/ella found on both sides'}

    # 3) U-Net 内部逐层随机对比（按出现顺序，两侧逐一匹配）
    print('[INFO] Comparing UNet internal modules (random IO, ordered pairwise)...')
    mods_old = collect_modules(model_old)
    mods_new = collect_modules(model_new)

    def _pairwise(cls_name):
        arr_old = mods_old.get(cls_name, [])
        arr_new = mods_new.get(cls_name, [])
        n = min(len(arr_old), len(arr_new))
        results = []
        for i in range(n):
            res = run_module_pair_random(
                cls_name,
                arr_old[i],
                arr_new[i],
                batch_size=args.batch_size,
                nframes=args.nframes,
                num_tokens_raw=args.num_tokens_raw,
                num_latents_cond=args.num_latents_cond,
            )
            results.append(res)
        return results

    per_layer_results = {
        'PerceiverResampler': _pairwise('PerceiverResampler'),
        'ResidualCrossAttentionLayer': _pairwise('ResidualCrossAttentionLayer'),
        'Downsample1d': _pairwise('Downsample1d'),
        'Upsample1d': _pairwise('Upsample1d'),
        'ResidualBlock': _pairwise('ResidualBlock'),
    }

    report['sections']['per_layer'] = per_layer_results

    # 4) UNet 顶层（整层随机 IO，对最终输出对比）
    print('[INFO] Comparing UNet top-level forward with random IO...')
    try:
        # 输入 [B, nfeats, L]，条件 [B, N_latents, D_latent]
        # 尽量从 NEW 的 unet 维度拿到 cond_dim；否则用文本潜在维 opt_new.text_latent_dim
        nfeats_new = getattr(model_new, 'input_feats', getattr(opt_new, 'input_feats', 263))
        nfeats_old = getattr(model_old, 'input_feats', getattr(opt_old, 'input_feats', 263))
        nfeats = min(nfeats_old, nfeats_new)
        D_latent = getattr(opt_new, 'text_latent_dim', getattr(opt_old, 'text_latent_dim', 256))
        N_latents = args.num_latents_cond

        x_old = torch.randn(args.batch_size, nfeats, args.nframes, device=dev_old)
        x_new = x_old.detach().to(dev_new)
        t_old = rand_int_timesteps(args.batch_size, dev_old)
        t_new = t_old.detach().to(dev_new)
        cond_old = torch.randn(args.batch_size, N_latents, D_latent, device=dev_old)
        cond_new = cond_old.detach().to(dev_new)
        idx_old = torch.arange(args.batch_size, device=dev_old, dtype=torch.long)
        idx_new = idx_old.detach().to(dev_new)

        with torch.no_grad():
            out_old = model_old.unet(x_old, t_old, cond_old, idx_old)
            out_new = model_new.unet(x_new, t_new, cond_new, idx_new)
        report['sections']['unet_top'] = compare_tensors(out_old, out_new)
    except Exception as e:
        report['sections']['unet_top'] = {'error': repr(e)}

    # 保存报告
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(report, f, indent=2)

    # 控制台摘要
    print('[RESULT] Saved JSON to:', args.out_json)
    sec = report['sections']
    if 'text_encoder_raw' in sec and isinstance(sec['text_encoder_raw'], dict) and 'l2' in sec['text_encoder_raw']:
        print('[RESULT] text_encoder_raw L2:', sec['text_encoder_raw']['l2'])
    if 'connector_sta_vs_sta' in sec and isinstance(sec['connector_sta_vs_sta'], dict) and 'l2' in sec['connector_sta_vs_sta']:
        print('[RESULT] connector_sta_vs_sta L2:', sec['connector_sta_vs_sta']['l2'])
    if 'unet_top' in sec and isinstance(sec['unet_top'], dict) and 'l2' in sec['unet_top']:
        print('[RESULT] unet_top L2:', sec['unet_top']['l2'])


if __name__ == '__main__':
    main()