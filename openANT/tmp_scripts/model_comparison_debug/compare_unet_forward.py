#!/usr/bin/env python3
"""
Compare forward outputs and key intermediates between OLD (backup) and NEW (openANT) T2M-Unet models.

- Builds both models from their respective opt.txt.
- Loads checkpoints (preferring EMA when requested).
- Generates a unified, deterministic input (motion x, timesteps t, and text).
- Registers forward hooks on key layers to capture intermediate activations.
- Runs a single forward pass on both models and reports:
  * Output tensor differences (L2, cosine, mean/std, shape)
  * Selected intermediate layer digests (shape/mean/std/min/max)
  * Attention weights if available via return_intermediates in NEW model

Usage example:
  python openANT/tmp_scripts/model_comparison_debug/compare_unet_forward.py \
    --old_opt /data/kuimou/backup/checkpoints/t2m/t2m_t5/opt.txt \
    --old_ckpt /data/kuimou/backup/checkpoints/t2m/t2m_t5/model/latest_120000.tar \
    --new_opt /data/kuimou/openANT/checkpoints/t2m/ant_t2m/opt.txt \
    --new_ckpt /data/kuimou/openANT/checkpoints/t2m/ant_t2m/model/old_best.tar \
    --device cpu --text "a person walks forward slowly" --batch_size 1 --nframes 64
"""

import os
from pickle import FALSE
import sys
import json
import argparse
import types
from types import SimpleNamespace
from collections import defaultdict

import torch
import torch.nn.functional as F
import ast


OPENANT_ROOT = "/data/kuimou/openANT"
BACKUP_ROOT = "/data/kuimou/backup"


def add_paths():
    """Ensure both repos are importable."""
    if OPENANT_ROOT not in sys.path:
        sys.path.insert(0, OPENANT_ROOT)
    if BACKUP_ROOT not in sys.path:
        sys.path.insert(0, BACKUP_ROOT)


def load_opt_via_get_opt(repo_root: str, opt_txt_path: str):
    """Load opt via repo-specific get_opt; fallback to parsing opt.txt.

    - For openANT: options.get_opt.get_opt
    - For backup: scripts.get_opt.get_opt
    - Fallback: parse opt.txt into SimpleNamespace
    """
    repo_root = os.path.abspath(repo_root)
    sys.path.insert(0, repo_root)
    try:
        opt = None
        # Prefer openANT's loader when repo_root is OPENANT_ROOT
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
            # Fallback: directly parse opt.txt into a SimpleNamespace
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
        # Some opts may miss dim_pose; ensure a reasonable default for construction
        if not hasattr(opt, 'dim_pose'):
            setattr(opt, 'dim_pose', 263)
        # Ensure device field exists; default to 'cpu' when absent
        if not hasattr(opt, 'device'):
            setattr(opt, 'device', 'cpu')
        return opt
    finally:
        if repo_root in sys.path:
            sys.path.remove(repo_root)


def build_model_old(opt):
    """Build OLD model from backup repo's models.build_models(opt)."""
    sys.path.insert(0, BACKUP_ROOT)
    try:
        # Provide a minimal stub for 'clip' to satisfy imports if backup unconditionally imports it
        if 'clip' not in sys.modules:
            clip_stub = types.ModuleType('clip')
            def _noop(*args, **kwargs):
                return None
            clip_stub.load = _noop
            sys.modules['clip'] = clip_stub
        from models import build_models as build_old
        model = build_old(opt)
    finally:
        # Prevent cross-pollution: ensure backup 'models' package doesn't shadow openANT's models
        if 'models' in sys.modules:
            try:
                # only remove if it points to backup repo
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
    """Build NEW model via UnetFactory.create_unet(opt)."""
    # Construct a proper 'models' package from disk so relative imports work,
    # then load 'models.unet_factory' directly without executing models/__init__.py
    import importlib.util, pathlib, types

    root = pathlib.Path(OPENANT_ROOT)
    models_dir = root / 'models'
    uf_path = models_dir / 'unet_factory.py'
    if not uf_path.exists():
        raise FileNotFoundError(f"unet_factory.py not found at {uf_path}")

    # Create a lightweight package placeholder for 'models'
    pkg = types.ModuleType('models')
    pkg.__file__ = str((models_dir / '__init__.py').resolve())
    pkg.__path__ = [str(models_dir.resolve())]
    sys.modules['models'] = pkg

    # Load submodule with package-qualified name so relative imports in it resolve
    spec = importlib.util.spec_from_file_location('models.unet_factory', str(uf_path))
    uf_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(uf_module)
    model = uf_module.UnetFactory.create_unet(opt)
    return model


def load_weights(model, ckpt_path, use_ema=True, device='cpu'):
    """Use openANT's load_model_weights for both models (compatible tar format)."""
    sys.path.insert(0, OPENANT_ROOT)
    try:
        from utils.model_load import load_model_weights
        niter = load_model_weights(model, ckpt_path, use_ema=use_ema, device=device)
        return niter
    finally:
        if OPENANT_ROOT in sys.path:
            sys.path.remove(OPENANT_ROOT)


def tensor_digest(x: torch.Tensor):
    """Return a compact digest for a tensor to compare numerically without dumping full arrays."""
    if not torch.is_tensor(x):
        return {'type': str(type(x))}
    return {
        'shape': list(x.shape),
        'dtype': str(x.dtype),
        'mean': float(x.mean().item()),
        'std': float(x.std().item()),
        'min': float(x.min().item()),
        'max': float(x.max().item()),
    }


def register_feature_hooks(model: torch.nn.Module, patterns=None):
    """
    Register forward hooks for modules whose qualified name contains any of the given patterns.
    Returns a dict to be filled at runtime and a list of hook handles for later removal.
    """
    if patterns is None:
        patterns = [
            'sta', 'ella', 'connector', 'perceiver', 'cross_attn', 'CrossAttention',
            'ResidualCrossAttention', 'mid', 'downs', 'ups', 'unet'
        ]

    captured = {}
    handles = []

    def _should_hook(name, module):
        lname = name.lower()
        for p in patterns:
            if p.lower() in lname:
                return True
        # Also hook by class name for attention/resampler
        cls = module.__class__.__name__.lower()
        for p in ['perceiver', 'resampler', 'crossattention', 'linearcrossattention']:
            if p in cls:
                return True
        return False

    def _hook(name):
        def fn(module, inp, out):
            # store a digest only
            try:
                captured[name] = tensor_digest(out)
            except Exception:
                captured[name] = {'error': 'digest_failed'}
        return fn

    for name, module in model.named_modules():
        if _should_hook(name, module):
            try:
                h = module.register_forward_hook(_hook(name))
                handles.append(h)
            except Exception:
                pass
    return captured, handles


def make_inputs(opt, batch_size=1, nframes=64, text_str="a person walks forward slowly", device='cpu'):
    torch.manual_seed(0)
    dim_pose = getattr(opt, 'dim_pose', 263)
    x = torch.randn(batch_size, nframes, dim_pose, device=device)
    # typical diffusion timesteps are int64 in [0, 1000)
    t = torch.randint(low=0, high=1000, size=(batch_size,), dtype=torch.long, device=device)
    return x, t, text_str


def compare_tensors(a: torch.Tensor, b: torch.Tensor):
    # Move both to CPU to ensure device match for arithmetic/metrics
    a_cpu = a.detach().to('cpu')
    b_cpu = b.detach().to('cpu')
    diff = (a_cpu - b_cpu).float()
    l2 = float(torch.norm(diff).item())
    mean_abs = float(diff.abs().mean().item())
    cos = float(F.cosine_similarity(a_cpu.reshape(1, -1), b_cpu.reshape(1, -1)).item())
    return {
        'a_shape': list(a_cpu.shape),
        'b_shape': list(b_cpu.shape),
        'l2': l2,
        'mean_abs': mean_abs,
        'cosine': cos,
        'a_digest': tensor_digest(a_cpu),
        'b_digest': tensor_digest(b_cpu),
    }


def main():
    parser = argparse.ArgumentParser("Single-pass forward comparison for OLD vs NEW T2M-Unet")
    parser.add_argument('--old_opt', required=True)
    parser.add_argument('--old_ckpt', required=True)
    parser.add_argument('--new_opt', required=True)
    parser.add_argument('--new_ckpt', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nframes', type=int, default=64)
    parser.add_argument('--text', type=str, default='a person walks forward slowly')
    parser.add_argument('--use_ema', action='store_true', help='Prefer EMA weights if available')
    parser.add_argument('--out_json', default='openANT/tmp_scripts/model_comparison_debug/output/forward_compare.json')
    parser.add_argument('--device_old', default=None, help='Device for OLD model, e.g., cuda:0 or cpu')
    parser.add_argument('--device_new', default=None, help='Device for NEW model, e.g., cuda:1 or cpu')

    args = parser.parse_args()

    add_paths()

    # Load opts
    print('[INFO] Loading opts...')
    opt_old = load_opt_via_get_opt(BACKUP_ROOT, args.old_opt)
    opt_new = load_opt_via_get_opt(OPENANT_ROOT, args.new_opt)

    # Resolve per-model devices (default to splitting across GPUs when available)
    def _resolve_dev(dev_str, fallback):
        if dev_str is None:
            return fallback
        if dev_str == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        if isinstance(dev_str, str) and dev_str.startswith('cuda') and torch.cuda.is_available():
            return torch.device(dev_str)
        return torch.device('cpu')

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        default_old = torch.device('cuda:0')
        default_new = torch.device('cuda:1')
    elif torch.cuda.is_available():
        default_old = torch.device('cuda:0')
        default_new = torch.device('cuda:0')
    else:
        default_old = torch.device('cpu')
        default_new = torch.device('cpu')

    device_old = _resolve_dev(args.device_old, default_old)
    device_new = _resolve_dev(args.device_new, default_new)
    print(f'[INFO] Using devices -> OLD: {device_old}, NEW: {device_new}')

    # Force opts to respect the chosen devices to avoid internal mismatches
    opt_old.device = str(device_old)
    opt_new.device = str(device_new)

    # Build models
    print('[INFO] Building models...')
    model_old = build_model_old(opt_old).to(device_old).eval()
    model_new = build_model_new(opt_new).to(device_new).eval()

    # Load weights
    print('[INFO] Loading checkpoints...')
    _ = load_weights(model_old, args.old_ckpt, use_ema=args.use_ema, device=device_old)
    _ = load_weights(model_new, args.new_ckpt, use_ema=args.use_ema, device=device_new)

    # Register hooks
    print('[INFO] Registering hooks...')
    old_feats, old_handles = register_feature_hooks(model_old)
    new_feats, new_handles = register_feature_hooks(model_new)

    # Prepare inputs
    print('[INFO] Preparing inputs...')
    x_old, t_old, text = make_inputs(opt_old, batch_size=args.batch_size, nframes=args.nframes, text_str=args.text, device=device_old)
    # Ensure identical tensors for NEW (move to its device while keeping values identical)
    x_new = x_old.detach().to(device_new)
    t_new = t_old.detach().to(device_new)

    # Forward
    print('[INFO] Running forward...')
    with torch.no_grad():
        # OLD forward signature: positional args only; no return_intermediates support
        out_old = model_old(x_old, t_old, text=text)
        # NEW forward signature: use named args for clarity; do not request intermediates
        out_new = model_new(x_new, timesteps=t_new, text=text, return_intermediates=False)

    # Remove hooks
    for h in old_handles + new_handles:
        try:
            h.remove()
        except Exception:
            pass

    # Compare outputs
    print('[INFO] Comparing final outputs...')
    # Handle potential tuple returns (new may return (out, intermediates) when requested)
    old_out, old_inter = (out_old if isinstance(out_old, (tuple, list)) else (out_old, {}))
    new_out, new_inter = (out_new if isinstance(out_new, (tuple, list)) else (out_new, {}))

    compare = compare_tensors(old_out, new_out)

    def digest_intermediates(inter):
        def _dig(val):
            if torch.is_tensor(val):
                return tensor_digest(val)
            if isinstance(val, (list, tuple)):
                return [_dig(v) for v in val]
            if isinstance(val, dict):
                return {k: _dig(v) for k, v in val.items()}
            return {'type': str(type(val))}
        try:
            return _dig(inter)
        except Exception:
            return {'error': 'digest_failed'}

    # Align intermediate keys for reporting
    def shortlist(feats: dict):
        # Only include a stable subset to keep JSON readable
        keys = sorted(feats.keys())
        selected = {}
        for k in keys:
            if any(p in k.lower() for p in ['sta', 'ella', 'perceiver', 'cross', 'mid', 'downs', 'ups', 'unet']):
                selected[k] = feats[k]
        return selected

    report = {
        'device_old': str(device_old),
        'device_new': str(device_new),
        'text': args.text,
        'batch_size': args.batch_size,
        'nframes': args.nframes,
        'old_opt': args.old_opt,
        'old_ckpt': args.old_ckpt,
        'new_opt': args.new_opt,
        'new_ckpt': args.new_ckpt,
        'output_compare': compare,
        'old_intermediates_hooks': shortlist(old_feats),
        'new_intermediates_hooks': shortlist(new_feats),
        'old_return_intermediates': digest_intermediates(old_inter),
        'new_return_intermediates': digest_intermediates(new_inter),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(report, f, indent=2)

    # Pretty print summary to stdout
    print('[RESULT] Output L2:', report['output_compare']['l2'])
    print('[RESULT] Output cosine:', report['output_compare']['cosine'])
    print('[RESULT] Output mean_abs:', report['output_compare']['mean_abs'])
    print('[RESULT] Saved JSON to:', args.out_json)


if __name__ == '__main__':
    main()