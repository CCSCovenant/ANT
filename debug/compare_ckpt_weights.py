#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【脚本作用】
- 对比两份训练检查点（.tar）中的“所有”权重张量键值，输出树状结构的差异报告。
- 特别处理：将重构前 ckpt 中的 ella_model 视为与重构后 ckpt 中的 sta_model 等价（做键名映射）。

【实现要点】
- 递归展开 ckpt 中的字典结构，提取所有张量型叶子（不忽略任何权重）。
- 尝试优先取常见的 state_dict/ema_state_dict/model 等作为根，若未找到则回退为全字典展开。
- 逐键比较：仅旧/仅新/形状不同/数值不同/完全一致，并统计汇总。
- 以点号分层构建树状结构，输出到 diff_weight.txt，同时导出 JSON 汇总便于后续记录。
"""

# 以上为本文件新增：用于说明改动内容与脚本目标（中文注释，便于 Review）

import os
import sys
import json
import argparse
from typing import Mapping

import torch

# ========== 工具函数：判定张量 ==========
# 新增：封装张量判定，便于后续扩展

def is_tensor(x):
    return isinstance(x, torch.Tensor)

# ========== 工具函数：递归展开所有张量叶子 ==========
# 新增：确保不遗漏任何权重，支持多层嵌套与列表/元组下标展开

def flatten_tensors(obj, prefix=''):
    result = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = f"{prefix}{k}" if prefix == '' else f"{prefix}.{k}"
            if is_tensor(v):
                result[key] = v.detach().cpu()
            elif isinstance(v, Mapping):
                result.update(flatten_tensors(v, key))
            else:
                # 处理列表/元组中包含张量/字典的情况，避免遗漏
                if isinstance(v, (list, tuple)):
                    for i, vi in enumerate(v):
                        idx_key = f"{key}.{i}"
                        if is_tensor(vi):
                            result[idx_key] = vi.detach().cpu()
                        elif isinstance(vi, Mapping):
                            result.update(flatten_tensors(vi, idx_key))
                # 其他类型忽略（非权重）
    return result

# ========== 选择权重根节点 ==========
# 新增：优先尝试常见键，尽量锁定真正的 state_dict，避免把训练日志等非权重内容混入

def choose_state_root(ckpt):
    priority_keys = [
        'state_dict', 'ema_state_dict', 'model_state_dict',
        'model', 'ema', 'net', 'unet', 'module'
    ]
    if isinstance(ckpt, Mapping):
        for pk in priority_keys:
            if pk in ckpt and isinstance(ckpt[pk], Mapping):
                return ckpt[pk]
        # 回退：直接返回整个字典，后续 flatten 会只提取张量
        return ckpt
    # 非字典：返回空
    return {}

# ========== 键名标准化（考虑 ella -> sta 映射） ==========
# 新增：重构前的键名中将 ella_model.* 标准化为 sta_model.*

def normalize_key(k: str, *, old_side: bool = False) -> str:
    if old_side:
        k = k.replace('ella_model.', 'sta_model.')
    return k

# ========== 加载并展开 ckpt ==========
# 新增：封装加载+标准化，返回扁平字典与计数信息

def load_and_flatten(path: str, old_side: bool = False):
    obj = torch.load(path, map_location='cpu')
    root = choose_state_root(obj)
    flat = flatten_tensors(root)
    norm = {normalize_key(k, old_side=old_side): v for k, v in flat.items()}
    return norm, {'raw_count': len(flat), 'norm_count': len(norm)}

# ========== 对比逻辑 ==========
# 新增：对比键集合并分别统计各种情况，同时记录数值差异的统计量

def compare_dicts(d_old, d_new, atol=1e-6, rtol=1e-5):
    keys = sorted(set(d_old.keys()) | set(d_new.keys()))
    stats = {
        'only_old': 0,
        'only_new': 0,
        'shared': 0,
        'shared_same': 0,
        'shared_diff_shape': 0,
        'shared_diff_value': 0,
    }
    detail = []  # (key, status, meta)

    for k in keys:
        in_old = k in d_old
        in_new = k in d_new
        if not in_new:
            stats['only_old'] += 1
            detail.append((k, 'only_old', {'old_shape': tuple(d_old[k].shape)}))
            continue
        if not in_old:
            stats['only_new'] += 1
            detail.append((k, 'only_new', {'new_shape': tuple(d_new[k].shape)}))
            continue

        stats['shared'] += 1
        a = d_old[k]
        b = d_new[k]
        if tuple(a.shape) != tuple(b.shape):
            stats['shared_diff_shape'] += 1
            detail.append((k, 'diff_shape', {
                'old_shape': tuple(a.shape),
                'new_shape': tuple(b.shape),
            }))
        else:
            same = torch.allclose(a, b, atol=atol, rtol=rtol)
            if same:
                stats['shared_same'] += 1
                detail.append((k, 'same', {'shape': tuple(a.shape)}))
            else:
                stats['shared_diff_value'] += 1
                diff = (a - b).abs()
                max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
                mean_abs = float(diff.mean().item()) if diff.numel() > 0 else 0.0
                detail.append((k, 'diff_value', {
                    'shape': tuple(a.shape),
                    'max_abs_diff': max_abs,
                    'mean_abs_diff': mean_abs,
                }))
    return stats, detail

# ========== 构建树状结构 ==========
# 新增：按点号分层组织，__leaves__ 存放叶子项

def build_tree(detail):
    tree = {}
    for k, status, meta in detail:
        parts = k.split('.')
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node.setdefault('__leaves__', []).append((parts[-1], status, meta))
    return tree

# ========== 打印树状结构 ==========
# 新增：按缩进打印，并在差异处标注来源

def write_tree(f, node, indent=0, old_path=None, new_path=None):
    sp = '  ' * indent
    for key in sorted([k for k in node.keys() if k != '__leaves__']):
        f.write(f"{sp}{key}/\n")
        write_tree(f, node[key], indent + 1, old_path=old_path, new_path=new_path)
    leaves = node.get('__leaves__', [])
    for name, status, meta in sorted(leaves, key=lambda x: x[0]):
        line = f"{sp}- {name}: "
        if status == 'same':
            line += f"相同 shape={meta['shape']}\n"
        elif status == 'only_old':
            line += f"仅旧（来源: {old_path}） shape={meta['old_shape']}\n"
        elif status == 'only_new':
            line += f"仅新（来源: {new_path}） shape={meta['new_shape']}\n"
        elif status == 'diff_shape':
            line += (
                f"两者皆有但形状不同 old_shape={meta['old_shape']} "
                f"new_shape={meta['new_shape']} （来源: 两者）\n"
            )
        elif status == 'diff_value':
            line += (
                f"两者皆有但数值不同 shape={meta['shape']} "
                f"max_abs_diff={meta['max_abs_diff']:.6g} "
                f"mean_abs_diff={meta['mean_abs_diff']:.6g} （来源: 两者）\n"
            )
        else:
            line += f"{status}\n"
        f.write(line)

# ========== 主流程 ==========
# 新增：支持 --summary_json 输出汇总，便于后续生成 change.md

def main():
    parser = argparse.ArgumentParser(
        description='对比两份 ckpt 的权重（考虑 ella->sta 键名映射），输出树状 diff 报告'
    )
    parser.add_argument('--old', required=True, help='重构前 ckpt 路径')
    parser.add_argument('--new', required=True, help='重构后 ckpt 路径')
    parser.add_argument('--output', required=True, help='diff 文本输出路径')
    parser.add_argument('--summary_json', default=None, help='可选：统计汇总 JSON 输出路径')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 加载与展开（旧侧执行 ella->sta 标准化）
    old_flat, old_info = load_and_flatten(args.old, old_side=True)
    new_flat, new_info = load_and_flatten(args.new, old_side=False)

    # 对比
    stats, detail = compare_dicts(old_flat, new_flat)

    # 汇总
    summary = {
        'old_ckpt': args.old,
        'new_ckpt': args.new,
        'old_raw_keys': old_info['raw_count'],
        'new_raw_keys': new_info['raw_count'],
        'old_norm_keys': len(old_flat),
        'new_norm_keys': len(new_flat),
        **stats,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('权重对比报告（考虑 ella_model ≡ sta_model 映射）\n')
        f.write(f"旧: {args.old}\n")
        f.write(f"新: {args.new}\n")
        f.write('\n[统计汇总]\n')
        # 固定顺序输出，便于读取
        ordered_keys = [
            'old_raw_keys','new_raw_keys','old_norm_keys','new_norm_keys',
            'only_old','only_new','shared','shared_same','shared_diff_shape','shared_diff_value'
        ]
        for k in ordered_keys:
            v = summary.get(k, None)
            if v is not None:
                f.write(f"- {k}: {v}\n")
        f.write('\n[详细树状结构]\n')
        tree = build_tree(detail)
        write_tree(f, tree, indent=0, old_path=args.old, new_path=args.new)

    if args.summary_json:
        with open(args.summary_json, 'w', encoding='utf-8') as jf:
            json.dump(summary, jf, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()