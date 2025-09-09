#!/usr/bin/env bash
# 复用 eval.sh 的评测入口，但允许自定义 opt 与 which_ckpt 指向 cut_*.tar
# 用法：
#   OPT_PATH=/path/to/opt.txt WHICH=latest_120000 GPU_ID=0 bash eval_old.sh
set -euo pipefail

OPT_PATH="${OPT_PATH:-}"
WHICH="${WHICH:-latest_120000}"
GPU_ID="${GPU_ID:-0}"

if [[ -z "$OPT_PATH" ]]; then
  echo "[ERROR] 必须设置 OPT_PATH 指向目标 opt.txt" >&2
  exit 2
fi
if [[ ! -f "$OPT_PATH" ]]; then
  echo "[ERROR] 找不到 opt: $OPT_PATH" >&2
  exit 2
fi

python -m scripts.evaluation.evaluation \
  --opt_path "$OPT_PATH" \
  --which_ckpt "$WHICH" \
  --num_inference_steps 10 \
  --gpu_id "$GPU_ID"