#!/usr/bin/env bash
# 调试单机多卡设备不一致问题的复现实验脚本
# 改动点：DDPMTrainer 中将 accelerator.prepare 移出 is_main_process，确保每个 rank 都正确放置到对应 GPU。
# 本脚本以较小训练步数运行 2 卡训练，以快速验证是否仍然触发 device mismatch 错误。
set -euo pipefail
TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --config_file 4gpu.yaml --gpu_ids 0,1,2,7 \
  -m scripts.train.train_ddp \
  --abstractor \
  --use_text_cache \
  --model-ema \
  --num_train_steps 150000 \
  --name ant_t2m_ddp \
  --batch_size_eval 32 \
  --batch_size 48 \