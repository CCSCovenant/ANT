import sys
import os
# 中文注释：使用当前项目目录，注释掉外部工程路径依赖，避免路径冲突
# os.chdir('/home/data2/bkjiahaozhe/StableMoFusion/wenshuo/StableMoFusion')
# sys.path.append('/home/data2/bkjiahaozhe/StableMoFusion/wenshuo/StableMoFusion')

# 中文注释：确保项目根目录在 sys.path 中，避免从 scripts/train 目录运行时无法导入 options 等本地包
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from os.path import join as pjoin
from options.train_options import TrainOptions
from utils.plot_script import *

# 中文注释：改为使用 UnetFactory 创建模型（与单卡脚本一致）
from models.unet_factory import UnetFactory
from utils.ema import ExponentialMovingAverage
from trainers.ddpm_trainer_ddp import DDPMTrainer  # 中文注释：使用分布式版本的DDPMTrainer
from motion_loader import get_dataset_loader

from accelerate.utils import set_seed
from accelerate import Accelerator
# 使用 DistributedDataParallel 进行单机多卡训练
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
if __name__ == '__main__':
    ddp_kwargs = DistributedDataParallelKwargs(
            static_graph=True,                 # 中文注释：静态图时能显著降低重建桶和参数跟踪的开销
            gradient_as_bucket_view=True,      # 中文注释：减少内存复制，加速梯度聚合
            bucket_cap_mb=64,                  # 中文注释：可按模型大小与硬件带宽调参（32~128均可尝试）
            broadcast_buffers=False            # 中文注释：通常无需广播 buffers，减少不必要通信
        )    
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    
    parser = TrainOptions()
    opt = parser.parse(accelerator)
    set_seed(opt.seed)
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if accelerator.is_main_process:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)

    train_datasetloader = get_dataset_loader(opt,  batch_size = opt.batch_size, split='train', accelerator=accelerator, mode='train') # 7169


    accelerator.print('\nInitializing model ...' )
    # 中文注释：与单卡脚本一致，使用工厂创建UNet
    encoder = UnetFactory.create_unet(opt)
    model_ema = None
    if opt.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        adjust = 106_667 * opt.model_ema_steps / opt.num_train_steps
        alpha = 1.0 - opt.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        print('EMA alpha:',alpha)
        model_ema = ExponentialMovingAverage(encoder, decay=1.0 - alpha)
    accelerator.print('Finish building Model.\n')
    accelerator.print('Model trainable parameters:', sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    accelerator.print('Model total parameters:', sum(p.numel() for p in encoder.parameters()))
    trainer = DDPMTrainer(opt, encoder,accelerator, model_ema)

    trainer.train(train_datasetloader)


