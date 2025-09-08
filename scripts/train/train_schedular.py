import sys
import os
#os.chdir('/data/wenshuo/project/StableMoFusion')
#sys.path.append('/data/wenshuo/project/StableMoFusion')
from os.path import join as pjoin
from options.train_options import TrainOptions
from utils.plot_script import *

from models import build_models
from utils.ema import ExponentialMovingAverage
from trainers import DDPMTrainer_schedular
from motion_loader import get_dataset_loader

from utils.model_load import load_model_weights
from accelerate.utils import set_seed
from accelerate import Accelerator
# 使用 DistributedDataParallel 进行单机多卡训练
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from accelerate.utils import DistributedDataParallelKwargs



if __name__ == '__main__':
    
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        bucket_cap_mb=25,          # 优化通信效率
        gradient_as_bucket_view=True
    )

    # 创建 accelerator
    accelerator = Accelerator(
        mixed_precision="no",      # 保持float32训练
        kwargs_handlers=[ddp_kwargs]
    )


    
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
    encoder = build_models(opt)
    ckpt_path = "/data/kuimou/SET/checkpoints/t2m/ella_200000/model/latest_120000.tar"
    niter = load_model_weights(encoder, ckpt_path, use_ema=True)
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

    for param in encoder.parameters():
        param.requires_grad = False
    # 解冻 cfg_scheduler 模块中所有参数
    for param in encoder.cfg_scheduler.parameters():
        param.requires_grad = True
    
    for param in encoder.cfg_scheduler_proj.parameters():
        param.requires_grad = True
        
    trainer = DDPMTrainer_schedular(opt, encoder,accelerator, model_ema)

    trainer.train(train_datasetloader)


