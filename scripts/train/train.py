import sys
import os
from os.path import join as pjoin
from options.train_options import TrainOptions
from utils.plot_script import *

from models.unet_factory import UnetFactory
from utils.ema import ExponentialMovingAverage
from trainers import DDPMTrainer
from motion_loader import get_dataset_loader

from accelerate.utils import set_seed
from accelerate import Accelerator
# 使用 DistributedDataParallel 进行单机多卡训练
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import TorchDynamoPlugin



if __name__ == '__main__':
    
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        bucket_cap_mb=25,          # 优化通信效率
        gradient_as_bucket_view=True
    )

    # Configure the compilation backend
    '''
    dynamo_plugin = TorchDynamoPlugin(
        backend="aot_eager",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
        mode="default",      # Options: "default", "reduce-overhead", "max-autotune"
        fullgraph=True,
        dynamic=False
    )
    '''

    # 创建 accelerator
    accelerator = Accelerator(
        mixed_precision="no",      # 保持float32训练
        kwargs_handlers=[ddp_kwargs]
        #dynamo_plugin=dynamo_plugin
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
    print(opt)
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
    '''
    do not compile model, accelerator will compile it later
    accelerator.print('compile model ...')
    encoder.compile()
    accelerator.print('Finish compiling Model.\n')
    '''
    accelerator.print('Model trainable parameters:', sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    accelerator.print('Model total parameters:', sum(p.numel() for p in encoder.parameters()))

    trainer = DDPMTrainer(opt, encoder,accelerator, model_ema,run_name=opt.name)

    trainer.train(train_datasetloader)


