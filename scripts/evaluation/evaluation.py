import sys
import os
import torch
from motion_loader import get_dataset_loader, get_motion_loader
from datasets import get_dataset
from models.unet_factory import UnetFactory
from eval import EvaluatorModelWrapper,evaluation
from utils.utils import *
from utils.model_load import load_model_weights

from os.path import join as pjoin

from models.gassuian_diffusion import DiffusePipeline
from accelerate.utils import set_seed

from options.evaluate_options import TestOptions
from eval.eval_t2m_base import evaluation as evaluation_fid


torch.multiprocessing.set_sharing_strategy('file_system')
if __name__ == '__main__':
    parser = TestOptions()
    opt = parser.parse()
    set_seed(0)
    # 兼容旧 opt.txt：若未定义 use_text_cache，则默认关闭以避免评估时报错
    if not hasattr(opt, 'use_text_cache'):
        opt.use_text_cache = False

    #opt.eval_mode = 'mmd'
    device_id = opt.gpu_id
    device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    opt.device = device

    # load evaluator
    eval_wrapper = EvaluatorModelWrapper(opt)

    # load dataset
    gt_loader = get_dataset_loader(opt, opt.batch_size, mode='gt_eval',split='test')
    gen_dataset = get_dataset(opt, mode='eval',split='test')



    # load model
    print(opt)
    model = UnetFactory.create_unet(opt)
    ckpt_path = pjoin(opt.model_dir, opt.which_ckpt + '.tar')  
    load_model_weights(model, ckpt_path, use_ema=not opt.no_ema, device=device)

    # Create a pipeline for generation in diffusion model framework
    pipeline = DiffusePipeline(
        opt = opt,
        model = model, 
        diffuser_name = opt.diffuser_name, 
        device=device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float32 )

    eval_motion_loaders = {
        'text2motion': lambda: get_motion_loader(
            opt,
            opt.batch_size,
            pipeline,
            gen_dataset,
            opt.mm_num_samples,
            opt.mm_num_repeats,
        )
    }

    save_dir = pjoin(opt.save_root,'eval') 
    os.makedirs(save_dir, exist_ok=True)
    if opt.no_ema:
        log_file = pjoin(save_dir,opt.diffuser_name)+f'_{str(opt.num_inference_steps)}setps.log'
    else:
        log_file = pjoin(save_dir,opt.diffuser_name)+f'_{str(opt.num_inference_steps)}steps_ema.log'
    if not os.path.exists(log_file):
        config_dict = dict(pipeline.scheduler.config)
        config_dict['no_ema'] = opt.no_ema
        with open(log_file, 'wt') as f:
            f.write('------------ Options -------------\n')
            for k, v in sorted(config_dict.items()):
                f.write('%s: %s\n' % (str(k), str(v)))
            f.write('-------------- End ----------------\n')
    '''
    if opt.evl_mode == 'mmd':
        all_metrics = evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, opt.replication_times, opt.diversity_times, opt.mm_num_times, run_mm=True,device=device)
    elif opt.evl_mode == 'fid':
    '''
    print("eval with fid")
    all_metrics = evaluation_fid(eval_wrapper, gt_loader, eval_motion_loaders, log_file, opt.replication_times, opt.diversity_times, opt.mm_num_times, run_mm=True)
