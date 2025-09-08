import argparse

import yaml
from .get_opt import get_opt
from os.path import join as pjoin
import os

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # base set
        self.parser.add_argument('--name', type=str, default="t2m", help='Name of this trial')
        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--feat_bias', type=float, default=5, help='Scales for global motion features and foot contact')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--log_every', type=int, default=500, help='Frequency of printing training progress (by iteration)')
        self.parser.add_argument('--save_interval', type=int, default=5_000, help='Frequency of evaluateing and saving models (by iteration)')


        # network hyperparams
        self.parser.add_argument('--num_layers', type=int, default=8, help='num_layers of transformer')
        self.parser.add_argument('--latent_dim', type=int, default=512, help='latent_dim of transformer')
        self.parser.add_argument('--text_latent_dim', type=int, default=256, help='latent_dim of text embeding')
        self.parser.add_argument('--time_dim', type=int, default=512, help='latent_dim of timesteps')
        self.parser.add_argument('--base_dim', type=int, default=512, help='Dimension of Unet base channel')
        self.parser.add_argument('--dim_mults', type=int, default=[2,2,2,2], nargs='+', help='Unet channel multipliers.')
        self.parser.add_argument('--no_eff', action='store_true', help='whether use efficient linear attention')
        self.parser.add_argument('--no_adagn', action='store_true', help='whether use adagn block')
        self.parser.add_argument('--diffusion_steps', type=int, default=1000, help='diffusion_steps of transformer')
        self.parser.add_argument('--prediction_type', type=str, default='sample', help='diffusion_steps of transformer')
        self.parser.add_argument('--text_encoder_type', type=str, default='t5', help='diffusion_steps of transformer')
        self.parser.add_argument('--abstractor', action='store_true', help='diffusion_steps of transformer')
        self.parser.add_argument('--disable_sta', action='store_true', help='enable or disable set module')
        self.parser.add_argument('--laten_size', type=int, default=77, help='laten_size of the ANT')
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout of the ANT')
        # train hyperparams
        self.parser.add_argument('--seed',  type=int, default=0, help='seed for train')
        self.parser.add_argument('--num_train_steps', type=int, default=50, help='Number of training iterations')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        self.parser.add_argument("--decay_rate", default=0.9, type=float, help="the decay rate of lr (0-1 default 0.9)")
        self.parser.add_argument("--update_lr_steps", default=5_000, type=int, help="")
        self.parser.add_argument("--cond_mask_prob", default=0.1, type=float,help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
        self.parser.add_argument('--clip_grad_norm', type=float, default=1, help='Gradient clip')
        self.parser.add_argument('--weight_decay', type=float, default=1e-2, help='Learning rate weight_decay')
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU')
        self.parser.add_argument("--beta_schedule", default='linear', type=str, help="Types of beta in diffusion (e.g. linear, cosine)")
        self.parser.add_argument('--enable_trace', action='store_true', help='enable trace')
        self.parser.add_argument('--use_text_cache', action='store_true', help='cache text embeddings to avoid recomputing')



        # continue training
        self.parser.add_argument('--is_continue', action="store_true", help='Is this trail continued from previous trail?')
        self.parser.add_argument('--continue_ckpt', type=str, default="latest.tar", help='previous trail to continue')
        self.parser.add_argument("--opt_path", type=str, default='',help='option file path for loading model')
        
        # schedular only
        self.parser.add_argument("--which_ckpt", type=str, default='latest_120000', help='name of checkpoint to load')

        # validation 

        # evaluator
        self.parser.add_argument("--evaluator_dir", type=str, default='./data/pretrained_models', help='Directory path where save T2M evaluator\'s checkpoints')
        self.parser.add_argument("--eval_meta_dir", type=str, default='./data', help='Directory path where save T2M evaluator\'s normalization data.')
        self.parser.add_argument("--glove_dir",type=str,default='./data/glove', help='Directory path where save glove')
        
        # inference
        self.parser.add_argument("--num_inference_steps", type=int, default=10, help='Number of iterative denoising steps during inference.')
        self.parser.add_argument("--diffuser_name", type=str, default='dpmsolver', help='sampler\'s scheduler class name in the diffuser library')
        self.parser.add_argument("--no_ema", action="store_true", help='Where use EMA model in inference')
        self.parser.add_argument("--no_fp16", action="store_true", help='Whether use FP16 in inference')
        self.parser.add_argument('--enable_cfg_scheduler', action='store_true', help='enable or disable set module')
        self.parser.add_argument('--cfg_scheduler_type',type=str,default='none',help='set scheduler type')
        # evaluation
        self.parser.add_argument("--replication_times", type=int, default=20, help='Number of generation rounds for each text description')
        self.parser.add_argument('--batch_size_eval', type=int, default=32, help='Batch size for eval')
        self.parser.add_argument('--diversity_times', type=int, default=300, help='')
        self.parser.add_argument('--mm_num_samples', type=int, default=100, help='Number of samples for evaluating multimodality')
        self.parser.add_argument('--mm_num_repeats', type=int, default=30, help='Number of generation rounds for each text description when evaluating multimodality')
        self.parser.add_argument('--mm_num_times', type=int, default=10, help='')
        self.parser.add_argument("--evl_mode", type=str, default='fid',help='option file path for loading model')
        self.parser.add_argument('--cfg_scale',type=float, default=2.5, help='')
        # EMA params
        self.parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
        )
        self.parser.add_argument(
            "--model-ema-steps",
            type=int,
            default=32,
            help="the number of iterations that controls how often to update the EMA model (default: 32)",
        )
        self.parser.add_argument(
            "--model-ema-decay",
            type=float,
            default=0.9999,
            help="decay factor for Exponential Moving Average of model parameters (default: 0.99988)",
        )
        
        self.initialized = True
       
    def parse(self,accelerator):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        
        if self.opt.is_continue:
            assert self.opt.opt_path.endswith('.txt')
            get_opt(self.opt, self.opt.opt_path)
            self.opt.is_train = True
            self.opt.is_continue=True
        elif accelerator.is_main_process:
            args = vars(self.opt)
            accelerator.print('------------ Options -------------')
            for k, v in sorted(args.items()):
                accelerator.print('%s: %s' % (str(k), str(v)))
            accelerator.print('-------------- End ----------------')
            # save to the disk
            expr_dir = pjoin(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
            os.makedirs(expr_dir,exist_ok=True)
            file_name = pjoin(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    if k =='opt_path':
                        continue
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
                # load the model options of T2m evaluator
        with open('./config/evaluator.yaml', 'r') as yaml_file:
            yaml_config = yaml.safe_load(yaml_file)
        opt_dict = vars(self.opt)
        opt_dict.update(yaml_config)

        if self.opt.dataset_name == 't2m' or self.opt.dataset_name == 'humanml':
            self.opt.joints_num = 22
            self.opt.dim_pose = 263
            self.opt.max_motion_length = 196
            self.opt.radius = 4
            self.opt.fps = 20
        elif self.opt.dataset_name == 'cmp':
            self.opt.joints_num = 22
            self.opt.dim_pose = 263
            self.opt.max_motion_length = 196
            self.opt.radius = 4
            self.opt.fps = 20
        elif self.opt.dataset_name == 'kit':
            self.opt.joints_num = 21
            self.opt.dim_pose = 251
            self.opt.max_motion_length = 196
            self.opt.radius = 240 * 8
            self.opt.fps = 12.5
        else:
            raise KeyError('Dataset not recognized')

        self.opt.device = accelerator.device
        self.opt.is_train = True
        return self.opt

        
