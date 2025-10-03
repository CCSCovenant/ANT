import argparse
from .get_opt import get_opt
import yaml

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):
        self.parser.add_argument("--opt_path", type=str, default='/data/kuimou/backup/checkpoints/t2m/t2m_t5/opt.txt',help='option file path for loading model')
        self.parser.add_argument("--gpu_id", type=int, default=7, help='GPU id')

        # evaluator
        self.parser.add_argument("--evaluator_dir", type=str, default='./data/pretrained_models', help='Directory path where save T2M evaluator\'s checkpoints')
        self.parser.add_argument("--eval_meta_dir", type=str, default='./data', help='Directory path where save T2M evaluator\'s normalization data.')
        self.parser.add_argument("--glove_dir",type=str,default='./data/glove', help='Directory path where save glove')
        
        # inference
        self.parser.add_argument("--num_inference_steps", type=int, default=10, help='Number of iterative denoising steps during inference.')
        self.parser.add_argument("--which_ckpt", type=str, default='latest_120000', help='name of checkpoint to load')
        self.parser.add_argument("--diffuser_name", type=str, default='dpmsolver', help='sampler\'s scheduler class name in the diffuser library')
        self.parser.add_argument("--no_ema", action="store_true", help='Where use EMA model in inference')
        self.parser.add_argument("--no_fp16", action="store_true", help='Whether use FP16 in inference')
        
        # evaluation
        self.parser.add_argument("--replication_times", type=int, default=20, help='Number of generation rounds for each text description')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size for eval')
        self.parser.add_argument('--diversity_times', type=int, default=300, help='')
        self.parser.add_argument('--mm_num_samples', type=int, default=100, help='Number of samples for evaluating multimodality')
        self.parser.add_argument('--mm_num_repeats', type=int, default=30, help='Number of generation rounds for each text description when evaluating multimodality')
        self.parser.add_argument('--mm_num_times', type=int, default=10, help='')
        self.parser.add_argument("--evl_mode", type=str, default='fid',help='option file path for loading model')
        self.parser.add_argument('--cfg_scale',type=float, default=2.5, help='')


    
    def parse(self,force_opt_path=None):
        # load evaluation options
        self.opt = self.parser.parse_args()
        opt_dict = vars(self.opt)

        # load the model options of T2m evaluator
        with open('./config/evaluator.yaml', 'r') as yaml_file:
            yaml_config = yaml.safe_load(yaml_file)
        opt_dict.update(yaml_config)

        # load the training options of the selected checkpoint
        if force_opt_path is not None:
            get_opt(self.opt, force_opt_path)
        else:
            get_opt(self.opt, self.opt.opt_path)
       
        return self.opt