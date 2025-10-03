import torch
import time
import torch.optim as optim
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin

from diffusers import  DDPMScheduler
from torch.utils.tensorboard import SummaryWriter
import time
import pdb
import sys
import os
from torch.optim.lr_scheduler import ExponentialLR
import torch.multiprocessing as mp

# 这个 sampler 可以把采样的数据分散到各个 CPU 上                                      
from torch.utils.data.distributed import DistributedSampler     

# 实现分布式数据并行的核心类        
from torch.nn.parallel import DistributedDataParallel as DDP         

# DDP 在每个 GPU 上运行一个进程，其中都有一套完全相同的 Trainer 副本（包括model和optimizer）
# 各个进程之间通过一个进程池进行通信，这两个方法来初始化和销毁进程池
from torch.distributed import init_process_group, destroy_process_group 

# === 新增导入（与单卡 Trainer 对齐）===
# 中文注释：为了实现训练中评估与生成，需要引入评测模块和生成 Pipeline
from eval.eval_t2m_base import evaluation as evaluation_fid  # 中文注释：评估入口
from eval import EvaluatorModelWrapper  # 中文注释：评估器包装器
from motion_loader import get_dataset_loader, get_motion_loader  # 中文注释：评估阶段的数据加载器
from datasets import get_dataset  # 中文注释：评估时需要取评测数据集
from models.gassuian_diffusion import DiffusePipeline  # 中文注释：扩散生成 Pipeline

class DDPMTrainer(object):

    def __init__(self, args, model,accelerator, model_ema=None):
        self.opt = args
        self.accelerator = accelerator
        self.device = self.accelerator.device
        # 中文注释：移除对 gpu_id/DDP 的手动包装，统一交给 accelerate.prepare 管理设备与并行
        self.model = model  # 中文注释：模型实例，稍后由 accelerator.prepare 放到各 rank 设备
        self.diffusion_steps = args.diffusion_steps
        self.noise_scheduler = DDPMScheduler(num_train_timesteps= self.diffusion_steps,
            beta_schedule=args.beta_schedule,
            variance_type="fixed_small",
            prediction_type= args.prediction_type,
            clip_sample=False)
        self.model_ema = model_ema
        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')

        self.accelerator.print('Diffusion_config:\n',self.noise_scheduler.config)

        if self.accelerator.is_main_process:
            starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
            print("Start experiment:", starttime)
            # 中文注释：主进程创建 TensorBoard 日志记录器
            self.writer = SummaryWriter(log_dir=pjoin(args.save_root,'logs_')+starttime[:16],comment=starttime[:16],flush_secs=60)
        self.accelerator.wait_for_everyone()

        # 中文注释：优化器与学习率调度器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=args.decay_rate) if args.decay_rate>0 else None

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    def clip_norm(self,network_list):
        # 中文注释：使用 accelerator 的 clip_grad_norm_ 以兼容分布式梯度
        for network in network_list:
            self.accelerator.clip_grad_norm_(network.parameters(), self.opt.clip_grad_norm) # 0.5 -> 1

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data):
        """
        中文注释：
        - 对齐单卡版本 forward，支持 text cache（raw_embeds），以避免重复编码文本。
        - 当 use_text_cache 为真时，从 batch 最后一项读取 raw_embeds，并在调用模型时将 text=None。
        """
        # 拆包（与单卡版本保持一致）
        caption = batch_data[0]
        motions = batch_data[1].detach().float()
        m_lens  = batch_data[2]
        raw_embeds = batch_data[-1] if (self.opt.use_text_cache and torch.is_tensor(batch_data[-1])) else None  # 中文注释：使用 text cache

        x_start = motions
        B, T = x_start.shape[:2]
        #cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
        cur_len = torch.as_tensor(m_lens, device=x_start.device).clamp_max(T).long()

        #self.src_mask = self.generate_src_mask(T, cur_len).to(x_start.device)
        self.src_mask = self.generate_src_mask(T, cur_len, device=x_start.device)
        # 1. Sample noise that we'll add to the motion
        real_noise = torch.randn_like(x_start)

        # 2. Sample a random timestep for each motion
        t = torch.randint(0, self.diffusion_steps, (B,), device=self.device)
        self.timesteps = t

        # 3. Add noise to the motion according to the noise magnitude at each timestep
        x_t = self.noise_scheduler.add_noise(x_start, real_noise, t)

        # 4. network prediction（支持 raw_embeds）
        self.prediction = self.model(
            x_t,
            t,
            text=None if raw_embeds is not None else caption,
            raw_embeds=raw_embeds
        )
        
        if self.opt.prediction_type =='sample':
            self.target = x_start
        elif self.opt.prediction_type == 'epsilon':
            self.target = real_noise
        elif self.opt.prediction_type == 'v_prediction':
            self.target = self.noise_scheduler.get_velocity(x_start, real_noise, t)

    def masked_l2(self, a, b, mask, weights):
        
        loss = self.mse_criterion(a, b).mean(dim=-1) # (bath_size, motion_length)
        
        loss = (loss * mask).sum(-1) / mask.sum(-1) # (batch_size, )

        loss = (loss * weights).mean()

        return loss

    def backward_G(self):
        loss_logs = OrderedDict({})
        mse_loss_weights = torch.ones_like(self.timesteps)
        loss_logs['loss_mot_rec']= self.masked_l2(self.prediction, self.target, self.src_mask, mse_loss_weights)

        self.loss = loss_logs['loss_mot_rec'] 

        return loss_logs

    def update(self):
        self.zero_grad([self.optimizer])
        loss_logs = self.backward_G()
        self.accelerator.backward(self.loss)
        self.clip_norm([self.model])
        self.step([self.optimizer])

        return loss_logs
    
    def generate_src_mask(self, T, length, device):
        # 中文注释【性能优化】：矢量化构造 mask；mask[b, t] = 1 若 t < length[b]，否则 0；避免 O(B*T) 的 Python 循环
        ar = torch.arange(T, device=device)              # [T]
        mask = ar.unsqueeze(0) < length.unsqueeze(1)     # [B, T] bool
        return mask.float()
    '''
    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask
    '''
    def train_mode(self):
        self.model.train()
        if self.model_ema:
            self.model_ema.train()

    def eval_mode(self):
        self.model.eval()
        if self.model_ema:
            self.model_ema.eval()

    def save(self, file_name,total_it):
        # 中文注释：仅保存非 text_encoder 的参数，避免在加载时强绑定文本编码器权重
        full_state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        filtered_state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith('text_encoder.')}
        state = {
            'opt_encoder': self.optimizer.state_dict(),
            'total_it': total_it,
            'encoder': filtered_state_dict,
        }
        if self.model_ema:
            full_ema_state_dict = self.accelerator.unwrap_model(self.model_ema).module.state_dict()
            filtered_ema_state_dict = {k: v for k, v in full_ema_state_dict.items() if not k.startswith('text_encoder.')}
            state["model_ema"] = filtered_ema_state_dict
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['opt_encoder'])
        if self.model_ema:
            self.model_ema.load_state_dict(checkpoint["model_ema"], strict=False)  # 中文注释：与单卡版本对齐，宽松加载
        self.model.load_state_dict(checkpoint['encoder'], strict=False)
       
        return checkpoint.get('total_it', 0)

    # === 新增：评估流程，与单卡版本保持一致 ===
    def evaluate(self, total_it, eval_wrapper, gt_loader, gen_dataset):
        """
        Performs evaluation on the validation set.
        在验证集上执行评估。
        """
        self.accelerator.print(f"--- Starting evaluation at iteration {total_it} ---")
        self.eval_mode()  # 中文注释：切换到评估模式

        # 中文注释：确定用于评估的模型（优先 EMA）
        model_to_eval = self.accelerator.unwrap_model(self.model)
        use_ema = self.model_ema and not self.opt.no_ema
        if use_ema:
            self.accelerator.print("Using EMA model for evaluation.")
            ema_model = self.accelerator.unwrap_model(self.model_ema)
            model_to_eval = ema_model.module
        else:
            self.accelerator.print("Using standard model for evaluation.")

        # 中文注释：更新评估器包装器中的模型引用
        eval_wrapper.model = model_to_eval

        # 中文注释：构造扩散生成 Pipeline
        pipeline = DiffusePipeline(
            opt=self.opt,
            model=model_to_eval,
            diffuser_name=self.opt.diffuser_name,
            device=self.device,
            num_inference_steps=self.opt.num_inference_steps,
            torch_dtype=torch.float32
        )

        # 中文注释：构建评估阶段的 motion loaders
        eval_motion_loaders = {
            'text2motion': lambda: get_motion_loader(
                self.opt,
                self.opt.batch_size_eval,
                pipeline,
                gen_dataset,
                self.opt.mm_num_samples,
                self.opt.mm_num_repeats,
            )
        }

        # 定义日志文件路径
        save_dir = pjoin(self.opt.save_root, 'eval')
        os.makedirs(save_dir, exist_ok=True)
        log_file = pjoin(save_dir, f'eval_log_it_{total_it}.log')

        # 运行评估（仅主进程打印日志，但评估内部只在主进程触发）
        all_metrics = evaluation_fid(
            eval_wrapper, gt_loader, eval_motion_loaders, log_file,
            2, self.opt.diversity_times, self.opt.mm_num_times, run_mm=True
        )

        self.accelerator.print(f"--- Evaluation at iteration {total_it} complete ---")
        self.train_mode()  # 中文注释：切回训练模式

    def train(self, train_loader):
        
        it = 0
        if self.opt.is_continue:
            model_path = pjoin(self.opt.model_dir, self.opt.continue_ckpt)         
            it = self.load(model_path)
            self.accelerator.print(f'continue train from  {it} iters in {model_path}')
        start_time = time.time()

        logs = OrderedDict()
        self.dataset = train_loader.dataset
        
        # 中文注释：将损失函数与模型/优化器/数据加载器统一放到各自 rank 的设备与进程上
        self.model, self.optimizer, train_loader,self.model_ema,self.mse_criterion= self.accelerator.prepare(
            self.model, self.optimizer, train_loader,self.model_ema,self.mse_criterion
        )


        num_epochs = (self.opt.num_train_steps-it)//len(train_loader)  + 1 
        if self.accelerator.is_main_process:
            self.accelerator.print('Setting up evaluation components...')
            eval_split = 'val'
            eval_wrapper = EvaluatorModelWrapper(self.opt)
            # 中文注释：评估阶段的GT DataLoader使用评估专用的batch_size_eval，
            # 避免训练batch过小（<3）导致R-precision中top_k=3索引越界
            gt_loader = get_dataset_loader(self.opt, self.opt.batch_size_eval, mode='gt_eval', split='test')
            gen_dataset = get_dataset(self.opt, mode='eval', split='test')
            self.accelerator.print('Evaluation components are ready.')
            self.accelerator.print(f'need to train for {num_epochs} epochs....')
        
        for epoch in range(0, num_epochs):
            self.train_mode()

            # 中文注释：如果存在分布式采样器，需在每个 epoch 设置随机种子，以确保各 rank 数据划分不同
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            for i, batch_data in enumerate(train_loader):
                self.forward(batch_data)
                log_dict = self.update()
                it += 1

                if self.model_ema and it % self.opt.model_ema_steps == 0:
                    self.accelerator.unwrap_model(self.model_ema).update_parameters(self.model)

                # update logger（累计 loss）
                for k, v in log_dict.items():
                    value = v.item() if torch.is_tensor(v) else v
                    logs[k] = logs.get(k, 0) + value
                
                if it % self.opt.log_every == 0 :                   
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(self.accelerator,start_time, it, mean_loss, epoch, inner_iter=i)
                    if self.accelerator.is_main_process:
                        self.writer.add_scalar("loss/train", mean_loss['loss_mot_rec'], it)
                    self.accelerator.wait_for_everyone()
                
                if it % self.opt.save_interval == 0 and self.accelerator.is_main_process: # 500
                    # 中文注释：保存 checkpoint（忽略 text_encoder）并执行评估
                    self.save(pjoin(self.opt.model_dir, f'latest_{it}.tar'), it)
                    # 仅主进程调起评估，避免重复计算
                    self.evaluate(it, eval_wrapper, gt_loader, gen_dataset)
                self.accelerator.wait_for_everyone()

                if (self.scheduler is not None) and (it % self.opt.update_lr_steps == 0) :
                    self.scheduler.step()

        # Save the last checkpoint if it wasn't already saved.
        if it % self.opt.save_interval != 0 and self.accelerator.is_main_process:
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), it)

        self.accelerator.wait_for_everyone()
        self.accelerator.print('FINISH')

 