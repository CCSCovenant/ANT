import sys
import torch
torch._dynamo.config.capture_dynamic_output_shape_ops = True
import time
import torch.optim as optim
from collections import OrderedDict
from os.path import join as pjoin
import os
import wandb
#import nvtx

from diffusers import DDPMScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

from eval.eval_t2m_base import evaluation as evaluation_fid
from eval import EvaluatorModelWrapper
from datasets import get_dataset
from motion_loader import get_dataset_loader, get_motion_loader
from models.gassuian_diffusion import DiffusePipeline
from utils.utils import print_current_loss
import time
import uuid


class DDPMTrainer(object):

    def __init__(self, args, model, accelerator, model_ema=None, run_name="ANT"):
        self.opt = args
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.model = model
        self.diffusion_steps = args.diffusion_steps
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.diffusion_steps,
                                             beta_schedule=args.beta_schedule,
                                             variance_type="fixed_small",
                                             prediction_type=args.prediction_type,
                                             clip_sample=False)
        self.model_ema = model_ema
        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')

        accelerator.print('Diffusion_config:\n', self.noise_scheduler.config)

        if self.accelerator.is_main_process:
                    # --- Generate the dynamic run name ---
                    # Get components: dataset, timestamp, and a unique ID
                    dataset_name = args.dataset_name
                    timestamp = time.strftime("%Y%m%d-%H")
                    unique_id = str(uuid.uuid4())[:6]  # 6-character short unique hash

                    # Combine them into the final name, e.g., "t2m_20250721-20_a1b2c3"
                    generated_run_name = f"{dataset_name}_{timestamp}_{unique_id}"
                    
                    # --- Use the generated name in wandb.init ---
                    #wandb.init(project=run_name, name=generated_run_name, config=vars(args))
                    
                    starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
                    print("Start experiment:", starttime)
                    self.writer = SummaryWriter(log_dir=pjoin(args.save_root, 'logs_') + starttime[:16],
                                                comment=starttime[:16], flush_secs=60)
        self.accelerator.wait_for_everyone()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=args.decay_rate) if args.decay_rate > 0 else None

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    def clip_norm(self, network_list):
        for network in network_list:
            self.accelerator.clip_grad_norm_(network.parameters(), self.opt.clip_grad_norm)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data):
        """
        batch_data 结构（启用缓存）:
            caption, motions, m_lens, ..., raw_embeds
        关闭缓存则不带 raw_embeds
        """
        #nvtx.push_range("Forward Pass", color="blue") # <--- 开始标记 forward
        # ------------- 1. 拆包 -------------
        #nvtx.push_range("Data Unpacking", color="green")

        caption = batch_data[0]
        motions = batch_data[1].detach().float()
        m_lens  = batch_data[2]
        raw_embeds = batch_data[-1] if (self.opt.use_text_cache and
                                        torch.is_tensor(batch_data[-1])) else None
        #nvtx.pop_range()
        # ------------- DEBUG -------------
        # print(f"[DEBUG] Motions Shape: {motions.shape}, M_lens: {m_lens}, Raw Embeds Shape: {raw_embeds.shape if raw_embeds is not None else 'None'}")
        # ------------- 2. Diffusion -------------
        #nvtx.push_range("Diffusion Step", color="yellow")
        x_start = motions
        B, T = x_start.shape[:2]
        #cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
        cur_len = torch.as_tensor(m_lens, device=x_start.device).clamp_max(T).long()

        #self.src_mask = self.generate_src_mask(T, cur_len).to(x_start.device)
        self.src_mask = self.generate_src_mask(T, cur_len, device=x_start.device)
        real_noise = torch.randn_like(x_start)
        t = torch.randint(0, self.diffusion_steps, (B,), device=self.device)
        self.timesteps = t
        x_t = self.noise_scheduler.add_noise(x_start, real_noise, t)
        #nvtx.pop_range()
        # ------------- 3. 调用模型 -------------
        #nvtx.push_range("Model Execution", color="red")
        self.prediction = self.model(
            x_t,
            t,
            text=None if raw_embeds is not None else caption,
            raw_embeds=raw_embeds
        )
        #nvtx.pop_range()
        # ------------- 4. 目标 -------------
        if self.opt.prediction_type == 'sample':
            self.target = x_start
        elif self.opt.prediction_type == 'epsilon':
            self.target = real_noise
        elif self.opt.prediction_type == 'v_prediction':
            self.target = self.noise_scheduler.get_velocity(x_start, real_noise, t)
        #nvtx.pop_range()

    def masked_l2(self, a, b, mask, weights):
        loss = self.mse_criterion(a, b).mean(dim=-1)
        loss = (loss * mask).sum(-1) / mask.sum(-1)
        loss = (loss * weights).mean()
        return loss

    def backward_G(self):
        loss_logs = OrderedDict({})
        mse_loss_weights = torch.ones_like(self.timesteps)
        loss_logs['loss_mot_rec'] = self.masked_l2(self.prediction, self.target, self.src_mask, mse_loss_weights)
        self.loss = loss_logs['loss_mot_rec']
        return loss_logs

    def update(self):
        #nvtx.push_range("Update Step", color="purple") 
        self.zero_grad([self.optimizer])
        #nvtx.push_range("Loss Calculation", color="orange")
        loss_logs = self.backward_G()
        #nvtx.pop_range()
        #nvtx.push_range("Backward", color="red")
        self.accelerator.backward(self.loss)
        #nvtx.pop_range()
        #nvtx.push_range("Gradient Clipping", color="cyan")
        self.clip_norm([self.model])
        #nvtx.pop_range()
        #nvtx.push_range("Optimizer Step", color="magenta")
        self.step([self.optimizer])
        #nvtx.pop_range()
        #nvtx.pop_range()

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

    def save(self, file_name, total_it):
        full_state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        filtered_state_dict = {
            k: v for k, v in full_state_dict.items() if not k.startswith('text_encoder.')
        }
        state = {
            'opt_encoder': self.optimizer.state_dict(),
            'total_it': total_it,
            'encoder': filtered_state_dict,
        }
        if self.model_ema:
            full_ema_state_dict = self.accelerator.unwrap_model(self.model_ema).module.state_dict()
            filtered_ema_state_dict = {
                 k: v for k, v in full_ema_state_dict.items() if not k.startswith('text_encoder.')
            }
            state["model_ema"] = filtered_ema_state_dict
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['opt_encoder'])
        if self.model_ema:
            self.model_ema.load_state_dict(checkpoint["model_ema"], strict=False)
        self.model.load_state_dict(checkpoint['encoder'], strict=False)
        return checkpoint.get('total_it', 0)

    def evaluate(self, total_it, eval_wrapper, gt_loader, gen_dataset):
        """
        Performs evaluation on the validation set.
        在验证集上执行评估。
        """
        self.accelerator.print(f"--- Starting evaluation at iteration {total_it} ---")
        self.eval_mode()  # Switch to evaluation mode

        # Determine which model to use for evaluation (EMA or standard)
        # 确定用于评估的模型（EMA或标准模型）
        model_to_eval = self.accelerator.unwrap_model(self.model)
        use_ema = self.model_ema and not self.opt.no_ema
        if use_ema:
            self.accelerator.print("Using EMA model for evaluation.")
            ema_model = self.accelerator.unwrap_model(self.model_ema)
            model_to_eval = ema_model.module
        else:
            self.accelerator.print("Using standard model for evaluation.")

        # Update the evaluator wrapper with the current model state
        # 使用当前模型状态更新评估器包装器
        eval_wrapper.model = model_to_eval

        # Create a pipeline for generation in diffusion model framework
        # 为扩散模型框架中的生成创建一个 pipeline
        pipeline = DiffusePipeline(
            opt=self.opt,
            model=model_to_eval,
            diffuser_name=self.opt.diffuser_name,
            device=self.device,
            num_inference_steps=self.opt.num_inference_steps,
            torch_dtype=torch.float32
        )

        # Create motion loader for generation
        # 为生成创建运动加载器
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

        # Define log file path
        # 定义日志文件路径
        save_dir = pjoin(self.opt.save_root, 'eval')
        os.makedirs(save_dir, exist_ok=True)
        log_file = pjoin(save_dir, f'eval_log_it_{total_it}.log')

        # Run the evaluation
        # 运行评估
        all_metrics = evaluation_fid(
            eval_wrapper, gt_loader, eval_motion_loaders, log_file,
            2, self.opt.diversity_times, self.opt.mm_num_times, run_mm=True
        )

        # Log metrics to TensorBoard and WandB
        # 将指标记录到 TensorBoard 和 WandB
        self.accelerator.print(f"--- Evaluation at iteration {total_it} complete ---")
        self.train_mode()  # Switch back to training mode

    def train(self, train_loader):
        it = 0
        if self.opt.is_continue:
            model_path = pjoin(self.opt.model_dir, self.opt.continue_ckpt)
            it = self.load(model_path)
            self.accelerator.print(f'Continue training from {it} iterations in {model_path}')
        
        start_time = time.time()
        logs = OrderedDict()
        self.dataset = train_loader.dataset
        num_epochs = (self.opt.num_train_steps - it) // len(train_loader) + 1
        if self.accelerator.is_main_process:
            self.accelerator.print("Setting up evaluation components...")
            eval_split = 'val' 
            eval_wrapper = EvaluatorModelWrapper(self.opt)
            gt_loader = get_dataset_loader(self.opt, self.opt.batch_size, mode='gt_eval', split='test')
            gen_dataset = get_dataset(self.opt, mode='eval', split='test')
            self.accelerator.print("Evaluation components are ready.")

            # 1. 将损失函数移动到正确的设备，它不需要 'prepare'
            self.mse_criterion.to(self.device)

            # 2. 一起准备核心的训练组件 (模型, 优化器, 数据加载器)
            #    这是 DeepSpeed 初始化的主要步骤
            self.model, self.optimizer, train_loader = self.accelerator.prepare(
                self.model, self.optimizer, train_loader
            )

            # 3. 如果使用EMA，单独准备EMA模型
            if self.model_ema is not None:
                self.model_ema = self.accelerator.prepare(self.model_ema)

            self.accelerator.print(f'Need to train for {num_epochs} epochs....')
        #rng = nvtx.start_range(message="main training loop", color="blue")
        for epoch in range(0, num_epochs):
            self.train_mode()
            for i, batch_data in enumerate(train_loader):
                #nvtx.push_range(f"Iteration_{it}", color="gray")
                self.forward(batch_data)
                log_dict = self.update()
                it += 1
                #nvtx.push_range(f"Iteration_other_process", color="black")
                if self.model_ema and it % self.opt.model_ema_steps == 0:
                    self.accelerator.unwrap_model(self.model_ema).update_parameters(self.model)

                for k, v in log_dict.items():
                    value = v.item() if torch.is_tensor(v) else v
                    logs[k] = logs.get(k, 0) + value
                
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(self.accelerator, start_time, it, mean_loss, epoch, inner_iter=i)
                    if self.accelerator.is_main_process:
                        self.writer.add_scalar("loss/train", mean_loss['loss_mot_rec'], it)
                        #wandb.log({"loss/train": mean_loss['loss_mot_rec'], "epoch": epoch, "iters": it})
                    self.accelerator.wait_for_everyone()
                
                if it % self.opt.save_interval == 0 and self.accelerator.is_main_process:
                    self.accelerator.print(f"Saving checkpoint at iteration {it}...")
                    self.save(pjoin(self.opt.model_dir, f'latest_{it}.tar'), it)
                    self.evaluate(it, eval_wrapper, gt_loader, gen_dataset)
                
                self.accelerator.wait_for_everyone()

                if (self.scheduler is not None) and (it % self.opt.update_lr_steps == 0):
                    self.scheduler.step()
                #nvtx.pop_range()
                #nvtx.pop_range()
            #nvtx.end_range(rng)
        self.accelerator.print(f"Training completed after {it} iterations.")        
        if it % self.opt.save_interval != 0 and self.accelerator.is_main_process:
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), it)

        self.accelerator.wait_for_everyone()
        self.accelerator.print('FINISH')
