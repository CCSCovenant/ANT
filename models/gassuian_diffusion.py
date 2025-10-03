from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, DDIMScheduler, PNDMScheduler, DEISMultistepScheduler
import torch
import yaml
import math
import tqdm
import time
import os

class DiffusePipeline(object):
    """
    A pipeline for generating motion sequences using a diffusion model.
    It handles the setup of various schedulers and orchestrates the iterative denoising process.
    """
    def __init__(self, opt, model, diffuser_name, num_inference_steps, device, torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.opt = opt
        self.model = model.to(device)
        self.num_inference_steps = num_inference_steps
        if self.torch_dtype == torch.float16:
            self.model.half()

        # Load scheduler parameters from a YAML configuration file
        with open('config/diffuser_params.yaml', 'r') as yaml_file:
            diffuser_params = yaml.safe_load(yaml_file)

        # Select and initialize the scheduler based on the diffuser_name
        if diffuser_name in diffuser_params:
            params = diffuser_params[diffuser_name]
            scheduler_class_name = params['scheduler_class']
            additional_params = params['additional_params']
            
            # Align scheduler parameters with training configuration
            additional_params.update({
                'num_train_timesteps': opt.diffusion_steps,
                'beta_schedule': opt.beta_schedule,
                'prediction_type': opt.prediction_type,
            })
            
            try:
                scheduler_class = globals()[scheduler_class_name]
                self.scheduler = scheduler_class(**additional_params)
            except KeyError:
                raise ValueError(f"Scheduler class '{scheduler_class_name}' not found.")
        else:
            raise ValueError(f"Unsupported diffuser: {diffuser_name}")
        
        print(f"Pipeline initialized with scheduler: {self.scheduler.__class__.__name__}")

    @torch.no_grad()
    def generate_batch(self, caption, m_lens):
        """
        Generates a single batch of motions.
        """
        B = len(caption)
        T = m_lens.max().item()
        shape = (B, T, self.model.input_feats)

        # 1. Start with random noise
        sample = torch.randn(shape, device=self.device, dtype=self.torch_dtype)

        # 2. Set the number of inference steps
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        
        # 3. Pre-calculate text embeddings to avoid re-computation in the loop
        raw_embeds = self.model.encode_text(text=caption)
        # 4. Iteratively denoise the sample
        for t in self.scheduler.timesteps:
            timestep_batch = torch.full((B,), t, device=self.device, dtype=torch.long)

            # Use classifier-free guidance if enabled in the model
            if getattr(self.model, 'cond_mask_prob', 0) > 0:
                model_output = self.model.forward_with_cfg(sample, timestep_batch, text=caption, raw_embeds=raw_embeds,opt=self.opt)
                print(model_output.shape)
            else:
                model_output = self.model(sample, timestep_batch, text=caption)

            # 5. Compute the previous noisy sample using the scheduler
            sample = self.scheduler.step(model_output, t, sample).prev_sample

        return sample

    def generate(self, caption, m_lens, batch_size=32):
        """
        Generates motions for a list of captions, handling batching.
        """
        N = len(caption)
        self.model.eval()
        
        infer_mode = 'classifier-free-guidance' if getattr(self.model, 'cond_mask_prob', 0) > 0 else 'standard'
        print(f'\nUsing {self.scheduler.__class__.__name__} with {infer_mode} to generate {N} motions ({self.num_inference_steps} steps).')

        all_outputs = []
        timed_batches = 0
        total_time = 0.0

        for i in tqdm.tqdm(range(0, N, batch_size)):
            batch_caption = caption[i : i + batch_size]
            batch_m_lens = m_lens[i : i + batch_size]
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            output = self.generate_batch(batch_caption, batch_m_lens)
            
            torch.cuda.synchronize()
            end_time = time.time()

            # Skip the first few batches for warm-up before timing
            if i // batch_size > 2:
                total_time += (end_time - start_time)
                timed_batches += 1

            # Crop generated motions to their specified lengths
            for j in range(len(batch_caption)):
                all_outputs.append(output[j, :batch_m_lens[j]])
        
        # Calculate and report average inference time
        if timed_batches > 0:
            avg_time_per_batch = total_time / timed_batches
            print(f'Average generation time per batch (bs={batch_size}): {avg_time_per_batch:.4f} seconds')
        else:
            avg_time_per_batch = -1.0
            print("Not enough batches to calculate a stable average generation time.")

        return all_outputs, avg_time_per_batch