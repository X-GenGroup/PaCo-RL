# scripts/train_qwenimage_edit.py
from argparse import Namespace
import contextlib
import datetime
import hashlib
import json
import math
import os
import random
import signal
import sys
import tempfile
import time
import torch
import tqdm as tqdm_
from typing import List, Tuple, Any, Optional
import shutil
from itertools import permutations, combinations, product

from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from collections import defaultdict, Counter
from concurrent import futures
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from functools import partial
from ml_collections import config_flags
import numpy as np
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

from paco_grpo.logging_utils import set_online_log
from paco_grpo.utils import tensor_list_to_pil_image, tensor_to_pil_image, all_gather_tensor_list, create_generator, pil_image_to_tensor, hash_pil_image
from paco_grpo.rewards.rewards import multi_score
from paco_grpo.diffusers_patch.qwenimage_edit_pipeline import qwenimage_edit_pipeline, compute_log_prob
from paco_grpo.ema import EMAModuleWrapper
from paco_grpo.stat_tracking import PerPromptStatTracker
from paco_grpo.data_utils.sampler import DistributedKRepeatSampler
from paco_grpo.data_utils.prompt_dataset import GenevalPromptImageDataset, ArrowPromptImageDataset
from paco_grpo.scheduler import FlowMatchNoiseScheduler
from paco_grpo.memory_tracker import MemoryProfiler

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def reward_compute(
    logging_platform,
    accelerator : Accelerator,
    pipeline : QwenImageEditPipeline,
    config : Namespace,
    samples : List[dict],
    reward_fn,
    executor : futures.ThreadPoolExecutor,
    max_log_num : int = 30,
    step : int = 0
):  
    log_items = []
    # Compute reward for each sample
    for i in tqdm(
        range(0, len(samples), config.train.batch_size),
        desc="Computing rewards",
        disable=not accelerator.is_local_main_process
    ):
        # Compute reward with train.batch_size to avoid OOM
        batch = samples[i : i + config.train.batch_size]
        heights = [sample.get('height', config.train.resolution) for sample in batch]
        widths = [sample.get('width', config.train.resolution) for sample in batch]
        ref_images = [sample['ref_image'] for sample in batch]
        images = [sample['image'] for sample in batch]  
        # Compute reward
        prompts = [sample['prompt'] for sample in batch]
        prompt_metadatas = [sample.get('metadata', {}) for sample in batch]
        future = executor.submit(reward_fn, images, prompts, prompt_metadatas, ref_images)
        rewards, reward_metadatas = future.result()

        # Convert rewards from dict of list to list of dict
        rewards = [
            dict(zip(rewards.keys(), value))
            for value in zip(*rewards.values())
        ]

        for sample, reward in zip(batch, rewards):
            sample['rewards'] = reward
        
        if accelerator.is_main_process and len(log_items) < max_log_num:
            log_items.extend(list(zip(ref_images, images, prompts, rewards)))

    if accelerator.is_main_process:
        # Catenate ref_image and edited image
        data = []
        for ref_img, edited_img, prompt, reward in log_items:
            # Create a new image with width = sum of both widths, height = max of both heights
            total_width = ref_img.width + edited_img.width
            max_height = max(ref_img.height, edited_img.height)
            new_img = Image.new('RGB', (total_width, max_height))
            new_img.paste(ref_img, (0, 0))
            new_img.paste(edited_img, (ref_img.width, 0))
            data.append((new_img, prompt, reward))
        
        logging_platform.log(
            {
                "train_samples": [
                    logging_platform.Image(
                        image,
                        caption=", ".join(f"{k}: {v:.2f}" for k, v in reward.items()) + f" | {prompt}",
                    )
                    for image, prompt, reward in data
                ]
            },
            step=step
        )

    return samples


def compute_ppo_loss(
    config : Namespace,
    accelerator : Accelerator,
    pipeline: QwenImageEditPipeline,
    transformer: QwenImageTransformer2DModel,
    sample : dict,
    timestep_index : int,
    autocast,
):
    info = {}
    batch_size = sample['all_latents'].shape[0]
    with autocast():
        transformer.module.set_adapter("default")
        prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
            transformer=transformer,
            pipeline=pipeline,
            sample=sample,
            timestep_index=timestep_index,
            config=config,
        )
        with torch.no_grad():
            transformer.module.set_adapter("old")
            _, old_log_prob, old_prev_sample_mean, _ = compute_log_prob(
                transformer=transformer,
                pipeline=pipeline,
                sample=sample,
                timestep_index=timestep_index,
                config=config,
            )
            with transformer.module.disable_adapter():
                _, ref_log_prob, ref_prev_sample_mean, _ = compute_log_prob(
                    transformer=transformer,
                    pipeline=pipeline,
                    sample=sample,
                    timestep_index=timestep_index,
                    config=config,
                )

    transformer.module.set_adapter("default")
    # grpo logic
    advantages = torch.clamp(
        sample["advantages"],
        -config.train.adv_clip_max,
        config.train.adv_clip_max,
    )

    ratio = torch.exp(log_prob - old_log_prob)
    # print("ratio", ratio)
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(
        ratio,
        1.0 - config.train.clip_range,
        1.0 + config.train.clip_range,
    )
    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

    kl_loss = ((prev_sample_mean - ref_prev_sample_mean) ** 2).mean(dim=tuple(range(1, prev_sample_mean.ndim)), keepdim=True) / (2 * std_dev_t ** 2 + 1e-7)
    kl_loss = torch.mean(kl_loss)
    
    loss = policy_loss + config.train.beta * kl_loss
    info["policy_loss"] = policy_loss.detach()
    info["unclipped_loss"] = unclipped_loss.mean().detach()
    info["clipped_loss"] = clipped_loss.mean().detach()
    info["kl_loss"] = kl_loss.mean().detach()
    info['loss'] = loss.detach()
    info['ratio'] = ratio.abs().mean()
    info["clipfrac"] = torch.mean(
        (
            torch.abs(ratio - 1.0) > config.train.clip_range
        ).float()
    )
    info["clipfrac_gt_one"] = torch.mean(
        (
            ratio - 1.0 > config.train.clip_range
        ).float()
    )
    info["clipfrac_lt_one"] = torch.mean(
        (
            1.0 - ratio > config.train.clip_range
        ).float()
    )
    info["clipfrac_lt_one"] = torch.mean(
        (
            1.0 - ratio > config.train.clip_range
        ).float()
    )

    return loss, info

def compute_guard_grpo_loss(
    config : Namespace,
    accelerator : Accelerator,
    pipeline: QwenImageEditPipeline,
    transformer: QwenImageTransformer2DModel,
    sample : dict,
    timestep_index : int,
    autocast,
):
    info = {}
    batch_size = sample['all_latents'].shape[0]
    with autocast():
        transformer.module.set_adapter("default")
        prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
            transformer=transformer,
            pipeline=pipeline,
            sample=sample,
            timestep_index=timestep_index,
            config=config,
        )
        with torch.no_grad():
            transformer.module.set_adapter("old")
            _, old_log_prob, old_prev_sample_mean, _ = compute_log_prob(
                transformer=transformer,
                pipeline=pipeline,
                sample=sample,
                timestep_index=timestep_index,
                config=config,
            )
            with transformer.module.disable_adapter():
                _, ref_log_prob, ref_prev_sample_mean, _ = compute_log_prob(
                    transformer=transformer,
                    pipeline=pipeline,
                    sample=sample,
                    timestep_index=timestep_index,
                    config=config,
                )

    transformer.module.set_adapter("default")


    # Get dt
    timestep = sample['timesteps'][:, timestep_index].to(accelerator.device)
    timestep_next = sample["timesteps"][:, timestep_index + 1] if timestep_index + 1 < sample["timesteps"].shape[1] else torch.zeros_like(timestep) # (B,)
    dt = (timestep_next - timestep).view(-1, *([1] * (prev_sample.ndim -1))) # (B, 1, 1, 1)
    dt = dt / 1000.0 # scale to [0, 1] for flux
    sqrt_dt = torch.sqrt(-dt)

    # Guard-GRPO logic
    advantages = torch.clamp(
        sample["advantages"],
        -config.train.adv_clip_max,
        config.train.adv_clip_max,
    )

    sigma_t = std_dev_t.mean()
    ratio_mean_bias = (prev_sample_mean - old_prev_sample_mean).pow(2).mean(dim=tuple(range(1, log_prob.ndim)))
    ratio_mean_bias = ratio_mean_bias / (2 * (sqrt_dt.mean() * sigma_t) ** 2)
    ratio = torch.exp((log_prob - old_log_prob + ratio_mean_bias) * (sqrt_dt.mean() * sigma_t))

    # print("ratio", ratio)
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(
        ratio,
        1.0 - config.train.clip_range,
        1.0 + config.train.clip_range,
    )
    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

    # Guard-GRPO normalization
    policy_loss = policy_loss / (sqrt_dt.mean()**2)

    kl_loss = ((prev_sample_mean - ref_prev_sample_mean) ** 2).mean(dim=tuple(range(1, prev_sample_mean.ndim)), keepdim=True) / (2 * std_dev_t ** 2 + 1e-7)
    kl_loss = torch.mean(kl_loss)
    
    loss = policy_loss + config.train.beta * kl_loss
    info["policy_loss"] = policy_loss.detach()
    info["unclipped_loss"] = unclipped_loss.mean().detach()
    info["clipped_loss"] = clipped_loss.mean().detach()
    info["kl_loss"] = kl_loss.mean().detach()
    info['loss'] = loss.detach()
    info['ratio'] = ratio.abs().mean()
    info["clipfrac"] = torch.mean(
        (
            torch.abs(ratio - 1.0) > config.train.clip_range
        ).float()
    )
    info["clipfrac_gt_one"] = torch.mean(
        (
            ratio - 1.0 > config.train.clip_range
        ).float()
    )
    info["clipfrac_lt_one"] = torch.mean(
        (
            1.0 - ratio > config.train.clip_range
        ).float()
    )
    info["clipfrac_lt_one"] = torch.mean(
        (
            1.0 - ratio > config.train.clip_range
        ).float()
    )

    return loss, info

@torch.no_grad()
def eval(pipeline : QwenImageEditPipeline,
         test_dataloader : DataLoader,
         config : Namespace,
         accelerator,
         logging_platform,
         global_step,
         reward_fn,
         executor,
         autocast,
         ema,
         transformer_trainable_parameters,
         memory_profiler : Optional[MemoryProfiler] = None,
         log_sample_num : int = 108 # 108 as max in wandb/swanlab
    ):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    
    log_data = {
        'ref_images': [],
        'images': [],
        'prompts': [],
        'rewards': defaultdict(list)
    }
    if memory_profiler is not None:
        memory_profiler.snapshot("before_eval")

    for batch_idx, test_batch in enumerate(tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        )):
        if memory_profiler is not None:
            memory_profiler.snapshot(f"eval_batch_{batch_idx}_start")

        prompts, prompt_metadata, ref_images = test_batch
        generator = create_generator(prompts, config.seed + accelerator.process_index)

        heights = [prompt_meta.get('height', config.test.resolution) for prompt_meta in prompt_metadata]
        widths = [prompt_meta.get('width', config.test.resolution) for prompt_meta in prompt_metadata]
        
        if not all(h == heights[0] for h in heights) or not all(w == widths[0] for w in widths):
            # Split the batch if there are different sizes
            images = []
            for i in tqdm(
                range(len(prompts)),
                desc="Eval: per sample",
                leave=False,
                position=1,
                disable=not accelerator.is_local_main_process,
            ):
                prompt = [prompts[i]]
                prompt_meta = [prompt_metadata[i]]
                ref_image = [ref_images[i]]
                with autocast():
                    imgs, _, _, _ = qwenimage_edit_pipeline(
                        pipeline,
                        image=ref_image,
                        prompt=prompt,
                        num_inference_steps=config.test.num_steps,
                        true_cfg_scale=config.sample.guidance_scale,
                        generator=generator[i],
                        output_type="pt",
                        height=heights[i],
                        width=widths[i],
                        noise_level=0,
                        max_area=config.test.resolution * config.test.resolution
                    )

                images.append(imgs.squeeze(0))  # (C, H, W)
        else:
            # Batch inference if all sizes are the same
            with autocast():
                images, _, _, _ = qwenimage_edit_pipeline(
                    pipeline,
                    image=ref_images,
                    prompt=prompts,
                    num_inference_steps=config.test.num_steps,
                    true_cfg_scale=config.sample.guidance_scale,
                    generator=generator,
                    output_type="pt",
                    height=heights[0],
                    width=widths[0],
                    noise_level=0,
                    max_area=config.test.resolution * config.test.resolution
                )
                images = list(images.unbind(0)) # List[torch.Tensor(C, H, W)]
        
        # reward_fn accepts torch.Tensor (B, C, H, W) or List[torch.Tensor(C, H, W)]
        future = executor.submit(reward_fn, images, prompts, prompt_metadata, ref_images)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = future.result()

        # ---------------------------------Collect log data--------------------------------
        for i, prompt in enumerate(prompts):
            log_data['ref_images'].append(ref_images[i])
            log_data['images'].append(images[i].cpu())
            log_data['prompts'].append(prompt)
            for key, value in rewards.items():
                if key not in log_data['rewards']:
                    log_data['rewards'][key] = []
                
                log_data['rewards'][key].append(value[i])
        
        # log memory after reward computation
        if memory_profiler is not None:
            memory_profiler.snapshot(f"eval_batch_{batch_idx}_end")

    if memory_profiler is not None:
        memory_profiler.snapshot("after_eval_before_gather_log_data")
    # ---------------------------Gather all Log data, with prompt-image-reward tuples--------------------------
    # 1. Gather all rewards and report average
    gathered_rewards = {}
    for key, value in log_data['rewards'].items():
        gathered_rewards[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    if accelerator.is_main_process:
        # Report detailed rewards values
        for key, value in gathered_rewards.items():
            print(key, np.mean(value))

        # Log eval metrics
        logging_platform.log(
            {
                **{f"eval/{key}": np.mean(value) for key, value in gathered_rewards.items()},
                **{f"eval/{key}_std": np.std(value) for key, value in gathered_rewards.items()},
            },
            step=global_step
        )

    # gathered_rewards = {'r1': [1,2,3], 'r2': [4,5,6]}
    # ->
    # gathered_rewards = [{'r1':1, 'r2':4}, {'r1':2, 'r2':5}, {'r1':3, 'r2':6}]
    gathered_rewards = [
        dict(zip(gathered_rewards.keys(), value))
        for value in zip(*gathered_rewards.values())
    ]

    if memory_profiler is not None:
        memory_profiler.snapshot("after_gather_rewards")

    # 2. Encode prompt to tensors for gpu communication
    prompt_ids = pipeline.tokenizer(
        log_data['prompts'],
        padding="max_length",
        max_length=config.max_sequence_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    gathered_prompt_ids = accelerator.gather(prompt_ids).cpu().numpy()
    gathered_prompts = pipeline.tokenizer.batch_decode(
        gathered_prompt_ids, skip_special_tokens=True
    )

    if memory_profiler is not None:
        memory_profiler.snapshot("after_gather_prompts")

    # 3. Gather all images
    # Approach : by saving them in a temp dir
    # This approach saves images as JPG files in a temporary directory
    # Since uploading images with jpg is faster, if we need to do it anyway.
    temp_dir = os.path.join(config.save_dir, 'eval_images', str(global_step))
    os.makedirs(temp_dir, exist_ok=True)
    for idx, img in enumerate(log_data['images']):
        # Save image to temp dir
        pil_img = tensor_to_pil_image(img)[0]
        pil_img.save(os.path.join(temp_dir, f"{accelerator.process_index}-{idx}.jpg"))
    for idx, pil_img in enumerate(log_data['ref_images']):
        # Save ref image to temp dir
        pil_img.save(os.path.join(temp_dir, f"ref-{accelerator.process_index}-{idx}.jpg"))
    accelerator.wait_for_everyone()
    # The order of images here should be guaranteed by the name of images
    # NOTE: it provides gathered_images as a list of file paths
    def sort_key(filename):
        key = []
        if filename.startswith('ref-'):
            key.append(0) # Add a prefix to make ref images come first
            filename = filename[4:]  # remove 'ref-' prefix
        key.extend([int(i) for i in filename.split('.')[0].split('-')])
        return tuple(key)
    
    gathered_images = [
        os.path.join(temp_dir, filename)
        for filename in sorted(os.listdir(temp_dir), key=sort_key)
        if not filename.startswith('ref-')
    ]
    gathered_ref_images = [
        os.path.join(temp_dir, filename)
        for filename in sorted(os.listdir(temp_dir), key=sort_key)
        if filename.startswith('ref-')
    ]

    if memory_profiler is not None:
        memory_profiler.snapshot("after_gather_images")

    # 4. Log images
    if accelerator.is_main_process:
        # 'Deterministically' sample 'random' `log_sample_num` data for logging
        # Make `num_processes` divides `log_sample_num` to make sure each process has same amount of data to log
        log_sample_num = int(math.ceil(log_sample_num / accelerator.num_processes) * accelerator.num_processes)
        generator = torch.Generator().manual_seed(config.seed + accelerator.process_index)
        sample_indices = torch.randperm(len(gathered_images), generator=generator)[:log_sample_num].tolist()
        gathered_images = [gathered_images[i] for i in sample_indices]
        gathered_ref_images = [gathered_ref_images[i] for i in sample_indices]
        gathered_prompts = [gathered_prompts[i] for i in sample_indices]
        gathered_rewards = [gathered_rewards[i] for i in sample_indices]
        # Catenate image and ref image
        images = []
        for ref_img, img in zip(gathered_ref_images, gathered_images):
            ref_img = Image.open(ref_img).convert("RGB")
            edited_img = Image.open(img).convert("RGB")
            # Create a new image with width = sum of both widths, height = ref image height
            # Resize the edited image to the same height as ref image for better visualization
            target_height = ref_img.height
            edited_img = edited_img.resize((int(edited_img.width * target_height / edited_img.height), target_height))
            total_width = ref_img.width + edited_img.width
            new_img = Image.new('RGB', (total_width, target_height))
            new_img.paste(ref_img, (0, 0))
            new_img.paste(edited_img, (ref_img.width, 0))
            images.append(new_img)
            
        logging_platform.log(
            {
                "eval_images": [
                    logging_platform.Image(
                        image,
                        caption=", ".join(f"{k}: {v:.2f}" for k, v in reward.items()) + f" | {prompt}",
                    )
                    for image, prompt, reward in zip(images, gathered_prompts, gathered_rewards)
                ]
            },
            step=global_step
        )
        # Clean up temp dir
        shutil.rmtree(temp_dir)
    
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def load_pipeline(config : Namespace, accelerator : Accelerator):
    """
    Load QwenImageEditPipeline with FlowMatchNoiseScheduler for controlled noise injection.
    
    This function:
    1. Loads the QwenImageEditPipeline from pretrained weights
    2. Replaces the default scheduler with FlowMatchNoiseScheduler
    3. Configures LoRA adapters if specified
    4. Sets up proper device placement and dtype
    
    Args:
        config: Configuration namespace containing model paths and training settings
        accelerator: Accelerate accelerator for distributed training
        
    Returns:
        pipeline
    """
    # -------------------------------Load models-----------------------------------
    # load scheduler, tokenizer and models.
    pipeline = QwenImageEditPipeline.from_pretrained(
        config.pretrained.model,
        low_cpu_mem_usage=True
    )

    if hasattr(config.sample, 'noise_steps') and config.sample.noise_steps is not None:
        noise_steps = config.sample.noise_steps
    else:
        noise_steps = list(range(config.sample.num_steps)) # Default to all steps
    
    if hasattr(config.sample, 'noise_level') and config.sample.noise_level is not None:
        noise_level = config.sample.noise_level
    else:
        noise_level = 0.3 # Default to 0.3

    if hasattr(config.sample, 'num_noise_steps'):
        num_noise_steps = config.sample.num_noise_steps
        

    scheduler = FlowMatchNoiseScheduler(
        noise_level=noise_level,
        noise_steps=noise_steps,
        num_noise_steps=num_noise_steps,
        seed=config.seed,
        **pipeline.scheduler.config.__dict__,
    )

    # Overwrite the original scheduler
    pipeline.scheduler = scheduler

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    # Note: VAE is kept in float32 for numerical stability
    pipeline.vae.to(accelerator.device, dtype=inference_dtype if config.use_lora else torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device, dtype=inference_dtype if config.use_lora else torch.float32)

    if config.use_lora:
        # Set correct lora layers for Flux Kontext
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. 
            # You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

        # Add "old" adapter for PPO-style training
        pipeline.transformer.add_adapter("old", transformer_lora_config)
        pipeline.transformer.set_adapter("default")

    if config.enable_gradient_checkpointing:
        pipeline.transformer.enable_gradient_checkpointing() # save memory

    return pipeline


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    # Flexible training only supports batch size 1, so
    # update gradient_accumulation_steps, and update train.batch_size to 1 later for logger info
    if config.enable_flexible_size:
        gradient_accumulation_steps = config.train.gradient_accumulation_steps * config.train.batch_size
    else:
        gradient_accumulation_steps = config.train.gradient_accumulation_steps

    if config.train.timesteps is None:
        # Default to all timesteps
        config.train.timesteps = list(range(config.sample.num_steps))

    train_timestep_indices = [int(i) for i in config.train.timesteps if i < config.sample.num_steps] # filter out invalid indices

    num_train_timesteps = len(train_timestep_indices)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=gradient_accumulation_steps * num_train_timesteps,
    )
    
    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    if config.enable_flexible_size and config.train.batch_size != 1:
        # Print a warning message and override config
        logger.info(
            "Only batch size 1 is supported for flexible size training: "
            f"Overriding config.train.gradient_accumulation_steps by multiplying it with config.train.batch_size {config.train.gradient_accumulation_steps}*{config.train.batch_size}={gradient_accumulation_steps}"
            f" and setting config.train.batch_size to 1")
        
        config.train.batch_size = 1
        config.train.gradient_accumulation_steps = gradient_accumulation_steps
    else:
        # config.train.batch_size should divide config.sample.batch_size * config.num_batches_per_epoch
        assert (config.sample.batch_size * config.sample.unique_sample_num_per_epoch) % config.train.batch_size == 0, \
            f"config.train.batch_size {config.train.batch_size} should divide config.sample.batch_size {config.sample.batch_size} * config.sample.unique_sample_num_per_epoch {config.sample.unique_sample_num_per_epoch}"

    if not config.project_name:
        config.project_name = 'PaCoGRPO-QwenImageEdit'

    run, logging_platform = set_online_log(accelerator, config)

    def safe_exit(sig, frame):
        print("Received signal to terminate.")
        if accelerator.is_main_process:
            logging_platform.finish()
        
        sys.exit(0)

    signal.signal(signal.SIGINT, safe_exit)
    signal.signal(signal.SIGTERM, safe_exit)
    
    logger.info(f"\n{config}")

    # -----------------------------------------------Set up memory profiler-----------------------------------
    memory_profiler = None
    if config.enable_mem_log:
        # Initialize memory profiler
        time_stamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        meme_log_file = f'memory_{time_stamp}.log'
        # clean up old log file
        if accelerator.is_main_process and os.path.exists(meme_log_file):
            os.remove(meme_log_file)
        memory_profiler = MemoryProfiler(accelerator, enable_tensor_accumulation=True, log_file=meme_log_file)

    # --------------------------------------Load pipeline----------------------------------
    pipeline = load_pipeline(config, accelerator)
    transformer = pipeline.transformer

    if memory_profiler is not None:
        # Register model to profiler
        memory_profiler.register_model(transformer, "transformer")
        memory_profiler.snapshot("after_model_loading")
    
    # Setup multiple adapters for PPO
    transformer.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer.set_adapter("default")

    for src_param, tgt_param in zip(
        transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
    ):
        tgt_param.data.copy_(src_param.detach().data)
        assert src_param is not tgt_param

    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    if memory_profiler is not None:
        memory_profiler.track_optimizer(optimizer)
        memory_profiler.snapshot("after_optimizer_init")

    if config.prompt_fn == 'general_editing':
        dataset_cls = GenevalPromptImageDataset
    elif config.prompt_fn == 'arrow_editing':
        dataset_cls = ArrowPromptImageDataset
    else:
        raise NotImplementedError("Specify `prompt_fn` in ['general_editing', 'arrow_editing']")

    train_dataset = dataset_cls(config.dataset, 'train')
    test_dataset = dataset_cls(config.dataset, 'test')

    train_sampler = DistributedKRepeatSampler( 
        dataset=train_dataset,
        batch_size=config.sample.batch_size,
        k=config.sample.num_images_per_prompt,
        m=config.sample.unique_sample_num_per_epoch,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=config.seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=dataset_cls.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.test.batch_size,
        collate_fn=dataset_cls.collate_fn,
        shuffle=False,
        num_workers=8,
    )

    if config.sample.num_images_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # autocast = accelerator.autocast
    autocast = partial(torch.autocast, device_type=accelerator.device.type, dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16)
    # autocast = contextlib.nullcontext

    # for deepspeed zero
    if accelerator.state.deepspeed_plugin:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.sample.batch_size
    
    # Prepare everything with our `accelerator`.
    transformer, optimizer, test_dataloader = accelerator.prepare(transformer, optimizer, test_dataloader)
    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    if memory_profiler is not None:
        memory_profiler.snapshot("after_accelerator_prepare")

    # -----------------------------------------Reward fn-----------------------------------------
    # prepare prompt and reward fn
    if accelerator.is_main_process:
        print(f"Train reward dict: {config.train.reward_fn}")
        print(f"Eval reward dict: {config.test.reward_fn}")
    train_reward_fn_kwargs = {
        k: {
            'device': accelerator.device,
            **(config.train.reward_fn_kwargs.get(k, {}))
        }
        for k in config.train.reward_fn.keys()
    }
    test_reward_fn_kwargs = {
        k: {
            'device': accelerator.device,
            **(config.test.reward_fn_kwargs.get(k, {}))
        }
        for k in config.test.reward_fn.keys()
    }
    if accelerator.is_main_process:
        print(f"Training reward fn kwargs: {train_reward_fn_kwargs}")
        print(f"Test reward fn kwargs: {test_reward_fn_kwargs}")
    train_reward_fn = multi_score(config.train.reward_fn, config.train.aggregate_fn, **train_reward_fn_kwargs)
    test_reward_fn = multi_score(config.test.reward_fn, config.test.aggregate_fn, **test_reward_fn_kwargs)

    if memory_profiler is not None:
        memory_profiler.snapshot("after_loading_reward_fn")

    # Train!
    samples_per_epoch = (
        config.sample.batch_size
        * accelerator.num_processes
        * train_sampler.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    logger.info(f"  Training timesteps: {train_timestep_indices}")

    if config.resume_from_id:
        global_step = config.resume_from_step
        epoch = global_step // 2
    else:
        global_step = 0
        epoch = 0

    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if config.eval_freq > 0 and epoch % config.eval_freq == 0:
            if memory_profiler is not None:
                memory_profiler.snapshot(f"epoch_{epoch}_before_eval")
            eval(
                pipeline,
                test_dataloader,
                config,
                accelerator,
                logging_platform,
                global_step,
                test_reward_fn,
                executor,
                autocast,
                ema,
                transformer_trainable_parameters,
                memory_profiler=memory_profiler,
            )
            if memory_profiler is not None:
                memory_profiler.snapshot(f"epoch_{epoch}_after_eval")
    
        if config.save_freq > 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        train_sampler.set_epoch(epoch)
        train_iter = iter(train_dataloader)
        pipeline.scheduler.set_seed(config.seed + epoch)

        # Use old policy to sample
        transformer.set_adapter("old")
        samples = []
        for i in tqdm(
            range(train_sampler.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            prompts, prompt_metadata, ref_images = next(train_iter)
            ref_images = [ref_image.resize((config.train.resolution, config.train.resolution)) for ref_image in ref_images]

            # the input of edit task is determined by both the image and the edit prompt
            prompt_ids = pipeline.tokenizer(
                prompts,
                padding="max_length",
                max_length=config.max_sequence_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # Get heights and widths
            heights = [prompt_meta.get('height', config.train.resolution) for prompt_meta in prompt_metadata]
            widths = [prompt_meta.get('width', config.train.resolution) for prompt_meta in prompt_metadata]

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch)
            else:
                generator = None
            
            # If all heights and widths are the same, we can batch them together
            if all(h == heights[0] for h in heights) and all(w == widths[0] for w in widths):
                with autocast():
                    with torch.no_grad():
                        (images, all_latents, all_timesteps, image_latents) = qwenimage_edit_pipeline(
                            pipeline,
                            image=ref_images,
                            prompt=prompts,
                            num_inference_steps=config.sample.num_steps,
                            true_cfg_scale=config.sample.guidance_scale,
                            output_type="pil",
                            height=heights[0],
                            width=widths[0],
                            max_area=config.train.resolution * config.train.resolution,
                            generator=generator,
                            cps=config.sample.cps
                        )
                    images = list(images) # List[PIL.Image] with length batch_size
                    all_latents = torch.stack(all_latents, dim=1) # (batch_size, num_steps + 1, seq_len, C)
                    all_latents = list(all_latents.unbind(0)) # List[Tensor(num_steps + 1, seq_len, C)] with length batch_size
                    all_timesteps = list(all_timesteps.unbind(0)) # List[Tensor(num_steps)] with length batch_size
                    image_latents = list(image_latents.unbind(0)) # List[Tensor(image_seq_len, C)] with length batch_size
            else:
                # Different sizes, have to do one by one
                images = []
                all_latents = []
                all_timesteps = []
                image_latents = []
                for index in range(len(prompts)):
                    with autocast():
                        with torch.no_grad():
                            (this_image, this_all_latents, this_timesteps, this_image_latents) = qwenimage_edit_pipeline(
                                pipeline,
                                image=[ref_images[index]],
                                prompt=[prompts[index]],
                                num_inference_steps=config.sample.num_steps,
                                true_cfg_scale=config.sample.guidance_scale,
                                output_type="pil",
                                height=heights[index],
                                width=widths[index],
                                max_area=config.train.resolution * config.train.resolution,
                                generator=generator[index] if generator is not None else None,
                                cps=config.sample.cps
                            )
                    images.append(this_image[0])  # add PIL.Image
                    all_latents.append(torch.stack(this_all_latents, dim=1).squeeze(0))  # add (num_steps + 1, seq_len, C)
                    all_timesteps.append(this_timesteps.squeeze(0)) # (1, num_steps) -> (num_steps,)
                    image_latents.append(this_image_latents.squeeze(0)) # (1, image_seq_len, C) -> (image_seq_len, C)

            # Final `samples` is List[Dict], with length = config.sample.batch_size * train_sampler.num_batches_per_epoch
            samples.extend(
                [
                    {
                        'height': heights[index],
                        'width': widths[index],
                        'prompt': prompts[index],
                        'metadata': prompt_metadata[index],
                        'ref_image': ref_images[index],
                        'timesteps': all_timesteps[index].unsqueeze(0), # Keep batch dimension as 1
                        'prompt_ids': prompt_ids[index].unsqueeze(0), # Keep batch dimension as 1
                        'all_latents': all_latents[index].unsqueeze(0), # Keep batch dimension as 1, shape (1, num_steps + 1, seq_len, C)
                        'image_latents': image_latents[index].unsqueeze(0), # Keep batch dimension as 1, shape (1, image_seq_len, C)
                        'image': images[index],
                    }
                    for index in range(len(prompts))
                ]
            )

            if memory_profiler is not None:
                memory_profiler.track_samples(samples, f"sampling")
                memory_profiler.snapshot(f"epoch_{epoch}_after_sampling_batch_{i}")

        transformer.set_adapter("default")

        # Compute reward for samples
        samples = reward_compute(
            logging_platform,
            accelerator,
            pipeline,
            config,
            samples,
            train_reward_fn,
            executor,
            max_log_num=30,
            step=global_step
        )

        # Gather rewards across all samples
        gathered_rewards = {
            key: torch.as_tensor([sample['rewards'][key] for sample in samples], device=accelerator.device)
            for key in samples[0]['rewards'].keys()
        }
        # Gather rewards across processes
        gathered_rewards = {
            key: accelerator.gather(value).cpu().numpy()
            for key, value in gathered_rewards.items()
        }

        # log rewards and images
        if accelerator.is_main_process:
            print(f"Epoch {epoch} rewards: ")
            for key, value in gathered_rewards.items():
                print(f"  {key}: {value.mean():.4f} ± {value.std():.4f}")
            logging_platform.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items()},
                    **{f"reward_{key}_std": value.std() for key, value in gathered_rewards.items()},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = torch.cat([s["prompt_ids"] for s in samples], dim=0)
            prompt_ids = accelerator.gather(prompt_ids).cpu().numpy()
            # Convert ref_image to tensor and gather
            local_ref_image_tensor = pil_image_to_tensor([s["ref_image"] for s in samples]).to(accelerator.device)
            gathered_ref_image_tensor = accelerator.gather(local_ref_image_tensor).cpu()
            # Reconstruct prompts with ref images in metadata to ensure uniqueness
            prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            gathered_ref_images = tensor_to_pil_image(gathered_ref_image_tensor)
            gathered_ref_images = [hash_pil_image(img) for img in gathered_ref_images] # Hash ref images
            # Construct unique prompt keys
            prompts = [
                f"{prompt}_{img_hash}"
                for prompt, img_hash in zip(prompts, gathered_ref_images)
            ]
            # Compute advantages with for GRPO
            advantages = stat_tracker.compute_advantages(
                prompts=prompts,
                rewards=gathered_rewards,  # Dict[str, np.ndarray]
                type='grpo',
                reward_weights=config.train.reward_fn,
                aggregate_fn=config.train.aggregate_fn,
            )
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

                (
                    avg_group_size,
                    trained_prompt_num,
                    avg_group_std,
                    global_std,
                    zero_std_ratio
                ) = stat_tracker.get_stats()

                if accelerator.is_main_process:
                    logging_platform.log(
                        {
                            "avg_group_size": avg_group_size,
                            "trained_prompt_num": trained_prompt_num,
                            "avg_group_std": avg_group_std,
                            "global_std": global_std,
                            "zero_std_ratio": zero_std_ratio,
                        },
                        step=global_step,
                    )
            # !!! Notice here, after every advantage calculation, the tracker is cleared so that no history is saved.
            # So comment the following clear code if `config.sample.use_history=True` is set
            stat_tracker.clear()
        else:
            # Compute advantages directly
            rewards_array = np.stack(
                [gathered_rewards[key] for key in config.train.reward_fn.keys()],
                axis=1
            )  # shape (num_samples, num_reward_types)
            reward_weights = np.array(
                [config.train.reward_fn[key] for key in config.train.reward_fn.keys()]
            )  # shape (num_reward_types,)

            # Weighted sum of rewards
            weighted_rewards = rewards_array @ reward_weights  # shape (num_samples,)

            # Normalize advantages
            advantages = (weighted_rewards - weighted_rewards.mean())
            advantages /= max(1e-6, advantages.std())

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        advantages = (
            advantages.reshape(accelerator.num_processes, -1, *advantages.shape[1:])[accelerator.process_index]
            .to(accelerator.device)
        )
        # Distribute advantages to samples
        for i, sample in enumerate(samples):
            sample['advantages'] = advantages[i].unsqueeze(0) # keep batch dimension

        if accelerator.is_local_main_process:
            print("len samples", len(samples))
            print("advantages has shape", advantages.shape)
            print("advantages: ", advantages.abs().mean())

        # clean up to save memory
        del gathered_rewards
        for sample in samples:
            del sample["rewards"]
            del sample["prompt_ids"]
            del sample['metadata']
            del sample['image']


        #################### TRAINING ####################
        if config.enable_mem_log:
            memory_profiler.snapshot(f"epoch_{epoch}_before_training")

        total_batch_size = len(samples) # = config.train.batch_size * config.train.num_batches_per_epoch

        pipeline.transformer.train()
        
        # Add some noise to default parameters for better exploration
        if hasattr(config.train, 'param_noise_std') and config.train.param_noise_std > 0:
            for p in transformer_trainable_parameters:
                p.data += torch.randn_like(p.data) * config.train.param_noise_std

        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples
            perm = torch.randperm(total_batch_size)
            samples = [samples[i] for i in perm]

            # sample:{
            # 'height': int,
            # 'width': int,
            # 'prompt': str,
            # 'all_latents': Tensor(1, config.sample.num_steps + 1, seq_len, c),
            # 'advantages': Tensor(1, 1),
            # }
            keys = samples[0].keys()
            samples = [samples[i:i+config.train.batch_size] for i in range(0, total_batch_size, config.train.batch_size)]
            samples = [
                {
                    # Catenate along batch dimension if the entry is Tensor
                    k: torch.cat([s[k] for s in batch], dim=0)
                    if isinstance(batch[0][k], torch.Tensor)
                    else [batch[_][k] for _ in range(len(batch))] # for other type -  cat to a list
                    for k in keys
                }
                for batch in samples
            ]

            info = defaultdict(list)

            for i, sample in tqdm(
                list(enumerate(samples)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                for j in tqdm(
                    train_timestep_indices,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    if config.enable_mem_log and i % 10 == 0:
                        memory_profiler.snapshot(f"epoch_{epoch}_step_{i}_timestep_{j}_before_forward")

                    with accelerator.accumulate(transformer):
                        loss_type_map = {
                            'ppo': compute_ppo_loss,
                            'nft': compute_nft_loss,
                            'guard_grpo': compute_guard_grpo_loss,
                        }
                        loss_fn = loss_type_map.get(config.train.loss_type.lower(), None)
                        assert loss_fn is not None, f"Unsupported loss type: {config.train.loss_type}, expected one of {list(loss_type_map.keys())}"

                        loss, loss_info = loss_fn(
                            config=config,
                            accelerator=accelerator,
                            pipeline=pipeline,
                            transformer=transformer,
                            sample=sample,
                            timestep_index=j,
                            autocast=autocast,
                        )

                        for k, v in loss_info.items():
                            info[k].append(v.detach())

                        # Track loss tensors
                        if config.enable_mem_log:
                            memory_profiler.track_tensors(info, "loss_info")
                            if i % 10 == 0:
                                memory_profiler.snapshot(f"epoch_{epoch}_step_{i}_timestep_{j}_before_backward")

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                        if config.enable_mem_log and i % 10 == 0:
                            memory_profiler.snapshot(f"epoch_{epoch}_step_{i}_timestep_{j}_after_backward")

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            logging_platform.log(info, step=global_step)

                        if config.enable_mem_log:
                            memory_profiler.snapshot(f"epoch_{epoch}_step_{i}_after_optimization")
                            memory_profiler.print_full_report(f"epoch_{epoch}_step_{i}")

                        global_step += 1
                        info = defaultdict(list)

                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients

        with torch.no_grad():
            decay = return_decay(global_step, config.train.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                # In-place update
                tgt_param.data.mul_(decay).add_(src_param.detach().data, alpha=1 - decay)
                assert src_param is not tgt_param

        if config.enable_mem_log:
            memory_profiler.cleanup_and_snapshot(f"epoch_{epoch}_end")
            # Clear tensor accumulation info in profiler to save memory
            memory_profiler.tensor_tracker.clear_stats()

        epoch += 1
        
if __name__ == "__main__":
    app.run(main)