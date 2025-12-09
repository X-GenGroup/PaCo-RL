# paco_grpo.diffusers_patch.flux_pipeline.py
from argparse import Namespace
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import torch
import numpy as np
import math
from typing import Optional, Union

from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.utils import logging
from diffusers.pipelines.flux.pipeline_flux import logger

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

from .sde_denoising_step import denoising_sde_step_with_logprob

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def set_scheduler_timesteps(
    scheduler,
    num_inference_steps: int,
    seq_len: int,
    sigmas: Optional[List[float]] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    # 5. Prepare scheduler, shift timesteps/sigmas according to image size (image_seq_len)
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None

    mu = calculate_shift(
        seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    return timesteps


def compute_log_prob(
        transformer : FluxTransformer2DModel,
        pipeline : FluxPipeline,
        sample : dict[str, torch.Tensor],
        timestep_index : int,
        config : Namespace
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    # 1. Prepare parameters
    latents = sample["all_latents"][:, timestep_index] # Latents at current timestep, shape (B, seq_len, C)
    next_latents = sample["all_latents"][:, timestep_index + 1] # Latents at next timestep, shape (B, seq_len, C)
    num_inference_steps = config.sample.num_steps
    scheduler = pipeline.scheduler
    timestep = sample["timesteps"][:, timestep_index] # (B,)
    timestep_next = sample["timesteps"][:, timestep_index + 1] if timestep_index + 1 < sample["timesteps"].shape[1] else torch.zeros_like(timestep) # (B,)
    timestep_max =  sample["timesteps"][:, 1]

    batch_size = latents.shape[0]
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    height = config.train.resolution if 'height' not in sample else sample['height'][0] # All height/width in the batch should be the same
    width = config.train.resolution if 'width' not in sample else sample['width'][0] # All height/width in the batch should be the same
    prompt = sample['prompt']
    device = latents.device
    dtype = latents.dtype


    # 1. Set the scheduler, shift timesteps/sigmas according to full image size (image_seq_len)
    _ = set_scheduler_timesteps(
        scheduler=pipeline.scheduler,
        num_inference_steps=num_inference_steps,
        seq_len=latents.shape[1],
        device=device,
    )
    noise_level = pipeline.scheduler.get_noise_level_for_timestep(timestep[0].item())

    # 2. Prepare prompt_embeds and latents if using dividing
    logger.setLevel(logging.ERROR) # To silent CLIP overflow warning
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
        max_sequence_length=config.max_sequence_length,
    )
    logger.setLevel(logging.WARNING) # Restore logger level
    

    # 3. Prepare image_ids according to the latents
    latents, image_ids = pipeline.prepare_latents(
        batch_size = batch_size,
        num_channels_latents = num_channels_latents,
        height = height,
        width = width,
        dtype=dtype,
        device=device,
        generator=None,
        latents=latents
    )

    # 4. Prepare guidance and predict the noise residual
    guidance = torch.tensor([config.sample.guidance_scale], device=device)

     # Predict the noise residual
    model_pred = transformer(
        hidden_states=latents,
        timestep=timestep / 1000, # which is scheduler.sigmas[timestep_index] exactly
        guidance=guidance.expand(latents.shape[0]),
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype),
        img_ids=image_ids,
        return_dict=False,
    )[0]
    
    
    # 5. Compute log prob
    # Compute the log prob of next_latents given latents under the current model
    # Here, use determistic denoising for normal diffusion process.
    prev_sample, log_prob, prev_sample_mean, std_dev_t = denoising_sde_step_with_logprob(
        scheduler=scheduler,
        model_output=model_pred.float(),
        sigma=timestep / 1000,
        sigma_prev=timestep_next / 1000,
        sample=latents.float(),
        noise_level=noise_level,
        prev_sample=next_latents.float(),
        cps=config.sample.cps,
        return_log_prob=True,
        sigma_max=timestep_max / 1000
    )


    return prev_sample, log_prob, prev_sample_mean, std_dev_t

@torch.no_grad()
def flux_pipeline(
    pipeline : FluxPipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt: Union[str, List[str]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 3.5,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    noise_level: Optional[float] = None,
    cps : bool = False,
) -> Tuple[
        torch.FloatTensor,
        List[torch.FloatTensor],
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipeline.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    pipeline._guidance_scale = guidance_scale
    pipeline._joint_attention_kwargs = joint_attention_kwargs
    pipeline._current_timestep = None
    pipeline._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
        prompt = [prompt]
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if isinstance(generator, torch.Generator):
        generator = [generator] * batch_size

    device = pipeline._execution_device

    lora_scale = (
        pipeline.joint_attention_kwargs.get("scale", None)
        if pipeline.joint_attention_kwargs is not None else None
    )
    
    # 3. Encode prompts
    logger.setLevel(logging.ERROR) # To silent CLIP overflow warning
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    logger.setLevel(logging.WARNING) # Restore logger level

    # 4. Prepare latent variables
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    latents, latent_image_ids = pipeline.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare scheduler, shift timesteps/sigmas according to image size (image_seq_len)
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(pipeline.scheduler.config, "use_flow_sigmas") and pipeline.scheduler.config.use_flow_sigmas:
        sigmas = None

    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    # FlowMatchEulerDiscreteScheduler has order 1, which gives num_warmup_steps=0
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)

    # handle guidance
    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)

    # 6. Denoising loop
    all_latents = [latents]
    all_noise_timestep_indices = []
    pipeline.scheduler.set_begin_index(0)
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            pipeline._current_timestep = t
            # Get noise_level. If not given in the arguments, use the sliding window scheduler's method to retrieve it.
            current_noise_level = noise_level if noise_level is not None else pipeline.scheduler.get_noise_level_for_timestep(t)


            current_prompt_embeds = prompt_embeds
            current_pooled_prompt_embeds = pooled_prompt_embeds
            img_ids = latent_image_ids

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            timestep_next = timesteps[i + 1].expand(latents.shape[0]).to(latents.dtype) if i + 1 < len(timesteps) else torch.zeros_like(timestep)

            noise_pred = pipeline.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance.expand(latents.shape[0]),
                pooled_projections=current_pooled_prompt_embeds,
                encoder_hidden_states=current_prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                joint_attention_kwargs=pipeline.joint_attention_kwargs,
                return_dict=False,
            )[0]

            noise_pred = noise_pred.to(prompt_embeds.dtype)
            latents_dtype = latents.dtype

            latents, _, _, _ = denoising_sde_step_with_logprob(
                scheduler=pipeline.scheduler,
                model_output=noise_pred.float(),
                sigma=timestep / 1000,
                sigma_prev=timestep_next / 1000,
                sample=latents.float(),
                noise_level=current_noise_level,
                prev_sample=None,
                sigma_max=timesteps[1].item() / 1000,
                cps=cps,
                return_log_prob=False,
            )
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            all_latents.append(latents)
            if current_noise_level > 0:
                all_noise_timestep_indices.append(i)
    
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    latents = latents.to(dtype=pipeline.vae.dtype)
    images = pipeline.vae.decode(latents, return_dict=False)[0]
    images = pipeline.image_processor.postprocess(images, output_type=output_type)

    # Offload all models
    pipeline.maybe_free_model_hooks()

    timesteps = timesteps.unsqueeze(0).expand(batch_size, -1) # (batch_size, num_inference_steps)
    return images, all_latents, prompt_embeds, pooled_prompt_embeds, all_noise_timestep_indices, timesteps 