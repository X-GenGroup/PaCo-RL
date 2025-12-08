# flowgrpo.diffusers_patch.flux_pipeline_kontext.py
from argparse import Namespace
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import torch
import numpy as np
import math
from typing import Optional, Union
from PIL import Image

from diffusers import QwenImageEditPipeline
from diffusers.utils import logging
from diffusers.pipelines.flux.pipeline_flux import logger
from diffusers.image_processor import PipelineImageInput

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

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


def compute_log_prob(
        transformer,
        pipeline : QwenImageEditPipeline,
        sample : dict[str, Union[torch.FloatTensor, List[int], List[str]]],
        timestep_index : int,
        config : Namespace
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    # 1. Prepare parameters
    latents: torch.FloatTensor = sample["all_latents"][:, timestep_index] # Latents at current timestep, shape (B, seq_len, C)
    next_latents: torch.FloatTensor = sample["all_latents"][:, timestep_index + 1] # Latents at next timestep, shape (B, seq_len, C)
    image_latents: torch.FloatTensor = sample["image_latents"] # Image latents, shape (B, image_seq_len, C)
    image: torch.FloatTensor = sample['ref_image'] # Conditioning image, PIL Image
    prompt: str = sample['prompt']
    negative_prompt = None
    negative_prompt_embeds = None
    negative_prompt_embeds_mask = None
    num_inference_steps: int = config.sample.num_steps
    scheduler = pipeline.scheduler
    timestep: torch.FloatTensor = sample["timesteps"][:, timestep_index] # (B,)
    timestep_next: torch.FloatTensor = sample["timesteps"][:, timestep_index + 1] if timestep_index + 1 < sample["timesteps"].shape[1] else torch.zeros_like(timestep) # (B,)
    timestep_max: torch.FloatTensor =  sample["timesteps"][:, 1]

    batch_size: int = latents.shape[0]
    num_channels_latents: int = pipeline.transformer.config.in_channels // 4
    # `height` and `width` in `sample` are optional and prioritized if present
    height: int = config.train.resolution if 'height' not in sample else sample['height'][0] # All height/width in the batch should be the same
    width: int = config.train.resolution if 'width' not in sample else sample['width'][0] # All height/width in the batch should be the same
    device = latents.device
    dtype = latents.dtype
    true_cfg_scale = config.train.guidance_scale

    max_area = config.train.resolution ** 2
    # `calulated_width` and `calculated_height` are used for resizing the input conditioning image
    # resize the conditioning image to be smaller than `config.train.resolution**2` while keeping aspect ratio
    image_size = image.size
    calculated_width, calculated_height, _ = calculate_dimensions(max_area, image_size[0] / image_size[1])
    # `height` and `width` are used for the output image size
    height = height or calculated_height
    width = width or calculated_width

    # 1. Set the scheduler, shift timesteps/sigmas according to full image size (image_seq_len)
    _ = set_scheduler_timesteps(
        scheduler=pipeline.scheduler,
        num_inference_steps=num_inference_steps,
        seq_len=latents.shape[1],
        device=device,
    )
    noise_level = pipeline.scheduler.get_noise_level_for_timestep(timestep[0].item())
    
    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
        image = pipeline.image_processor.resize(image, calculated_height, calculated_width)
        prompt_image = image
        image = pipeline.image_processor.preprocess(image, calculated_height, calculated_width)
        image = image.unsqueeze(2)
    
    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
        image=prompt_image,
        prompt=prompt,
        device=device,
        max_sequence_length=config.max_sequence_length,
    )
    if do_true_cfg:
        negative_prompt_embeds, negative_prompt_embeds_mask = pipeline.encode_prompt(
            image=prompt_image,
            prompt=negative_prompt,
            prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=negative_prompt_embeds_mask,
            device=device,
            max_sequence_length=config.max_sequence_length,
        )

    # 4. Prepare latent variables
    latents, image_latents = pipeline.prepare_latents(
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        None,
        latents,
    )
    img_shapes = [
        [
            (1, height // pipeline.vae_scale_factor // 2, width // pipeline.vae_scale_factor // 2),
            (1, calculated_height // pipeline.vae_scale_factor // 2, calculated_width // pipeline.vae_scale_factor // 2),
        ]
    ] * batch_size

    # 5. Prepare timesteps
    timesteps = set_scheduler_timesteps(
        scheduler=pipeline.scheduler,
        num_inference_steps=num_inference_steps,
        seq_len=latents.shape[1],
        device=device,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)

    # handle guidance
    guidance = None

    if pipeline.attention_kwargs is None:
        pipeline._attention_kwargs = {}

    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
    negative_txt_seq_lens = (
        negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
    )

    latent_model_input = torch.cat([latents, image_latents], dim=1)

    with pipeline.transformer.cache_context("cond"):
        noise_pred = pipeline.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs=pipeline.attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, : latents.size(1)]

    if do_true_cfg:
        with pipeline.transformer.cache_context("uncond"):
            neg_noise_pred = pipeline.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                encoder_hidden_states=negative_prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=negative_txt_seq_lens,
                attention_kwargs=pipeline.attention_kwargs,
                return_dict=False,
            )[0]
        neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
        comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        noise_pred = comb_pred * (cond_norm / noise_norm)

    # 7. Compute the log probability
    # Compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = denoising_sde_step_with_logprob(
        scheduler=pipeline.scheduler,
        model_output=noise_pred.float(),
        sigma=timestep / 1000,
        sigma_prev=timestep_next / 1000,
        sample=latents.float(),
        noise_level=noise_level,
        prev_sample=next_latents.float(),
        sigma_max=timestep_max / 1000,
        cps=config.sample.cps,
        return_log_prob=True,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

@torch.no_grad()
def qwenimage_edit_pipeline(
    self : QwenImageEditPipeline,
    image: Optional[PipelineImageInput] = None,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    true_cfg_scale: float = 4.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 1.0,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    max_area: int = 1024**2,
    noise_level: Optional[float] = None,
    cps : bool = False,
) -> Tuple[
        Union[List[Image.Image], torch.FloatTensor],
        List[torch.FloatTensor],
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
    image_size = image[0].size if isinstance(image, list) else image.size
    # `calulated_width` and `calculated_height` are used for resizing the input conditioning image
    # `height` and `width` are used for the output image size
    calculated_width, calculated_height, _ = calculate_dimensions(max_area, image_size[0] / image_size[1])
    # init diffusers code logic: `max_area`` will be ignored if both height and width are given
    height = height or calculated_height
    width = width or calculated_width

    multiple_of = self.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # 3. Preprocess image
    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
        image = self.image_processor.resize(image, calculated_height, calculated_width)
        prompt_image = image
        image = self.image_processor.preprocess(image, calculated_height, calculated_width)
        image = image.unsqueeze(2)

    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    prompt_embeds, prompt_embeds_mask = self.encode_prompt(
        image=prompt_image,
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    if do_true_cfg:
        negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
            image=prompt_image,
            prompt=negative_prompt,
            prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=negative_prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )


    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, image_latents = self.prepare_latents(
        image,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    img_shapes = [
        [
            (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
            (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
        ]
    ] * batch_size

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    if self.attention_kwargs is None:
        self._attention_kwargs = {}

    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
    negative_txt_seq_lens = (
        negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
    )
    
    # 6. Denoising loop
    all_latents = [latents]
    all_noise_timestep_indices = []
    self.scheduler.set_begin_index(0)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            self._current_timestep = t
            # Get noise_level. If not given in the arguments, use the sliding window scheduler's method to retrieve it.
            current_noise_level = noise_level if noise_level is not None else self.scheduler.get_noise_level_for_timestep(t)

            # Concatenate image_latents to latents
            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            timestep_next = timesteps[i + 1].expand(latents.shape[0]).to(latents.dtype) if i + 1 < len(timesteps) else torch.zeros_like(timestep)

            with self.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

            if do_true_cfg:
                with self.transformer.cache_context("uncond"):
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        encoder_hidden_states=negative_prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=negative_txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)


            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype

            latents, _, _, _ = denoising_sde_step_with_logprob(
                scheduler=self.scheduler,
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
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            all_latents.append(latents)
            if current_noise_level > 0:
                all_noise_timestep_indices.append(i)
    
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    self._current_timestep = None
    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    timesteps = timesteps.unsqueeze(0).expand(batch_size, -1) # (batch_size, num_inference_steps)
    return image, all_latents, timesteps, image_latents