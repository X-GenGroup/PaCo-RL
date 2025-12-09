# paco_grpo.diffusers_patch.sde_denoising_step.py
# Modified from https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py

import math
from typing import Optional, Union, Tuple
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from ..utils import to_broadcast_tensor

def denoising_sde_step_with_logprob(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    sigma: Union[float, torch.FloatTensor],
    sigma_prev: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: Union[int, float, list[float], torch.FloatTensor] = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
    sigma_max: Optional[float] = 0.98,
    cps : bool = False,
    return_log_prob : bool = True,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Predict the sample from the previous timestep by **reversing** the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity). Specially, when noise_level is zero, the process becomes deterministic.

    Args:
        scheduler (`FlowMatchEulerDiscreteScheduler`):
            A scheduler object that handles the scheduling of the diffusion process.
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        sigma (`float` | `torch.FloatTensor`):
            The current noise level (sigma) in the diffusion chain. This can be different for each sample in the batch.
        sigma_prev (`float` | `torch.FloatTensor`):
            The previous noise level (sigma) in the diffusion chain. This can be different for each sample in the batch.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        noise_level (`int` | `float` | `list[float]` | `torch.FloatTensor`, *optional*, defaults to 0.7):
            The noise level parameter, can be different for each sample in the batch. This parameter controls the standard deviation of the noise added to the denoised sample.
        prev_sample (`torch.FloatTensor`):
            The next insance of the sample. If given, calculate the log_prob using given `prev_sample` as predicted value.
        generator (`torch.Generator`, *optional*):
            A random number generator for SDE solving. If not given, a random generator will be used.
        sigma_max (`float`, *optional*, defaults to 0.98):
            The maximum noise level (sigma) used to avoid numerical issues when sigma is 1.
        cps (`bool`, *optional*, defaults to False):
            Whether to use coefficient preserving sampling (CPS) in the denoising step.
        return_log_prob (`bool`, *optional*, defaults to True):
            Whether to return the log probability of the transition.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    # Convert noise_level to a tensor with shape (batch_size, 1, 1)
    noise_level = to_broadcast_tensor(noise_level, sample)
    sigma = to_broadcast_tensor(sigma, sample)
    sigma_prev = to_broadcast_tensor(sigma_prev, sample)

    dt = sigma_prev - sigma # dt is negative, (batch_size, 1, 1)

    if not cps:
        sigma_max = to_broadcast_tensor(sigma_max, sample) # To avoid dividing by zero
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level # (batch_size, 1, 1)
        
        # FlowGRPO sde
        # Equation (9):
        #              sigma <-> t
        #        noise_level <-> a below Equation (9) - gives sigma_t = sqrt(t/(1-t))*a in the paper - corresponsds to std_dev_t = sqrt(sigma/(1-sigma))*noise_level here
        #                 dt <-> -\delta_t
        #       model_output <-> v_\theta(x_t, t)
        #             sample <-> x_t
        #        prev_sample <-> x_{t+\delta_t}
        #          std_dev_t <-> sigma_t

        prev_sample_mean = sample * (1 + std_dev_t**2 / (2 * sigma) * dt) + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        
        if prev_sample is None:
            # Non-determistic step, add noise to it
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            # Last term of Equation (9)
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

        if return_log_prob:
            log_prob = (
                -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
                - torch.log(std_dev_t * torch.sqrt(-1 * dt))
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    else:
        # FlowCPS
        std_dev_t = sigma_prev * torch.sin(noise_level * torch.pi / 2) # sigma_t in paper
        pred_original_sample = sample - sigma * model_output # predicted x_0 in paper
        noise_estimate = sample + model_output * (1 - sigma) # predicted x_1 in paper
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)
    
        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        if return_log_prob:
            log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    if not return_log_prob:
        log_prob = torch.zeros(sample.shape[0], device=sample.device)

    # Returns x_{t+\delta_t}, log_prob, x_{t+\delta_t} mean, sigma_t
    return prev_sample, log_prob, prev_sample_mean, std_dev_t