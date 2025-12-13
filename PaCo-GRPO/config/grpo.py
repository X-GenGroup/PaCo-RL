import ml_collections
import os
import math
from typing import Optional, Union, Dict, Callable
from importlib.util import spec_from_file_location, module_from_spec
import inspect
from logging import getLogger


import numpy as np
from scipy.stats import gmean, hmean
import torch
from datetime import datetime

logger = getLogger(__name__)

time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

spec = spec_from_file_location('base', os.path.join(os.path.dirname(__file__), "base.py"))
base = module_from_spec(spec)
spec.loader.exec_module(base)

FLUX_MODEL_PATH = "black-forest-labs/FLUX.1-dev"
FLUX_KONTEXT_MODEL_PATH = "black-forest-labs/FLUX.1-Kontext-dev"
QWEN_EDIT_MODEL_PATH = "Qwen/Qwen-Image-Edit"
SAVE_DIR = 'logs' # Save dir for checkpoints and evaluation results

def get_gpu_count():
    """
        Get gpu number
    """
    # 1. Get CUDA_VISIBLE_DEVICES first
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible = os.environ['CUDA_VISIBLE_DEVICES']
        if cuda_visible:
            return len(cuda_visible.split(','))
    
    # 2. Use torch
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    
    return 1

def get_config(name):
    return globals()[name]()

# --------------------------------------------------base------------------------------------------------------------
def get_base_config():
    config = base.get_config()
    # Run info
    config.resume_from_id = None
    config.resume_from_step = None
    config.resume_from_epoch = None
    config.project_name = 'Paco-GRPO'

    gpu_number = get_gpu_count()

    # Debug settings
    config.enable_mem_log = False

    # Testing
    config.test.resolution = 1024
    config.test.save_eval_images = True
    config.test.batch_size = 5
    config.test.num_steps = 20

    # Sampling    
    ## sde window scheduler
    config.sample.global_std = True
    config.sample.cps = False # Use cps sampling or not
    config.sample.num_steps = 10
    config.sample.noise_steps = [1] # Use sde sampling within these noise steps, e.g., [1,2,3,4]
    config.sample.num_noise_steps = None # Default to all noise steps.
    # If set noise_steps=[1,2,3,4] and num_noise_steps=1, then randomly select 1 from [1,2,3,4] for each iteration.
    config.sample.noise_level = 0.7
    config.sample.guidance_scale = 3.5
    config.sample.same_latent = False  # Whether to use the same init noise for the same prompt

    # Training
    ## batches
    config.enable_gradient_checkpointing = False
    config.sample.batch_size = 1
    config.sample.reward_batch_size = min(config.sample.batch_size * 4, 8) # Reward computation batch size
    config.sample.num_images_per_prompt = 16
    config.sample.unique_sample_num_per_epoch = 42 # Number of unique prompts used in each epoch all gathered

    # Search for proper `unique_sample_num_per_epoch`
    sample_num_per_iteration = config.sample.batch_size * gpu_number
    step = sample_num_per_iteration // math.gcd(config.sample.num_images_per_prompt, sample_num_per_iteration)
    new_unique_sample_num = (config.sample.unique_sample_num_per_epoch + step - 1) // step * step
    if new_unique_sample_num != config.sample.unique_sample_num_per_epoch:
        logger.warning(f"""Adjusting `unique_sample_num_per_epoch` from {config.sample.unique_sample_num_per_epoch} to {new_unique_sample_num} to ensure even batching across GPUs.""")
        config.sample.unique_sample_num_per_epoch = new_unique_sample_num
    
    # Total number of samples across all processes
    config.sample.sample_num_per_epoch = config.sample.num_images_per_prompt * config.sample.unique_sample_num_per_epoch

    # number of batches per epoch per GPU
    config.sample.num_batches_per_epoch = int(config.sample.sample_num_per_epoch / (gpu_number * config.sample.batch_size))

    # Training
    # Whether to add noise to param after sampling, before training.
    # Less than 0.01 is not harmful, but not improve performance. Large value collapse the training, for sure.
    config.train.resolution = 512
    config.train.param_noise_std = 0
    config.train.loss_type = 'ppo' # options: ['ppo', 'guard_grpo', 'nft'], where `ppo` is equivalent to Flow-GRPO
    config.train.batch_size = config.sample.batch_size
    config.train.learning_rate = 3e-4
    config.train.gradient_step_per_epoch = 1 if 'nft' in config.train.loss_type else 2
    assert config.sample.num_batches_per_epoch % config.train.gradient_step_per_epoch == 0, f"""Make sure num_batches_per_epoch is divisible by gradient_step_per_epoch."""
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // config.train.gradient_step_per_epoch
    config.train.num_inner_epochs = 1
    config.train.guidance_scale = 3.5
    config.train.timesteps = config.sample.noise_steps # Train on all noise steps
    config.train.beta = 0
    config.train.nft_beta = 1
    config.train.decay_type = 1 if 'nft' in config.train.loss_type else 0 # Diffusion-NFT uses ema for decoupling sampling and training policies.
    config.train.ema = True

    # Other settings
    config.per_prompt_stat_tracking = True
    config.max_sequence_length = 512
    config.use_lora = True # Use lora, `False` is not tested
    config.enable_flexible_size = False # Use flexible resolution for training, `True` is not tested

    return config

# --------------------------------------------------Some general aggregation functions------------------------------------------------------------
# Default: none (means simple weighted sum)

def geometric_mean_aggregate_fn(**grouped_rewards: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Aggregate multiple rewards using geometric mean for each prompt group.
    
    Geometric mean is useful when rewards are on different scales and you want
    to balance their influence multiplicatively.
    
    Args:
        **grouped_rewards: Each kwarg is {prompt: reward_array}
                          e.g., quality={'p1': [0.8, 0.9], 'p2': [0.6, 0.7]}
    
    Returns:
        Dictionary mapping each prompt to its aggregated advantages using geometric mean.
    
    Example:
        >>> grouped_rewards = {
        ...     'quality': {'p1': np.array([0.8, 0.9]), 'p2': np.array([0.6, 0.7])},
        ...     'safety': {'p1': np.array([0.9, 0.95]), 'p2': np.array([0.7, 0.75])}
        ... }
        >>> result = geometric_mean_aggregate_fn(**grouped_rewards)
        >>> # result['p1'] ≈ [√(0.8*0.9), √(0.9*0.95)]
    """
    prompts = next(iter(grouped_rewards.values())).keys()
    result = {}
    
    for prompt in prompts:
        # Stack all reward arrays for this prompt: (num_rewards, sample_size)
        reward_stack = np.stack([
            grouped_rewards[reward_name][prompt] 
            for reward_name in grouped_rewards
        ], axis=0)
        
        # Compute geometric mean across reward types (axis=0)
        result[prompt] = gmean(reward_stack, axis=0)
    
    return result

def harmonic_mean_aggregate_fn(**grouped_rewards: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Aggregate multiple rewards using harmonic mean for each prompt group.
    
    Harmonic mean is useful when you want to penalize low values more heavily,
    ensuring all rewards are reasonably high.
    
    Args:
        **grouped_rewards: Each kwarg is {prompt: reward_array} e.g., quality={'p1': [0.8, 0.9], 'p2': [0.6, 0.7]}
    
    Returns:
        Dictionary mapping each prompt to its aggregated advantages using harmonic mean.
    
    Example:
        >>> grouped_rewards = {
        ...     'quality': {'p1': np.array([0.8, 0.9]), 'p2': np.array([0.6, 0.7])},
        ...     'safety': {'p1': np.array([0.9, 0.95]), 'p2': np.array([0.7, 0.75])}
        ... }
        >>> result = harmonic_mean_aggregate_fn(**grouped_rewards)
        >>> # result['p1'] ≈ [2/(1/0.8 + 1/0.9), 2/(1/0.9 + 1/0.95)]
    """
    prompts = next(iter(grouped_rewards.values())).keys()
    result = {}
    
    for prompt in prompts:
        # Stack all reward arrays for this prompt: (num_rewards, sample_size)
        reward_stack = np.stack([
            grouped_rewards[reward_name][prompt] 
            for reward_name in grouped_rewards
        ], axis=0)
        
        # Compute harmonic mean across reward types (axis=0)
        result[prompt] = hmean(reward_stack, axis=0)
    
    return result

def log_tame_aggregate_fn(
    threshold: Union[float, str] = 'mean',
    reward_weights: Dict[str, float] = {}
):
    """
    Factory function that creates a log-tame aggregation function.
    
    Log-tame transformation applies log(1 + x) to rewards with high variance
    (high coefficient of variation h = std/mean) to reduce the influence of outliers
    and stabilize training.
    
    Args:
        threshold: Threshold for coefficient of variation (h = std/mean).
                  - float: Use this value as threshold
                  - 'mean': Use mean of all h values as threshold
                  - 'median': Use median of all h values as threshold
        reward_weights: Optional weights for each reward type. If None, use equal weights (1.0).
    
    Returns:
        Aggregation function that accepts **grouped_rewards and returns aggregated advantages.
    
    Algorithm:
        1. For each reward type, compute h = std(reward) / (mean(reward) + eps)
        2. If h > threshold, apply log-tame: reward = log(1 + reward)
        3. Apply weighted sum with provided weights
        4. Return aggregated rewards for each prompt
    
    Example:
        >>> # Create log-tame aggregation with mean threshold
        >>> agg_fn = log_tame_aggregate_fn(threshold='mean', 
        ...                                 reward_weights={'quality': 0.7, 'safety': 0.3})
        >>> 
        >>> # Use in config
        >>> config.train.aggregate_fn = agg_fn
        >>> config.train.aggregate_fn_code = inspect.getsource(log_tame_aggregate_fn)
    """
    def aggregate(**grouped_rewards: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
            Args:
                **grouped_rewards: Each kwarg is {prompt: reward_array} e.g., quality={'p1': [0.8, 0.9], 'p2': [0.6, 0.7]}
        """
        prompts = next(iter(grouped_rewards.values())).keys()
        
        # Step 1: Compute coefficient of variation (h) for each reward type
        h_values = {}
        for reward_name, prompt_dict in grouped_rewards.items():
            # Collect all reward values across prompts for this reward type
            all_rewards = np.concatenate([prompt_dict[p] for p in prompts])
            h = np.std(all_rewards) / (np.mean(all_rewards) + 1e-6)
            h_values[reward_name] = h
        
        # Step 2: Determine threshold
        if isinstance(threshold, float):
            h_threshold = threshold
        elif threshold == 'mean':
            h_threshold = np.mean(list(h_values.values()))
        elif threshold == 'median':
            h_threshold = np.median(list(h_values.values()))
        else:
            raise ValueError(f"Invalid threshold value: {threshold}. Expected float, 'mean', or 'median'.")
        
        # Step 3: Apply log-tame transformation to high-variance rewards
        transformed_rewards = {}
        for reward_name, prompt_dict in grouped_rewards.items():
            h = h_values[reward_name]
            if h > h_threshold:
                # Apply log transformation
                transformed_rewards[reward_name] = {
                    prompt: np.log(1 + reward_array)
                    for prompt, reward_array in prompt_dict.items()
                }
            else:
                # Keep original rewards
                transformed_rewards[reward_name] = prompt_dict
        
        # Step 4: Apply weighted sum        
        result = {}
        for prompt in prompts:
            # Stack rewards from all types at once
            reward_stack = np.array([
                transformed_rewards[reward_name][prompt] * reward_weights.get(reward_name, 1.0)
                for reward_name in transformed_rewards
            ])
            result[prompt] = np.sum(reward_stack, axis=0)
        
        return result
    
    return aggregate

def weighted_advantage_sum_aggregate_fn(
    global_std: bool = True,
    reward_weights: Dict[str, float] = {},
):
    """
    Factory function that creates an aggregation function using weighted sum of per-reward advantages.
    
    This approach computes advantages for each reward type independently, then combines them
    using weighted sum. Each reward is normalized separately using GRPO (reward - mean) / std
    before applying weights.
    
    Args:
        reward_weights: Optional weights for each reward type. If None, use equal weights (1.0).
        global_std: If True, use global std across all prompts for each reward type.
                   If False, use per-prompt std for each reward type.
    
    Returns:
        Aggregation function that accepts **grouped_rewards and returns aggregated advantages.
    
    Algorithm:
        For each reward type r:
            1. Compute advantages: adv_r = (reward_r - mean_r) / std_r
            2. Apply weight: weighted_adv_r = weight_r * adv_r
        Final: aggregated_advantage = sum(weighted_adv_r for all r)
    
    Example:
        >>> agg_fn = weighted_advantage_sum_aggregate_fn(
        ...     reward_weights={'quality': 0.7, 'safety': 0.3},
        ...     global_std=True
        ... )
        >>> config.train.aggregate_fn = agg_fn
        >>> config.train.aggregate_fn_code = inspect.getsource(weighted_advantage_sum_aggregate_fn)
    """
    def aggregate(**grouped_rewards: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
            Args:
                **grouped_rewards: Each kwarg is {prompt: reward_array} e.g., quality={'p1': [0.8, 0.9], 'p2': [0.6, 0.7]}
        """
        prompts = list(next(iter(grouped_rewards.values())).keys())

        normalized_advantages = {}
        
        # Compute advantages for each reward type separately
        for reward_name, prompt_dict in grouped_rewards.items():
            normalized_advantages[reward_name] = {}
            
            if global_std:
                all_rewards = np.concatenate([prompt_dict[p] for p in prompts])
                global_std_value = np.maximum(np.std(all_rewards, axis=0, keepdims=True), 1e-6)
            
            for prompt in prompts:
                rewards = prompt_dict[prompt]
                mean = np.mean(rewards, axis=0, keepdims=True)
                
                if global_std:
                    std = global_std_value
                else:
                    std = np.maximum(np.std(rewards, axis=0, keepdims=True), 1e-6)
                
                normalized_advantages[reward_name][prompt] = (rewards - mean) / std
        
        # Weighted sum of advantages
        result = {}
        for prompt in prompts:
            advantage_stack = np.array([
                normalized_advantages[reward_name][prompt] * reward_weights.get(reward_name, 1.0)
                for reward_name in grouped_rewards
            ])
            result[prompt] = np.sum(advantage_stack, axis=0)
        
        return result
    
    return aggregate

# -----------------------------------------------------------Flux---------------------------------------------------------------

def t2is():
    config = get_base_config()
    resolution = 512
    dataset_map = {
        256: "dataset/T2IS/half_2by2_micro_train",
        384: "dataset/T2IS/half_2by2_mini_train",
        512: "dataset/T2IS/half_2by2_small_train",
        768: "dataset/T2IS/half_2by2_medium_train",
        1024: "dataset/T2IS/half_2by2"
    }
    config.pretrained.model = FLUX_MODEL_PATH
    config.dataset = os.path.join(os.getcwd(), dataset_map[resolution])
    config.run_name = "Flux-T2IS-" + time_stamp
    config.prompt_fn = 'geneval'

    config.train.resolution = resolution
    config.test.resolution = 1024 # Keep test resolution to 1024 for evaluation
    # Add time stamp to save dir suffix to avoid overwriting
    config.save_dir = os.path.join(SAVE_DIR, time_stamp)
    config.save_freq = 10 # epoch
    config.eval_freq = 10 # 0 for no eval applied

    # Evaluation functions
    config.train.reward_fn = {
        "consistency_score": 0.2,
        "subfig_clipT" : 1,
        'pickscore': 1,
    }
    config.train.reward_fn_kwargs = {
        'consistency_score': {
            'model': 'PaCo-Reward-7B',
            'port': 8000
            },
    } # No special kwargs for subfig_clipT

    agg_fn = None # Use default weighted sum
    config.train.aggregate_fn_code = inspect.getsource(agg_fn) if agg_fn is not None else None
    config.train.aggregate_fn = agg_fn

    # Use the same reward function for testing, can be changed if needed
    config.test.reward_fn = config.train.reward_fn
    config.test.reward_fn_kwargs = config.train.reward_fn_kwargs
    config.test.aggregate_fn_code = config.train.aggregate_fn_code
    config.test.aggregate_fn = config.train.aggregate_fn
    # For example, for cross-model evaluation using gemma-4b-it, you can set:
    # config.test.reward_fn_kwargs = {
    #     'consistency_score': {
    #         'model': 'Gemma-4B-IT',
    #         'port': 8001
    #         },
    # }

    return config


# -----------------------------------------------------------Flux-Kontext---------------------------------------------------------------
def kontext_editing():
    config = get_base_config()
    config.dataset = 'dataset/GEdit-Bench/train_split'
    config.prompt_fn = 'arrow_editing'
    config.pretrained.model = FLUX_MODEL_PATH
    config.run_name = 'Kontext-' + time_stamp

    # Set some important params
    config.train.resolution = 384
    config.test.resolution = 512
    config.test.batch_size = 4
    config.test.num_steps = 20
    config.train.reward_fn = {
        "consistency_for_editing": 1.0,
    }
    config.train.reward_fn_kwargs = {
        'consistency_for_editing': {
            'model': 'PaCo-Reward-7B',
            'port': 8000
        }
    }
    agg_fn = None
    config.train.aggregate_fn = agg_fn
    config.train.aggregate_fn_code = inspect.getsource(config.train.aggregate_fn) if config.train.aggregate_fn is not None else None

    config.test.reward_fn = {
        "consistency_for_editing": 1.0,
    }
    config.test.reward_fn_kwargs = config.train.reward_fn_kwargs
    config.test.aggregate_fn = agg_fn
    config.test.aggregate_fn_code = inspect.getsource(config.test.aggregate_fn) if config.test.aggregate_fn is not None else None

    # Testing
    config.test.save_eval_images = True
    config.test.batch_size = 4
    config.test.num_steps = 20

    # Sampling
    ## sliding window scheduler
    config.sample.global_std = False
    config.sample.guidance_scale = 2.5
    config.sample.cps = False
    config.sample.num_steps = 8
    config.sample.noise_steps = [1]
    config.sample.noise_level = 0.9

    return config
    

# -----------------------------------------------------------Qwen-Image-Edit---------------------------------------------------------------
def qwen_editing():
    config = get_base_config()
    config.dataset = 'dataset/GEdit-Bench/train_split'
    config.prompt_fn = 'arrow_editing'
    config.pretrained.model = QWEN_EDIT_MODEL_PATH
    config.run_name = 'QwenImageEdit-' + time_stamp

    # Set some important params
    config.train.resolution = 384
    config.test.resolution = 512
    config.test.batch_size = 4
    config.test.num_steps = 20
    config.train.reward_fn = {
        "consistency_for_editing": 1.0,
    }
    config.train.reward_fn_kwargs = {
        'consistency_for_editing': {
            'model': 'PaCo-Reward-7B',
            'port': 8000
        }
    }
    agg_fn = None
    config.train.aggregate_fn = agg_fn
    config.train.aggregate_fn_code = inspect.getsource(config.train.aggregate_fn) if config.train.aggregate_fn is not None else None
    config.test.reward_fn = {
        "consistency_for_editing": 1.0,
    }
    config.test.reward_fn_kwargs = config.train.reward_fn_kwargs
    config.test.aggregate_fn = agg_fn
    config.test.aggregate_fn_code = inspect.getsource(config.test.aggregate_fn) if config.test.aggregate_fn is not None else None

    # Testing
    config.test.save_eval_images = True
    config.test.batch_size = 4
    config.test.num_steps = 20

    # Sampling
    ## sliding window scheduler
    config.sample.global_std = False
    config.sample.guidance_scale = 4.0
    config.sample.cps = False
    config.sample.num_steps = 10
    config.sample.noise_steps = [1,2,3,4]
    config.sample.noise_level = 1.0

    return config