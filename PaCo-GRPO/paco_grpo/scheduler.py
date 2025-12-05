import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers import FlowMatchEulerDiscreteScheduler

class FlowMatchNoiseScheduler(FlowMatchEulerDiscreteScheduler):
    """
        A scheduler with noise level provided within the given steps
    """
    def __init__(
        self,
        noise_level : float = 0.7,
        noise_steps : Optional[Union[int, list, torch.Tensor]] = None,
        num_noise_steps : Optional[int] = None,
        seed : int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        if noise_steps is None:
            noise_steps = range(0, self.config.num_train_timesteps)

        self.noise_level = noise_level

        assert self.noise_level >= 0, "Noise level must be non-negative."

        self.noise_steps = torch.tensor(noise_steps, dtype=torch.int64)
        self.num_noise_steps = num_noise_steps if num_noise_steps is not None else len(noise_steps) # Default to all noise steps
        self.seed = seed

    @property
    def current_noise_steps(self) -> torch.Tensor:
        """
            Returns the current noise steps under the self.seed.
            Randomly select self.num_noise_steps from self.noise_steps.
        """
        if self.num_noise_steps >= len(self.noise_steps):
            return self.noise_steps
        generator = torch.Generator().manual_seed(self.seed)
        selected_indices = torch.randperm(len(self.noise_steps), generator=generator)[:self.num_noise_steps]
        return self.noise_steps[selected_indices]

    def get_noise_timesteps(self) -> torch.Tensor:
        """
            Returns timesteps within the current window.
            If `left_boundary` is provided, use it instead of the current left boundary.
            If `window_size` is provided, use it instead of the current window size.
        """

        return self.timesteps[self.noise_steps]

    def get_noise_sigmas(self, left_boundary : Optional[int] = None, window_size : Optional[int] = None) -> torch.Tensor:
        """
            Returns sigmas within the current window.
            If `left_boundary` is provided, use it instead of the current left boundary.
            If `window_size` is provided, use it instead of the current window size.
        """

        return self.sigmas[self.noise_steps]

    def get_noise_levels(self) -> torch.Tensor:
        """ Returns noise levels on all timesteps, where noise level is non-zero only within the current window. """
        noise_levels = torch.zeros_like(self.timesteps, dtype=torch.float32)
        noise_levels[self.current_noise_steps] = self.noise_level
        return noise_levels

    def get_noise_level_for_timestep(self, time_step) -> float:
        """
            Return the noise level for a specific timestep.
        """
        time_step_index = self.index_for_timestep(time_step)
        if time_step_index in self.noise_steps:
            return self.noise_level

        return 0.0
    
    def set_seed(self, seed: int):
        """
            Set the random seed for noise generation.
        """
        self.seed = seed