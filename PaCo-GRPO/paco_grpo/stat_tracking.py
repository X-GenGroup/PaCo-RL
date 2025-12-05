from typing import List, Optional, Union, Callable, Dict
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerPromptStatTracker:
    """
    Tracker for computing per-prompt advantages in reinforcement learning from human feedback (RLHF).
    
    This class groups samples by prompt and computes normalized advantages using various algorithms
    such as GRPO (Group Relative Policy Optimization) and rank-based GRPO. It supports multi-reward
    aggregation with customizable weighting and aggregation functions.
    
    Attributes:
        global_std (bool): If True, use global standard deviation across all prompts for normalization.
                          If False, use per-prompt standard deviation.
        stats (dict): Historical statistics for each prompt (stores aggregated rewards).
        history_prompts (set): Set of hashed prompts seen in history.
    
    Example:
        >>> tracker = PerPromptStatTracker(global_std=True)
        >>> prompts = ['p1', 'p1', 'p2', 'p2']
        >>> rewards = {'quality': [0.8, 0.9, 0.6, 0.7], 'safety': [0.9, 0.95, 0.7, 0.75]}
        >>> advantages = tracker.compute_advantages(prompts, rewards, 
        ...                                         reward_weights={'quality': 0.7, 'safety': 0.3})
    """
    
    def __init__(self, global_std: bool = True):
        """
        Initialize the PerPromptStatTracker.
        
        Args:
            global_std: Whether to use global standard deviation for normalization.
                       True: normalize using std across all prompt groups (default).
                       False: normalize using std within each prompt group.
        """
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    def _group_prompt_rewards(
        self,
        prompts: np.ndarray,
        rewards: Dict[str, np.ndarray],
        unique_prompts: np.ndarray,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Group reward values by their corresponding prompts.
        
        Args:
            prompts: Array of prompt strings for each sample.
            rewards: Dictionary mapping reward names to reward value arrays.
            unique_prompts: Array of unique prompt strings.
        
        Returns:
            Nested dictionary with structure {reward_name: {prompt: reward_array}}.
            
        Example:
            Input: prompts=['p1', 'p1', 'p2'], rewards={'r1': [1, 2, 3]}
            Output: {'r1': {'p1': [1, 2], 'p2': [3]}}
        """
        grouped_rewards = {}
        
        for reward_name, reward_values in rewards.items():
            grouped_rewards[reward_name] = {}
            for prompt in unique_prompts:
                mask = prompts == prompt
                grouped_rewards[reward_name][prompt] = reward_values[mask]
        
        return grouped_rewards

    def update_history(self, unique_prompts: np.ndarray) -> set:
        """
        Update the history with newly seen prompts.
        
        Args:
            unique_prompts: Array of unique prompt strings to add to history.
        
        Returns:
            Updated set of hashed prompts in history.
        """
        self.history_prompts.update(hash(p) for p in unique_prompts)
        return self.history_prompts
    
    def _update_stats(self, combined_rewards: Dict[str, np.ndarray]):
        """
        Update statistics with combined (aggregated) rewards for each prompt.
        
        Args:
            combined_rewards: Dictionary mapping each prompt to its aggregated reward array.
        """
        for prompt, rewards in combined_rewards.items():
            if prompt not in self.stats:
                self.stats[prompt] = rewards
            else:
                self.stats[prompt] = np.concatenate([self.stats[prompt], rewards])

    def _calc_grpo_advantage(
        self, 
        grouped_rewards: Dict[str, Dict[str, np.ndarray]],
        reward_weights: Dict[str, float] = {},
    ) -> Dict[str, np.ndarray]:
        """
        Calculate advantages using Group Relative Policy Optimization (GRPO).
        
        GRPO normalizes rewards within each prompt group using (reward - mean) / std.
        Multiple reward types are combined via summation before normalization.
        
        Args:
            grouped_rewards: Nested dictionary {reward_name: {prompt: reward_array}}.
            reward_weights: Dictionary mapping reward names to their weights for combination.
        
        Returns:
            Dictionary mapping each prompt to its advantage array.
            
        Algorithm:
            1. Sum all reward types for each prompt group
            2. For each group: advantage = (reward - group_mean) / std
            3. std is either global (across all groups) or local (per group)
        """
        prompts = list(next(iter(grouped_rewards.values())).keys())
        
        combined_rewards = {}
        for prompt in prompts:
            reward_stack = np.array([
                grouped_rewards[reward_name][prompt] * reward_weights.get(reward_name, 1.0)
                for reward_name in grouped_rewards
            ])
            combined_rewards[prompt] = np.sum(reward_stack, axis=0)
        
        # Update stats with aggregated rewards
        self._update_stats(combined_rewards)
        
        if self.global_std:
            all_rewards = np.concatenate(list(combined_rewards.values()))
            global_std = np.maximum(np.std(all_rewards, axis=0, keepdims=True), 1e-6)
        
        advantages = {}
        for prompt, rewards in combined_rewards.items():
            mean = np.mean(rewards, axis=0, keepdims=True)
            
            if self.global_std:
                std = global_std
            else:
                std = np.maximum(np.std(rewards, axis=0, keepdims=True), 1e-6)
            
            advantages[prompt] = (rewards - mean) / std
        
        return advantages

    def _calc_rank_grpo_advantage(
        self, 
        grouped_rewards: Dict[str, Dict[str, np.ndarray]],
        reward_weights: Dict[str, float] = {},
    ) -> Dict[str, np.ndarray]:
        """
        Calculate advantages using rank-based GRPO.
        
        Rank-based GRPO first converts absolute reward values to normalized ranks [0, 1],
        then applies standard GRPO normalization. This reduces sensitivity to reward scale
        and outliers.
        
        Args:
            grouped_rewards: Nested dictionary {reward_name: {prompt: reward_array}}.
        
        Returns:
            Dictionary mapping each prompt to its rank-based advantage array.
            
        Algorithm:
            1. Within each prompt group, convert rewards to ranks normalized to [0, 1]
            2. Apply standard GRPO on the ranked rewards
        """
        rank_based_rewards = {}
        
        for reward_name, reward_dict in grouped_rewards.items():
            rank_based_rewards[reward_name] = {}
            for prompt, rewards in reward_dict.items():
                ranks = np.argsort(np.argsort(rewards, axis=0), axis=0)
                group_size = rewards.shape[0]
                rank_based_rewards[reward_name][prompt] = (
                    ranks / max(group_size - 1, 1.0)
                )
        
        return self._calc_grpo_advantage(rank_based_rewards, reward_weights)
    
    def _map_advantages_to_order(
        self, 
        prompts: np.ndarray,
        unique_prompts: np.ndarray,
        advantages: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Map computed advantages back to the original sample order.
        
        Args:
            prompts: Array of prompt strings in original order.
            unique_prompts: Array of unique prompt strings.
            advantages: Dictionary mapping prompts to their advantage arrays.
        
        Returns:
            Advantage array aligned with the original sample order.
        """
        n_samples = len(prompts)
        mapped_advantages = np.empty(n_samples, dtype=np.float64)
        
        for prompt in unique_prompts:
            mask = prompts == prompt
            mapped_advantages[mask] = advantages[prompt]
        
        return mapped_advantages

    def compute_advantages(
        self,
        prompts: List[str],
        rewards: Dict[str, Union[np.ndarray, torch.Tensor]],
        type: str = 'grpo',
        reward_weights: Dict[str, float] = {},
        aggregate_fn: Optional[
            Callable[[Dict[str, Dict[str, np.ndarray]]], Dict[str, np.ndarray]]
        ] = None,
    ) -> np.ndarray:
        """
        Compute advantages from single or multi-dimensional rewards.
        
        This is the main interface for advantage computation. It groups samples by prompt,
        optionally aggregates multiple reward types, and computes normalized advantages
        using the specified algorithm.
        
        Args:
            prompts: List of prompt strings, one for each sample. Samples with the same
                    prompt are grouped together for advantage computation.
            rewards: Dictionary mapping reward names to reward value arrays. Each array
                    should have length equal to len(prompts).
                    Example: {'quality': [0.8, 0.9], 'safety': [0.7, 0.8]}
            type: Advantage computation algorithm. Options:
                 - 'grpo': Group Relative Policy Optimization (default)
                 - 'rank-grpo': Rank-based GRPO
            reward_weights: Optional dictionary of weights for each reward type. Used only
                           when aggregate_fn is None. Default weight is 1.0 for all rewards.
                           Example: {'quality': 0.7, 'safety': 0.3}
            aggregate_fn: Optional custom aggregation function. If provided, reward_weights
                         and type are ignored. The function should accept **kwargs where
                         each kwarg is a dict {prompt: reward_array} and return a dict
                         {prompt: advantage_array}.
                         Example: lambda **rewards: {p: max(r[p] for r in rewards.values())
                                                    for p in rewards[list(rewards.keys())[0]]}
        
        Returns:
            NumPy array of advantages with the same length and order as the input prompts.
        
        Examples:
            >>> # Single reward with default GRPO
            >>> tracker.compute_advantages(['p1', 'p1', 'p2'], {'score': [1, 2, 3]})
            
            >>> # Multi-reward with weights
            >>> tracker.compute_advantages(
            ...     ['p1', 'p1', 'p2', 'p2'],
            ...     {'quality': [1, 2, 3, 4], 'safety': [0.5, 0.8, 0.6, 0.9]},
            ...     reward_weights={'quality': 0.7, 'safety': 0.3}
            ... )
            
            >>> # Rank-based GRPO
            >>> tracker.compute_advantages(
            ...     ['p1', 'p1', 'p2', 'p2'],
            ...     {'score': [10, 50, 5, 25]},
            ...     type='rank-grpo'
            ... )
            
            >>> # Custom aggregation (take maximum)
            >>> def max_aggregate(**grouped_rewards):
            ...     prompts = next(iter(grouped_rewards.values())).keys()
            ...     return {p: np.max([grouped_rewards[r][p] for r in grouped_rewards], axis=0)
            ...             for p in prompts}
            >>> tracker.compute_advantages(
            ...     ['p1', 'p1'],
            ...     {'r1': [1, 2], 'r2': [2, 1]},
            ...     aggregate_fn=max_aggregate
            ... )
        """
        prompts_array = np.asarray(prompts)
        unique_prompts = np.unique(prompts_array)
        
        rewards_array = {
            k: (
                v.astype(np.float64, copy=False) if isinstance(v, np.ndarray)
                else np.array(v, dtype=np.float64)
            )
            for k, v in rewards.items()
        }
        
        grouped_rewards = self._group_prompt_rewards(
            prompts_array, rewards_array, unique_prompts
        )
        
        self.update_history(unique_prompts)

        if aggregate_fn is not None:
            # Custom aggregation function returns advantages directly
            advantages = aggregate_fn(**grouped_rewards)
            
            # For custom aggregation, we need to extract combined rewards for stats
            # Since aggregate_fn returns advantages, we need to reverse engineer the rewards
            # So we use the advantages as combined rewards for stats update
            self._update_stats(advantages)            
        else:
            if type == 'grpo':
                advantages = self._calc_grpo_advantage(grouped_rewards, reward_weights)
            elif type == 'rank-grpo':
                advantages = self._calc_rank_grpo_advantage(grouped_rewards, reward_weights)
            else:
                raise ValueError(
                    f"Unsupported advantage type: '{type}'. "
                    f"Supported types: 'grpo', 'rank-grpo'"
                )

        return self._map_advantages_to_order(
            prompts_array, unique_prompts, advantages
        )

    def get_stats(self):
        """
        Get statistics about the tracker's current state.
        
        Returns:
            Tuple of (avg_group_size, history_prompts, avg_group_std, global_std, zero_std_ratio):
            - avg_group_size: Average number of samples per prompt group
            - history_prompts: Total number of unique prompts seen
            - avg_group_std: Average standard deviation within prompt groups
            - global_std: Standard deviation across all samples
            - zero_std_ratio: Fraction of groups with near-zero std (< 1e-5)
        """
        if not self.stats:
            return 0, 0, 0, 0, 0
        
        all_values = list(self.stats.values())
        avg_group_size = sum(len(v) for v in all_values) / len(all_values)
        history_prompts = len(self.history_prompts)
        
        stds = np.array([np.std(v) for v in all_values])
        avg_group_std = np.mean(stds)
        
        global_std = np.std(np.concatenate(all_values))
        zero_std_ratio = np.sum(stds < 1e-5) / len(stds)
        
        return avg_group_size, history_prompts, avg_group_std, global_std, zero_std_ratio

    def clear(self):
        """
        Clear all stored statistics.
        
        Note: This clears the stats dict but preserves history_prompts.
        To fully reset, create a new tracker instance.
        """
        self.stats = {}


def main():
    """Run tests and benchmarks for PerPromptStatTracker."""
    import time
    
    tracker = PerPromptStatTracker()
    
    print("=== Performance Benchmark ===")
    n_samples = 100000
    n_prompts = 1000
    
    prompts = [f"prompt_{i % n_prompts}" for i in range(n_samples)]
    rewards = {
        'quality': np.random.randn(n_samples),
        'safety': np.random.randn(n_samples),
        'coherence': np.random.randn(n_samples),
    }
    reward_weights = {'quality': 0.5, 'safety': 0.3, 'coherence': 0.2}
    
    _ = tracker.compute_advantages(prompts[:1000], {k: v[:1000] for k, v in rewards.items()})
    
    start = time.time()
    advantages = tracker.compute_advantages(
        prompts, rewards, reward_weights=reward_weights
    )
    elapsed = time.time() - start
    
    print(f"Processed {n_samples} samples with {n_prompts} unique prompts")
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {n_samples/elapsed:.0f} samples/sec")
    print(f"Advantages shape: {advantages.shape}")
    print(f"Advantages mean: {np.mean(advantages):.4f}")
    print(f"Advantages std: {np.std(advantages):.4f}")
    
    # Check stats
    avg_group_size, history_prompts, avg_group_std, global_std, zero_std_ratio = tracker.get_stats()
    print(f"Stats - Avg group size: {avg_group_size:.2f}, History prompts: {history_prompts}")
    print(f"Stats - Avg group std: {avg_group_std:.4f}, Global std: {global_std:.4f}")
    print()
    
    print("=== Test 1: Single reward ===")
    tracker.clear()  # Clear for fresh test
    prompts = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = {'default': np.array([1, 2, -1, 4, 2, 1], dtype=np.float64)}
    advantages = tracker.compute_advantages(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts, avg_group_std, global_std, zero_std_ratio = tracker.get_stats()
    print(f"Stats - Avg group size: {avg_group_size:.2f}, History prompts: {history_prompts}")
    print()
    
    print("=== Test 2: Multi-reward with weights ===")
    tracker.clear()
    prompts = ['p1', 'p1', 'p2', 'p2']
    rewards = {
        'quality': np.array([1.0, 2.0, 3.0, 4.0]),
        'safety': np.array([0.5, 0.8, 0.6, 0.9])
    }
    advantages = tracker.compute_advantages(
        prompts, rewards, reward_weights={'quality': 0.7, 'safety': 0.3}
    )
    print("Advantages:", advantages)
    avg_group_size, history_prompts, avg_group_std, global_std, zero_std_ratio = tracker.get_stats()
    print(f"Stats - Avg group size: {avg_group_size:.2f}, History prompts: {history_prompts}")
    print()
    
    print("=== Test 3: Custom aggregation ===")
    
    def weighted_harmonic_mean(**grouped_rewards: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute weighted harmonic mean of rewards."""
        weights = {'quality': 0.6, 'safety': 0.4}
        prompts = next(iter(grouped_rewards.values())).keys()
        result = {}
        
        for prompt in prompts:
            weighted_sum = 0.0
            for reward_name in grouped_rewards:
                w = weights.get(reward_name, 1.0)
                r = grouped_rewards[reward_name][prompt]
                weighted_sum += w / (r + 1e-6)
            
            result[prompt] = len(grouped_rewards) / (weighted_sum + 1e-6)
        
        return result
    
    tracker.clear()
    prompts = ['x', 'x', 'y', 'y']
    rewards = {
        'quality': np.array([1.0, 2.0, 3.0, 4.0]),
        'safety': np.array([0.5, 0.8, 0.6, 0.9])
    }
    advantages = tracker.compute_advantages(
        prompts, rewards, aggregate_fn=weighted_harmonic_mean
    )
    print("Advantages:", advantages)
    print("Note: Stats not updated when using custom aggregate_fn")


if __name__ == "__main__":
    main()