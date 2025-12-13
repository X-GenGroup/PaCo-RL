import math
import torch
from torch.utils.data import Sampler, Dataset, DataLoader
from collections import Counter
from logging import getLogger

logger = getLogger(__name__)


class DistributedGroupKRepeatSampler(Sampler):
    """
    
        This class make sure all samples in one group is on the same device.
    """
    def __init__(self, dataset : Dataset, batch_size : int, k : int, m : int, num_replicas : int, rank : int, seed :int = 0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.m = m                    # `Least` number of unique sample per epoch
        self.num_replicas = num_replicas  # Total number of replicas, process num, gpu num
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization

        # To make sure all samples in one group is on the same device, num_samples_per_device should be multiple of k
        assert (k * m) % (num_replicas * batch_size) == 0, f"Please set config.sample.m (={m}) to make sure k*m (={k*m}) is divisible by num_replicas*batch_size (={num_replicas*batch_size})!"
        self.sample_num_per_epoch = k * m  # Total sample number per epoch
        self.num_batches_per_epoch = int(self.sample_num_per_epoch // (self.num_replicas * self.batch_size))  # Number of batches per epoch per replica
        self.epoch = 0
    
    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()

            # Repeat each sample k times to generate m*k total samples.
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            # Distribute samples to replicas
            for i in range(0, self.sample_num_per_epoch // self.num_replicas, self.batch_size):
                start = self.rank * (self.sample_num_per_epoch // self.num_replicas) + i
                end = start + self.batch_size
                yield repeated_indices[start:end]

    def set_epoch(self, epoch : int):
        self.epoch = epoch  # Used to synchronize random state across epochs

class DistributedKRepeatSampler(Sampler):
    """
    """
    def __init__(self, dataset : Dataset, batch_size : int, k : int, m : int, num_replicas : int, rank : int, seed : int = 0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.m = m                    # `Least` number of unique sample per epoch
        self.num_replicas = num_replicas  # Total number of replicas, process num, gpu num
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of samples for each batch iteration
        self.sample_num_per_iteration = self.num_replicas * self.batch_size
        step = self.sample_num_per_iteration // math.gcd(self.k, self.sample_num_per_iteration)
        new_m = (self.m + step - 1) // step * step  # Round up m to be multiple of step, `new_m` is the least multiple of step that is larger than `m`
        if new_m != self.m:
            logger.warning(f"Adjusted `m` from {self.m} to {new_m} to make sure `m`*`k` is multiple of `batch_size`*`num_replicas` for even distribution.")
            self.m = new_m
        
        self.num_batches_per_epoch = (self.m * self.k) // self.sample_num_per_iteration

        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()

            # Repeat each sample k times to generate m*k total samples.
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            for i in range(self.num_batches_per_epoch):
                # Offset for current iteration
                offset = i * self.sample_num_per_iteration
                # Compute start and end indices for current replica
                start = offset + self.rank * self.batch_size
                end = start + self.batch_size
                yield shuffled_samples[start:end]

    def set_epoch(self, epoch : int):
        self.epoch = epoch  # Used to synchronize random state across epochs


def test_distributed_k_repeat_sampler():
    from prompt_dataset import TextPromptDataset
    num_processes = 4 # 3
    train_batch_size = 4 # 5
    num_images_per_prompt = 13 # 2
    unique_sample_per_epoch = 1 # 13
    sample_num_per_batch = num_processes * train_batch_size
    sample_num_per_epoch = math.lcm(num_images_per_prompt * unique_sample_per_epoch, sample_num_per_batch)
    num_batches_per_epoch = int(sample_num_per_epoch // sample_num_per_batch)
    unique_sample_per_epoch = sample_num_per_epoch // num_images_per_prompt

    for var_name in ['train_batch_size', 'num_images_per_prompt', 'num_processes', 'sample_num_per_epoch', 'sample_num_per_batch', 'num_batches_per_epoch', 'unique_sample_per_epoch']:
        var_value = vars()[var_name]
        print(f"{var_name:>30}: {var_value:<10}")
    

    # assert num_batches_per_epoch % 2 == 0, f"Please set config.sample.num_batches_per_epoch {num_batches_per_epoch} to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."


    dataset = 'dataset/ocr'
    train_dataset = TextPromptDataset(dataset, 'train')
    train_samplers = [
        DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=train_batch_size,
        k=num_images_per_prompt,
        m=unique_sample_per_epoch,
        num_replicas=num_processes,
        rank=i,
        seed=0)
        for i in range(num_processes)
    ]

    train_dataloaders = [
        DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        ) for train_sampler in train_samplers
    ]

    all_samples = [[] for _ in range(num_processes)]
    epoch_num = 3
    for epoch in range(epoch_num):
        for j, sampler in enumerate(train_samplers):
            sampler.set_epoch(epoch)
            loader_iter = iter(train_dataloaders[j])
            for i in range(num_batches_per_epoch):
                samples = next(loader_iter)
                all_samples[j].extend(samples[0])

    for i in range(num_processes):
        counter = Counter(all_samples[i])
        print(f"\nProcess {i} - Total unique samples: {len(counter)} / {unique_sample_per_epoch * epoch_num}")
        print(f"Sample length {Counter(list(counter.values()))}")


def test_distributed_group_k_repeat_sampler():
    from prompt_dataset import TextPromptDataset
    num_processes = 2 # 3
    train_batch_size = 2 # 5
    num_images_per_prompt = 3 # 2
    unique_sample_per_epoch = 16 # 13
    sample_num_per_batch = num_processes * train_batch_size
    sample_num_per_epoch = math.lcm(num_images_per_prompt * unique_sample_per_epoch, sample_num_per_batch)
    num_batches_per_epoch = int(sample_num_per_epoch // sample_num_per_batch)
    unique_sample_per_epoch = sample_num_per_epoch // num_images_per_prompt

    for var_name in ['train_batch_size', 'num_images_per_prompt', 'num_processes', 'sample_num_per_epoch', 'sample_num_per_batch', 'num_batches_per_epoch', 'unique_sample_per_epoch']:
        var_value = vars()[var_name]
        print(f"{var_name:>30}: {var_value:<10}")
    

    # assert num_batches_per_epoch % 2 == 0, f"Please set config.sample.num_batches_per_epoch {num_batches_per_epoch} to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."


    dataset = 'dataset/ocr'
    train_dataset = TextPromptDataset(dataset, 'train')
    train_samplers = [
        DistributedGroupKRepeatSampler(
        dataset=train_dataset,
        batch_size=train_batch_size,
        k=num_images_per_prompt,
        m=unique_sample_per_epoch,
        num_replicas=num_processes,
        rank=i,
        seed=0)
        for i in range(num_processes)
    ]

    train_dataloaders = [
        DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        ) for train_sampler in train_samplers
    ]

    all_samples = [[] for _ in range(num_processes)]
    epoch_num = 3
    for epoch in range(epoch_num):
        for j, sampler in enumerate(train_samplers):
            sampler.set_epoch(epoch)
            loader_iter = iter(train_dataloaders[j])
            for i in range(num_batches_per_epoch):
                samples = next(loader_iter)
                all_samples[j].extend(samples[0])

    for i in range(num_processes):
        counter = Counter(all_samples[i])
        print(f"\nProcess {i} - Total unique samples: {len(counter)} / {unique_sample_per_epoch * epoch_num}")
        print(f"Sample length {Counter(list(counter.values()))}")

    # for k,v in sorted

if __name__ == "__main__":
    # test_distributed_k_repeat_sampler()
    test_distributed_group_k_repeat_sampler()