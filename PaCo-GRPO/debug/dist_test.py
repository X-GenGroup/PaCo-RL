import torch
from accelerate import Accelerator
import torch.distributed as dist
import os

def main():
    accelerator = Accelerator()
    if accelerator.process_index == 0:
        tensor_list = [
            torch.randn((3, 2, 1), device=accelerator.device)
            for _ in range(3)
        ]
    else:
        tensor_list = [
            torch.randn((3, 1, 1), device=accelerator.device),
            torch.randn((3, 4, 1), device=accelerator.device),
            torch.randn((3, 2, 2), device=accelerator.device),
        ]

    tensor_dim = tensor_list[0].dim()
    tensor_dtype = tensor_list[0].dtype
    device = 'cpu'

    # Step 1: Gather lengths of tensor_list from all ranks
    local_length = torch.tensor([len(tensor_list)], device=accelerator.device, dtype=torch.long)
    gathered_lengths = [torch.zeros(1, dtype=torch.long, device=accelerator.device) for _ in range(accelerator.num_processes)]
    dist.gather(gathered_lengths, local_length)
    gathered_lengths = [int(length.item()) for length in gathered_lengths]

    # Step 2: Gather shapes of each tensor in tensor_list from all ranks
    local_shapes = torch.tensor([list(t.shape) for t in tensor_list], device=accelerator.device, dtype=torch.long)
    gathered_shapes = [
        torch.zeros((length, tensor_dim), dtype=torch.long, device=accelerator.device)
        for length in gathered_lengths
    ]
    dist.all_gather(gathered_shapes, local_shapes)

    # Compute the total length of flattened tensors for each rank, [rank0_total_length, rank1_total_length, ...]
    flat_lengths = [
        sum(int(shape.prod().item()) for shape in this_rank_shapes)
        for this_rank_shapes in gathered_shapes
    ]

    # Step 3: Gather all tensors by flattening and concatenating
    local_flat_tensor = torch.cat([t.flatten() for t in tensor_list], dim=0)
    gathered_flat_tensor = [
        torch.zeros(length, dtype=tensor_dtype, device=accelerator.device)
        for length in flat_lengths
    ]
    dist.all_gather(gathered_flat_tensor, local_flat_tensor)

    # Step 4: Reconstruct the original tensors from gathered shapes and flattened tensors
    gathered_tensor = []
    for rank, (this_rank_shapes, this_rank_flat_tensor) in enumerate(zip(gathered_shapes, gathered_flat_tensor)):
        offset = 0
        for shape in this_rank_shapes:
            length = int(shape.prod().item())
            this_tensor = this_rank_flat_tensor[offset:offset+length].reshape(shape.tolist()).to(device)
            gathered_tensor.append(this_tensor)
            offset += length

    print(f"Process {accelerator.process_index} gathered: {[t.shape for t in gathered_tensor]}\n")


if __name__ == "__main__":
    main()