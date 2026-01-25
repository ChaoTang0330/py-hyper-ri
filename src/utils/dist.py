import numpy as np
import torch
from torch import distributed as dist


def gather_ch_to_tensor(
    tensor_ch_rank: torch.Tensor, dest: int
) -> torch.Tensor:
    """
    Gather tensors have different number of channels from all ranks to the tensor at destination rank.
    Args:
        tensor_ch_rank (torch.Tensor): Tensor on each rank with shape (1, num_ch_rank, H, W).
        dest (int): Destination rank to gather tensors.
    Returns:
        torch.Tensor: Gathered tensor on the destination rank with shape (1, total_num_ch, H, W)
            at the destination rank, None on other ranks.
    """

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed is not available or not initialized")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_ch_rank = tensor_ch_rank.shape[1]
    num_ch_each_rank = torch.zeros(world_size, dtype=torch.int64, device=tensor_ch_rank.device)
    dist.all_gather_into_tensor(num_ch_each_rank, torch.tensor([num_ch_rank], device=tensor_ch_rank.device))
    num_ch_each_rank = num_ch_each_rank.cpu().numpy().tolist()

    max_num_ch = max(num_ch_each_rank)
    if tensor_ch_rank.shape[1] < max_num_ch:
        tensor_pad = torch.cat(
            [
                tensor_ch_rank,
                tensor_ch_rank.new_zeros(
                    1,
                    max_num_ch - tensor_ch_rank.shape[1],
                    tensor_ch_rank.shape[2],
                    tensor_ch_rank.shape[3],
                ),
            ],
            dim=1,
        )
    else:
        tensor_pad = tensor_ch_rank

    gather_list = (
        [torch.empty_like(tensor_pad) for _ in range(world_size)]
        if rank == dest
        else None
    )
    dist.gather(tensor_pad, gather_list=gather_list, dst=dest)

    if rank != dest:
        return None

    tensor_full = torch.cat(
        [gather_list[r][:, : num_ch_each_rank[r], :, :] for r in range(world_size)],
        dim=1,
    )

    return tensor_full
