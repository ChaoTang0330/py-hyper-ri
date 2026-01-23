#!/usr/bin/env python3
"""
Two-node torch.distributed + NCCL demo (default stream only)

What it shows:
- init_process_group (env://)
- rank / world_size / local_rank / node info printing
- point-to-point: node 1 GPU 0 -> node 0 GPU 0 (typical rank mapping: rank 4 -> rank 0)
- broadcast: rank 0 -> all ranks
- all_reduce: SUM across all ranks
- barrier
- destroy_process_group

Typical launch (2 nodes, 4 GPUs per node):

Node 0:
  torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=<node0_addr> --master_port=29500 dist_2node_nccl_demo.py

Node 1:
  torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
    --master_addr=<node0_addr> --master_port=29500 dist_2node_nccl_demo.py

Assumption about rank mapping:
- With torchrun and nproc_per_node=4, ranks are usually contiguous per node:
  node 0: ranks 0..3   (GPU 0..3)
  node 1: ranks 4..7   (GPU 0..3)
So "node 1 GPU 0" corresponds to global rank 4 and local_rank 0.
"""

import os
import socket
import torch
import torch.distributed as dist


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def main() -> None:
    rank = env_int("RANK", 0)
    world_size = env_int("WORLD_SIZE", 1)
    local_rank = env_int("LOCAL_RANK", 0)
    local_world_size = env_int("LOCAL_WORLD_SIZE", 1)
    node_rank = env_int("NODE_RANK", 0)

    # Bind process to one GPU
    torch.cuda.set_device(0)
    device = torch.device("cuda")

    host = socket.gethostname()

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda:0"),
    )

    # Small header
    dist.barrier()
    if rank == 0:
        print("=== dist demo: 2 nodes / NCCL / default stream only ===", flush=True)
    dist.barrier()

    print(
        f"[rank {rank}/{world_size} | node_rank {node_rank} | "
        f"local_rank {local_rank}/{local_world_size} | {host}] init ok",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # 1) Point-to-point: rank 4 -> rank 0 (node 1 GPU 0 -> node 0 GPU 0)
    # -------------------------------------------------------------------------
    x = torch.full((4,), float(rank), device=device)

    if world_size >= 2:
        sender = 4
        receiver = 0
        if rank == sender:
            dist.send(tensor=x, dst=receiver)
            print(f"[rank {rank} | {host}] sent x={x.tolist()} -> rank {receiver}", flush=True)

        if rank == receiver:
            y = torch.empty_like(x, device=device)
            dist.recv(tensor=y, src=sender)
            print(f"[rank {rank} | {host}] received y={y.tolist()} <- rank {sender}", flush=True)
    else:
        if rank == 0:
            print("[rank 0] world_size=1, skipping send/recv", flush=True)

    dist.barrier()

    # -------------------------------------------------------------------------
    # 2) Broadcast: rank 0 -> all ranks
    # -------------------------------------------------------------------------
    b = torch.zeros(3, device=device)
    if rank == 0:
        b[:] = torch.tensor([10.0, 20.0, 30.0], device=device)

    dist.broadcast(b, src=0)
    print(f"[rank {rank} | {host}] after broadcast b={b.tolist()}", flush=True)

    dist.barrier()

    # -------------------------------------------------------------------------
    # 3) All-reduce SUM: every rank ends with the same scalar
    # -------------------------------------------------------------------------
    a = torch.tensor([float(rank)], device=device)
    dist.all_reduce(a, op=dist.ReduceOp.SUM)

    # Expected sum = world_size*(world_size-1)/2
    expected = world_size * (world_size - 1) / 2
    print(f"[rank {rank} | {host}] after all_reduce a={a.item()} (expected {expected})", flush=True)

    dist.barrier()

    # -------------------------------------------------------------------------
    # 5) Shutdown
    # -------------------------------------------------------------------------
    if rank == 0:
        print("=== done ===", flush=True)
    
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()