# Distributed Imaging

This tutorial collects distributed-launch examples for pyHyperRI. It is a
starting point for future multi-node and scheduler-specific recipes.

pyHyperRI uses `torch.distributed` when `WORLD_SIZE` is greater than `1`.
Distributed runs partition effective channels across ranks, so the number of
processes must not exceed `ch_effective`.

## Single Node

For a single node with multiple processes, use a PyTorch launcher such as
`torchrun`:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m hyri.cli \
  --config config/example_hs_short.json \
  --algorithm hyperairi \
  --data_file <DATA_DIRECTORY> \
  --result_path results
```

The example above launches four ranks on one node. For GPU runs, this usually
means one rank per GPU. Make sure `ch_effective` is at least the number of
processes.

## Distributed Environment

The runtime reads these distributed environment variables:

`WORLD_SIZE`:
: Total number of ranks.

`RANK`:
: Global rank of the current process.

`LOCAL_WORLD_SIZE`:
: Number of ranks on the current node.

`LOCAL_RANK`:
: Rank index on the current node.

`NODE_RANK`:
: Node index in a multi-node run.

`MASTER_ADDR`, `MASTER_PORT`:
: Rendezvous endpoint used by `dist_init_method: "env://"`.

The distributed backend and initialisation method are configured in the
`computing` section with `dist_backend` and `dist_init_method`. Use `nccl` for
CUDA multi-GPU runs when available, otherwise use `gloo`.

## SLURM

On SLURM systems, launch one process per GPU with `srun`, set the PyTorch
distributed environment variables from SLURM variables, and run the module
entry point:

```bash
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT

srun --hint=nomultithread --distribution=block:block bash -lc '
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export NODE_RANK=$SLURM_NODEID
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

  python -m hyri.cli \
    --config config/example_hs.json \
    --algorithm hyperairi \
    --data_file <DATA_DIRECTORY> \
    --result_path results
'
```

Repository SLURM examples are available in `scripts/test_hs.slurm`,
`scripts/hyperairi_single.slurm`, and `scripts/hyperairi_array.slurm`.
