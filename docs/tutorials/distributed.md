# Distributed imaging

This tutorial describes how to run pyHyperRI for distributed hyperspectral imaging.
Distributed imaging is enabled when the distributed environment variables are
set. Each rank is mapped to one accelerator and reconstructs a subset of the effective
frequency channels, so the number of distributed processes should not exceed
`ch_effective`.

The documentation demonstrates launching distributed imaging on machines with
CUDA devices. For a single machine, `torchrun` is the simplest launcher. On
SLURM systems, `srun` with explicit PyTorch distributed environment variables
is usually more convenient because the same SLURM script can cover single-node
and multi-node jobs. Kubernetes can achieve the same functionality via Indexed
Jobs.

## What this tutorial covers

- Distributed channel partitioning in pyHyperRI.
- CUDA/NCCL configuration for hyperspectral imaging.
- Single-node launching with `torchrun`.
- Multi-node examples with SLURM `srun` and Kubernetes Indexed Jobs.
- Basic visualisation of reconstruction results.

## Tutorial data and config

This tutorial uses one hyperspectral 3C 353 measurement set with 36
frequency channels:

```text
id_00_3c353_dt_9.46_6.21_freqratio_1.70
```

It is a simulated measurement from the test set in [Tang et al. (2026)](#ref-tang-2026),
and is available from
[Zenodo](https://doi.org/10.5281/zenodo.23601176) in the archive `simulated_hyper_vla_0.zip`.

Download the archive containing this measurement folder from the Zenodo record
and extract it under `data/`.

The configuration file used in this tutorial is `config/tutorial_hs.json`. If
you are following the tutorial outside a source checkout, download the config
file from the repository:

```bash
mkdir -p config
curl -L -o config/tutorial_hs.json \
  https://raw.githubusercontent.com/ChaoTang0330/py-hyper-ri/main/config/tutorial_hs.json
```

All launch examples below assume the measurement folder is available at:

```text
data/id_00_3c353_dt_9.46_6.21_freqratio_1.70
```

## Imaging configuration file

The essential settings related to distributed imaging in the configuration file
are listed below. The tutorial uses 36 effective channels, with each data
channel corresponding to 1 image channel.

```json
{
  "main": {
    "ch_start": 1,
    "ch_last": 36,
    "ch_effective": 36
  },
  "computing": {
    "meas_device": "cuda",
    "regulariser_device": "cuda",
    "dist_backend": "nccl",
    "dist_init_method": "env://"
  }
}
```

## Channels and ranks

Physical channels are the measured frequency channels stored on disk. Effective
channels are the channels reconstructed by pyHyperRI after optional grouping.
The `ch_partition` option controls how many physical channels are concatenated
into each effective channel.

When `ch_partition` is omitted or empty, pyHyperRI groups the selected physical
channels evenly according to `ch_effective`. In this tutorial, `ch_start = 1`,
`ch_last = 36`, and `ch_effective = 36`, so the 36 physical channels become 36
effective channels with no concatenation.

Effective channels are distributed as evenly as possible across ranks. For
example, with 36 effective channels and 4 ranks, each rank reconstructs 9
effective channels. `WORLD_SIZE` must not exceed `ch_effective`.

## Backend and initialisation

PyTorch provides backend selection guidance in
[Backends](https://docs.pytorch.org/docs/stable/distributed.html#backends)
section. Its rule of thumb is to use NCCL for distributed CUDA GPU runs, XCCL
for distributed XPU GPU runs, and Gloo for distributed CPU runs. For GPU hosts
with InfiniBand or Ethernet interconnects, PyTorch recommends NCCL; Gloo can be
used as a fallback if NCCL is not available or causes problems.

As each rank will only manage 1 accelerator, the device indexes should be bound to
the local rank before distributed initialisation. With `torchrun`, users normally
expose all local GPUs and {func}`hyri.interface.imager.set_rank_device` will set the
current device from `LOCAL_RANK`. Users can also explicitly set the visible devices
for each process via corresponding environment variables. For CUDA devices, the
variable `CUDA_VISIBLE_DEVICES` can be used.

After the devices are set, {func}`hyri.interface.imager.imager` will then initialise
the PyTorch distributed environment. This tutorial uses `"env://"` as the initialisation
method, which will read necessary environment variables for distributed setup.
When `torchrun` is used, it automatically fills in these environment variables for each
local process. Otherwise, users must set these variables explicitly. See
{ref}`distributed-environment-variables` below for details.
There are also other initialisation methods, such as `"file://"` and `"tcp://"`, which require
different setups. See the PyTorch distributed documentation for details.

## Single node with torchrun

Launch imaging through the `hyri` Python module with `torchrun -m`.

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m hyri imaging \
    --config "config/tutorial_hs.json" \
    --algorithm "hyperairi" \
    --data_file "data/id_00_3c353_dt_9.46_6.21_freqratio_1.70" \
    --result_path "results/tutorial_hs"
```

The example launches four processes on one node. Replace `hyperairi` with
`usara`, `airi`, or `hyper-usara` to launch the other imaging algorithms
with the same distributed runtime.

(distributed-environment-variables)=
## Distributed environment variables

When `torchrun` is not used, the launcher must provide the environment required
by the initialisation method `"env://"`:

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
: Rendezvous endpoint used by all ranks.

## SLURM with srun

On SLURM systems, use one task per GPU and export the PyTorch distributed
environment from SLURM variables. This pattern works for both single-node and
multi-node jobs. The example below launches a distributed job with two nodes,
four ranks per node.

```bash
#!/bin/bash
#SBATCH --job-name=pyhyperri-hs
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4       # 4 ranks per node
#SBATCH --gres=gpu:4              # Assume 4 GPUs per node
#SBATCH --cpus-per-task=8         # Assume 32 CPU cores per node
#SBATCH --time=00:30:00           # Adjust walltime as needed

set -euo pipefail

export CONFIG="config/tutorial_hs.json"
export ALGORITHM="hyperairi"
export DATA_DIR="data/id_00_3c353_dt_9.46_6.21_freqratio_1.70"
export RESULT_DIR="results/tutorial_hs"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT
export PYTHONUNBUFFERED=1

# Site-specific NCCL diagnostics and network settings.
# Adjust or omit these variables as needed for the target cluster.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"

srun --hint=nomultithread --distribution=block:block bash -lc '
  set -euo pipefail

  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export NODE_RANK=$SLURM_NODEID
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  export MASTER_ADDR="'"$MASTER_ADDR"'"
  export MASTER_PORT="'"$MASTER_PORT"'"

  hyri-imaging \
    --config "$CONFIG" \
    --algorithm "$ALGORITHM" \
    --data_file "$DATA_DIR" \
    --result_path "$RESULT_DIR"
'
```

The `CUDA_VISIBLE_DEVICES` assignment restricts each SLURM task to one local
GPU. Inside that task, the visible GPU is device `cuda:0`. pyHyperRI also sets
the current CUDA device from `LOCAL_RANK` before distributed initialisation if
the script can see more than one GPU.

The NCCL settings are site-specific. On some systems the automatically selected
network path is sufficient, while on others, explicit values are required for NCCL to
use the intended interface or InfiniBand/RDMA path. In the NCCL environment variables,
`NCCL_DEBUG=INFO` prints NCCL startup information.
`NCCL_SOCKET_IFNAME` selects a network interface, and `"ib"` represents InfiniBand interfaces.
`NCCL_IB_DISABLE=0` enables InfiniBand/RDMA paths when available.
`NCCL_IB_HCA` specifies which Host Channel Adapter (RDMA) interfaces to use for communication, and normally `"mlx5"` support the full RDMA features.
`NCCL_SOCKET_NTHREADS` with `NCCL_NSOCKS_PERTHREAD` tunes the CPUs required for communication.
Replace the example interface and adapter values with values appropriate for the target cluster.
See the [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
for details on these and other environment variables.

## Kubernetes with torchrun

For Kubernetes, a flexible pattern is to let each pod run `torchrun`. Each pod
may request one or more GPUs, and `torchrun` starts one pyHyperRI process per
local GPU in that pod. The number of pods becomes `--nnodes`; the number of
GPUs requested by each pod becomes `--nproc-per-node`.

The minimal example below uses a Kubernetes Indexed Job with two pods and four
GPUs per pod. The built-in `JOB_COMPLETION_INDEX` becomes the `torchrun`
`--node-rank`. Use scheduling policy, affinity, or cluster-specific controls if
pods must be placed on distinct physical nodes. The headless Service is a
separate Kubernetes Service object that enables deterministic pod DNS names for
rendezvous. In this example, rank 0 is reached at `pyhyperri-0.pyhyperri`.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: pyhyperri
spec:
  clusterIP: None
  publishNotReadyAddresses: true
  selector:
    batch.kubernetes.io/job-name: pyhyperri
  ports:
    - name: dist
      port: 29500
      targetPort: 29500
---
apiVersion: batch/v1
kind: Job
metadata:
  name: pyhyperri
spec:
  completions: 2
  parallelism: 2
  completionMode: Indexed
  template:
    spec:
      subdomain: pyhyperri
      restartPolicy: Never
      containers:
        - name: pyhyperri
          image: <PYHYPERRI_IMAGE>
          workingDir: /mnt/pyHyperRI
          resources:
            limits:
              nvidia.com/gpu: 4
          env:
            - name: NNODES
              value: "2"
            - name: GPUS_PER_NODE
              value: "4"
            - name: MASTER_ADDR
              value: "pyhyperri-0.pyhyperri"
            - name: MASTER_PORT
              value: "29500"
          command:
            - bash
            - -lc
            - |
              set -euo pipefail

              torchrun \
                --nnodes="${NNODES}" \
                --nproc-per-node="${GPUS_PER_NODE}" \
                --node-rank="${JOB_COMPLETION_INDEX}" \
                --master-addr="${MASTER_ADDR}" \
                --master-port="${MASTER_PORT}" \
                -m hyri imaging \
                  --config "config/tutorial_hs.json" \
                  --algorithm "hyperairi" \
                  --data_file "data/id_00_3c353_dt_9.46_6.21_freqratio_1.70" \
                  --result_path "results/tutorial_hs"
          volumeMounts:
            - name: tutorial
              mountPath: /mnt/pyHyperRI
      volumes:
        - name: tutorial
          persistentVolumeClaim:
            claimName: <DATA_PVC>
```

This single YAML file contains both the Service and the Job, separated by
`---`. They can be applied together with `kubectl apply -f <file>.yaml`. 
It shows one common storage pattern with persistent volume claims mounted 
at `./pyHyperRI` under the container working directory. 
Depending on the cluster DNS setup, you may
need to use the fully qualified name
`pyhyperri-0.pyhyperri.<namespace>.svc.cluster.local` for `MASTER_ADDR`.

## Visualise the results

After imaging finishes, pyHyperRI writes final products under the effective
output directory:

```text
<result_path>/<src_name>/
```

For the commands in this tutorial, the default `src_name` is the measurement
folder name, so the expected output directory is:

```text
results/tutorial_hs/id_00_3c353_dt_9.46_6.21_freqratio_1.70
```

The following scripts can be used to visualise the results. Here we show the dirty image,
reconstruction, and residual dirty image for the first and last channels, as well as 
the spectral index map fit from the reconstructed hyperspectral cube.

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

result_dir = Path(
    "./results/tutorial_hs/"
    "id_00_3c353_dt_9.46_6.21_freqratio_1.70"
)


def read_cube(path):
    return np.squeeze(np.asarray(fits.getdata(path), dtype=float))


def rlog(x, a=1e4, x_max=1.0):
    return x_max * np.log10(a * x / x_max + 1.0) / np.log10(a)


dirty = read_cube(result_dir / "dirty.fits")
model = read_cube(next(result_dir.glob("*model_image.fits")))
residual = read_cube(next(result_dir.glob("*residual_dirty_image.fits")))
psf = read_cube(next(result_dir.glob("*psf.fits")))
spec_idx = np.squeeze(fits.getdata(next(result_dir.glob("*spectral_index_map.fits"))))

psf_peak = np.max(psf)
dirty /= psf_peak
residual /= psf_peak

channels = [0, model.shape[0] - 1]
channel_names = ["Channel 1", "Channel 36"]
rows = [
    ("Dirty image", dirty, "twilight_shifted", -1, 20, False),
    ("Reconstruction", model, "magma", 0.0, 1.0, True),
    ("Residual", residual, "twilight_shifted", -6e-3, 6e-3, False),
]

fig, axes = plt.subplots(3, 2, figsize=(8, 9), constrained_layout=True)
for i, (name, cube, cmap, vmin, vmax, use_log) in enumerate(rows):
    for j, channel in enumerate(channels):
        ax = axes[i, j]
        image = np.nan_to_num(cube[channel], nan=0.0)
        if use_log:
            image = rlog(np.clip(image, 0.0, None), a=1e4, x_max=1.0)
        im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}, {channel_names[j]}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

fig.savefig("tutorial_dist_results.svg", dpi=180)

fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
im = ax.imshow(spec_idx, origin="lower", cmap="rainbow", vmin=-6.0, vmax=0.0)
ax.set_title("Spectral index map")
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
fig.savefig("tutorial_dist_spec_idx.svg", dpi=180)
```

The generated images are shown below. The reconstructed images are in logarithmic 
scale, while others are in linear scale. The dirty and residual dirty images are
normalized by the PSF peak.

```{figure} ../_images/tutorials/tutorial_dist_results.svg
:alt: HyperAIRI reconstruction results.

The dirty images, reconstructed images, and residual dirty image for the first and last
of the HyperAIRI reconstruction results.
```

```{figure} ../_images/tutorials/tutorial_dist_spec_idx.svg
:alt: Spectral index map of the HyperAIRI reconstruction.

The spectral-index map estimated from the reconstructed hyperspectral cube of HyperAIRI.
```

## Summary and further reading

Use `torchrun` for the simplest single-machine CUDA run. Use `srun` on SLURM
systems when you want one launch script for single-node and multi-node jobs.
On Kubernetes or another scheduler, pyHyperRI can run with the same PyTorch
distributed environment as long as each rank can reach the rendezvous endpoint
and the selected data paths are visible to every process.

Further reading:

- [PyTorch torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)
- [PyTorch distributed package](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch CUDA environment variables](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)
- [Kubernetes Indexed Jobs](https://kubernetes.io/docs/tasks/job/indexed-parallel-processing-static/)
- [Kubernetes Job with pod-to-pod communication](https://kubernetes.io/docs/tasks/job/job-with-pod-to-pod-communication/)

## References

(ref-tang-2026)=
- Tang, C., Dabbech, A., Jackson, A., & Wiaux, Y. (2026).
  [HyperAIRI: A Plug-and-play Algorithm for Precise Hyperspectral Image
  Reconstruction in Radio Interferometry](https://doi.org/10.3847/1538-4365/ae3149).
  *ApJS*, 283(1), 9.
