# Distributed imaging

This tutorial describes how to run pyHyperRI with `torch.distributed`.
Distributed imaging is enabled when `WORLD_SIZE` is greater than `1`. Each rank
reconstructs a subset of the effective frequency channels, so the number of
distributed processes must not exceed `ch_effective`.

The documented and tested path is CUDA with one process per GPU. For a single
machine, `torchrun` is the simplest launcher. On SLURM systems, `srun` with
explicit PyTorch distributed environment variables is usually more convenient
because the same script can cover single-node and multi-node jobs. Other
systems, including Kubernetes, can run pyHyperRI as long as they provide the
same distributed runtime information.

## Tutorial data and config

The examples below use one hyperspectral 3C353 measurement set with 36
frequency channels:

```text
3c353_uvw_id_00_dt_9.46_6.21_freqratio_1.70_nfreq_36_rotation_330.40
```

The tutorial dataset will be available from Zenodo:

```text
<ZENODO_LINK_TODO>
```

Download the archive and unzip it under `data/`:

```bash
mkdir -p data
curl -L -o data/pyhyperri_tutorial_data.zip <ZENODO_LINK_TODO>
unzip data/pyhyperri_tutorial_data.zip -d data/
```

The configuration file used in this tutorial is `config/tutorial_hs.json`. If
you are following the tutorial outside a source checkout, download the config
file from the repository:

```bash
curl -L -o config/tutorial_hs.json \
  https://raw.githubusercontent.com/ChaoTang0330/small-scale-ri-imaging-hs/main/config/tutorial_hs.json
```

All launch examples below assume the measurement folder is available at:

```text
data/3c353_uvw_id_00_dt_9.46_6.21_freqratio_1.70_nfreq_36_rotation_330.40
```

## Imaging configuration file

Only a small part of `config/tutorial_hs.json` is essential for distributed
execution. The essential settings are listed in the JSON excerpt below. The
tutorial uses 36 effective channels and does not set `ch_partition`, so
pyHyperRI uses the default empty partition and each physical frequency channel
maps to one effective channel.

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

Use `nccl` for CUDA distributed imaging. Other PyTorch initialisation methods,
such as `tcp://` and `file://`, can be used through `dist_init_method`; see the
PyTorch distributed documentation for details.

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

## Backend and device behaviour

PyTorch provides backend selection guidance in its
[Which backend to use?](https://docs.pytorch.org/docs/stable/distributed.html#which-backend-to-use)
section. Its rule of thumb is to use NCCL for distributed CUDA GPU runs, XCCL
for distributed XPU GPU runs, and Gloo for distributed CPU runs. For GPU hosts
with InfiniBand or Ethernet interconnects, PyTorch recommends NCCL; Gloo can be
used as a fallback if NCCL is not available or causes problems.

pyHyperRI sets the current CUDA device from the local rank before accelerator
tensors are created, so unindexed devices such as `torch.device("cuda")`
resolve to the rank-local GPU. With `torchrun`, users normally expose all local
GPUs and let `torchrun` provide the local rank. With scheduler-managed launches
such as SLURM, it is also common to restrict each process with
`CUDA_VISIBLE_DEVICES`.

The backend also determines which tensor devices can be used for distributed
collectives. pyHyperRI currently places its internal distributed coordination
tensors on CUDA for `nccl`, XPU for `xccl`, and CPU for other backends:

`nccl`:
: Uses CUDA tensors.

`gloo`:
: PyTorch supports CPU collectives and can support GPU collectives with Gloo,
  but NCCL is recommended for CUDA GPU runs. pyHyperRI documents Gloo mainly as
  a CPU and debugging fallback.

`mpi`:
: Depends on the MPI-enabled PyTorch build. PyTorch must be built from source
  with the target MPI implementation if this backend is needed; MPI is not
  necessarily CPU-only.

`xccl`:
: Uses XPU tensors. XPU/XCCL support is provided by PyTorch, but it has not
  been fully tested with pyHyperRI, so this tutorial focuses on CUDA/NCCL.

See PyTorch's
[Backends](https://docs.pytorch.org/docs/stable/distributed.html#backends)
section for backend capabilities and build requirements.

## Single node with torchrun

The `hyri-imaging` command can be launched directly with `torchrun`.
`torchrun` automatically sets the necessary PyTorch distributed environment
variables for each local process.

```bash
torchrun --standalone --nproc-per-node=4 \
  hyri-imaging \
  --config config/tutorial_hs.json \
  --algorithm hyperairi \
  --data_file data/3c353_uvw_id_00_dt_9.46_6.21_freqratio_1.70_nfreq_36_rotation_330.40 \
  --result_path results/tutorial_hs
```

The example launches four processes on one node. For CUDA runs, this usually
means one process per local GPU. Replace `hyperairi` with `usara`, `airi`, or
`hyper-usara` to launch the other imaging algorithms with the same distributed
runtime.

## Distributed environment

When `torchrun` is not used, the launcher must provide the environment required
by `dist_init_method: "env://"`:

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
multi-node jobs.

```bash
#!/bin/bash
#SBATCH --job-name=pyhyperri-hs
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

set -euo pipefail

export CONFIG="config/tutorial_hs.json"
export ALGORITHM="hyperairi"
export DATA_DIR="data/3c353_uvw_id_00_dt_9.46_6.21_freqratio_1.70_nfreq_36_rotation_330.40"
export RESULT_DIR="results/tutorial_hs"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT
export PYTHONUNBUFFERED=1

# Site-specific NCCL diagnostics and network settings.
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
GPU. Inside that task, the visible GPU is device `cuda:0`; pyHyperRI also sets
the current CUDA device from `LOCAL_RANK` before distributed initialisation.

The NCCL settings are site-specific. On some systems the automatically selected
network path is sufficient; on others, explicit values are required for NCCL to
use the intended interface or InfiniBand/RDMA path. `NCCL_DEBUG=INFO` prints
NCCL startup information, `NCCL_SOCKET_IFNAME` selects a network interface,
`NCCL_IB_DISABLE=0` enables InfiniBand/RDMA paths when available, `NCCL_IB_HCA`
selects host channel adapters, and `NCCL_SOCKET_NTHREADS` with
`NCCL_NSOCKS_PERTHREAD` tunes socket transport. Replace the example interface
and adapter values with values appropriate for the target cluster.

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
rendezvous; it is not the Job itself. In this example, rank 0 is reached at
`pyhyperri-0.pyhyperri`.

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
          workingDir: /workspace
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
                hyri-imaging \
                  --config config/tutorial_hs.json \
                  --algorithm hyperairi \
                  --data_file data/3c353_uvw_id_00_dt_9.46_6.21_freqratio_1.70_nfreq_36_rotation_330.40 \
                  --result_path results/tutorial_hs
          volumeMounts:
            - name: tutorial-data
              mountPath: /workspace/data
            - name: tutorial-results
              mountPath: /workspace/results
      volumes:
        - name: tutorial-data
          persistentVolumeClaim:
            claimName: <DATA_PVC>
        - name: tutorial-results
          persistentVolumeClaim:
            claimName: <RESULTS_PVC>
```

This single YAML file contains both the Service and the Job, separated by
`---`; they can be applied together with `kubectl apply -f <file>.yaml`. The
manifest intentionally omits image pull configuration, node selectors,
tolerations, and security policy. It shows one common storage pattern with
persistent volume claims mounted at `./data` and `./results` under the container
working directory. If `config/tutorial_hs.json` is not packaged into the image,
mount the config directory as well. Depending on the cluster DNS setup, you may
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
results/tutorial_hs/3c353_uvw_id_00_dt_9.46_6.21_freqratio_1.70_nfreq_36_rotation_330.40
```

Use the notebook-style code below to plot the first and last channel of the
reconstruction, dirty image, and residual dirty image, plus the spectral-index
map. The code also saves the figures used by this documentation page.

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


result_dir = Path(
    "results/tutorial_hs/"
    "3c353_uvw_id_00_dt_9.46_6.21_freqratio_1.70_nfreq_36_rotation_330.40"
)
figure_dir = Path("docs/_images/tutorials")
figure_dir.mkdir(parents=True, exist_ok=True)


def read_cube(path):
    cube = np.squeeze(np.asarray(fits.getdata(path), dtype=float))
    return cube[None, :, :] if cube.ndim == 2 else cube


def log_display(image, vmin=0.0, vmax=1.0):
    image = np.nan_to_num(image, nan=0.0, posinf=vmax, neginf=vmin)
    image = np.clip(image, vmin, vmax)
    image = (image - vmin) / (vmax - vmin)
    return np.log10(1.0 + 9.0 * image)


dirty = read_cube(result_dir / "dirty.fits")
model = read_cube(next(result_dir.glob("*model_image.fits")))
residual = read_cube(next(result_dir.glob("*residual_dirty_image.fits")))
spec_idx = np.squeeze(fits.getdata(next(result_dir.glob("*spectral_index_map.fits"))))

channels = [0, model.shape[0] - 1]
channel_names = ["first channel", "last channel"]
rows = [
    ("Reconstruction", model, "magma", True),
    ("Dirty image", dirty, "magma", True),
    ("Residual", residual, "twilight_shifted", False),
]

fig, axes = plt.subplots(3, 2, figsize=(8, 9), constrained_layout=True)
for i, (name, cube, cmap, use_log) in enumerate(rows):
    for j, channel in enumerate(channels):
        ax = axes[i, j]
        if use_log:
            image = log_display(cube[channel], vmin=0.0, vmax=1.0)
            im = ax.imshow(image, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            im = ax.imshow(
                cube[channel],
                origin="lower",
                cmap=cmap,
                vmin=-6e-3,
                vmax=6e-3,
            )
        ax.set_title(f"{name}, {channel_names[j]}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

fig.savefig(figure_dir / "distributed_channel_diagnostics.svg", dpi=180)

fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
im = ax.imshow(spec_idx, origin="lower", cmap="rainbow", vmin=-6.0, vmax=0.0)
ax.set_title("Spectral index map")
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="spectral index")
fig.savefig(figure_dir / "distributed_spectral_index.svg", dpi=180)

plt.show()
```

The reconstruction and dirty-image panels are clipped to `[0, 1]` and displayed
with the log transform `log10(1 + 9I)`, so the displayed range is `[0, 1]`.
The residual panels use the shifted twilight colour map with range
`[-6e-3, 6e-3]`. The spectral-index map uses the rainbow colour map with range
`[-6, 0]`, where `0` maps to red and `-6` maps to purple.

```{figure} ../_images/tutorials/distributed_channel_diagnostics.svg
:alt: Placeholder for distributed channel diagnostics.

First and last channel diagnostics for the reconstruction, dirty image, and
residual dirty image.
```

```{figure} ../_images/tutorials/distributed_spectral_index.svg
:alt: Placeholder for the distributed tutorial spectral-index map.

Spectral-index map estimated from the reconstructed hyperspectral cube.
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
