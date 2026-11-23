# Usage

## Overview

pyHyperRI provides a top-level `hyri` command with subcommands for the main
workflows:

```bash
hyri -h
hyri imaging --help
hyri config --help
hyri ms2mat --help
```

The standalone commands `hyri-imaging`, `hyri-config`, and `hyri-ms2mat` are
also available for compatibility. For example, `hyri imaging` and
`hyri-imaging` launch the same imaging workflow.

After the input data is prepared, launch imaging jobs with `hyri imaging`.
If no configuration file is provided, the command reads the default
configuration bundled with the package. Command-line overrides are available
for common run settings such as the input data directory, output directory,
algorithm, image size, denoiser shelf, channel range, and run ID.

## Run an Imaging Job

Run an imaging algorithm with the packaged default configuration:

```bash
hyri imaging --data_file <DATA_DIRECTORY>
```

Run with an explicit configuration file:

```bash
hyri imaging --config <CONFIG_FILE>
```

`--config` can also be written as `-c`. If it is omitted, pyHyperRI reads the
bundled `hyri/config/default.json` template. See [Configuration](configuration)
for details on the configuration file generations and contents.

`hyri imaging` also accepts the following command-line arguments to override the
corresponding default imaging settings or the ones in the provided configuration
file:

`--src_name`:
: Source or experiment tag used for the output subdirectory name.

`--data_file`:
: Path to the input measurement directory containing `msSpecs.mat` and
  `data_ch_*.mat` files.

`--result_path`:
: Base directory where the source-specific output directory is created.

`--algorithm`:
: Imaging algorithm. Valid values are `usara`, `hyper-usara`, `airi`,
  `cairi`, and `hyperairi`.

`--im_dim_x`, `--im_dim_y`:
: Reconstructed image width and height in pixels.

`--im_pixel_size`, `--superresolution`:
: Image pixel size in arcsec, or the super-resolution factor used to derive it
  when the pixel size is omitted.

`--dnn_shelf_path`:
: Denoiser shelf CSV for `airi`, `cairi`, and `hyperairi`.

`--ch_start`, `--ch_last`, `--ch_effective`, `--ch_partition`:
: Channel-selection and channel-grouping controls.

`--run_id`:
: Optional identifier appended to output file prefixes.

(run-in-a-container)=
## Run in a Container

See {ref}`Docker Images <docker-images>` for the available images and
GPU requirements. The runtime image already includes pyHyperRI, so a source
checkout is not needed to run imaging jobs.

The same images can also run with [Podman](https://docs.podman.io/en/latest/markdown/podman-run.1.html)
or in [Kubernetes Pods](https://kubernetes.io/docs/concepts/containers/images/),
with volume mounts and GPU access configured for the chosen environment.

Both images start Bash. Use `-it` for an interactive shell, or pass
`-c 'hyri ...'` after the image name to run a command directly.

For example, to run an imaging pipeline defined in a configuration file, one can use:

```bash
docker run --rm --gpus all --shm-size=2g \
  --mount type=bind,source="$(pwd)",target=/workspace \
  ghcr.io/chaotang0330/py-hyper-ri:v1.0.1-runtime \
  -c 'hyri imaging --config "<CONFIG_FILE>"'
```

`$(pwd)` is your current host directory, mounted at `/workspace` inside the container.
Replace `<CONFIG_FILE>` with the configuration file's path inside the container.
Set `result_path` to a directory under `/workspace`, so outputs remain on the host
after the container exits.

## Select Channels

pyHyperRI can load a range of physical channels and group them into effective
channels for hyperspectral imaging.

`ch_start`:
: First physical channel to load. Channel numbering is 1-based.

`ch_last`:
: Last physical channel to load, inclusive. If omitted, all available channels
  from `ch_start` onward are considered.

`ch_effective`:
: Number of effective channels after grouping. When `ch_partition` is empty,
  the selected physical channels are divided evenly into this many effective
  channels.

`ch_partition`:
: List of group sizes mapping selected physical channels to effective channels.
  The sum of the entries must equal the number of selected physical channels.
  If provided, its length sets `ch_effective`.

The frequency of each effective channel is the mean of the physical-channel
frequencies in that group. In distributed execution, each rank loads only its
assigned effective-channel slice, and the loader records the global effective
channel offset for that rank.

If `ch_partition` is not provided, the loader uses equal-size groups. Any
remaining physical channels that do not fit into equal groups are not loaded.

Examples:

```bash
# Load physical channels 1 through 36 and reconstruct 36 effective channels.
hyri imaging \
  --data_file <DATA_DIRECTORY> \
  --ch_start 1 \
  --ch_last 36 \
  --ch_effective 36

# Group 36 physical channels into 4 effective channels of 9 physical channels each.
hyri imaging \
  --data_file <DATA_DIRECTORY> \
  --ch_start 1 \
  --ch_last 36 \
  --ch_partition 9 9 9 9
```

(convert-measurement-sets)=
## Convert Measurement Sets

Measurement Set data can be converted into the `.mat` layout expected by
pyHyperRI with `hyri ms2mat`. Install Measurement Set support first; see
{ref}`Measurement Set Support <measurement-set-support>`.

Inspect the available spectral windows and frequency-channel grouping plan
without writing files:

```bash
hyri ms2mat path/to/target.ms --dry_run
```

Convert an MS table:

```bash
hyri ms2mat path/to/target.ms --output_path data --src_name target --src_id 0
```

The converter writes:

```text
data/target/
  msSpecs.mat
  data_ch_1.mat
  data_ch_2.mat
  ...
```

The `--src_id` option selects the Measurement Set `FIELD_ID`. The default value is `0`.
If `--src_name` is omitted, the output folder is derived from the MS filename
with a trailing `.ms` suffix removed. Channels from all spectral windows are
flattened, grouped by shared observation frequency, and sorted from low to
high, so `data_ch_1.mat` corresponds to the lowest grouped frequency. If
different spectral windows contain the same frequency, their visibilities are
merged into the same output channel.

Use `--freq_tol_hz` to merge nearly equal frequencies. The converter caps the
effective tolerance below the smallest frequency spacing within any single
spectral window so neighbouring channels in one spectral window are not merged.

Use `--ch_start` and `--ch_last` to convert a 1-based inclusive subset after
frequency grouping and sorting. The converter still stores the full grouped
frequency list in `msSpecs.mat`; partial conversion only limits which
`data_ch_<index>.mat` files are written. File indices remain global, so
`--ch_start 5` writes `data_ch_5.mat`.

The shared `uvw` array in `msSpecs.mat` contains all rows for the selected
field, not only rows from the selected spectral windows, so the metadata file is
stable when `--ch_start` or `--ch_last` changes. Per-channel files mark
non-contributing rows as flagged.

By default, the converter reads `CORRECTED_DATA` and falls back to `DATA` when
`CORRECTED_DATA` is unavailable. It extracts Stokes I from the first and last
correlations using their natural weights, stores `weightsNat` as
`sqrt(w_first + w_last)`, applies `FLAG`, `FLAG_ROW`, invalid-data, non-positive
weight, and `--min_uv` baseline filtering, and writes MS imaging weights when
available.

## Use AIRI Denoisers

Packaged AIRI denoisers can be selected through the imaging configuration.

The packaged denoiser shelf CSV files are installed as package data under
`hyri/airi_denoisers`. In the source tree, they are located at
`src/hyri/airi_denoisers`:

- `shelf_airi_oaid.csv`
- `shelf_airi_mrid.csv`
- `shelf_airi_mix.csv`
- `shelf_hyperairi_mix.csv`

For AIRI family of algorithms, set `dnn_shelf_path` in the matching algorithm
section of the configuration file, or override it from the command line:

```bash
hyri imaging \
  --algorithm airi \
  --data_file <DATA_DIRECTORY> \
  --dnn_shelf_path shelf_airi_oaid.csv
```

In addition to the packaged shelves, users can create custom shelves with
compatible ONNX denoisers. Custom denoisers are supported by providing a 
shelf CSV maps training noise levels to ONNX model files:

```text
1e-4,path/to/denoiser_sigma_0p001.onnx
5e-5,path/to/denoiser_sigma_0p002.onnx
```
where the first column is the training noise level and the second column
is the path to the ONNX file. The path of shelf CSV and ONNX files in them
can be provided as an absolute path or a path relative to the current working 
directory.

## Outputs

Imaging jobs write reconstructed images and related output files to the output
directory configured for the job.

The effective output directory is:

```text
<result_path>/<src_name>/
```

If `src_name` is omitted, pyHyperRI uses the input data directory name. If
`result_path` is omitted, it uses `./results`.

Every imaging run writes:

`dirty.fits`:
: Dirty image formed from the adjoint measurement operator.

`psf.fits`:
: Point spread function used by the optimiser.

`<PREFIX>model_image.fits`:
: Final reconstructed image or image cube.

`<PREFIX>residual_dirty_image.fits`:
: Residual dirty image after reconstruction.

Hyperspectral outputs also write:

`<PREFIX>spectral_index_map.fits`:
: Spectral-index map estimated from the reconstructed cube.

AIRI, cAIRI, and HyperAIRI can also save intermediate snapshots when
`itr_save` is less than or equal to `im_max_itr`:

`<PREFIX>tmp_model_itr_<N>.fits`:
: Model-image snapshot at iteration `N`.

`<PREFIX>tmp_residual_itr_<N>.fits`:
: Residual-image snapshot at iteration `N`.

When `reweighting_save` is enabled, uSARA and Hyper-uSARA save intermediate
reweighting products during outer iterations.

Output file prefixes encode the algorithm and the main scale parameter:

```text
AIRI_heuScale_<value>_
cAIRI_heuScale_<value>_
uSARA_heuRegScale_<value>_
HyperAIRI_heuScale_<value>_
Hyper-uSARA_heuRegScale_<value>_
```

If `run_id` is set, `runID_<run_id>_` is appended to the prefix.

Runtime logs are printed to standard output. In batch systems, capture stdout
and stderr with the scheduler options. Metrics are not computed automatically
by `hyri imaging`.

## Examples

### uSARA

```bash
hyri imaging \
  --algorithm usara \
  --data_file <DATA_DIRECTORY> \
  --im_dim_x 512 \
  --im_dim_y 512 \
  --ch_start 1 \
  --ch_last 1 \
  --ch_effective 1
```

### HyperAIRI

```bash
hyri imaging \
  --algorithm hyperairi \
  --data_file <DATA_DIRECTORY> \
  --ch_start 1 \
  --ch_last 4 \
  --ch_effective 4 \
  --dnn_shelf_path shelf_hyperairi_mix.csv
```

### Distributed Hyperspectral Imaging

pyHyperRI uses `torch.distributed` when `WORLD_SIZE` is greater than `1`.
For a simple single-node run, launch one process per device with `torchrun`.
The number of processes must not exceed `ch_effective`.

```bash
torchrun --standalone --nproc-per-node=4 \
  -m hyri imaging \
  --algorithm hyper-usara \
  --data_file <DATA_DIRECTORY> \
  --ch_start 1 \
  --ch_last 36 \
  --ch_effective 36
```

See [Distributed Imaging](tutorials/distributed) for more examples of distributed execution.
