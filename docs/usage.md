# Usage

## Overview

After the input data is prepared, launch imaging jobs with `hyri-imaging`.
If no configuration file is provided, the command reads the default
configuration bundled with the package. Command-line overrides are available
for common run settings such as the input data directory, output directory,
algorithm, image size, denoiser shelf, channel range, and run ID.

## Run an Imaging Job

Run an imaging algorithm with the packaged default configuration:

```bash
hyri-imaging --data_file <DATA_DIRECTORY>
```

Run with an explicit configuration file:

```bash
hyri-imaging --config <CONFIG_FILE>
```

`--config` can also be written as `-c`. If it is omitted, pyHyperRI reads the
bundled `hyri/config/default.json` template. See [Configuration](configuration)
for details on the configuration file generations and contents.

`hyri-imaging` also accepts the following command-line arguments to override the 
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
hyri-imaging \
  --data_file <DATA_DIRECTORY> \
  --ch_start 1 \
  --ch_last 36 \
  --ch_effective 36

# Group 36 physical channels into 4 effective channels of 9 physical channels each.
hyri-imaging \
  --data_file <DATA_DIRECTORY> \
  --ch_start 1 \
  --ch_last 36 \
  --ch_partition 9 9 9 9
```

## Use AIRI Denoisers

Packaged AIRI denoisers can be selected through the imaging configuration.

The packaged denoiser shelf CSV files are installed as package data under
`hyri/airi_denoisers`. In the source tree, they are located at
`src/hyri/airi_denoisers`:

- `shelf_airi_oaid.csv`
- `shelf_airi_mrid.csv`
- `shelf_hyperairi_mix.csv`

For AIRI family of algorithms, set `dnn_shelf_path` in the matching algorithm
section of the configuration file, or override it from the command line:

```bash
hyri-imaging \
  --algorithm airi \
  --data_file <DATA_DIRECTORY> \
  --dnn_shelf_path shelf_airi_oaid.csv
```

Besides the three packaged shelves, users can create custom shelves with
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
by `hyri-imaging`; helper scripts for evaluation are available under `scripts`.

## Examples

### uSARA

```bash
hyri-imaging \
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
hyri-imaging \
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
  hyri-imaging \
  --algorithm hyper-usara \
  --data_file <DATA_DIRECTORY> \
  --ch_start 1 \
  --ch_last 36 \
  --ch_effective 36
```

See [Distributed Imaging](tutorials/distributed) for more examples of distributed execution.