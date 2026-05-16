# Configuration

## Overview

The `hyri-imaging` command accepts both command-line inputs and a JSON
configuration file. Command-line inputs are useful for quick overrides, while
the configuration file gives full control of the imaging settings, including
the input data, image grid, channel selection, weighting, compute devices,
measurement operator, algorithm, and algorithm-specific parameters.

pyHyperRI also provides the `hyri-config` wizard to help create a configuration
file from the packaged template.

## Create a Configuration File

Create a default configuration file without prompts:

```bash
hyri-config --default --output hyri_config.json
```

Create a configuration file through the interactive wizard:

```bash
hyri-config --output hyri_config.json
```

Overwrite an existing output file:

```bash
hyri-config --default --output hyri_config.json --overwrite
```

The wizard can ask for data, image-grid, channel, weighting, output, compute,
NUFFT, and algorithm-specific options. After creating a configuration file,
edit the JSON file directly to access the full set of imaging settings.

## Configuration Structure

The default configuration template is packaged at `hyri/config/default.json`.
It is organised into common sections and algorithm-specific sections:

- `main`
- `flag`
- `weighting`
- `computing`
- `nufft`
- `airi` and `airi_default`
- `cairi` and `cairi_default`
- `hyperairi` and `hyperairi_default`
- `usara` and `usara_default`
- `hyper-usara` and `hyper-usara_default`

For each run, pyHyperRI reads `main`, the common sections, and the sections
matching the selected `algorithm`. Command-line options override matching JSON
fields when provided.

## Main Settings

The parameters in the `main` section can be overridden by the corresponding command-line options.

`src_name`:
: Experiment or target source name tag used in the output file names. If empty,
  pyHyperRI uses the input data directory name.

`data_file`:
: Path to the input measurement data directory.

`result_path`:
: Base directory where output files are written. pyHyperRI creates a
  source-specific subdirectory under this path.

`algorithm`:
: Imaging algorithm. Valid values are `usara`, `hyper-usara`, `airi`, `cairi`,
  and `hyperairi`.

`im_dim_x`:
: Horizontal dimension of the reconstructed image in pixels.

`im_dim_y`:
: Vertical dimension of the reconstructed image in pixels.

`ch_start`:
: First physical channel to load. Channel numbering is 1-based.

`ch_last`:
: Last physical channel to load, inclusive. Leave empty to use all available
  channels from `ch_start` onward.

`ch_effective`:
: Number of effective channels in the reconstructed image cube.

`ch_partition`:
: Number of physical channels assigned to each effective channel. Leave empty
  to partition the selected physical channels evenly. If provided, the list
  length sets `ch_effective`.

`im_pixel_size`:
: Pixel size of the reconstructed image in arcsec. Leave empty to derive the
  pixel size from `superresolution` and the maximum projected baseline.

`superresolution`:
: Imaging super-resolution factor used when `im_pixel_size` is empty.

`run_id`:
: Optional identification number for the current run. If provided, it is
  appended to output file prefixes.

## Flag Settings

`flag_imaging`:
: Enable imaging. If `false`, pyHyperRI generates only the dirty image and PSF.

`flag_data_weighting`:
: Enable the data-weighting scheme.

`invert_uvw_flag`:
: Whether to invert the loaded UVW flag logic before filtering samples.

`verbose`:
: Enable verbose runtime output.

## Weighting Settings

`weight_load`:
: Whether to read imaging weights from the channel files when data weighting is
  enabled.

`weight_type`:
: Data-weighting scheme used when weights are generated. Valid values are
  `briggs` and `uniform`.

`weight_robustness`:
: Briggs robustness parameter used when `weight_type` is `briggs`.

`weight_gridsize`:
: Padding factor used when estimating the sampling density for imaging-weight
  generation.

## Computing Settings

`meas_device`:
: Device that the measurement operators run on. Valid values are `cpu`, `cuda`,
  `mps`, and `xpu`; unavailable devices fall back to `cpu`.

`meas_dtype`:
: Numeric precision accepted and returned by the measurement operators. Valid
  values are `single`, `float`, `float32`, `double`, and `float64`.

`regulariser_device`:
: Device that the regularisers and denoisers run on. Valid values are `cpu`,
  `cuda`, `mps`, and `xpu`; unavailable devices fall back to an available
  device.

`regulariser_dtype`:
: Numeric precision accepted and returned by regularisers and denoisers. Valid
  values are `single`, `float`, `float32`, `double`, and `float64`.

`dist_backend`:
: PyTorch distributed backend. Valid values are `gloo`, `nccl`, `mpi`, and
  `xccl`; unavailable backends fall back to `gloo`.

`dist_init_method`:
: Initialisation URL passed to `torch.distributed.init_process_group`. Supported
  schemes are `env://`, `file://`, and `tcp://`.

## Measurement Operator Settings

`nufft_package`:
: NUFFT package used by the measurement operator. Valid values are `finufft`
  and `tkbnufft`.

`nufft_mode`:
: TorchKbNUFFT interpolation mode. Valid values are `table` and `matrix`; this
  option is only used when `nufft_package` is `tkbnufft`.

## uSARA Settings

These fields are used when `algorithm` is `usara`.

`heu_reg_param_scale`:
: Adjusting factor applied to the regularisation parameter computed from the
  heuristic noise level. Larger values usually give stronger regularisation and
  smoother reconstructions.

`reweighting`:
: Enable the reweighting algorithm.

`approx_meas_op`:
: Use the approximate measurement operator during reconstruction.

`im_min_itr`:
: Minimum number of inner forward-backward iterations.

`im_max_itr`:
: Maximum number of inner forward-backward iterations.

`im_max_outer_itr`:
: Maximum number of outer reweighting iterations.

`im_var_tol`:
: Tolerance on the relative variation of the estimate for stopping the inner
  iterations.

`im_var_outer_tol`:
: Tolerance on the relative variation of the estimate for stopping the outer
  reweighting iterations.

`reweighting_save`:
: Whether to save intermediate reweighting products.

## Hyper-uSARA Settings

These fields are used when `algorithm` is `hyper-usara`. They have the same
meaning as the corresponding uSARA settings, applied to the hyperspectral
variant.

`heu_reg_param_scale`:
: Adjusting factor applied to the regularisation parameter computed from the
  heuristic noise level.

`reweighting`:
: Enable the reweighting algorithm.

`approx_meas_op`:
: Use the approximate measurement operator during reconstruction.

`im_min_itr`:
: Minimum number of inner forward-backward iterations.

`im_max_itr`:
: Maximum number of inner forward-backward iterations.

`im_max_outer_itr`:
: Maximum number of outer reweighting iterations.

`im_var_tol`:
: Tolerance on the relative variation of the estimate for stopping the inner
  iterations.

`im_var_outer_tol`:
: Tolerance on the relative variation of the estimate for stopping the outer
  reweighting iterations.

`reweighting_save`:
: Whether to save intermediate reweighting products.

## AIRI Settings

These fields are used when `algorithm` is `airi`.

`heu_noise_scale`:
: Adjusting factor applied to the heuristic noise-level estimate. Larger values
  usually give stronger denoising and smoother reconstructions.

`dnn_shelf_path`:
: Path to the CSV file defining the denoiser shelf. The shelf maps noise levels
  to ONNX denoisers.

`im_peak_est`:
: Estimated maximum intensity of the reconstructed image. Leave empty to
  estimate it from the dirty image normalised by the PSF peak.

`dnn_adaptive_peak`:
: Whether to adaptively select denoisers during reconstruction from the current
  reconstructed image peak.

`dnn_apply_transform`:
: Apply random rotations and flips before denoising, then undo the transform
  after denoising.

`approx_meas_op`:
: Use the approximate measurement operator during reconstruction.

`im_min_itr`:
: Minimum number of forward-backward iterations.

`im_max_itr`:
: Maximum number of forward-backward iterations.

`im_var_tol`:
: Tolerance on the relative variation of the estimate for stopping iterations.

`itr_save`:
: Interval for saving intermediate FITS snapshots. If larger than
  `im_max_itr`, no intermediate snapshot is saved.

`dnn_adaptive_peak_tol_max`:
: Initial relative peak-variation tolerance for adaptive denoiser selection.

`dnn_adaptive_peak_tol_min`:
: Minimum relative peak-variation tolerance for adaptive denoiser selection.

`dnn_adaptive_peak_tol_step`:
: Decay factor applied after each denoiser reselection.

## cAIRI Settings

These fields are used when `algorithm` is `cairi`.

`heu_noise_scale`:
: Adjusting factor applied to the heuristic noise-level estimate.

`dnn_shelf_path`:
: Path to the CSV file defining the denoiser shelf. The shelf maps noise levels
  to ONNX denoisers.

`precond_flag`:
: Whether to enable preconditioned weighting for the constrained algorithm.

`im_peak_est`:
: Estimated maximum intensity of the reconstructed image. Leave empty to
  estimate it from the dirty image normalised by the PSF peak.

`dnn_adaptive_peak`:
: Whether to adaptively select denoisers during reconstruction.

`dnn_apply_transform`:
: Apply random rotations and flips before denoising, then undo the transform
  after denoising.

`im_min_itr`:
: Minimum number of primal-dual iterations.

`im_max_itr`:
: Maximum number of primal-dual iterations.

`im_var_tol`:
: Tolerance on the relative variation of the estimate for stopping iterations.

`itr_save`:
: Interval for saving intermediate FITS snapshots.

`dnn_adaptive_peak_tol_max`:
: Initial relative peak-variation tolerance for adaptive denoiser selection.

`dnn_adaptive_peak_tol_min`:
: Minimum relative peak-variation tolerance for adaptive denoiser selection.

`dnn_adaptive_peak_tol_step`:
: Decay factor applied after each denoiser reselection.

## HyperAIRI Settings

These fields are used when `algorithm` is `hyperairi`.

`heu_noise_scale`:
: Adjusting factor applied to the heuristic noise-level estimate.

`dnn_shelf_path`:
: Path to the CSV file defining the denoiser shelf. The shelf maps noise levels
  to ONNX denoisers.

`im_peak_est`:
: Estimated maximum intensity of the reconstructed image. Leave empty to
  estimate it from the dirty image normalised by the PSF peak.

`dnn_adaptive_peak`:
: Whether to adaptively select denoisers during reconstruction.

`dnn_apply_transform`:
: Apply random rotations and flips before denoising, then undo the transform
  after denoising.

`approx_meas_op`:
: Use the approximate measurement operator during reconstruction.

`spec_update`:
: Whether to update the spectral-index map adaptively during reconstruction.

`spec_idx_init`:
: Spectral-index map initialisation. Valid values are `zero`, `dirty`, or a
  path to a FITS file.

`im_min_itr`:
: Minimum number of forward-backward iterations.

`im_max_itr`:
: Maximum number of forward-backward iterations.

`im_var_tol`:
: Tolerance on the relative variation of the estimate for stopping iterations.

`itr_save`:
: Interval for saving intermediate FITS snapshots.

`dnn_adaptive_peak_tol_max`:
: Initial relative peak-variation tolerance for adaptive denoiser selection.

`dnn_adaptive_peak_tol_min`:
: Minimum relative peak-variation tolerance for adaptive denoiser selection.

`dnn_adaptive_peak_tol_step`:
: Decay factor applied after each denoiser reselection.

`spec_max_update_count`:
: Maximum number of spectral-index updates.

`spec_max_update_interval`:
: Maximum number of iterations between spectral-index update checks.

`spec_update_tol_max`:
: Initial relative variation tolerance for spectral-index updates.

`spec_update_tol_min`:
: Minimum relative variation tolerance for spectral-index updates.

`spec_update_tol_step`:
: Decay factor applied to the spectral-index update tolerance.

## Example Configuration

```json
[
    {
        "__comment": "Main",
        "main": {
            "src_name": [],
            "data_file": "<DATA_DIRECTORY>",
            "result_path": "results",
            "algorithm": "hyperairi",
            "im_dim_x": 512,
            "im_dim_y": 512,
            "ch_start": 1,
            "ch_last": [],
            "ch_effective": 1,
            "ch_partition": [],
            "im_pixel_size": [],
            "superresolution": 1.5,
            "run_id": []
        }
    }
]
```

This example shows the `main` section only. A complete configuration file also
contains the common sections and the algorithm-specific sections described
above.

## Validation Note

Empty lists, written as `[]`, are used in the default template for optional
values that should be inferred at runtime.

If `im_pixel_size` is provided, it takes priority over `superresolution`.

If `ch_partition` is provided and all entries are positive, its length sets the
number of effective channels.

Unsupported devices, distributed backends, NUFFT packages, NUFFT modes, and
invalid numeric values fall back to runtime defaults where supported by the
parameter parser.
