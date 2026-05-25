# Data

## Overview

pyHyperRI expects measurement data stored in a directory containing `.mat`
files. The directory contains one metadata file for the sampling pattern and
frequency information, plus one data file per channel for visibilities,
weights, and flags.

## Measurement Directory

The input measurement directory is expected to contain:

```text
<DATA_DIRECTORY>/
  msSpecs.mat
  data_ch_1.mat
  data_ch_2.mat
  ...
```

`msSpecs.mat` stores global metadata shared across channels. Each
`data_ch_<index>.mat` file stores data for one physical frequency channel.
Channel indices are 1-based. When loading a channel range, pyHyperRI looks for
files matching `*data_ch_<index>.mat` for each selected physical channel.
The selected range must contain one channel file for each requested physical
channel.

## Sampling Pattern Metadata

`msSpecs.mat` contains the sampling pattern and frequency metadata.

Required fields:

`freqs`:
: Physical channel frequencies in Hz. The array is flattened before use.

`uvw`:
: Sampling coordinates in metres. The expected layout is either `(N, 3)` or
  `(3, N)`, where the three columns or rows correspond to `u`, `v`, and `w`.
  During loading, coordinates are converted to wavelength units for each
  physical channel.

Optional fields:

`super_resolution`:
: Super-resolution factor used to derive the image pixel size when no explicit
  pixel size is provided.

## Channel Data

Each `data_ch_<index>.mat` file contains visibility data and optional
channel-specific weights or flags.

Optional fields:

`data_I`:
: Complex Stokes I visibility data for the channel. If this field is absent,
  the loader still prepares the sampling pattern and weights, but no visibility
  tensor is populated for that channel.

`flag`:
: Boolean or boolean-like mask for invalid samples. Samples with `flag == True`
  are excluded before tensors are created. The interpretation can be inverted
  with the corresponding loading option.

`weightsNat`:
: Natural weights for the channel. This may be a scalar or a vector matching
  the number of valid samples. If absent, natural weights are set to one.

`weightsImg`:
: Imaging weights for the channel. These are loaded only when requested. If
  file-based imaging weights are requested but unavailable, pyHyperRI falls
  back to unweighted data. If file-based weights are not requested and imaging
  weighting is enabled, pyHyperRI generates imaging weights from the sampling
  pattern and natural weights.

All per-sample channel arrays are flattened before loading. After flagged
samples are removed, per-channel arrays must be compatible with the remaining
number of valid samples.

## Image Data

Image files read or written by pyHyperRI use the FITS format.

## Measurement Set Conversion

[TODO: finalise this section after finising the converter]

Measurement Set data can be converted into the `.mat` layout expected by
pyHyperRI.

The `ms2mat` utility prepares Measurement Set data for pyHyperRI by writing the
`msSpecs.mat` metadata file and the per-channel `data_ch_<index>.mat` files.
Use the converter when starting from Measurement Set data rather than a
pre-existing `.mat` measurement directory.

See the Measurement Set conversion documentation for supported assumptions and
commands.

## Example Data

Example hyperspectral measurement files are available under:

```text
data/3c353_sim_vla_meas_hs/
```

This directory follows the expected layout with `msSpecs.mat` and per-channel
data files.

## Note

The loader reads MATLAB v7.3 files with `h5py` and older MAT files with
`scipy.io.loadmat`.

Coordinates in `msSpecs.mat` are converted internally from metres to wavelength
units using the physical channel frequency. The loaded `u` and `v` coordinates
are additionally normalised for the measurement operator using the image pixel
size.

Visibility tensors are complex-valued. Coordinate and weight tensors use the
real dtype selected by the runtime configuration.
