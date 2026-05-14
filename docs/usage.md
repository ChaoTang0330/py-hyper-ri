# Usage

## Overview

The standard pyHyperRI workflow is:

1. Prepare input data.
2. Create or edit a configuration file.
3. Run an imaging algorithm.
4. Inspect the generated outputs.

TODO:

- Add a short description of the supported imaging workflows.
- Link to the data and configuration pages.

## Create a Configuration File

Create a configuration file before running an imaging job.

```bash
hyri-config <OPTIONS>
```

TODO:

- Add the final command for generating a default configuration file.
- Explain where the generated file is written.
- Link to the configuration reference.

## Run an Imaging Job

Run an imaging algorithm from a configuration file.

```bash
hyri-imaging --config <CONFIG_FILE>
```

TODO:

- Confirm the exact configuration option name.
- Add a minimal working example.
- Add algorithm-specific examples.

## Select Channels

pyHyperRI can load a range of physical channels and group them into effective
channels for hyperspectral imaging.

`ch_start`:
: First physical channel to load. Channel numbering is 1-based.

`ch_last`:
: Last physical channel to load, inclusive. If omitted, all available channels
  from `ch_start` onward are considered.

`ch_effective`:
: Number of effective channels after grouping. If omitted, each selected
  physical channel is treated as an effective channel.

`ch_partition`:
: List of group sizes mapping selected physical channels to effective channels.
  The sum of the entries must equal the number of selected physical channels.

The frequency of each effective channel is the mean of the physical-channel
frequencies in that group. In distributed execution, each rank loads only its
assigned effective-channel slice, and the loader records the global effective
channel offset for that rank.

## Use AIRI Denoisers

Packaged AIRI denoisers can be selected through the imaging configuration.

TODO:

- Explain where packaged denoisers are stored.
- Explain the relevant configuration fields.
- Explain whether users can provide custom denoisers.

## Distributed Execution

Run distributed imaging jobs using the supported PyTorch execution workflow.

TODO:

- Confirm the recommended distributed launcher.
- Add a single-node example.
- Add a multi-node example if supported.
- Add notes for SLURM-based execution.

## Outputs

Imaging jobs write reconstructed images and related output files to the output
directory configured for the job.

TODO:

- Document reconstructed image outputs.
- Document logs and metrics.
- Document naming conventions.

## Examples

TODO:

- Add a monochromatic example.
- Add a hyperspectral example.
- Add an AIRI example.
- Add a distributed example.
