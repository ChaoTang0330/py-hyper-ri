:orphan:

# Notebook Tutorial Outlines

This page records the planned notebook tutorials before the notebooks are
written. The notebooks will use the repository example data at
`data/3c353_sim_vla_meas_hs/` and image channel 2 with:

```text
ch_start = 2
ch_last = 2
ch_effective = 1
```

The starting configuration is `config/tutorial.json`.

## uSARA Mono-GPU Imaging

Planned notebook: `docs/tutorials/usara_mono_gpu.ipynb`

Goal:
: Run a single-channel uSARA reconstruction on one GPU and inspect the main
  outputs.

Outline:

1. Introduce the task: image channel 2 from `data/3c353_sim_vla_meas_hs/`.
2. Check the environment: import `torch` and `hyri`, report CUDA availability,
   and select a GPU device when available.
3. Inspect the measurement directory: confirm `msSpecs.mat` and
   `data_ch_2.mat` exist.
4. Use `config/tutorial.json` and specify the data, algorithm, and result path with command-line arguments.
5. Run `hyri-imaging` from the notebook or call the CLI entry point through a
   subprocess.
6. Load the generated FITS products, such as the reconstruction, PSF, dirty
   image, and residual image.
7. Plot the dirty image, reconstructed image, and residual side by side.
8. Summarise the key output files and the uSARA parameters that most often need
   adjustment.

Notes:

- Keep the first notebook command-line driven so it mirrors the documented CLI
  workflow.
- Use modest iteration settings for a tutorial run if runtime is too long.
- Include a CPU fallback only for inspecting data and operators; reconstruction
  examples are intended for mono-GPU use.

## AIRI Mono-GPU Imaging

Planned notebook: `docs/tutorials/airi_mono_gpu.ipynb`

Goal:
: Run a single-channel AIRI reconstruction on one GPU using the packaged AIRI
  denoiser shelf.

Outline:

1. Introduce AIRI as the learned-denoiser counterpart to the uSARA example.
2. Check the environment: import `torch`, `hyri`, and `onnx2torch`; report CUDA
   availability.
3. Locate the packaged denoiser shelf, using `shelf_airi_oaid.csv` by default.
4. Inspect the measurement directory and confirm `data_ch_2.mat` exists.
5. Use `config/tutorial.json` and specify the data, algorithm, result path with command-line arguments.
6. Run `hyri-imaging` from the notebook or call the CLI entry point through a
   subprocess.
7. Load and plot the dirty image, reconstructed image, and residual image.

Notes:

- Keep the same output plotting code as the uSARA notebook for easy
  visual comparison.

## Hyperspectral Measurement Simulation with Package APIs

Planned notebook: `docs/tutorials/hs_meas_simulation.ipynb`

Goal:
: Demonstrate how to use pyHyperRI's functions to build a
  small synthetic hyperspectral measurement experiment from an existing
  sampling pattern. The notebook should show how `msSpecs.mat`, the data
  loader, and `MeasOpHyperspectral` fit together, without running a full
  reconstruction algorithm.

Working idea:
: Use the existing `uvw` coordinates in
  `data/3c353_sim_vla_meas_hs/msSpecs.mat`, define three uniformly spaced
  channel frequencies from 1 to 2 GHz, create a synthetic three-channel sky
  cube, simulate noisy visibilities with the hyperspectral forward operator,
  save the result in pyHyperRI's measurement-directory format, and reload it
  with `load_data_to_tensor`. This is more interesting than only loading
  `data_ch_2.mat`, while still teaching the original data-loader and
  measurement-operator workflow.

Proposed learning outcomes:

1. Understand the relationship between `msSpecs.mat`, channel files, and the
   tensors consumed by measurement operators.
2. Create a small pyHyperRI-compatible measurement directory with `msSpecs.mat`
   and `data_ch_<index>.mat` files.
3. Build `MeasOpHyperspectral` directly and inspect the expected image and
   visibility tensor shapes.
4. Use `forward_op` to generate synthetic hyperspectral visibilities from a
   known image cube.
5. Use `adjoint_op` and `get_psf` to form dirty images and PSFs, then interpret
   channel-dependent sampling effects.
6. Add controlled complex Gaussian noise to the simulated visibilities and
   inspect its effect in the dirty image domain.

Draft outline:

1. Introduce the task: simulate a small hyperspectral RI measurement set using
   the existing 3C 353 sampling pattern and image in the repository.
2. Import the package APIs:
   `load_mat_to_np`, `load_data_to_tensor`, `gen_img_cube`,
   `MeasOpHyperspectral`, and the chosen monochromatic backend such as
   `MeasOpFinufft`.
3. Set paths and basic parameters:
   `DATA_DIR = $REPO/data/3c353_sim_vla_meas_hs/`,
   `TARGET_DIR = $REPO/data/sim_meas_tutorial/`,
   `NUM_CH = 3`, image size `(512, 512)`, super-resolution `1.5`, CPU device,
   and double precision. Create `TARGET_DIR` if needed and overwrite the
   notebook-generated files in that folder when they already exist.
4. Load the example `msSpecs.mat`, print the shape of `uvw`, and explain that
   sampling patterns can also be generated externally, for example with CASA
   `simobserve`; this notebook reuses the repository sampling pattern for a
   self-contained demonstration.
5. Define three channel frequencies uniformly over 1-2 GHz and save a new
   `msSpecs.mat` in the target measurement folder containing:
   - `uvw`: copied from the example `msSpecs.mat`;
   - `freqs`: the three simulated frequencies in Hz.
6. Generate a synthetic hyperspectral image cube with `gen_img_cube`.
   Use the first channel of `groundtruth.fits` as the reference image and the
   three simulated frequencies as the spectral axis. The helper uses a
   power-law/curvature model and takes the first selected frequency as the
   reference frequency.
7. Plot the synthetic cube: three channel images, the spectral-index map, and
   optionally the curvature map.
8. Build initial sampling tensors and a `MeasOpHyperspectral` instance from the
   new `msSpecs.mat`. Use natural weights for this simulation; imaging weights
   can be generated by `load_data_to_tensor` in other workflows.
9. Plot the `uv` coverage for each channel in wavelength units. Use the
   maximum projected baseline across all channels to set shared symmetric axis
   limits.
10. Simulate noiseless visibilities with `meas_op.forward_op(sky_cube)`.
   Inspect each channel's complex visibility tensor shape, dtype, amplitude
   range, and phase range.
11. Add controlled complex Gaussian noise in visibility space. Briefly discuss
    the AIRI heuristic image-domain noise relation
    $\sigma = \tau / \sqrt{2L}$, where $L = \|\Phi^\dagger\Phi\|_S$ is the
    channel-wise operator norm. Choose a target image-domain noise level
    $\sigma = 10^{-4}$, corresponding to dynamic range $10^4$, compute
    channel-wise operator norms, and set $\tau$ from the last channel's
    operator norm. Reference the first AIRI paper in the notebook reference
    section.
12. Form dirty image cubes with `meas_op.adjoint_op(y_sim)` and
    `meas_op.adjoint_op(y_noisy)`. Compute PSFs with `meas_op.get_psf()`.
13. Save only the noisy simulated visibilities to the target measurement
    directory. Follow the documented pyHyperRI file layout:
    - `msSpecs.mat` with `uvw` and `freqs`;
    - `data_ch_1.mat`, `data_ch_2.mat`, `data_ch_3.mat`;
    - each channel file should include complex `data_I` and vector-valued
      `weightsNat` set to one for natural weighting.
    Do not save `flag`; the loader will treat all samples as valid by default.
    Also save the simulated ground-truth image cube in the target folder for
    later inspection.
14. Reload the data from the target directory with `load_data_to_tensor` and
    build a new `MeasOpHyperspectral` from the reloaded tensors. Use natural
    weighting for simplicity, and mention that imaging weighting can be enabled
    through the loading function.
15. Visualise the PSF and dirty image for each simulated channel in linear
    scale using `twilight_shifted`; visualise the synthetic sky and selected
    dirty images with a consistent image colormap such as `magma`.
16. Summarise when to use direct package APIs:
    emphasise that, due to its modular design, pyHyperRI can be a useful
    platform for testing, simulation, developing, in addition to imaging.

Notebook title:

- `Simulating hyperspectral RI measurements`

Design notes:

- This notebook should be more exploratory and code-centric than the uSARA and
  AIRI CLI notebooks.
- Prefer a CPU-compatible path where possible.
- Keep generated measurement files under `data/sim_meas_tutorial/`.
- Do not overwrite the repository example data.
- Avoid promising a complete reconstruction from the simulated data; the focus
  is to provide an example of using the package functions directly.
- The notebook can reuse plotting conventions from the imaging tutorials:
  `twilight_shifted` for PSF/dirty/residual-style images and `magma` for sky
  images.

## Open Decisions

- Decide whether notebooks should execute full reconstructions by default or use
  shortened iteration counts for documentation builds.
  Answer: Yes, and I've shortened the iteration counts in `config/tutorial.json`.
- Decide whether the notebook outputs should be committed as cleared notebooks
  or with selected plots saved in the notebook.
  Answer: Commit with selected plots saved in the notebook, so that the doc build step doesn't have to run the full reconstructions.
- Decide whether to include the outline page in the public tutorials toctree or
  keep it as an orphan planning document.
  Answer: this page is for planning purpose only and will not be included
