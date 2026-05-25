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

## Measurement Operator Walkthrough

Planned notebook: `docs/tutorials/measurement_operator.ipynb`

Goal:
: Demonstrate how to use pyHyperRI's data loader and measurement operator
  directly, without running a full reconstruction algorithm.

Outline:

1. Introduce the forward model: image to visibilities with `forward_op`, and
   visibilities to dirty image with `adjoint_op`.
2. Import `load_data_to_tensor` and the relevant measurement operator class.
3. Load channel 2 from `data/3c353_sim_vla_meas_hs/` with tutorial settings:
   image size `(512, 512)`, super-resolution `1.5`, Briggs weighting, and
   `ch_start = 1, and ch_last = 2`.
4. Inspect loaded tensors: `u`, `v`, `w`, `y`, natural weights, imaging weights,
   preconditioning weights, frequencies, and image pixel size.
5. Build the measurement operator for the loaded channel on the selected device.
6. Compute the PSF with `get_psf`.
7. Compute the dirty image with `adjoint_op(y)` and normalise by the PSF peak.
8. Optionally apply `forward_op` to a simple synthetic image and inspect the
   predicted visibility shape.
9. Plot the PSF and dirty image, and compare them with the reference
   `psf.fits` and `dirty_normalised.fits` included in the example data.
10. Summarise when to use the direct measurement-operator API instead of the
    high-level imaging CLI.

Notes:

- Keep this notebook more code-centric than the imaging tutorials.
- Prefer CPU-compatible cells where possible, then show the one-line change to
  run the operator on CUDA.
- Reuse checks from `test/test_data_load.py` and `test/test_data_integration.py`
  as lightweight assertions inside the notebook.

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
