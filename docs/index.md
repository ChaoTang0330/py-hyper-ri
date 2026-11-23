---
title: pyHyperRI
---

# pyHyperRI

```{raw} html
<h1 class="hyri-index-logo-title">
  <img src="logo/pyHyperRI_logo_text_black.svg" alt="pyHyperRI" class="hyri-index-logo-lockup only-light">
  <img src="logo/pyHyperRI_logo_text_white.svg" alt="pyHyperRI" class="hyri-index-logo-lockup only-dark">
</h1>
```

```{raw} html
<style>
.prev-next-footer {
  display: none !important;
}

.toctree-wrapper.compound + .toctree-wrapper.compound {
  margin-top: -1rem;
}
</style>
```

pyHyperRI is a Python package for radio-interferometric imaging in astronomy, 
with support for distributed hyperspectral image reconstruction. Built
on PyTorch, it can run across the diverse hardware and execution environments
supported by the PyTorch ecosystem.

## Citation and License

If you use pyHyperRI in your research, please cite this software repository
using the metadata in
[`CITATION.cff`](https://github.com/ChaoTang0330/py-hyper-ri/blob/main/CITATION.cff).
If you use specific methods implemented in the package, please also cite the
relevant method papers listed in {ref}`Algorithms <algorithms>`.

For redistributed or modified versions, please keep
[`CITATION.cff`](https://github.com/ChaoTang0330/py-hyper-ri/blob/main/CITATION.cff)
with the software, so the recommended citation remains available.

pyHyperRI is distributed under the
[BSD 3-Clause License](https://github.com/ChaoTang0330/py-hyper-ri/blob/main/LICENSE).
Redistributed source or binary forms, with or without modification, must retain
or reproduce the copyright notice, license conditions, and disclaimer as
required by the license.

## User Guide
```{toctree}
:maxdepth: 1

background
installation
data
usage
configuration
```
```{toctree}
:maxdepth: 2

tutorials/index
api/index
```
