"""Astronomical hyperspectral radio interferometric imaging package.

The :mod:`hyri` package provides reusable components for building imaging
pipelines, including measurement operators, proximal and denoising
regularisers, and optimisation algorithms for hyperspectral radio
interferometric image reconstruction.
"""

from .imager import imager

__all__ = ["imager"]
