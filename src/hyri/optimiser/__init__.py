"""Expose public optimiser classes.

This module provides the base optimiser interface together with the
forward-backward and primal-dual variants used by the imaging pipeline.
"""

from .fb_airi import FBAIRI
from .fb_hyperairi import FBHyperAIRI
from .fb_hypersara import FBHyperSARA
from .fb_sara import FBSARA
from .optimiser import Optimiser
from .pd_airi import PDAIRI

__all__ = [
    "Optimiser",
    "FBAIRI",
    "FBHyperAIRI",
    "FBHyperSARA",
    "FBSARA",
    "PDAIRI",
]
