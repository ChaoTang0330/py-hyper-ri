"""Expose public optimiser classes.

This module provides the base optimiser interface together with the
forward-backward and primal-dual variants used by the imaging pipeline.
"""

from .optimiser import Optimiser
from .fb_airi import FBAIRI
from .pd_airi import PDAIRI
from .fb_sara import FBSARA
from .fb_hyperairi import FBHyperAIRI
from .fb_hypersara import FBHyperSARA
