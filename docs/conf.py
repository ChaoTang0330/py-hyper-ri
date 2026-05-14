"""Sphinx configuration for the pyHyperRI documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "pyHyperRI"
author = "Chao Tang"
copyright = "2026, Chao Tang"

release = "1.0.0"
version = release
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
    "myst_nb",
]

root_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

autoclass_content = "class"
autodoc_inherit_docstrings = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]

nb_execution_mode = "off"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "pytorch_sphinx_theme2"
html_title = f"{project} documentation"
html_short_title = project
pygments_style = "sphinx"

html_theme_options = {
    "pytorch_project": "docs",
    "collapse_navigation": False,
    "display_version": True,
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

nitpicky = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable/", {}),
    "torch": ("https://docs.pytorch.org/docs/stable/", {}),
    "scipy": ("https://docs.scipy.org/doc/scipy/", {}),
}

autodoc_mock_imports = [
    "torch",
    "onnx2torch",
    "ptwt",
    "torchkbnufft",
    "finufft",
    "pytorch_finufft",
    "cufinufft",
    "astropy",
    "h5py",
    "scipy",
    "skimage",
]
