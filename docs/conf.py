"""Sphinx configuration for the pyHyperRI documentation."""

from __future__ import annotations

import sys
from pathlib import Path

from docutils import nodes

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
    "sphinx.ext.extlinks",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

root_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
templates_path = ["_templates"]

autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "special-members": "__call__",
    "exclude-members": "__init__",
    "show-inheritance": True,
}

autoclass_content = "both"
autodoc_inherit_docstrings = False
autodoc_typehints = "signature"
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_admonition_for_notes = True
napoleon_custom_sections = [("Shape", "params_style")]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "pytorch_sphinx_theme2"
html_title = f"{project} documentation"
html_short_title = project
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
pygments_style = "sphinx"

html_theme_options = {
    "pytorch_project": "docs",
    "collapse_navigation": False,
    "display_version": True,
    "article_header_end": [],
    "article_footer_items": [],
    "navbar_end": ["search-field", "theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],
    "show_lf_footer": False,
    "show_pytorch_org_link": False,
    "use_edit_page_button": False,
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

nitpicky = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable/", {}),
    "torch": ("https://docs.pytorch.org/docs/stable/", {}),
    "scipy": ("https://docs.scipy.org/doc/scipy/", {}),
    "astropy": ("https://docs.astropy.org/en/stable/", {}),
    "h5py": ("https://docs.h5py.org/en/stable/", {}),
    "skimage": ("https://scikit-image.org/docs/stable/", {}),
    "ptwt": ("https://pytorch-wavelet-toolbox.readthedocs.io/en/stable/", {}),
    "pytorch_finufft": ("https://flatironinstitute.github.io/pytorch-finufft/", {}),
    "torchkbnufft": ("https://torchkbnufft.readthedocs.io/en/stable/", {}),
}

extlinks = {
    "ptfinufft": ("https://flatironinstitute.github.io/pytorch-finufft/%s", "%s"),
    "ptwt-docs": ("https://pytorch-wavelet-toolbox.readthedocs.io/en/stable/%s", "%s"),
    "torchkbnufft-docs": ("https://torchkbnufft.readthedocs.io/en/stable/%s", "%s"),
}


def _rename_notes_to_note(app, doctree, docname):
    for admonition in doctree.findall(nodes.admonition):
        if "admonition-notes" not in admonition.get("classes", []):
            continue
        if not admonition or not isinstance(admonition[0], nodes.title):
            continue
        if admonition[0].astext() == "Notes":
            admonition[0].clear()
            admonition[0] += nodes.Text("Note")


def setup(app):
    app.connect("doctree-resolved", _rename_notes_to_note)
