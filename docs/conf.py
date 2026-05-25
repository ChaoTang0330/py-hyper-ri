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
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
]

root_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
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

nb_execution_mode = "off"

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
    "logo": {"text": "pyHyperRI"},
    "collapse_navigation": False,
    "display_version": True,
    "show_prev_next": True,
    "header_links_before_dropdown": 10,
    "article_header_end": [],
    "article_footer_items": [],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-theme-versions.html"],
    "navbar_end": ["search-field", "theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ChaoTang0330/small-scale-ri-imaging-hs",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            # TODO: Replace this with the package's PyPI project URL once it is available.
            "url": "https://github.com/ChaoTang0330/small-scale-ri-imaging-hs",
            "icon": "fa-brands fa-python",
        },
    ],
    "secondary_sidebar_items": [
        "page-toc",
        "notebook-download.html",
        "github-source.html",
    ],
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

external_module_links = {
    "ptwt": "https://pytorch-wavelet-toolbox.readthedocs.io/en/stable/",
    "pytorch_finufft": "https://flatironinstitute.github.io/pytorch-finufft/",
    "torchkbnufft": "https://torchkbnufft.readthedocs.io/en/stable/",
}

github_source_base = (
    "https://github.com/ChaoTang0330/small-scale-ri-imaging-hs/blob/main/"
)


def _github_path_for_page(pagename, context):
    page_source_map = {
        "api/cli": "src/hyri/cli.py",
        "api/interface": "src/hyri/interface/__init__.py",
        "api/measurement_operator": "src/hyri/measurement_operator/__init__.py",
        "api/optimiser": "src/hyri/optimiser/__init__.py",
        "api/regulariser": "src/hyri/regulariser/__init__.py",
    }
    if pagename in page_source_map:
        return page_source_map[pagename]

    if pagename.startswith("api/generated/"):
        module_name = pagename.removeprefix("api/generated/")
        module_path = SRC / f"{module_name.replace('.', '/')}.py"
        package_path = SRC / module_name.replace(".", "/") / "__init__.py"
        if module_path.exists():
            return module_path.relative_to(ROOT).as_posix()
        if package_path.exists():
            return package_path.relative_to(ROOT).as_posix()

    sourcename = context.get("sourcename")
    if sourcename:
        docs_source_path = ROOT / "docs" / sourcename
        if docs_source_path.exists():
            return docs_source_path.relative_to(ROOT).as_posix()

    return None


def _add_github_source_url(app, pagename, templatename, context, doctree):
    notebook_source = f"{pagename}.ipynb"
    notebook_path = ROOT / "docs" / notebook_source
    context["notebook_source"] = (
        notebook_source if notebook_path.exists() else None
    )

    github_path = _github_path_for_page(pagename, context)
    if github_path is None and context["notebook_source"] is not None:
        github_path = f"docs/{context['notebook_source']}"
    context["github_source_url"] = (
        f"{github_source_base}{github_path}" if github_path is not None else None
    )


def _rename_notes_to_note(app, doctree, docname):
    for admonition in doctree.findall(nodes.admonition):
        if "admonition-notes" not in admonition.get("classes", []):
            continue
        if not admonition or not isinstance(admonition[0], nodes.title):
            continue
        if admonition[0].astext() == "Notes":
            admonition[0].clear()
            admonition[0] += nodes.Text("Note")


def _resolve_external_modules(app, env, node, contnode):
    if node.get("refdomain") != "py" or node.get("reftype") != "mod":
        return None

    url = external_module_links.get(node.get("reftarget"))
    if url is None:
        return None

    ref = nodes.reference("", "", internal=False, refuri=url)
    ref += contnode
    return ref


def setup(app):
    app.connect("html-page-context", _add_github_source_url)
    app.connect("doctree-resolved", _rename_notes_to_note)
    app.connect("missing-reference", _resolve_external_modules)
