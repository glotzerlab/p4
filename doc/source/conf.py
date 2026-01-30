# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "p4"
copyright = "2025-2026, The Regents of the University of Michigan"
author = "Joseph Burkhart"
release = "0.0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

# For sphincontrib.bibtex (as of v2.0).
# bibtex_bibfiles = []

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []  # type: ignore[var-annotated]

source_suffix = ".rst"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org", None),
    "gsd": ("https://gsd.readthedocs.io/en/stable/", None),
    "hoomd": ("https://hoomd-blue.readthedocs.io/en/stable/", None),
}

autodoc_default_options = {
    "inherited-members": False,
    "show-inheritance": False,
    "autosummary": False,
}

html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": False,
}
html_static_path = ["_static"]

# Override default nbsphinx CSS for HTML tables, as it doesn't respect dark mode
nbsphinx_prolog = """
.. raw:: html

    <style>
        div.rendered_html table {
            color: var(--color-code-foreground);
        }
        div.rendered_html thead {
            border-bottom: 1px solid var(--color-code-foreground);
        }
        div.rendered_html thead tr {
            background: var(--color-background-secondary);
        }
        div.rendered_html tbody tr:nth-child(2n) {
            background: var(--color-background-primary);
        }
        div.rendered_html tbody tr:nth-child(2n+1) {
            background: var(--color-background-secondary);
        }
    </style>
"""