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
    "sphinx_autodoc_typehints", # possibly remove for type aliases
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

maximum_signature_line_length = 100
python_display_short_literal_types = True


def setup(app):
  app.add_css_file("custom.css")