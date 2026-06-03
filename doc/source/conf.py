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
    "sphinx.ext.intersphinx",
    # "sphinx_autodoc_typehints", # possibly remove for type aliases
    "sphinx_copybutton",
    "autoclasstoc",
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
    "gsd": ("https://gsd.readthedocs.io/en/stable/", None),
    "hoomd": ("https://hoomd-blue.readthedocs.io/en/stable/", None),
    "coxeter": ("https://coxeter.readthedocs.io/en/stable/", None),
}

autodoc_default_options = {
    "inherited-members": False,
    "show-inheritance": False,
    "autosummary": False,
    "special-members": False,
    'private-members': False,
    'inherited-members': False,
    'undoc-members': False,
    # 'exclude-members': '__weakref__',
}

html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": False,
}
html_static_path = ["_static"]

maximum_signature_line_length = 68
python_display_short_literal_types = True


# Custom sections for autoclasstoc
# Docs: https://autoclasstoc.readthedocs.io/en/latest/advanced_usage.html

from autoclasstoc import Section, is_method, is_data_attr, is_special

class Properties(Section):
    key = "properties"
    title = "Properties"

    def predicate(self, name, attr, meta):
        return is_data_attr(name, attr) and not is_special(name)

class Operations(Section):
    key = "operations"
    title = "Operations"

    def predicate(self, name, attr, meta):
        return "operation" in meta or name in ["__add__", "__sub__"]

class CreationFrom(Section):
    key = "creation-from"
    title = "Creation From"

    def predicate(self, name, attr, meta):
        return is_method(name, attr) and name.startswith("from_")

class ConversionTo(Section):
    key = "conversion-to"
    title = "Conversion To"

    def predicate(self, name, attr, meta):
        return is_method(name, attr) and name.startswith("to_")

class Plotting(Section):
    key = "plotting"
    title = "Plotting"

    def predicate(self, name, attr, meta):
        return is_method(name, attr) and name == "plot"

class Measure(Section):
    key = "measure"
    title = "Measure"
  
    def predicate(self, name, attr, meta):
        return "measure" in meta

autoclasstoc_sections = [
    "properties",
    "operations",
    "creation-from",
    "conversion-to",
    "plotting",
    "measure"
]

# Custom CSS
def setup(app):
    app.add_css_file("custom.css")
