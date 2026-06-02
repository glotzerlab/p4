# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

"""Utility functions used by multiple classes and/or functions.

In contrast to other classes and functions in p4, functions in this submodule
generally do not have argument defaults, and they never validate or sanitize
user input.
"""

from . import colors
from . import polyhedron_intersection
from . import simulation
from . import data
from . import plotting
