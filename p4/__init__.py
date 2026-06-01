# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from . import body
from .body import Body
from . import arrangement
from .arrangement import Arrangement
from . import interaction
from .interaction import Interaction
from . import system
from .system import System
from . import field
from .field import Field

from .top_level_functions import (
    positions_on_regular_grid,
    # exclude_positions_by_shape,
    orientations_about_axis,
    orientations_from_fibonacci_lattice,
    # exclude_orientations_with_shape_overlap,
    plot_positions,
    # plot_orientations,
    # plot_state,
    # plot_bodies,
    # plot_interactions,
)

__all__ = [
    "body",
    "Body",
    "arrangement",
    "Arrangement",
    "interaction",
    "Interaction",
    "system",
    "System",
    "field",
    "Field",
    "positions_on_regular_grid",
    # "exclude_positions_by_shape",
    "orientations_about_axis",
    "orientations_from_fibonacci_lattice",
    # "exclude_orientations_with_shape_overlap",
    "plot_positions",
    # "plot_orientations",
    # "plot_state",
    # "plot_bodies",
    # "plot_interactions",
]

__version__ = "0.0.1"
