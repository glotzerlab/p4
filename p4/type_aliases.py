# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

"""Aliases for common data types.

Type aliases are provided as a convenience for users with static type checking
tools such as `mypy`_ and `pyright`_. Their definitions are documented here as a
reference for all users.

.. _`mypy`: https://github.com/python/mypy
.. _`pyright`: https://github.com/microsoft/pyright
"""

from collections.abc import Iterable
from typing import Literal

import hoomd
import gsd.hoomd
import numpy as np


type state_like = (hoomd.State | hoomd.Snapshot | gsd.hoomd.Frame)
"""Types that either are or are mappable to a HOOMD-blue simulation state."""

type axis_like = (
    list[float]
    | tuple[float, float, float]
    | np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]
)
"""Types that represent a cartesian axis as a 3-vector of floats."""

type positions_like = (
    Iterable[list[float]]
    | Iterable[tuple[float, float, float]]
    | Iterable[np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]]
    | np.ndarray[tuple[*tuple[int, ...], Literal[3]], np.dtype[np.floating]]
)
"""Types that represent an array of positions in 3D space."""

type orientation_like = (
    list[float]
    | tuple[float, float, float, float]
    | np.ndarray[tuple[Literal[4]], np.dtype[np.floating]]
)
"""Types that represent an orientation as a quaternion."""

type orientations_like = (
    Iterable[list[float]]
    | Iterable[tuple[float, float, float, float]]
    | Iterable[np.ndarray[tuple[Literal[4]], np.dtype[np.floating]]]
    | np.ndarray[tuple[*tuple[int, ...], Literal[3]], np.dtype[np.floating]]
)
"""Types that represent an array of orientations as quaternions."""

type moi_like = (
    tuple[float, float, float]
    | np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]
)
"""Types that represent an array of moments of inertia ("MoI").

Following the convention in HOOMD-blue, MoIs are expressed as 3-vectors
containing the diagonal terms of the MoI tensor.
"""
