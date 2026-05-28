# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

"""Type aliases."""

from collections.abc import Iterable
from typing import Literal

import hoomd
import gsd
import numpy as np


type state_like = (hoomd.State | hoomd.Snapshot | gsd.hoomd.Frame)

type bodies_holder = (hoomd.md.constrain.Rigid | state_like)

type axis_like = (
    list[float]
    | tuple[float, float, float]
    | np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]
)

type positions_like = (
    Iterable[list[float]]
    | Iterable[tuple[float, float, float]]
    | Iterable[np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]]
    | np.ndarray[tuple[*tuple[int, ...], Literal[3]], np.dtype[np.floating]]
)

type orientations_like = (
    Iterable[list[float]]
    | Iterable[tuple[float, float, float, float]]
    | Iterable[np.ndarray[tuple[Literal[4]], np.dtype[np.floating]]]
    | np.ndarray[tuple[*tuple[int, ...], Literal[3]], np.dtype[np.floating]]
)

type moi_like = (
    tuple[float, float, float]
    | np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]
)
