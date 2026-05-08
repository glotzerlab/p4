import os
from typing import Iterable

import coxeter
import gsd
import hoomd
import numpy as np

from p4 import Body

class Configuration:
    def __init__(
        self,
        bodies: list[Body],
        positions_by_type: dict[str, list[list[float]]],
        orientations_by_type: dict[str, list[list[float]]] = {}
    ):
        # Ensure bodies is the right type [TODO: consider duck-typing instead]
        if not (
            isinstance(bodies, Iterable)
            or all(isinstance(b, Body) for b in bodies)
        ):
            raise TypeError("`bodies` must be a list of Bodies.")

        # Ensure positions are provided for all primary types
        primary_types = [b.primary_type for b in bodies]
        if ts := [t for t in primary_types if t not in positions_by_type]:
            raise ValueError(
                "Missing required keys in `positions_by_type`: "
                + f"'{"', '".join(ts)}'. Positions must be "
                + "provided for all bodies' primary types."
            )

        # Ensure orientations are either provided for all primary types, or
        # not provided at all
        if orientations_by_type:
            ts = [
                t for t in primary_types if t not in orientations_by_type
            ]
            if ts:
                raise ValueError(
                    f"Missing required keys in `orientations_by_type`: "
                    + f"'{"', '".join(ts)}'. If orientations are provided at "
                    " all, they must be provided for all bodies' primary types."
                )


        # Ensure that for every secondary type, the number of provided
        # orientations matches the number of provided positions
        if orientations_by_type:
            ts = [
                t
                for t in primary_types
                if len(positions_by_type[t]) != len(orientations_by_type[t])
            ]
            if ts:
                raise ValueError(
                    "Mismatched numbers of orientations and positions for the "
                    + "following body primary types: "
                    + f"'{"', '".join(ts)}'. For every primary type, the "
                    + "numbers of positions and orientations must be the same."
                )

        self.bodies = bodies
        self.positions_by_type = positions_by_type
        self.orientations_by_type = orientations_by_type

    # --------------------------------- IMPORT ---------------------------------

    @classmethod
    def from_gsd_frame(cls, frame: gsd.hoomd.Frame):
        pass

    @classmethod
    def from_hoomd_snapshot(cls, snapshot: hoomd.Snapshot):
        # Convert snapshot to gsd frame and then call from_gsd_frame
        pass

    @classmethod
    def from_hoomd_simulation(cls, simulation: hoomd.Simulation):
        pass

    @classmethod
    def from_json(
        cls,
        filename: os.PathLike,
        json_path: str | None = "p4.configuration"
    ):
        pass

    @classmethod
    def _convert_json_dict(cls, json_dict: dict):
        pass

    # --------------------------------- EXPORT ---------------------------------

    def to_gsd_frame(self):
        pass

    def to_hoomd_snapshot(self, ignore_types=[]):
        # call to_gsd_frame and then convert that to a snapshot
        pass

    # -------------------------------- PLOTTING --------------------------------

    def plot(
        self,
        type_shapes: dict[str, coxeter.shapes.Polyhedron] = {},
        type_styles: dict[str, dict] = {},
        ignore_types: list[str] = [],
        slice: dict[str, float] = {},
        schematic_slice: bool = False,
        schematic_slice_scale: float = 1,
        schematic_slice_color: str = "red",
        schematic_slice_opacity: float = 1,
        schematic_slice_line_width: float = 10,
        show_legend: bool = True,
    ):
        pass

    def _plot_traces_schematic_slice(
        self,
        particle_data: list,
        slice: dict[str, int],
        scale: float,
        color: str,
        opacity: float,
        line_width: float
    ) -> dict:
        pass

    def _plot_traces_3d(
        self,
        particle_data: np.ndarray,
        type_shapes: dict[str, coxeter.shapes.Polyhedron],
        type_styles: dict[str, dict],
        default_colors: list[str],
    ):
        pass

    def _plot_traces_2d(
        self,
        particle_data: np.ndarray,
        type_shapes: dict[str, coxeter.shapes.Polyhedron],
        type_styles: dict[str, dict],
        slice: dict[str, float],
        default_colors: list[str],
        point_size_for_slice: float = 1e-6
    ):
        pass

    def _plot_traces_1d(
        self,
        particle_data: np.ndarray,
        type_shapes: dict[str, coxeter.shapes.Polyhedron],
        type_styles: dict[str, dict],
        slice: dict[str, float],
        default_colors: list[str],
        point_size_for_slice: float = 1e-6
    ):
        pass