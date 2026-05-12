from copy import copy
import json
import os
from typing import Iterable
from pathlib import Path

import coxeter
import gsd
import hoomd
import numpy as np
import rowan

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
    def from_hoomd_simulation(cls, simulation: hoomd.Simulation):
        """Parse a HOOMD-blue `Simulation`_ to create a configuration.

        .. _Simulation: https://hoomd-blue.readthedocs.io/en/latest/hoomd/simulation.html
        
        Parameters
        ----------
        simulation : hoomd.Simulation
            The simulation to parse.
        """
        bodies = Body.from_hoomd_simulation(simulation)

        snapshot = simulation.state.get_snapshot()

        single_particle_conf = cls.from_hoomd_snapshot(snapshot)

        positions_by_type = {
            t: single_particle_conf.positions_by_type[t]
            for t in single_particle_conf.positions_by_type
            if any(t == b.primary_type for b in bodies)
        }
        orientations_by_type = {
            t: single_particle_conf.orientations_by_type[t]
            for t in single_particle_conf.orientations_by_type
            if any(t == b.primary_type for b in bodies)
        }

        return cls(
            bodies=bodies,
            positions_by_type=positions_by_type,
            orientations_by_type=orientations_by_type
        )

    @classmethod
    def from_hoomd_snapshot(cls, snapshot: hoomd.Snapshot):
        """Parse a HOOMD-blue `Snapshot`_ to create a configuration.

        .. _Snapshot: https://hoomd-blue.readthedocs.io/en/latest/hoomd/snapshot.html

        A snapshot does not contain rigid body constraint data, so it is parsed
        into single-particle bodies.
        
        Parameters
        ----------
        snapshot : hoomd.Snapshot
            The snapshot to parse.
        """
        frame = gsd.hoomd.Frame()

        frame.particles.N = snapshot.particles.position.shape[0]
        frame.particles.types = snapshot.particles.types
        frame.particles.typeid = snapshot.particles.typeid
        frame.particles.position = snapshot.particles.position
        frame.particles.orientation = snapshot.particles.orientation
        
        return cls.from_gsd_frame(frame)


    @classmethod
    def from_gsd_frame(cls, frame: gsd.hoomd.Frame):
        """Parse a GSD `Frame`_ to create a configuration.

        .. _Frame: https://gsd.readthedocs.io/en/latest/python-module-gsd.hoomd.html#gsd.hoomd.Frame

        A Frame does not contain rigid body constraint data, so it is parsed
        into single-particle bodies.
        
        Parameters
        ----------
        frame : gsd.hoomd.Frame
            The frame to parse.
        """
        types = frame.particles.types
        typeids = frame.particles.typeid
        positions = frame.particles.position
        orientations = frame.particles.orientation

        bodies = [Body(t) for t in types]

        positions_by_type = {
            t: [p for tid, p in zip(typeids, positions) if types[tid] == t]
            for t in types
        }

        orientations_by_type = {
            t: [p for tid, p in zip(typeids, orientations) if types[tid] == t]
            for t in types
        }

        return cls(
            bodies=bodies,
            positions_by_type=positions_by_type,
            orientations_by_type=orientations_by_type
        )

    @classmethod
    def from_json(
        cls,
        filename: os.PathLike,
        json_path: str = "p4.configuration"
    ):
        """Create a configuration from JSON.

        a JSON path may be provided to control the location that the
        configuration data is retrieved from. See
        :meth:`~p4.Configuration.to_json` for an explanation of JSON path
        formatting.

        .. note::
            The internal data structures for the ``Configuration`` class have
            native JSON analogues, so the JSON representation is simply
            ``Configuration.__dict__``.

        Parameters
        ----------
        filename : os.PathLike
            The name or path of the JSON file.
        json_path : str, default='p4.configuration'
            The location within the JSON file to retrieve the configuration's
            representation from.
        
        Raises
        ------
        ValueError
            If the JSON file does not have the keys and values required for
            instantiating a Configuration.
        """
        with open(filename, "r") as f:
            data = json.load(f)

        if json_path == ".":
            data = cls._convert_json_dict(data)            
        
        else:
            current_container = data
            for name in json_path.split("."):
                current_container = current_container[name]
            data = cls._convert_json_dict(current_container)
        
        required_args = [
            "bodies",
            "positions_by_type",
            "orientations_by_type"
        ]
        for required_arg in required_args:
            if required_arg not in data:
                raise ValueError(
                    f"Required arg {required_arg} not found in '{filename}' "
                    + f"at path '{json_path}'."
                )
        for key in copy(data):
            if key not in required_args:
                del data[key]

        return cls(**data)

    @classmethod
    def _convert_json_dict(cls, json_dict: dict):
        """Convert a JSON-compliant dict into an instantiation-ready dict."""
        data = copy(json_dict)
        data["bodies"] = [Body(**b) for b in data["bodies"]]
        return json_dict

    # --------------------------------- EXPORT ---------------------------------

    def to_hoomd_snapshot(self, ignore_types: list[str] = []):
        """Convert the configuration to a HOOMD-blue `Snapshot`_.
        
        .. _Snapshot: https://hoomd-blue.readthedocs.io/en/latest/hoomd/snapshot.html

        Parameters
        ----------
        ignore_types : list[str], default=[]
            Primary types to omit from the snapshot.
        """
        frame = self.to_gsd_frame(ignore_types)
        return hoomd.Snapshot.from_gsd_frame(
            gsd_snap=frame,
            communicator=hoomd.communicator.Communicator()
        )

    def to_gsd_frame(self, ignore_types: list[str] = []):
        """Convert the configuration to a GSD `Frame`_.
        
        .. _Frame: https://gsd.readthedocs.io/en/latest/python-module-gsd.hoomd.html#gsd.hoomd.Frame

        Parameters
        ----------
        ignore_types : list[str], default=[]
            Primary types to omit from the frame.
        """
        types = []
        typeids = []
        positions = np.empty((0, 3), dtype=np.float32)
        orientations = np.empty((0, 4), dtype=np.float32)

        for body in self.bodies:
            types.extend([body.primary_type] + body.secondary_types)
            
            body_positions = self.positions_by_type[body.primary_type]
            body_orientations = self.orientations_by_type[body.primary_type]
            
            for primary_p, primary_o in zip(body_positions, body_orientations):
                # Primary particle
                typeids.append(types.index(body.primary_type))
                positions = np.vstack((positions, primary_p))
                orientations = np.vstack((orientations, primary_o))
                
                # Secondary particles
                for secondary_t in body.secondary_types:
                    secondary_ps = body.positions_by_type[secondary_t]
                    secondary_os = body.orientations_by_type.get(
                        secondary_t,
                        np.array([(1, 0, 0, 0) for _ in secondary_ps])
                    )

                    # typeids
                    typeids.extend([types.index(secondary_t) for _ in secondary_ps])

                    # positions
                    positions = np.vstack((
                        positions,
                        rowan.rotate(primary_o, secondary_ps) + primary_p
                    ))

                    # orientations
                    orientations = np.vstack((
                        orientations,
                        rowan.multiply(primary_o, secondary_os)
                    ))

        frame = gsd.hoomd.Frame()

        frame.particles.N = positions.shape[0]
        frame.particles.types = types
        frame.particles.typeid = typeids
        frame.particles.position = positions
        frame.particles.orientation = orientations

        return frame

    def to_json(
        self,
        filename: os.PathLike,
        json_path: str = "p4.configuration",
        indent: str | int | None = None
    ):
        """Export the configuration to JSON.
        
        If ``filename`` points to an existing file, a JSON path may be provided
        to ensure the configuration data does not clash with existing data in
        the file.

        A JSON path that looks like ``'a.b.c'`` represents the following
        location:

        .. code-block::

            <root>
            └─ a
               └─ b
                  └─ c
                     └─ <data will go here>

        If the path specifies a location that already contains data, the
        contents of that location may be overwritten.

        .. note::
            The internal data structures for the ``Configuration`` class have
            native JSON analogues, so the JSON representation is simply
            ``Configuration.__dict__``.
                     
        Parameters
        ----------
        filename : os.PathLike
            The name or path of the JSON file.
        json_path : str or None, default='p4.configuration'
            The location within the JSON file to put the configuration's
            representation in. Only used if ``filename`` already exists. If
            ``'.'`` is provided, then the representation is placed at the root
            level.
        indent : str or int, optional
            The string or number of spaces to use when indenting newlines in the
            JSON file. If not provided, there are no newlines.
        """
        path = Path(filename)
        data = self._to_json_dict()

        if not path.exists():
            path.touch()
            existing_data = {}
        else:
            with open(path, "r") as f:
                existing_data = json.load(f)
            
        if json_path == ".":
            for k, v in data.items():
                existing_data[k] = v
        
        else:
            names = json_path.split(".")
            current_container = existing_data
            for i, name in enumerate(names):
                if name not in current_container:
                    current_container[name] = {}
                if i < (len(names) - 1):
                    current_container = current_container[name]
                else:
                    # try to write alongside existing data if possible...
                    if isinstance(current_container[name], dict):
                        current_container[name].update(data)
                    elif isinstance(current_container[name], list):
                        current_container[name].append(data)
                    # ... and insert or overwrite if not
                    else:
                        current_container[name] = data

        with open(path, "w") as f:
            json.dump(existing_data, f, indent=indent)

    def _to_json_dict(self):
        """Return a JSON-compliant dictionary representing this configuration."""
        data = copy(self.__dict__)
        data["bodies"] = [b._to_json_dict() for b in self.bodies]
        return data

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