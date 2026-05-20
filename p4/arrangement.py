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

from . import util
from .body import Body

class Arrangement:
    def __init__(
        self,
        bodies: list[Body],
        positions_by_type: dict[str, list[list[float]]],
        orientations_by_type: dict[str, list[list[float]]] | None = None
    ):
        self._bodies = bodies
        self._positions_by_type = util.sanitize(positions_by_type)
        if orientations_by_type:
            self._orientations_by_type = util.sanitize(orientations_by_type)
        else:
            self._orientations_by_type = {}
        
        self.validate()

    def validate(self):
        """Ensure this arrangement adheres to the :ref:`arrangement schema`."""
        # Ensure bodies is the right type
        if not (
            isinstance(self.bodies, Iterable)
            and all(isinstance(b, Body) for b in self.bodies)
        ):
            raise TypeError("`bodies` must be a list of Bodies.")

        # Ensure positions are provided for all primary types
        primary_types = [b.primary_type for b in self.bodies]
        if ts := [t for t in primary_types if t not in self.positions_by_type]:
            raise ValueError(
                "Missing required keys in `positions_by_type`: "
                + f"'{"', '".join(ts)}'. Positions must be "
                + "provided for all bodies' primary types."
            )

        # Ensure that for every secondary type, the number of provided
        # orientations matches the number of provided positions
        if self.orientations_by_type:
            ts = [
                t
                for t in primary_types
                if (
                    t in self.orientations_by_type
                    and (
                        len(self.positions_by_type[t])
                        != len(self.orientations_by_type[t])
                    )
                )
            ]
            if ts:
                raise ValueError(
                    "Mismatched numbers of orientations and positions for the "
                    + "following body primary types: "
                    + f"'{"', '".join(ts)}'. For every primary type, the "
                    + "numbers of positions and orientations must be the same."
                )

    # ------------------------------- PROPERTIES -------------------------------

    @property
    def bodies(self) -> list[Body]:
        """The bodies present in this arrangement."""
        return self._bodies

    @bodies.setter
    def bodies(self, value):
        """Set the bodies present in this arrangement."""
        self._bodies = value

    @property
    def positions_by_type(self) -> dict[str, list[list[float]]]:
        """A mapping from body primary types to positions in 3D space."""
        return self._positions_by_type

    @positions_by_type.setter
    def positions_by_type(self, value):
        """Set a mapping from body primary types to positions in 3D space."""
        self._positions_by_type = util.sanitize(value)

    @property
    def orientations_by_type(self) -> dict[str, list[list[float]]]:
        """A mapping from body primary types to orientations as quaternions.
        
        If not specified for a type, defaults to an array of ``(1,0,0,0)``
        quaternions.
        """
        return self._orientations_by_type

    @orientations_by_type.setter
    def orientations_by_type(self, value):
        """Set a mapping from body primary types to orientations as quaternions.
        """
        self._orientations_by_type = util.sanitize(value)

    # --------------------------------- IMPORT ---------------------------------

    @classmethod
    def from_hoomd_simulation(
        cls,
        simulation: hoomd.Simulation,
        include_singles: bool = False
    ):
        """Parse a HOOMD-blue `Simulation`_ to create an arrangement.

        .. _Simulation: https://hoomd-blue.readthedocs.io/en/latest/hoomd/simulation.html

        The returned list contains only bodies defined in the simulation's
        state. Therefore this method is equivalent to

        .. code-block:: python

            arrangement.from_hoomd_snapshot(
                simulation.state.get_snapshot(),
                include_singles
            )

        Parameters
        ----------
        simulation : hoomd.Simulation
            The simulation to parse.
        include_singles : bool, default=False
            Whether to include single-particle bodies when parsing the
            simulation.
        """
        if not simulation.state:
            raise ValueError("`simulation` has no state.")
        return cls.from_hoomd_snapshot(
            snapshot=simulation.state.get_snapshot(),
            include_singles=include_singles
        )

    @classmethod
    def from_hoomd_snapshot(
        cls,
        snapshot: hoomd.Snapshot,
        include_singles: bool = False
    ):
        """Parse a HOOMD-blue `Snapshot`_ to create an arrangement.

        .. _Snapshot: https://hoomd-blue.readthedocs.io/en/latest/hoomd/snapshot.html

        Body definitions in a snapshot are encoded in `particles.data`_, an
        array of integers that specify body ids as the indices of central
        particles.

        .. _particles.data: https://gsd.readthedocs.io/en/latest/schema-hoomd.html#chunk-particles-body
        
        Parameters
        ----------
        snapshot : hoomd.Snapshot
            The snapshot to parse.
        include_singles : bool, default=False
            Whether to include single-particle bodies when parsing the
            snapshot.
        """
        bodies = Body.from_hoomd_snapshot(snapshot, include_singles)

        if not include_singles:
            bodies = [b for b in bodies if b.secondary_types]

        positions_by_type = {}
        orientations_by_type = {}

        for b in bodies:
            tid = snapshot.particles.types.index(b.primary_type)
            indices = np.argwhere(snapshot.particles.typeid == tid).flatten()
            positions = snapshot.particles.position[indices]
            orientations = snapshot.particles.orientation[indices]

            positions_by_type[b.primary_type] = positions.tolist()

            if not (orientations == [1, 0, 0, 0]).all(axis=1).all():
                orientations_by_type[b.primary_type] = orientations.tolist()

        return cls(
            bodies=bodies,
            positions_by_type=positions_by_type,
            orientations_by_type=orientations_by_type
        )

    @classmethod
    def from_gsd(cls, filename: os.PathLike, index: int = -1):
        """Parse a GSD file to create an arrangement from an indexed frame.

        .. _Frame: https://gsd.readthedocs.io/en/latest/python-module-gsd.hoomd.html#gsd.hoomd.Frame

        A Frame does not contain rigid body constraint data, so it is parsed
        into single-particle bodies.
        
        TODO: mention that box must be big enough to prevent image problem because images
        are not present in GSD file - the onus for this is on the user

        Parameters
        ----------
        filename : os.PathLike
            The name or path of the GSD file.
        index : int, default=-1
            The index of the frame to parse. Defaults to the last frame in the
            file.
        """
        with gsd.hoomd.open(filename, "r") as f:
            frame = f[index]
        
        snapshot = hoomd.Snapshot.from_gsd_frame(
            gsd_snap=frame,
            communicator=hoomd.communicator.Communicator()
        )
        
        return cls.from_hoomd_snapshot(snapshot)

    @classmethod
    def from_json(
        cls,
        filename: os.PathLike,
        json_path: str = "p4.arrangement"
    ):
        """Create an arrangement from JSON.

        a JSON path may be provided to control the location that the
        arrangement data is retrieved from. See
        :meth:`~p4.Arrangement.to_json` for an explanation of JSON path
        formatting.

        .. note::
            The internal data structures for the ``Arrangement`` class have
            native JSON analogues, so the JSON representation is simply
            ``Arrangement.__dict__``.

        Parameters
        ----------
        filename : os.PathLike
            The name or path of the JSON file.
        json_path : str, default='p4.arrangement'
            The location within the JSON file to retrieve the arrangement's
            representation from.
        
        Raises
        ------
        ValueError
            If the JSON file does not have the keys and values required for
            instantiating a Arrangement.
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
        return data

    # --------------------------------- EXPORT ---------------------------------

    def to_hoomd_rigid(
        self,
        rigid: hoomd.md.constrain.Rigid | None = None,
    ) -> hoomd.md.constrain.Rigid:
        """Convert the arrangement to a HOOMD-blue `rigid constraint`_.

        An existing rigid constraint may be passed to this method, in which case
        this arrangement is merely added to it.

        .. _rigid constraint: https://hoomd-blue.readthedocs.io/en/stable/hoomd/md/constrain/rigid.html

        Parameters
        ----------
        rigid : hoomd.md.constrain.Rigid, optional
            An existing constraint instance to use. If not provided, a new one
            is created.
        """
        if rigid is None:
            rigid = hoomd.md.constrain.Rigid()
        
        for b in self.bodies:
            rigid = b.to_hoomd_rigid(rigid)

        return rigid 

    def to_hoomd_snapshot(self):
        """Convert the arrangement to a HOOMD-blue `Snapshot`_.
        
        .. _Snapshot: https://hoomd-blue.readthedocs.io/en/latest/hoomd/snapshot.html
        """
        # Create initial state with only primary type particles
        types = []
        typeids = []
        positions = np.empty((0, 3), dtype=np.float32)
        orientations = np.empty((0, 4), dtype=np.float32)
        masses = []
        mois = np.empty((0, 3), dtype=np.float32)
        bodyids = []

        # Loop over each type of body
        for body in self.bodies:
            types.extend([body.primary_type] + body.secondary_types)
            
            # Calculate quantities that do not vary between instances
            body_typeids = [types.index(body.primary_type)]
            body_masses = [body.mass_by_type.get(body.primary_type, 1)]
            body_mois = np.array(
                [body.moi_by_type.get(body.primary_type, [1, 1, 1])],
                dtype=np.float32
            )
            for t, ps in body.positions_by_type.items():
                body_typeids.extend([types.index(t) for _ in ps])
                body_masses.extend([body.mass_by_type.get(t, 1) for _ in ps])
                body_mois = np.vstack((
                    body_mois,
                    [body.moi_by_type.get(t, [1, 1, 1]) for _ in ps]
                ))

            # Add quantities that do not vary between instances
            for _ in self.positions_by_type[body.primary_type]:
                typeids.extend(body_typeids)
                masses.extend(body_masses)
                mois = np.vstack((mois, body_mois))

            # Calculate the positions and orientations for all instances of
            # this body
            instance_primary_positions = self.positions_by_type[
                body.primary_type
            ]
            instance_primary_orientations = self.orientations_by_type.get(
                body.primary_type,
                np.array([[1, 0, 0, 0] for _ in instance_primary_positions])
            )
            
            # Loop over each instance
            for i, primary_p in enumerate(instance_primary_positions):
                body_instance_id = positions.shape[0]    # equals primary's tag
                primary_o = instance_primary_orientations[i]

                # Add primary particle's bodyid, position and orientation
                bodyids.append(body_instance_id)
                positions = np.vstack((positions, primary_p))
                orientations = np.vstack((orientations, primary_o))
                
                # Add secondary particles' bodyids, positions and orientations
                for secondary_t in body.secondary_types:
                    secondary_ps = body.positions_by_type[secondary_t]
                    secondary_os = body.orientations_by_type.get(
                        secondary_t,
                        [(1, 0, 0, 0) for _ in secondary_ps]
                    )

                    bodyids.extend([body_instance_id for _ in secondary_ps])
                    positions = np.vstack((
                        positions,
                        rowan.rotate(primary_o, secondary_ps) + primary_p
                    ))
                    orientations = np.vstack((
                        orientations,
                        rowan.multiply(primary_o, secondary_os)
                    ))

        # Insert the calculated particle data into a GSD Frame, then convert the
        # Frame to a HOOMD Snapshot
        frame = gsd.hoomd.Frame()
        frame.configuration.box = [
            3*max(np.abs(positions[:,0].max()), np.abs(positions[:,0].min())+1),
            3*max(np.abs(positions[:,1].max()), np.abs(positions[:,1].min())+1),
            3*max(np.abs(positions[:,2].max()), np.abs(positions[:,2].min())+1),
            0.0,
            0.0,
            0.0
        ]
        frame.particles.N = positions.shape[0]
        frame.particles.types = types
        frame.particles.typeid = typeids
        frame.particles.position = positions
        frame.particles.orientation = orientations
        frame.particles.mass = masses
        frame.particles.moment_inertia = mois
        frame.particles.body = bodyids

        return hoomd.Snapshot.from_gsd_frame(
            gsd_snap=frame,
            communicator=hoomd.communicator.Communicator()
        )

    def to_gsd(
        self,
        filename: os.PathLike,
        type_shapes: dict | None = None
    ):
        """Export the arrangement to a frame in a GSD file.

        Parameters
        ----------
        filename : os.PathLike
            The name or path of the GSD file.
        type_shapes: dict[str, coxeter.shapes], optional
            If provided, encodes geometry for provided particle types. Specify
            a geometry using Coxeter's `shapes module`_.
        
        .. _shapes module: https://coxeter.readthedocs.io/en/latest/package-shapes.html
        """
        snapshot = self.to_hoomd_snapshot()

        frame = util.snapshot_to_frame(snapshot)

        if type_shapes:
            gsd_shape_specs = []
            for t in frame.particles.types:
                s = type_shapes.get(t)
                accepted_types = (
                    coxeter.shapes.Sphere,
                    coxeter.shapes.Ellipsoid,
                    coxeter.shapes.Polygon,
                    coxeter.shapes.Polyhedron,
                )
                if isinstance(s, (accepted_types)):
                    gsd_shape_specs.append(s.gsd_shape_spec)
                else:
                    gsd_shape_specs.append({})

            frame.particles.type_shapes = gsd_shape_specs

        with gsd.hoomd.open(filename, "w") as f:
            f.append(frame)

    def to_json(
        self,
        filename: os.PathLike,
        json_path: str = "p4.arrangement",
        indent: str | int | None = None
    ):
        """Export the arrangement to JSON.
        
        If ``filename`` points to an existing file, a JSON path may be provided
        to ensure the arrangement data does not clash with existing data in
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
            The internal data structures for the ``Arrangement`` class have
            native JSON analogues, so the JSON representation is simply
            ``Arrangement.__dict__``.
                     
        Parameters
        ----------
        filename : os.PathLike
            The name or path of the JSON file.
        json_path : str or None, default='p4.arrangement'
            The location within the JSON file to put the arrangement's
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
        """Return a JSON-compliant dictionary representing this arrangement."""
        return dict(
            bodies=[b._to_json_dict() for b in self.bodies],
            positions_by_type=self.positions_by_type,
            orientations_by_type=self.orientations_by_type
        )

    # -------------------------------- PLOTTING --------------------------------

    def plot(
        self,
        type_shapes: dict[str, coxeter.shapes.Polyhedron] | None = None,
        type_styles: dict[str, dict] | None = None,
        ignore_types: list[str] | None = None,
        slice: dict[str, float] | None = None,
        schematic_slice: bool = False,
        schematic_slice_scale: float = 1,
        schematic_slice_color: str = "red",
        schematic_slice_opacity: float = 1,
        schematic_slice_line_width: float = 10,
        show_legend: bool = True,
    ):
        # TODO
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

    # --------------------------------- OTHER ----------------------------------
    
    def __repr__(self):
        return (
            "Arrangement ("
            + f"\n\tbodies='{self.bodies}',"
            + f"\n\tpositions_by_type={self.positions_by_type},"
            + f"\n\torientations_by_type={self.orientations_by_type},"
            + "\n)"
        )
    
    def __eq__(self, other):
        """Arrangements are equal if their properties are identical."""
        bodies_equivalent = (
            len(self.bodies) == len(other.bodies)
            and all([i in other.bodies for i in self.bodies])
        )
        positions_by_type_equivalent = (
            self.positions_by_type.keys() == other.positions_by_type.keys()
            and all(
                self.positions_by_type[t] == other.positions_by_type[t]
                for t in self.positions_by_type
            )
        )
        orientations_by_type_equivalent =  (
            all(
                (
                    self.orientations_by_type.get(t, [1, 0, 0, 0])
                    == other.orientations_by_type.get(t, [1, 0, 0, 0])
                )
                for t in set(
                    self.orientations_by_type.keys()
                ).union(other.orientations_by_type.keys())
            )
            or (
                self.orientations_by_type == {}
                and all(
                    list(i) == [1, 0, 0, 0]
                    for v in other.orientations_by_type.values()
                    for i in v
                )
            )
            or (
                other.orientations_by_type == {}
                and all(
                    list(i) == [1, 0, 0, 0]
                    for v in self.orientations_by_type.values()
                    for i in v
                )
            )
        )
        return (
            bodies_equivalent
            and positions_by_type_equivalent
            and orientations_by_type_equivalent
        )
