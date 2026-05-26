# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from copy import copy, deepcopy
import json
import os
from pathlib import Path
from typing import Iterable, Literal
import multiprocessing

import hoomd

from . import util
from .body import Body
from .interaction import Interaction
from .arrangement import Arrangement

class System:
    """A System is defined by a probe, an analyte, and their interactions.
    
    Measure a system's spatial distributions of potential energy, force, and
    torque using :meth:`~p4.System.measure`.

    .. code-block:: python
        :caption: A system composed of a probe point particle 'A' and a an analyte cubic body 'B' with secondary particles 'C'. 'A' and 'C' interact via an LJ potential.

        import p4
        import hoomd
        
        probe = p4.Body("A")
        analyte = p4.Body(
            primary_type="B",
            secondary_types=["C"],
            positions_by_type=dict(
                C=[
                    [-1, -1, -1],
                    [-1, -1,  1],
                    [-1,  1, -1],
                    [-1,  1,  1],
                    [ 1, -1, -1],
                    [ 1, -1,  1],
                    [ 1,  1, -1],
                    [ 1,  1,  1]
                ]
            )
        )
        interactions = [
            p4.Interaction(
                hoomd_class=hoomd.md.pair.LJ,
                initial_args=dict(),
                default_params=dict(
                    r_cut=0,
                    params=dict(epsilon=0, sigma=1)
                ),
                typed_params={
                    ("A", "C"): dict(
                        r_cut=5.0,
                        params=dict(epsilon=1, sigma=1)
                    )
                }
            )
        ]
        system = p4.System(probe, analyte, interactions)

    Parameters
    ----------
    probe : Body
        The body for the probe.
    analyte : Body | Arrangement
        The body or arrangement for the analyte.
    interactions : list[Interaction]
        A collection of interactions that define how one or more particle types
        in the probe interact with one or more types in the analyte.
    """
    def __init__(
        self,
        probe: Body,
        analyte: Body | Arrangement,
        interactions: list[Interaction],
    ):
        # Validate inputs
        self.validate(
            probe=probe,
            analyte=analyte,
            interactions=interactions
        )

        # Set instance attributes
        self._probe = probe
        self._analyte = analyte
        self._interactions = interactions

    @classmethod
    def validate(
        cls,
        probe: Body,
        analyte: Body | Arrangement,
        interactions: list[Interaction],
    ):
        """Ensure this system adheres to the :ref:`system schema`."""
        # Ensure types are correct
        if not isinstance(probe, Body):
            raise TypeError("`probe` must be an instance of 'Body'.")
        
        if not isinstance(analyte, (Body, Arrangement)):
            raise TypeError(
                "`analyte` must be an instance of 'Body' or 'Arrangement'."
            )
        
        if (
            not isinstance(interactions, Iterable)
            or not all(type(i) is Interaction for i in interactions)
        ):
            raise TypeError(
                "`interactions` must be a list of Interaction instances."
            )
        
        # Ensure there are no body clashes
        if isinstance(analyte, Body):
            if (
                probe.primary_type == analyte.primary_type
                and probe != analyte
            ):
                raise ValueError(
                    "Clashing body definitions: if `probe` and `analyte` have "
                    + "the same primary type, they must be identical bodies."
                )
            if probe.primary_type in analyte.secondary_types:
                raise ValueError(
                    "Clashing body definitions: probe primary type must not "
                    + "be included in analyte secondary types."
                )
            if analyte.primary_type in probe.secondary_types:
                raise ValueError(
                    "Clashing body definitions: analyte primary type must not "
                    + "be included in probe secondary types."
                )

        elif isinstance(analyte, Arrangement):
            if (
                any(
                    probe.primary_type == b.primary_type for b in analyte.bodies
                )
                and probe not in analyte.bodies
            ):
                raise ValueError(
                    "Clashing body definitions: if `probe` and has the same "
                    + "primary type as a body in `analyte`, the probe must be "
                    + "identical to that body."
                )
            if any(
                probe.primary_type in b.secondary_types for b in analyte.bodies
            ):
                raise ValueError(
                    "Clashing body definitions: probe primary type must not "
                    + "be included in any analyte body's secondary types."
                )
            if any(
                b.primary_type in probe.secondary_types for b in analyte.bodies
            ):
                raise ValueError(
                    "Clashing body definitions: the probe secondary types "
                    + "must not include any of the analyte bodies' primary "
                    + "types."
                )

    # ------------------------------- PROPERTIES -------------------------------

    @property
    def probe(self) -> Body:
        """The system's probe.
        
        Measurements correspond to the energies and forces experienced by the
        probe.
        """
        return self._probe

    @probe.setter
    def probe(self, value):
        """Set the system's probe."""
        self.validate(
            probe=value,
            analyte=self.analyte,
            interactions=self.interactions
        )
        self._probe = value
        
        
    @property
    def analyte(self) -> Body | Arrangement:
        """The system's analyte.
        
        The analyte is a static entity (either a :py:class:`~p4.Body` or a
        :py:class:`~p4.Arrangement`) whose effective energy and force fields
        are measured by the probe.
        """
        return self._analyte

    @analyte.setter
    def analyte(self, value):
        """Set the system's analyte."""
        self.validate(
            probe=self.probe,
            analyte=value,
            interactions=self.interactions
        )
        self._analyte = value
        
    @property
    def interactions(self) -> list[Interaction]:
        """The interactions between particles in the probe and analyte."""
        return self._interactions

    @interactions.setter
    def interactions(self, value):
        """Set the interactions between particles in the probe and analyte."""
        self.validate(
            probe=self.probe,
            analyte=self.analyte,
            interactions=value
        )
        self._interactions = value

    # --------------------------------- IMPORT ---------------------------------

    @classmethod
    def from_hoomd_simulation(
        cls,
        simulation: hoomd.Simulation,
        probe_primary_type: str,
        analyte_primary_type: str | None = None
    ):
        """Parse a HOOMD-blue `Simulation`_ to create a system.

        .. _Simulation: https://hoomd-blue.readthedocs.io/en/latest/hoomd/simulation.html

        The simulation must have an integrator, and the integrator must have one
        or more forces. Optionally, the integrator may also have a rigid
        constraint. If it does have one, and if this constraint's keys
        include ``probe_primary_type`` or ``analyte_primary_type``, then the
        constraint is parsed to determine secondary types, positions, and
        orientations for the resulting probe and/or analyte.

        Parameters
        ----------
        simulation : hoomd.Simulation
            The simulation to parse.
        probe_primary_type : str
            The name of the particle type to use for the probe's primary type.
            This name must be represented in the simulation's current state.
        analyte_primary_type : str | None, optional
            The name of the particle type to use for the analyte's primary type.
            This name must be represented in the simulation's current state. If
            not provided, the entire simulation state becomes the analyte.
            (See :py:class`~p4.Arrangement`.)
        """
        if (
            not simulation.operations.integrator
            or not simulation.operations.integrator.forces 
        ):
            raise ValueError("simulation must have an integrator with forces.")

        types_in_state = simulation.state.get_snapshot().particles.types
        if probe_primary_type not in types_in_state:
            raise ValueError(
                "simulation state does not have particle type "
                + f"{probe_primary_type}"
            )
        if (
            isinstance(analyte_primary_type, str)
            and analyte_primary_type not in types_in_state
        ):
            raise ValueError(
                "simulation state does not have particle type "
                + f"{analyte_primary_type}"
            )
        
        interactions = Interaction.from_hoomd_simulation(simulation)

        bodies = Body.from_hoomd_simulation(
            simulation,
            include_singles=True
        )

        if not any(b.primary_type == probe_primary_type for b in bodies):
            probe = Body(probe_primary_type)
        else:
            probe = [
                b for b in bodies if b.primary_type == probe_primary_type
            ][0]
        
        if analyte_primary_type is None:
            analyte = Arrangement.from_hoomd_snapshot(
                simulation.state.get_snapshot()
            )
        else:
            if not any(b.primary_type == analyte_primary_type for b in bodies):
                analyte = Body(analyte_primary_type)
            else:
                analyte = [
                    b for b in bodies if b.primary_type == analyte_primary_type
                ][0]
        
        return cls(
            probe=probe,
            analyte=analyte,
            interactions=interactions
        )

    @classmethod
    def from_json(
        cls,
        filename: os.PathLike,
        json_path: str = "p4.system"
    ):
        """Create a system from JSON.

        a JSON path may be provided to control the location that the system
        data is retrieved from. See :meth:`~p4.System.to_json` for an
        explanation of JSON path formatting.
        
        Parameters
        ----------
        filename : os.PathLike
            The name or path of the JSON file.
        json_path : str, default='p4.system'
            The location within the JSON file to retrieve the system's
            representation from.
        
        Raises
        ------
        ValueError
            If the JSON file does not have the keys and values required for
            instantiating a System.
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
            "probe",
            "analyte",
            "interactions",
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
        
        data["probe"] = Body(**data["probe"])
        
        # TODO: see if there's a better way to distinguish a body json from
        # an arrangement json
        if "bodies" in data["analyte"]:
            data["analyte"] = Arrangement(**data["analyte"])
        else:
            data["analyte"] = Body(**data["analyte"])
        interactions = [
            Interaction(**i_dict) for i_dict in data["interactions"]
        ]
        data["interactions"] = interactions
        
        return cls(**data)

    @classmethod
    def _convert_json_dict(cls, data: dict):
        """Convert a JSON-compliant dict into an instantiation-ready dict."""
        data["probe"] = Body._convert_json_dict(data["probe"])
        
        # TODO: see if there's a better way to distinguish a body json from
        # an arrangement json
        if "bodies" in data["analyte"]:
            data["analyte"] = Arrangement._convert_json_dict(data["analyte"])
        else:
            data["analyte"] = Body._convert_json_dict(data["analyte"])

        data["interactions"] = [
            Interaction._convert_json_dict(i_dict)
            for i_dict in data["interactions"]
        ]

        return data

    # --------------------------------- EXPORT ---------------------------------

    def to_json(
        self,
        filename: os.PathLike,
        json_path: str = "p4.system",
        indent: str | int | None = None
    ):
        """Export the system to JSON.
        
        If ``filename`` points to an existing file, a JSON path may be specified
        to ensure the system data does not clash with existing data in the
        file.

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
                     
        Parameters
        ----------
        filename : os.PathLike
            The name or path of the JSON file.
        json_path : str or None, default='p4.system'
            The location within the JSON file to put the system's
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
        """Convert the system to a JSON-compliant dictionary."""
        data = {}

        data["probe"] = self.probe._to_json_dict()
        data["analyte"] = self.analyte._to_json_dict()
        data["interactions"] = [
            i._to_json_dict() for i in self.interactions
        ]

        return data

    # ------------------------------- PROPERTIES -------------------------------

    @property
    def active_interactions(self) -> list[Interaction]:
        """Interactions with typed params for both the probe and the analyte."""
        probe_types = [self.probe.primary_type] + self.probe.secondary_types
        if isinstance(self.analyte, Body):
            analyte_types = (
                [self.analyte.primary_type] + self.analyte.secondary_types
            )
        else:
            analyte_types = []
            for b in self.analyte.bodies:
                analyte_types.extend([b.primary_type] + b.secondary_types)
        
        included_interactions = []

        for interaction in self.interactions:
            # For single types, there must be at least one in probe and one in
            # analyte
            if (
                any(
                    t in probe_types
                    for t in interaction.interacting_types("single")
                )
                and any(
                    t in analyte_types
                    for t in interaction.interacting_types("single")
                )
            ):
                included_interactions.append(interaction)

            # For pair types, there must be at least one pair that contains a
            # type in the probe and a type (could be the same one) in the
            # analyte
            else:
                straddlers = []
                for type_pair in interaction.interacting_types("pair"):
                    if (
                        any(t in probe_types for t in type_pair)
                        and any(t in analyte_types for t in type_pair)
                    ):
                        straddlers.append(type_pair)
                if len(straddlers) > 0:
                    included_interactions.append(interaction)

        return included_interactions
    
    @property
    def all_types(self) -> list[str]:
        """All unique particle types in the probe and analyte."""
        all_types = [self.probe.primary_type]
        all_types.extend(self.probe.secondary_types)

        if isinstance(self.analyte, Body):
            all_types.append(self.analyte.primary_type)
            all_types.extend(self.analyte.secondary_types)
        else:
            for b in self.analyte.bodies:
                all_types.append(b.primary_type)
                all_types.extend(b.secondary_types)

        return list(set(all_types))

    # -------------------------------- MEASURE ---------------------------------

    def measure(
        self,
        quantities: Literal["U", "F", "T"] | list[Literal["U", "F", "T"]],
        position_resolutions: list[float],
        orientation_resolutions: list[list[float]],
        symmetries: list[int],
        csv_filename: str,
        nlist: hoomd.md.nlist.NeighborList,
        outside_cutoff: float,
        box_safety_factor: float = 100,
        n_processes: int = 1,
        save_gsd: bool = False,
    ):
        """Measure named quantities for the system.

        Parameters
        ----------
        quantities : one or more of 'U', 'F', 'T'
            The quantities to measure. 'U' is the potential energy measured for
            the entire system, and is saved as a single scalar quantity. 'F' and
            'T' are the net Force and Torque experienced by the probe, and are
            saved as vector quantities.
        position_resolutions : list[float]
            The number of samples along each dimension :math:`[X, Y, Z]` of the
            position grid.
        orientation_resolutions : list[float]
            The number of samples along each dimension :math:`[X, Y, Z]` of the
            axis-angle orientation grid.
        orientation_symmetries : list[int]
            The rotational symmetry for each axis $[X, Y, Z]$. If not provided,
            C1 symmetry is assumed for every axis. 
        csv_filename : str
            The name of the CSV file to save.
        nlist : hoomd.md.nlist.NeighborList
            The neighbor list to use for the interactions.
        outside_cutoff : float
            The cutoff distance outside which no positions will be probed. This
            distance represents the side lengths of a cube centered on the
            origin.
        box_safety_factor : float, default=100
            The scale factor for the simulation box, since it must be bigger
            than the probe box to prevent the minimum image problem. Defaults
            to 100, which should be sufficient in most cases.
        n_processes : int, default=1
            The number of processes to distribute the probe operation between.
            Parallelization is implemented at the Python level, so each process
            creates and runs its own simulation and then the table results are
            combined in the output CSV. Note that if ``save_gsd`` is set to
            True, each simulation will produce a separate GSD file. Set this
            parameter to -1 to use the maximum allowed number of processes for
            your machine.
        save_gsd : bool, default=False
            Whether to save a GSD file alongside the output CSV file. If True,
            the GSD has the same name as the CSV. The name of the GSD file will
            be almost identical to that of the CSV file, with a suffix
            the process whose simulation wrote the GSD file. This option is
            available for debugging purposes, but generally should not be used.
        """
        # Calculate probe box based on cutoff distances
        probe_box = [outside_cutoff, outside_cutoff, outside_cutoff]
        
        # Determine the frame's box from the probe box
        simulation_box = [d * box_safety_factor for d in probe_box]
        simulation_box.extend([0, 0, 0])

        # Figure out number of processes that will be used formultiprocessing.
        # Each process will ultimately receive its own simulation.
        if n_processes < -1 or n_processes == 0:
            raise ValueError("`n_processes` must be -1 or a positive integer.")

        if n_processes == -1:
            n_processes = os.process_cpu_count()
        
        # Calculate the probe positions and orientations
        probe_positions = util.get_probe_positions(
            probe_box,
            position_resolutions
        )
        probe_orientations = util.get_probe_orientations(
            orientation_resolutions,
            symmetries
        )

        # Remove positions that are too far away
        probe_positions = util.exclude_positions_by_shape(
            positions=probe_positions,
            exclude_inside=False,
            shape=util.get_cube(outside_cutoff),
            buffer=0.0
        )

        # If multiprocessing, run copies of the probe simulation with chunks
        # of the set of positions across a collection of processes
        if n_processes != 1:
            with multiprocessing.Pool(processes=n_processes) as pool:
                if save_gsd:
                    gsd_filenames = [
                        csv_filename.rsplit(".", 1)[0] + f"_{i}.gsd"
                        for i in range(n_processes)
                    ]
                else:
                    gsd_filenames = [None for _ in range(n_processes)]

                args = zip(
                    [deepcopy(self) for _ in range(n_processes)],
                    [quantities for _ in range(n_processes)],
                    util.subdivide(probe_positions, n_processes),
                    [probe_orientations for _ in range(n_processes)],
                    [self.active_interactions for _ in range(n_processes)],
                    [nlist for _ in range(n_processes)],
                    [probe_box for _ in range(n_processes)],
                    [simulation_box for _ in range(n_processes)],
                    gsd_filenames,
                )
                tables = pool.starmap(util.measure, args)
            
            table = util.clean_header(util.merge_tables(tables))
        
        # If not multiprocessing, don't initialize a pool (easier for debugging)
        else:
            if save_gsd:
                gsd_filename = csv_filename.rsplit(".", 1)[0] + ".gsd"
            else:
                gsd_filename = None
            table = util.measure(
                system=self,
                quantities=quantities,
                positions=probe_positions,
                orientations=probe_orientations,
                included_interactions=self.active_interactions,
                nlist=nlist,
                measurement_box=probe_box,
                simulation_box=simulation_box,
                gsd_filename=gsd_filename
            )

            table = util.clean_header(table)

        with open(csv_filename, "w") as file:
            table.seek(0)
            file.write(table.read())

    # --------------------------------- OTHER ----------------------------------

    def __eq__(self, other):
        """Systems are equal if their settable properties are equal."""
        return (
            self.interactions == other.interactions
            and self.probe == other.probe
            and self.analyte == other.analyte
        )
