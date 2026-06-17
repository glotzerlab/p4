# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from copy import copy, deepcopy
import json
import os
from pathlib import Path
from typing import Iterable, Literal
import multiprocessing

import hoomd
import numpy as np
from tqdm import tqdm

from . import util
from .types import PositionsLike, OrientationLike, OrientationsLike
from .body import Body
from .interaction import Interaction
from .arrangement import Arrangement
from .field import Field

class System:
    """A System is defined by a probe, an analyte, and their interactions.

    Instantiate a system directly using its constructor, or create one by parsing
    parsing an existing HOOMD-blue simulation using
    :py:meth:`~p4.system.System.from_hoomd_simulation`. Systems can also be saved
    to and created from JSON files using
    :py:meth:`~p4.system.System.to_json` and
    :py:meth:`~p4.system.System.from_json`.

    Measure a system's U, F, and T fields using
    :py:meth:`~p4.system.System.measure`.

    This class is self-validating: it cannot be instantiated or modified without
    adhering to the :ref:`system-schema`. This rule is enforced by
    :py:meth:`~p4.system.System.validate`.

    Parameters
    ----------
    probe : Body
        The probe, which is translated and rotated around the system's analyte
        during the field measurement process. Recorded F and T values correspond
        to F and T exerted on the probe by the analyte. 
    analyte : Body | Arrangement
        The analyte, which remains static at the simulation box's center during
        the field measurement process.
    interactions : list[Interaction]
        The interactions that collectively produce U, F, and T fields when the
        probe and analyte are within ``r_cut``.


    Example
    -------

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
    """
    def __init__(
        self,
        probe: Body,
        analyte: Body | Arrangement,
        interactions: list[Interaction],
    ):
        # Set instance attributes
        self._probe = probe
        self._analyte = analyte
        self._interactions = interactions

        # Validate instance attributes
        self.validate()

    def validate(self):
        """Ensure this system adheres to the :ref:`system-schema`."""
        # Ensure types are correct
        if not isinstance(self.probe, Body):
            raise TypeError("`probe` must be an instance of 'Body'.")
        
        if not isinstance(self.analyte, (Body, Arrangement)):
            raise TypeError(
                "`analyte` must be an instance of 'Body' or 'Arrangement'."
            )
        
        if (
            not isinstance(self.interactions, Iterable)
            or not all(type(i) is Interaction for i in self.interactions)
        ):
            raise TypeError(
                "`interactions` must be a list of Interaction instances."
            )
        
        # Ensure there are no body clashes
        if isinstance(self.analyte, Body):
            if (
                self.probe.primary_type == self.analyte.primary_type
                and self.probe != self.analyte
            ):
                raise ValueError(
                    "Clashing body definitions: if `probe` and `analyte` have "
                    + "the same primary type, they must be identical bodies."
                )
            if self.probe.primary_type in self.analyte.secondary_types:
                # TODO: is this still necessary for a single-particle probe?
                raise ValueError(
                    "Clashing body definitions: probe primary type must not "
                    + "be included in analyte secondary types."
                )
            if self.analyte.primary_type in self.probe.secondary_types:
                raise ValueError(
                    "Clashing body definitions: analyte primary type must not "
                    + "be included in probe secondary types."
                )

        elif isinstance(self.analyte, Arrangement):
            if (
                any(
                    self.probe.primary_type == b.primary_type
                    for b in self.analyte.bodies
                )
                and self.probe not in self.analyte.bodies
            ):
                raise ValueError(
                    "Clashing body definitions: if `probe` and has the same "
                    + "primary type as a body in `analyte`, the probe must be "
                    + "identical to that body."
                )
            if any(
                self.probe.primary_type in b.secondary_types
                for b in self.analyte.bodies
            ):
                raise ValueError(
                    "Clashing body definitions: probe primary type must not "
                    + "be included in any analyte body's secondary types."
                )
            if any(
                b.primary_type in self.probe.secondary_types
                for b in self.analyte.bodies
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
    def probe(self, value: Body):
        """Set the system's probe."""
        original_value = self._probe
        self._probe = value
        try:
            self.validate()
        except:
            self._probe = original_value
            raise
        
    @property
    def analyte(self) -> Body | Arrangement:
        """The system's analyte.
        
        The analyte is a static entity (either a :py:class:`~p4.Body` or a
        :py:class:`~p4.Arrangement`) whose effective energy and force fields
        are measured by the probe.
        """
        return self._analyte

    @analyte.setter
    def analyte(self, value: Body | Arrangement):
        """Set the system's analyte."""
        original_value = self._analyte
        self._analyte = value
        try:
            self.validate()
        except:
            self._analyte = original_value
            raise
        
    @property
    def interactions(self) -> list[Interaction]:
        """The interactions between particles in the probe and analyte."""
        return self._interactions

    @interactions.setter
    def interactions(self, value: list[Interaction]):
        """Set the interactions between particles in the probe and analyte."""
        original_value = self._interactions
        self._interactions = value
        try:
            self.validate()
        except:
            self._interactions = original_value
            raise

    # ---------------------------------- FROM ----------------------------------

    @classmethod
    def from_hoomd_simulation(
        cls,
        simulation: hoomd.Simulation,
        probe_primary_type: str,
        analyte_primary_type: str | None = None
    ) -> System:
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
    ) -> System:
        """Create a system from JSON.

        A JSON path may be provided to control the location that the
        system data is retrieved from. See :py:meth:`to_json` for an
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
    def _convert_json_dict(cls, data: dict) -> dict:
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

    # ----------------------------------- TO -----------------------------------

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
                    for t in interaction._interacting_types("single")
                )
                and any(
                    t in analyte_types
                    for t in interaction._interacting_types("single")
                )
            ):
                included_interactions.append(interaction)

            # For pair types, there must be at least one pair that contains a
            # type in the probe and a type (could be the same one) in the
            # analyte
            else:
                straddlers = []
                for type_pair in interaction._interacting_types("pair"):
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

    # --------------------------------- MEASURE --------------------------------

    def measure(
        self,
        quantities: Literal["U", "F", "T"] | list[Literal["U", "F", "T"]],
        positions: PositionsLike,
        orientations: OrientationLike | OrientationsLike | Iterable[OrientationsLike],
        filename: os.PathLike | None = None,
        n_processes: int = -1,
        save_gsd: bool = False,
        nlist: hoomd.md.nlist.NeighborList | None = None,
        disable_pbar: bool = False,
    ) -> Field:
        """Measure named quantities for the system and return them as a field.

        This method coordinates the parallelization of the measurement
        simulations and the merging, cleaning, and formatting of the resulting
        datasets. For the actual implementation of the simulation loop, see
        :py:func:`p4.util.simulation.measure`.

        .. _multiprocessing module: https://docs.python.org/3/library/multiprocessing.html

        .. _measure-multiprocessing-warning:
        .. warning::
            By default, :py:meth:`~p4.system.System.measure` uses Python's
            `multiprocessing module`_ to distribute the measurement process
            across the maximum number of processes allowed on your computer.
            Using the maximum number of processes may not minimize the program
            execution time due to the added overhead of creating and closing new
            processes. The smaller the number of positions and orientations, the
            lower the optimum number of processes. For example, for a 10 x 10 x
            1 grid of positions and only 1 orientation, a single process is
            best.
            

        :meta measure:

        Parameters
        ----------
        quantities : one or more of 'U', 'F', 'T'
            The quantities to measure. 'U' is the potential energy measured for
            the entire system, and is saved as a scalar quantity. 'F' and 'T'
            are the net force and torque experienced by the probe, and are saved
            as vector quantities.
        positions : (N, 3) array of floats
            The positions at which to measure.
        orientations : (4,), (M, 4), or (N, M, 4) array of floats
            The orientations at which to measure. If the array is 1D or 2D, it
            is broadcast so that each orientation is sampled at each position.
            If the array is 3D, it and ``positions`` must have identically-sized
            first axes (i.e., if ``positions`` has the shape ``(N, 3)``,
            ``orientations`` must have the shape ``(N, M, 4)``).
        filename : os.PathLike, optional
            The name or path for a file containing the measurement data. If not
            provided, no file is written. Must end in '.csv'.
        n_processes : int, default=-1
            The number of processes to distribute the measure operation between.
            Parallelization is implemented at the Python level, so each process
            creates and runs its own simulation and then the table results are
            combined in the output CSV. Note that if ``save_gsd`` is set to
            True, each simulation will produce a separate GSD file. Defaults to
            ``-1``, which uses the maximum allowed number of processes on the
            user's computer.
        save_gsd : bool, default=False
            Whether to save a GSD file alongside the output CSV file. If True,
            the GSD has the same name as the CSV. The name of the GSD file will
            be almost identical to that of the CSV file, with a suffix
            the process whose simulation wrote the GSD file. This option is
            available for debugging purposes, but generally should not be used.
        nlist : hoomd.md.nlist.NeighborList, optional
            The neighbor list with which to instantiate the class. If not
            provided, a bounding volume hierarchy-based neighbor list is created
            on the fly. This neighbor list is sufficient in most cases.
        disable_pbar : bool, default=False
            Whether to disable the progress bar.
        """
        # Default nlist
        if nlist is None:
            nlist = hoomd.md.nlist.Tree(2)

        # Ensure positions and orientations are numpy arrays (TODO: validate)
        probe_positions = np.asarray(positions)
        probe_orientations = np.asarray(orientations)

        # Ensure positions is a 2D array in 3D space
        if len(probe_positions.shape) != 2 or probe_positions.shape[1] != 3:
            raise ValueError("Positions must be a (N, 3) array.")

        # If necessary, broadcast orientations
        if (
            len(probe_orientations.shape) < 3
            and probe_orientations.shape[-1] == 4
        ):
            probe_orientations = np.array(
                [np.atleast_2d(probe_orientations) for _ in probe_positions]
            )
        elif (
            len(probe_orientations.shape) != 3
            or probe_orientations.shape[0] != probe_positions.shape[0]
            or probe_orientations.shape[-1] == 4
        ):
            raise ValueError("Could not broadcast orientations onto positions.")
        
        # Ensure filename has correct extension
        if (
            filename is not None
            and str(filename).rsplit(".")[-1] not in ("csv", "CSV")
        ):
            raise ValueError(
                "`filename` must be None or a string ending in '.csv'."
            )
        
        # Calculate a simulation box that contains all the particles
        max_distance = probe_positions.max()
        if isinstance(self.analyte, Body):
            for ps in self.analyte.positions_by_type.values():
                max_distance = max(max_distance, np.asarray(ps).max())
        elif isinstance(self.analyte, Arrangement):
            for b in self.analyte.bodies:
                if not b.secondary_types:
                    continue
                for primary_p in self.analyte.positions_by_type[b.primary_type]:
                    max_distance = max(
                        max_distance,
                        *[
                            (np.asarray(primary_p) + np.asarray(ps).max()).max()
                            for ps in b.positions_by_type.values()
                        ]
                    )
        max_r_cut = 0
        for i in self.interactions:
            max_r_cut = max(max_r_cut, i.initial_args.get("default_r_cut", 0))
            max_r_cut = max(max_r_cut, i.default_params.get("r_cut", 0))
            for param_dict in i.typed_params.values():
                max_r_cut = max(max_r_cut, param_dict.get("r_cut", 0))
        
        simulation_box = 1.1 * np.array([
            2 * (max_distance + max_r_cut),
            2 * (max_distance + max_r_cut),
            2 * (max_distance + max_r_cut),
            0.0,
            0.0,
            0.0
        ])

        # Figure out number of processes that will be used formultiprocessing.
        # Each process will ultimately receive its own simulation.
        if n_processes < -1 or n_processes == 0:
            raise ValueError("`n_processes` must be -1 or a positive integer.")

        if n_processes == -1:
            n_processes = os.process_cpu_count()

        # If multiprocessing, run copies of the probe simulation with chunks
        # of the set of positions across a collection of processes
        def subdivide(array: list, n: int):
            """Subdivide an array into n chunks of consecutive items.

            Disclaimer: the body of this function was written by ChatGPT.

            Parameters
            ----------
            array : list
                The array to subdivide
            n : int
                The number of chunks to subdivide the array into.

            Returns
            -------
            subarrays
                An array of sections of the input array.
            """
            k, m = divmod(len(array), n)
            return [
                array[i*k + min(i, m):(i+1)*k + min(i+1, m)]
                for i in range(n)
            ]
        
        if n_processes != 1:
            # Set up for multiprocessing with tqdm
            # Ref: https://github.com/tqdm/tqdm#nested-progress-bars
            tqdm.set_lock(multiprocessing.RLock())

            with (
                multiprocessing.Pool(
                    n_processes,
                    initializer=tqdm.set_lock,
                    initargs=(tqdm.get_lock(),)
                ) as pool
            ):
                if save_gsd:
                    gsd_filenames = [
                        filename.rsplit(".", 1)[0] + f"_{i}.gsd"
                        for i in range(n_processes)
                    ]
                else:
                    gsd_filenames = [None for _ in range(n_processes)]

                args = zip(
                    [deepcopy(self) for _ in range(n_processes)],
                    [quantities for _ in range(n_processes)],
                    subdivide(probe_positions, n_processes),
                    [probe_orientations for _ in range(n_processes)],
                    [self.active_interactions for _ in range(n_processes)],
                    [simulation_box for _ in range(n_processes)],
                    gsd_filenames,
                    [nlist for _ in range(n_processes)],
                    list(range(n_processes)),
                    [disable_pbar for _ in range(n_processes)]
                )
                tables = pool.starmap(util.simulation.measure, args)
            
            table = util.data.clean_header(util.data.merge_tables(tables))
        
        # If not multiprocessing, don't initialize a pool (easier for debugging)
        else:
            if save_gsd:
                gsd_filename = filename.rsplit(".", 1)[0] + ".gsd"
            else:
                gsd_filename = None
            table = util.simulation.measure(
                system=self,
                quantities=quantities,
                positions=probe_positions,
                orientations=probe_orientations,
                included_interactions=self.active_interactions,
                simulation_box=simulation_box,
                gsd_filename=gsd_filename,
                nlist=nlist,
                pbar_number=0,
                disable_pbar=disable_pbar,
            )

            table = util.data.clean_header(table)

        # Write file if necessary
        if filename is not None:
            with open(filename, "w") as file:
                table.seek(0)
                file.write(table.read())
        
        # Create Field
        table.seek(0)
        columns = table.readline().strip("\n").split(",")

        table.seek(0)
        return Field(
            np.rec.array(
                np.genfromtxt(
                    table,
                    names=columns,
                    skip_header=1,
                    dtype=None,
                    delimiter=","
                )
            )
        )
        


    # --------------------------------- OTHER ----------------------------------

    def __eq__(self, other) -> bool:
        """Systems are equal if their settable properties are equal."""
        return (
            self.interactions == other.interactions
            and self.probe == other.probe
            and self.analyte == other.analyte
        )
