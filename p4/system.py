# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from copy import deepcopy
import os
from typing import TYPE_CHECKING, Callable, Iterable
import multiprocessing

import coxeter
import hoomd

import p4.util
from p4 import Body, Interaction

class System:
    """A System is defined by two bodies and a set of interactions.
    
    Once the system is instantiated, its potential energy landscape can be
    measured using the `probe_potential()` method.

    Parameters
    ----------
    probe : Body
        The body for the probe.
    analyte : Body
        The body for the analyte.
    interactions : list[Interaction]
        A collection of interactions that define how one or more particle types
        in the probe interact with one or more types in the analyte.
    """
    def __init__(
        self,
        probe: Body,
        analyte: Body,
        interactions: list[Interaction],
    ):
        if not isinstance(probe, Body):
            raise TypeError("`probe` must be an instance of 'Body'.")
        if not isinstance(analyte, Body):
            raise TypeError("`analyte` must be an instance of 'Body'.")
        if (
            not isinstance(interactions, Iterable)
            or not all(type(i) is Interaction for i in interactions)
        ):
            raise TypeError(
                "`interactions` must be a list of Interaction instances."
            )

        self.probe = probe
        self.analyte = analyte
        self.interactions = interactions

        self.validate()

    def validate(self):
        """Ensure the system can be simulated."""
        # Ensure there is no rigid body clash
        if self.probe.primary_type == self.analyte.primary_type:
            if self.probe != self.analyte:
                raise ValueError(
                    "If `probe` and `analyte` have the same primary_type, they "
                    + "must also have the same secondary types, positions, and "
                    + "orientations."
                )
        if self.probe.primary_type in self.analyte.secondary_types:
            raise ValueError(
                "probe primary type cannot appear in analyte secondary types."
            )
        if self.analyte.primary_type in self.probe.secondary_types:
            raise ValueError(
                "analyte primary type cannot appear in probe secondary types."
            )

        # Ensure bodies do not contain types not covered by interactions.        
        covered_types = [t for i in self.interactions for t in i.all_types]

        def err_msg(probe_or_analyte, primary_or_secondary, particle_type):
            return (
                f"{probe_or_analyte} {primary_or_secondary} type "
                + f"{self.probe.primary_type} is not covered by the provided "
                + f"interactions. Covered types are {covered_types}."
            )
        
        if self.probe.primary_type not in covered_types:
            raise ValueError(
                err_msg("probe", "primary", self.probe.primary_type)
            )
        for t in self.probe.secondary_types:
            if t not in covered_types:
                raise ValueError(
                    err_msg("probe", "secondary", t)
                )
        if self.analyte.primary_type not in covered_types:
            raise ValueError(
                err_msg("analyte", "primary", self.analyte.primary_type)
            )
        for t in self.analyte.secondary_types:
            if t not in covered_types:
                raise ValueError(
                    err_msg("analyte", "secondary", t)
                )

    @property
    def active_interactions(self) -> list[Interaction]:
        """Interactions with 'yes' types in both the probe and the analyte."""
        probe_types = [self.probe.primary_type] + self.probe.secondary_types
        analyte_types = [self.analyte.primary_type] + self.analyte.secondary_types
        
        included_interactions = []

        for interaction in self.interactions:
            # For single types, there must be at least one in probe and one in
            # analyte
            if (
                any(t in probe_types for t in interaction.yes_single_types)
                and any(t in analyte_types for t in interaction.yes_single_types)
            ):
                included_interactions.append(interaction)

            # For pair types, there must be at least one pair that contains a
            # type in the probe and a type (could be the same one) in the
            # analyte
            else:
                straddlers = []
                for type_pair in interaction.yes_pair_types:
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
        all_types.append(self.analyte.primary_type)
        all_types.extend(self.analyte.secondary_types)
        return list(set(all_types))

    def probe_potential(
        self,
        position_resolutions: list[list[float]],    # TODO: sampling_strategy: 'grid' with p_res and o_res, 'dynamic' with ???
        orientation_resolutions: list[list[float]],
        symmetries: list[int],
        csv_filename: str,
        nlist: hoomd.md.nlist.NeighborList,
        outside_cutoff: float,
        inside_cutoff: float | None = None,
        cutoff_shape: coxeter.shapes.ConvexPolyhedron | None = None,
        box_safety_factor: float = 100,
        n_processes: int = 1,
        save_gsd: bool = False,
    ):
        """Probe the potential energy landscape of the system.

        Parameters
        ----------
        position_resolutions : list[list[float]]
            The number of samples along each dimension of the position grid.
            $[X, Y, Z]$
        orientation_resolutions : list[list[float]]
            The number of samples along each dimension of the orientation grid.
            $[X, Y, Z]$
        orientation_symmetries : list[int]
            The rotational symmetry for each axis. If not provided, C1 symmetry
            is assumed for every axis. $[X, Y, Z]$
        csv_filename : str
            The name of the CSV file to save.
        nlist : hoomd.md.nlist.NeighborList
            The neighbor list to use for the interactions.
        outside_cutoff : float
            The cutoff distance outside which no positions will be probed. If
            `cutoff_shape` is provided, this distance represents a buffer
            distance around the shape, otherwise it distance represents the side
            lengths of a cube centered on the origin.
        inside_cutoff : float, optional
            The cutoff distance inside which no positions will be probed.
            If `probe_cutoff_shape` is provided, this distance represents a
            buffer distance inside the shape, otherwise it distance represents
            the side lengths of a cube centered on the origin. If not provided,
            all positions inside the outer cutoff distance will be probed.
        cutoff_shape : coxeter.shapes.ConvexPolyhedron, optional
            A convex polyhedron representing a shape to which cutoff distances
            are relative, enabling the user to sample non-cubic boxes. If not
            provided, cutoff distances describe the side lengths of a cube.
        box_safety_factor : float, default=100
            The scale factor for the simulation box, since it must be bigger
            than the probe box to prevent the minimum image problem. Defaults
            to 100, which should be sufficient in most cases.
        n_processes : int, default=1
            The number of processes to distribute the probe operation between.
            Parallelization is implemented at the Python level, so each process
            creates and runs its own simulation and then the table results are
            combined in the output CSV. Note that if save_gsd is set to True,
            each simulation will produce a separate GSD file. Set this parameter
            to -1 to use the maximum allowed number of processes for your
            machine.
        save_gsd : bool, default=False
            Whether to save a GSD file alongside the output CSV file. If True,
            the GSD has the same name as the CSV. The name of the GSD file will
            be almost identical to that of the CSV file, with a suffix
            the process whose simulation wrote the GSD file. This option is
            available for debugging purposes, but generally should not be used.
        """
        # Calculate probe box based on cutoff distances
        if cutoff_shape is None:
            probe_box = [outside_cutoff, outside_cutoff, outside_cutoff]
        else:
            shape_maxes = cutoff_shape.vertices.max(axis=0)
            probe_box = [m + outside_cutoff for m in shape_maxes]

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
        probe_positions = p4.util.get_probe_positions(
            probe_box,
            position_resolutions
        )
        probe_orientations = p4.util.get_probe_orientations(
            orientation_resolutions,
            symmetries
        )

        # Remove positions that are too far away
        probe_positions = p4.util.exclude_positions_by_shape(
            positions=probe_positions,
            exclude_inside=False,
            shape=(
                cutoff_shape
                if cutoff_shape is not None
                else p4.util.get_cube(outside_cutoff)
            ),
            buffer=outside_cutoff if cutoff_shape is not None else 0.0
        )

        # Remove positions that are too close
        # Review: allow distance to be negative?
        if inside_cutoff is not None:
            # if inside_cutoff <= 0:
            #     raise ValueError(
            #         "'inside_cutoff' must be a value greater than 0."
            #     ) 
            if inside_cutoff > outside_cutoff:
                raise ValueError("inside cutoff must be smaller than outside cutoff.")
            probe_positions = p4.util.exclude_positions_by_shape(
                positions=probe_positions,
                exclude_inside=True,
                shape=(
                    cutoff_shape
                    if cutoff_shape is not None
                    else p4.util.get_cube(inside_cutoff)
                ),
                buffer=(
                    inside_cutoff
                    if cutoff_shape is not None
                    else 0.0
                )
            )

        # If multiprocessing, run copies of the probe simulation with chunks
        # of the set of positions across a collection of processes
        if n_processes != 1:
            with multiprocessing.Pool(processes=n_processes) as pool:
                if save_gsd:
                    gsd_filenames = [
                        csv_filename.split(".")[-2] + f"_{i}.gsd"
                        for i in range(n_processes)
                    ]
                else:
                    gsd_filenames = [None for _ in range(n_processes)]

                args = zip(
                    [deepcopy(self) for _ in range(n_processes)],
                    p4.util.subdivide(probe_positions, n_processes),
                    [probe_orientations for _ in range(n_processes)],
                    [self.active_interactions for _ in range(n_processes)],
                    [nlist for _ in range(n_processes)],
                    [probe_box for _ in range(n_processes)],
                    [simulation_box for _ in range(n_processes)],
                    gsd_filenames,
                )
                tables = pool.starmap(p4.util.run_probe, args)
            
            table = p4.util.clean_header(p4.util.merge_tables(tables))
        
        # If not multiprocessing, don't initialize a pool (easier for debugging)
        else:
            if save_gsd:
                gsd_filename = csv_filename.split(".")[-2] + ".gsd"
            else:
                gsd_filename = None
            table = p4.util.run_probe(
                system=self,
                probe_positions=probe_positions,
                probe_orientations=probe_orientations,
                included_interactions=self.active_interactions,
                nlist=nlist,
                probe_box=probe_box,
                simulation_box=simulation_box,
                gsd_filename=gsd_filename
            )

            table = p4.util.clean_header(table)

        with open(csv_filename, "w") as file:
            table.seek(0)
            file.write(table.read())

    @classmethod
    def from_hoomd_simulation(
        cls,
        simulation: hoomd.Simulation,
        probe_primary_type: str,
        analyte_primary_type: str
    ):
        """Parse a hoomd.Simulation object into a System.

        The simulation must have an integrator, and the integrator must have one
        or more forces. Optionally, the integrator may also have a rigid
        constraint. If it does have one, and if this constraint's keys
        include the probe's and/or the analyte's primary type, then the
        constraint is parsed to determine secondary types, positions, and
        orientations for the probe and/or analyte bodies.

        Parameters
        ----------
        simulation : hoomd.Simulation
            The simulation to parse.
        probe_primary_type : str
            The name of the particle type to use for the probe's primary type.
            This name must be represented in the simulation's current state.
        analyte_primary_type : str
            The name of the particle type to use for the analyte's primary type.
            This name must be represented in the simulation's current state.
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
        if analyte_primary_type not in types_in_state:
            raise ValueError(
                "simulation state does not have particle type "
                + f"{analyte_primary_type}"
            )
        
        interactions = Interaction.from_hoomd_simulation(simulation)

        # [Review: is there a better way to do this?]
        try:
            probe = Body.from_hoomd_simulation(simulation, probe_primary_type)
        except ValueError:
            probe = Body(probe_primary_type)
        try:
            analyte = Body.from_hoomd_simulation(simulation, analyte_primary_type)
        except ValueError:
            analyte = Body(analyte_primary_type)
        
        return cls(
            probe=probe,
            analyte=analyte,
            interactions=interactions
        )
    
    def __eq__(self, other):
        """Two Systems are equivalent if their settable properties are, too."""
        return (
            self.interactions == other.interactions
            and self.probe == other.probe
            and self.analyte == other.analyte
        )
