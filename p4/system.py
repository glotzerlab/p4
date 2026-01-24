# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from copy import deepcopy
import os
from typing import TYPE_CHECKING, Callable

import coxeter
import hoomd

import p4.util

if TYPE_CHECKING:
    from p4 import Interaction, Body

class System:
    """A System is defined by its body models and an interaction model.
    
    Once the system is instantiated, its potential energy landscape can be
    measured using the `probe()` method.

    Parameters
    ----------
    probe : BodyModel
        The body model for the probe.
    analyte : BodyModel
        The body model for the analyte.
    interaction_model : dict[str, Interaction]
        A collection of named interactions. All keys should have the `str` type
        and all values should have the `Interaction` type.
    """
    def __init__(
        self,
        probe: Body,
        analyte: Body,
        interactions: list[Interaction],
    ):
        self.probe = probe
        self.analyte = analyte
        self.interactions = interactions

        self.validate()

    def validate(self):
        """Ensure all possible interacting types appear in body models."""
        valid_types = []
        for model in [self.probe, self.analyte]:
            valid_types.append(model.primary_type)
            valid_types.extend(model.secondary_types)
        
        invalid_types = {}
        for i, interaction in enumerate(self.interactions):
            invalid_types[i] = [
                t
                for t in interaction.all_types if t not in valid_types
            ]

        if any([len(v) > 0 for v in invalid_types.values()]):
            raise ValueError(
                f"The provided interaction model contains types that are "
                f"not in any of the provided interaction models: "
                f"{invalid_types}."
            )

    @property
    def active_interactions(self) -> list[Interaction]:
        """The interactions with 'yes' particle types in the probe or analyte.
        """
        included_interactions = []

        for interaction in self.interactions:
            for type_name in interaction.yes_single_types:
                if type_name in self.all_types:
                    included_interactions.append(interaction)
            for type_pair in interaction.yes_pair_types:
                if any(t in self.all_types for t in type_pair):
                    included_interactions.append(interaction)

        return included_interactions
    
    @property
    def all_types(self) -> list[str]:
        """All particle types in the probe and analyte."""
        all_types = [self.probe.primary_type]
        all_types.extend(self.probe.secondary_types)
        all_types.append(self.analyte.primary_type)
        all_types.extend(self.analyte.secondary_types)
        return all_types

    @property
    def interacting_types(self) -> list[str]:
        """The particle types that are 'yes' types in the active interactions."""
        interacting_types = []
        for t in self.all_types:
            for i in self.active_interactions:
                if (
                    t in i.yes_single_types
                    or any(t in p for p in i.yes_pair_types)
                ):
                    interacting_types.append(t)
        return interacting_types

    def probe_potential(
        self,
        position_resolutions: list[list[float]],    # TODO: sampling_strategy: 'grid' with p_res and o_res, 'dynamic' with ???
        orientation_resolutions: list[list[float]],
        symmetries: list[int],  # TODO: auto-calculate
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
        # TODO: allow distance to be negative?
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
            import pathos
            mp = pathos.helpers.mp
            with mp.Pool() as pool:
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
        else:   # TODO: move header cleaning to its own function
            table = p4.util.run_probe(
                self, probe_positions, probe_orientations,
                self.active_interactions, nlist, probe_box, simulation_box
            )

            table = p4.util.clean_header(table)

        with open(csv_filename, "w") as file:
            table.seek(0)
            file.write(table.read())

    @classmethod
    def from_hoomd_simulation(cls, simulation, probe_primary_type, analyte_primary_type):
        pass
