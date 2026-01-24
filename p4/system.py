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
        interaction_model: dict[str, Interaction],
    ):
        self.probe = probe
        self.analyte = analyte
        self.interaction_model = interaction_model

        self.validate()

    def validate(self):
        """Ensure all possible interacting types appear in body models."""
        valid_types = []
        for model in [self.probe, self.analyte]:
            valid_types.append(model.primary_type)
            valid_types.extend(model.secondary_types)
        
        invalid_types = {}
        for name, interaction in self.interaction_model.items():
            invalid_types[name] = [
                t
                for t in interaction.yes_types if t not in valid_types
            ]

        if any([len(v) > 0 for v in invalid_types.values()]):
            raise ValueError(
                f"The provided interaction model contains types that are "
                f"not in any of the provided interaction models: "
                f"{invalid_types}."
            )

    def interactions(self, names: list[str] = []) -> list[Interaction]:
        """A list of interactions, all of them or just those named.

        Parameters
        ----------
        names : list[str], optional
            The names of the interactions to include. If not provided, all
            interactions from the interaction model are included.

        Returns
        -------
        interactions
        """
        if names != []:
            return [self.interaction_model[name] for name in names]
        else:
            return list(self.interaction_model.values())

    def interacting_types(self, names: list[str] | None = None) -> set[str]:
        """The interacting types, for all interactions or just those named.
        
        Parameters
        ----------
        names : list[str], optional
            The names of the interactions to include. If not provided, particle
            types for all interactions in the interaction model are included.

        Returns
        -------
        types
        """
        interacting_types = set([
            t
            for interaction in self.interactions(names)
            for t in interaction.yes_types
        ])
        return interacting_types

    def probe_potential(
        self,
        position_resolutions: list[list[float]],
        orientation_resolutions: list[list[float]],
        orientation_symmetries: list[int],
        interactions_to_include: list[str],
        csv_filename: str,
        nlist: hoomd.md.nlist.NeighborList,
        probe_cutoff_outside_distance: Callable,  # needs to be a callable if the user can change params after instantiation
        probe_cutoff_inside_distance: Callable | None = None,
        probe_cutoff_shape: coxeter.shapes.ConvexPolyhedron | None = None,
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
        interactions_to_include : list[str]
            The names of the interactions to include.
        csv_filename : str
            The name of the CSV file to save.
        nlist : hoomd.md.nlist.NeighborList
            The neighbor list to use for the interactions.
        probe_cutoff_outside_distance : Callable
            A callable that takes `self` as its only argument and returns a
            float representing the cutoff distance outside which no positions
            will be probed. If`probe_cutoff_shape` is provided, this distance
            represents a buffer distance around the shape, otherwise it
            distance represents the side lengths of a cube centered on the
            origin.
        probe_cutoff_inside_distance : Callable, optional
            A callable that takes `self` as its only argument and returns a
            float representing the cutoff distance inside which no positions
            will be probed. If`probe_cutoff_shape` is provided, this distance
            represents a buffer distance inside the shape, otherwise it
            distance represents the side lengths of a cube centered on the
            origin. If not provided, all positions inside the outer cutoff
            distance will be probed.
        probe_cutoff_shape : coxeter.shapes.ConvexPolyhedron, optional
            A convex polyhedron representing a shape to which cutoff distances
            are relative, enabling the user to probe non-cubic boxes. If not
            provided, cutoff distances describe the side lengths of a cube.
        box_safety_factor : float, default=100
            The scale factor for the simulation box, since it must be bigger
            than the probe box to prevent the minimum image problem. Defaults
            to 100.
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
            available for debugging, but is not necessary for most users.
        """
        # Calculate probe box based on distance cutoff callables for included
        # interactions
        outside_cutoff = probe_cutoff_outside_distance(self)
        if probe_cutoff_shape is None:
            probe_box = [outside_cutoff, outside_cutoff, outside_cutoff]
        else:
            shape_maxes = probe_cutoff_shape.vertices.max(axis=0)
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
            orientation_symmetries
        )

        # Remove positions that are too far away
        probe_positions = p4.util.exclude_positions_by_shape(
            positions=probe_positions,
            exclude_inside=False,
            shape=(
                probe_cutoff_shape
                if probe_cutoff_shape is not None
                else p4.util.get_cube(outside_cutoff)
            ),
            buffer=outside_cutoff if probe_cutoff_shape is not None else 0.0
        )

        # Remove positions that are too close
        # TODO: allow distance to be negative?
        if probe_cutoff_inside_distance is not None:
            inside_cutoff = probe_cutoff_inside_distance(self)
            # if inside_cutoff <= 0:
            #     raise ValueError(
            #         "'probe_cutoff_inside_distance' must return a value "
            #         f"greater than 0."
            #     ) 
            if inside_cutoff > outside_cutoff:
                raise ValueError("inside cutoff must be smaller than outside cutoff.")
            probe_positions = p4.util.exclude_positions_by_shape(
                positions=probe_positions,
                exclude_inside=True,
                shape=(
                    probe_cutoff_shape
                    if probe_cutoff_shape is not None
                    else p4.util.get_cube(inside_cutoff)
                ),
                buffer=(
                    inside_cutoff
                    if probe_cutoff_shape is not None
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
                    [interactions_to_include for _ in range(n_processes)],
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
                interactions_to_include, nlist, probe_box, simulation_box
            )

            table = p4.util.clean_header(table)

        with open(csv_filename, "w") as file:
            table.seek(0)
            file.write(table.read())

    @classmethod
    def from_hoomd_simulation(cls, simulation, probe_primary_type, analyte_primary_type):
        pass
