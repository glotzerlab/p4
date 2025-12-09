from copy import deepcopy
from typing import Callable
import hoomd
import numpy as np
import tqdm
import util

class Interaction:
    # An interaction contains the info needed to initialize an interaction instance
    # with the correct parameters for every pair of types. It also knows how to
    # calculate inside and outside cutoff distances.

    # Initialize an interaction instance with types and other params

    # Do we want this to be static, or should user be able to change params of
    # an interaction after they've registered it? I think we do want this, since
    # the interactions are attached to the simulation object every time run is called.

    # Note: some interactions require things besides params to be set (e.g., ALJ requires 'shape')...
    #       should we provide an additional dictionary parameter or similar?
    def __init__(
        self,
        hoomd_class,
        initial_inputs,
        default_single_typed_attributes: dict,
        default_pair_typed_attributes: dict,
        yes_types,
        yes_single_typed_attributes,
        yes_pair_typed_attributes,
        probe_cutoff_shape,
        probe_cutoff_outside_callable,  # needs to be a callable if the user can change params after instantiation
        probe_cutoff_inside_callable
    ):
        self.hoomd_class = hoomd_class
        self.inputs = initial_inputs
        self.default_single_typed_attributes = default_single_typed_attributes
        self.default_pair_typed_attributes = default_pair_typed_attributes
        self.yes_types = yes_types
        self.yes_single_typed_attributes = yes_single_typed_attributes
        self.yes_pair_typed_attributes = yes_pair_typed_attributes
        self.probe_cutoff_shape = probe_cutoff_shape
        self.probe_cutoff_outside_callable = probe_cutoff_outside_callable
        self.probe_cutoff_inside_callable = probe_cutoff_inside_callable

    def validate_inputs():
        pass

    def validate_typed_attributes():
        pass


class ParticleModel:
    """The names and positions of a particle's primary and secondary types.

    Every particle model must have a primary type, but secondary types are
    optional.
    
    When secondary types **are not** provided, the model represents a
    simple particle with a single type and no further information is needed.
    
    When secondary types **are** provided, the model represents a rigid body
    with a central particle (`primary_type`) and one or more constituent
    particles (`secondary_types`). In this case, the model needs a way to
    determine the position(s) of each type of constituent particle, and the user
    must provide a function that does so.
    
    Parameters
    ----------
    primary_type : str
        The name of the primary type.
    secondary_types : list[str], optional
        The names of the secondary types.
    get_secondary_positions_by_type : Callable, optional
        A function that returns the position(s) of particle(s) with the
        secondary types. The function must have at least one parameter, and the
        first parameter must be a string representing the name of a secondary
        type. Required if `secondary_types` is provided, otherwise ignored.
    """
    def __init__(
        self,
        primary_type: str,
        secondary_types: list[str] = [],
        get_secondary_positions_by_type: Callable | None = None
    ):
        if secondary_types and get_secondary_positions_by_type is None:
            raise ValueError(
                "'get_secondary_positions_by_type' is required if "
                + "'secondary_types' is provided"
            )

        self.primary_type = primary_type
        self.secondary_types = secondary_types
        self.get_secondary_positions_by_type = get_secondary_positions_by_type
    
    def can_be_rigid_body(self) -> bool:
        """Whether the particle can be a rigid body."""
        return len(self.secondary_types) == 0
    
    def must_be_rigid_body(self, interaction: Interaction) -> bool:
        """Whether the particle must be a rigid body for some interaction."""
        return any([t in self.secondary_types for t in interaction.yes_types])


class System:
    """A System is defined by its particle models and an interaction model.
    
    Once the system is instantiated, its potential energy landscape can be
    measured using the `probe()` method.

    Parameters
    ----------
    probe : ParticleModel
        The particle model for the probe.
    analyte : ParticleModel
        The particle model for the analyte.
    interaction_model : dict[str, Interaction]
        A collection of named interactions. All keys should have the `str` type
        and all values should have the `Interaction` type.
    """
    def __init__(
        self,
        probe_model: ParticleModel,
        analyte_model: ParticleModel,
        interaction_model: dict[str, Interaction],
    ):
        self.probe_model = probe_model
        self.analyte_model = analyte_model
        self.interaction_model = interaction_model

        self.validate()

    def validate(self):
        """Ensure all possible interacting types appear in particle models."""
        valid_types = []
        for model in [self.probe_model, self.analyte_model]:
            valid_types.append(model.primary_type)
            valid_types.extend(model.secondary_types)

        for interaction in self.interaction_model.values():
            assert all([t in valid_types for t in interaction.yes_types])

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
        nlist: hoomd.md.nlist.NeighborList,
        interactions_to_include: list[str] = [],
        probe_box: list[float] | None = None,
        gsd_filename: str | None = None,
        csv_filename: str | None = None,
        box_safety_factor: float = 10
    ):
        """Probe the potential energy landscape of the system.

        Parameters
        ----------
        position_resolutions : list[list[float]]
            The number of samples along each dimension of the position grid.
            [X, Y, Z]
        orientation_resolutions : list[list[float]]
            The number of samples along each dimension of the orientation grid.
            [X, Y, Z]
        orientation_symmetries : list[int]
            The rotational symmetry for each axis. If not provided, C1 symmetry
        is assumed for every axis. [X, Y, Z]
        nlist : hoomd.md.nlist.NeighborList
            The neighborlist to use for the interactions.
        interactions_to_include : list[str], optional
            The names of the interactions to include. If not provided, all
            interactions in the model are included.
        probe_box : list[float], optional
            The side lengths of the probe box. If provided, overrides the
            distances provided by the interactions' cutoff callables.
        gsd_filename : str, optional
            The name of the GSD file to save. If not provided, no GSD file is
            saved.
        csv_filename : str, optional
            The name of the CSV file to save. If not provided, no CSV file is
            saved.
        box_safety_factor : float, optional
            The scale factor for the simulation box, since it must be bigger
            than the probe box to prevent the minimum image problem. Defaults
            to 10.
        """
        # determine needed interactions and particle types
        interactions = self.interactions(interactions_to_include)
        types = self.interacting_types(interactions_to_include)

        # calculate probe box based on distance cutoff callables for included interactions
        if probe_box is None:
            cutoffs = util.maximal_outer_interaction_cutoffs(interactions)
            probe_box = [2 * c for c in cutoffs]

        # determine the frame's box from the probe box
        simulation_box = [d * box_safety_factor for d in probe_box]
        simulation_box.extend([0, 0, 0])
        
        # create initial frame
        frame = util.get_initial_frame(
            self.probe_model,
            self.analyte_model,
            types,
            probe_box,
            simulation_box
        )

        # initialize Simulation
        simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
        simulation.create_state_from_snapshot(frame)

        # add integrator
        simulation = util.add_integrator(simulation)

        # add rigid bodies if necessary
        probe_is_rigid = util.particle_must_be_rigid_body(
            self.probe_model,
            interactions
        )
        analyte_is_rigid = util.particle_must_be_rigid_body(
            self.analyte_model,
            interactions
        )
        if probe_is_rigid:
            simulation, rigid = util.add_rigid_constraint(
                simulation,
                self.probe_model,
                False if analyte_is_rigid else True,
                [t for t in self.probe_model.secondary_types if t in types]
            )
        if analyte_is_rigid:
            simulation, _ = util.add_rigid_constraint(
                simulation,
                self.analyte_model,
                True,
                [t for t in self.analyte_model.secondary_types if t in types],
                rigid if probe_is_rigid else None
            )

        # add gsd/table writers if file names are provided
        if gsd_filename:
            simulation, compute = util.add_gsd_writer(simulation, gsd_filename)

        if csv_filename:
            csv_file = open(csv_filename, "w")
            if gsd_filename:
                simulation, _ = util.add_table_writer(
                    simulation, csv_file, self.probe_model, compute
                )
            else:
                simulation, _ = util.add_table_writer(
                    simulation, csv_file, self.probe_model
                )

        # add required interactions
        for interaction in interactions:
            simulation = util.add_interaction(simulation, nlist, interaction)

        # calculate the probe positions and orientations
        probe_positions = util.get_probe_positions(
            probe_box,
            position_resolutions
        )
        probe_orientations = util.get_probe_orientations(
            orientation_resolutions,
            orientation_symmetries
        )

        # TODO: remove positions that are too close or too far away?
        # I'm not sure how to do this currently because the particle model
        # has no notion of shape. We could allow user to pass a flag and if true
        # then attempt to construct shapes for each of the types of secondary
        # particles for the analyte. The smallest shape would be used for r_in
        # and the largest shape would be used for r_out

        # run the probe simulation
        state = simulation.state.get_snapshot()
        all_types = state.particles.types
        probe_index = deepcopy(np.where(
            state.particles.typeid == all_types.index(self.probe_model.primary_type)
        ))
        analyte_index = deepcopy(np.where(
            state.particles.typeid == all_types.index(self.analyte_model.primary_type)
        ))
        analyte_orientation = deepcopy(
            state.particles.orientation[analyte_index,:]
        )
        integrate_rotational_dof = deepcopy(    # currently always True
            simulation.operations.integrator.integrate_rotational_dof
        )

        for p in tqdm(probe_positions):
            for o in probe_orientations:
                with simulation.state.cpu_local_snapshot as state:

                    state.particles.position[state.particles.rtag[analyte_index]] = np.array([0.0, 0.0, 0.0])
                    state.particles.velocity[state.particles.rtag[analyte_index]] = np.array([0.0, 0.0, 0.0])
                    if integrate_rotational_dof:
                        state.particles.angmom[state.particles.rtag[analyte_index]] = np.array([0.0, 0.0, 0.0, 0.0])
                        state.particles.orientation[state.particles.rtag[analyte_index]] = analyte_orientation

                    state.particles.position[state.particles.rtag[probe_index]] = p
                    state.particles.velocity[state.particles.rtag[probe_index]] = np.array([0.0, 0.0, 0.0])
                    if integrate_rotational_dof:
                        state.particles.angmom[state.particles.rtag[probe_index]] = np.array([0.0, 0.0, 0.0, 0.0])
                        state.particles.orientation[state.particles.rtag[probe_index]] = o

                simulation.run(1)

        # close the csv file if necessary
        if csv_filename:
            csv_file.close()


class ProbeResults:     # or maybe "Field"
    # Handle mean/min, plotting
    def __init__(self, csv_filename):
        pass