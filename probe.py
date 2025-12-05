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
        hoomd_class,    # ???
        inputs,
        default_params,  # if this is a dictionary with a 'params' key, then class must have more than just params, but if not then class just has params
        yes_types,  # if this is a dictionary with a 'params' key, then class must have more than just params, but if not then class just has params
        yes_params,
        probe_outside_cutoff_callable,  # needs to be a callable if the user can change params after instantiation
        probe_inside_cutoff_callable
    ):
        self.hoomd_class = hoomd_class
        self.inputs = inputs
        self.default_params = default_params
        self.yes_types = yes_types
        self.yes_params = yes_params
        self.probe_outside_cutoff_callable = probe_outside_cutoff_callable
        self.probe_inside_cutoff_callable = probe_inside_cutoff_callable


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
    
    Once instantiated, the potential energy landscape can be measured using the
    `probe()` method.

    Parameters
    ----------
    probe : ParticleModel
        The particle model for the probe.
    analyte : ParticleModel
        The particle model for the analyte.
    interaction_model : dict[str, Interaction]
        A collection of named interactions. All keys should have the type string
        and all values should have the type Interaction.
    """
    def __init__(
        self,
        probe: ParticleModel,
        analyte: ParticleModel,
        interaction_model: dict[str, Interaction],
    ):
        self.probe = probe
        self.analyte = analyte
        self.interaction_model = interaction_model

        self.validate()

    def validate(self):
        """Ensure all possible interacting types appear in particle models."""
        valid_types = []
        for model in [self.probe, self.analyte]:
            valid_types.append(model.primary_type)
            valid_types.extend(model.secondary_types)

        for interaction in self.interaction_model.items():
            assert all([t in valid_types for t in interaction.yes_types])

    def interactions(self, names: list[str] | None = None) -> list[Interaction]:
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
        if names is not None:
            return [self.interaction_model[name] for name in names]
        else:
            return list(self.interaction_model.values())

    def interacting_types(self, names: list[str] | None = None) -> list[str]:
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
        interacting_types = []
        for interaction in self.interactions(names):
            for t in interaction.yes_types:
                if t not in interacting_types:
                    interacting_types.append(t)
        return interacting_types

    def probe(
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
        types = self.interacting_types(interactions_to_include)
        interactions = self.interactions(interactions_to_include)

        # calculate probe box based on distance cutoff callables for included interactions
        if probe_box is None:
            cutoffs = util.maximal_outer_interaction_cutoffs(interactions)
            probe_box = [2 * c for c in cutoffs]

        # determine the frame's box from the probe box
        simulation_box = [d * box_safety_factor for d in probe_box]
        
        # create initial frame
        frame = util.get_initial_frame(
            self.probe,
            self.analyte,
            types,
            probe_box,
            simulation_box
        )

        # initialize Simulation
        simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
        simulation.create_state_from_snapshot(frame)

        # add rigid bodies if necessary
        probe_is_rigid = util.particle_must_be_rigid_body(
            self.probe,
            interactions
        )
        analyte_is_rigid = util.particle_must_be_rigid_body(
            self.analyte,
            interactions
        )
        if probe_is_rigid:
            simulation, rigid = util.add_rigid_bodies(
                simulation,
                self.probe,
                [t for t in self.probe.secondary_types if t in types]
            )
        if analyte_is_rigid:
            simulation, _ = util.add_rigid_bodies(
                simulation,
                self.probe,
                [t for t in self.analyte.secondary_types if t in types],
                rigid if probe_is_rigid else None
            )

        # add gsd/table writers if file names are provided
        if gsd_filename:
            simulation, compute = util.add_gsd_writer(simulation, gsd_filename)

        if csv_filename:
            csv_file = open(csv_filename, "w")
            if gsd_filename:
                simulation, _ = util.add_csv_writer(simulation, csv_file, compute)
            else:
                simulation, _ = util.add_csv_writer(simulation, csv_file)

        # add required interactions
        for interaction in interactions:
            simulation = util.add_interaction(simulation, interaction)

        # add integrator
        simulation = util.add_integrator(simulation)

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
            state.particles.typeid == all_types.index(self.probe.primary_type)
        ))
        analyte_index = deepcopy(np.where(
            state.particles.typeid == all_types.index(self.analyte.primary_type)
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