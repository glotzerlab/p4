from copy import deepcopy
from io import TextIOWrapper
import itertools
from typing import Tuple
import gsd.hoomd
import hoomd
import numpy as np
import rowan

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from probe_potential import Interaction, ParticleModel

def maximal_outer_interaction_cutoffs(interactions: list[Interaction]):
    """The maximal useful distance for probing a set of interactions."""
    cutoffs = []
    for interaction in interactions:
        cutoffs.append(interaction.probe_cutoff_outside_callable())
    cutoffs = np.array(cutoffs)
    return [cutoffs[:,0].max(), cutoffs[:,1].max(), cutoffs[:,2].max()]

def particle_must_be_rigid_body(
    particle_model: ParticleModel,
    interactions: list[Interaction]
) -> bool:
    """Whether the particle must be rigid for any of the given interactions."""
    return any([
        particle_model.must_be_rigid_body(interaction)
        for interaction in interactions
    ])

def get_probe_positions(
    box: list[float],
    resolutions: list[int]
) -> np.ndarray:
    """Return an array of positions in an origin-centered box.

    Parameters
    ----------
    box : list[float]
        The lengths of the sides of the box in X, Y, Z order.
    resolutions : list[int]
        The resolution of the point grid for each dimension in X, Y, Z order.
    
    Returns
    -------
    np.ndarray
        A numpy array (..., 3) of positions.
    """
    positions = np.array(list(itertools.product(
        np.linspace(-box[0]/2, box[0]/2, resolutions[0], endpoint=False),
        np.linspace(-box[1]/2, box[1]/2, resolutions[1], endpoint=False),
        np.linspace(-box[2]/2, box[2]/2, resolutions[2], endpoint=False),
    )))

    positions[:,0] += (box[0]/resolutions[0])/2
    positions[:,1] += (box[1]/resolutions[1])/2
    positions[:,2] += (box[2]/resolutions[2])/2

    return positions

def exclude_positions_by_shape():
    # TODO
    pass

def get_probe_orientations(
    resolutions: list[int],
    symmetries: list[int] | None = None,
) -> np.ndarray:
    """Calculate an grid of orientations evenly sampling given axes.

    Parameters
    ----------
    resolutions : list[int]
        The number of samples per axis. [X, Y, Z]
    symmetries : list[int], optional
        The rotational symmetry for each axis. If not provided, C1 symmetry is
        assumed for every axis. [X, Y, Z]

    Returns
    -------
    orientations
        A numpy array (..., 4) of orientations as quaternions.
    """
    if symmetries is None:
        symmetries = [1, 1, 1]
    angles = np.array(list(itertools.product(
        np.linspace(0, 2*np.pi/symmetries[2], resolutions[2], endpoint=False),
        np.linspace(0, 2*np.pi/symmetries[1], resolutions[1], endpoint=False),
        np.linspace(0, 2*np.pi/symmetries[0], resolutions[0], endpoint=False),
    )))
    return rowan.from_euler(angles[:,0], angles[:,1], angles[:,2])

def get_initial_frame(
    probe_model: ParticleModel,
    analyte_model: ParticleModel,
    included_types: list[str],
    probe_box: list[float],
    simulation_box: list[float],
) -> gsd.hoomd.Frame:
    """Return a simulation frame with analyte at center and probe at edge.

    All provided types are included in the particle type data, but only primary
    types specified on the provided particle models are actually placed. In
    other words, secondary types **are not** placed in the frame and must be
    added separately using `create_rigid_bodies()`.

    Parameters
    ----------
    probe_model : ParticleModel
        The particle model for the probe.
    analyte_model : ParticleModel
        The particle model for the analyte.
    included_secondary_types : list[str]
        The secondary types to include in the particle data, accessible via
        frame.particles.types.
    probe_box : list[float]
        The side lengths of the box that will be probed. [Lx, Ly, Lz]
    simulation_box : list[float]
        The side lengths of the simulation box. [Lx, Ly, Lz]

    Returns
    -------
    frame
        The initial frame.
    """
    frame = gsd.hoomd.Frame()

    all_types = list(
        set(
            [analyte_model.primary_type, probe_model.primary_type]
        ).union(included_types)
    )
    all_types.sort()
    frame.particles.types = all_types

    positions = np.array([
        [0.0, 0.0, 0.0],
        [-probe_box[0]/2, -probe_box[1]/2, -probe_box[2]/2]
    ]) 
    frame.particles.N = 2
    frame.particles.position = positions
    frame.particles.typeid = [
        frame.particles.types.index(analyte_model.primary_type),
        frame.particles.types.index(probe_model.primary_type)
    ]
    frame.configuration.box = simulation_box
    frame.particles.mass = [1] * frame.particles.N
    frame.particles.moment_inertia = [1, 1, 1] * frame.particles.N
    frame.particles.orientation = [(1, 0, 0, 0)] * frame.particles.N

    return frame

def add_rigid_constraint(
    simulation: hoomd.Simulation,
    particle_model: ParticleModel,
    create_bodies: bool,
    included_secondary_types: list[str] | None = None,
    rigid: hoomd.md.constrain.Rigid | None = None,
) -> Tuple[hoomd.Simulation, hoomd.md.constrain.Rigid]:
    """Add rigid body constraints for a single particle model to the simulation.

    TODO: consider allowing the particle model to provide orientations like
    it already does for positions.

    Parameters
    ----------
    simulation : hoomd.Simulation
        The simulation to modify.
    particle_model : ParticleModel
        The particle model containing the rigid body information. The model's
        `primary_type` corresponds to the rigid body's central particle, while
        the `secondary_types` correspond to the constituent particles.
    create_bodies : bool
        Whether to modify the simulation state by calling
        `rigid.create_bodies(<simulation.state>)`. Only set the value to True
        when no more constraints will be added.
    included_secondary_types : list[str], optional
        The names of the secondary types to include in the rigid body. If not
        provided, all secondary types are included.
    rigid : hoomd.md.constrain.Rigid, optional
        An existing constraint instance to use. If not provided, a new one is
        created.

    Returns
    -------
    simulation, rigid
        The modified simulation and its rigid constraint.
    """
    if rigid is None:
        rigid = hoomd.md.constrain.Rigid()
    
    if included_secondary_types is None:
        included_secondary_types = particle_model.secondary_types
    
    types_and_positions = [
        [t, position]
        for t in included_secondary_types
        for position in particle_model.get_secondary_positions_by_type(t)
    ]

    rigid.body[particle_model.primary_type] = {
        "constituent_types": [row[0] for row in types_and_positions],
        "positions": [row[1] for row in types_and_positions],
        "orientations": [(1.0, 0.0, 0.0, 0.0) for _ in types_and_positions]
    }

    if create_bodies:
        rigid.create_bodies(simulation.state)
        simulation.operations.integrator.rigid = rigid

    return simulation, rigid

def add_gsd_writer(
    simulation: hoomd.Simulation,
    gsd_filename: str,
    compute: hoomd.md.compute.ThermodynamicQuantities | None = None
) -> Tuple[hoomd.Simulation, hoomd.md.compute.ThermodynamicQuantities]:
    """Add a GSD writer that logs potential energy to the simulation.

    Parameters
    ----------
    simulation : hoomd.Simulation
        The simulation to modify.
    gsd_filename : str
        The name of the GSD file to write.
    compute : hoomd.logging.Logger, optional
        An existing thermodynamic computer instance to use. If not provided, a
        new one is created.

    Returns
    -------
    simulation, compute
        The modified simulation and its thermodynamic computer.
    """
    gsd_writer = hoomd.write.GSD(
        1,
        filename=gsd_filename,
        mode="wb"
    )

    if compute is None: 
        compute = hoomd.md.compute.ThermodynamicQuantities(
            filter=hoomd.filter.All()
        )
        simulation.operations.computes.append(compute)

    logger = hoomd.logging.Logger()
    logger.add(compute, quantities=["potential_energy"])
    
    simulation.operations.writers.append(gsd_writer)

    return simulation, compute

def add_table_writer(
    simulation: hoomd.Simulation,
    csv_file: TextIOWrapper,
    probe_model: ParticleModel,
    compute: hoomd.md.compute.ThermodynamicQuantities | None = None
) -> Tuple[hoomd.Simulation, hoomd.logging.Logger]:
    """Add a table writer that logs potential energy to the simulation.

    Parameters
    ----------
    simulation : hoomd.Simulation
        The simulation to modify.
    csv_file : TextIOWrapper
        The file object to which the table will be written.
    compute : hoomd.logging.Logger, optional
        An existing thermodynamic computer instance to use. If not provided, a
        new one is created.

    Returns
    -------
    simulation, compute
        The modified simulation and its thermodynamic computer.
    """
    logger = hoomd.logging.Logger(categories=["scalar", "string"])
    logger.add(simulation, quantities=["timestep"])

    snapshot = simulation.state.get_snapshot()
    probe_index = get_primary_particle_index(snapshot, probe_model)

    def probe_position():
        with simulation.state.cpu_local_snapshot as snapshot:
            return snapshot.particles.position[
                snapshot.particles.rtag[probe_index]
            ][0]

    def probe_orientation():
        with simulation.state.cpu_local_snapshot as snapshot:
            return snapshot.particles.orientation[
                snapshot.particles.rtag[probe_index]
            ][0]

    logger["x"] = (lambda: probe_position()[0], "scalar")
    logger["y"] = (lambda: probe_position()[1], "scalar")
    logger["z"] = (lambda: probe_position()[2], "scalar")
    logger["q0"] = (lambda: probe_orientation()[0], "scalar")
    logger["q1"] = (lambda: probe_orientation()[1], "scalar")
    logger["q2"] = (lambda: probe_orientation()[2], "scalar")
    logger["q3"] = (lambda: probe_orientation()[3], "scalar")

    if compute is None: 
        compute = hoomd.md.compute.ThermodynamicQuantities(
            filter=hoomd.filter.All()
        )
        simulation.operations.computes.append(compute)
    
    logger.add(compute, quantities=["potential_energy"])

    table_writer = hoomd.write.Table(
        trigger=1,
        output=csv_file,
        logger=logger,
        delimiter=",",
        pretty=False
    )

    simulation.operations.writers.append(table_writer)

    return simulation, compute

def add_interaction(
    simulation: hoomd.Simulation,
    nlist: hoomd.md.nlist.NeighborList,
    interaction: Interaction
) -> hoomd.Simulation:
    """Add an interaction to the simulation.

    Parameters
    ----------
    simulation : hoomd.Simulation
        The simulation to modify
    nlist : hoomd.md.nlist.NeighborList
        The neighborlist to use for the interaction.
    interaction : Interaction
        The interaction to add.

    Returns
    -------
    simulation
        The modified simulation.
    """
    force = interaction.hoomd_class(nlist, **interaction.inputs)
    all_types = simulation.state.particle_types
    all_type_pairs = list(itertools.combinations(all_types, 2))
    all_type_pairs.extend([(t, t) for t in all_types])
    yes_type_pairs = []
    for p in all_type_pairs:
        if (
            p[0] in interaction.yes_types
            and p[1] in interaction.yes_types
            and p[0] != p[1]
        ):
            yes_type_pairs.append(p)
    
    def wrong_type_msg(att, default_or_yes, single_or_double, hoomd_class):
        return (
            f"'{att}' was provided as a {default_or_yes} {single_or_double}-"
            f"typed-attribute, but no such attribute was found in hoomd class "
            f"'{hoomd_class}'."
        )

    # Set default single attributes
    for a_t in all_types:
        for k, v in interaction.default_single_typed_attributes.items():
            try:
                getattr(force, k)[a_t] = v
            except AttributeError:
                raise AttributeError(
                    wrong_type_msg(
                        k, "default", "single", interaction.hoomd_class
                    )
                )

    # Set default pair attributes
    for a_p in all_type_pairs:
        for k, v in interaction.default_pair_typed_attributes.items():
            try:
                getattr(force, k)[a_p] = v
            except AttributeError:
                raise AttributeError(
                    wrong_type_msg(
                        k, "default", "pair", interaction.hoomd_class
                    )
                )

    # Modify single attributes for interacting types
    for y_t in interaction.yes_types:
        for k, v in interaction.yes_single_typed_attributes.items():
            try:
                getattr(force, k)[y_t] = v
            except AttributeError:
                raise AttributeError(
                    wrong_type_msg(
                        k, "yes", "single", interaction.hoomd_class
                    )
                )

    # Modify pair attributes for interacting types
    for y_p in yes_type_pairs:
        for k, v in interaction.yes_pair_typed_attributes.items():
            try:
                getattr(force, k)[y_p] = v
            except AttributeError:
                raise AttributeError(
                    wrong_type_msg(
                        k, "yes", "pair", interaction.hoomd_class
                    )
                )

    simulation.operations.integrator.forces.append(force)

    return simulation

def add_integrator(
    simulation: hoomd.Simulation,
    rigid: hoomd.md.constrain.Rigid | None = None,
) -> hoomd.Simulation:
    """Add a constant Volume MD integrator to the simulation.

    Parameters
    ----------
    simulation : hoomd.Simulation
        The simulation to modify.
    rigid : hoomd.md.constrain.Rigid, optional
        The rigid constraint.

    Returns
    -------
    simulation
        The modified simulation.
    """
    integrator = hoomd.md.Integrator(
        dt=0.000000000001,
        integrate_rotational_dof=True
    )
    simulation.operations.integrator = integrator
    
    if rigid is not None:
        integrator.rigid = rigid

    # NOTE: no method is needed because no particle movement is wanted.
    
    return simulation

def get_primary_particle_index(snapshot, particle_model):
    """TODO"""
    all_types = snapshot.particles.types
    particle_index = deepcopy(np.where(
        snapshot.particles.typeid == all_types.index(particle_model.primary_type)
    ))
    return particle_index
