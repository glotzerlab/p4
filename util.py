"""Utiliity functions for p4.

In general, these functions have no defaults and do not protect themselves
from wrong/fallible inputs.
"""

from copy import deepcopy
import csv
from io import StringIO, TextIOWrapper
import itertools
from typing import Tuple
import coxeter
import gsd.hoomd
import hoomd
import numpy as np
import rowan

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from p4 import Interaction, ParticleModel, System

def get_cube(side_length: float):
    """Return a coxeter cube with a given side length."""
    s = side_length/2
    vertices = [
        [-s, -s, -s],
        [-s, -s,  s],
        [-s,  s, -s],
        [-s,  s,  s],
        [ s, -s, -s],
        [ s, -s,  s],
        [ s,  s, -s],
        [ s,  s,  s]
    ]
    return coxeter.shapes.ConvexPolyhedron(vertices)

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

def exclude_positions_by_shape(
    positions: list[list[float]],
    shape: coxeter.shapes.ConvexPolyhedron,
    exclude_inside: bool,
    buffer: float=0.0,
):
    """Remove positions inside or outside a shape with an optional buffer.

    Parameters
    ----------
    positions : list[list[float]]
        Array of positions. (..., 3)
    shape : coxeter.shapes.ConvexPolyhedron
        The shape to check against positions.
    exclude_inside : bool
        Whether to exclude positions that are inside the shape (True) or
        outside (False).
    buffer : float, optional
        An buffer distance for the shape. If greater than zero, the provided
        `shape` is converted into a ConvexSpheroPolyhedron and exclusion checks
        are performed on that instead. If smaller than zero, the provided shape
        is shrunk by the factor ((r-b)/r), where b is the absolute value of the
        buffer distance and r is the radius of the maximal centered bounded
        sphere. [TODO: check that this is ok]

    Returns
    -------
    np.ndarray
        The positions that are not excluded.
    """
    if buffer > 0:
        shape = coxeter.shapes.ConvexSpheropolyhedron(shape.vertices, buffer)
    if buffer < 0:
        r = shape.maximal_centered_bounded_sphere_radius
        shape.vertices *= (r - buffer) / r
    
    if exclude_inside:
        return np.array(positions)[~shape.is_inside(positions)]
    else:
        return np.array(positions)[shape.is_inside(positions)]

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

    if particle_model.get_secondary_orientations_by_type is not None:
        orientations = [
            orientation
            for t in included_secondary_types
            for orientation in particle_model.get_secondary_orientations_by_type(t)
        ]
    else:
        orientations = [(1.0, 0.0, 0.0, 0.0) for _ in types_and_positions]

    rigid.body[particle_model.primary_type] = {
        "constituent_types": [t for (t, p) in types_and_positions],
        "positions": [p for (t, p) in types_and_positions],
        "orientations": [o for o in orientations]
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
    force = interaction.hoomd_class(nlist, **interaction.initial_inputs)
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

def get_primary_particle_index(
    snapshot: hoomd.Snapshot,
    particle_model: ParticleModel
):
    """Return the snapshot's particle index for the model's primary type.

    The returned index corresponds to the **first** occurance of the given type.
    
    Parameters
    ----------
    snapshot : Snapshot
        The snapshot containing the particle of interest.
    particle_model : ParticleModel
        The model whose primary type is of interest.
    
    Returns
    -------
    index
    """
    all_types = snapshot.particles.types
    particle_index = deepcopy(np.where(
        snapshot.particles.typeid == all_types.index(particle_model.primary_type)
    ))
    return particle_index

def find_nearest(array, value):
    """TODO"""
    # Ref: https://stackoverflow.com/a/2566508/15426433
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_simulation(
    system: System,
    interactions_to_include: list[str],
    nlist: hoomd.md.nlist.NeighborList,
    probe_box: list[float],
    simulation_box: list[float]
) -> hoomd.Simulation:
    """Return a simulation for a System with specified boxes and interactions.

    Parameters
    ----------
    system : System
        The system containing the probe and analyte particle models, as well as
        the interaction model for the simulation.
    interactions_to_include : list[str]
        The names of the interactions from the system's interaction model to
        include in the simulation.
    nlist : hoomd.md.nlist.NeighborList
        The neighbor list to use for the interactions in the simulation.
    probe_box : list[float]
        The side lengths $[Lx, Ly, Lz]$ of the box containing the positions to
        be probed.
    simulation_box : list[float]
        The simulation's box in HOOMD notation. $[Lx, Ly, Lz, xy, xz, yz]$

    Returns
    -------
    hoomd.Simulation
        The simulation object, fully prepared and ready to be run.
    """
    # Determine needed interactions and particle types
    interactions = system.interactions(interactions_to_include)
    types = system.interacting_types(interactions_to_include)

    # Create initial frame
    frame = get_initial_frame(
        system.probe_model,
        system.analyte_model,
        types,
        probe_box,
        simulation_box
    )

    # Initialize Simulation
    simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    simulation.create_state_from_snapshot(frame)

    # Add integrator
    simulation = add_integrator(simulation)

    # Add rigid bodies if necessary
    probe_is_rigid = particle_must_be_rigid_body(
        system.probe_model,
        interactions
    )
    analyte_is_rigid = particle_must_be_rigid_body(
        system.analyte_model,
        interactions
    )
    if probe_is_rigid:
        simulation, rigid = add_rigid_constraint(
            simulation,
            system.probe_model,
            False if analyte_is_rigid else True,
            [t for t in system.probe_model.secondary_types if t in types]
        )
    if analyte_is_rigid:
        simulation, _ = add_rigid_constraint(
            simulation,
            system.analyte_model,
            True,
            [t for t in system.analyte_model.secondary_types if t in types],
            rigid if probe_is_rigid else None
        )

    # Add required interactions
    for interaction in interactions:
        simulation = add_interaction(simulation, nlist, interaction)
    
    return simulation

def run_probe(
    system: System,
    probe_positions: list[list[float]],
    probe_orientations: list[list[float]],
    interactions_to_include: list[str],
    nlist: hoomd.md.nlist.NeighborList,
    probe_box: list[float],
    simulation_box: list[float],
    gsd_filename: str | None = None,
) -> StringIO:
    """Return the probe data table for a system.

    Parameters
    ----------
    system : System
        The System to probe.
    probe_positions : list[list[float]]
        The positions to probe at.
    probe_orientations : list[list[float]]
        The orientations to probe at each position (in quaternion form).
    interactions_to_include : list[str]
        The names of the interactions from the system's interaction model to
        include in the simulation.
    gsd_filename : str
        The name of the final output CSV file. This is not used to actually
        write 
    nlist : hoomd.md.nlist.NeighborList
        The neighbor list to use for the interactions.
    probe_box : list[float]
        The side lengths $[Lx, Ly, Lz]$ of the box containing the positions to
        be probed.
    simulation_box : list[float]
        The simulation's box in HOOMD notation. $[Lx, Ly, Lz, xy, xz, yz]$
    gsd_filename : str, optional
        The name of the GSD file to save. If not provided, no GSD file will be
        saved.

    Returns
    -------
    table
        The tabular results of the probe simulation, formatted as a CSV and
        stored in a string buffer.
    """
    # Create simulation
    simulation = get_simulation(
        system,
        interactions_to_include,
        nlist,
        probe_box,
        simulation_box
    )

    # Add file writers
    if gsd_filename is not None:
        simulation, compute = add_gsd_writer(simulation, gsd_filename)
    
    table = StringIO()
    simulation, _ = add_table_writer(
        simulation=simulation,
        csv_file=table,
        probe_model=system.probe_model,
        compute=None if gsd_filename is None else compute
    )
    
    probe_index = get_primary_particle_index(
        simulation.state.get_snapshot(),
        system.probe_model
    )

    # Iterate over positions
    for p in probe_positions:
        for o in probe_orientations:
            with simulation.state.cpu_local_snapshot as state:

                # Note: only probe position and orientation need to be
                # reset. No forces can change the probe particle's velocity
                # or angular momentum, nor can anything change the analyte's
                # properties because there is no integration method.
                state.particles.position[
                    state.particles.rtag[probe_index]
                ] = p
                state.particles.orientation[
                    state.particles.rtag[probe_index]
                ] = o

            simulation.run(1)

    # Flush all writers, just to make sure
    for writer in simulation.operations.writers:
        if hasattr(writer, "flush"):
            writer.flush()

    return table

def subdivide(array: list, n: int):
    """Subdivide an array into some number of chunks of consecutive items.

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
    return [array[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def merge_tables(table_csvs: list[StringIO]):
    """Combine an array of tables stored in string buffers.

    Additionally, the header is modified to have shorter column names.

    Parameters
    ----------
    table_csvs : list[StringIO]
        The tables to merge. The tables should be in CSV format and stored
        in string buffers.

    Returns
    -------
    table
        The merged table.
    """
    merged_csv = StringIO()
    writer = csv.writer(merged_csv)

    # Write simpler header
    writer.writerow(["t","x","y","z","q0","q1","q2","q3","PE"])

    for i, table in enumerate(table_csvs):
        table.seek(0)
        reader = csv.reader(table)
        
        next(reader)    # skip the header

        for row in reader:
            writer.writerow(row)
        
    return merged_csv
