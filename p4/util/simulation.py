# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

"""Functions for creating and manipulating HOOMD-blue simulations."""

from __future__ import annotations
from collections.abc import Iterable
from io import StringIO, TextIOWrapper
from typing import Literal, Tuple
import warnings
from packaging.version import Version
from functools import partial

import gsd
import hoomd
import numpy as np

# Tqdm must be imported differently for python vs ipython
try:
    get_ipython()
except NameError:
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm

from ..types import OrientationsLike, PositionsLike


def snapshot_to_frame(snapshot: hoomd.Snapshot):
    """Convert a HOOMD-blue Snapshot to a GSD Frame.
    
    This function only copies the following data:
    
    * configuration.box
    * particles.N
    * particles.types
    * particles.typeid
    * particles.position
    * particles.orientation
    * particles.mass
    * particles.moment_inertia
     
    All other data is ignored.
    """
    frame = gsd.hoomd.Frame()

    frame.configuration.box = snapshot.configuration.box
    frame.particles.N = snapshot.particles.N
    frame.particles.types = snapshot.particles.types
    frame.particles.typeid = snapshot.particles.typeid
    frame.particles.position = snapshot.particles.position
    frame.particles.orientation = snapshot.particles.orientation
    frame.particles.mass = snapshot.particles.mass
    frame.particles.moment_inertia = snapshot.particles.moment_inertia
    frame.particles.body = snapshot.particles.body

    return frame

def get_initial_frame(
    probe: "Body",
    analyte: "Body" | "Arrangement",
    simulation_box: list[float],
) -> gsd.hoomd.Frame:
    """Return a simulation frame with the probe and analyte at the origin.

    Parameters
    ----------
    probe : Body
        The probe body.
    analyte : Body | Arrangement
        The analyte body or arrangement.
    simulation_box : list[float]
        The dimensions of the simulation box. Must be an array of 6 floats
        representing ``[Lx, Ly, Lz, xy, xz, yz]``.

    Returns
    -------
    gsd.hoomd.Frame
    """
    # Create separate frames
    probe_frame = snapshot_to_frame(probe.to_hoomd_snapshot())
    analyte_frame = snapshot_to_frame(analyte.to_hoomd_snapshot())

    # Start with the probe at [0, 0, 0] - this shouldn't be a problem if it is
    # moved before simulation.run()
    probe_frame.particles.position -= np.array([0, 0, 0])

    # Merge all data together, placing the probe particles after the analyte
    # particles in the frame data
    merged_frame = gsd.hoomd.Frame()

    merged_frame.configuration.box = simulation_box
    merged_frame.particles.N = (
        probe_frame.particles.N + analyte_frame.particles.N
    )
    merged_frame.particles.types = (
        analyte_frame.particles.types
        + [
            t
            for t in probe_frame.particles.types
            if t not in analyte_frame.particles.types
        ]
    )
    merged_frame.particles.typeid = np.hstack((
        analyte_frame.particles.typeid,
        [
            merged_frame.particles.types.index(
                probe_frame.particles.types[tid]
            )
            for tid in probe_frame.particles.typeid
        ]
    ))
    merged_frame.particles.position = np.vstack((
        analyte_frame.particles.position,
        probe_frame.particles.position
    ))
    merged_frame.particles.orientation = np.vstack((
        analyte_frame.particles.orientation,
        probe_frame.particles.orientation
    ))
    merged_frame.particles.mass = np.hstack((
        analyte_frame.particles.mass,
        probe_frame.particles.mass
    ))
    merged_frame.particles.moment_inertia = np.vstack((
        analyte_frame.particles.moment_inertia,
        probe_frame.particles.moment_inertia
    ))
    if probe_frame.particles.body[0] == -1:
        merged_frame.particles.body = np.hstack((
            analyte_frame.particles.body,
            probe_frame.particles.body
        ))
    else:
        merged_frame.particles.body = np.hstack((
            analyte_frame.particles.body,
            probe_frame.particles.body + len(analyte_frame.particles.body)
        ))

    return merged_frame

def add_gsd_writer(
    simulation: hoomd.Simulation,
    gsd_filename: str,
    compute: hoomd.md.compute.ThermodynamicQuantities | None = None
) -> tuple[hoomd.Simulation, hoomd.md.compute.ThermodynamicQuantities]:
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
    tuple[hoomd.Simulation, hoomd.md.compute.ThermodynamicQuantities]
        The modified simulation and its thermodynamic computer.
    """
    if compute is None: 
        compute = hoomd.md.compute.ThermodynamicQuantities(
            filter=hoomd.filter.All()
        )
        simulation.operations.computes.append(compute)

    logger = hoomd.logging.Logger()
    logger.add(compute, quantities=["potential_energy"])

    gsd_writer = hoomd.write.GSD(
        1,
        filename=gsd_filename,
        mode="wb",
        logger=logger
    )

    simulation.operations.writers.append(gsd_writer)

    return simulation, compute

def add_table_writer(
    simulation: hoomd.Simulation,
    csv_file: TextIOWrapper,
    quantities: Literal["U", "F", "T"] | list[Literal["U", "F", "T"]],
    probe_is_rigid: bool,
    probe_index: int,
    compute: hoomd.md.compute.ThermodynamicQuantities | None = None
) -> Tuple[hoomd.Simulation, hoomd.md.compute.ThermodynamicQuantities]:
    """Add a table writer that logs named quantities to the simulation.

    Parameters
    ----------
    simulation : hoomd.Simulation
        The simulation to modify.
    csv_file : TextIOWrapper
        The file object to which the table will be written.
    quantities : one or more of 'U', 'F', 'T'
        The quantities to measure. 'U' is the potential energy measured for
        the entire system. 'F' and 'T' are the net Force and Torque experienced
        by the probe.
    probe_is_rigid : bool
        Whether the probe is a rigid body. Required for proper summing of forces
        and torques.
    probe_index : int
        The index of the probe's central particle.
    compute : hoomd.md.compute.ThermodynamicQuantities, optional
        An existing thermodynamic computer instance to use. If not provided, a
        new one is created.

    Returns
    -------
    tuple[hoomd.Simulation, hoomd.md.compute.ThermodynamicQuantities]
        The modified simulation and its thermodynamic computer.
    """
    logger = hoomd.logging.Logger(categories=["scalar", "string"])

    def probe_position():
        with simulation.state.cpu_local_snapshot as snapshot:
            return np.array(snapshot.particles.position[
                snapshot.particles.rtag[probe_index]
            ])

    def probe_orientation():
        with simulation.state.cpu_local_snapshot as snapshot:
            return np.array(snapshot.particles.orientation[
                snapshot.particles.rtag[probe_index]
            ])
    
    def probe_net_force():
        if probe_is_rigid:
            return simulation.operations.integrator.rigid.forces[probe_index]
        else:
            net_force = np.array([0.0, 0.0, 0.0])
            for f in simulation.operations.integrator.forces:
                net_force += f.forces[probe_index]
            return net_force

    def probe_net_torque():
        if probe_is_rigid:
            return simulation.operations.integrator.rigid.torques[probe_index]
        else:
            net_torque = np.array([0.0, 0.0, 0.0])
            for f in simulation.operations.integrator.forces:
                net_torque += f.torques[probe_index]
            return net_torque

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
    
    if isinstance(quantities, str):
        quantities = [quantities]
    
    if "U" in quantities:
        logger.add(compute, quantities=["potential_energy"])
    
    if "F" in quantities:
        logger["Fx"] = (lambda: probe_net_force()[0], "scalar")
        logger["Fy"] = (lambda: probe_net_force()[1], "scalar")
        logger["Fz"] = (lambda: probe_net_force()[2], "scalar")

    if "T" in quantities:
        logger["Tx"] = (lambda: probe_net_torque()[0], "scalar")
        logger["Ty"] = (lambda: probe_net_torque()[1], "scalar")
        logger["Tz"] = (lambda: probe_net_torque()[2], "scalar")

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
    interaction: "Interaction",
) -> hoomd.Simulation:
    """Add an interaction to a simulation.

    Parameters
    ----------
    simulation : hoomd.Simulation
        The simulation to modify.
    nlist : hoomd.md.nlist.NeighborList
        The neighbor list to use for the interaction.
    interaction : Interaction
        The interaction to add.

    Returns
    -------
    simulation
        The modified simulation.
    """
    force = interaction.to_hoomd_pair(nlist=nlist)
    simulation.operations.integrator.forces.append(force)
    return simulation

def add_integrator(
    simulation: hoomd.Simulation,
    rigid: hoomd.md.constrain.Rigid | None = None,
) -> hoomd.Simulation:
    """Add a constant volume MD integrator to a simulation.

    Parameters
    ----------
    simulation : hoomd.Simulation
        The simulation to modify.
    rigid : hoomd.md.constrain.Rigid, optional
        The rigid constraint. If not provided, none is attached to the
        integrator.

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

def get_simulation(
    system: "System",
    nlist: hoomd.md.nlist.NeighborList,
    simulation_box: list[float]
) -> hoomd.Simulation:
    """Return a simulation for a system with specified boxes and interactions.

    Parameters
    ----------
    system : System
        The system.
    nlist : hoomd.md.nlist.NeighborList
        The neighbor list to use for the interactions in the simulation.
    simulation_box : list[float]
        The simulation's box in HOOMD notation. Must be an array of 6 floats
        representing ``[Lx, Ly, Lz, xy, xz, yz]``.

    Returns
    -------
    hoomd.Simulation
        The simulation object.
    """
    # Create initial frame
    frame = get_initial_frame(
        system.probe,
        system.analyte,
        simulation_box
    )

    # Initialize Simulation
    simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    simulation.create_state_from_snapshot(
        hoomd.Snapshot.from_gsd_frame(
            gsd_snap=frame,
            communicator=hoomd.communicator.Communicator()
        )
    )

    # Fix out-of-date body tags for the probe
    n_total = simulation.state.get_snapshot().particles.N
    n_probe = system.probe.to_hoomd_snapshot().particles.N
    probe_start = n_total - n_probe

    simulation.state.get_snapshot().particles.body[probe_start] = probe_start
    simulation.state.get_snapshot().particles.body[
        probe_start + 1 : n_total
    ] = probe_start

    # Create rigid constraint
    # NOTE: probe is added to rigid constraint even if its secondary
    # particles are not "actively interacting"
    rigid = system.analyte.to_hoomd_rigid()
    rigid = system.probe.to_hoomd_rigid(rigid)

    # Add integrator
    simulation = add_integrator(simulation, rigid)

    # Add required interactions
    for interaction in system.interactions:
        simulation = add_interaction(
            simulation=simulation,
            nlist=nlist,
            interaction=interaction
        )

    return simulation

def measure(
    system: "System",
    quantities: Literal["U", "F", "T"] | list[Literal["U", "F", "T"]],
    positions: PositionsLike,
    orientations: list[OrientationsLike],
    simulation_box: list[float],
    gsd_filename: str | None,
    nlist: hoomd.md.nlist.NeighborList,
    pbar_number: int,
    disable_pbar: bool
) -> StringIO:
    """Measure named quantities for a system.

    Parameters
    ----------
    system : System
        The System to measure.
    quantities : one or more of 'U', 'F', 'T'
        The quantities to measure. 'U' is the potential energy measured for
        the entire system, and is saved as a single scalar quantity. 'F' and
        'T' are the net Force and Torque experienced by the probe, and are
        saved as vector quantities.
    positions : PositionsLike
        The positions to measure at.
    orientations : list[OrientationsLike]
        The orientations (in quaternion form) to measure at. A separate array
        of orientations must be provided for every position.
    simulation_box : list[float]
        The simulation's box in HOOMD notation. Must be an array of 6 floats
        representing ``[Lx, Ly, Lz, xy, xz, yz]``.
    gsd_filename : str
        The name of the GSD file to save. If not provided, no GSD file will be
        saved.
    nlist : hoomd.md.nlist.NeighborList
        The neighbor list to use for the interactions.
    pbar_position : int
        The position of the tqdm progress bar. ``0`` is outermost, ``1`` is
        next, and so on.
    disable_pbar : bool
        Whether to disable the progress bar.

    Returns
    -------
    StringIO
        The tabular results of the measurement simulation, formatted as a CSV
        and stored in a string buffer.
    """
    # Warn if there is a risk of table writer erroring
    if Version(hoomd.version.version) < Version("6.1.0"):
        warnings.warn(
            f"Outdated HOOMD-blue version: '{hoomd.version.version}'. Table "
            + "writer will error if U, F, or T are NaN or Inf. Resolve this "
            + "issue by upgrading to the latest version of HOOMD-blue."
        )

    # Create simulation
    simulation = get_simulation(system, nlist, simulation_box)

    # Calculate the index of the probe's central particle
    n_probe = system.probe.to_hoomd_snapshot().particles.N
    probe_index = simulation.state.N_particles - n_probe

    # Add file writers
    if gsd_filename is not None:
        simulation, compute = add_gsd_writer(simulation, gsd_filename)
    
    table = StringIO()
    simulation, _ = add_table_writer(
        simulation=simulation,
        csv_file=table,
        quantities=quantities,
        probe_is_rigid=system.probe._is_rigid(system.interactions),
        probe_index=probe_index,
        compute=None if gsd_filename is None else compute,
    )
    # breakpoint()

    # Iterate over positions
    pbar = partial(
        tqdm,
        total=positions.shape[0],
        desc=f"#{pbar_number}",
        position=pbar_number,
        disable=disable_pbar,
    )
    for p, os in pbar(zip(positions, orientations)):
        for o in os:
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
