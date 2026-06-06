# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

import itertools

import hoomd
import gsd
import numpy as np
import plotly
import rowan
import coxeter
import scipy

from . import util
from .types import (
    AxisLike,
    StateLike,
    PositionsLike,
    OrientationsLike
)
from .arrangement import Arrangement
from .body import Body


# ---------------------------------- SAMPLING ----------------------------------


def positions_on_regular_grid(
    box: list[float] | tuple[float, float, float],
    resolution: list[int] | tuple[int, int, int]
) -> np.ndarray:
    """Return a regular grid of positions bounded by a box.

    Parameters
    ----------
    box : (3,) array of floats
        The lengths of the sides of the box in X, Y, Z order.
    resolution : (3,) array of ints
        The resolution of the point grid for each dimension in X, Y, Z order.
    
    Returns
    -------
    (..., 3) np.ndarray
    """
    positions = np.array(list(itertools.product(
        np.linspace(-box[0]/2, box[0]/2, resolution[0], endpoint=False),
        np.linspace(-box[1]/2, box[1]/2, resolution[1], endpoint=False),
        np.linspace(-box[2]/2, box[2]/2, resolution[2], endpoint=False),
    )))

    positions[:,0] += (box[0]/resolution[0])/2
    positions[:,1] += (box[1]/resolution[1])/2
    positions[:,2] += (box[2]/resolution[2])/2

    return positions

# def exclude_positions_by_shape(
#     positions: PositionsLike,
#     shape: coxeter.shapes.Polyhedron,
#     inside: bool = True,
# ):
#     """Remove positions inside or outside a shape with an optional buffer.

#     Parameters
#     ----------
#     positions : (N, 3) array of floats
#         The positions to filter.
#     shape : coxeter.shapes.Polyhedron
#         The shape to check against positions.
#     inside : bool, default=True
#         Whether to exclude positions that are inside the shape (``True``) or
#         outside (``False``).
    
#     Returns
#     -------
#     (..., 3) np.ndarray
#     """
#     if inside:
#         return np.array(positions)[~shape.is_inside(positions)]
#     else:
#         return np.array(positions)[shape.is_inside(positions)]

def orientations_about_axis(n: int, axis: AxisLike, k: int = 1) -> np.ndarray:
    """Return an array of quaternions sampling orientation about a given axis.

    Parameters
    ----------
    n : int
        The number of samples.
    axis : (3,) array of floats
        The axis of rotation. For example, ``[1, 0, 0]`` is the x-axis.
    k : int, default=1
        The order of the rotational symmetry about the axis. Samples are
        constrained to the range ``0 - 360/k``. ``1`` corresponds to **C1**
        symmetry, ``2`` to **C2**, and so on.

    Returns
    -------
    (..., 4) np.ndarray
    """
    return rowan.from_axis_angle(
        axes=axis,
        angles=np.linspace(0, 2*np.pi/k, n, endpoint=False)
    )

def orientations_from_fibonacci_lattice(
    n: int,
    group: str | None = None,
    ideal_group_index = 0   # TODO: remove
) -> np.ndarray:
    """A near-uniform grid of `n` quaternions.

    This is equivalent to a Fibonacci lattice on the 3-sphere. See
    `this paper <https://ieeexplore.ieee.org/document/9878746>`_ for a
    derivation.

    To sample only a section of the 3-sphere, pass a string representing the
    symmetry group.

    Parameters
    ----------
    n : int
        The number of points in the lattice.
    group : str, optional
        The symmetry group. TODO: elaborate
    
    Returns
    -------
    (..., 4) np.ndarray
    """
    if group is not None:
        group_quaternions = (
            scipy.spatial.transform.Rotation
                .create_group(group)
                .as_quat(scalar_first=True)
        )
        n *= group_quaternions.shape[0]

    PSI = 1.533751168755204288118041413  # Solution to ψ**4 = ψ + 4
    s = np.arange(n) + 1 / 2
    t = s / n
    d = 2 * np.pi * s
    r0, r1 = (np.sqrt(t), np.sqrt(1 - t))
    a, b = (d / np.sqrt(2), d / PSI)

    # Allocate as rows and then transpose, rather than stacking columns
    result = np.empty((4, n))
    result[...] = r0 * np.sin(a), r0 * np.cos(a), r1 * np.sin(b), r1 * np.cos(b)
    quaternions = result.T
    
    # Filter quaternions, only keeping ones which are "closest" to the same
    # group quaternion. The chosen group quaternion is the first one, and
    # distance is evaluated as the symmetric intrinsic distance.
    # TODO: validate this
    if group is not None:
        filtered_quaternions = []
        for q in quaternions:
            best_group_index = -1
            best_distance = float("inf")
            for i, g in enumerate(group_quaternions):
                d = rowan.geometry.sym_intrinsic_distance(g, q)
                if d < best_distance:
                    best_distance = d
                    best_group_index = i
            
            if best_group_index == ideal_group_index:
                filtered_quaternions.append(q)

        return np.asarray(filtered_quaternions)
    
    else:
        return quaternions

# def exclude_orientations_with_shape_overlap(
#     positions,
#     probe,
#     analyte,
#     probe_type_shapes,
#     analyte_type_shapes
# ):
#     pass


# ---------------------------------- PLOTTING ----------------------------------


def plot_positions(
    positions: PositionsLike,
    box: list[float] | None = None,
    color: str = "cornflowerblue",
    opacity: float = 1.0,
    size: float = 5.0,
    box_color: str = "grey",
    box_opacity: float = 1.0,
    box_line_width: float = 4.0,
    **layout_kwargs # TODO
):
    """Plot positions in 3D space.
    
    Informally evaluate sampling coverage by plotting sampled positions.

    Parameters
    ----------
    positions : (N, 3) array of floats
        The positions to plot.
    box : (3,) array of floats, optional
        The size of the system box, expressed as a (3,) array of side lengths.
        If not provided, no box is plotted.
    color : str, default='cornflowerblue'
        The color of the position markers.
    opacity : float, default=1.0
        The opacity of the position markers.
    size : float, default=5.0
        The size of the position markers.
    box_color : str, default='grey'
        The color of the box.
    box_opacity : float, default=1.0
        The opacity of the box.
    box_line_width : float, default=4.0
        The line width of the box.
    """
    positions = np.asarray(positions)

    figure = plotly.graph_objects.Figure()

    positions_trace = plotly.graph_objects.Scatter3d(
        x=positions[:,0],
        y=positions[:,1],
        z=positions[:,2],
        mode="markers",
        marker=dict(
            size=size,
            color=color,
            opacity=opacity,
            line=dict(width=2, color="DarkSlateGrey")
        ),
        showlegend=False
    )

    figure.add_trace(positions_trace)

    if box is not None:
        cube = coxeter.families.PlatonicFamily.get_shape("Cube")

        vertices = np.asarray(box) * cube.vertices
        edges = cube.edges
    
        data = []   # TODO: make this numpy

        for edge in edges:
            data.append([vertices[edge[0], i] for i in [0, 1, 2]])
            data.append([vertices[edge[1], i] for i in [0, 1, 2]])
            data.append([None, None, None])
        
        data = np.asarray(data)
        
        figure.add_trace(
            plotly.graph_objects.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                mode='lines',
                opacity=box_opacity,
                line=dict(color=box_color, width=box_line_width),
                showlegend=False
            )
        )
    
    figure.update_layout(util.plot_layout(**layout_kwargs))

    return figure, positions_trace

def plot_state(
    obj: hoomd.Simulation | StateLike,
    include_singles: bool = False,
    type_shapes: dict[str, coxeter.shapes.Polyhedron] | None = None,
    type_styles: dict[str, dict] | None = None,
    ignore_types: list[str] | None = None,
    **layout_kwargs
):
    """Visualize the state of a system.
    
    This convenience function for :py:meth:~p4.Arrangement.plot` accepts any
    object that resembles or contains a simulation state.

    Parameters
    ----------
    TODO
    """
    if isinstance(obj, hoomd.Simulation):
        arrangement = Arrangement.from_hoomd_simulation(obj, include_singles)
    elif isinstance(obj, hoomd.State):
        snapshot = obj.get_snapshot()
        arrangement = Arrangement.from_hoomd_snapshot(snapshot, include_singles)
    elif isinstance(obj, hoomd.Snapshot):
        arrangement = Arrangement.from_hoomd_snapshot(obj, include_singles)
    elif isinstance(obj, gsd.hoomd.Frame):
        snapshot = hoomd.Snapshot.from_gsd_frame(
            gsd_snap=obj,
            communicator=hoomd.communicator.Communicator()
        )
        arrangement = Arrangement.from_hoomd_snapshot(snapshot, include_singles)
    
    figure, traces = arrangement.plot(
        type_shapes=type_shapes,
        type_styles=type_styles,
        ignore_types=ignore_types,
        **layout_kwargs
    )

    figure.show()

    return figure, traces


# ------------------------------------ OTHER -----------------------------------


# def default_params(hoomd_class: hoomd.md.pair.Pair):
#     pass