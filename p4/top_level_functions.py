# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from copy import copy
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
    **kwargs
) -> tuple[plotly.graph_objs._figure.Figure, list]:
    """Visualize sampled positions to evaluate coverage around an analyte.

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
    **kwargs
        Other keyword arguments are passed to the function
        :py:func:`p4.plot_layout`.
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
    
    figure.update_layout(plot_layout(slice={}, **kwargs))

    return figure, positions_trace

def plot_state(
    obj: hoomd.Simulation | StateLike,
    type_shapes: dict[str, coxeter.shapes.Polyhedron] | None = None,
    type_styles: dict[str, dict] | None = None,
    ignore_types: list[str] | None = None,
    **kwargs
) -> tuple[plotly.graph_objects.Figure, list]:
    """Visualize the state of a system.
    
    This is a convenience function that creates an
    :py:class:`~p4.arrangement.Arrangement` from an object and then calls its 
    plot method.

    Parameters
    ----------
    obj : hoomd.Simulation or StateLike
        The simulation or state-like object to plot.
    type_shapes : dict, optional
        A mapping from particle type name [``str``] to shape
        [``coxeter.shapes.Polyhedron``]. If no shape is provided for a type,
        it will be plotted as a sphere.
    type_styles : dict, optional
        A mapping from particle type name to style, where style is given as
        a dictionary which may have the keys 'color', 'opacity', and 'size'.
        See above for more information.
    ignore_types : list[str], optional
        The names of the particle types to exclude from the plot.
    **kwargs
        Other keyword arguments are passed to the function
        :py:func:`p4.plot_layout`.
    """
    from .arrangement import Arrangement

    include_singles = True
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
        **kwargs
    )

    return figure, traces

def plot_layout(
    slice: dict[str, float],
    show_axes: bool = True,
    show_title: bool = False,
    show_ticks: bool = True,
    show_grid: bool = False,
    show_border: bool = True,
    show_legend: bool | None = None,
    clim: list[float] | None = None,
) -> dict:
    """Return a Plotly layout dictionary customized for a given slice.

    Parameters
    ----------
    slice : dict[str, float]
        Axes and positions along which to slice. Keys are limited to 'x',
        'y', and 'z'. There can be at most two keys.
    show_axes : bool, default=True
        Whether to show the axes.
    show_title : bool, default=False
        Whether to show the title.
    show_ticks : bool, default=True
        Whether to show tick marks on the axes.
    show_grid : bool, default=False
        Whether to show the axes grid.
    show_border : bool, default=True
        Whether to show the plot border.
    show_legend : bool, optional
        Whether to show the plot legend.
    clim : list of floats, optional
        The lower and upper limits of the colorscale.
    """
    # Build initial dictionary
    axis_style = dict(
        visible=show_axes,
        ticks="outside" if show_ticks else "",
        gridcolor="#e5e5e5" if show_grid else "rgba(0,0,0,0)",
        zerolinecolor="#e5e5e5" if show_grid else "rgba(0,0,0,0)",
        showline=show_border,
        linewidth=1,
        linecolor="black",
        mirror=True
    )

    if len(slice) == 0:
        axis_style["showbackground"] = False
        layout = dict(
            scene=dict(
                xaxis=axis_style,
                yaxis=axis_style,
                zaxis=axis_style,
                aspectmode="data",
            ),
            plot_bgcolor="rgba(0,0,0,0)"
        )
    
    elif len(slice) == 1:
        layout = dict(
            xaxis=copy(axis_style),
            yaxis=copy(axis_style),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        layout["xaxis"].update(
            scaleanchor="y",
            scaleratio=1,
            constrain="domain"
        )
        layout["yaxis"].update(
            scaleanchor="x",
            scaleratio=1,
            constrain="domain"
        )

    elif len(slice) == 2:
        layout = dict(
            xaxis=copy(axis_style),
            yaxis=copy(axis_style),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        if clim is not None:
            layout["yaxis_range"] = [min(clim), max(clim)]

    # Update layout dictionary with axis and figure titles (only 1D and 2D)
    if len(slice) == 1:
        if "x" in slice:
            x_title = "y"
            y_title = "z"
            fig_title = f"x = {float(slice["x"])}"
        elif "y" in slice:
            x_title = "x"
            y_title = "z"
            fig_title = f"y = {float(slice["y"])}"
        elif "z" in slice:
            x_title = "x"
            y_title = "y"
            fig_title = f"z = {float(slice["z"])}"
    
    elif len(slice) == 2:
        y_title = ""
        if "x" in slice:
            if "y" in slice:
                x_title = "z"
                fig_title = f"x = {float(slice["x"])}, y = {float(slice["y"])}"
            else:
                x_title = "y"
                fig_title = f"x = {float(slice["x"])}, z = {float(slice["z"])}"
        else:
            x_title = "x"
            fig_title = f"y = {float(slice["y"])}, z = {float(slice["z"])}"

    if len(slice) in (1, 2):
        layout["xaxis"]["title"] = dict(text=x_title, font=util.plotting.AXIS_TITLE_FONT)
        layout["yaxis"]["title"] = dict(text=y_title, font=util.plotting.AXIS_TITLE_FONT)
        layout["title"] = dict(
            text=fig_title if show_title else "",
            font=util.plotting.FIG_TITLE_FONT,
            xanchor="center",
            yanchor="top",
            x=0.5,
        )
    
    # Miscellaneous other layout settings
    layout["autosize"] = False
    layout["width"] = 500
    layout["height"] = 500
    layout["margin"] = dict(t=20, b=20, l=20, r=20)
    layout["showlegend"] = show_legend
    
    return layout

def snapshot_schematic_slice_trace(
    snapshot: hoomd.Snapshot,
    type_shapes: dict[str, coxeter.shapes.Polyhedron],
    slice: dict[str, int],
    scale: float = 1,
    color: str = "red",
    opacity: float = 1,
    line_width: float = 10,
) -> plotly.graph_objs._scatter3d.Scatter3d:
    """Return the plotly trace for a schematic slice through a snapshot.

    A slice with 1 key is represented as a plane, while a slice with 2 keys
    is represented as a line.
    
    Parameters
    ----------
    snapshot : hoomd.Snapshot
        The HOOMD snapshot. The positions of the snapshot's particles help
        determine the extents of the schematic slice.
    type_shapes : dict, optional
        A mapping from particle type name to shape. The vertices of the shapes
        help determine the extents of the schematic slice.
    slice : dict
        Axes and positions along which to slice. Keys are limited to 'x', 'y',
        and 'z'. There can be at most two keys.
    scale : float, default=1
        The scale of the schematic slice.
    color : str, default='red'
        The color of the schematic slice. Must satisfy plotly's color
        naming/formatting conventions.
    opacity : float, default=1
        The opacity of the schematic slice. Must be between 0 and 1.
    line_width : float, default=10
        The width of the schematic slice if it is a line. Ignored if the slice
        is a plane.
    """
    # Calculate extents
    extents = []
    for i in [0, 1, 2]:
        # Minimum
        min_index = np.argmin(snapshot.particles.position[:, i])
        p_at_i_min = snapshot.particles.position[min_index]
        o_at_i_min = snapshot.particles.orientation[min_index]
        t = snapshot.particles.types[snapshot.particles.typeid[min_index]]
        
        if t in type_shapes:
            vertices = type_shapes[t].vertices
            vertices = rowan.rotate(o_at_i_min, vertices)
            vertices += p_at_i_min        
            extents.append([min(p_at_i_min[i], vertices[:, i].min())])
        else:
            extents.append([p_at_i_min[i]])
        
        # Maximum
        max_index = np.argmax(snapshot.particles.position[:, i])
        p_at_i_max = snapshot.particles.position[max_index]
        o_at_i_max = snapshot.particles.orientation[max_index]
        t = snapshot.particles.types[snapshot.particles.typeid[max_index]]
        
        if t in type_shapes:
            vertices = type_shapes[t].vertices
            vertices = rowan.rotate(o_at_i_max, vertices)
            vertices += p_at_i_max        
            extents[i].append(max(p_at_i_max[i], vertices[:, i].max()))
        else:
            extents[i].append(p_at_i_max[i])

    xmin, xmax = extents[0]
    ymin, ymax = extents[1]
    zmin, zmax = extents[2]

    # A slice with 1 key is represented as a plane
    if len(slice) == 1:
        if "x" in slice:
            x = slice["x"]
            x = np.array([x, x, x, x])
            y = np.array([ymin, ymax, ymax, ymin]) * scale
            z = np.array([zmin, zmin, zmax, zmax]) * scale

        elif "y" in slice:
            y = slice["y"]
            x = np.array([xmin, xmax, xmax, xmin]) * scale
            y = np.array([y, y, y, y])
            z = np.array([zmin, zmin, zmax, zmax]) * scale

        elif "z" in slice:
            z = slice["z"]
            x = np.array([xmin, xmax, xmax, xmin]) * scale
            y = np.array([ymin, ymin, ymax, ymax]) * scale
            z = np.array([z, z, z, z])
        
        i = [0, 0]
        j = [1, 2]
        k = [2, 3]

        return plotly.graph_objects.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            name="slice",
            color=color,
            opacity=opacity,
            flatshading=True,
            showlegend=True
        )

    # A slice with 2 keys is represented as a line
    if len(slice) == 2:
        if "x" in slice:
            x = slice["x"]
            if "y" in slice:
                y = slice["y"]
                x = np.array([x, x])
                y = np.array([y, y])
                z = np.array([zmin, zmax]) * scale
            
            else:
                z = slice["z"]
                x = np.array([x, x])
                y = np.array([ymin, ymax]) * scale
                z = np.array([z, z])
        
        else:
            y = slice["y"]
            z = slice["z"]
            x = np.array([xmin, xmax]) * scale
            y = np.array([y, y])
            z = np.array([z, z])
        
        return plotly.graph_objects.Scatter3d(
            x=x,
            y=y,
            z=z,
            name="slice",
            opacity=opacity,
            line=dict(color=color, width=line_width),
            mode="lines",
            showlegend=True
        )

# ------------------------------------ OTHER -----------------------------------


# def default_params(hoomd_class: hoomd.md.pair.Pair):
#     pass