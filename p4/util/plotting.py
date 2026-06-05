# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

"""Functions and constants for plotting data and formatting plots."""

from copy import copy

import coxeter
import hoomd
import numpy as np
import plotly
import rowan

from .polyhedron_intersection import (
    point_plane_distance,
    polyhedron_line_intersection,
    polyhedron_plane_intersection
)

AXIS_TITLE_FONT = dict(weight=1000, size=16)
FIG_TITLE_FONT = dict(style="italic", size=16)

def find_nearest(array, value):
    """Find the item nearest to a given value in an array."""
    # Ref: https://stackoverflow.com/a/2566508/15426433
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

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

    Returns
    -------
    dict     
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
        layout["xaxis"]["title"] = dict(text=x_title, font=AXIS_TITLE_FONT)
        layout["yaxis"]["title"] = dict(text=y_title, font=AXIS_TITLE_FONT)
        layout["title"] = dict(
            text=fig_title if show_title else "",
            font=FIG_TITLE_FONT,
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
):
    """Return plotly plot traces for schematic slice in 3D.

    A slice with 1 key is represented as a plane, while a slice with 2 keys
    is represented as a line.
    
    Parameters
    ----------
    snapshot : hoomd.Snapshot
        The HOOMD snapshot. The positions of snapshot's particles help determine
        the extents of the schematic slice.
    type_shapes : dict, optional
        A mapping from particle type name to shape. The vertices of the shapes
        help determine the extents of the schematic slice.
    slice : dict
        Axes and positions along which to slice. Keys are limited to 'x',
        'y', and 'z'. There can be at most two keys.
    scale : float, default=1
        The scale of the schematic slice.
    color : str, default='red'
        The color of the schematic slice. Must satisfy plotly's color
        naming/formatting conventions.
    opacity : float, default=1
        The opacity of the schematic slice. Must be between 0 and 1.
    line_width : float, default=10
        The width of the schematic slice if it is a line. Ignored if the
        slice is a plane.
    
    Returns
    -------
    A dictionary containing the plotly trace.
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

def snapshot_3D_traces(
    snapshot: hoomd.Snapshot,
    type_shapes: dict[str, coxeter.shapes.Polyhedron],
    type_styles: dict[str, dict],
    ignore_types: list[str],
    default_colors: list[str],
) -> list:
    """Return plotly plot traces representing a snapshot.

    Parameters
    ----------
    snapshot : hoomd.Snapshot
        The HOOMD snapshot to plot.
    type_shapes : dict
        A mapping from particle type names to coxeter Polyhedra. If no shape is
        provided for a type, it will be plotted as a sphere.
    type_styles : dict
        A dictionary mapping particle type names to styles. A style is a
        dictionary which may have the following keys: 'color', 'opacity',
        and 'size'. See ``Body.plot`` for more information.
    ignore_types : list[str]
        The names of the particle types to exclude from the plot.
    default_colors : list of strings
        The default list of colors to use if no color is specified for a
        type in ``type_styles``.

    Returns
    -------
    list
        The plotly trace objects.
    """
    traces = []
    
    for tid, t in enumerate(snapshot.particles.types):
        if t in ignore_types:
            continue

        indices = np.nonzero(snapshot.particles.typeid == tid)[0]
        type_data = np.column_stack((
            snapshot.particles.position[indices][:,0],
            snapshot.particles.position[indices][:,1],
            snapshot.particles.position[indices][:,2],
            snapshot.particles.orientation[indices][:,0],
            snapshot.particles.orientation[indices][:,1],
            snapshot.particles.orientation[indices][:,2],
            snapshot.particles.orientation[indices][:,3],
        ))

        # Get user-provided style information. Unless user specifies a type's
        # color, color markers by index from the default colors list.
        trace_style = type_styles.get(t, {})
        trace_color = trace_style.get("color", default_colors[tid])
        trace_opacity = trace_style.get("opacity", 1.0)
        trace_size = trace_style.get("size")

        default_line_width = 10
        default_point_size = 10

        # Use Scatter3d when a shape is not specified
        if t not in type_shapes:
            traces.append(
                plotly.graph_objects.Scatter3d(
                    name=t,
                    x=type_data[:,0],
                    y=type_data[:,1],
                    z=type_data[:,2],
                    customdata=type_data[:,3:],
                    mode="markers",
                    marker=dict(
                        size=default_point_size if trace_size is None else trace_size,
                        color=trace_color,
                        opacity=trace_opacity,
                        line=dict(width=2, color="DarkSlateGrey")
                    ),
                    hovertemplate=
                        "<b>r</b> (%{x:.0f}, %{y:.0f}, %{z:.0f})<br>" +
                        "<b>q</b> (%{customdata[0]}, %{customdata[1]}, " +
                        "%{customdata[2]}, %{customdata[3]})"
                )
            )
        
        # Use Mesh3d when a shape is specified
        else:
            shape_vertices = type_shapes[t].vertices
            shape_faces = type_shapes[t].faces

            vertices = np.empty((0,3), dtype=np.float32)
            triangle_faces = np.empty((0,3), dtype=np.int32)

            for row in type_data:
                # Transform shape vertices to the specified orientation/position
                row_vertices = rowan.rotate(row[3:], copy(shape_vertices)) 
                row_vertices += row[0:3]

                # Get triangle indices
                row_shape = coxeter.shapes.Polyhedron(row_vertices, shape_faces)
                
                row_triangle_faces = np.empty((0,3), dtype=int)
                for t_verts in row_shape._surface_triangulation():
                    face = np.array(
                        [
                            np.where(np.all(t_vert == row_vertices, axis=1))
                            for t_vert in t_verts
                        ]
                    ).flatten()
                    row_triangle_faces = np.vstack((row_triangle_faces, face))
                
                # Add the row's shape data to the arrays
                triangle_faces = np.vstack((
                    triangle_faces,
                    row_triangle_faces + vertices.shape[0]
                ))
                vertices = np.vstack((vertices, row_vertices))

            # Construct trace
            traces.append(
                plotly.graph_objects.Mesh3d(
                    name=t,
                    x=vertices[:,0],
                    y=vertices[:,1],
                    z=vertices[:,2],
                    color=trace_color,
                    i=triangle_faces[:,0],
                    j=triangle_faces[:,1],
                    k=triangle_faces[:,2],
                    flatshading=True,
                    showlegend=True,
                    opacity=trace_opacity
                )
            )

    return traces

def snapshot_2D_traces(
    snapshot: hoomd.Snapshot,
    slice: dict[str, float],
    type_shapes: dict[str, coxeter.shapes.Polyhedron],
    type_styles: dict[str, dict],
    ignore_types: list[str],
    default_colors: list[str],
    point_size_for_slice: float = 1e-6
) -> list:
    """Return plotly plot traces representing a snapshot sliced by one plane.

    Parameters
    ----------
    snapshot : hoomd.Snapshot
        The HOOMD snapshot to plot.
    slice : dict
        Axes and positions along which to slice. Keys are limited to 'x',
        'y', and 'z'. There must be exactly one key.
    type_shapes : dict
        A mapping from particle type names to coxeter Polyhedra. If no shape is
        provided for a type, it will be plotted as a sphere.
    type_styles : dict
        A dictionary mapping particle type names to styles. A style is a
        dictionary which may have the following keys: 'color', 'opacity',
        and 'size'. See ``Body.plot`` for more information.
    ignore_types : list[str]
        The names of the particle types to exclude from the plot.
    default_colors : list of strings
        The default list of colors to use if no color is specified for a
        type in ``type_styles``.
    point_size_for_slice : float, default=1e-6
        The distance within which a point is considered to be contained by
        a plane.

    Returns
    -------
    list
        The plotly trace objects.
    """
    # Calculate plane from slice
    slice_axis, slice_value = list(slice.items())[0]
    if slice_axis == "x":
        plane = [1, 0, 0, slice_value]
    elif slice_axis == "y":
        plane = [0, 1, 0, slice_value]
    elif slice_axis == "z":
        plane = [0, 0, 1, slice_value]

    # Construct a separate trace for each type
    traces = []
    for tid, t in enumerate(snapshot.particles.types):
        if t in ignore_types:
            continue

        indices = np.nonzero(snapshot.particles.typeid == tid)[0]
        type_data = np.column_stack((
            snapshot.particles.position[indices][:,0],
            snapshot.particles.position[indices][:,1],
            snapshot.particles.position[indices][:,2],
            snapshot.particles.orientation[indices][:,0],
            snapshot.particles.orientation[indices][:,1],
            snapshot.particles.orientation[indices][:,2],
            snapshot.particles.orientation[indices][:,3],
        ))

        type_slice_data = []
        for (px, py, pz, q0, q1, q2, q3) in type_data:
            # Types without shapes must be within the distance tolerance to be
            # included
            if t not in type_shapes:
                if (
                    point_plane_distance([px, py, pz], plane)
                    < point_size_for_slice
                ):
                    type_slice_data.append([
                        np.array([[px, py, pz]]),
                        np.array([[q0, q1, q2, q3]]),
                        np.array([[px, py, pz]]),
                        "point"
                    ])

            # Types with shapes must be sliced
            else:
                vertices = type_shapes[t].vertices
                faces = type_shapes[t].faces

                # Transform shape vertices to the specified orientation/position
                row_vertices = rowan.rotate([q0,q1,q2,q3], copy(vertices)) 
                row_vertices += [px, py, pz]

                # Create shape and slice it
                row_shape = coxeter.shapes.Polyhedron(row_vertices, faces)            
                slice_geometries = polyhedron_plane_intersection(
                    row_shape, plane
                )

                # Slice geometries are packaged in a new data structure that
                # formats data for easier passing to the plotly constructors.
                for geometry in slice_geometries:
                    geometry = np.array(geometry)
                    # Point
                    if geometry.shape[0] == 1:
                        type_slice_data.append([
                            np.array([[px, py, pz]]),
                            np.array([[q0, q1, q2, q3]]),
                            geometry,
                            "point"
                        ])
                    
                    # Single segment
                    elif geometry.shape[0] == 2:
                        type_slice_data.append([
                            np.array([[px, py, pz]]),
                            np.array([[q0, q1, q2, q3]]),
                            geometry,
                            "line"
                        ])
                    
                    # Polygon
                    elif geometry.shape[0] > 2:
                        type_slice_data.append([
                            np.array([[px, py, pz]]),
                            np.array([[q0, q1, q2, q3]]),
                            geometry,
                            "polygon"
                        ])

        # Merge identical symbols
        merged_type_slice_data = []
        merged_row_indices = []
        for i, current_row in enumerate(type_slice_data):
            rows_to_merge = []
            for j, other_row in enumerate(type_slice_data):
                if j != i and j not in merged_row_indices:
                    rows_are_equivalent = (
                        np.array_equal(current_row[0], other_row[0])
                        # and current_row[3] == other_row[3]
                    )
                    if rows_are_equivalent: # TODO: check this doesn't break for lines
                        rows_to_merge.append(other_row)
                        merged_row_indices.append(j)
            
            if len(rows_to_merge) > 0:
                rows_to_merge = [current_row] + rows_to_merge
                merged_row_indices.append(i)

                m_p = np.vstack([row[0] for row in rows_to_merge])
                m_q = np.vstack([row[1] for row in rows_to_merge])
                m_geometry = np.vstack([    # note the need for breaking rows
                    np.vstack([row[2], np.array([[None, None, None]])])
                    for row in rows_to_merge
                ])
                m_trace_type = rows_to_merge[0][3]
                merged_type_slice_data.append([
                    m_p,
                    m_q,
                    m_geometry,
                    m_trace_type
                ])
            
            elif i not in merged_row_indices:
                merged_type_slice_data.append(current_row)

        # Sort the merged slice data so that polygons are under lines and lines
        # are under points
        draw_order = ["polygon", "line", "point"]
        merged_type_slice_data = sorted(
            merged_type_slice_data,
            key=lambda row: draw_order.index(row[3])
        )

        # Build trace from slice data
        for (p, q, geometry, trace_type) in merged_type_slice_data:
            # For polygons, sometimes the circuit/loop is not actually closed.
            # If this is the case, close it.
            if trace_type == "polygon":
                if not np.array_equal(geometry[0, :], geometry[-1, :]):
                    geometry = np.vstack((
                        geometry,
                        np.atleast_2d(geometry[0, :])
                    ))

            # Transform geometry's point coordinates to match the plotting axis,
            # assuming that geometry is a 2D array with 3 columns and a row for
            # each point
            x = geometry[:,0]
            y = geometry[:,1]
            z = geometry[:,2]

            if slice_axis == "x":
                trace_x = y
                trace_y = z

            elif slice_axis == "y":
                trace_x = x
                trace_y = z

            elif slice_axis == "z":
                trace_x = x
                trace_y = y
            
            # Get user-provided style information. Unless user specifies a
            # type's color, color markers by index from the default colors list.
            trace_style = type_styles.get(t, {})
            trace_color = trace_style.get("color", default_colors[tid])
            trace_opacity = trace_style.get("opacity", 1.0)
            trace_size = trace_style.get("size")

            default_line_width = 10
            default_point_size = 10
            
            # Construct traces by symbol type     
            if trace_type == "polygon":
                traces.append(
                    plotly.graph_objects.Scatter(
                        x=trace_x,
                        y=trace_y,
                        name=t,
                        mode="lines",
                        fill="toself",
                        fillcolor=trace_color,
                        opacity=trace_opacity,
                        # hoveron="points+fills",
                        line=dict(
                            color=trace_color,
                            width=2
                        ),
                        cliponaxis=False,
                    #     hovertemplate=
                    #         f"<b>r</b> ({p[0,0]}, {p[0,1]}, {p[0,2]})<br>" +        # TODO: this text does not show up
                    #         f"<b>q</b> ({q[0,0]}, {q[0,1]}, {q[0,2]}, {q[0,3]})"                        
                    )
                )

            elif trace_type == "line":  # TODO: check that this works
                traces.append(
                    plotly.graph_objects.Scatter(
                        x=trace_x,
                        y=trace_y,
                        mode="lines",
                        opacity=trace_opacity,
                        name=t,
                        line=dict(
                            color=trace_color,
                            width=(
                                default_line_width if trace_size is None
                                else trace_size
                            )
                        ),
                        cliponaxis=False,
                    #     hovertemplate=
                    #         f"<b>r</b> ({p[0,0]}, {p[0,1]}, {p[0,2]})<br>" +        # TODO: this text does not show up
                    #         f"<b>q</b> ({q[0,0]}, {q[0,1]}, {q[0,2]}, {q[0,3]})"                        
                    )
                )

            elif trace_type == "point":
                traces.append(
                    plotly.graph_objects.Scatter(
                        x=trace_x,
                        y=trace_y,
                        mode="markers",
                        name=t,
                        marker=dict(
                            size=(
                                default_point_size if trace_size is None
                                else trace_size
                            ),
                            color=trace_color,
                            opacity=trace_opacity,
                            line=dict(width=2, color="DarkSlateGrey")
                        ),
                        cliponaxis=False,
                        customdata=np.hstack((p, q)),
                        hovertemplate=
                            "<b>r</b> (%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<br>" +
                            "<b>q</b> (%{customdata[3]}, %{customdata[4]}, %{customdata[5]}, %{customdata[6]})"                        
                    )
                )

    return traces

def snapshot_1D_traces(
    snapshot: hoomd.Snapshot,
    slice: dict[str, float],
    type_shapes: dict[str, coxeter.shapes.Polyhedron],
    type_styles: dict[str, dict],
    ignore_types: list[str],
    default_colors: list[str],
    point_size_for_slice: float = 1e-6
) -> list:
    """Return plotly plot traces representing a snapshot sliced by two planes.

    Parameters
    ----------
    snapshot : hoomd.Snapshot
        The HOOMD snapshot to plot.
    slice : dict
        Axes and positions along which to slice. Keys are limited to 'x',
        'y', and 'z'. There must be exactly two keys.
    type_shapes : dict
        A mapping from particle type names to coxeter Polyhedra. If no shape is
        provided for a type, it will be plotted as a sphere.
    type_styles : dict
        A dictionary mapping particle type names to styles. A style is a
        dictionary which may have the following keys: 'color', 'opacity',
        and 'size'. See ``Body.plot`` for more information.
    default_colors : list of strings
        The default list of colors to use if no color is specified for a
        type in ``type_styles``.
    point_size_for_slice : float, default=1e-6
        The distance within which a point is considered to be contained by
        a plane.

    Returns
    -------
    list
        The plotly trace objects.
    """
    # Calculate line from slice (assume that slice has exactly 2 keys)
    slice_x = slice.get("x")
    slice_y = slice.get("y")
    slice_z = slice.get("z")
    if slice_x is not None:
        plane1 = [1, 0, 0, slice_x]
        if slice_y is not None:
            plane2 = [0, 1, 0, slice_y]
            # line is defined by 2 end points
            line = [[slice_x, slice_y, -1e9], [slice_x, slice_y, 1e9]]  # Review: there must be a better way...
        else:
            plane2 = [0, 0, 1, slice_z]
            line = [[slice_x, -1e9, slice_z], [slice_x, 1e9, slice_z]]
    else:
        plane1 = [0, 1, 0, slice_y]
        plane2 = [0, 0, 1, slice_z]
        line = [[-1e9, slice_y, slice_z], [1e9, slice_y, slice_z]]

    # Construct a separate trace for each type
    traces = []
    for tid, t in enumerate(snapshot.particles.types):
        if t in ignore_types:
            continue
    
        indices = np.nonzero(snapshot.particles.typeid == tid)[0]
        type_data = np.column_stack((
            snapshot.particles.position[indices][:,0],
            snapshot.particles.position[indices][:,1],
            snapshot.particles.position[indices][:,2],
            snapshot.particles.orientation[indices][:,0],
            snapshot.particles.orientation[indices][:,1],
            snapshot.particles.orientation[indices][:,2],
            snapshot.particles.orientation[indices][:,3],
        ))

        type_slice_data = []
        for (px, py, pz, q0, q1, q2, q3) in type_data:
            # Types without shapes must be within the distance tolerance to be
            # included
            if t not in type_shapes:
                if (
                    point_plane_distance([px, py, pz], plane1)
                    < point_size_for_slice
                ):
                    type_slice_data.append([
                        np.array([[px, py, pz]]),
                        np.array([[q0, q1, q2, q3]]),
                        np.array([[px, py, pz]]),
                        "point"
                    ])

            # Types with shapes must be sliced
            else:
                vertices = type_shapes[t].vertices
                faces = type_shapes[t].faces

                # Rotate shape vertices to the specified orientation
                row_vertices = rowan.rotate(
                    q=np.repeat([[q0, q1, q2, q3]], len(vertices), axis=0),
                    v=np.array(copy(vertices))
                )

                # Translate shape vertices to the specified position
                row_vertices += np.array([px, py, pz]).astype(float)

                # Create shape and slice it
                row_shape = coxeter.shapes.Polyhedron(row_vertices, faces)            
                slice_geometries = polyhedron_line_intersection(
                    row_shape, [plane1, plane2]
                )

                # Slice geometries are packaged in a new data structure that
                # formats data for easier passing to the plotly constructors.
                for geometry in slice_geometries:
                    geometry = np.array(geometry)
                    # Point
                    if geometry.shape[0] == 1:
                        type_slice_data.append(
                            [
                                t,
                                np.array([[px, py, pz]]),
                                np.array([[q0, q1, q2, q3]]),
                                geometry,
                                "point"
                            ]
                        )
                    
                    # Single segment
                    elif geometry.shape[0] == 2:
                        type_slice_data.append(
                            [
                                t,
                                np.array([[px, py, pz]]),
                                np.array([[q0, q1, q2, q3]]),
                                geometry,
                                "line"
                            ]
                        )
                    
                    # Polygons should not be possible
                    elif geometry.shape[0] > 2:
                        raise Exception(
                            "Uh oh! Found polygons when plotting in 1D. Check the "
                            + "code for the second slice..."
                        )

    # Merge identical symbols
    merged_type_slice_data = []
    merged_row_indices = []
    for i, current_row in enumerate(type_slice_data):
        rows_to_merge = []
        for j, other_row in enumerate(type_slice_data):
            if j != i and j not in merged_row_indices:
                rows_are_equivalent = ( # TODO: check that this is sufficient
                    np.array_equal(current_row[0], other_row[0])
                    # and current_row[3] == other_row[3]
                )
                if rows_are_equivalent:
                    rows_to_merge.append(other_row)
                    merged_row_indices.append(j)
        
        if len(rows_to_merge) > 0:
            rows_to_merge = [current_row] + rows_to_merge
            merged_row_indices.append(i)

            m_t = rows_to_merge[0][0]
            m_p = np.vstack([row[1] for row in rows_to_merge])
            m_q = np.vstack([row[2] for row in rows_to_merge])
            m_geometry = np.vstack([
                np.vstack([row[3], np.array([[None, None, None]])])
                for row in rows_to_merge
            ])
            m_trace_type = rows_to_merge[0][4]
            merged_type_slice_data.append((m_t, m_p, m_q, m_geometry, m_trace_type))
        
        elif i not in merged_row_indices:
            merged_type_slice_data.append(current_row)

    # Sort the merged slice data so that polygons are under lines and lines
    # are under points
    draw_order = ["line", "point"]
    merged_type_slice_data = sorted(
        merged_type_slice_data,
        key=lambda row: draw_order.index(row[3])
    )

    # Build trace from slice data
    traces = []
    for (p, q, geometry, trace_type) in merged_type_slice_data:
        # Transform geometry's point coordinates to match the plotting axis,
        # assuming that geometry is a 2D array with 3 columns and a row for each
        # point
        x = geometry[:,0]
        y = geometry[:,1]
        z = geometry[:,2]

        if slice_x is not None:
            if slice_y is not None:
                trace_x = z
            else:
                trace_x = y
        else:
            trace_x = x
        
        # Get user-provided style information. Unless user specifies a type's
        # style, color markers by index from the default colors list.
        trace_style = type_styles.get(t, {})
        trace_color = trace_style.get(
            "color",
            default_colors[snapshot.particles.types.index(t)]
        )
        trace_opacity = trace_style.get("opacity", 1.0)
        trace_size = trace_style.get("size")

        default_line_width = 10
        default_point_size = 10

        # Construct traces by symbol type     
        if trace_type == "line":
            traces.append(
                plotly.graph_objects.Scatter(
                    x=trace_x,
                    y=[0 for _ in trace_x],
                    mode="lines",
                    name=t,
                    opacity=trace_opacity,
                    line=dict(
                        color=trace_color,
                        width=default_line_width if trace_size is None else trace_size
                    ),
                    cliponaxis=False
                #     hovertemplate=
                #         f"<b>r</b> ({p[0,0]}, {p[0,1]}, {p[0,2]})<br>" +        # TODO: this text does not show up
                #         f"<b>q</b> ({q[0,0]}, {q[0,1]}, {q[0,2]}, {q[0,3]})"                        
                )
            )

        if trace_type == "point":
            traces.append(
                plotly.graph_objects.Scatter(
                    x=trace_x,
                    y=[0 for _ in trace_x],
                    mode="markers",
                    name=t,
                    marker=dict(
                        size=default_point_size if trace_size is None else trace_size,
                        color=trace_color,
                        opacity=trace_opacity,
                        line=dict(width=2, color="DarkSlateGrey")
                    ),
                    cliponaxis=False,
                    customdata=np.hstack((p, q)),
                    hovertemplate=
                        "<b>r</b> (%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<br>" +
                        "<b>q</b> (%{customdata[3]}, %{customdata[4]}, %{customdata[5]}, %{customdata[6]})"
                )
            )
    
    return traces
