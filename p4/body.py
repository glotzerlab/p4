# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from __future__ import annotations
import coxeter
import hoomd
import numpy as np
import plotly
from copy import copy

import rowan
from .util import (
    WONG_COLORS,
    point_segment_distance,
    point_plane_distance,
    intersection_of_polygon_with_plane,
    intersection_of_polyhedron_with_plane,
    intersection_of_segment_with_plane
)

class Body:
    """The names and spatial data for a body's primary and secondary types.

    Every body must have a primary particle type, but secondary types are
    optional.
    
    When secondary types **are not** provided, the body represents a
    simple particle with a single type and no further information is needed.
    
    When secondary types **are** provided, the body represents a rigid body
    with a central particle (`primary_type`) and one or more constituent
    particles (`secondary_types`). In this case, the body needs a way to
    determine the position(s) of each type of constituent particle, and the user
    must provide a function that does so.

    .. code-block::
        :caption: A cubic body with primary particle 'A' at the center and
            secondary particles 'B' at the vertices.

        body = p4.Body(
            primary_type="A",
            secondary_types=["B"],
            positions_by_type=dict(
                B=[
                    [-1, -1, -1],
                    [-1, -1,  1],
                    [-1,  1, -1],
                    [-1,  1,  1],
                    [ 1, -1, -1],
                    [ 1, -1,  1],
                    [ 1,  1, -1],
                    [ 1,  1,  1]
                ]
            )
        )


    Parameters
    ----------
    primary_type : str
        The name of the primary type.
    secondary_types : list[str], optional
        The names of the secondary types.
    positions_by_type : dict[str, list[list[float]]], optional
        A mapping of secondary particle type names to position(s). Required if
        `secondary_types` is provided, otherwise ignored.
    orientations_by_type : dict[str, list[list[float]]], optional
        A mapping of secondary particle type names to orientation(s) in
        quaternion form. Can only be provided if `secondary_types` and
        `positions_by_type` are also provided.
    """
    def __init__(
        self,
        primary_type: str,
        secondary_types: list[str] = [],
        positions_by_type: dict[str, list[list[float]]] = {},
        orientations_by_type: dict[str, list[list[float]]] = {}
    ):
        if secondary_types != [] and not positions_by_type:
            raise ValueError(
                "'positions_by_type' is required if "
                + "'secondary_types' is provided"
            )

        self.primary_type = str(primary_type)
        self.secondary_types = [str(t) for t in secondary_types]
        self.positions_by_type = positions_by_type
        self.orientations_by_type = orientations_by_type

        if self.positions_by_type:
            self._validate_secondary_positions()
        if self.orientations_by_type:
            self._validate_secondary_orientations()
        if (
            self.positions_by_type
            and self.orientations_by_type
        ):
            self._validate_secondary_orientations_and_positions_match()
    
    def is_rigid(self, interactions: list["Interaction"]) -> bool:
        """Whether the body must represent a rigid body for some interactions."""
        common_single_types = any(
            t in self.secondary_types
            for interaction in interactions
            for t in interaction.interacting_types("single")
        )
        common_pair_types = any(
            p[0] in self.secondary_types or p[1] in self.secondary_types
            for interaction in interactions
            for p in interaction.interacting_types("pair")
        )
        nonzero_default_r_cut = any(
            (
                interaction.default_params["r_cut"] > 0
                or interaction.initial_args.get("default_r_cut", 0) > 0
            )
            for interaction in interactions
        )
        return common_single_types or common_pair_types or nonzero_default_r_cut

    def plot(
        self,
        type_shapes: dict[str, coxeter.shapes.Polyhedron] = {},
        type_styles: dict[str, dict] = {},
        ignore_types: list[str] = [],
        slice: dict[str, float] = {},
        schematic_slice: bool = False,
        schematic_slice_scale: float = 1,
        schematic_slice_color: str = "red",
        schematic_slice_opacity: float = 1,
        schematic_slice_line_width: float = 10
    ):
        """Interactively plot the body using plotly.

        Shapes and styles may be specified for specific types. A shape must be
        specified as a coxeter Polyhedron. A style must specified as a
        dictionary which may have the following keys and values:

        - **color**: A string representing the color for the symbol of the
        specified type. Must satisfy plotly's color name/formatting conventions.
        - **opacity**: A float between 0 and 1, where 0 represents fully
        transparent and 1 represents fully opaque.
        - **size**: A positive float representing the size of the symbol for the
        specified type. When the type is represented by a point, this setting
        corresponds to the point's ``size`` attribute. When the type is
        represented by a line, this setting corresponds to the line's ``width``
        attribute. When the type is represented by a polygon or Mesh/Polyhedron,
        this setting is ignored.
        
        Parameters
        ----------
        type_shapes : dict, default={}
            A dictionary mapping particle type names to
            coxeter.shapes.ConvexPolyhedron. If no shape is provided for a type,
            it will be plotted as a sphere.
        type_styles : dict, default={}
            A dictionary mapping particle type names to styles. A style is a
            dictionary which may have the following keys: 'color', 'opacity',
            and 'size'. See above for more information.
        ignore_types : list[str], default=[]
            The names of the particle types to exclude from the plot.
        slice : dict, default={}
            Axes and positions along which to slice. Keys are limited to 'x',
            'y', and 'z'. There can be at most two keys.
        schematic_slice : bool, default=False
            If True, the slice is shown schematically in a 3D view. A 2D slice
            appears like a plane intersecting with the body, while a 1D slice
            appears like a line intersecting with the body.
        schematic_slice_scale : float, default=1
            The scale of the schematic slice. Defaults to 1, which is to-scale.
            Increase this number if the schematic slice is fully contained by
            the body geometry and the true slice scale is not essential.
        schematic_slice_color : float, default='red'
            The color of the schematic slice. Must satisfy plotly's color
            name/formatting conventions.
        schematic_slice_opacity : float, default=1
            The opacity of the schematic slice.
        schematic_slice_line_width : float, default=10
            The width of the schematic slice if it is a line. Ignored if the
            slice is a plane.
        """
        default_colors = WONG_COLORS

        # Make a list of all types that will be plotted
        all_types = copy(self.secondary_types)
        all_types.append(self.primary_type)
        all_types = [t for t in all_types if t not in ignore_types]

        # Construct an array that stores each particle's type, position, and
        # orientation)
        particle_data = []
        for t in all_types:
            # only primary type should be at [0, 0, 0]
            for i, p in enumerate(self.positions_by_type.get(t, [[0, 0, 0]])):
                if t in self.orientations_by_type:
                    o = self.orientations_by_type[t][i]
                else:
                    o = [1, 0, 0, 0]

                particle_data.append([t,p[0],p[1],p[2],o[0],o[1],o[2],o[3]])
        
        particle_data = np.array(particle_data, dtype=object)

        # Construct the figure type-by-type
        figure = plotly.graph_objects.Figure()
        
        if len(slice) == 0 or schematic_slice:
            traces = self._plot_traces_3d(
                particle_data, type_shapes, type_styles, default_colors
            )

            if schematic_slice:
                traces.append(
                    self._plot_traces_schematic_slice(
                        particle_data,
                        slice,
                        schematic_slice_scale,
                        schematic_slice_color,
                        schematic_slice_opacity,
                        schematic_slice_line_width
                    )
                )
        
        elif len(slice) == 1 and not schematic_slice:
            traces = self._plot_traces_2d(
                particle_data, type_shapes, type_styles, slice, default_colors
            )
        
        elif len(slice) == 2 and not schematic_slice:
            traces = self._plot_traces_1d(
                particle_data, type_shapes, type_styles, slice, default_colors
            )

        for trace in traces:
            figure.add_trace(trace)

        figure.update_layout(showlegend=True)

        # Ensure that 2D plots have correct axis titles, figure title, and aspect ratio
        if len(slice) == 1:
            if "x" in slice:
                x_title = "y"
                y_title = "z"
                fig_title = f"x = {slice["x"]}"
            elif "y" in slice:
                x_title = "x"
                y_title = "z"
                fig_title = f"y = {slice["y"]}"
            elif "z" in slice:
                x_title = "x"
                y_title = "y"
                fig_title = f"z = {slice["z"]}"
            figure.update_layout(
                    xaxis=dict(
                        scaleanchor="y",
                        scaleratio=1,
                        constrain='domain',
                        title=dict(text=x_title, font=dict(weight=1000, size=16))
                    ),
                    yaxis=dict(
                        scaleanchor="x",
                        scaleratio=1,
                        constrain='domain',
                        title=dict(text=y_title, font=dict(weight=1000, size=16))
                    ),
                    # plot_bgcolor="rgba(0,0,0,0)"
                    title=dict(
                        text=fig_title,
                        font=dict(style="italic", size=16),
                        xanchor="center",
                        yanchor="top",
                        x=0.5
                    )
                )
        
        # Ensure that 1D plots have correct axis titles and caption
        if len(slice) == 2:
            if "x" in slice:
                if "y" in slice:
                    x_title = "z"
                    fig_title = f"x = {slice["x"]}, y = {slice["y"]}"
                else:
                    x_title = "y"
                    fig_title = f"x = {slice["x"]}, z = {slice["z"]}"
            else:
                x_title = "x"
                fig_title = f"y = {slice["y"]}, z = {slice["z"]}"

            figure.update_layout(
                    xaxis=dict(
                        title=dict(text=x_title, font=dict(weight=1000, size=16))
                    ),
                    # plot_bgcolor="rgba(0,0,0,0)"
                    title=dict(
                        text=fig_title,
                        font=dict(style="italic", size=16),
                        xanchor="center",
                        yanchor="top",
                        x=0.5
                    )
                )

        return figure, traces

    def _plot_traces_schematic_slice(
        self,
        particle_data: list,
        slice: dict[str, int],
        scale: float,
        color: str,
        opacity: float,
        line_width: float
    ) -> dict:
        """Return plotly plot traces for schematic slice in 3D.

        A slice with 1 key is represented as a plane, while a slice with 2 keys
        is represented as a line.
        
        Parameters
        ----------
        particle_data : np.array
            Type, position, and orientation data for all of the particles in the
            body. Must be formatted as a (N, 8) numpy array with the following
            columnsL type name, position x, position y, position z, q0, q1, q2,
            q3. This parameter is only needed to determine the extents of the
            schematic.
        slice : dict
            Axes and positions along which to slice. Keys are limited to 'x',
            'y', and 'z'. There can be at most two keys.
        scale : float
            The scale of the schematic slice.
        color : str
            The color of the schematic slice. Must satisfy plotly's color
            naming/formatting conventions.
        opacity : float
            The opacity of the schematic slice. Must be between 0 and 1.
        line_width : float
            The width of the schematic slice if it is a line. Ignored if the
            slice is a plane.
        
        Returns
        -------
        A dictionary representing the plotly trace.
        """
        positions = particle_data[:,1:4]
        xmin, xmax = positions[:,0].min(), positions[:,0].max()
        ymin, ymax = positions[:,1].min(), positions[:,1].max()
        zmin, zmax = positions[:,2].min(), positions[:,2].max()

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

    def _plot_traces_3d(
        self,
        particle_data: np.ndarray,
        type_shapes: dict[str, coxeter.shapes.Polyhedron],
        type_styles: dict[str, dict],
        default_colors: list[str],
    ) -> list[dict]:
        """Return plotly plot traces for body plotting in 3D.
        
        This function requires pre-calculated particle data. It should only be
        called from ``Body.plot``.

        Parameters
        ----------
        particle_data : np.array
            Type, position, and orientation data for all of the particles in the
            body. Must be formatted as a (N, 8) numpy array with the following
            columnsL type name, position x, position y, position z, q0, q1, q2,
            q3.
        type_shapes : dict
            A dictionary mapping particle type names to
            coxeter.shapes.ConvexPolyhedron. If no shape is provided for a type,
            it will be plotted as a sphere.
        type_styles : dict
            A dictionary mapping particle type names to styles. A style is a
            dictionary which may have the following keys: 'color', 'opacity',
            and 'size'. See ``Body.plot`` for more information.
        default_colors : list of strings
            The default list of colors to use if no color is specified for a
            type in ``type_styles``.

        Returns
        -------
        A list of dictionaries representing plotly traces.
        """

        # TODO: merge duplicate symbols

        traces = []
        all_particle_types = list(np.unique([row[0] for row in particle_data])[::-1])
        for t in all_particle_types:
            type_data = particle_data[particle_data[:,0] == t]

            # Get user-provided style information. Unless user specifies a type's
            # color, color markers by index from the default colors list.
            trace_style = type_styles.get(t, {})
            trace_color = trace_style.get("color", default_colors[all_particle_types.index(t)])
            trace_opacity = trace_style.get("opacity", 1.0)
            trace_size = trace_style.get("size")

            default_line_width = 10
            default_point_size = 10
            
            # Use Scatter3d when a shape is not specified
            if t not in type_shapes:
                traces.append(
                    plotly.graph_objects.Scatter3d(
                        name=t,
                        x=type_data[:,1],
                        y=type_data[:,2],
                        z=type_data[:,3],
                        customdata=type_data[:,4:],
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
                vertices = type_shapes[t].vertices
                faces = type_shapes[t].faces

                for row in type_data:
                    # Rotate shape vertices to the specified orientation
                    row_vertices = rowan.rotate(
                        q=np.repeat([row[4:]], len(vertices), axis=0),
                        v=np.array(copy(vertices))
                    )

                    # Translate shape vertices to the specified position
                    row_vertices += np.array(row[1:4]).astype(float)

                    # Get triangle indices
                    row_shape = coxeter.shapes.Polyhedron(row_vertices, faces)
                    
                    triangle_faces = np.empty((0,3), dtype=int)
                    for t_verts in row_shape._surface_triangulation():
                        face = np.array(
                            [
                                np.where(np.all(t_vert == row_vertices, axis=1))
                                for t_vert in t_verts
                            ]
                        ).flatten()
                        triangle_faces = np.vstack((triangle_faces, face))

                    # Construct trace
                    traces.append(
                        plotly.graph_objects.Mesh3d(
                            name=t,
                            x=row_vertices[:,0],
                            y=row_vertices[:,1],
                            z=row_vertices[:,2],
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

    def _plot_traces_2d(
        self,
        particle_data: np.ndarray,
        type_shapes: dict[str, coxeter.shapes.Polyhedron],
        type_styles: dict[str, dict],
        slice: dict[str, float],
        default_colors: list[str],
        point_size_for_slice: float = 1e-6
    ) -> list[dict]:
        """Return plotly plot traces for body plotting in 2D.
        
        This function requires pre-calculated particle data and assumes that
        ``slice`` has 1 key. It should only be called from ``Body.plot``.

        Parameters
        ----------
        particle_data : np.array
            Type, position, and orientation data for all of the particles in the
            body. Must be formatted as a (N, 8) numpy array with the following
            columnsL type name, position x, position y, position z, q0, q1, q2,
            q3.
        type_shapes : dict
            A dictionary mapping particle type names to
            coxeter.shapes.ConvexPolyhedron. If no shape is provided for a type,
            it will be plotted as a sphere.
        type_styles : dict
            A dictionary mapping particle type names to styles. A style is a
            dictionary which may have the following keys: 'color', 'opacity',
            and 'size'. See ``Body.plot`` for more information.
        slice : dict
            Axes and positions along which to slice. Keys are limited to 'x',
            'y', and 'z'. There can be at most two keys.
        default_colors : list of strings
            The default list of colors to use if no color is specified for a
            type in ``type_styles``.
        point_size_for_slice : float, default=1e-6
            The distance within which a point is considered to be contained by
            a plane.

        Returns
        -------
        A list of dictionaries representing plotly traces.
        """
        all_particle_types = list(np.unique([row[0] for row in particle_data])[::-1])

        # Calculate plane from slice
        slice_axis, slice_value = list(slice.items())[0]
        if slice_axis == "x":
            plane = [1, 0, 0, slice_value]
        elif slice_axis == "y":
            plane = [0, 1, 0, slice_value]
        elif slice_axis == "z":
            plane = [0, 0, 1, slice_value]

        # Construct the particle data for the slice. The row is included only if
        # points are returned for the slice
        slice_data = []
        for (t, px, py, pz, q0, q1, q2, q3) in particle_data:
            # Types without shapes must be within the distance tolerance to be
            # included
            if t not in type_shapes:
                if point_plane_distance([px, py, pz], plane) < point_size_for_slice:
                    slice_data.append((t, np.array([[px, py, pz]]), np.array([[q0, q1, q2, q3]]), np.array([[px, py, pz]]), "point"))

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
                slice_geometries = intersection_of_polyhedron_with_plane(
                    row_shape, plane
                )

                # Slice geometries are packaged in a new data structure that
                # formats data for easier passing to the plotly constructors.
                for geometry in slice_geometries:
                    geometry = np.array(geometry)
                    # Point
                    if geometry.shape[0] == 1:
                        slice_data.append([t, np.array([[px, py, pz]]), np.array([[q0, q1, q2, q3]]), geometry, "point"])
                    
                    # Single segment
                    elif geometry.shape[0] == 2:
                        slice_data.append([t, np.array([[px, py, pz]]), np.array([[q0, q1, q2, q3]]), geometry, "line"])
                    
                    # Polygon
                    elif geometry.shape[0] > 2:
                        slice_data.append([t, np.array([[px, py, pz]]), np.array([[q0, q1, q2, q3]]), geometry, "polygon"])


        # Merge identical symbols
        merged_slice_data = []
        merged_row_indices = []
        for i, current_row in enumerate(slice_data):
            rows_to_merge = []
            for j, other_row in enumerate(slice_data):
                if j != i and j not in merged_row_indices:
                    rows_are_equivalent = (
                        current_row[0] == other_row[0]
                        and current_row[4] == other_row[4]
                    )
                    if rows_are_equivalent: # TODO: check this doesn't break for lines
                        rows_to_merge.append(other_row)
                        merged_row_indices.append(j)
            
            if len(rows_to_merge) > 0:
                rows_to_merge = [current_row] + rows_to_merge
                merged_row_indices.append(i)

                m_t = rows_to_merge[0][0]
                m_p = np.vstack([row[1] for row in rows_to_merge])
                m_q = np.vstack([row[2] for row in rows_to_merge])
                m_geometry = np.vstack([    # note the need for breaking rows
                    np.vstack([row[3], np.array([[None, None, None]])])
                    for row in rows_to_merge
                ])
                m_trace_type = rows_to_merge[0][4]
                merged_slice_data.append((m_t, m_p, m_q, m_geometry, m_trace_type))
            
            elif i not in merged_row_indices:
                merged_slice_data.append(current_row)

        # Sort the merged slice data so that polygons are under lines and lines
        # are under points
        draw_order = ["polygon", "line", "point"]
        merged_slice_data = sorted(
            merged_slice_data,
            key=lambda row: draw_order.index(row[4])
        )

        # Build trace from slice data
        traces = []
        for (t, p, q, geometry, trace_type) in merged_slice_data:
            # For polygons, sometimes the circuit/loop is not actually closed. If
            # this is the case, close it.
            if trace_type == "polygon":
                if not np.array_equal(geometry[0, :], geometry[-1, :]):
                    geometry = np.vstack((geometry, np.atleast_2d(geometry[0, :])))

            # Transform geometry's point coordinates to match the plotting axis,
            # assuming that geometry is a 2D array with 3 columns and a row for each
            # point
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
            
            # Get user-provided style information. Unless user specifies a type's
            # color, color markers by index from the default colors list.
            trace_style = type_styles.get(t, {})
            trace_color = trace_style.get("color", default_colors[all_particle_types.index(t)])
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
                        name=t,
                        line=dict(
                            color=trace_color,
                            opacity=trace_opacity,
                            width=default_line_width if trace_size is None else trace_size
                        ),
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
                            size=default_point_size if trace_size is None else trace_size,
                            color=trace_color,
                            opacity=trace_opacity,
                            line=dict(width=2, color="DarkSlateGrey")
                        ),
                        customdata=np.hstack((p, q)),
                        hovertemplate=
                            "<b>r</b> (%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<br>" +
                            "<b>q</b> (%{customdata[3]}, %{customdata[4]}, %{customdata[5]}, %{customdata[6]})"                        
                    )
                )

        return traces

    def _plot_traces_1d(
        self,
        particle_data: np.ndarray,
        type_shapes: dict[str, coxeter.shapes.Polyhedron],
        type_styles: dict[str, dict],
        slice: dict[str, float],
        default_colors: list[str],
        point_size_for_slice: float = 1e-6
    ) -> list[dict]:
        """Return plotly plot traces for body plotting in 1D.
        
        This function requires pre-calculated particle data and assumes that
        ``slice`` has 2 keys. It should only be called from ``Body.plot``.

        Parameters
        ----------
        particle_data : np.array
            Type, position, and orientation data for all of the particles in the
            body. Must be formatted as a (N, 8) numpy array with the following
            columnsL type name, position x, position y, position z, q0, q1, q2,
            q3.
        type_shapes : dict
            A dictionary mapping particle type names to
            coxeter.shapes.ConvexPolyhedron. If no shape is provided for a type,
            it will be plotted as a sphere.
        type_styles : dict
            A dictionary mapping particle type names to styles. A style is a
            dictionary which may have the following keys: 'color', 'opacity',
            and 'size'. See ``Body.plot`` for more information.
        slice : dict
            Axes and positions along which to slice. Keys are limited to 'x',
            'y', and 'z'. There can be at most two keys.
        default_colors : list of strings
            The default list of colors to use if no color is specified for a
            type in ``type_styles``.
        point_size_for_slice : float, default=1e-6
            The distance within which a point is considered to be contained by
            a plane.

        Returns
        -------
        A list of dictionaries representing plotly traces.
        """
        all_particle_types = list(np.unique([row[0] for row in particle_data])[::-1])

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

        # Construct the particle data for the first slice. The row is included only
        # if points are returned for the slice
        slice_data = []
        for (t, px, py, pz, q0, q1, q2, q3) in particle_data:
            # Types without shapes must be within the distance tolerance to be
            # included
            if t not in type_shapes:
                if point_segment_distance([px, py, pz], line) < point_size_for_slice:
                    slice_data.append((t, np.array([[px, py, pz]]), np.array([[q0, q1, q2, q3]]), np.array([[px, py, pz]]), "point"))

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
                slice_geometries = intersection_of_polyhedron_with_plane(
                    row_shape, plane1
                )

                # Slice the current slice with the second plane
                slice_geometries2 = []
                for geometry in slice_geometries:
                    # Points must be within the distance tolerance
                    if np.array(geometry).shape[0] == 1:
                        px, py, pz = geometry
                        if point_segment_distance([px, py, pz], line) < point_size_for_slice:
                            slice_geometries2.append(geometry)
                    
                    # Segments must pass the intersection test
                    if np.array(geometry).shape[0] == 2:
                        segment_is_intersected = (
                            (
                                plane2[1] == 1
                                and slice["y"] >= np.array(geometry)[:,1].min()
                                and slice["y"] <= np.array(geometry)[:,1].max()
                            )
                            or
                            (
                                plane2[2] == 1
                                and slice["z"] >= np.array(geometry)[:,2].min()
                                and slice["z"] <= np.array(geometry)[:,2].max()
                            )
                        )
                        if segment_is_intersected:
                            slice_geometries2.append(
                                intersection_of_segment_with_plane(geometry, plane2)
                            )

                    # Polygons are sliced like in 2D plotting
                    if np.array(geometry).shape[0] > 2:
                        geometries = intersection_of_polygon_with_plane(geometry, plane2)
                        slice_geometries2.extend(geometries)

                # Slice geometries are packaged in a new data structure that
                # formats data for easier passing to the plotly constructors.
                for geometry in slice_geometries2:
                    geometry = np.array(geometry)
                    # Point
                    if geometry.shape[0] == 1:
                        slice_data.append([t, np.array([[px, py, pz]]), np.array([[q0, q1, q2, q3]]), geometry, "point"])
                    
                    # Single segment
                    elif geometry.shape[0] == 2:
                        slice_data.append([t, np.array([[px, py, pz]]), np.array([[q0, q1, q2, q3]]), geometry, "line"])
                    
                    # Polygons should not be possible
                    elif geometry.shape[0] > 2:
                        raise Exception("Uh oh! Found polygons when plotting in 1D. Check the code for the second slice...")

        # Merge identical symbols
        merged_slice_data = []
        merged_row_indices = []
        for i, current_row in enumerate(slice_data):
            rows_to_merge = []
            for j, other_row in enumerate(slice_data):
                if j != i and j not in merged_row_indices:
                    if current_row[0] == other_row[0] and current_row[4] == other_row[4]:
                        rows_to_merge.append(other_row)
                        merged_row_indices.append(j)
            
            if len(rows_to_merge) > 0:
                rows_to_merge = [current_row] + rows_to_merge
                merged_row_indices.append(i)

                m_t = rows_to_merge[0][0]
                m_p = np.vstack([row[1] for row in rows_to_merge])
                m_q = np.vstack([row[2] for row in rows_to_merge])
                m_geometry = np.vstack([np.vstack([row[3], np.array([[None, None, None]])]) for row in rows_to_merge])
                m_trace_type = rows_to_merge[0][4]
                merged_slice_data.append((m_t, m_p, m_q, m_geometry, m_trace_type))
            
            elif i not in merged_row_indices:
                merged_slice_data.append(current_row)

        # Sort the merged slice data so that polygons are under lines and lines
        # are under points
        draw_order = ["line", "point"]
        merged_slice_data = sorted(
            merged_slice_data,
            key=lambda row: draw_order.index(row[4])
        )

        # Build trace from slice data
        traces = []
        for (t, p, q, geometry, trace_type) in merged_slice_data:
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
            trace_color = trace_style.get("color", default_colors[all_particle_types.index(t)])
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
                        customdata=np.hstack((p, q)),
                        hovertemplate=
                            "<b>r</b> (%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<br>" +
                            "<b>q</b> (%{customdata[3]}, %{customdata[4]}, %{customdata[5]}, %{customdata[6]})"                        
                    )
                )
        
        return traces

    def _validate_secondary_positions(self):
        """Ensure that the provided callable works for all secondary types."""
        for t in self.secondary_types:
            if t not in self.positions_by_type.keys():
                raise ValueError(
                    "`positions_by_type` does not specify positions "
                    + f"for secondary type '{t}'."
                )

    def _validate_secondary_orientations(self):
        """Ensure that the provided callable works for all secondary types."""
        for t in self.secondary_types:
            if t not in self.orientations_by_type.keys():
                raise ValueError(
                    "`orientations_by_type` does not specify "
                    + f"orientations for secondary type '{t}'."
                )

    def _validate_secondary_orientations_and_positions_match(self):
        """Ensure secondary types' numbers of positions and orientations match."""
        for t in self.secondary_types:
            n_positions = len(self.positions_by_type[t])
            n_orientations = len(self.orientations_by_type[t])
            
            if n_positions != n_orientations:
                raise ValueError(
                    "The number of positions and orientations for secondary  "
                    + f"type {t} do not match."
                )

    @classmethod
    def from_hoomd_rigid(
        cls,
        rigid: hoomd.md.constrain.Rigid,
        primary_type: str | None = None
    ) -> list[Body] | Body:
        """Parse a `hoomd.md.constrain.Rigid <https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/constrain/rigid.html>`_ to create one or more :class:`~p4.Body`.
        
        Parameters
        ----------
        rigid : hoomd.md.constrain.Rigid
            The constraint that defines rigid bodies.
        primary_type : str, optional
            The name of the primary type of a single body. If provided, just
            that body is returned. If not provided, all possible bodies are
            returned in a list. If there is no body defined for the provided
            primary type, a single-particle body is returned.
        """
        # If primary type is supplied, it must be in the rigid's primary types
        if primary_type and primary_type not in rigid.body.keys():
            raise ValueError(
                f"`primary_type` ({primary_type}) not in rigid's primary types "
                f"({list(rigid.body.keys())})"
            )
        
        # Hoomd does not detect nested body definitions until sim.run(), so a
        # check is needed here
        for p_t in rigid.body.keys():
            for k, v in rigid.body.items():
                if v is not None:
                    if p_t in v["constituent_types"] and rigid.body[p_t] is not None:
                        raise ValueError("Nested bodies are not supported.")

        if primary_type:
            primary_types = [primary_type]
        else:
            primary_types = rigid.body.keys()

        def unique(strings):
            """Find unique values in a list of strings."""
            searched = []
            for s in strings:
                if s not in searched:
                    searched.append(s)
            return searched
        
        def data_by_type(types, data):
            """Return a mapping of unique types to their corresponding data."""
            d = {}
            for t in unique(types):
                d[t] = [x for i, x in enumerate(data) if types[i] == t]
            return d
        
        # Construct bodies
        bodies = []
        for p_t in primary_types:
            if rigid.body[p_t] is not None:
                bodies.append(cls(
                    primary_type=p_t,
                    secondary_types=unique(rigid.body[p_t]["constituent_types"]),
                    positions_by_type=data_by_type(
                        rigid.body[p_t]["constituent_types"],
                        [list(p) for p in rigid.body[p_t]["positions"]]
                    ),
                    orientations_by_type=data_by_type(
                        rigid.body[p_t]["constituent_types"],
                        [list(p) for p in rigid.body[p_t]["orientations"]]
                    )
                ))

        if len(bodies) == 0:
            return cls(primary_type=primary_type)
        if len(bodies) == 1:
            return bodies[0]
        else:
            return bodies

    @classmethod
    def from_hoomd_simulation(
        cls,
        simulation: hoomd.Simulation,
        primary_type: str | None = None
    ) -> list[Body] | Body:
        """Parse a `hoomd.Simulation <https://hoomd-blue.readthedocs.io/en/latest/hoomd/simulation.html>`_ to create one or more :class:`~p4.Body`.

        This is a convenience method that is equivalent to
        
        .. code-block::
            
            p4.Body.from_hoomd_rigid(sim.operations.integrator.rigid, primary_type)
        
        Parameters
        ----------
        simulation : hoomd.Simulation
            The simulation to parse.
        primary_type : str, optional
            The name of the primary type of a single body. If provided, just
            that body is returned. If not provided, all possible bodies are
            returned in a list.
        """
        if simulation.operations.integrator is None:
            raise ValueError("`simulation` must have an integrator")
        if simulation.operations.integrator.rigid is None:
            raise ValueError("integrator must have a rigid constraint")
        types_in_state = simulation.state.get_snapshot().particles.types
        if primary_type is not None and primary_type not in types_in_state:
            raise ValueError(
                f"`simulation` does not contain primary_type {primary_type}"
            )
        return cls.from_hoomd_rigid(
            simulation.operations.integrator.rigid, primary_type
        )

    def __eq__(self, other):
        """Bodies are equal if their attributes are the same or equivalent."""
        primary_same = self.primary_type == other.primary_type
        if not primary_same:
            return False
        
        secondary_same = self.secondary_types == other.secondary_types
        if not secondary_same:
            return False
        
        positions_same = self.positions_by_type == other.positions_by_type
        if not positions_same:
            return False

        orientations_same = (
            self.orientations_by_type == other.orientations_by_type
        )

        orientations_missing_from_self = (
            set(other.orientations_by_type) - set(self.orientations_by_type)
        )
        orientations_missing_from_other = (
            set(self.orientations_by_type) - set(other.orientations_by_type)
        )
        common_types = set(self.orientations_by_type).intersection(
            set(other.orientations_by_type)
        )
        orientations_equivalent = (
            all(
                self.orientations_by_type[t] == other.orientations_by_type[t]
                for t in common_types
            )
            and
            all(
                all(list(i) == [1,0,0,0] for i in other.orientations_by_type[t])
                for t in orientations_missing_from_self
            )
            and
            all(
                all(list(i) == [1,0,0,0] for i in self.orientations_by_type[t])
                for t in orientations_missing_from_other
            )
        )

        return (
            primary_same and secondary_same and positions_same
            and (orientations_same or orientations_equivalent)
        )
    
    def __repr__(self):
        return (
            "Body ("
            + f"\n\tprimary_type='{self.primary_type}',"
            + f"\n\tsecondary_types={self.secondary_types},"
            + f"\n\tpositions_by_type={self.positions_by_type},"
            + f"\n\torientations_by_type={self.orientations_by_type},"
            + "\n)"
        )
