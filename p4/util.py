# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

"""Utility functions for p4.

In general, these functions have no defaults and do not protect themselves
from wrong/fallible inputs.
"""

import csv
from io import StringIO, TextIOWrapper
import itertools
from typing import Tuple
import coxeter
import gsd.hoomd
import hoomd
import numpy as np
import rowan
from collections import defaultdict


# https://davidmathlogic.com/colorblind (original source IBM Design Library)
IBM_COLORS = [
    "#5b8efd",
    "#725def",
    "#dd217d",
    "#ff5f00",
    "#ffb00d",
]

# https://davidmathlogic.com/colorblind (original source https://doi.org/10.1038/nmeth.1618)
WONG_COLORS = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]

# https://davidmathlogic.com/colorblind (original source Paul Tol)
TOL_COLORS = [
    "#332288",
    "#117733",
    "#44AA99",
    "#88CCEE",
    "#DDCC77",
    "#CC6677",
    "#AA4499",
    "#882255",
]


# ---------------------------------- GEOMETRY ----------------------------------


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

def angle_between_vectors(a, b):
    """Return the smallest angle between vectors a and b in radians."""
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def point_plane_distance(point, plane):
    """The distance of a point from a plane.
    Ref: https://mathinsight.org/distance_point_plane
    """
    x, y, z, = point
    a, b, c, d = plane
    return np.abs(a*x + b*y + c*z - d)/(np.sqrt((a**2 + b**2 + c**2)))

def point_line_distance(point, line):
    """Return the distance of a point from a line.
    Ref: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    """
    x0 = point
    x1, x2 = line
    x0, x1, x2 = np.array(x0), np.array(x1), np.array(x2)
    d = np.linalg.norm(np.cross(x0 - x1, x0 - x2)) / np.linalg.norm(x2 - x1)
    return d

def intersection_of_segment_with_plane(segment, plane):
    """Return the intersection of a line segment with a plane in 3D.
    
    The segment is defined by a pair of points with coordinates in X, Y, Z order.

    The plane is provided as an array of floats [a, b, c, d], which correspond
    to the equation of a plane

    a*x + b*y + c*z = d

    This function only works if the plane has a normal with one non-zero
    component.
    """
    p1, p2 = segment
    if plane[0] == 1:
        f = ((plane[3] - p1[0])/(p2[0] - p1[0]))
        px = plane[3]
        py = f * (p2[1] - p1[1]) + p1[1]
        pz = f * (p2[2] - p1[2]) + p1[2]

    elif plane[1] == 1:
        f = ((plane[3] - p1[1])/(p2[1] - p1[1]))
        py = plane[3]
        px = f * (p2[0] - p1[0]) + p1[0]
        pz = f * (p2[2] - p1[2]) + p1[2]

    elif plane[2] == 1:
        f = ((plane[3] - p1[2])/(p2[2] - p1[2]))
        pz = plane[3]
        px = f * (p2[0] - p1[0]) + p1[0]
        py = f * (p2[1] - p1[1]) + p1[1]

    return [px, py, pz]

def intersection_of_polygon_with_plane(polygon, plane):
    """Return the intersection of a polygon with a plane in 3D.
    
    The polygon is provided as an ordered list of consecutive (i.e., connected)
    points. Each point has coordinates in X, Y, Z order. It is assumed that
        
        - the points are co-planar with each other
        - the points are in CCW order

    The plane is provided as an array of floats [a, b, c, d], which correspond
    to the equation of a plane

    a*x + b*y + c*z = d

    This function only works if the plane has a normal with one non-zero
    component.

    The intersection is returned as a ...
    
    TODO: can't be a (N, 3) numpy array, where N is the number of intersection
    points. Have to return *segments*, not points.

    """
    polygon = np.array(polygon)

    # If all of the polygon's vertices are co-planar, return the polygon.
    # If some of the vertices are co-planar, still proceed with the line segment
    # intersection checks.
    if len(np.array(polygon).shape) == 1: # TODO: fix this
        breakpoint()

    coplanar_points = [p for p in polygon if point_plane_distance(p, plane) < 1e-6]
    if coplanar_points:
        if len(coplanar_points) == polygon.shape[0]:
            return [polygon]    # note it must be in an array

    # Check for intersections between line segments and the plane. There can
    # only be intersections if the points on either end of a line segment
    # straddle the plane.
    def points_straddle_plane(point1, point2, plane):
        """Return True if the points are on opposite sides of the plane."""
        px, py, pz = point1
        qx, qy, qz = point2
        a, b, c, d = plane
        return (
            (
                (px*a + py*b + pz*c < d)
                and (qx*a + qy*b + qz*c > d)
            )
            or (
                (px*a + py*b + pz*c > d)
                and (qx*a + qy*b + qz*c < d)
            )
        )

    intersected_segments = []
    for i in range(polygon.shape[0]):
        p1 = polygon[np.mod(i, polygon.shape[0])]
        p2 = polygon[np.mod(i+1, polygon.shape[0])]
        if points_straddle_plane(p1, p2, plane):
            intersected_segments.append([p1, p2])
    
    # Determine the intersection point of the plane along each of the
    # intersected line segments. This is the part that only works for planes
    # with one non-zero normal component.
    if intersected_segments: # TODO: return here to fix tangent points problem: add `or coplanar_points`
        intersection_points = []
        for segment in intersected_segments:
            intersection_point = intersection_of_segment_with_plane(segment, plane)
            intersection_points.append(intersection_point)

        # # If the polygon is convex, there will only be 2 intersection points.
        # # In this case, simply return them. TODO: what if there are 2 intersection points for a concave shape...
        # if len(intersection_points) == 2:
        #     return np.vstack([[intersection_points[0]], [intersection_points[1]]])
        
        # If the polygon is concave, there can be more than 2 intersection
        # points. We have to determine which points connect together to form
        # line segments that are inside the polygon.
        potential_intersection_segments = [
            [intersection_points[i], intersection_points[i+1]]
            for i in range(len(intersection_points)-1)
        ]
        if potential_intersection_segments: # may not exist yet if only coplanar points
            potential_intersection_segments.append( # make sure we close the loop
                [intersection_points[-1], intersection_points[0]]
            )

        # If there are no intersected segments, we still have to check
        # the coplanar points
        if not intersected_segments:
            intersection_points = coplanar_points

        # If there are previously-found coplanar points, we have to check them
        # just like we check the segments. Treat them as intersecting the line
        # segments that *start* at them. TODO: return here to fix tangent points problem
        # if coplanar_points:
        #     if not intersected_segments:
        #         for cp in coplanar_points:
        #             for i in range(polygon.shape[0]):
        #                 p1 = polygon[np.mod(i, polygon.shape[0])]
        #                 p2 = polygon[np.mod(i+1, polygon.shape[0])]
        #                 if np.array_equal(cp, p1):
        #                     intersected_segments.append([p1, p2])
            
        #     potential_intersection_segments.append( # make sure we close the loop
        #         [intersection_points[-1], intersection_points[0]]
        #     )

        #     for i in range(len(coplanar_points)-1):
        #         potential_intersection_segments.append(
        #             [coplanar_points[i], coplanar_points[i+1]]
        #         )

        # If there are only 2 intersection points, either the points define a
        # line segment (always the case for convex polygons), or the points
        # define two tangent point intersections. 

        # For every potential intersection segment, check if it is inside
        # the polygon. This check consists of the following steps:
        #   1) let a be a vector representing a potential intersection segment
        #   2) let b be a vector from a[0] to the end of the polygon
        #      segment that that point lies on
        #   3) Take the cross product b x a = c
        #   4) Compare the cross product c to the normal vector n for the
        #      polygon. If the angle between c and n is approximately 0,
        #      then the potential intersection segment is inside the polygon
        #      and counts as a legitimate intersection.
        normal = np.cross(
            polygon[1] - polygon[0],
            polygon[2] - polygon[1]
        )

        intersection_segments = []
        indices_of_intersection_segment_points = []
        for i in range(len(intersection_points)):
            a = np.array(potential_intersection_segments[i][1]) - np.array(potential_intersection_segments[i][0])
            b = np.array(intersected_segments[i][1]) - np.array(potential_intersection_segments[i][0])
            c = np.cross(b, a)
            if angle_between_vectors(c, normal) < np.pi/2: # TODO: add `or np.array_equal(a, b)` to fix warning
                # Check also if any other intersection point is on this
                # potential segment. If so, reject the segment. This could be
                # optimized away if the intersection points were ordered such
                # that wrapping problems were preventable.
                def point_in_segment(point, segment):
                    # Ref: https://stackoverflow.com/a/328122/15426433
                    a = np.array(segment[0])
                    b = np.array(segment[1])
                    c = np.array(point)
                    return (
                        (not np.array_equal(c, a) and not np.array_equal(c, b)) # c is not endpoint
                        and all(np.cross(b - a, c - a) == 0)                    # c is aligned with line segment
                        and (                                                   # c is between a and b
                            np.dot(b - a, c - a) > 0
                            and np.dot(b - a, c - a) < np.linalg.norm(b - a)**2
                        )
                    )
                if not any(
                    point_in_segment(p, potential_intersection_segments[i])
                    for p in intersection_points
                ):
                    intersection_segments.append(potential_intersection_segments[i])
                    indices_of_intersection_segment_points.extend([i, np.mod(i+1, len(intersection_points))])
        
        # Any intersection points that are not included in the points that make
        # up the intersection segments must be tangent point intersections
        tangent_intersection_points = []
        for i in range(len(intersection_points)):
            if i not in indices_of_intersection_segment_points:
                tangent_intersection_points.append(intersection_points[i])

        return intersection_segments + tangent_intersection_points

    else:
        return []

def joinable_segments_to_polygon(segments):
    """Take a list of pairs of points and return a list of single points that
    define a non-self-intersecting polygon.
    """
    # Start by converting all segments into native python data structures
    segments = [[[float(i) for i in p] for p in s] for s in segments]
    processed_indices = []
    points = []

    points.extend(segments[0])
    processed_indices.append(0)
    
    n_points = len(segments)
    for i in range(1, n_points-1):
        endpoint = points[i]
        other_segments_and_indices = [
            (j, s) for j, s in enumerate(segments) if j not in processed_indices
        ]
        for j, segment in other_segments_and_indices:
            if j not in processed_indices:
                p1, p2 = segment
                if p1 == endpoint:
                    points.append(p2)
                    processed_indices.append(j)
                    break
                elif p2 == endpoint:
                    points.append(p1)
                    processed_indices.append(j)
                    break
                else:
                    continue

    return np.array(points)

def geometry_contains_point(geometry, point):
    """geometry and point are 2D arrays of row-wise vertices."""
    return (geometry == point).all(axis=1).any()

def geometry_contains_segment(geometry, segment):
    """geometry and segment are 2D arrays of row-wise vertices."""
    # This function was written with assistance from GPT-4.1
    pairs = np.stack([geometry[:-1], geometry[1:]], axis=1)
    wrap_pair = np.stack(
        [geometry[-1], geometry[0]],
        axis=0
    ).reshape(1, 2, np.array(geometry).shape[1])
    geometry_segments = np.concatenate([pairs, wrap_pair], axis=0)
    return (
        np.any(np.all(geometry_segments == segment, axis=(1,2)))
        or
        np.any(np.all(geometry_segments == segment[::-1], axis=(1,2)))
    )

def find_loop(start, adj, visited):
    """Find and return a loop of connected points that are not already visited.

    Requires an adjacency dictionary whose keys are points and whose values are
    points that neighbor that point.

    If there is no loop containing both the start point and at least 2 other
    non-visited points, an empty list is returned.
    
    This function was written with assistance from GPT-4.1.
    """
    loop = []
    current = start
    neighbors = adj[current]
    if set(neighbors) - visited == set():   
        return []
    next_pt = neighbors[0] if neighbors[0] not in visited else neighbors[1]

    while True:
        loop.append(current)
        visited.add(current)

        # if all neighbors are visited, the loop is complete
        neighbors = adj[current]
        if set(neighbors) - visited == set():   
            break
        
        # otherwise, go to the next neighbor that isn't the previous point
        next_pt = neighbors[0] if neighbors[0] not in visited else neighbors[1]

        current = next_pt

    if len(loop) < 3:
        return []
    else:
        return loop

def pairs_into_polygons(pairs):
    """Join pairs of points into point-arrays representing polygons.
    
    pairs that cannot be joined together are returned as-is.
    """
    # Points must be tuples so they can be used as keys
    pairs = tuple((tuple(point1), tuple(point2)) for point1, point2 in pairs)
    
    # Adjacency dictionary
    adj = defaultdict(list)
    for p1, p2 in pairs:
        adj[p1].append(p2)
        adj[p2].append(p1)
    
    # Isolated pairs can be detected in and culled from the dictionary
    isolated_pairs = []
    for k, v in adj.items():
        if len(v) == 1:
            isolated_pairs.append([k, v])
    for p1, p2 in isolated_pairs:
        del adj[p1]

    visited = set()
    loops = []
    
    for pt in adj:
        if pt not in visited:
            loop = find_loop(pt, adj, visited)
            if loop == []:
                isolated_pairs.append([pt] + [n for n in adj[pt]])  # Review: this shouldn't be necessary
            else:
                loops.append(loop)

    return loops, isolated_pairs

def intersection_of_polyhedron_with_plane(polyhedron, plane):
    """Return an array of point-sets corresponding to the intersection of a
    polyhedron with a plane.
    
    The polyhedron must be given as a coxeter.shapes.Polyhedron instance. It does
    not need to be convex.

    The returned point-sets are all (N, 3) numpy arrays. When N = 1 (1 row), the
    point-set represents a single tangent point. When N = 2, it is a single
    tangent edge. When N = 3, it is a polygon.
    """
    # Slice each of the faces as a polygon in 3D
    slice_geometries = []
    for face in polyhedron.faces:
        polygon = polyhedron.vertices[face]
        intersection = intersection_of_polygon_with_plane(
            polygon,
            plane
        )
        if len(intersection) > 0:
            slice_geometries.extend(
                intersection
            )

    # If any of the faces were co-planar with the slice, then they
    # were returned as polygons and that might contain other segments
    # or points that were returned from intersections of other faces.
    # Remove those duplicate points and segments.
    polygons = [g for g in slice_geometries if np.array(g).shape[0] > 2]
    nonpolygons = [g for g in slice_geometries if np.array(g).shape[0] <= 2]
    clean_nonpolygons = []
    clean_indices = []
    removed_indices = []

    if polygons:
        for polygon in polygons:
            for i, nonpolygon in enumerate(nonpolygons):
                if i not in clean_indices and i not in removed_indices:
                    # Points
                    if np.array(nonpolygon).shape[0] == 1:
                        is_duplicate = geometry_contains_point(
                            polygon, nonpolygon
                        )
                    # Line segments
                    else:
                        is_duplicate = geometry_contains_segment(
                            polygon, nonpolygon
                        )

                    if is_duplicate:
                        removed_indices.append(i)
                    else:
                        clean_nonpolygons.append(nonpolygon)
                        clean_indices.append(i)
    else:
        clean_nonpolygons = nonpolygons
    
    # The remaining nonpolygon geometries must be checked to see if they
    # can be merged together:
    #   - points contained by line segments should be removed
    #   - line segments that are equivalent should be merged
    #   - line segments that connect should be joined into polygons
    points = [g for g in clean_nonpolygons if np.array(g).shape[0] == 1]
    segments = [g for g in clean_nonpolygons if np.array(g).shape[0] == 2]
    clean_points = []
    clean_segments = []
    removed_point_indices = []
    removed_segment_indices = []
    
    # Remove points contained by line segments
    for i, p in enumerate(points):
        if any(geometry_contains_point(g, p) for g in segments):
            removed_point_indices.append(i)
        else:
            clean_points.append(p)
    
    # Remove equivalent line segments
    for i, s in enumerate(segments):
        other_segments = [
            o
            for j, o in enumerate(segments)
            if j != i and j not in removed_segment_indices
        ]
        # if any(geometry_contains_segment(g, s) for g in other_segments):
        if any(
            (o[0] == s[0] and o[1] == s[1]) or (o[1] == s[0] and o[0] == s[1])
            for o in other_segments
        ):
            removed_segment_indices.append(i)
        else:
            clean_segments.append(s)

    # Join line segments into polygons
    polygons_from_segments, isolated_segments = pairs_into_polygons(
        clean_segments
    )
    
    # The geometries for this slice now consist of polygons from
    # co-planar faces, polygons created from line segments, line
    # segments that could not be connected together, and points not
    # contained by other polygons or line segments.
    slice_geometries = (
        polygons + polygons_from_segments + isolated_segments
        + clean_points
    )

    return slice_geometries


# ---------------------------------- SAMPLING ----------------------------------


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
        is shrunk by the factor ((r+b)/r), where b is the absolute value of the
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
        shape._vertices *= (r + buffer) / r
    
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

def find_nearest(array, value):
    """Find the item nearest to a given value in an array."""
    # Ref: https://stackoverflow.com/a/2566508/15426433
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# --------------------------------- SIMULATION ---------------------------------


def get_initial_frame(
    probe_body: "Body",
    analyte_body: "Body",
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
    probe_model : Body
        The body for the probe.
    analyte_model : Body
        The body for the analyte.
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
            [analyte_body.primary_type, probe_body.primary_type]
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
        frame.particles.types.index(analyte_body.primary_type),
        frame.particles.types.index(probe_body.primary_type)
    ]
    frame.configuration.box = simulation_box
    frame.particles.mass = [1] * frame.particles.N
    frame.particles.moment_inertia = [1, 1, 1] * frame.particles.N
    frame.particles.orientation = [(1, 0, 0, 0)] * frame.particles.N

    return frame

def add_rigid_constraint(
    simulation: hoomd.Simulation,
    body: "Body",
    create_bodies: bool,
    included_secondary_types: list[str] | None = None,
    rigid: hoomd.md.constrain.Rigid | None = None,
) -> Tuple[hoomd.Simulation, hoomd.md.constrain.Rigid]:
    """Add rigid body constraints for a single particle model to the simulation.

    Parameters
    ----------
    simulation : hoomd.Simulation
        The simulation to modify.
    body : Body
        The body containing the rigid body information. The model's
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
        included_secondary_types = body.secondary_types
    
    types_and_positions = [
        [t, position]
        for t in included_secondary_types
        for position in body.positions_by_type[t]
    ]

    if body.orientations_by_type:
        orientations = [
            orientation
            for t in included_secondary_types
            for orientation in body.orientations_by_type[t]
        ]
    else:
        orientations = [(1.0, 0.0, 0.0, 0.0) for _ in types_and_positions]

    rigid.body[body.primary_type] = {
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
    probe_index = 1

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
    interaction: "Interaction",
    all_types: list[str]
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
    all_types : list[str]
        The names of the particle types to parameterize the interaction for.

    Returns
    -------
    simulation
        The modified simulation.
    """
    force = interaction.to_parameterized_hoomd_instance(
        nlist=nlist,
        all_types=all_types
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

def get_simulation(
    system: "System",
    included_interactions: list["Interaction"],
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
    interactions_to_include : list[Interaction]
        The interactions to include in the simulation.
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

    # Create initial frame
    included_secondary_types = [
        t
        for t in system.all_types
        if t in system.probe.secondary_types or t in system.analyte.secondary_types
    ]
    frame = get_initial_frame(
        system.probe,
        system.analyte,
        included_secondary_types,
        probe_box,
        simulation_box
    )

    # Initialize Simulation
    simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    simulation.create_state_from_snapshot(frame)

    # Add integrator
    simulation = add_integrator(simulation)

    # Add rigid bodies if necessary
    if system.probe.is_rigid(included_interactions):
        simulation, rigid = add_rigid_constraint(
            simulation,
            system.probe,
            False if system.analyte.is_rigid(included_interactions) else True,
            [t for t in system.probe.secondary_types if t in included_secondary_types]
        )
    if system.analyte.is_rigid(included_interactions):
        simulation, _ = add_rigid_constraint(
            simulation,
            system.analyte,
            True,
            [t for t in system.analyte.secondary_types if t in included_secondary_types],
            rigid if system.probe.is_rigid(included_interactions) else None
        )

    # Add required interactions
    for interaction in included_interactions:
        simulation = add_interaction(
            simulation=simulation,
            nlist=nlist,
            interaction=interaction,
            all_types=system.all_types
        )
    
    return simulation

def run_probe(
    system: "System",
    probe_positions: list[list[float]],
    probe_orientations: list[list[float]],
    included_interactions: list["Interaction"],
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
    included_interactions : list[Interactions]
        The Interactions to include in the simulation.
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
        included_interactions,
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
        compute=None if gsd_filename is None else compute
    )
    
    probe_index = 1

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


# ----------------------------------- TABLES -----------------------------------


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
    merged_table = StringIO()
    writer = csv.writer(merged_table)

    for i, table in enumerate(table_csvs):
        table.seek(0)
        
        # skip header for all tables after the first
        if i != 0:
            next(table)

        reader = csv.reader(table)

        for row in reader:
            writer.writerow(row)
        
    return merged_table

def clean_header(table: StringIO):
    """Simplify the header of a table string buffer.

    This function is not robust. It does not search for expected column
    names and modify them in place. It requires a CSV-formatted string buffer
    with the following columns (in order):
    
    1. 'Simulation.timestep'
    2. '       x        '
    3. '       y        '
    4. '       z        '
    5. '       q0       '
    6. '       q1       '
    7. '       q2       '
    8. '       q3       '
    9. 'md.compute.ThermodynamicQuantities.potential_energy'
    
    This function replaces the header row with a new row with these columns:

    1. 't'
    2. 'x'
    3. 'y'
    4. 'z'
    5. 'q0'
    6. 'q1'
    7. 'q2'
    8. 'q3'
    9. 'PE'

    Parameters
    ----------
    table : StringIO
        The CSV-formatted string buffer.

    Returns
    -------
    cleaned_table
    """
    cleaned_table = StringIO()
    writer = csv.writer(cleaned_table)

    writer.writerow(["t","x","y","z","q0","q1","q2","q3","PE"])

    table.seek(0)
    next(table)

    for row in csv.reader(table):
        writer.writerow(row)
    
    return cleaned_table
