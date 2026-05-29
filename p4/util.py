# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

"""Utility functions for p4.

In general, these functions have no defaults and do not protect themselves
from wrong/fallible inputs.
"""

from __future__ import annotations
import csv
from io import StringIO, TextIOWrapper
import itertools
from typing import Literal, Tuple
import warnings
import coxeter
import gsd.hoomd
import hoomd
import numpy as np
import rowan
from collections import defaultdict
from copy import copy
from scipy.spatial import Delaunay
from packaging.version import Version


# ------------------------------------ COLOR -----------------------------------


# https://davidmathlogic.com/colorblind
# (original source IBM Design Library)
IBM_COLORS = [
    "#5b8efd",
    "#725def",
    "#dd217d",
    "#ff5f00",
    "#ffb00d",
]

# https://davidmathlogic.com/colorblind
# (original source https://doi.org/10.1038/nmeth.1618)
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

# https://davidmathlogic.com/colorblind
# (original source Paul Tol)
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


def get_cube(side_length: float) -> coxeter.shapes.ConvexPolyhedron:
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

def angle_between_vectors(a: list[float], b: list[float]) -> float:
    """Return the smallest angle between vectors a and b in radians."""
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def angle_sign_3d(
    v1: list[float],
    v2: list[float],
    normal: list[float]
) -> float:
    """Calculate the sign of the angle of two 3D vectors relative to a normal.

    This function was mostly written by Google Gemini.
    
    Parameters
    ----------
    v1 : list[float]
        The vector to measure the angle from.
    v2 : list[float]
        The vector to measure the angle to.
    normal : list[float]
        The reference normal vector.

    Returns
    -------
    A float representing a scaled angle. Positive is CCW, negative is CW.
    """
    # v1 /= np.linalg.norm(v1)
    # v2 /= np.linalg.norm(v2)

    dot = np.dot(v1, v2)
    cross = np.cross(v1, v2)
    
    # Use the sign of the dot product between cross product and normal
    # to determine the direction relative to the reference axis
    angle = np.arctan2(np.dot(cross, normal), dot)
    return angle

def newells_normal(polygon: list[tuple[float]]) -> list[float]:
    """Calculate the normal for an arbitrary 3D polygon using Newell's method.
    
    Newell's method is required for concave polygons, since the usual 3-point
    method will give different signs depending on whether they are around a
    concave corner.

    Note that this vector does NOT have a magnitude of 1.

    Ref: https://doi.org/10.1016/B978-0-08-050755-2.50052-X

    Parameters
    ----------
    polygon : list[tuple[float]]
        The polygon in 3D.
    """
    n = len(polygon)

    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    z = [p[2] for p in polygon]

    a = sum((y[i] - y[(i+1) % n]) * (z[i] + z[(i+1) % n]) for i in range(n))
    b = sum((z[i] - z[(i+1) % n]) * (x[i] + x[(i+1) % n]) for i in range(n))
    c = sum((x[i] - x[(i+1) % n]) * (y[i] + y[(i+1) % n]) for i in range(n))

    return [a, b, c]

def point_plane_distance(point: list[float], plane: list[float]) -> float:
    """Return the smallest distance between a point and a plane.

    Ref: https://mathinsight.org/distance_point_plane
    
    Parameters
    ----------
    point : (1, 3) or (3,) array of floats
        The 3D point.
    plane : (4,) array of floats
        The linear coefficients of the plane, corresponding to [a, b, c, d],
        where the coefficients satisfy the conventional plane equation
        a*x + b*y + c*z = d.
    """
    x, y, z, = point
    a, b, c, d = plane
    return np.abs(a*x + b*y + c*z - d)/(np.sqrt((a**2 + b**2 + c**2)))

def point_segment_distance(
    point:list[float],
    segment: list[list[float]]
) -> float:
    """Return the smallest distance between a point and a line segment.

    Ref: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

    Parameters
    ----------
    point : (1, 3) or (3,) array of floats
        The 3D point.
    segment : (2, 3) array of floats
        Two points in 3D space that define a line segment.
    """
    x0 = point
    x1, x2 = segment
    x0, x1, x2 = np.array(x0), np.array(x1), np.array(x2)
    d = np.linalg.norm(np.cross(x0 - x1, x0 - x2)) / np.linalg.norm(x2 - x1)
    return d

def pointset_can_be_polygon(
    pointset: list[tuple[float]],
    unique_points: bool = False,
    noncolinear_points: bool = False,
    no_intersections: bool = False,
) -> bool:
    """Whether a pointset can be a polygon.
    
    In order for this check to return True, the pointset must have no
    overlapping or intersecting sides. Short-circuiting is enabled with
    optional inputs.

    Parameters
    ----------
    pointset : list[tuple[float]]
        The array of 3D points.
    unique_points : bool, default=False
        Whether it is guaranteed that the points are already unique.
    noncolinear_points : bool, default=False
        Whether it is already guaranteed that the points are already
        noncolinear.
    no_intersections : bool, default=False
        Whether it is already guaranteed that the sides already have no
        intersections.
    """
    # Ensure enough points
    if len(pointset) < 3:
        return False
    
    # Ensure unique points
    if not unique_points:
        if not len(set(pointset)) == len(pointset):
            return False

    # Ensure not all points are colinear (check by making sure the cross
    # products of all vectors between the starting point and other points are
    # equal to zero) - TODO: technically, we should check all triples of points
    if not noncolinear_points:
        diff = np.array(pointset)[1:,:] - np.array(pointset)[0,:]
        if np.all(np.cross(diff[0,:], diff[1:,:]) == 0):
            return False
    
    # Ensure no intersections
    if not no_intersections:
        segments = polygon_to_segments(pointset)
        for s1, s2 in itertools.combinations(segments, 2):
            # Don't consider overlapping endpoints as an intersection
            if not any(point in s2 for point in s1):
                if segment_segment_intersection(s1, s2) is not None:
                    return False

    return True

def points_straddle_plane(
    point1: list[float],
    point2: list[float],
    plane: list[float]
) -> bool:
    """Return True if the points are on opposite sides of the plane.
    
    Parameters
    ----------
    point1 : (3,) array of floats
        A 3D point.
    point2 : (3,) array of floats
        A 3D point.
    plane : (4,) array of floats
        The linear coefficients of the plane, corresponding to [a, b, c, d],
        where the coefficients satisfy the conventional plane equation
        a*x + b*y + c*z = d.
    """
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

def segment_plane_intersection(
    segment: list[list[float]],
    plane: list[float]
) -> list[tuple[float]]:
    """Return the intersection of a line segment with a plane in 3D.

    This function only works if the plane has a normal with one non-zero
    component.

    Parameters
    ----------
    segment : (2, 3) array of floats
        Two points in 3D space that define a line segment.
    plane : (4,) array of floats
        The linear coefficients of the plane, corresponding to [a, b, c, d],
        where the coefficients satisfy the conventional plane equation
        a*x + b*y + c*z = d. Only one of a, b, or c may be non-zero.
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

    return (px, py, pz)

def polygon_plane_intersection(
    polygon: list[list[float]],
    plane: list[float],
) -> list[list[tuple[float]]]:
    """Return the intersection of a polygon with a plane in 3D.
    
    The polygon is provided as an ordered list of consecutive (i.e., connected)
    points. It is assumed that
        
        - the points are co-planar with each other
        - the points are in CCW order (looking down the normal)
        - the first point is not repeated as the last point

    Because this function relies on ``intersection_of_segment_with_plane``,
    it requires that the plane normal has only one non-zero component.

    Parameters
    ----------
    polygon : (N, 3) array of floats
        The points in 3D space that consecutively define the vertices of a
        polygon.
    plane : (4,) array of floats
        The linear coefficients of the plane, corresponding to [a, b, c, d],
        where the coefficients satisfy the conventional plane equation
        a*x + b*y + c*z = d. Only one of a, b, or c may be non-zero.

    Returns
    -------
    A (N,) array of (M, 3) subarrays, where each subarray represents a set of M
    points. For M = 1, the subarray represents a single point; for M = 2, a
    line segment. In the case of no intersection, an empty list is returned.
    """
    polygon = [tuple(float(i) for i in p) for p in polygon]

    # Simplify the polygon, merging all colinear segments
    colinear_indices = []
    for i, p in enumerate(polygon):
        p_before = polygon[(i - 1) % len(polygon)]
        p_after = polygon[(i + 1) % len(polygon)]
        if point_in_segment(p, [p_before, p_after]):
            colinear_indices.append(i)
    polygon = [p for i, p in enumerate(polygon) if i not in colinear_indices]
    
    normal = newells_normal(polygon)

    # If the polygon's points are arranged CW when looking down the normal,
    # reverse the order. The handedness check works like this:
    #   1) let *p0* be the first polygon point, likewise for *p1* and *p2*
    #   2) let *c* be the polygon's centroid
    #   3) let *u* be the vector from *c* to *p0*
    #   4) let *v* be the vector from *c* to *p1*
    #   5) let *a* be the angle (relative to the normal) from *u* to *v*
    #   6) if *a* is negative, the polygon points are arranged CW and their
    #      order must be reversed.
    c = np.mean(polygon, axis=0)
    u = np.array(polygon[0]) - c
    v = np.array(polygon[1]) - c
    if angle_sign_3d(u, v, normal) < 0:
        polygon = list(reversed(polygon))
        normal = newells_normal(polygon)

    # If all of the polygon's vertices are co-planar, return the polygon.
    # If some of the vertices are co-planar, still proceed with the line segment
    # intersection checks.
    coplanar_points = [
        p for p in polygon if point_plane_distance(p, plane) < 1e-6
    ]
    if coplanar_points:
        if len(coplanar_points) == len(polygon):
            return [polygon]    # NOTE: must be wrapped in an array

    # Check for intersections between line segments and the plane. There can
    # only be intersections if the points on either end of a line segment
    # straddle the plane.
    intersected_segments = []
    for i in range(len(polygon)):
        p1 = polygon[np.mod(i, len(polygon))]
        p2 = polygon[np.mod(i+1, len(polygon))]
        if points_straddle_plane(p1, p2, plane):
            intersected_segments.append([p1, p2])
    
    # Handle the edge case where there is a single tangent point. In this case,
    # the intersection checks below will not work.
    if not intersected_segments and len(coplanar_points) == 1:
        return [coplanar_points]    # NOTE: must be wrapped in an array

    # There is only an intersection if some points are co-planar or some
    # segments are intersected.
    if not (intersected_segments or coplanar_points):
        return []
    
    else:
        # Determine the intersection point of the plane along each of the
        # intersected line segments. This is the part that only works for planes
        # with one non-zero normal component.
        intersection_points = []
        for segment in intersected_segments:
            intersection_point = segment_plane_intersection(segment, plane)
            intersection_points.append(intersection_point)
        
        # Arrange the intersection points and co-planar points together in
        # ascending order along the line formed from the intersection of the
        # plane with the polygon.
        merged_intersection_points = intersection_points + coplanar_points
        
        # The line segments containing coplanar points are guaranteed to have
        # those points as their starting points.
        coplanar_point_segments = []
        for p in coplanar_points:
            i = polygon.index(p)
            if i + 1 == len(polygon):
                coplanar_point_segments.append([polygon[i], polygon[0]])
            else:
                coplanar_point_segments.append([polygon[i], polygon[(i+1)]])
        
        merged_intersected_segments = (
            intersected_segments + coplanar_point_segments
        )

        direction = (
            np.array(merged_intersection_points[1])
            - np.array(merged_intersection_points[0])
        )
        metrics = [np.dot(direction, p) for p in merged_intersection_points]
        sorted_merged_intersection_points = [
            p for _, p in sorted(zip(metrics, merged_intersection_points))
        ]
        sorted_merged_intersected_segments = [
            s for _, s in sorted(zip(metrics, merged_intersected_segments))
        ]

        # In order for the rest of this function to work, the order of the
        # intersection points must follow the handedness of the polygon.
        must_be_reversed = False
        
        # For 3+ intersection points, the handedness check works like this:
        #   1) let *s0* be the segment containing the first intersection point,
        #      likewise for *s1* and *s2*
        #   2) let *P* be 
        #   3) In the array of segments representing the polygon, start at *s0*
        #      and look forward. If *s2* comes before *s1* then the order of the
        #      intersection points must be reversed, otherwise the handedness is
        #      preserved.
        if len(sorted_merged_intersection_points) > 2:
            s0, s1, s2 = sorted_merged_intersected_segments[0:3]
            polygon_segments = polygon_to_segments(polygon)
            start = polygon_segments.index(s0)
            for i in range(1, len(polygon)):
                current_s = polygon_segments[(start + i) % len(polygon)]
                if current_s == s2:
                    must_be_reversed = True
                    break
                elif current_s == s1:
                    break

        # For 2+ intersection points, the handedness check works like this:
        #   1) let *i0* be the first intersection point, likewise for *i1*
        #   2) let *c* be the polygon's centroid
        #   3) let *p* be the polygon point immediately after *i0* in the
        #      current order of the intersection segments array
        #   4) let *u* be the vector from *c* to *i0*
        #   5) let *v* be the vector from *c* to *p*
        #   6) let *w* be the vector from *c* to *i1*
        #   7) let *n* be the polygon normal
        #   8) let *a* be the angle (relative to *n*) from *u* to *v*
        #   9) let *b* be the angle (relative to *n*) from *u* to *w*
        #   10) if *a* and *b* have opposite signs, then the order of the
        #       intersection points must be reversed, otherwise the handedness
        #       is preserved.
        elif len(sorted_merged_intersection_points) == 2:
            i0, i1 = sorted_merged_intersection_points
            c = np.mean(polygon, axis=0)
            p = sorted_merged_intersected_segments[0][1]
            u = np.array(i0) - c
            v = np.array(p) - c
            w = np.array(i1) - c
            a = angle_sign_3d(u, v, normal)
            b = angle_sign_3d(u, w, normal)

            if ((a < 0) and (b > 0)) or ((a > 0) and (b < 0)):
                must_be_reversed = True
        
        if must_be_reversed:
            sorted_merged_intersection_points = list(
                reversed(sorted_merged_intersection_points)
            )
            sorted_merged_intersected_segments = list(
                reversed(sorted_merged_intersected_segments)
            )

        # Check each sequential pair of intersection points to see if the
        # segment between them is inside the polygon. If it is, then they form
        # an intersection segment. If not, then the first one is a tangent point
        # and the next one is checked as the start point of the next pair.

        # This check consists of the following steps:
        #   1) let *a* be a vector representing a potential intersection segment
        #   2) let *b* be a vector from the start point of the potential
        #      intersection segment to the end of the polygon
        #      segment that that point lies on
        #   3) Take the cross product b x a = c
        #   4) Compare the cross product *c* to the normal vector *n* for the
        #      polygon. If the angle between *c* and *n* is within 90 degrees,
        #      then the potential intersection segment is inside the polygon
        #      and counts as a legitimate intersection.

        # In order to calculate *a*, we need an array of potential intersection
        # segments (this is what sorted_merged_points is for) and in order to
        # calculate *b* we need an array of polygon sides that the intersection
        # points are from.

        intersection_segments = []
        tangent_intersection_points = []

        for i in range(len(sorted_merged_intersection_points) - 1):
            a = (
                np.array(sorted_merged_intersection_points[i+1])
                - np.array(sorted_merged_intersection_points[i])
            )
            b = (
                np.array(sorted_merged_intersected_segments[i][1])
                - np.array(sorted_merged_intersection_points[i])
            )
            c = np.cross(b, a)

            if (
                np.array_equal(a, b)
                or angle_between_vectors(c, normal) < np.pi/2
            ):
                intersection_segments.append([
                    sorted_merged_intersection_points[i],
                    sorted_merged_intersection_points[i+1]
                ])
            else:
                # NOTE: each tangent point must be wrapped in an array
                tangent_intersection_points.append(
                    [sorted_merged_intersection_points[i]] 
                )
                if i == len(sorted_merged_intersection_points) - 1:
                    tangent_intersection_points.append(
                        [sorted_merged_intersection_points[i+1]]
                    )

        # Merge colinear segments
        if intersection_segments:
            merged_segments = []
            current_start, current_end = intersection_segments[0]
            for i in range(len(intersection_segments) - 1):
                if intersection_segments[i+1][0] == current_end:
                    current_end = intersection_segments[i+1][1]
                else:
                    merged_segments.append([current_start, current_end])
                    current_start, current_end = intersection_segments[i+1]
            
            if not merged_segments or merged_segments[-1][1] != current_end:
                merged_segments.append([current_start, current_end])
            
            # Remove points that are contained by segments (not sure why this
            # sometimes happens)
            uncontained_tangent_intersection_points = [
                p
                for p in tangent_intersection_points
                if not any(p[0] in s for s in intersection_segments)
            ]

            return merged_segments + uncontained_tangent_intersection_points
        
        else:
            return tangent_intersection_points

def joinable_segments_to_polygon(
    segments: list[list[tuple[float]]]
) -> list[tuple[float]]:
    """Convert mutually overlapping line segments into a polygon.

    Line segments, which are represented by their start and end points, are
    flipped and shuffled around so that they form a consecutively overlapping
    loop. This loop of segments is then collapsed: duplicate points are merged
    together, and the resulting (N, 1, 3) array is flattened into a (N, 3) array
    of consecutive points that define a polygon.

    It is assumed that the segments already form a consecutively overlapping
    loop - no checks are performed to ensure that the segments all overlap and
    form a perfect loop.

    Parameters
    ----------
    segments : (N,) array of (2, 3) arrays of floats
        The line segments that overlap with one another.
    
    Returns
    -------
    A (N, 3) array of floats representing the vertices of the resulting polygon.
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

def find_loop(
    start: tuple[float],
    adj: dict[tuple[float], list[tuple[float]]],
    visited: set[tuple[float]]
) -> list[tuple[float]]:
    """Find and return a loop of connected points that are not already visited.

    If there is no loop containing both the start point and at least 2 other
    non-visited points, an empty list is returned.

    This function is not pure - it mutates the provided ``visited`` set.

    This function was written with assistance from GPT-4.1.

    Parameters
    ----------
    start : (3,) tuple of floats
        The starting point.
    adj : dict (keys: (3,) tuples of floats; values: lists of the same)
        An adjacency dictionary that maps each point to its neighbors.
    visited : set of (3,) tuples of floats
        The previously visited points.
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

def segments_to_polygons(
    segments: list[list[tuple[float]]],
) -> Tuple[list[tuple[float]], list[list[tuple[float]]]]:
    """Convert overlapping line segments into a polygon, returning others as-is.

    Line segments, which are represented by their start and end points, are
    effectively flipped and shuffled around so that they form a consecutively
    overlapping loop, which is then collapsed: duplicate points are merged
    together, and the resulting (N, 1, 3) array is flattened into a (N, 3) array
    of consecutive points that define a polygon.

    Segments that don't form polygonal (length > 2) loops are returned as-is
    separately.

    This function relies on ``find_loop``, which implements loop-finding as a
    graph-traversal algorithm.
    
    Parameters
    ----------
    segments : (N,) array of (2, 3) arrays of floats
        The line segments that overlap with one another.
    
    Returns
    -------
    A tuple of 2 arrays. The first is a (N,) array of (M, 3) subarrays of floats
    that each represent the M vertices of a loop, i.e., a polygon. The second is
    a (P,) array of (2, 3) subarrays of floats that each represent a line
    segment that could not be used to form a polygonal (length > 2) loop.
    """
    # Points must be tuples so they can be used as keys in the adjacency dict
    segments = [
        [tuple(point1), tuple(point2)]
        for point1, point2 in segments
    ]

    # Adjacency dictionary
    adj = defaultdict(list)
    for p1, p2 in segments:
        adj[p1].append(p2)
        adj[p2].append(p1)
    
    # Isolated pairs can be detected in and culled from the dictionary
    isolated_pairs = []
    for k, v in adj.items():
        if len(v) == 1:
            isolated_pairs.append([k, v[0]])
    for p1, p2 in isolated_pairs:
        del adj[p1]

    visited = set()
    loops = []
    
    for pt in copy(adj):
        if pt not in visited:
            loop = find_loop(pt, adj, visited)
            # reject loops that cannot be polygons
            if loop and pointset_can_be_polygon(loop, unique_points=True):
                loops.append(loop)

    # Remove duplicate isolated pairs
    unique_isolated_pairs = []
    for g in isolated_pairs:
        if not any(
            segments_are_equivalent(g, u) for u in unique_isolated_pairs
        ):
            unique_isolated_pairs.append(g)

    return loops, unique_isolated_pairs

def pointset_contains_point(
    pointset: list[list[float]],
    point: list[float]
) -> bool:
    """Whether a pointset contains a point.
    
    Parameters
    ----------
    pointset : (N, 3) array of floats
        The points to check. For N = 1, the pointset represents a single point;
        for N = 2, a line segment; for N > 2, a polygon.
    point : (3,) array of floats
        The point that may be contained.
    """
    return (np.atleast_2d(pointset) == np.atleast_2d(point)).all(axis=1).any()

def pointset_contains_segment(
    pointset: list[tuple[float]],
    segment: list[tuple[float]]
) -> bool:
    """Whether a pointset contains a line segment.
    
    Parameters
    ----------
    pointset : (N, 3) array of floats
        The points to check. For N = 1, the pointset represents a single point;
        for N = 2, a line segment; for N > 2, a polygon.
    segment : (2, 3) array of floats
        The line segment that may be contained.
    """
    if len(pointset) < 2:
        return False

    pointset = np.atleast_2d(pointset)
    pairs = np.stack([pointset[:-1], pointset[1:]], axis=1)
    wrap_pair = np.stack(
        [pointset[-1], pointset[0]],
        axis=0
    ).reshape(1, 2, np.array(pointset).shape[1])
    pointset_segments = np.concatenate([pairs, wrap_pair], axis=0)
    return (
        np.any(np.all(pointset_segments == segment, axis=(1,2)))
        or
        np.any(np.all(pointset_segments == segment[::-1], axis=(1,2)))
    )

def segment_segment_intersection(
    segment1: list[tuple[float]],
    segment2: list[tuple[float]],
) -> list[float] | None:
    """Return the intersection point between two segments.
    
    None is returned in the following cases:
    
    * when the segments are colinear, even if they partially overlap,
    * if the segments do not intersect at all

    If the segments are colinear but share a single point, or if they are not
    colinear and intersect at a single point, that point is returned.
    
    Ref: https://math.stackexchange.com/a/271366
    """
    # Check first if the segments share a single endpoint. If so, that is the
    # intersection point.
    if (
        (
            np.array_equal(segment1[0], segment2[0])
            and not np.array_equal(segment1[0], segment2[0])
        )
        or
        (
            np.array_equal(segment1[0], segment2[1])
            and not np.array_equal(segment1[1], segment2[0])
        )
    ):
        return segment1[0]
    
    if (
        (
            np.array_equal(segment1[1], segment2[1])
            and not np.array_equal(segment1[0], segment2[0])
        )
        or
        (
            np.array_equal(segment1[1], segment2[0])
            and not np.array_equal(segment1[0], segment2[1])
        )
    ):
        return segment1[1]
    

    c = np.array(segment1[0])
    d = np.array(segment2[0])
    g = d - c   # guaranteed not to have magnitude zero
    e = np.array(segment1[1]) - np.array(segment1[0])
    f = np.array(segment2[1]) - np.array(segment2[0])
    
    h = np.cross(f, g)
    k = np.cross(f, e)

    # If k has magnitude zero, the lines are parallel, so the only way for the
    # segments to intersect is if they share one set of endpoints. We already
    # checked for that, so we can conclude they either do not intersect at all
    # or that they partially overlap. Either way, we return None.
    if np.linalg.norm(k) == 0:
        return

    # If h has magnitude zero but g does not, then both d and c lie on the same
    # line that passes through segment2. Since k is nonzero we know that the
    # segments are not colinear. And we've already checked to make sure
    # that the two segments do not share endpoints. The only other option then
    # is for there to be no intersection.
    elif np.linalg.norm(h) == 0:
        return
    
    # At this point, we know that the lines passing through the two segments
    # must intersect at a single point. The question is whether that point lies
    # on one of (and hence both of) the segments. The position of the
    # intersection point p is given by the following relation.
    if np.dot(h, k) > 0:
        p = c + (e * np.linalg.norm(h) / np.linalg.norm(k))
    else:
        p = c - (e * np.linalg.norm(h) / np.linalg.norm(k))
    
    s1 = segment1
    s2 = segment2
    if (
        (   # Check for x
            (segment1[0][0] < p[0] and p[0] < segment1[1][0])
            or (segment1[1][0] < p[0] and p[0] < segment1[0][0])
        )
        and
        (   # Check for y
            (segment1[0][1] < p[0] and p[0] < segment1[1][1])
            or (segment1[1][1] < p[0] and p[0] < segment1[0][1])
        )
        and
        (   # Check for z
            (segment1[0][2] < p[0] and p[0] < segment1[1][2])
            or (segment1[1][2] < p[0] and p[0] < segment1[0][2])
        )
    ):
        return p
    
    else:
        return None

def polygon_to_segments(polygon: list[tuple[float]]) -> list[list[float]]:
    """Convert a polygon into an array of segments.
    
    The polygon is given as a pointset without duplicates, i.e., an (N,3) array
    of floats representing an N-vertex polygon.

    The returned segments are each (2,3) arrays of floats, where each row
    represents a point.
    """
    polygon = list(polygon)
    segments = []
    for i in range(len(polygon)):
        segments.append([polygon[i], polygon[(i+1) % len(polygon)]])
    return segments

def point_in_polygon(point: tuple[float], polygon: list[tuple[float]]) -> bool:
    """Whether a polygon contains a point.
    
    Ref: https://stackoverflow.com/a/60672266/15426433

    NOTE: because we are only slicing along coordinate axes, the polygon is
    guaranteed to already be in a coordinate plane. Therefore we only need to
    detect the degenerate coordinate and remove it (no rotating required). This
    function will not work for polygons with other orientations.
    """
    for i in [0, 1, 2]:
        if all(polygon[j][i] == polygon[j+1][i] for j in range(len(polygon)-1)):
            coordinate_to_remove = i
    
    polygon = np.array(polygon)
    polygon = polygon[:,[i for i in [0,1,2] if i != coordinate_to_remove]]
    
    point = np.array(point).flatten()
    point = [v for i, v in enumerate(point) if i != coordinate_to_remove]

    return Delaunay(polygon).find_simplex(point) >= 0

def point_in_segment(point: tuple[float], segment: list[tuple[float]]) -> bool:
    """Whether a point is on a line segment.
    
    Parameters
    ----------
    point : tuple[float]
        The 3D point.
    segment : list[tuple[float]]
        The 3D line segment, defined by two endpoints.
    """
    u = np.array(segment[1]) - np.array(segment[0])
    v = np.array(point) - np.array(segment[0])

    try:
        _ = np.dot(u, v)
    except ValueError:
        breakpoint()

    return (
        point in segment
        or (
            np.all(np.cross(u, v) == 0) # vectors are colinear
            and np.dot(u, v) > 0        # vectors point in same direction
            and np.linalg.norm(v) < np.linalg.norm(u) # point is closer than end
        )
    )

def pointset_in_polygon(
    pointset: list[tuple[float]],
    polygon: list[tuple[float]],
) -> bool:
    """Whether every point in a pointset is within a polygon.
    
    Points on the edges and vertices of a polygon count as inside.
    
    Parameters
    ----------
    pointset : list[tuple[float]]
        An array of 3D points.
    segment : list[tuple[float]]
        The polygon, defined as a sequential list of points.
    """
    is_inside = []
    if isinstance(pointset[0], list):
        breakpoint()
    for p in pointset:
        on_edge = any(
            point_in_segment(p, s) for s in polygon_to_segments(polygon)
        )

        if on_edge:
            is_inside.append(True)
        
        else:
            is_inside.append(point_in_polygon(p, polygon))

    return all(is_inside)

def polygon_contains_polygon(polygon1, polygon2) -> int:
    """Check if polygon1 fully contains polygon2.
    
    Check for intersections between all pairs of polygon sides. If there are no
    intersections and one of the points in polygon1 is inside polygon2, then
    polygon1 is fully contained by polygon2.

    The result of the contains check is returned as an integer. The values mean
    the following:
    
    * 0 - neither polygon contains the other
    * 1 - polygon1 contains polygon2
    * 2 - polygon2 contains polygon1
    """
    segments1 = polygon_to_segments(polygon1)
    segments2 = polygon_to_segments(polygon2)

    for s1, s2 in itertools.product(segments1, segments2):
        if segment_segment_intersection(s1, s2) is not None:
            return 0
    
    else:
        for point in polygon1:
            if point_in_polygon(point, polygon2):
                return 2
        for point in polygon2:
            if point_in_polygon(point, polygon1):
                return 1
    
    return 0

def segments_are_equivalent(
    segment1: list[tuple[float]],
    segment2: list[tuple[float]]
) -> bool:
    """Return whether segment1 and segment2 contain the same points.

    Parameters
    ----------
    segment1 : list[tuple[float]]
        A 3D line segment consisting of a (2,3) array of floats.
    segment2 : list[tuple[float]]
        A 3D line segment consisting of a (2,3) array of floats.
    """
    return (
        (segment1[0] == segment2[0] and segment1[1] == segment2[1])
        or (segment1[1] == segment2[0] and segment1[0] == segment2[1])
    )

def polygons_are_equivalent(
    polygon1: list[tuple[float]],
    polygon2: list[tuple[float]]
) -> bool:
    """Return whether polygon1 and polygon2 contain the same points.

    Polygons are equivalent if they have the same points ordered in the same
    handedness.

    Parameters
    ----------
    polygon1 : list[tuple[float]]
        A 3D polygon consisting of a (N,3) array of floats.
    polygon2 : list[tuple[float]]
        A 3D polygon consisting of a (N,3) array of floats.
    """
    # Same length
    if not len(polygon1) == len(polygon2):
        return False

    # Same constituent points
    if not all(p in polygon2 for p in polygon1):
        return False
    
    # Same handedness
    p2_start = polygon2.index(polygon1[0])
    for i in range(len(polygon1)):
        if not polygon1[i] == polygon2[(p2_start+i) % len(polygon2)]:
            return False
        
    return True

def polyhedron_plane_intersection(
    polyhedron: coxeter.shapes.Polyhedron,
    plane: list[float],
) -> list[list[tuple[float]]]:
    """Return the intersection of a polyhedron with a plane.

    Parameters
    ----------
    polyhedron : coxeter.shapes.Polyhedron
        The polyhedron to slice with the plane.
    plane : (4,) array of floats
        The linear coefficients of the plane, corresponding to [a, b, c, d],
        where the coefficients satisfy the conventional plane equation
        a*x + b*y + c*z = d. Only one of a, b, or c may be non-zero.

    Returns
    -------
    A (N,) array of (M, 3) subarrays, where each subarray represents a set of M
    points. For M = 1, the subarray represents a single point; for M = 2, a
    line segment; for M > 2, a polygon. In the case of no intersection, an empty
    list is returned.
    """
    # Slice each of the faces as a polygon in 3D
    slice_geometries = []
    for face in polyhedron.faces:
        polygon = polyhedron.vertices[face]
        intersection = polygon_plane_intersection(polygon, plane)
        if len(intersection) > 0:
            slice_geometries.extend(intersection)

    # Remove all duplicate geometries:
    #   - points are duplicates if they are identical
    #   - segments are equivalent if they contain the same points
    #   - polygons are equivalent if they have the same points arranged in the
    #     same handedness
    unique_geometries = []
    for i, g in enumerate(slice_geometries):
        if len(g) == 1:
            if g not in unique_geometries:
                unique_geometries.append(g)
        elif len(g) == 2:
            if not any(
                segments_are_equivalent(g, u)
                for u in unique_geometries
                if len(u) == 2
            ):
                unique_geometries.append(g)
        else:
            if not any(
                polygons_are_equivalent(g, u)
                for u in unique_geometries
                if len(u) == len(g)
            ):
                unique_geometries.append(g)

    # Remove all geometries contained by other geometries. At this stage, points
    # and segments can be contained by other geometries, but polygons cannot.
    uncontained_geometries = []
    for i, g in enumerate(unique_geometries):
        other_geometries = [
            h for j, h in enumerate(unique_geometries) if j != i
        ]
        if len(g) == 1:
            if not any(
                pointset_contains_point(o, g) for o in other_geometries
            ):
                uncontained_geometries.append(g)
        elif len(g) == 2:
            if not any(
                pointset_contains_segment(o, g) for o in other_geometries
            ):
                uncontained_geometries.append(g)
        else:
            uncontained_geometries.append(g)
       
    # For bookkeeping, split geometries into arrays of different types
    points = [g for g in uncontained_geometries if len(g) == 1]
    segments = [g for g in uncontained_geometries if len(g) == 2]
    polygons = [g for g in uncontained_geometries if len(g) > 2]

    # Join overlapping line segments into polygons
    polygons_from_segments, isolated_segments = segments_to_polygons(
        segments,
    )

    # Remove points, segments, and now polygons that are contained by other
    # polygons that were constructed from line segments.
    contained_points_indices = []
    contained_isolated_segments_indices = []
    contained_polygons_indices = []
    for polygon_from_segments in polygons_from_segments:

        # Check for points
        for i, point in enumerate(points):
            if i not in contained_points_indices:
                if pointset_in_polygon(point, polygon_from_segments):
                    contained_points_indices.append(i)
        
        # Check for isolated segments
        for i, segment in enumerate(isolated_segments):
            if i not in contained_isolated_segments_indices:
                try:
                    if pointset_in_polygon(segment, polygon_from_segments):
                        contained_isolated_segments_indices.append(i)
                except:
                    breakpoint()
        
        # Check for polygons that were not constructed from segments
        for i, polygon in enumerate(polygons):
            if i not in contained_polygons_indices:
                if pointset_in_polygon(polygon, polygon_from_segments):
                    contained_polygons_indices.append(i)
    
    uncontained_points = [
        p for i, p in enumerate(points) if i not in  contained_points_indices
    ]
    uncontained_isolated_segments = [
        s
        for i, s in enumerate(isolated_segments)
        if i not in contained_isolated_segments_indices
    ]
    uncontained_polygons = [
        p for i, p in enumerate(polygons) if i not in contained_polygons_indices
    ]

    return (
        uncontained_polygons + polygons_from_segments
        + uncontained_isolated_segments + uncontained_points
    )

def polyhedron_line_intersection(
    polyhedron: coxeter.shapes.Polyhedron,
    line: list[list[float]],
    point_size_for_slice: float = 1e-6,
) -> list[list[tuple[float]]]:
    """Return the intersection of a polyhedron with a line.

    Parameters
    ----------
    polyhedron : coxeter.shapes.Polyhedron
        The polyhedron to slice with the plane.
    line : list[list[float]]
        The two planes whose intersection defines the line. Each element of this
        array is an array of 4 numbers, corresponding to [a, b, c, d],
        where the coefficients satisfy the conventional plane equation
        a*x + b*y + c*z = d. Only one of a, b, or c may be non-zero.
    point_size_for_slice : float, default=1e-6
        The distance within which a point is considered to be contained by
        a plane.

    Returns
    -------
    A (N,) array of (M, 3) subarrays, where each subarray represents a set of M
    points. For M = 1, the subarray represents a single point; for M = 2, a
    line segment; for M > 2, a polygon. In the case of no intersection, an empty
    list is returned.
    """
    plane1, plane2 = line
    
    slice_x, slice_y, slice_z = None, None, None
    if plane1[0] == 1:
        slice_x = plane1[-1]
    elif plane2[0] == 1:
        slice_x = plane2[-1]
    if plane1[1] == 1:
        slice_y = plane1[-1]
    elif plane2[1] == 1:
        slice_y = plane2[-1]
    if plane1[2] == 1:
        slice_z = plane1[-1]
    elif plane2[2] == 1:
        slice_z = plane2[-1]

    # Create pseudoline for point intersection checks
    # Review: there must be a better way...
    if slice_x is not None:
        if slice_y is not None:
            pseudoline = [[slice_x, slice_y, -1e9], [slice_x, slice_y, 1e9]]
        else:
            pseudoline = [[slice_x, -1e9, slice_z], [slice_x, 1e9, slice_z]]
    else:
        pseudoline = [[-1e9, slice_y, slice_z], [1e9, slice_y, slice_z]]

    slice_geometries1 = polyhedron_plane_intersection(polyhedron, plane1)

    # Slice the current slice with the second plane
    slice_geometries2 = []
    for geometry in slice_geometries1:
        # Points must be within the distance tolerance
        if len(geometry) == 1:
            # breakpoint()
            px, py, pz = geometry[0]
            if point_segment_distance([px, py, pz], pseudoline) < point_size_for_slice:
                slice_geometries2.append(geometry)
        
        # Segments must pass the intersection test
        if len(geometry) == 2:
            segment_is_intersected = (
                (
                    plane2[1] == 1
                    and slice_y >= np.array(geometry)[:,1].min()
                    and slice_y <= np.array(geometry)[:,1].max()
                )
                or
                (
                    plane2[2] == 1
                    and slice_z >= np.array(geometry)[:,2].min()
                    and slice_z <= np.array(geometry)[:,2].max()
                )
            )
            if segment_is_intersected:
                slice_geometries2.append(
                    segment_plane_intersection(geometry, plane2)
                )

        # Polygons are sliced like in 2D plotting
        if len(geometry) > 2:
            geometries = polygon_plane_intersection(geometry, plane2)
            slice_geometries2.extend(geometries)
    
    return slice_geometries2


# --------------------------------- SIMULATION ---------------------------------


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
    """Return a simulation frame with analyte at center and probe at edge.

    All provided types are included in the particle type data, but only primary
    types specified on the provided particle models are actually placed.
    Secondary types **are not** placed in the frame and must be added separately
    using `create_rigid_bodies()`.

    Parameters
    ----------
    probe : Body
        The body for the probe.
    analyte : Body | Arrangement
        The body or arrangement for the analyte.
    simulation_box : list[float]
        The side lengths of the simulation box. [Lx, Ly, Lz]

    Returns
    -------
    frame
        The initial frame.
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
    quantities: Literal["U", "F", "T"] | list[Literal["U", "F", "T"]],
    probe_is_rigid: bool,
    compute: hoomd.md.compute.ThermodynamicQuantities | None = None
) -> Tuple[hoomd.Simulation, hoomd.logging.Logger]:
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
    simulation, compute
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
    force = interaction.to_hoomd_pair(nlist=nlist)
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
    simulation_box : list[float]
        The simulation's box in HOOMD notation. $[Lx, Ly, Lz, xy, xz, yz]$

    Returns
    -------
    hoomd.Simulation
        The simulation object, fully prepared and ready to be run.
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

    # Create rigid constraint   [TODO: refactor to make this less hacky]
    rigid = system.analyte.to_hoomd_rigid()
    if system.probe._is_rigid(system.interactions):
        rigid = system.probe.to_hoomd_rigid(rigid)

    # Add integrator
    simulation = add_integrator(simulation, rigid)

    # Add required interactions
    for interaction in included_interactions:
        simulation = add_interaction(
            simulation=simulation,
            nlist=nlist,
            interaction=interaction
        )

    return simulation

def measure(
    system: "System",
    quantities: Literal["U", "F", "T"] | list[Literal["U", "F", "T"]],
    positions: list[list[float]],
    orientations: list[list[float]],
    included_interactions: list["Interaction"],
    simulation_box: list[float],
    gsd_filename: str | None,
    nlist: hoomd.md.nlist.NeighborList,
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
    positions : list[list[float]]
        The positions to measure at.
    orientations : list[list[float]]
        The orientations (in quaternion form) to measure at for each position.
    included_interactions : list[Interactions]
        The Interactions to include in the simulation.
    gsd_filename : str
        The name of the final output CSV file. This is not used to actually
        write.
    measurement_box : list[float]
        The side lengths $[Lx, Ly, Lz]$ of the box containing the positions to
        measure at.
    simulation_box : list[float]
        The simulation's box in HOOMD notation. $[Lx, Ly, Lz, xy, xz, yz]$
    gsd_filename : str
        The name of the GSD file to save. If not provided, no GSD file will be
        saved.
    nlist : hoomd.md.nlist.NeighborList
        The neighbor list to use for the interactions.

    Returns
    -------
    table
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
    simulation = get_simulation(
        system,
        included_interactions,
        nlist,
        simulation_box
    )

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
        probe_is_rigid=system.probe._is_rigid(included_interactions),
        probe_index=probe_index,
        compute=None if gsd_filename is None else compute,
    )

    # Iterate over positions
    for p, os in zip(positions, orientations):
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


# ----------------------------------- TABLES -----------------------------------


def merge_tables(table_csvs: list[StringIO]):
    """Combine an array of tables stored in string buffers.

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

    This function strips whitespace from column names and if a column named 
    'md.compute.ThermodynamicQuantities.potential_energy' is present, it is
    rebnamed to 'U'.

    Parameters
    ----------
    table : StringIO
        The CSV-formatted string buffer.

    Returns
    -------
    cleaned_table
    """
    # Calculate the clean column names
    table.seek(0)
    raw_header = table.readline()

    raw_columns = raw_header.split(",")
    clean_columns = []
    for c in raw_columns:
        if c.strip() == "md.compute.ThermodynamicQuantities.potential_energy":
            clean_columns.append("U")
        else:
            clean_columns.append(c.strip())

    # Build the clean table
    cleaned_table = StringIO()
    writer = csv.writer(cleaned_table)
    writer.writerow(clean_columns)

    table.seek(0)
    next(table)

    for row in csv.reader(table):
        writer.writerow(row)
    
    return cleaned_table


# --------------------------------- VALIDATION ---------------------------------


def sanitize(d: dict):
    """Traverse a dictionary, converting numpy types to native python analogues.
    
    Parameters
    ----------
    d : dict
        The dictionary to traverse.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            sanitize(v)
        else:
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, np.number):
                d[k] = v.item()
            elif isinstance(v, (list, tuple)):
                if v:
                    new_v = None
                    if isinstance(v[0], np.ndarray):
                        new_v = [i.tolist() for i in v]
                    elif isinstance(v[0], np.floating):
                        new_v = [float(i) for i in v]
                    elif isinstance(v[0], np.integer):
                        new_v = [int(i) for i in v]
                    
                    # Convert back to tuple if necessary
                    if new_v and isinstance(v, tuple):
                        new_v = tuple(new_v)
                    
                    if new_v:
                        d[k] = new_v
                    else:
                        d[k] = v
        
    return d
