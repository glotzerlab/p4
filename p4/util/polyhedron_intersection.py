# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

"""Functions for calculating polyhedron-plane intersections.

These functions are not optimized for performance and require adherence to 
specific types for representing geometric primitives - see :py:type:`plane`,
:py:type:`point`, :py:type:`segment`, and :py:type:`pointset`.

These functions obey the following assumptions:

* Points, segments, and planes are all embedded in 3D space.

* Polyhedra do not have self-intersections.

* Planes are orthogonal to the principle coordinate axes.

* Polygons are represented by **open** pointsets, meaning that the final point
  is not a duplicate of the first point.
"""

from collections import defaultdict
from copy import copy
import itertools

import numpy as np
import coxeter
from scipy.spatial import Delaunay


type Plane = tuple[float, float, float, float]
"""An array of 4 coefficients that represent a plane in 3D space.

These coefficients corresponding to ``[a, b, c, d]``, which define the plane
through the equation ``a*x + b*y + c*z = d``.
"""

type Point = tuple[float, float, float]
"""A point in 3D space."""

type Segment = tuple[Point, Point]
"""A pair of points that define a line segment."""

type PointSet = tuple[Point, ...]
"""An ordered array of 3D points."""

type Vector3D = tuple[float, float, float]
"""An 3-dimensional vector."""

type VectorND = tuple[float, ...]
"""An N-dimensional vector."""


def get_cube(s: float) -> coxeter.shapes.ConvexPolyhedron:
    """Return a coxeter cube with a given side length ``s``."""
    vertices = [
        [-s/2, -s/2, -s/2],
        [-s/2, -s/2,  s/2],
        [-s/2,  s/2, -s/2],
        [-s/2,  s/2,  s/2],
        [ s/2, -s/2, -s/2],
        [ s/2, -s/2,  s/2],
        [ s/2,  s/2, -s/2],
        [ s/2,  s/2,  s/2]
    ]
    return coxeter.shapes.ConvexPolyhedron(vertices)

def angle_between_vectors(v1: VectorND, v2: VectorND) -> float:
    """Return the smallest angle between two vectors in radians."""
    return np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )

def angle_sign_3d(v1: VectorND, v2: VectorND, normal: VectorND) -> float:
    """Calculate the sign of the angle of two 3D vectors relative to a normal.

    ..
        Disclaimer: this function was mostly written by Google Gemini.
    
    Parameters
    ----------
    v1 : VectorND
        The vector to measure the angle from.
    v2 : VectorND
        The vector to measure the angle to.
    normal : VectorND
        The reference normal vector.

    Returns
    -------
    float
        The scaled angle. Positive is CCW, negative is CW.
    """
    # v1 /= np.linalg.norm(v1)
    # v2 /= np.linalg.norm(v2)

    dot = np.dot(v1, v2)
    cross = np.cross(v1, v2)
    
    # Use the sign of the dot product between cross product and normal
    # to determine the direction relative to the reference axis
    angle = np.arctan2(np.dot(cross, normal), dot)
    return angle

def newells_normal(polygon: PointSet) -> Vector3D:
    """Calculate the normal for an arbitrary 3D polygon using Newell's method.
    
    Newell's method is required for concave polygons, since the usual 3-point
    method will give different signs depending on whether they are around a
    concave corner.

    .. note::
        The normal vector returned by this function does NOT have a magnitude of
        1.

    This implementation follows `Tampieri 1992`_.
    
    .. _`Tampieri 1992`: https://doi.org/10.1016/B978-0-08-050755-2.50052-X

    Parameters
    ----------
    polygon : PointSet
        The polygon.
    """
    n = len(polygon)

    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    z = [p[2] for p in polygon]

    a = sum((y[i] - y[(i+1) % n]) * (z[i] + z[(i+1) % n]) for i in range(n))
    b = sum((z[i] - z[(i+1) % n]) * (x[i] + x[(i+1) % n]) for i in range(n))
    c = sum((x[i] - x[(i+1) % n]) * (y[i] + y[(i+1) % n]) for i in range(n))

    return (a, b, c)

def point_plane_distance(point: Point, plane: Plane) -> float:
    """Return the smallest distance between a point and a plane.

    ..
        Ref: https://mathinsight.org/distance_point_plane
    """
    x, y, z, = point
    a, b, c, d = plane
    return np.abs(a*x + b*y + c*z - d)/(np.sqrt((a**2 + b**2 + c**2)))

def point_segment_distance(point: Point, segment: Segment) -> float:
    """Return the smallest distance between a point and a line segment.

    This implementation follows `Wolfram Alpha`_.
    
    .. _`Wolfram Alpha`: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    """
    x0 = point
    x1, x2 = segment
    x0, x1, x2 = np.array(x0), np.array(x1), np.array(x2)
    d = np.linalg.norm(np.cross(x0 - x1, x0 - x2)) / np.linalg.norm(x2 - x1)
    return d

def pointset_can_be_polygon(
    pointset: PointSet,
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
    pointset : PointSet
        The set of 3D points.
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

def points_straddle_plane(point1: Point, point2: Point, plane: Plane) -> bool:
    """Whether two points are on opposite sides of a plane."""
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

def segment_plane_intersection(segment: Segment, plane: Plane) -> Point:
    """Return the intersection point of a line segment with a plane.

    This function only works if the plane has a normal with one non-zero
    component. In other words, the plane must be perpendicular to one of the
    coordinate axes.

    This function assumes that the segment does intersect with the plane. The
    results are invalid if it does not.
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
    polygon: PointSet,
    plane: Plane,
) -> list[PointSet]:
    """Return the intersection of a polygon with a plane in 3D.
    
    The polygon is provided as an ordered list of consecutive (i.e., connected)
    points. It is assumed that
        
    * the points are co-planar with each other
    
    * the points are in CCW order (looking down the normal)
    
    * the first point is not repeated as the last point

    Because this function relies on
    :py:func:`~p4.util.polyhedron_intersection.segment_plane_intersection`,
    it requires that the plane be orthogonal to the principle coordinate axes.

    Parameters
    ----------
    polygon : PointSet
        A sequence of points that comprise the vertices of a polygon.
    plane : Plane
        The plane, whose normal can only have one non-zero component.

    Returns
    -------
    list[PointSet]
        The intersection geometries, returned as an (N, M, 3) array of floats.
        Each of the ``N`` contiguous geometries is defined by some number of
        points ``M``. For ``M = 1``, the subarray represents a single point; for
        ``M = 2``, a line segment. In the case of no intersection, an empty list
        is returned.
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

    # Determine the intersection point of the plane along each of the
    # intersected line segments. This is the part that only works for planes
    # with one non-zero normal component.
    intersection_points = []
    for segment in intersected_segments:
        intersection_points.append(segment_plane_intersection(segment, plane))

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
    #   2) In the array of segments representing the polygon, start at *s0*
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

    # For 2 intersection points, the handedness check works like this:
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
    #      intersection segment to the end of the polygon segment that that
    #      point lies on
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
            or np.isclose(normalize(c), normalize(normal)).all()
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

def joinable_segments_to_polygon(segments: list[Segment]) -> PointSet:
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
    segments : list[Segment]
        The line segments that overlap with one another.
    
    Returns
    -------
    PointSet
        The vertices of the resulting polygon.
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

    return points

def find_loop(
    start: Point,
    adj: dict[Point, list[Point]],
    visited: set[Point]
) -> list[Point]:
    """Find and return a loop of connected points that are not already visited.

    If there is no loop containing both the start point and at least 2 other
    non-visited points, an empty list is returned.

    This function is not pure - it mutates the provided ``visited`` set.

    ..
        This function was written with assistance from GPT-4.1.

    Parameters
    ----------
    start : Point
        The starting point.
    adj : dict[Point, list[Point]]
        An adjacency dictionary that maps each point to its neighbors.
    visited : set[Point]
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
    segments: list[Segment],
) -> tuple[PointSet, list[Segment]]:
    """Convert overlapping line segments into a polygon, returning others as-is.

    Line segments, which are represented by their start and end points, are
    effectively flipped and shuffled around so that they form a consecutively
    overlapping loop, which is then collapsed: duplicate points are merged
    together, and the resulting (N, 1, 3) array is flattened into a (N, 3) array
    of consecutive points that define a polygon.

    Segments that do not form polygonal (length > 2) loops are returned as-is
    separately.

    This function relies on
    :py:func:`~p4.util.polyhedron_intersection.find_loop`, which implements
    loop-finding as a graph-traversal algorithm.
    
    Parameters
    ----------
    segments : list[Segments]
        The line segments that may overlap with one another.
    
    Returns
    -------
    tuple[PointSet, list[Segment]]
        The first item is a (N,) array of (M, 3) subarrays of floats
        that each represent the M vertices of a loop, i.e., a polygon. The
        second array is a (P,) array of (2, 3) subarrays of floats that each
        represent a line segment that could not be used to form a polygonal
        (length > 2) loop.
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

def pointset_contains_point(pointset: PointSet, point: Point) -> bool:
    """Whether a pointset contains a point."""
    return (np.atleast_2d(pointset) == np.atleast_2d(point)).all(axis=1).any()

def pointset_contains_segment(pointset: PointSet, segment: Segment) -> bool:
    """Whether a pointset contains a line segment."""
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
    segment1: Segment,
    segment2: Segment,
) -> Point | None:
    """Return the intersection point between two segments.
    
    None is returned in the following cases:
    
    * when the segments are colinear, even if they partially overlap,

    * if the segments do not intersect at all

    If the segments are colinear but share a single point, or if they are not
    colinear and intersect at a single point, that point is returned.
    
    This implementation follows Michael E2's `derivation`_, published on
    Math Stack Exchange under the CC BY-SA 4.0 license.
    
    .. _`derivation`: https://math.stackexchange.com/a/271366
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

def polygon_to_segments(polygon: PointSet) -> list[Segment]:
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

def point_in_polygon(point: Point, polygon: PointSet) -> bool:
    """Whether a point is inside a polygon (tangent points count as inside).
    
    This implementation follows BottleNick's `implementation`_, published on
    StackOverflow under the CC BY-SA 4.0 license.
    
    .. _`implementation`: https://stackoverflow.com/a/60672266/15426433

    .. note:
        Because slicing is only allowed along coordinate axes, the polygon is
        guaranteed to already be in a coordinate plane. Therefore this
        implementation only needs to detect the degenerate coordinate and remove
        it - no rotating into a principle axis is required. This function will
        not work for polygons created by non-orthogonal slicing.
    """
    for i in [0, 1, 2]:
        if all(polygon[j][i] == polygon[j+1][i] for j in range(len(polygon)-1)):
            coordinate_to_remove = i
    
    polygon = np.array(polygon)
    polygon = polygon[:,[i for i in [0,1,2] if i != coordinate_to_remove]]
    
    point = np.array(point).flatten()
    point = [v for i, v in enumerate(point) if i != coordinate_to_remove]

    return Delaunay(polygon).find_simplex(point) >= 0

def point_in_segment(point: Point, segment: Segment) -> bool:
    """Whether a point is on a line segment."""
    u = np.array(segment[1]) - np.array(segment[0])
    v = np.array(point) - np.array(segment[0])

    try:
        _ = np.dot(u, v)
    except ValueError:
        breakpoint()    # TODO

    return (
        point in segment
        or (
            np.all(np.cross(u, v) == 0) # vectors are colinear
            and np.dot(u, v) > 0        # vectors point in same direction
            and np.linalg.norm(v) < np.linalg.norm(u) # point is closer than end
        )
    )

def pointset_in_polygon(pointset: PointSet, polygon: PointSet) -> bool:
    """Whether every point in a pointset is inside a polygon.
    
    Points on the edges and vertices of a polygon count as inside.
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

def polygon_contains_polygon(polygon1: PointSet, polygon2: PointSet) -> int:
    """Check if ``polygon1`` fully contains ``polygon2``.
    
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

def segments_are_equivalent(segment1: Segment, segment2: Segment) -> bool:
    """Whether two segments are defined by the same endpoints in any order."""
    return (
        (segment1[0] == segment2[0] and segment1[1] == segment2[1])
        or (segment1[1] == segment2[0] and segment1[0] == segment2[1])
    )

def polygons_are_equivalent(polygon1: PointSet, polygon2: PointSet) -> bool:
    """Whether two polygons are equivalent.

    Polygons are equivalent if they have the same points ordered in the same
    handedness.
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
    plane: Plane,
) -> list[PointSet]:
    """Return the intersection of a polyhedron with a plane.

    Because this function relies on
    :py:func:`~p4.util.polyhedron_intersection.polygon_plane_intersection`,
    it requires that the plane be orthogonal to the principle coordinate axes.

    Parameters
    ----------
    polyhedron : coxeter.shapes.Polyhedron
        The polyhedron to slice.
    plane : Plane
        The plane plane, whose normal can only have one non-zero component.

    Returns
    -------
    list[PointSet]
        The intersection geometries, returned as an (N, M, 3) array of floats.
        Each of the ``N`` contiguous geometries is defined by some number of
        points ``M``. For ``M = 1``, the subarray represents a single point; for
        ``M = 2``, a line segment. In the case of no intersection, an empty list
        is returned.
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
    line: tuple[Plane, Plane],
    point_size_for_slice: float = 1e-6,
) -> list[PointSet]:
    """Return the intersection of a polyhedron with a line.

    Because this function relies on
    :py:func:`~p4.util.polyhedron_intersection.polyhedron_plane_intersection`,
    it requires that the plane be orthogonal to the principle coordinate axes.

    Parameters
    ----------
    polyhedron : coxeter.shapes.Polyhedron
        The polyhedron to slice.
    line : tuple[Plane, Plane]
        The two planes whose intersection defines the line. The planes, whose
        normals can each have only one non-zero component.
    point_size_for_slice : float, default=1e-6
        The distance within which a point is considered to be contained by
        a plane.

    Returns
    -------
    list[PointSet]
        The intersection geometries, returned as an (N, M, 3) array of floats.
        Each of the ``N`` contiguous geometries is defined by some number of
        points ``M``. For ``M = 1``, the subarray represents a single point; for
        ``M = 2``, a line segment. In the case of no intersection, an empty list
        is returned.
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
