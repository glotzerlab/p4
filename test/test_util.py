import coxeter
import numpy as np
import pytest
from pathlib import Path
import linecache

from p4 import util # [Review] how should I import util

# ---------------------------------- GEOMETRY ----------------------------------

REFERENCE_FOLDER = Path(__file__).parent / "data"

def stl_to_polyhedron(filepath):
    """Convert an ASCII STL file to a coxeter Polyhedron."""
    with open(filepath, "r") as f:
        # Find the line numbers corresponding to the start of each facet loop
        loop_starts = [i+2 for i, line in enumerate(f) if "outer loop" in line]

    # Parse each loop into a set of unique vertices and a list of the vertices
    # in each face. Linecache improves performance for large STL files.
    vertices = set()
    vertices_by_face = []
    for face_index, start in enumerate(loop_starts):
        loop_vertices = []
        for line_number in range(start, start + 3): 
            line = linecache.getline(str(filepath), line_number)
            
            if "vertex" not in line:
                raise Exception(
                    "Malformed STL file: detected a loop with n_vertices != 3."
                )
            
            component_str = line.rsplit("vertex ")[1]
            vertex = tuple(
                float(i) for i in component_str.split(" ")
            )

            loop_vertices.append(vertex)
            vertices.add(vertex)
        
        vertices_by_face.append(loop_vertices)
    
    # Convert the set and mapping into a list of vertices and a list of vertex
    # indices for each face.
    vertices = list(vertices)
    faces = []
    for face in vertices_by_face:
        faces.append([vertices.index(v) for v in face])

    return coxeter.shapes.Polyhedron(
        vertices=vertices, faces=faces, faces_are_convex=True
    )

def obj_to_polyhedron(filepath):
    """Convert an OBJ file to a coxeter Polyhedron."""
    vertices = []
    faces = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            
            # A vertex line starts with 'v' and has 3 or more components
            # separated by spaces. We only use the first 3 components, since
            # those are guaranteed to be x, y, and z.
            if len(line) > 0 and line[:2] == "v ":
                component_str = line.rsplit("v ")[1]
                vertices.append(
                    tuple(float(i) for i in component_str.split(" ")[:3])
                )
            
            # A face line starts with 'f' and has 3 or more components
            # separated by spaces. A component may consist of multiple indices
            # separated by '/'. The first index in a component is always the
            # vertex index, while the other indices are used for vertex normals
            # and texture mapping. Indexing is 1-based.
            elif len(line) > 0 and line[0] == "f":
                indices_str = line.rsplit("f ")[1]

                vertex_indices = []
                for substr in indices_str.split(" "):
                    vertex_indices.append(int(substr.split("/")[0]) - 1)
                
                faces.append(vertex_indices)
    
    return coxeter.shapes.Polyhedron(vertices=vertices, faces=faces)

def pointsets_are_equivalent(pointset1, pointset2):
    """Whether the sets share the same points and the points connect in the same way."""
    pointset1 = [tuple(i) for i in pointset1]
    pointset2 = [tuple(i) for i in pointset2]

    if pointset1 == pointset2:
        return True

    same_length = len(pointset1) == len(pointset2)
    same_points = True
    for p in pointset1:
        if p not in pointset2:
            same_points = False

    same_points = same_points and same_length

    if not same_points:
        return False
    
    start = pointset2.index(pointset1[0])
    same_dir = pointset1[1] == pointset2[(start + 1) % len(pointset2)]
    
    d = 1 if same_dir else -1

    same_connectivity = True
    for i, point1 in enumerate(pointset1):
        if point1 != pointset2[(start + d*i) % len(pointset2)]:
            same_connectivity = False

    return same_connectivity

@pytest.mark.parametrize("slice,expected", [
    [   # Test internal polygon, tangent polygon, tangent edge, and tangent vertex
        dict(y=0),
        [
            [   # Internal polygon
                ( 0.5,   0.0,   0.5 ),
                ( 0.5,   0.0,  -0.5 ),
                ( 0.25,  0.0,  -0.5 ),
                ( 0.25,  0.0,   0.5 ),
            ],
            [   # Tangent polygon
                (-0.5,   0.0,   0.5 ),
                (-0.5,   0.0,  -0.5 ),
                (-0.25,  0.0,  -0.5 ),
                (-0.25,  0.0,   0.5 ),
            ],
            [   # Tangent edge 1
                ( 0.0,   0.0,   0.5 ),
                ( 0.0,   0.0,   0.25),
            ],
            [   # Tangent edge 2
                ( 0.0,   0.0,   0.0 ),
                ( 0.0,   0.0,  -0.25),
            ],
            [   # Tangent vertex 1
                ( 0.0,   0.0,   0.125),
            ],
            [   # Tangent vertex 2
                ( 0.0,   0.0,  -0.5 ),
            ],
        ],
    ],
    [   # Test merging of multiple internal polygons
        dict(y=-0.25),
        [
            [   # Single merged polygon
                ( 0.5,  -0.25,  -0.5 ),
                ( 0.5,  -0.25,   0.5 ),
                (-0.5,  -0.25,   0.5 ),
                (-0.5,  -0.25,  -0.5 ),
            ],
        ],
    ],
])
def test_polyhedron_intersection_2d(slice, expected):
    """Ensure that slices return expected geometric elements."""
    filepath = REFERENCE_FOLDER / "2d.obj"

    shape = obj_to_polyhedron(filepath)

    slice_axis, slice_value = list(slice.items())[0]
    if slice_axis == "x":
        plane = [1, 0, 0, slice_value]
    elif slice_axis == "y":
        plane = [0, 1, 0, slice_value]
    elif slice_axis == "z":
        plane = [0, 0, 1, slice_value]

    actual = util.polyhedron_plane_intersection(shape, plane)
    actual = [np.array(i) for i in actual]

    expected = [np.array(i) for i in expected]

    assert len(actual) == len(expected)
    for e in expected:
        is_in_actual = False

        for a in actual:
            if pointsets_are_equivalent(a, e):
                is_in_actual = True

        assert is_in_actual

# def test_polyhedron_intersection_1d(filename, slice, expected):
#     pass


