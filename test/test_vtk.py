# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from copy import deepcopy
from pathlib import Path

import pytest
import numpy as np
import tempfile

import p4

def image_table(dx, dy, dz):
    """Return a field table with constant spacing in X, Y, and Z."""
    n_side = 5
    rows = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                rows.append([
                    -(n_side * dx)/2 + (i * dx),
                    -(n_side * dy)/2 + (j * dy),
                    -(n_side * dz)/2 + (k * dz),
                    1,
                    0,
                    0,
                    0,
                    i + j + k,
                    i,
                    j,
                    k,
                    -i,
                    -j,
                    -k
                ])
    return np.rec.array(rows, names=["x", "y", "z", "q0", "q1", "q2", "q3", "U", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])

def rectilinear_table(xs, ys, zs):
    """Return a field table with combinations of X, Y, and Z values."""
    rows = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                rows.append([
                    x,
                    y,
                    z,
                    1,
                    0,
                    0,
                    0,
                    i + j + k,
                    i,
                    j,
                    k,
                    -i,
                    -j,
                    -k
                ])
    return np.rec.array(rows, names=["x", "y", "z", "q0", "q1", "q2", "q3", "U", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])

IMAGE_FIELD = p4.Field(image_table(1, 2, 3))

RECTILINEAR_FIELD = p4.Field(rectilinear_table(
    xs=[-3, -2, -0.15, 0.15, 2, 3],
    ys=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    zs=[-1, 0, 1]
))

UNSTRUCTURED_FIELD = p4.Field(np.rec.array(
    [
        [0, 0, 0,  1,  0,  0,  0,  0, 0, 0, 0, -0, 0, 0],
        [1, 0, 0,  1,  0,  0,  0,  1, 1, 0, 0, -1, 0, 0],
        [0, 2, 0,  1,  0,  0,  0,  2, 2, 0, 0, -2, 0, 0],
        [9, 0, 0,  1,  0,  0,  0,  3, 3, 0, 0, -3, 0, 0],
    ],
    names=["x", "y", "z", "q0", "q1", "q2", "q3", "U", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
))

UFT_FIELD = p4.Field(np.rec.array(
    [
        [-3, -3, -3,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 3, -3, -3,  1,  0,  0,  0,  1,  0,  1,  0,  0, -1,  0],
        [ 3,  3, -3,  1,  0,  0,  0,  2,  1,  1,  0, -1, -1,  0],
        [-3,  3, -3,  1,  0,  0,  0,  3,  1,  0,  0, -1,  0,  0],
        [-3, -3,  3,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 3, -3,  3,  1,  0,  0,  0,  0,  0,  1,  0,  0, -1,  0],
        [ 3,  3,  3,  1,  0,  0,  0,  4,  0,  0, -1,  0,  0,  1],
        [-3,  3,  3,  1,  0,  0,  0,  0,  1,  0,  0, -1,  0,  0],
        
    ],
    names=["x", "y", "z", "q0", "q1", "q2", "q3", "U", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
))

FIELD_KWARGS_IMAGE = [
    dict(filename="field.vti"),
    dict(filename="field.pvti"),
]
FIELD_KWARGS_RECTILINEAR = [
    dict(filename="field.vtr"),
    dict(filename="field.pvtr"),
    dict(filename="field-r-false.vtk", legacy_format="rectilinear", legacy_binary=False),
    dict(filename="field-r-true.vtk", legacy_format="rectilinear", legacy_binary=True),
]
FIELD_KWARGS_UNSTRUCTURED = [
    dict(filename="field.vtu"),
    dict(filename="field.pvtu"),
    dict(filename="field-u-false.vtk", legacy_format="unstructured", legacy_binary=False),
    dict(filename="field-u-true.vtk", legacy_format="unstructured", legacy_binary=True),
]

REFERENCE_FOLDER = Path(__file__).parent / "data"

@pytest.mark.parametrize("field", [IMAGE_FIELD, RECTILINEAR_FIELD, UFT_FIELD])
@pytest.mark.parametrize("kwargs", FIELD_KWARGS_IMAGE + FIELD_KWARGS_RECTILINEAR + FIELD_KWARGS_UNSTRUCTURED)
def test_field_to_vtk(field, kwargs):
    """Ensure field export to VTK only errors for invalid combinations of kwargs and field grid types."""
    # Unpack kwargs
    kwargs = deepcopy(kwargs)
    filename = kwargs["filename"]
    legacy_format = kwargs.get("legacy_format", "unstructured")

    # Calculate allowed grid types
    extension = str(filename).rsplit(".")[-1].lower()
    match extension:
        case "vti" | "pvti":
            allowed_grid_types = ["image"]
        case "vtr" | "pvtr":
            allowed_grid_types = ["rectilinear", "image"]
        case "vtu" | "pvtu":
            allowed_grid_types = ["unstructured", "rectilinear", "image"]
        case "vtk":
            match legacy_format:
                case "unstructured":
                    allowed_grid_types = ["unstructured", "rectilinear", "image"]
                case "rectilinear":
                    allowed_grid_types = ["rectilinear", "image"]

    # Ensure error is thrown if the field has the wrong grid type...
    if field.grid_type not in allowed_grid_types:
        with pytest.raises(ValueError):
            field.to_vtk(**kwargs)

    # ...otherwise ensure that no error is thrown.
    else:
        with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
            kwargs["filename"] = Path(tempdir) / kwargs["filename"]
            field.to_vtk(**kwargs)

@pytest.mark.parametrize("field,kwargss", [
    [IMAGE_FIELD, FIELD_KWARGS_IMAGE + FIELD_KWARGS_RECTILINEAR + FIELD_KWARGS_UNSTRUCTURED],
    [RECTILINEAR_FIELD, FIELD_KWARGS_RECTILINEAR + FIELD_KWARGS_UNSTRUCTURED],
    [UNSTRUCTURED_FIELD, FIELD_KWARGS_UNSTRUCTURED]
])
def test_field_to_from_vtk_equivalence(field, kwargss):
    """Ensure fields can be exported to and read from VTK without changing the data.
    
    Multiple sets of kwargs are provided in an array. Each set of kwargs is
    compatible with the provided field.
    """
    for kwargs in kwargss:
        kwargs = deepcopy(kwargs)
        with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
                kwargs["filename"] = Path(tempdir) / kwargs["filename"]
                field.to_vtk(**kwargs)

                test_field = p4.Field.from_vtk(kwargs["filename"])

                assert test_field == field
