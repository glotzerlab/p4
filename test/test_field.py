# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from pathlib import Path

import pytest
import numpy as np
from copy import copy
import p4

import tempfile

VALID_LOCATIONS = np.array([
    [-1.0, -1.0, -1.0, 1, 0, 0, 0],
    [-1.0, -1.0, -1.0, 0, 1, 0, 0],
    [-1.0, -1.0, -1.0, 0, 0, 1, 0],
    [-1.0, -1.0,  0.0, 1, 0, 0, 0],
    [-1.0, -1.0,  0.0, 0, 1, 0, 0],
    [-1.0, -1.0,  0.0, 0, 0, 1, 0],
    [-1.0, -1.0,  1.0, 1, 0, 0, 0],
    [-1.0, -1.0,  1.0, 0, 1, 0, 0],
    [-1.0, -1.0,  1.0, 0, 0, 1, 0],
    [-1.0,  0.0, -1.0, 1, 0, 0, 0],
    [-1.0,  0.0, -1.0, 0, 1, 0, 0],
    [-1.0,  0.0, -1.0, 0, 0, 1, 0],
    [-1.0,  0.0,  0.0, 1, 0, 0, 0],
    [-1.0,  0.0,  0.0, 0, 1, 0, 0],
    [-1.0,  0.0,  0.0, 0, 0, 1, 0],
    [-1.0,  0.0,  1.0, 1, 0, 0, 0],
    [-1.0,  0.0,  1.0, 0, 1, 0, 0],
    [-1.0,  0.0,  1.0, 0, 0, 1, 0],
    [-1.0,  1.0, -1.0, 1, 0, 0, 0],
    [-1.0,  1.0, -1.0, 0, 1, 0, 0],
    [-1.0,  1.0, -1.0, 0, 0, 1, 0],
    [-1.0,  1.0,  0.0, 1, 0, 0, 0],
    [-1.0,  1.0,  0.0, 0, 1, 0, 0],
    [-1.0,  1.0,  0.0, 0, 0, 1, 0],
    [-1.0,  1.0,  1.0, 1, 0, 0, 0],
    [-1.0,  1.0,  1.0, 0, 1, 0, 0],
    [-1.0,  1.0,  1.0, 0, 0, 1, 0],
    [ 0.0, -1.0, -1.0, 1, 0, 0, 0],
    [ 0.0, -1.0, -1.0, 0, 1, 0, 0],
    [ 0.0, -1.0, -1.0, 0, 0, 1, 0],
    [ 0.0, -1.0,  0.0, 1, 0, 0, 0],
    [ 0.0, -1.0,  0.0, 0, 1, 0, 0],
    [ 0.0, -1.0,  0.0, 0, 0, 1, 0],
    [ 0.0, -1.0,  1.0, 1, 0, 0, 0],
    [ 0.0, -1.0,  1.0, 0, 1, 0, 0],
    [ 0.0, -1.0,  1.0, 0, 0, 1, 0],
    [ 0.0,  0.0, -1.0, 1, 0, 0, 0],
    [ 0.0,  0.0, -1.0, 0, 1, 0, 0],
    [ 0.0,  0.0, -1.0, 0, 0, 1, 0],
    [ 0.0,  0.0,  0.0, 1, 0, 0, 0],
    [ 0.0,  0.0,  0.0, 0, 1, 0, 0],
    [ 0.0,  0.0,  0.0, 0, 0, 1, 0],
    [ 0.0,  0.0,  1.0, 1, 0, 0, 0],
    [ 0.0,  0.0,  1.0, 0, 1, 0, 0],
    [ 0.0,  0.0,  1.0, 0, 0, 1, 0],
    [ 0.0,  1.0, -1.0, 1, 0, 0, 0],
    [ 0.0,  1.0, -1.0, 0, 1, 0, 0],
    [ 0.0,  1.0, -1.0, 0, 0, 1, 0],
    [ 0.0,  1.0,  0.0, 1, 0, 0, 0],
    [ 0.0,  1.0,  0.0, 0, 1, 0, 0],
    [ 0.0,  1.0,  0.0, 0, 0, 1, 0],
    [ 0.0,  1.0,  1.0, 1, 0, 0, 0],
    [ 0.0,  1.0,  1.0, 0, 1, 0, 0],
    [ 0.0,  1.0,  1.0, 0, 0, 1, 0],
    [ 1.0, -1.0, -1.0, 1, 0, 0, 0],
    [ 1.0, -1.0, -1.0, 0, 1, 0, 0],
    [ 1.0, -1.0, -1.0, 0, 0, 1, 0],
    [ 1.0, -1.0,  0.0, 1, 0, 0, 0],
    [ 1.0, -1.0,  0.0, 0, 1, 0, 0],
    [ 1.0, -1.0,  0.0, 0, 0, 1, 0],
    [ 1.0, -1.0,  1.0, 1, 0, 0, 0],
    [ 1.0, -1.0,  1.0, 0, 1, 0, 0],
    [ 1.0, -1.0,  1.0, 0, 0, 1, 0],
    [ 1.0,  0.0, -1.0, 1, 0, 0, 0],
    [ 1.0,  0.0, -1.0, 0, 1, 0, 0],
    [ 1.0,  0.0, -1.0, 0, 0, 1, 0],
    [ 1.0,  0.0,  0.0, 1, 0, 0, 0],
    [ 1.0,  0.0,  0.0, 0, 1, 0, 0],
    [ 1.0,  0.0,  0.0, 0, 0, 1, 0],
    [ 1.0,  0.0,  1.0, 1, 0, 0, 0],
    [ 1.0,  0.0,  1.0, 0, 1, 0, 0],
    [ 1.0,  0.0,  1.0, 0, 0, 1, 0],
    [ 1.0,  1.0, -1.0, 1, 0, 0, 0],
    [ 1.0,  1.0, -1.0, 0, 1, 0, 0],
    [ 1.0,  1.0, -1.0, 0, 0, 1, 0],
    [ 1.0,  1.0,  0.0, 1, 0, 0, 0],
    [ 1.0,  1.0,  0.0, 0, 1, 0, 0],
    [ 1.0,  1.0,  0.0, 0, 0, 1, 0],
    [ 1.0,  1.0,  1.0, 1, 0, 0, 0],
    [ 1.0,  1.0,  1.0, 0, 1, 0, 0],
    [ 1.0,  1.0,  1.0, 0, 0, 1, 0],
])

LOCATION_NAMES = ["x", "y", "z", "q0", "q1", "q2", "q3"]

VALID_U = np.linspace(0, 1, len(VALID_LOCATIONS)).reshape(len(VALID_LOCATIONS), 1)
VALID_F = np.linspace(1, 2, 3*len(VALID_LOCATIONS)).reshape(len(VALID_LOCATIONS), 3)
VALID_T = np.linspace(2, 3, 3*len(VALID_LOCATIONS)).reshape(len(VALID_LOCATIONS), 3)

U_NAMES = ["U"]
F_NAMES = ["Fx", "Fy", "Fz"]
T_NAMES = ["Tx", "Ty", "Tz"]

REFERENCE_FOLDER = Path(__file__).parent / "data"

INVALID_RECARRAYS = [
    np.hstack((VALID_LOCATIONS, VALID_U)),          # wrong type
    np.rec.fromarrays(                              # missing 1 position column
        np.hstack((VALID_LOCATIONS[:,1:], VALID_U)).T,
        names=LOCATION_NAMES[1:] + U_NAMES
    ),
    np.rec.fromarrays(                              # missing all position columns
        np.hstack((VALID_LOCATIONS[:,3:], VALID_U)).T,
        names=LOCATION_NAMES[3:] + U_NAMES
    ),
    np.rec.fromarrays(                              # missing quantities
        VALID_LOCATIONS, names=LOCATION_NAMES
    ),
    np.rec.fromarrays(                              # incomplete quantities F
        np.hstack((VALID_LOCATIONS, VALID_F[:,1:])).T,
        names=LOCATION_NAMES + F_NAMES[1:]
    ),
    np.rec.fromarrays(                              # incomplete quantities T
        np.hstack((VALID_LOCATIONS, VALID_T[:,1:])).T,
        names=LOCATION_NAMES + T_NAMES[1:]
    ),
]


def recarray_from_quantities(quantities: list[str]) -> np.recarray:
    """Create a numpy recarray with locations and the given quantities."""
    arrays = []
    names = copy(LOCATION_NAMES)
    if "U" in quantities:
        arrays.append(VALID_U)
        names.extend(U_NAMES)
    if "F" in quantities:
        arrays.append(VALID_F)
        names.extend(F_NAMES)
    if "T" in quantities:
        arrays.append(VALID_T)
        names.extend(T_NAMES)

    return np.rec.fromarrays(
        np.hstack((VALID_LOCATIONS,*arrays)).T,
        names=names
    )

def recarrays_are_equal(r1, r2) -> bool:
    """Whether two recarrays have identical rows/records."""
    return (r1 == r2).view(np.ndarray).all()


@pytest.mark.parametrize("quantities", [
    ["U"],
    ["F"],
    ["T"],
    ["U", "F"],
    ["U", "T"],
    ["F", "T"],
    ["U", "F", "T"],
])
def test_instantiation_valid(quantities):
    """Ensure instantiation works from a valid recarray with varying quantities."""
    _ = p4.Field(recarray_from_quantities(quantities))

@pytest.mark.parametrize("recarray", INVALID_RECARRAYS)
def test_instantiation_invalid(recarray):
    """Ensure instantiation fails predictably for invalid recarrays."""
    with pytest.raises((ValueError, TypeError)):
        _ = p4.Field(recarray)

def test_from_csv():
    """Ensure creation from CSV produces the expected Field."""
    filename = REFERENCE_FOLDER / "field-uft.csv"
    f1 = p4.Field.from_csv(filename)
    f2 = p4.Field(recarray_from_quantities(["U", "F", "T"]))

    assert recarrays_are_equal(f1._table, f2._table)

def test_to_csv():
    """Ensure export to CSV produces the expected output."""
    with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
        test_path = Path(tempdir) / f"test_field-uft.json"
        
        p4.Field(recarray_from_quantities(["U", "F", "T"])).to_csv(test_path)

        with open(REFERENCE_FOLDER / "field-uft.csv", "r") as f:
            control_lines = f.readlines()

        with open(test_path, "r") as f:
            test_lines = f.readlines()

        assert control_lines == test_lines

@pytest.mark.parametrize("quantities", [
    ["U"],
    ["F"],
    ["T"],
    ["U", "F"],
    ["U", "T"],
    ["F", "T"],
    ["U", "F", "T"],
])
def test_properties(quantities):
    """Ensure properties work for fields with varying quantities."""
    # Test table getting and setting, columns, positions, orientations,
    # quantities
    recarray = recarray_from_quantities(quantities)
    f = p4.Field(recarray)

    assert recarrays_are_equal(f.table, copy(recarray))
    assert f.columns == list(recarray.dtype.fields.keys())
    assert np.array_equal(
        f.positions,
        np.unique(
            np.column_stack((
                recarray["x"],
                recarray["y"],
                recarray["z"]
            )),
            axis=0
        )
    )
    assert np.array_equal(
        f.orientations,
        np.unique(
            np.column_stack((
                recarray["q0"],
                recarray["q1"],
                recarray["q2"],
                recarray["q3"],
            )),
            axis=0
        )
    )
    correct_quantities = [i for i in recarray.dtype.fields.keys() if i not in LOCATION_NAMES]
    if all(n in correct_quantities for n in F_NAMES):
        correct_quantities.append("F")
    if all(n in correct_quantities for n in T_NAMES):
        correct_quantities.append("T")
    assert f.quantities == correct_quantities

def test_table_setter_valid():
    """Ensure the table can be set to a new recarray and other properties update accordingly."""
    f = p4.Field(recarray_from_quantities(["U"]))

    new_locations = np.vstack((VALID_LOCATIONS, np.array([[2, 2, 2, 0, 0, 1, 0]])))
    new_quantities = np.hstack((
        np.vstack((VALID_U, [[0]])),
        np.vstack((VALID_F, [[0, 0, 0]])),
        np.vstack((VALID_T, [[0, 0, 0]])),
    ))
    new_recarray = np.rec.fromarrays(
        np.hstack((new_locations, new_quantities)).T,
        names=LOCATION_NAMES + U_NAMES + F_NAMES + T_NAMES
    )

    f.table = new_recarray

    assert f.columns == LOCATION_NAMES + U_NAMES + F_NAMES + T_NAMES
    assert np.array_equal(f.positions, np.unique(new_locations[:, :3], axis=0))
    assert np.array_equal(f.orientations, np.unique(new_locations[:, 3:], axis=0))
    assert f.quantities == U_NAMES + F_NAMES + T_NAMES + ["F", "T"]

@pytest.mark.parametrize("recarray", INVALID_RECARRAYS)
def test_table_setter_invalid(recarray):
    """Ensure setting the table property fails predictably with invalid input."""
    f = p4.Field(recarray_from_quantities(["U"]))
    with pytest.raises((TypeError, ValueError)):
        f.table = recarray

@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize("slice", [
    dict(),                # 3D slice through a single orientation
    dict(x=-1),            # 2D slice through X
    dict(y=-1),            # 2D slice through X
    dict(z=-1),            # 2D slice through X
    dict(x=-1, y=-1),      # 1D slice through X and Y
    dict(x=-1, z=-1),      # 1D slice through X and Z
    dict(y=-1, z=-1),      # 1D slice through Y and Z
    dict(x=-1, y=-1, z=-1),# 0D slice, i.e., indexing (just for completeness) 
])
@pytest.mark.parametrize("quantity", ["U", "F", "T", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
def test_gridded_array_valid(quantity, vectors, slice):
    """Ensure gridded_array produces the expected result for multiple slices."""
    f = p4.Field(recarray_from_quantities(["U", "F", "T"]))

    q = (1, 0, 0, 0)
    q0, q1, q2, q3 = q
    sub_table = f.subset(q0=q0, q1=q1, q2=q2, q3=q3, **slice).table

    # 3D
    if len(slice) == 0:
        grid_xs = np.unique(VALID_LOCATIONS[:,0])
        grid_ys = np.unique(VALID_LOCATIONS[:,1])
        grid_zs = np.unique(VALID_LOCATIONS[:,2])
        if (quantity not in ("F", "T")) or (not vectors):
            shape = (len(grid_zs), len(grid_ys), len(grid_xs))
            correct_array = np.empty(shape)

            if quantity == "F":
                components = np.array([[row["Fx"], row["Fy"], row["Fz"]] for row in sub_table])
                magnitudes = np.sqrt(np.sum(np.square(components), axis=1))
            elif quantity == "T":
                components = np.array([[row["Tx"], row["Ty"], row["Tz"]] for row in sub_table])
                magnitudes = np.sqrt(np.sum(np.square(components), axis=1))

            for i, row in enumerate(sub_table):
                x_idx = grid_xs.tolist().index(row["x"])
                y_idx = grid_ys.tolist().index(row["y"])
                z_idx = grid_zs.tolist().index(row["z"])

                if quantity in ("F", "T"):
                    correct_array[z_idx, y_idx, x_idx] = magnitudes[i]
                else:
                    correct_array[z_idx, y_idx, x_idx] = row[quantity]
        
        else:
            shape = (len(grid_zs), len(grid_ys), len(grid_xs), 3)
            correct_array = np.empty(shape)
            
            if quantity == "F":
                components = np.array([[row["Fx"], row["Fy"], row["Fz"]] for row in sub_table])
            elif quantity == "T":
                components = np.array([[row["Tx"], row["Ty"], row["Tz"]] for row in sub_table])
            
            for i, row in enumerate(sub_table):
                x_idx = grid_xs.tolist().index(row["x"])
                y_idx = grid_ys.tolist().index(row["y"])
                z_idx = grid_zs.tolist().index(row["z"])
                correct_array[z_idx, y_idx, x_idx] = components[i]

    # 2D slices
    if len(slice) == 1:
        slice_dim = list(slice.keys())[0]
        if slice_dim == "x":
            grid_x_dim = "y"  # y along x axis
            grid_y_dim = "z"  # z along y axis
            grid_xs = np.unique(VALID_LOCATIONS[:,1])   
            grid_ys = np.unique(VALID_LOCATIONS[:,2])
        else:
            grid_x_dim = "x"
            grid_xs = np.unique(VALID_LOCATIONS[:,0])   # x along x axis
            if slice_dim == "y":
                grid_y_dim = "z"
                grid_ys = np.unique(VALID_LOCATIONS[:,2])   # z along y axis
            else:
                grid_y_dim = "y"
                grid_ys = np.unique(VALID_LOCATIONS[:,1])   # y along y axis
        
        if (quantity not in ("F", "T")) or (not vectors):
            shape = (len(grid_ys), len(grid_xs))
            correct_array = np.empty(shape)
            
            if quantity == "F":
                components = np.array([[row["Fx"], row["Fy"], row["Fz"]] for row in sub_table])
                magnitudes = np.sqrt(np.sum(np.square(components), axis=1))

            elif quantity == "T":
                components = np.array([[row["Tx"], row["Ty"], row["Tz"]] for row in sub_table])
                magnitudes = np.sqrt(np.sum(np.square(components), axis=1))

            for i, row in enumerate(sub_table):
                x_idx = grid_xs.tolist().index(row[grid_x_dim])
                y_idx = grid_ys.tolist().index(row[grid_y_dim])

                if quantity in ("F", "T"):
                    correct_array[y_idx, x_idx] = magnitudes[i]
                else:
                    correct_array[y_idx, x_idx] = row[quantity]
        
        else:
            shape = (len(grid_ys), len(grid_xs), 3)
            correct_array = np.empty(shape)
            
            if quantity == "F":
                components = np.array([[row["Fx"], row["Fy"], row["Fz"]] for row in sub_table])
            elif quantity == "T":
                components = np.array([[row["Tx"], row["Ty"], row["Tz"]] for row in sub_table])
            
            for i, row in enumerate(sub_table):
                x_idx = grid_xs.tolist().index(row[grid_x_dim])
                y_idx = grid_ys.tolist().index(row[grid_y_dim])
                correct_array[y_idx, x_idx] = components[i]

    # 1D slices
    elif len(slice) == 2:
        if (quantity not in ("F", "T")) or (not vectors):
            if quantity == "F":
                components = np.array([[row["Fx"], row["Fy"], row["Fz"]] for row in sub_table])
                correct_array = np.sqrt(np.sum(np.square(components), axis=1))
            elif quantity == "T":
                components = np.array([[row["Tx"], row["Ty"], row["Tz"]] for row in sub_table])
                correct_array = np.sqrt(np.sum(np.square(components), axis=1))
            else:
                correct_array = np.array([row[quantity] for row in sub_table])
        
        else:
            if quantity == "F":
                correct_array = np.array([[row["Fx"], row["Fy"], row["Fz"]] for row in sub_table])
            elif quantity == "T":
                correct_array = np.array([[row["Tx"], row["Ty"], row["Tz"]] for row in sub_table])

    # 0D slices
    elif len(slice) == 3:  
        if (quantity not in ("F", "T")) or (not vectors):
            if quantity == "F":
                components = [sub_table[0]["Fx"], sub_table[0]["Fy"], sub_table[0]["Fz"]]
                magnitude = np.sqrt(np.sum(np.square(components)))
                correct_array = np.array([magnitude])
            elif quantity == "T":
                components = [sub_table[0]["Tx"], sub_table[0]["Ty"], sub_table[0]["Tz"]]
                magnitude = np.sqrt(np.sum(np.square(components)))
                correct_array = np.array([magnitude])
            else:
                correct_array = np.array([sub_table[0][quantity]])
        
        else:
            if quantity == "F":
                correct_array = np.array([sub_table[0]["Fx"], sub_table[0]["Fy"], sub_table[0]["Fz"]])
            elif quantity == "T":
                correct_array = np.array([sub_table[0]["Tx"], sub_table[0]["Ty"], sub_table[0]["Tz"]])
        
    assert np.array_equal(
        correct_array,
        f.to_gridded_array(quantity=quantity, vectors=vectors, q=q, **slice)
    )

def test_gridded_array_invalid():
    """Ensure gridded_array fails expectedly for invalid input."""
    f = p4.Field(recarray_from_quantities(["U"]))

    # Multiple orientations but q is not specified
    with pytest.raises(ValueError):
        _ = f.to_gridded_array("U")

    # Incorrect q (not an iterable)
    with pytest.raises(TypeError):
        _ = f.to_gridded_array("U", q=1)

    # Incorrect q (elements are not ints or floats)
    with pytest.raises(TypeError):
        _ = f.to_gridded_array("U", q=("a", "b", "c"))
    
    # Incorrect q (not in the locations)
    with pytest.raises(ValueError):
        _ = f.to_gridded_array("U", q=(0.5, 0.5, 0, 0))
    
    # Incorrect quantity
    with pytest.raises(ValueError):
        _ = f.to_gridded_array("F", q=(1, 0, 0, 0))

@pytest.mark.parametrize("method", ["min", "max", "mean"])
def test_aggregate_over_orientations_valid(method):
    """Ensure aggregate_over_orientations produces the correct result."""
    recarray = recarray_from_quantities(["U", "F", "T"])
    f = p4.Field(recarray)

    for q in ["U", "F", "T"]:
        correct_records = []
        record_groups_by_position = [
            [
                np.array(row.tolist())
                for row in recarray
                if np.array_equal(position, np.array(row.tolist())[:3])
            ]
            for position in f.positions
        ]
        
        correct_records = []
        for group in record_groups_by_position:
            group = np.array(group)

            if q == "U":
                array_to_aggregate = group[:,7]
            elif q == "F":
                array_to_aggregate = group[:,8:11]
            elif q == "T":
                array_to_aggregate = group[:,11:]
            
            if method == "min":
                if q == "U":
                    i = np.argmin(array_to_aggregate)
                    aggregated_result = array_to_aggregate[i]
                
                elif q in ("F", "T"):
                    magnitudes = np.sqrt(np.sum(np.square(array_to_aggregate), axis=1))
                    i = np.argmin(magnitudes)
                    aggregated_result = array_to_aggregate[i, :]
            
            elif method == "max":
                if q == "U":
                    i = np.argmax(array_to_aggregate)
                    aggregated_result = array_to_aggregate[i]

                elif q in ("F", "T"):
                    magnitudes = np.sqrt(np.sum(np.square(array_to_aggregate), axis=1))
                    i = np.argmax(magnitudes)
                    aggregated_result = array_to_aggregate[i, :]
            
            elif method == "mean":
                aggregated_result = np.mean(array_to_aggregate, axis=0)

            correct_records.append(np.hstack((group[0,:3], aggregated_result)))
        
        correct_names = ["x", "y", "z"]
        if q == "U":
            correct_names += ["U"]
        elif q == "F":
            correct_names += ["Fx", "Fy", "Fz"]
        elif q == "T":
            correct_names += ["Tx", "Ty", "Tz"]

        correct_recarray = np.rec.fromrecords(correct_records, names=correct_names)

    assert recarrays_are_equal(correct_recarray, f.aggregate_over_orientations(q, method).table)

def test_aggregate_over_orientations_invalid():
    """Ensure aggregate_over_orientations fails predictably with invalid input."""
    recarray = recarray_from_quantities(["U", "F"])
    f = p4.Field(copy(recarray))

    # No orientations
    f_no_orientations = f.aggregate_over_orientations("U", "min")
    with pytest.raises(ValueError):
        _ = f_no_orientations.aggregate_over_orientations("U", "min")

    # Missing quantity
    with pytest.raises(ValueError):
        _ = f.aggregate_over_orientations("T", "min")
    
    # Unrecognized method
    with pytest.raises(ValueError):
        _ = f.aggregate_over_orientations("U", "wrong")
    
    # Invalid quantity
    with pytest.raises(ValueError):
        _ = f.aggregate_over_orientations("Fx", "min")

def test_operations_valid():
    """Ensure addition and subtraction work when locations are identical."""
    recarray = recarray_from_quantities(["U", "F", "T"])
    f1 = p4.Field(copy(recarray))
    f2 = p4.Field(copy(recarray))

    # Test addition
    correct_recarray = np.rec.fromrecords(
        np.hstack((VALID_LOCATIONS, VALID_U*2, VALID_F*2, VALID_T*2)).T,
        names=LOCATION_NAMES + U_NAMES + F_NAMES + T_NAMES
    )
    assert p4.Field(correct_recarray) == f1 + f2

    # test subtraction
    correct_recarray = np.rec.fromrecords(
        np.hstack((VALID_LOCATIONS, VALID_U*0, VALID_F*0, VALID_T*0)).T,
        names=LOCATION_NAMES + U_NAMES + F_NAMES + T_NAMES
    )
    assert p4.Field(correct_recarray) == f1 - f2

def test_operations_invalid():
    """Ensure addition and subtraction fail expectedly when locations and quantities are different."""
    recarray = recarray_from_quantities(["U"])
    f1 = p4.Field(copy(recarray))

    # Different locations
    new_locations = np.vstack((VALID_LOCATIONS, np.array([[2, 2, 2, 0, 0, 1, 0]])))
    new_quantities = np.hstack((
        np.vstack((VALID_U, [[0]])),
    ))
    new_recarray = np.rec.fromarrays(
        np.hstack((new_locations, new_quantities)).T,
        names=LOCATION_NAMES + U_NAMES
    )

    f2 = p4.Field(new_recarray)

    with pytest.raises(ValueError):
        _ = f1 + f2
    with pytest.raises(ValueError):
        _ = f1 - f2
    
    # Different quantities
    new_recarray = recarray_from_quantities(["U", "F", "T"])

    f2 = p4.Field(new_recarray)

    with pytest.raises(ValueError):
        _ = f1 + f2
    with pytest.raises(ValueError):
        _ = f1 - f2

    # Different locations and quantities
    new_locations = np.vstack((VALID_LOCATIONS, np.array([[2, 2, 2, 0, 0, 1, 0]])))
    new_quantities = np.hstack((
        np.vstack((VALID_U, [[0]])),
        np.vstack((VALID_F, [[0, 0, 0]])),
        np.vstack((VALID_T, [[0, 0, 0]])),
    ))
    new_recarray = np.rec.fromarrays(
        np.hstack((new_locations, new_quantities)).T,
        names=LOCATION_NAMES + U_NAMES + F_NAMES + T_NAMES
    )

    f2 = p4.Field(new_recarray)

    with pytest.raises(ValueError):
        _ = f1 + f2
    with pytest.raises(ValueError):
        _ = f1 - f2

def test_subset_valid():
    """Ensure subset works for valid arguments."""
    f = p4.Field(recarray_from_quantities(["U"]))

    # Test every argument singly and with multiple values
    kwargs = LOCATION_NAMES
    for variant in ["single", "multiple"]:
        for i, kwarg in enumerate(kwargs):
            values = list(np.unique(f.table[kwarg]))
            
            if variant == "single":
                correct_values = [values[0]]    
            elif variant == "multiple":
                correct_values = values[:2]

            correct_records = []
            for j, row in enumerate(VALID_LOCATIONS):
                if row[i] in correct_values:
                    correct_records.append(np.hstack((row, VALID_U[j]))) 
            correct_recarray = np.rec.fromrecords(correct_records, names=LOCATION_NAMES + U_NAMES)
            correct_sub_f = p4.Field(correct_recarray)

            kwarg_dict = {f"{kwarg}": correct_values[0] if variant == "single" else correct_values}
            test_sub_f = f.subset(**kwarg_dict)

            assert correct_sub_f == test_sub_f

    # Test two pairs of arguments (assume that all other pairs and groupings
    # will work if two pairs do)
    for kwarg1, kwarg2 in [("x", "z"), ("x", "q2")]:
        for variant in ["single", "multiple"]:
            values = [
                list(np.unique(f.table[kwarg1])),
                list(np.unique(f.table[kwarg2])),
            ]
            
            if variant == "single":
                correct_values = [[v[0]] for v in values]
            elif variant == "multiple":
                correct_values = [v[:2] for v in values]

            correct_records = []
            i, j = kwargs.index(kwarg1), kwargs.index(kwarg2)
            for k, row in enumerate(VALID_LOCATIONS):
                if row[i] in correct_values[0] and row[j] in correct_values[1]:
                    correct_records.append(np.hstack((row, VALID_U[k]))) 
            correct_recarray = np.rec.fromrecords(correct_records, names=LOCATION_NAMES + U_NAMES)
            correct_sub_f = p4.Field(correct_recarray)

            kwarg_dict = {
                f"{kwarg1}": correct_values[0][0] if variant == "single" else correct_values[0],
                f"{kwarg2}": correct_values[1][0] if variant == "single" else correct_values[1],
            }
            assert correct_sub_f == f.subset(**kwarg_dict)

def test_subset_invalid():
    """Ensure subset fails predictably for invalid arguments."""
    f = p4.Field(recarray_from_quantities(["U"]))
    with pytest.raises(ValueError):
        _ = f.subset(a=None)

def test_equality():
    """Ensure comparison returns True for fields with identical tables/recarrays."""
    recarray = recarray_from_quantities(["U", "F", "T"])
    f1 = p4.Field(copy(recarray))
    f2 = p4.Field(copy(recarray))
    assert f1 == f2
