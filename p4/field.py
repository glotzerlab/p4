# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.


from __future__ import annotations
from inspect import signature
import numbers
import os
from types import NoneType
from typing import Iterable, Literal
import csv

import numpy as np
from numpy.lib import recfunctions
import plotly
import plotly.figure_factory

from .top_level_functions import plot_layout
from . import util


class Field:
    """Analyze and plot scalar and vector fields in 3D, 2D, and 1D.
    
    This class is built the NumPy `recarray`_, which is a lightweight tabular
    data structure similar to the Pandas `DataFrame`_.

    .. _recarray: https://numpy.org/doc/stable/reference/generated/numpy.recarray.html
    .. _DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

    Access a Field's tabular data using the :py:attr:`~p4.field.Field.table`
    property, which returns the underlying recarray.
    
    Instantiate a Field directly from a recarray, or build one from CSV using
    :py:meth:`~p4.field.Field.from_csv`. In order for instantiation to work,
    the tabular data must have at least 'x', 'y', and 'z' columns to indicate
    position, and at least one of the following sets of columns to indicate
    measured quantities:

    * 'U' - potential energy
    * 'Fx', 'Fy', and 'Fz' - force
    * 'Tx', 'Ty', and 'Tz' - torque

    Optionally, the tabular data may also include orientation columns; if
    included, these must be exactly 'q0', 'q1', 'q2', and 'q3', which
    correspond to the components of a quaternion.

    If there are no orientation columns, or if there is only one orientation
    across the dataset, then the measured quantities may be plotted immediately,
    otherwise it is necessary to first call
    :meth:`~p4.Field.aggregate_over_orientations`.

    Plotting is done using `Plotly`_, and the :meth:`~p4.Field.plot` method
    returns both figure and trace so that the trace may be added manually to
    other figures. Orthogonal slicing in 2D and 1D is supported for scalar
    plots, and in 2D for vector plots.

    .. _Plotly: https://plotly.com

    Parameters
    ----------
    recarray : np.rec.recarray
        The underlying recarray.
    
    Raises
    ------
    TypeError
        If the provided array of an incorrect type
    ValueError
        If the columns do not contain the required names.
    """
    def __init__(self, recarray: np.rec.recarray):
        if type(recarray) != np.rec.recarray:
            raise TypeError("The array must be an instance of np.rec.recarray.")
        
        columns = set(list(recarray.dtype.fields.keys()))
        if {"x", "y", "z"} - columns != set():
            raise ValueError("The array must have columns 'x', 'y', and 'z'.")
        
        if (
            ("U" not in columns)
            and ({"Fx", "Fy", "Fz"} - columns != set())
            and ({"Tx", "Ty", "Tz"} - columns != set())
        ):
            raise ValueError(
                "The array must have columns that contain either "
                + "'U', or all of 'Fx', 'Fy', and 'Fz', or all of 'Tx', 'Ty', "
                + "and 'Tz'."
            )
        
        self._positions = None
        self._orientations = None
        self._table = recarray

    # ------------------------------- PROPERTIES -------------------------------

    @property
    def table(self) -> np.rec.recarray:
        """The tabular dataset, returned as a record array."""
        return self._table
    
    @table.setter
    def table(self, recarray: np.rec.recarray):
        """Set the table equal to a new record array."""
        if type(recarray) != np.rec.recarray:
            raise TypeError("The array must be an instance of np.rec.recarray.")
        
        columns = set(list(recarray.dtype.fields.keys()))
        if {"x", "y", "z"} - columns != set():
            raise ValueError("The array must have columns 'x', 'y', and 'z'.")
        
        if (
            ("U" not in columns)
            and ({"Fx", "Fy", "Fz"} - columns != set())
            and ({"Tx", "Ty", "Tz"} - columns != set())
        ):
            raise ValueError(
                "The array must have columns that contain either "
                + "'U', or all of 'Fx', 'Fy', and 'Fz', or all of 'Tx', 'Ty', "
                + "and 'Tz'."
            )
        
        self._positions = None
        self._orientations = None
        self._table = recarray

    @property
    def columns(self) -> list[str]:
        """The names of the columns in the table."""
        return list(self.table.dtype.fields.keys())

    @property
    def positions(self) -> np.ndarray:
        """The unique positions."""
        if self._positions is None:
            self._positions = np.unique(
                np.column_stack((
                    self.table["x"],
                    self.table["y"],
                    self.table["z"]
                )),
                axis=0
            )
        
        return self._positions

    @property
    def orientations(self) -> np.ndarray | None:
        """The unique orientations.
        
        If the columns do not include all of 'q0', 'q1', 'q2', and 'q3', None is
        returned.
        """
        columns = set(self.table.dtype.fields.keys())
        if {"q0", "q1", "q2", "q3"} - columns != set():
            return None
        
        else:
            return np.unique(
                np.column_stack((
                    self.table["q0"],
                    self.table["q1"],
                    self.table["q2"],
                    self.table["q3"]
                )),
                axis=0
            )
    
    @property
    def quantities(self) -> list[str]:
        """The measured quantities."""
        location_columns = ["x", "y", "z", "q0", "q1", "q2", "q3"]
        quantities = [f for f in self.columns if f not in location_columns]
        
        # Include F and T shorthand names
        force_names = ["Fx", "Fy", "Fz"]
        torque_names = ["Tx", "Ty", "Tz"]
        
        if all(n in quantities for n in force_names):
            quantities.append("F")
        if all(n in quantities for n in torque_names):
            quantities.append("T")
        
        return quantities

    # def has_regular_grid(self) -> bool:
        # """Whether this Field's grid has constant intervals along each axis."""
        # positions = self.positions
        
        # x = np.unique(positions[:,0])
        # y = np.unique(positions[:,1])
        # z = np.unique(positions[:,2])
        
        # regular_x = np.nonzero(np.diff(x, n=2))[0].size == 0
        # regular_y = np.nonzero(np.diff(y, n=2))[1].size == 0
        # regular_z = np.nonzero(np.diff(z, n=2))[2].size == 0

        # return (regular_x and regular_y and regular_z)

    # ------------------------------- OPERATIONS -------------------------------

    def aggregate_over_orientations(
        self,
        quantity: Literal["U", "F", "T"],
        method: Literal["min", "max", "mean"],
    ) -> Field:
        """Aggregate one quantity over all orientations using a chosen method.
        
        The return value is a new ``Field`` with either 1 measured quantity
        (if vectors is False), or 3 measured quantities (if vectors is True).

        Methods:

        * **min**

          * **U**: take the minimum value
          * **F** or **T**: take the components with the minimum magnitude

        * **max**
        
          * **U**: take the maximum value
          * **F** or **T**: take the components with the maximum magnitude

        * **mean**
        
          * **U**: calculate the mean value
          * **F** or **T**: calculate the mean of each component

        :meta operation:

        Parameters
        ----------
        quantity : 'U', 'F', or 'T'
            The name of the quantity whose values will be aggregated.
        method : 'min', 'max', or 'mean'
            The method to use when aggregating the quantity. At each position,
            the values of the quantity are grouped together and this method is
            used to calculate a new value for that position.
        """
        # Ensure there are orientations
        if self.orientations is None:
            raise ValueError("Cannot aggregate if there are no orientations.")

        # Ensure the quantity is represented in this field
        if quantity not in self.quantities:
            raise ValueError(
                "`quantity` is not one of this field's quantities "
                f"({self.quantities})."
            )
        
        # Ensure the method is recognized
        if method not in ["min", "max", "mean"]:
            raise ValueError("`method` must be 'min', 'max', or 'mean'.")

        # Ensure the quantity is constrained to U, F, and T
        if quantity not in ["U", "F", "T"]:
            raise ValueError("`quantity` must be 'U', 'F', or 'T'.")

        # Unique positions serve as group keys
        positions = np.column_stack((
            self.table["x"],
            self.table["y"],
            self.table["z"]
        ))
        positions, group_ids = np.unique(positions, axis=0, return_inverse=True)

        # Calculate indices of the group boundaries
        ends = np.r_[
            np.flatnonzero(group_ids[1:] != group_ids[:-1]),
            len(group_ids) - 1
        ]
        starts = np.r_[0, ends[:-1] + 1]

        # Values are chosen by the user
        if quantity == "U":
            values = self.table["U"]
        
        elif quantity == "F":
            components = np.column_stack((
                self.table["Fx"],
                self.table["Fy"],
                self.table["Fz"]
            ))
            magnitudes = np.sqrt(np.sum(np.square(components), axis=1))
            values = np.hstack((components, magnitudes[:, np.newaxis]))
        
        elif quantity == "T":
            components = np.column_stack((
                self.table["Tx"],
                self.table["Ty"],
                self.table["Tz"]
            ))
            magnitudes = np.sqrt(np.sum(np.square(components), axis=1))
            values = np.hstack((components, magnitudes[:, np.newaxis]))
        
        # Build the tall aggregated array group by group
        if quantity == "U":
            agg_values = np.empty((len(starts), 1), dtype=float)
        else:
            agg_values = np.empty((len(starts), 3), dtype=float)

        for i, (start_index, end_index) in enumerate(zip(starts, ends)):
            if end_index == len(values) - 1:
                group_values = values[start_index:]
            else:
                group_values = values[start_index:(end_index+1)]
            
            if method == "min":
                if quantity == "U":   # min outright
                    agg_values[i] = group_values.min()
                else: # min magnitude
                    index = np.argmin(group_values[:,-1])
                    agg_values[i, :] = group_values[index, :-1]
            
            elif method == "max":
                if quantity == "U":   # max outright
                    agg_values[i] = group_values.max()
                else: # max magnitude
                    index = np.argmax(group_values[:,-1])
                    agg_values[i, :] = group_values[index, :-1]
            
            elif method == "mean":
                if quantity == "U":
                    agg_values[i] = group_values.mean()
                else:
                    mean_components = group_values[:,:-1].mean(axis=0)
                    agg_values[i, :] = mean_components

        locations = np.column_stack((
            self.table["x"][starts],
            self.table["y"][starts],
            self.table["z"][starts],
        ))
        agg_tall = np.hstack((locations, agg_values))

        location_columns = ["x", "y", "z"]
        if quantity == "U":
            new_columns = location_columns + ["U"]
        elif quantity == "F":
            new_columns = location_columns + ["Fx", "Fy", "Fz"]
        elif quantity == "T":
            new_columns = location_columns + ["Tx", "Ty", "Tz"]
        
        agg_tall = recfunctions.unstructured_to_structured(
            agg_tall,
            names=new_columns
        )

        return Field(agg_tall.view(np.recarray))

    # def resample(
    #     self,
    #     resolutions: list[int],
    #     # method: Literal[]
    # ) -> Field:
    #     pass

    # def fill_gaps_with_nan(self) -> Field:
    #     pass

    def subset(self, **kwargs) -> Field:
        """Return a new Field that is a subset of the current one.
        
        :meta operation:

        Parameters
        ----------
        x : float or array of floats, optional
            Restrict the new Field to these ``x`` values.
        y : float or array of floats, optional
            Restrict the new Field to these ``y`` values.
        z : float or array of floats, optional
            Restrict the new Field to these ``z`` values.
        q0 : float or array of floats, optional
            Restrict the new Field to these ``q0`` values.
        q1 : float or array of floats, optional
            Restrict the new Field to these ``q1`` values.
        q2 : float or array of floats, optional
            Restrict the new Field to these ``q2`` values.
        q3 : float or array of floats, optional
            Restrict the new Field to these ``q3`` values.
        
        Returns
        -------
        field
            The new Field.

        Raises
        ------
        ValueError
            If an unrecognized kwarg is provided.
        """
        recognized = {"x", "y", "z", "q0", "q1", "q2", "q3"}
        unrecognized = set(kwargs) - recognized
        if unrecognized:
            raise ValueError(f"Unrecognized columns: {unrecognized}.")
        
        return Field(self._subset_of_recarray(**kwargs))

    def _subset_of_recarray(self, **kwargs) -> np.rec.recarray:
        """Return a new recarray that is a subset of the current one.
        
        Kwargs
        ------
        x : float or array of floats, optional
            Restrict the new Field to these ``x`` values.
        y : float or array of floats, optional
            Restrict the new Field to these ``y`` values.
        z : float or array of floats, optional
            Restrict the new Field to these ``z`` values.
        q0 : float or array of floats, optional
            Restrict the new Field to these ``q0`` values.
        q1 : float or array of floats, optional
            Restrict the new Field to these ``q1`` values.
        q2 : float or array of floats, optional
            Restrict the new Field to these ``q2`` values.
        q3 : float or array of floats, optional
            Restrict the new Field to these ``q3`` values.
        
        Returns
        -------
        np.recarray
            The new recarray.

        Raises
        ------
        ValueError
            If an unrecognized kwarg is provided.
        """
        recognized = {"x", "y", "z", "q0", "q1", "q2", "q3"}
        unrecognized = set(kwargs) - recognized
        if unrecognized:
            raise ValueError(f"Unrecognized columns: {unrecognized}.")
        
        conditions = []
        for column, values in kwargs.items():
            if values is None:
                continue
            if not isinstance(values, Iterable):
                values = [values]
            subconditions = []
            for v in values:
                subconditions.append(self.table[column] == v)
            
            conditions.append(np.logical_or.reduce(subconditions))
        
        results = np.logical_and.reduce(conditions)

        return self.table[results]
    
    def __sub__(self, other: Field) -> Field:
        """Calculate the difference between two fields' measured quantities.
        
        Raises
        ------
        ValueError
            If the other Field has a different sampling grid or quantities.
        """
        if set(self.columns) != set(other.columns):
            raise ValueError(
                "Cannot calculate difference between two Fields with different "
                + "columns."
            )

        if self.orientations is not None:
            location_columns = ["x", "y", "z", "q0", "q1", "q2", "q3"]
        else:
            location_columns = ["x", "y", "z"]

        self_locations = np.column_stack([
            self.table[name] for name in location_columns
        ])
        other_locations = np.column_stack([
            other.table[name] for name in location_columns
        ])

        if not np.array_equal(self_locations, other_locations):
            raise ValueError(
                "Cannot calculate difference between two Fields with different "
                + "measurement locations."
            )
        
        self_quantity_names = [
            name for name in self.quantities if name in self.columns
        ]
        other_quantity_names = [
            name for name in other.quantities if name in other.columns
        ]
        
        if not sorted(self_quantity_names) == sorted(other_quantity_names):
            raise ValueError(
                "Cannot calculate difference between two Fields with different "
                + "measured quantities."
            )

        self_quantities = np.column_stack([
            self.table[name] for name in self_quantity_names
        ])
        other_quantities = np.column_stack([    # *self* ensures same order
            other.table[name] for name in self_quantity_names
        ])

        new_quantities = np.array(self_quantities) - np.array(other_quantities)

        new_recarray = np.rec.fromarrays(
            np.hstack((self_locations, new_quantities)),
            names=location_columns + self_quantity_names
        )

        return Field(new_recarray)

    def __add__(self, other: Field) -> Field:
        """Calculate the sum of two fields' measured quantities.
        
        Raises
        ------
        ValueError
            If the other Field has a different sampling grid or quantities.
        """
        if set(self.columns) != set(other.columns):
            raise ValueError(
                "Cannot calculate difference between two Fields with different "
                + "columns."
            )

        if self.orientations is not None:
            location_columns = ["x", "y", "z", "q0", "q1", "q2", "q3"]
        else:
            location_columns = ["x", "y", "z"]

        self_locations = np.column_stack([
            self.table[name] for name in location_columns
        ])
        other_locations = np.column_stack([
            other.table[name] for name in location_columns
        ])

        if not np.array_equal(self_locations, other_locations):
            raise ValueError(
                "Cannot calculate difference between two Fields with different "
                + "measurement locations."
            )
        
        self_quantity_names = [
            name for name in self.quantities if name in self.columns
        ]
        other_quantity_names = [
            name for name in other.quantities if name in other.columns
        ]
        
        if not sorted(self_quantity_names) == sorted(other_quantity_names):
            raise ValueError(
                "Cannot calculate difference between two Fields with different "
                + "measured quantities."
            )

        self_quantities = np.column_stack([
            self.table[name] for name in self_quantity_names
        ])
        other_quantities = np.column_stack([    # *self* ensures same order
            other.table[name] for name in self_quantity_names
        ])

        new_quantities = np.array(self_quantities) + np.array(other_quantities)

        new_recarray = np.rec.fromarrays(
            np.hstack((self_locations, new_quantities)),
            names=location_columns + self_quantity_names
        )

        return Field(new_recarray)

    # ---------------------------------- FROM ----------------------------------

    @classmethod
    def from_csv(cls, filename: os.PathLike):
        """Create a Field from a CSV file."""
        with open(filename) as f:
            columns = f.readline().strip("\n").split(",")

        return cls(
            np.rec.array(
                np.genfromtxt(
                    filename,
                    names=columns,
                    skip_header=1,
                    dtype=None,
                    delimiter=","
                )
            )
        )

    # ----------------------------------- TO -----------------------------------

    def to_csv(self, filename: os.PathLike):
        """Save this field to CSV."""
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(self.table.dtype.names)
            writer.writerows(self.table.tolist())

    def to_gridded_array(
        self,
        quantity: Literal["U", "F", "T", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        vectors: bool = False,
        q: tuple[float] | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> np.ndarray:
        """Return a gridded array of a given scalar or vector quantity.

        The shape of the returned array depends on the measurement locations,
        the slice, and whether vectors is True. In general, for a Field with
        ``nx`` x positions, ``ny`` y positions, and ``nz`` z positions, the
        shape of the returned array is as follows.

        .. code-block::
            
            * vectors = False
                * slice along x           -> (nz, ny)
                * slice along y           -> (nz, nx)
                * slice along z           -> (ny, nx)
                * slice along x and y     -> (nz,)
                * slice along x and z     -> (ny,)
                * slice along y and z     -> (nx,)
                * slice along x, y, and z -> (1,)

            * vectors = True
                * slice along x           -> (nz, ny, 3)
                * slice along y           -> (nz, nx, 3)
                * slice along z           -> (ny, nx, 3)
                * slice along x and y     -> (nz, 3)
                * slice along x and z     -> (ny, 3)
                * slice along y and z     -> (nx, 3)
                * slice along x, y, and z -> (3,)

        Parameters
        ----------
        quantity : 'U', 'F', 'T', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', or 'Tz'
            The name of the quantity with which to populate the cells of the
            gridded array.
        vectors : bool, default=False
            Whether to populate the gridded array with scalars or vectors. If
            `quantity` is 'U', only scalar values are possible and this
            parameter is ignored.
        q : tuple of 4 floats, optional
            If provided, only rows with the corresponding q0, q1, q2, and q3
            values will be used to create the gridded array. Required if
            ``table`` includes multiple orientations.
        x : float, optional
            If provided, only rows with the corresponding x values will be used
            to create the gridded array.
        y : float, optional
            If provided, only rows with the corresponding y values will be used
            to create the gridded array.
        z : float, optional
            If provided, only rows with the corresponding z values will be used
            to create the gridded array.

        Returns
        -------
        np.ndarray
            The gridded array.

        Raises
        ------
        ValueError
            If ``q`` is required but not provided. 
        TypeError
            If ``q`` is not an iterable of 4 floats.
        ValueError
            If ``quantity`` is not recognized.
        """
        # [Review: get closest values of x, y, z?]
        # Input Validation
        if self.orientations is not None and self.orientations.shape[0] > 1 and q is None:
            raise ValueError(
                "The table has more than one orientation. Choose one "
                + "using `q` or call `aggregate_over_orientations()`."
            )
        
        if (
            self.orientations is not None and self.orientations.shape[0] > 1
            and (
                not isinstance(q, Iterable)
                or len(q) != 4
                or any(not isinstance(i, (float, int)) for i in q)
            )
        ):
            raise TypeError("`q` must be an array of 4 floats.")
        
        if quantity not in self.quantities:
            raise ValueError(
                "`quantity` is not one of this field's quantities "
                f"({self.quantities})."
            )

        # Build the recarray
        if q is not None:
            table = self._subset_of_recarray(
                x=x, y=y, z=z, q0=q[0], q1=q[1], q2=q[2], q3=q[3]
            )
        else:
            table = self._subset_of_recarray(x=x, y=y, z=z)

        if len(table) == 0:
            raise ValueError(
                "No measurements found for the given x, y, z, and q."
            )

        if quantity == "U":
            columns_to_drop = [f for f in self.quantities if f != "U"]
        elif quantity == "F":
            columns_to_drop = [f for f in self.quantities if "F" not in f]
        elif quantity == "T":
            columns_to_drop = [f for f in self.quantities if "T" not in f]
        else:
            columns_to_drop = [f for f in self.quantities if f != quantity]
        
        table = recfunctions.drop_fields(
            table, columns_to_drop, usemask=False, asrecarray=True
        )

        # If vectors is False, convert F and T components into magnitudes
        if not vectors:
            if quantity in ("F", "T"):
                component_columns = [
                    f"{quantity}x",
                    f"{quantity}y",
                    f"{quantity}z",
                ]
                components = np.column_stack(
                    [table[f].flatten() for f in component_columns]
                )
                magnitudes = np.sqrt(np.sum(np.square(components), axis=1))

                columns_to_drop = [f for f in self.quantities if f != quantity]
                table = recfunctions.drop_fields(
                    table, columns_to_drop, usemask=False, asrecarray=True
                )
                table = recfunctions.append_fields(
                    table, quantity, magnitudes, usemask=False, asrecarray=True
                )
        
        # Create a gridded array and orient it to respect the conventions.
        gridded_array = np.atleast_1d(
            self._recarray_to_gridded_array(table).squeeze()
        )

        return gridded_array

    def _recarray_to_gridded_array(
        self,
        recarray: np.rec.recarray
    ) -> np.ndarray:
        """Convert a recarray like a table into a gridded array for plotting.
        
        It is assumed that either the recarray has a 1 measured quantity,
        or that it has 3 with that correspond to qx, qy, qz, where q is either
        F or T.
        
        If the recarray has a single measured quantity, the gridded array will
        be a 3D array of scalars (Z along axis 0, Y along axis 1, X along axis
        2).

        If the recarray has three measured quantities, the gridded array will
        be 4D (Z along axis 0, Y along axis 1, X along axis 2, q-components
        along axis 3 in qx, qy, qz order).

        Parameters
        ----------
        table : np.recarray
            The recarray to convert.
        
        Returns
        -------
        np.ndarray
            The gridded array.
        
        Raises
        ------
        ValueError
            If `recarray` has neither 1 nor 3 measured quantities.
        """
        location_columns = ["x", "y", "z", "q0", "q1", "q2", "q3"]
        columns = recarray.dtype.names
        measured_quantities = [f for f in columns if f not in location_columns]

        x_values, x_indices = np.unique(recarray["x"], return_inverse=True)
        y_values, y_indices = np.unique(recarray["y"], return_inverse=True)
        z_values, z_indices = np.unique(recarray["z"], return_inverse=True)
        
        if len(measured_quantities) == 1:
            q = measured_quantities[0]
            shape = (len(z_values), len(y_values), len(x_values))
            gridded_array = np.full(shape, np.nan)
            gridded_array[z_indices, y_indices, x_indices] = recarray[q]

        elif len(measured_quantities) == 3:
            shape = (len(z_values), len(y_values), len(x_values), 3)
            gridded_array = np.full(shape, np.nan)
            components = np.column_stack(
                [recarray[q] for q in measured_quantities]
            )
            gridded_array[z_indices, y_indices, x_indices] = components

        else:
            raise ValueError(
                "`recarray` must have either 1 or 3 measured quantities."
            )

        return gridded_array

    # -------------------------------- PLOTTING --------------------------------

    def plot(
        self,
        quantity: Literal["U", "F", "T", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"] | None = None,
        vectors: bool = False,
        slice: dict[str, float] | None = None,
        clim: list[float] | None = None,
        contours: int | None = 10,
        cmap: str = "RdYlBu_r",
        fill_nan_with_inf: bool = False,
        show_cbar: bool = True,
        marker_mode_1d: Literal["lines+markers", "lines", "markers"] = "lines",
        marker_color_1d: str = "black",
        marker_size_1d: float = 6,
        line_width_1d: float = 2,
        ylim_1d: list[float] | None = None,
        opacity_scalar_3d: float = 0.2,
        opacityscale_scalar_3d: float | str = "uniform",
        **kwargs
    ) -> tuple[plotly.graph_objects.Figure, list]:
        """Interactively plot the field using `Plotly`_.
        
        Slicing is supported along the X, Y, and Z axes via the ``slice``
        parameter. A slice along one axis (``slice={"x": 1}``) is 2D, while a
        slice along two axes (``slice={"x": 1, "y": 1}``) is 1D.

        Any of the measured quantities (including individual components of
        Forces or Torques) can be plotted. Set ``quantity`` to 'F' or 'T' to
        plot Force and Torque magnitudes as scalar quantities. Set ``vectors``
        to ``True`` to plot Forces or Torques as vectors. Vector plotting is
        only available in 2D and 3D.
        
        Parameters
        ----------
        quantity : 'U', 'F', 'T', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', or 'Tz'
            The name of the quantity to plot. If this field only contains one of
            'U' or 'F' or 'T' quantities, this parameter is optional.
        vectors : bool, default=False
            Whether to plot the quantity as a scalar or vector. If ``quantity``
            is not 'F' or 'T', this is always False.
        slice : dict, optional
            Axes and positions along which to slice. Keys are limited to 'x',
            'y', and 'z'. There can be at most two keys. If the slice contains
            a position that does not exactly match the grid, the nearest grid
            position will be used.
        clim : list[float], optional
            The lower and upper limits of the colorscale. If not provided,
            the lower and upper limits will be set to the 0th and 90th
            percentile values, respectively.
        contours : int or None, default=10
            The number of values to draw contours around. Only used in 3D and
            2D scalar plots. In 2D, pass None to instead use a continuous
            colorscale.
        cmap : str, default='RdYlBu_r'
            The name of the Plotly colormap to use. Only used in 3D and 2D
            scalar plots and 3D vector plots.
        fill_nan_with_inf : bool, default=False
            Whether to plot NaN values as though they were very large values.
            Only used in 3D and 2D scalar plots.
        show_cbar : bool, default=True
            Whether to show the colorbar.
        marker_mode_1d : 'lines+markers', 'lines', or 'markers', default='lines'
            In a 1D scalar plot, whether to show only lines, only markers, or
            both. Ignored for all other plot types.
        marker_color_1d : str, default='black'
            In a 1D scalar plot, the color of the plot symbol. Ignored for all
            other plot types.
        marker_size_1d : float, default=6
            In a 1D scalar plot, the size of the marker in pixels.
        line_width_1d : float, default=2
            In a 1D scalar plot, the width of the line in pixels.
        ylim_1d : list[float], optional
            In a 1D scalar plot, the limits of the y axis. Defaults to the
            0th and 90th percentile of the plotted quantity.
        opaopacity_scalar_3dcity : float, default=0.2
            In a 3D scalar plot, the opacity of the surface. Passed directly to
            ``plotly.graph_objects.Volume()``.
        opacityscale_scalar_3d : float | str, default='uniform'
            In a 3D scalar plot, the mapping between data values and opacity
            values. Specified as either an array of value pairs or as a string
            referring to the name of a preset scale. Passed directly to
            ``plotly.graph_objects.Volume()``.
        **kwargs
            Other keyword arguments are passed to ``p4.plot_layout()``.
            TODO: add link.
        
        Returns
        -------
        figure, traces
            The Plotly figure and associated traces.
        """
        # Default slice
        if not slice:
            slice = {}

        # Ensure there is no ambiguity around orientations
        if self.orientations is not None and self.orientations.shape[0] > 1:
            raise ValueError(
                "Cannot plot field with more than one orientation. "
                + "Aggregate over orientations."
            )
        
        # Ensure that there is no ambguity around quantity
        if (
            quantity is None
            and (
                set(self.quantities) != {"U"}
                and set(self.quantities) != {"F", "Fx", "Fy", "Fz"}
                and set(self.quantities) != {"T", "Tx", "Ty", "Tz"}
            )
        ):
            raise ValueError(
                "Plot quantity is not inferrable. Specify one of the "
                + f"following: {self.quantities}"
            )
        
        # Ensure that the requested quantity is actually present
        if (quantity is not None) and (quantity not in self.quantities):
            raise ValueError(
                "`quantity` is not one of this field's quantities "
                f"({self.quantities})."
            )
        
        # Ensure slice has only the allowed keys
        unrecognized_keys = set(slice) - {"x", "y", "z"}
        if unrecognized_keys:
            raise ValueError(f"Unrecognized slice keys: {unrecognized_keys}.")
        
        # Ensure slice has at most two keys if the plot is scalar
        if not vectors and len(slice) > 2:
            raise ValueError(
                "For a scalar plot, `slice` may have at most 2 keys."
            )
        
        # Ensure slice has at most one key if the plot is vector
        if vectors and len(slice) > 1:
            raise ValueError(
                "For a vector plot, `slice` may have at most 1 keys."
            )
        
        # Ensure that contours has the correct format given the slice
        if len(slice) == 0 and contours is None:
            raise ValueError("In 3D, `contours` must be an integer.")
        if len(slice) == 1 and not isinstance(contours, (int, NoneType)):
            raise ValueError("In 2D, `contours` must be an integer or None.")
        
        # Ensure that ylim_1d is the correct type
        if (
            ylim_1d is not None
            and (
                not isinstance(ylim_1d, Iterable)
                or len(ylim_1d) != 2
                or not all(isinstance(i, numbers.Number) for i in ylim_1d)
            )
        ):
            raise ValueError("`ylim_1d` must be a list of 2 floats.")
    
        # Ensure that opacity is correct
        accepted_presets = ["min", "max", "extremes", "uniform"]
        if opacity_scalar_3d < 0 or opacity_scalar_3d > 1:
            raise ValueError("`opacity_scalar_3d` must be between 0 and 1.")
        if not (
            (
                isinstance(opacityscale_scalar_3d, Iterable)
                and not isinstance(opacityscale_scalar_3d, str)
                and all(
                    len(i) == 2 and all(isinstance(j, numbers.Real))
                    for i in opacity_scalar_3d
                    for j in i
                )
            )
            or (
                isinstance(opacityscale_scalar_3d, str)
                and opacityscale_scalar_3d in accepted_presets
            )
        ):
            raise ValueError(
                "`opacityscale_scalar_3d` must be an array of length-2 arrays "
                + "or a valid string. See https://plotly.com/python/reference/volume/#volume-opacityscale."
            )
        
        # Infer quantity if necessary
        if quantity is None:
            if set(self.quantities) == {"U"}:
                quantity = "U"
            elif set(self.quantities) == {"F", "Fx", "Fy", "Fz"}:
                quantity = "F"
            elif set(self.quantities) == {"T", "Tx", "Ty", "Tz"}:
                quantity = "T"

        # Set slice values to the closest values in the recarray
        for dimension, value in slice.items():
            if value not in self.table[dimension]:
                slice[dimension] = util.plotting.find_nearest(
                    self.table[dimension], value
                )

        # If the array only has one value along any of the dimensions, treat
        # that dimension and value as part of the provided slice
        # [TODO: this is duplicated in private methods]
        slice_recarray = self._subset_of_recarray(**slice)
        extents = {
            d: [slice_recarray[d].min(), slice_recarray[d].max()]
            for d in ["x", "y", "z"]
        }
        for dimension, limits in extents.items():
            if limits[0] == limits[1]:
                slice[dimension] = limits[0]

        # If quantity is not F or T, the plot must be a scalar plot
        if quantity not in ("F", "T"):
            vectors = False
        
        # Build trace
        if not vectors:
            if len(slice) == 0:
                trace = self._plot_trace_scalar_3d(
                    quantity=quantity,
                    clim=clim,
                    contours=contours,
                    cmap=cmap,
                    fill_nan_with_inf=fill_nan_with_inf,
                    show_cbar=show_cbar,
                    opacity=opacity_scalar_3d,
                    opacityscale=opacityscale_scalar_3d,
                )
            elif len(slice) == 1:
                trace = self._plot_trace_scalar_2d(
                    quantity=quantity,
                    slice=slice,
                    clim=clim,
                    contours=contours,
                    cmap=cmap,
                    fill_nan_with_inf=fill_nan_with_inf,
                    show_cbar=show_cbar,
                )
            elif len(slice) == 2:
                trace = self._plot_trace_scalar_1d(
                    quantity=quantity,
                    slice=slice,
                    marker_color=marker_color_1d,
                    marker_mode=marker_mode_1d,
                    marker_size=marker_size_1d,
                    line_width=line_width_1d,
                )
        
        else:
            if len(slice) == 0:
                trace = self._plot_trace_vector_3d(
                    quantity=quantity,
                    clim=clim,
                    cmap=cmap,
                    show_cbar=show_cbar,
                )
            elif len(slice) == 1:
                trace = self._plot_trace_vector_2d(
                    quantity=quantity,
                    slice=slice,
                    marker_color=marker_color_1d,
                )

        # Create and style the figure.
        figure = plotly.graph_objects.Figure()
        figure.add_trace(trace)

        allowed_kwarg_names = (
            signature(plot_layout)
                .parameters
                .keys()
        )
        layout_kwargs = {
            k: v for k, v in kwargs.items() if k in allowed_kwarg_names
        }
        layout = plot_layout(slice=slice, **layout_kwargs)
        figure.update_layout(layout)

        if len(slice) == 2:
            figure.update_layout(yaxis=dict(title=dict(text=quantity)))
        
        if len(slice) == 2:
            if ylim_1d is None:
                ylim_1d = [trace.y.min(), np.percentile(trace.y, 90)]   # TODO: return here - find a better approach
            figure.update_layout(yaxis=dict(range=ylim_1d))

        return figure, trace

    def _plot_trace_scalar_3d(
        self,
        quantity: Literal["U", "F", "T", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        clim: list[float] | None,
        contours: int,
        cmap: str,
        opacity: float,
        opacityscale: float | str,
        fill_nan_with_inf: bool,
        show_cbar: bool,
    ) -> plotly.graph_objs._volume.Volume:
        """Return the Plotly trace for a 3D scalar (Volume) plot.
        
        Parameters
        ----------
        quantity : 'U', 'F', 'T', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', or 'Tz'
            The name of the quantity to plot.
        clim : list[float]
            The lower and upper limits of the colorscale. If None, then
            the lower and upper limits will be set to the 0th and 90th
            percentile values, respectively.
        contours : int
            The number of values to draw contours around. Only uesd in 3D and
            2D scalar plots. In 2D, pass None to instead use a continuous
            colorscale.
        cmap : str
            The name of the Plotly colormap to use. Only used in 3D and 2D
            scalar plots and 3D vector plots.
        opacity : float
            The opacity of the surface. Passed directly to the plotly trace's
            constructor.
        opacityscale : float | str
            The mapping between data values and opacity values. Specified as
            either an array of value pairs or as a string referring to the name
            of a preset scale. Passed directly to the plotly trace's
            constructor.
        fill_nan_with_inf : bool
            Whether to plot NaN values as though they were very large values.
            Only used in 3D and 2D scalar plots.
        show_cbar : bool
            Whether to show the colorbar.

        Returns
        -------
        plotly.graph_objs._volume.Volume
            The trace.
        """
        array = self.to_gridded_array(quantity=quantity, vectors=False)

        array = np.rot90(array, 1, axes=(0, 2))   # TODO: check if this should be in gridded_array
        array = np.flip(array, axis=0)

        if fill_nan_with_inf:
            array = np.nan_to_num(array, nan=1e99)

        # [TODO: this is duplicated in private methods]
        extents = {
            d: [self.table[d].min(), self.table[d].max()]
            for d in ["x", "y", "z"]
        }

        x, y, z = np.mgrid[
            extents["x"][0]:extents["x"][1]:complex(0, array.shape[2]),
            extents["y"][0]:extents["y"][1]:complex(0, array.shape[1]),
            extents["z"][0]:extents["z"][1]:complex(0, array.shape[0]),
        ]

        if clim is None:
            clim = [np.percentile(array, 0), np.percentile(array, 90)]
        
        return plotly.graph_objects.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=array.flatten(),
            isomin=min(clim),
            isomax=max(clim),
            opacity=opacity,
            opacityscale=opacityscale,
            surface_count=contours,
            colorscale=cmap,
            showscale=show_cbar,
            colorbar=dict(
                title=dict(
                    text=quantity,
                    font=util.plotting.AXIS_TITLE_FONT
                )
            )
        )

    def _plot_trace_scalar_2d(
        self,
        quantity: Literal["U", "F", "T", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        slice: dict[str, float],
        clim: list[float] | None,
        contours: int | None,
        cmap: str,
        fill_nan_with_inf: bool,
        show_cbar: bool
    ) -> plotly.graph_objs._contour.Contour | plotly.graph_objs._heatmap.Heatmap:
        """Return the Plotly trace for a 2D scalar (Heatmap or Contour) plot.

        Parameters
        ----------
        quantity : 'U', 'F', 'T', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', or 'Tz'
            The name of the quantity to plot.
        slice : dict
            Axes and positions along which to slice. Keys are limited to 'x',
            'y', and 'z'. There can be at most two keys.
        clim : list[float]
            The lower and upper limits of the colorscale. If None, then
            the lower and upper limits will be set to the 0th and 90th
            percentile values, respectively.
        contours : int or None
            The number of values to draw contours around. Only uesd in 3D and
            2D scalar plots. In 2D, pass None to instead use a continuous
            colorscale.
        cmap : str
            The name of the Plotly colormap to use. Only used in 3D and 2D
            scalar plots and 3D vector plots.
        fill_nan_with_inf : bool
            Whether to plot NaN values as though they were very large values.
            Only used in 3D and 2D scalar plots.
        show_cbar : bool
            Whether to show the colorbar.

        Returns
        -------
        plotly.graph_objs._contour.Contour | plotly.graph_objs._heatmap.Heatmap
            The trace.
        """
        array = self.to_gridded_array(quantity=quantity, vectors=False, **slice)

        slice_recarray = self._subset_of_recarray(**slice)

        # [Review: sorting necessary?]
        if "z" in slice:
            x = np.unique(slice_recarray["x"])
            y = np.unique(slice_recarray["y"])
        elif "y" in slice:
            x = np.unique(slice_recarray["x"])
            y = np.unique(slice_recarray["z"])
        elif "x" in slice:
            x = np.unique(slice_recarray["y"])
            y = np.unique(slice_recarray["z"])
        
        if fill_nan_with_inf:
            array = np.nan_to_num(array, nan=1e99)

        if clim is None:
            clim = [np.percentile(array, 0), np.percentile(array, 90)]
        if np.isnan(clim[0]):
            clim[0] = -1e99
        if np.isnan(clim[1]):
            clim[1] = 1e99

        cmin, cmax = min(clim), max(clim)

        if isinstance(contours, int):
            return plotly.graph_objects.Contour(
                x=x,
                y=y,
                z=array,
                colorscale=cmap,
                opacity=1,
                contours=dict(
                    start=cmin,
                    end=cmax,
                    size=(cmax - cmin) / contours,
                    # type="constraint",
                    # operation=">=",
                    # value=isosurfaces,
                    # showlabels=True,
                    coloring="fill"
                ),
                zmin=cmin,
                zmax=cmax,
                showscale=show_cbar,
                colorbar=dict(
                    title=dict(
                        text=quantity,
                        font=util.plotting.AXIS_TITLE_FONT
                    )
                )
            )

        elif contours is None:
            return plotly.graph_objects.Heatmap(
                x=x,
                y=y,
                z=array,
                colorscale=cmap,
                zmax=cmax,
                zmin=cmin,
                showscale=show_cbar,
                colorbar=dict(
                    title=dict(
                        text=quantity,
                        font=util.plotting.AXIS_TITLE_FONT
                    )
                )
            )

    def _plot_trace_scalar_1d(
        self,
        quantity: Literal["U", "F", "T", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        slice: dict[str, float],
        marker_mode: Literal["lines+markers", "lines", "markers"],
        marker_color: str,
        marker_size: float,
        line_width: float,
    ) -> plotly.graph_objs._scatter.Scatter:
        """Return the Plotly trace for a 1D scalar (Scatter) plot.

        Parameters
        ----------
        quantity : 'U', 'F', 'T', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', or 'Tz'
            The name of the quantity to plot.
        slice : dict
            Axes and positions along which to slice. Keys are limited to 'x',
            'y', and 'z'. There can be at most two keys.
        marker_mode : 'lines+markers', 'lines', or 'markers'
            In a 1D scalar plot, whether to show only lines, only markers, or
            both. Ignored for all other plot types.
        marker_color : str
            In a 1D scalar plot, the color of the plot symbol. Ignored for all
            other plot types.
        marker_size : float
            The size of the marker in pixels.
        line_width : float
            The width of the line in pixels.

        Returns
        -------
        plotly.graph_objs._scatter.Scatter
            The trace.
        """
        array = self.to_gridded_array(quantity=quantity, vectors=False, **slice)

        slice_recarray = self._subset_of_recarray(**slice)

        # [Review: sorting necessary?]
        x_dim = [i for i in ["x", "y", "z"] if i not in slice][0]
        if "z" in slice:
            x = slice_recarray[x_dim]
        elif "y" in slice:
            x = slice_recarray[x_dim]
        elif "x" in slice:
            x = slice_recarray[x_dim]

        return plotly.graph_objects.Scatter(
            x=x,
            y=array,
            name=quantity,
            cliponaxis=False,
            mode=marker_mode,
            marker_color=marker_color,
            marker=dict(size=marker_size),
            line=dict(width=line_width),
        )

    def _plot_trace_vector_3d(
        self,
        quantity: Literal["F", "T"],
        clim: list[float],
        cmap: str,
        show_cbar: bool,
    ) -> plotly.graph_objs._cone.Cone:
        """Return the Plotly trace for a 3D vector (Cone) plot.

        Parameters
        ----------
        quantity : 'F' or 'T'
            The name of the quantity to plot.
        clim : list[float]
            The lower and upper limits of the colorscale. If None, then
            the lower and upper limits will be set to the 0th and 90th
            percentile of the magnitudes, respectively.
        cmap : str
            The name of the Plotly colormap to use. Only used in 3D and 2D
            scalar plots and 3D vector plots.
        show_cbar : bool
            Whether to show the colorbar.

        Returns
        -------
        plotly.graph_objs._cone.Cone
            The trace.
        """
        # Calculate scaled vectors to ensure that the cones all fit on the plot.
        # The smallest scaled magnitude is 0.05 * the longest dimension.
        # The largest scaled magnitude is 0.15 * the longest dimension.
        # The low end of the scale is mapped to the 10th percentile magnitude,
        # with all smaller magnitudes clamped to it. Likewise for the high
        # end of the scale and the 90th percentile magnitude.
        # The scale is linear.
        components = np.column_stack((
            self.table[f"{quantity}x"],
            self.table[f"{quantity}y"],
            self.table[f"{quantity}z"],
        ))
        magnitudes = np.sqrt(np.sum(np.square(components), axis=1))

        m_10 = np.percentile(magnitudes, 10)
        m_90 = np.percentile(magnitudes, 90)
        m_min = magnitudes.min()
        m_max = magnitudes.max()

        longest_side = max([
            self.table["x"].max(),
            self.table["y"].max(),
            self.table["z"].max(),
        ])
        s_min = longest_side * 0.05
        s_max = longest_side * 0.15

        scale_factors = (
            (
                (((magnitudes - m_min) / (m_max - m_min)) * (s_max - s_min))
                + s_min
            ) / magnitudes
        )
        
        for i, m in enumerate(magnitudes):
            if m < m_10:
                scale_factors[i] = s_min / m
            elif m > m_90:
                scale_factors[i] = s_max / m

        if clim is None:
            clim = [np.percentile(magnitudes, 0), m_90]

        customdata = np.column_stack((
            self.table[f"{quantity}x"],
            self.table[f"{quantity}y"],
            self.table[f"{quantity}z"],
            magnitudes
        ))

        return plotly.graph_objects.Cone(
            x=self.table["x"],
            y=self.table["y"],
            z=self.table["z"],
            u=self.table[f"{quantity}x"] * scale_factors,
            v=self.table[f"{quantity}y"] * scale_factors,
            w=self.table[f"{quantity}z"] * scale_factors,
            colorscale=cmap,
            cmin=s_min,#min(clim),
            cmax=s_max,#max(clim),
            showscale=show_cbar,
            colorbar=dict(
                title=dict(
                    text=f"<b>{quantity}</b>",
                    font=util.plotting.AXIS_TITLE_FONT
                ),
                tickvals=[
                    s_min + (s_max - s_min) * i
                    for i in np.linspace(0, 1, 5)
                ],
                ticktext=[
                    f"{(m_10 + (m_90 - m_10) * i):.2e}"
                    for i in np.linspace(0, 1, 5)
                ]
            ),
            sizemode="raw",   # TODO: add customdata 
            customdata=customdata,
            hovertemplate=
                f"<b>|{quantity}|</b> " + "(%{customdata[3]:.2e})<br>"
                + "(%{customdata[0]:.2e}, %{customdata[1]:.2e}, %{customdata[2]:.2e})"
        )

    def _plot_trace_vector_2d(
        self,
        quantity: Literal["F", "T"],
        slice: dict[str, float],
        marker_color: str,
    ) -> plotly.graph_objs._scatter.Scatter:
        """Return the Plotly trace for a 2D vector (Quiver) plot.

        Parameters
        ----------
        quantity : 'U', 'F', 'T', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', or 'Tz'
            The name of the quantity to plot.
        slice : dict
            Axes and positions along which to slice. Keys are limited to 'x',
            'y', and 'z'. There can be at most two keys.
        marker_color : str
            In a 1D scalar plot, the color of the plot symbol. Ignored for all
            other plot types.        

        Returns
        -------
        plotly.graph_objs._scatter.Scatter
            The trace.
        """
        slice_recarray = self._subset_of_recarray(**slice)
        
        if "z" in slice:
            x = slice_recarray["x"]
            y = slice_recarray["y"]
            vec_x = slice_recarray[f"{quantity}x"]
            vec_y = slice_recarray[f"{quantity}y"]
        elif "y" in slice:
            x = slice_recarray["x"]
            y = slice_recarray["z"]
            vec_x = slice_recarray[f"{quantity}x"]
            vec_y = slice_recarray[f"{quantity}z"]
        elif "x" in slice:
            x = slice_recarray["y"]
            y = slice_recarray["z"]
            vec_x = slice_recarray[f"{quantity}y"]
            vec_y = slice_recarray[f"{quantity}z"]

        # Calculate scaled vectors to ensure that the cones all fit on the plot.
        # The smallest scaled magnitude is 0.05 * the longest dimension.
        # The largest scaled magnitude is 0.15 * the longest dimension.
        # The low end of the scale is mapped to the 10th percentile magnitude,
        # with all smaller magnitudes clamped to it. Likewise for the high
        # end of the scale and the 90th percentile magnitude.
        # The scale is linear.
        components = np.column_stack((vec_x, vec_y))
        magnitudes = np.sqrt(np.sum(np.square(components), axis=1))

        m_10 = np.percentile(magnitudes, 10)
        m_90 = np.percentile(magnitudes, 90)
        m_min = magnitudes.min()
        m_max = magnitudes.max()

        longest_side = max([x.max(), y.max()])
        s_min = longest_side * 0.05
        s_max = longest_side * 0.15

        scale_factors = (
            (
                (((magnitudes - m_min) / (m_max - m_min)) * (s_max - s_min))
                + s_min
            ) / magnitudes
        )
        
        for i, m in enumerate(magnitudes):
            if m < m_10:
                scale_factors[i] = s_min / m
            elif m > m_90:
                scale_factors[i] = s_max / m

        customdata = np.column_stack((vec_x, vec_y, magnitudes))

        fig = plotly.figure_factory.create_quiver(
            x=x,
            y=y,
            u=vec_x * scale_factors,
            v=vec_y * scale_factors,
            scale=1,
            arrow_scale=0.4,
            marker_color=marker_color,
            name=quantity,
            customdata=customdata,
            hovertemplate=
                f"<b>|{quantity}|</b> " + "(%{customdata[2]:.2e})<br>"
                + "(%{customdata[0]:.2e}, %{customdata[1]:.2e})"
        )

        return fig.data[0]

    # ---------------------------------- OTHER ---------------------------------

    def __eq__(self, other) -> bool:
        """Whether this field and another are equivalent."""
        return (self._table == other._table).view(np.ndarray).all()
