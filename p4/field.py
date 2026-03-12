# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

import json
from typing import Literal
import warnings
import PIL
import plotly
import numpy as np
import pandas as pd
import vtk.util.numpy_support   # type: ignore
from copy import copy, deepcopy

class Field:
    """A Field is defined by an array of values and an array of extents.
    
    This class can be instantiated directly from a numpy array and an optional
    array of extents, but most users will want to construct it from a CSV, TIFF,
    or VTI file. To do so, use the corresponding class methods `from_csv()`,
    `from_tiff()`, and `from_vti()`.

    [TODO: revise next paragraph - the central question is how we should
    handle fields with different dimensionalities.]

    Both the array and the extents are **always** 3D. If a Field is constructed
    with a 2D array, the array is coerced to 3D and represents a 2D slice
    embedded within a 3D grid. The extents thus has the form::
    
        [
            [xmin, xmax],
            [ymin, ymax],
            [zmin, zmax],
        ]
    
    where a 2D array represents a slice at the position ``z = zmin = zmax``.

    Parameters
    ----------
    array : np.ndarray
        The 2D or 3D numpy array of values.
    extents : list[list[float]]
        The minimum and maximum values along the X, Y, and Z axes.

    Raises
    ------
    ValueError
        If the provided array is not 2D or 3D.
    """
    def __init__(
        self,
        array: np.ndarray,
        extents: list[list[float]] | None = None
    ):
        if (n := len(array.shape)) not in [2, 3]:
            raise ValueError(f"`array` should be 2D or 3D but it is {n}D")

        self.array = array
        self.extents = extents

    @classmethod
    def from_csv(cls, filename, orientation: Literal["mean", "min", "boltzmann"]):
        """Construct a Field from a CSV file.
        
        This constructor is intended for use only with CSV files created with
        `System.probe_potential()`.

        Parameters
        ----------
        filename : str
            The name of the CSV file.
        orientation : 'mean' or 'min' or 'boltzmann'
            Whether to average potential values over all orientations ('mean')
            or take the minimum potential for each orientation ('min'), or the
            boltzmann average over all orientations ('boltzmann').
        """
        df = pd.read_csv(filename)
        array = cls._df_to_array(df, orientation)
        extents = [
            [df["x"].min(), df["x"].max()],
            [df["y"].min(), df["y"].max()],
            [df["z"].min(), df["z"].max()],
        ]
        return cls(array, extents)
    
    @classmethod
    def from_tiff(cls, filename):
        """Construct a Field from a TIFF file.

        This constructor is intended for use with TIFF files created with
        `Field.save_image()`, since they have the extents encoded, but it will
        still work with other TIFF files.

        Parameters
        ----------
        filename : str
            The name of the TIFF file.
        """
        image = PIL.Image.open(filename)
        
        extents_str = image.tag.get(270, None)
        if extents_str:
            extents = json.loads(extents_str[0])
            extents = [
                [extents["xmin"], extents["xmax"]],
                [extents["ymin"], extents["ymax"]],
                [extents["zmin"], extents["zmax"]],
            ]
            return cls(np.array(image), extents)
        else:
            warnings.warn("Could not read extents from TIFF.")
            return cls(np.array(image))

    @classmethod
    def from_vti(cls, filename):
        """Construct a Field from a VTI file.

        This constructor is intended for use with VTI files created with
        `Field.save_image()`, since they have the extents encoded, but it will
        still work with other VTI files.

        Parameters
        ----------
        filename : str
            The name of the VTI file.
        """
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(filename)
        reader.Update()
        vtk_image = reader.GetOutput()
        
        scalars = vtk_image.GetPointData().GetScalars()

        array = vtk.util.numpy_support.vtk_to_numpy(scalars)
        array = array.reshape(*vtk_image.GetDimensions())

        spacing = vtk_image.GetSpacing()
        if spacing:
            extents = [
                [-(spacing[2] * array.shape[2])/2, (spacing[2] * array.shape[2])/2],
                [-(spacing[1] * array.shape[1])/2, (spacing[1] * array.shape[1])/2],
                [-(spacing[0] * array.shape[0])/2, (spacing[0] * array.shape[0])/2],
            ]
            return cls(array, extents)
        else:
            return cls(array)

    @property
    def n_dimensions(self):
        """The number of dimensions represented by the array (2 or 3)."""
        return len(self.array.shape)

    @staticmethod
    def _df_to_array(df, orientation: Literal["mean", "min", "boltzmann"]):
        """Convert a raw field dataframe into a 2D or 3D numpy array.

        This processing has the following steps:
        1. Average over orientations and then drop orientation columns.
        2. Change all NaN values to Inf

        Parameters
        ----------
        orientation : 'mean' or 'min'
            Whether to average potential values over all orientations ('mean')
            or take the minimum potential for each orientation ('min'), or the
            boltzmann average over all orientations ('boltzmann')..

        Raises
        ------
        ValueError
            If `orientation` is not 'mean' or 'min'.
        """
        # Average over orientation, then drop orientation columns
        if orientation == "mean":
            df = df.groupby(["x", "y", "z"]).mean()
            df = df.reset_index(level=[0,1,2])
            df = df.drop(labels=["t", "q0", "q1", "q2", "q3"], axis=1)
        
        elif orientation == "min":
            df = df.groupby(["x", "y", "z"]).min()
            df = df.reset_index(level=[0,1,2])
            df = df.drop(labels=["t", "q0", "q1", "q2", "q3"], axis=1)
        
        elif orientation == "boltzmann":
            def boltzmann(x):
                return np.sum(x * np.exp(-x)) / np.sum(np.exp(-x))
            df = df.groupby(["x", "y", "z"])[["PE"]].agg(boltzmann)
            df = df.reset_index(level=[0,1,2])

        else:
            raise ValueError("`orientation` must be 'mean' or 'min' or 'boltzmann'.")
        
        # df = df.reset_index(level=[0,1,2])
        # df = df.drop(labels=["t", "q0", "q1", "q2", "q3"], axis=1)

        # TODO: consider replacing NaN with Inf
        # df = df.replace([np.nan], np.inf)

        # Reshape DataFrame into a numpy grid
        ndim = 2 if df["z"].max() == df["z"].min() else 3
        return Field._tall_df_to_array(df, ndim)

    @staticmethod
    def _tall_df_to_array(df: pd.DataFrame, ndim: Literal[2, 3]) -> np.ndarray:
        """Convert a tall DataFrame into a 3D numpy array.

        Adapted from https://stackoverflow.com/a/35049899/15426433.

        Parameters
        ----------
        df : pd.DataFrame
            The tall dataframe.
        ndim : 2 or 3
            The number of dimensions of the field represented by the dataframe.

        Returns
        -------
        array
            The 3D numpy array.
        """
        df = df.set_index(["z", "y", "x"])

        shape = tuple(map(len, df.index.levels))
        
        array = np.full(shape, np.nan)
        array[tuple(df.index.codes)] = df["PE"].values

        if ndim == 3:
            return array
        else:
            return array[0,:,:]     # TODO: check indexing

    @staticmethod
    def _save_2d_array_to_tiff(
        array: np.ndarray,
        extents: list[list[float]],
        filename: str
    ):
        """Write a 2D numpy array to a TIFF file.

        Parameters
        ----------
        array : np.ndarray
            The 2D numpy array.
        extents : list[list[float]]
            The X, Y, and Z extents of the array.
        filename : str
            The name of the TIFF file.
        """
        extents_info_str = json.dumps(dict(
            xmin=extents[0][0],
            xmax=extents[0][1],
            ymin=extents[1][0],
            ymax=extents[1][1],
            zmin=extents[2][0],
            zmax=extents[2][1],
        ))

        tiffinfo = PIL.TiffImagePlugin.ImageFileDirectory_v2()
        tiffinfo[270] = extents_info_str

        PIL.Image.fromarray(array).save(filename, tiffinfo=tiffinfo)

    @staticmethod
    def _save_3d_array_to_vti(
        array: np.ndarray,
        extents: list[list[float]],
        filename: str,
    ):
        """Write a 3D numpy array to a VTI file.
        
        Adapted from https://discourse.paraview.org/t/help-needed-with-vtk-and
        -paraview-converting-saving-and-rendering-3d-numpy-array/13526

        Parameters
        ----------
        array : np.ndarray
            The 3D numpy array.
        extents : list[list[float]]
            The X, Y, and Z extents of the array.
        filename : str
            The name of the VTI file.
        """
        vtk_data = vtk.util.numpy_support.numpy_to_vtk(
            num_array=array.flatten(),
            deep=True,
            array_type=vtk.VTK_FLOAT
        )

        img = vtk.vtkImageData()
        img.GetPointData().SetScalars(vtk_data)
        img.SetDimensions(*array.shape)

        spacing = [
            (extents[0][1] - extents[0][0])/array.shape[0], # TODO: check the indexing
            (extents[1][1] - extents[1][0])/array.shape[1],
            (extents[2][1] - extents[2][0])/array.shape[2],
        ]
        img.SetSpacing(*spacing)

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(img)
        writer.Write()

    def save_image(self, filename):
        """Save the array of values to an image file.
        
        If the array is 2D, it will be saved to TIFF. If the array is 3D, it
        will be saved to VTI.

        Parameters
        ----------
        filename : str
            The name of the output file.
        
        Raises
        ------
        ValueError
            If the filename extension does not match the number of dimensions of
            the array (2D -> tiff, 3D -> vti).
        """
        if self.n_dimensions == 2:
            # TODO: consider allowing saving to VTI in 2D (coerce to 3D)
            if filename.split(".")[-1] != "tiff":
                raise ValueError(
                    "To save a 2D array, `filename` must end in '.tiff'."
                )
            self._save_2d_array_to_tiff(self.array, self.extents, filename)
             
        elif self.n_dimensions == 3:
            if filename.split(".")[-1] != "vti":
                raise ValueError(
                    "To save a 3D array, `filename` must end in '.vti'."
                )
            self._save_3d_array_to_vti(self.array, self.extents, filename)

    def plot(
        self,
        slice: dict[str, float] = {},
        clim: list[float] = [-1,10],
        contours: int | None = 10,
        cmap="RdYlBu_r",
        show_cbar=True,
        show_axes=True,
        show_title=True,
        fill_nan_with_inf=False
    ):
        """Interactively plot the field using plotly.
        
        Slicing is supported along the X, Y, and Z axes via the ``slice``
        parameter. A slice along one axis (e.g., ``slice={"x": 1}``) is 2D,
        while a slice along two axes (e.g., ``slice={"x": 1, "y": 1}``) is 1D.

        Parameters
        ----------
        slice : dict, default={}
            Axes and positions along which to slice. Keys are limited to 'x',
            'y', and 'z'. There can be at most two keys.
        clim : list of floats, default=[-1, 10]
            The limits of the colorscale, expressed as an array of 2 or 3
            numbers. The lowest value is used as the colorscale's minimum,
            while the highest is used for the colorscale's maximum. If a 3rd
            number is provided, it is used as the colorscale's midpoint.
        contours : int, default=10
            The number of values to draw contours around. In 3D, this parameter
            must be a positive integer. In 2D, None may be passed to use a
            continuous colorscale. In 1D, this parameter is ignored.
        cmap : str, default='RdYlBu_r'
            The name of the plotly colormap to use.
        show_cbar : bool, default=True
            Whether to show the colorbar.
        show_axes : bool, default=True
            Whether to show the axes.
        show_title : bool, default=True
            Whether to show the title.
        fill_nan_with_inf : bool, default=False
            Whether to plot NaN values as though they were very large values.
        
        Returns
        -------
        figure, traces
            The plotly figure and associated traces.
        """
        # Calculate color axis limits
        cmin = min(clim)
        cmax = max(clim)
        if len(clim) == 2:
            cmid = (cmax + cmin) / 2
        elif len(clim) == 3:
            cmid = [i for i in clim if i not in (min(clim), max(clim))][0]
        else:
            raise ValueError("`clim` must be of length 2 or 3.")

        # If 2 slice dimensions are provided, plot will be 1D
        if len(slice) == 2:
            # Calculate the sliced array (use nearest values to the ones provided)
            dimensions, values = list(slice.keys()), list(slice.values())
            d_indices = [["z", "y", "x"].index(d) for d in dimensions]
            d_values = [
                np.linspace(
                    self.extents[d_index][0],
                    self.extents[d_index][1],
                    self.array.shape[d_index]
                )
                for d_index in d_indices
            ]
            v_indices = [
                (np.abs(d_values[i] - v)).argmin()
                for i, v in enumerate(values)
            ]

            array = self.array[v_indices[0], v_indices[1], :]
            x_axis_dim = [i for i in [0, 1, 2] if i not in d_indices][0]

            x = np.linspace(
                self.extents[x_axis_dim][0],
                self.extents[x_axis_dim][1],
                self.array.shape[x_axis_dim]
            )

            traces = [
                plotly.graph_objects.Scatter(
                    x=x,
                    y=array,
                    name="PE",
                    mode="lines+markers",
                    cliponaxis=False
                )
            ]

        # If 1 slice dimension is provided, plot will be 2D
        elif len(slice) == 1:
            # Calculate the sliced array (use nearest value to the one provided)
            dimension, value = list(slice.items())[0]
            d_index = ["z", "y", "x"].index(dimension)
            d_values = np.linspace(
                self.extents[d_index][0],
                self.extents[d_index][1],
                self.array.shape[d_index]
            )
            v_index = (np.abs(d_values - value)).argmin()
            
            if d_index == 0:
                array = self.array[v_index, :, :]
                x_axis_dim = 2
                y_axis_dim = 1
            elif d_index == 1:
                array = self.array[:, v_index, :]
                x_axis_dim = 2
                y_axis_dim = 0
            elif d_index == 2:
                array = self.array[:, :, v_index]
                x_axis_dim = 1
                y_axis_dim = 0
            
            else:
                raise ValueError("`slice` can only contain the keys 'x', 'y', or 'z'.")
            
            if fill_nan_with_inf:
                array = np.nan_to_num(copy(array), nan=1e99)
            
            # Calculate plot data
            x = np.linspace(
                self.extents[x_axis_dim][0],
                self.extents[x_axis_dim][1],
                self.array.shape[x_axis_dim]
            )
            y = np.linspace(
                self.extents[y_axis_dim][0],
                self.extents[y_axis_dim][1],
                self.array.shape[y_axis_dim]
            )

            traces = []

            # Contour plot
            if isinstance(contours, int):
                traces.append(
                    plotly.graph_objects.Contour(
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
                    )
                )

            # heatmap
            elif contours is None:
                traces.append(
                    plotly.graph_objects.Heatmap(
                        x=x,
                        y=y,
                        z=array,
                        colorscale=cmap,
                        zmax=cmax,
                        zmin=cmin
                    )
                )

            else:
                raise ValueError("In 2D, `contours` must be an int or None.")
        
        # If 0 slice dimensions are provided, plot will be 3D
        elif len(slice) == 0:
            array = deepcopy(self.array)

            array = np.rot90(array, 1, axes=(0, 2))
            array = np.flip(array, axis=0)

            if fill_nan_with_inf:
                array = np.nan_to_num(array, nan=1e99)
            
            traces = []

            x, y, z = np.mgrid[
                self.extents[0][0]:self.extents[0][1]:complex(0, array.shape[2]),
                self.extents[1][0]:self.extents[1][1]:complex(0, array.shape[1]),
                self.extents[2][0]:self.extents[2][1]:complex(0, array.shape[0]),
            ]
            
            # Volume plot
            if isinstance(contours, int):
                traces.append(
                    plotly.graph_objects.Volume(
                        x=x.flatten(),
                        y=y.flatten(),
                        z=z.flatten(),
                        value=array.flatten(),
                        isomin=cmin,
                        isomax=cmax,
                        cmid=cmid,
                        opacity=0.1,
                        surface_count=contours,
                        colorscale=cmap,
                        showscale=show_cbar,
                    )
                )

            else:
                raise ValueError("In 3D, `contours` must be an int.")

        else:
            raise ValueError("`slice` can only specify up to 2 dimensions.")
        
        # Create figure and add traces one by one
        figure = plotly.graph_objects.Figure()
        for trace in traces:
            figure.add_trace(trace)

        # # Move legend if necessary
        # if len(slice) == 0 and body_traces:
        #     figure.update_layout(
        #         legend=dict(yanchor="top", xanchor="left", x=0, y=1)
        #     )
        
        # Ensure that 3D plots have correctly visible axes
        if len(slice) == 0:
            figure.update_layout(
                scene=dict(
                    xaxis=dict(visible=show_axes),
                    yaxis=dict(visible=show_axes),
                    zaxis=dict(visible=show_axes),
                )
            )

        # Set aspect ratio if necessary
        if len(slice) == 1:
            figure.update_layout(
                xaxis=dict(scaleanchor="y", scaleratio=1, constrain='domain'),
                yaxis=dict(scaleanchor="x", scaleratio=1, constrain='domain'),
                plot_bgcolor="rgba(0,0,0,0)"
            )

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
                        title=dict(text=x_title, font=dict(weight=1000, size=16)),
                        visible=show_axes,
                    ),
                    yaxis=dict(
                        scaleanchor="x",
                        scaleratio=1,
                        constrain='domain',
                        title=dict(text=y_title, font=dict(weight=1000, size=16)),
                        visible=show_axes,
                    ),
                    # plot_bgcolor="rgba(0,0,0,0)"
                    title=dict(
                        text=fig_title if show_title else "",
                        font=dict(style="italic", size=16),
                        xanchor="center",
                        yanchor="top",
                        x=0.5,
                    ),
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
                        title=dict(text=x_title, font=dict(weight=1000, size=16)),
                        visible=show_axes
                    ),
                    yaxis=dict(
                        title=dict(text="Potential Energy (k<sub>b</sub>T)", font=dict(weight=1000, size=16)),
                        visible=show_axes
                    ),
                    # plot_bgcolor="rgba(0,0,0,0)"
                    title=dict(
                        text=fig_title if show_title else "",
                        font=dict(style="italic", size=16),
                        xanchor="center",
                        yanchor="top",
                        x=0.5,
                    ),
                )

        # Set y axis range if necessary
        if len(slice) == 2:
            figure.update_layout(yaxis_range=[cmin, cmax])

        return figure, traces
