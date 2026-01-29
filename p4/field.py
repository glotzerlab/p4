# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

import json
import math
from typing import Literal
import warnings
import PIL
import coxeter
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Polygon
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import Affine2D
import pandas as pd
import vtk.util.numpy_support   # type: ignore


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
    embedded within a 3D grid. The extents thus has the form
    
    ```
    [
        [xmin, xmax],
        [ymin, ymax],
        [zmin, zmax],
    ]
    ```
    
    where a 2D array represents a slice at the position z =  zmin = zmax.

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
        core_shape: coxeter.shapes.ConvexPolygon | coxeter.shapes.ConvexPolyhedron | None = None,
        rotate_core_deg: float = 90.0,
        vmin: float = -1.0,
        vmax: float = 10.0,
        # slice_x: float | None = None,         # TODO: decide whether to allow slice
        # slice_y: float | None = None,
        # slice_lim: list[float] | None = None,
        show_cbar: bool = True,
        show_axes: bool = True, # TODO
        core_scale_factor: float = 1.0,
        cmap_name: str = "RdYlBu_r",
        core_color: str = "#FFCF00",
        fill_nan_with_inf: bool = False
    ):
        """TODO"""
        # In 2D, use matplotlib
        if self.n_dimensions == 2:
            mpl_extents = [
                self.extents[0][0],
                self.extents[0][1],
                self.extents[1][0],
                self.extents[1][1]
            ]
            
            # Initialize empty axes
            # if slice_x is not None or slice_y is not None:
            #     if show_cbar:
            #         fig, ax = plt.subplots(1, 3, width_ratios=[1.0, 1.0, 0.1])
            #         sl_ax = ax[0]
            #         im_ax = ax[1]
            #         cbar_ax = ax[2]
            #     else:
            #         fig, ax = plt.subplots(1, 3, width_ratios=[1.0, 1.0, 0.1])
            #         sl_ax = ax[0]
            #         im_ax = ax[1]
            # else:
            if show_cbar:
                fig, ax = plt.subplots(1, 2, width_ratios=[1.0, 0.05])
                im_ax = ax[0]
                cbar_ax = ax[1]
            else:
                fig, ax = plt.subplots(1)
                im_ax = ax

            # Configure figure
            fig.set_size_inches(14, 6)
            fig.set_dpi(300)

            # Configure colormap
            cmap = colormaps[cmap_name]
            # cmap = cmap.set_over(cmap(1.0))

            # Plot image and outline
            if fill_nan_with_inf:
                array = np.nan_to_num(self.array, nan=1e99)
            else:
                array = self.array
            im = im_ax.imshow(
                array,
                extent=mpl_extents,
                cmap=cmap,
                norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
            )

            # Plot core if necessary
            if core_shape is not None:
                # Construct particle outline polygon
                if core_scale_factor != 1.0:
                    core_shape = coxeter.shapes.ConvexPolygon(
                        core_shape.vertices * core_scale_factor
                    )
                particle_outline = Polygon(
                    xy=core_shape.vertices[:,:2],
                    fill=True,
                    facecolor=core_color,
                    edgecolor="black",
                    linestyle="-",
                    linewidth=2.0
                )

                # Rotate if necessary
                if rotate_core_deg is not None:
                    particle_outline.set_transform(
                        Affine2D().rotate(math.radians(rotate_core_deg))
                        + im_ax.transData
                    )
                im_ax.add_patch(particle_outline)

            # # If necessary, plot slice
            # if slice_x is not None:
            #     nearest_x = util.find_nearest(np.array(df.index), slice_x)
            #     slice_path = Path([[nearest_x, mpl_extents[2]], [nearest_x, mpl_extents[3]]])
            #     slice_domain = df.columns
            #     slice_range = np.array(df.loc[nearest_x])
            #     # if slice_zmax is not None:
            #     #     slice_range[slice_range > slice_zmax] = slice_zmax
            
            # elif slice_y is not None:
            #     nearest_y = util.find_nearest(df.columns, slice_y)
            #     slice_path = Path([[mpl_extents[0], nearest_y], [mpl_extents[1], nearest_y]])
            #     slice_domain = np.array(df.index)
            #     slice_range = df[nearest_y]
            #     # if slice_zmax is not None:
            #     #     slice_range[slice_range > slice_zmax] = slice_zmax
            
            # if slice_x is not None or slice_y is not None:
            #     im_ax.add_patch(PathPatch(path=slice_path, linestyle="-.", edgecolor="black", linewidth=2.0))
            #     sl_ax.plot(slice_domain, slice_range, c="red")
            #     sl_ax.scatter(slice_domain, slice_range, c="red", )
            #     if slice_lim is not None:
            #         sl_ax.set_ylim(slice_lim)
            
            if show_cbar:
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_ticks([vmin, 0, vmax])

            return fig, ax

        # In 3D, use plotly
        elif self.n_dimensions == 3:
            raise NotImplementedError("3D plotting is currently not supported.")

    def plot_on_provided_axes(
        self,
        ax: matplotlib.axes.Axes,
        core_shape: coxeter.shapes.ConvexPolygon | coxeter.shapes.ConvexPolyhedron | None = None,
        rotate_core_deg: float = 90.0,
        vmin: float = -1.0,
        vmax: float = 10.0,
        core_scale_factor: float = 1.0,
        cmap_name: str = "RdYlBu_r",
        core_color: str = "#FFCF00",
    ):
        if not self.n_dimensions == 2:
            raise ValueError("This method only works for 2D fields.")
        
        mpl_extents = [
            self.extents[0][0],
            self.extents[0][1],
            self.extents[1][0],
            self.extents[1][1]
        ]

        # Plot image and outline
        im = ax.imshow(
            # np.nan_to_num(self.array),
            self.array,
            extent=mpl_extents,
            cmap=colormaps[cmap_name],
            norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
        )

        # Plot core if necessary
        if core_shape is not None:
            # Construct particle outline polygon
            if core_scale_factor != 1.0:
                core_shape = coxeter.shapes.ConvexPolygon(
                    core_shape.vertices * core_scale_factor
                )
            particle_outline = Polygon(
                xy=core_shape.vertices[:,:2],
                fill=True,
                facecolor=core_color,
                edgecolor="black",
                linestyle="-",
                linewidth=2.0
            )

            # Rotate if necessary
            if rotate_core_deg is not None:
                particle_outline.set_transform(
                    Affine2D().rotate(math.radians(rotate_core_deg))
                    + ax.transData
                )
            ax.add_patch(particle_outline)
