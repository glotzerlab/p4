from copy import deepcopy
import json
from typing import Callable, Literal
import warnings
import PIL
import PIL.TiffImagePlugin
import coxeter
import hoomd
import numpy as np
import pandas as pd
from tqdm import tqdm
import util
import vtk


class Interaction:
    """A container for the data to  create and parametrize an MD pair potential.

    Parameters
    ----------
    hoomd_class : hoomd.md.pair.Pair
        The constructor for the HOOMD class. Must be in `hoomd.md.pair`.
    initial_inputs : dict[str, float | str]
        All parameters (that aren't `nlist`) that are needed for instantiating
        the class from its constructor.
    default_single_typed_attributes : dict
        The names and default values of attributes that must be set for
        individual particle types. This is usually an empty dict for isotropic
        interactions.
    default_pair_typed_attributes : dict[str, float]
        The names and default values of attributes that must be set for
        pairs of particle types. This is usually an empty dict for isotropic
        interactions.
    yes_types : list[str]
        The particle types that can interact with each other under this
        potential. All interacting type pairs are assumed to use the same
        interaction attributes.
    yes_single_typed_attributes : dict[str, float]
        The names and values of the attributes for interacting particle types
        that must be set for individual types. This is usually an empty dict for
        isotropic interactions.
    yes_pair_typed_attributes : dict[str, float]
        The names and values values of attributes for interacting particle types
        that must be set for pairs of types. All interacting type pairs are
        assumed to use the same interaction attributes.

    Raises
    ------
    ValueError
        If the provided initial_inputs or typed attributes are not correct.        
    """
    def __init__(
        self,
        hoomd_class: hoomd.md.pair.Pair,
        initial_inputs: dict[str, float | str],
        default_single_typed_attributes: dict,
        default_pair_typed_attributes: dict[str, float],
        yes_types: list[str],
        yes_single_typed_attributes: dict[str, float],
        yes_pair_typed_attributes: dict[str, float],
    ):
        self.hoomd_class = hoomd_class
        self.initial_inputs = initial_inputs
        self.default_single_typed_attributes = default_single_typed_attributes
        self.default_pair_typed_attributes = default_pair_typed_attributes
        self.yes_types = yes_types
        self.yes_single_typed_attributes = yes_single_typed_attributes
        self.yes_pair_typed_attributes = yes_pair_typed_attributes

        self.validate()

    def validate(self):
        """Instantiate the HOOMD class and attempt to run a simulation."""
        nlist = hoomd.md.nlist.Cell(2)
        try:
            test_instance = self.hoomd_class(nlist, **self.initial_inputs)
        except TypeError, hoomd.error.TypeConversionError:
            raise ValueError(
                "The 'initial_inputs' are wrong. See traceback for details."
            )
        
        simulation = hoomd.util.make_example_simulation(
            particle_types=self.yes_types
        )
        simulation = util.add_integrator(simulation)
        simulation = util.add_interaction(simulation, nlist, self)

        try:
           simulation.run(0)
        except:
            raise ValueError(
                "The typed attributes are wrong. See traceback for details."
            )

class ParticleModel:
    """The names and positions of a particle's primary and secondary types.

    Every particle model must have a primary type, but secondary types are
    optional.
    
    When secondary types **are not** provided, the model represents a
    simple particle with a single type and no further information is needed.
    
    When secondary types **are** provided, the model represents a rigid body
    with a central particle (`primary_type`) and one or more constituent
    particles (`secondary_types`). In this case, the model needs a way to
    determine the position(s) of each type of constituent particle, and the user
    must provide a function that does so.
    
    Parameters
    ----------
    primary_type : str
        The name of the primary type.
    secondary_types : list[str], optional
        The names of the secondary types.
    get_secondary_positions_by_type : Callable, optional
        A function that returns the position(s) of particle(s) with the
        secondary types. The function must have at least one parameter, and the
        first parameter must be a string representing the name of a secondary
        type. Required if `secondary_types` is provided, otherwise ignored.
    """
    def __init__(
        self,
        primary_type: str,
        secondary_types: list[str] = [],
        get_secondary_positions_by_type: Callable | None = None
    ):
        if secondary_types and get_secondary_positions_by_type is None:
            raise ValueError(
                "'get_secondary_positions_by_type' is required if "
                + "'secondary_types' is provided"
            )

        self.primary_type = primary_type
        self.secondary_types = secondary_types
        self.get_secondary_positions_by_type = get_secondary_positions_by_type
    
    def can_be_rigid_body(self) -> bool:
        """Whether the particle can be a rigid body."""
        return len(self.secondary_types) == 0
    
    def must_be_rigid_body(self, interaction: Interaction) -> bool:
        """Whether the particle must be a rigid body for some interaction."""
        return any([t in self.secondary_types for t in interaction.yes_types])

    def validate_get_secondary_positions_by_type(self):
        # TODO
        pass

class System:
    """A System is defined by its particle models and an interaction model.
    
    Once the system is instantiated, its potential energy landscape can be
    measured using the `probe()` method.

    Parameters
    ----------
    probe : ParticleModel
        The particle model for the probe.
    analyte : ParticleModel
        The particle model for the analyte.
    interaction_model : dict[str, Interaction]
        A collection of named interactions. All keys should have the `str` type
        and all values should have the `Interaction` type.
    """
    def __init__(
        self,
        probe_model: ParticleModel,
        analyte_model: ParticleModel,
        interaction_model: dict[str, Interaction],
    ):
        self.probe_model = probe_model
        self.analyte_model = analyte_model
        self.interaction_model = interaction_model

        self.validate()

    def validate(self):
        """Ensure all possible interacting types appear in particle models."""
        valid_types = []
        for model in [self.probe_model, self.analyte_model]:
            valid_types.append(model.primary_type)
            valid_types.extend(model.secondary_types)
        
        invalid_types = {}
        for name, interaction in self.interaction_model.items():
            invalid_types[name] = [
                t
                for t in interaction.yes_types if t not in valid_types
            ]

        if len(invalid_types.keys()) > 0:
            raise ValueError(
                f"The provided interaction model contains types that are "
                f"not in any of the provided interaction models: "
                f"{invalid_types}."
            )

    def interactions(self, names: list[str] = []) -> list[Interaction]:
        """A list of interactions, all of them or just those named.

        Parameters
        ----------
        names : list[str], optional
            The names of the interactions to include. If not provided, all
            interactions from the interaction model are included.

        Returns
        -------
        interactions
        """
        if names != []:
            return [self.interaction_model[name] for name in names]
        else:
            return list(self.interaction_model.values())

    def interacting_types(self, names: list[str] | None = None) -> set[str]:
        """The interacting types, for all interactions or just those named.
        
        Parameters
        ----------
        names : list[str], optional
            The names of the interactions to include. If not provided, particle
            types for all interactions in the interaction model are included.

        Returns
        -------
        types
        """
        interacting_types = set([
            t
            for interaction in self.interactions(names)
            for t in interaction.yes_types
        ])
        return interacting_types

    def probe_potential(
        self,
        position_resolutions: list[list[float]],
        orientation_resolutions: list[list[float]],
        orientation_symmetries: list[int],
        nlist: hoomd.md.nlist.NeighborList,
        interactions_to_include: list[str],
        probe_cutoff_outside_distance: Callable,  # needs to be a callable if the user can change params after instantiation
        probe_cutoff_inside_distance: Callable | None = None,
        probe_cutoff_shape: coxeter.shapes.ConvexPolyhedron | None = None,
        gsd_filename: str | None = None,
        csv_filename: str | None = None,
        box_safety_factor: float = 10
    ):
        """Probe the potential energy landscape of the system.

        Parameters
        ----------
        position_resolutions : list[list[float]]
            The number of samples along each dimension of the position grid.
            [X, Y, Z]
        orientation_resolutions : list[list[float]]
            The number of samples along each dimension of the orientation grid.
            [X, Y, Z]
        orientation_symmetries : list[int]
            The rotational symmetry for each axis. If not provided, C1 symmetry
        is assumed for every axis. [X, Y, Z]
        nlist : hoomd.md.nlist.NeighborList
            The neighborlist to use for the interactions.
        interactions_to_include : list[str]
            The names of the interactions to include.
        probe_cutoff_outside_distance : Callable
            A callable that takes `self` as its only argument and returns a
            float representing the cutoff distance outside which no positions
            will be probed. If`probe_cutoff_shape` is provided, this distance
            represents a buffer distance around the shape, otherwise it
            distance represents the side lengths of a cube centered on the
            origin.
        probe_cutoff_inside_distance : Callable, optional
            A callable that takes `self` as its only argument and returns a
            float representing the cutoff distance inside which no positions
            will be probed. If`probe_cutoff_shape` is provided, this distance
            represents a buffer distance inside the shape, otherwise it
            distance represents the side lengths of a cube centered on the
            origin. If not provided, all positions inside the outer cutoff
            distance will be probed.
        probe_cutoff_shape : coxeter.shapes.ConvexPolyhedron, optional
            A convex polyhedron representing a shape to which cutoff distances
            are relative, enabling the user to probe non-cubic boxes. If not
            provided, cutoff distances describe the side lengths of a cube.
        gsd_filename : str, optional
            The name of the GSD file to save. If not provided, no GSD file is
            saved.
        csv_filename : str, optional
            The name of the CSV file to save. If not provided, no CSV file is
            saved.
        box_safety_factor : float, optional
            The scale factor for the simulation box, since it must be bigger
            than the probe box to prevent the minimum image problem. Defaults
            to 10.
        """
        # determine needed interactions and particle types
        interactions = self.interactions(interactions_to_include)
        types = self.interacting_types(interactions_to_include)

        # calculate probe box based on distance cutoff callables for included interactions
        outside_cutoff = probe_cutoff_outside_distance(self)
        if probe_cutoff_shape is None:
            probe_box = [2*outside_cutoff, 2*outside_cutoff, 2*outside_cutoff]
        else:
            shape_maxes = probe_cutoff_shape.vertices.max(axis=0)
            probe_box = [m + outside_cutoff for m in shape_maxes]

        # determine the frame's box from the probe box
        simulation_box = [d * box_safety_factor for d in probe_box]
        simulation_box.extend([0, 0, 0])
        
        # create initial frame
        frame = util.get_initial_frame(
            self.probe_model,
            self.analyte_model,
            types,
            probe_box,
            simulation_box
        )

        # initialize Simulation
        simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
        simulation.create_state_from_snapshot(frame)

        # add integrator
        simulation = util.add_integrator(simulation)

        # add rigid bodies if necessary
        probe_is_rigid = util.particle_must_be_rigid_body(
            self.probe_model,
            interactions
        )
        analyte_is_rigid = util.particle_must_be_rigid_body(
            self.analyte_model,
            interactions
        )
        if probe_is_rigid:
            simulation, rigid = util.add_rigid_constraint(
                simulation,
                self.probe_model,
                False if analyte_is_rigid else True,
                [t for t in self.probe_model.secondary_types if t in types]
            )
        if analyte_is_rigid:
            simulation, _ = util.add_rigid_constraint(
                simulation,
                self.analyte_model,
                True,
                [t for t in self.analyte_model.secondary_types if t in types],
                rigid if probe_is_rigid else None
            )

        # add gsd/table writers if file names are provided
        if gsd_filename:
            simulation, compute = util.add_gsd_writer(simulation, gsd_filename)

        if csv_filename:
            csv_file = open(csv_filename, "w")
            if gsd_filename:
                simulation, _ = util.add_table_writer(
                    simulation, csv_file, self.probe_model, compute
                )
            else:
                simulation, _ = util.add_table_writer(
                    simulation, csv_file, self.probe_model
                )

        # add required interactions
        for interaction in interactions:
            simulation = util.add_interaction(simulation, nlist, interaction)

        # calculate the probe positions and orientations
        probe_positions = util.get_probe_positions(
            probe_box,
            position_resolutions
        )
        probe_orientations = util.get_probe_orientations(
            orientation_resolutions,
            orientation_symmetries
        )

        # remove positions that are too far away
        probe_positions = util.exclude_positions_by_shape(
            positions=probe_positions,
            exclude_inside=False,
            shape=(
                probe_cutoff_shape
                if probe_cutoff_shape is not None
                else util.get_cube(outside_cutoff)
            ),
            buffer=outside_cutoff if probe_cutoff_shape is not None else 0.0
        )

        # remove positions that are too close
        if probe_cutoff_inside_distance is not None:
            inside_cutoff = probe_cutoff_inside_distance(self)
            if inside_cutoff <= 0:
                raise ValueError(
                    "'probe_cutoff_inside_distance' must return a value "
                    f"greater than 0."
                ) 
            probe_positions = util.exclude_positions_by_shape(
                positions=probe_positions,
                exclude_inside=True,
                shape=(
                    probe_cutoff_shape
                    if probe_cutoff_shape is not None
                    else util.get_cube(probe_cutoff_inside_distance(self))
                ),
                buffer=(
                    probe_cutoff_inside_distance(self)
                    if probe_cutoff_shape is not None
                    else 0.0
                )
            )

        # run the probe simulation
        state = simulation.state.get_snapshot()
        all_types = state.particles.types
        probe_index = deepcopy(np.where(
            state.particles.typeid == all_types.index(self.probe_model.primary_type)
        ))
        analyte_index = deepcopy(np.where(
            state.particles.typeid == all_types.index(self.analyte_model.primary_type)
        ))
        analyte_orientation = deepcopy(
            state.particles.orientation[analyte_index,:]
        )
        integrate_rotational_dof = deepcopy(    # currently always True
            simulation.operations.integrator.integrate_rotational_dof
        )

        for p in tqdm(probe_positions):
            for o in probe_orientations:
                with simulation.state.cpu_local_snapshot as state:

                    state.particles.position[state.particles.rtag[analyte_index]] = np.array([0.0, 0.0, 0.0])
                    state.particles.velocity[state.particles.rtag[analyte_index]] = np.array([0.0, 0.0, 0.0])
                    if integrate_rotational_dof:
                        state.particles.angmom[state.particles.rtag[analyte_index]] = np.array([0.0, 0.0, 0.0, 0.0])
                        state.particles.orientation[state.particles.rtag[analyte_index]] = analyte_orientation

                    state.particles.position[state.particles.rtag[probe_index]] = p
                    state.particles.velocity[state.particles.rtag[probe_index]] = np.array([0.0, 0.0, 0.0])
                    if integrate_rotational_dof:
                        state.particles.angmom[state.particles.rtag[probe_index]] = np.array([0.0, 0.0, 0.0, 0.0])
                        state.particles.orientation[state.particles.rtag[probe_index]] = o

                simulation.run(1)

        # close the csv file if necessary and clean its columns
        if csv_filename:
            csv_file.close()
            util.simplify_probe_csv_columns(csv_filename)

class Field:
    """A Field is defined by an array of values and an array of extents.
    
    This class can be instantiated directly from a numpy array and an optional
    array of extents, but most users will want to construct it from a CSV, TIFF,
    or VTI file. To do so, use the corresponding class methods `from_csv()`,
    `from_tiff()`, and `from_vti()`.

    While the array can be 2D or 3D, the extents array is **always** 3D and has
    the following form:
    ```
    [
        [xmin, xmax],
        [ymin, ymax],
        [zmin, zmax],
    ]
    ```
    
    If the array is 2D, zmin = zmax and the array therefore represents a 2D
    slice through a 3D field at the position z = zmin = zmax.

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
    def from_csv(cls, filename, orientation: Literal["mean", "min"]):
        """Construct a Field from a CSV file.
        
        This constructor is intended for use only with CSV files created with
        `System.probe_potential()`.

        Parameters
        ----------
        filename : str
            The name of the CSV file.
        orientation : 'mean' or 'min'
            Whether to average potential values over all orientations ('mean')
            or take the minimum potential for each orientation ('min').
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
        """The number of dimensions of the array. Should be 2 or 3."""
        return len(self.array.shape)
    @staticmethod

    def _df_to_array(df, orientation: Literal["mean", "min"]):
        """Convert a raw field dataframe into a 2D or 3D numpy array.

        This processing has the following steps:
        1. Average over orientations and then drop orientation columns.
        2. Change all NaN values to Inf

        Parameters
        ----------
        orientation : 'mean' or 'min'
            Whether to average potential values over all orientations ('mean')
            or take the minimum potential for each orientation ('min').

        Raises
        ------
        ValueError
            If `orientation` is not 'mean' or 'min'.
        """
        # Average over orientation, then drop orientation columns
        if orientation == "mean":
            df = df.groupby(["x", "y", "z"]).mean()
        elif orientation == "min":
            df = df.groupby(["x", "y", "z"]).min()
        else:
            raise ValueError("`orientation` must be 'mean' or 'min'.")
        
        df = df.reset_index(level=[0,1,2])
        df = df.drop(labels=["q0", "q1", "q2", "q3"], axis=1)

        # Replace NaN with Inf
        df = df.pivot(columns="x", index="y", values="PE")
        df = df.replace(to_replace=np.nan, value=np.inf)

        # If Convert pandas dataframe to PIL image
        # image_array = np.nan_to_num(
        #     df.array, nan=np.inf, posing=np.inf, neginf=-np.inf
        # )
        return df.array

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
        filename: str,
        extents: list[list[float]] | None = None
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
        if extents:
            spacing = [
                (extents[0][1] - extents[0][1])/array.shape[2], # TODO: check the indexing
                (extents[1][1] - extents[1][1])/array.shape[2],
                (extents[2][1] - extents[2][1])/array.shape[2],
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
            self._save_3d_array_to_vti()
        
    def plot(
        self,
        core_shape,
        rotate_core_deg,
        vmin,
        vmax,
        slice_x,
        slice_y,
        slice_lim,
        show_cbar,
        core_scale_factor
    ):
        # TODO
        pass
