# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from __future__ import annotations

from collections import defaultdict
from copy import copy, deepcopy
import inspect
import itertools
import importlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Literal
from packaging.version import Version

import hoomd
import numpy as np
import plotly

from .top_level_functions import plot_layout
from . import util

REQUIRED_PARENT_CLASS = hoomd.md.pair.pair.Pair
EXCLUDED_TYPE_STRINGS = [
    "hoomd.md.pair.pair.Pair",
    "hoomd.md.pair.aniso.AnisotropicPair",
    "hoomd.md.pair.aniso.Patchy",
    "hoomd.md.pair.friction.FrictionalPair"
]

class Interaction:
    """The data for making and parameterizing a `HOOMD-blue MD pair potential`_.

    .. _HOOMD-blue MD pair potential: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/pair/pair.html

    This class is self-validating: it cannot be instantiated or modified without
    adhering to the :ref:`interaction-schema`. This rule is enforced by
    :py:meth:`~p4.interaction.Interaction.validate`.

    Instantiate an interaction directly using its constructor, or create one by
    parsing existing HOOMD-blue objects using
    :py:meth:`~p4.interaction.Interaction.from_hoomd_simulation`,
    :py:meth:`~p4.interaction.Interaction.from_hoomd_integrator`, or
    :py:meth:`~p4.interaction.Interaction.from_hoomd_pair`. Interactions can
    also be saved to and created from JSON files using
    :py:meth:`~p4.interaction.Interaction.to_json` and
    :py:meth:`~p4.interaction.Interaction.from_json`.

    Interactively visualize an isotropic interaction's potential energy curve
    using :py:meth:`~p4.interaction.Interaction.plot`.

    Parameters
    ----------
    hoomd_class : hoomd.md.pair.Pair
        A HOOMD-blue MD pairwise potential type. Must be in the
        ``hoomd.md.pair`` module or one of its submodules. Cannot be one of the
        following types: :py:class:`~hoomd.md.pair.aniso.AnisotropicPair`,
        :py:class:`~hoomd.md.pair.aniso.Patchy`,
        :py:class:`~hoomd.md.pair.friction.FrictionalPair`.
    initial_args : dict
        All parameters (except for ``nlist``) that are needed for instantiating
        the class from its constructor.
    default_params : dict
        The names and values of parameters that will be set by default for
        all particle types and pairs of types. To determine the names and values
        required for parametrizing the HOOMD-class, consult the
        `documentation`_.
    typed_params : dict
        A mapping of particle types and pairs of types to parameter names and
        values. These names and values will override the default parameters for
        those particle types. Consult the documentation to determine the
        allowed parameter names and values.

    
    .. _documentation: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/module-pair.html


    Example
    -------

    .. code-block:: python
        :caption: A Lennard-Jones potential that allows A-B interactions but not A-A or B-B.

        import p4
        import hoomd
        
        interaction = p4.Interaction(
            hoomd_class=hoomd.md.pair.LJ,
            initial_args=dict(),
            default_params=dict(
                r_cut=0,
                params=dict(epsilon=0, sigma=1)
            ),
            typed_params={
                ("A", "B"): dict(
                    r_cut=5.0,
                    params=dict(epsilon=1, sigma=1)
                )
            }
        )
    """
    def __init__(
        self,
        hoomd_class: hoomd.md.pair.Pair,
        initial_args: dict,
        default_params: dict,
        typed_params: dict,
    ):
        # Validate inputs
        def cls_to_str(cls):
            """A string representation of the class path, including its name."""
            return cls.__module__ + "." + cls.__name__
        
        if not issubclass(hoomd_class, hoomd.md.pair.Pair):
            raise TypeError(
                "Incorrect `hoomd_class`: must be a subclass of "
                + "hoomd.md.pair.Pair"
            )
        if cls_to_str(hoomd_class) in EXCLUDED_TYPE_STRINGS:
            raise TypeError(
                "Incorrect `hoomd_class`: must be a subclass of "
                + f"{cls_to_str(hoomd_class)}"
            )
        if (
            hoomd_class is hoomd.md.pair.Table
            and Version(hoomd.version.version) < Version("7.0.2")
        ):
            raise TypeError(
                "`hoomd.md.pair.Table` is not compatible with p4 below "
                + "HOOMD-blue version 7.0.2."
            )
        
        # Set instance attributes
        self._hoomd_class = hoomd_class
        self._initial_args = util.data.sanitize(initial_args)
        self._default_params = util.data.sanitize(default_params)
        self._typed_params = util.data.sanitize(typed_params)

        # Validate instance attributes
        self.validate()

    def validate(self):
        """Ensure this interaction adheres to the :ref:`interaction-schema`.
        
        .. note:
            Unlike :py:class:`~p4.Body` and :py:class:`~p4.Arrangement`,
            interaction validation requires instance methods and so it is an
            instance method that evaluates the instance's attributes, rather
            than a class method that evaluates keyword arguments.
        """
        # Ensure the hoomd class can be instantiated
        nlist = hoomd.md.nlist.Tree(2)
        try:
            _ = self.hoomd_class(nlist=nlist, **self.initial_args)
        except ValueError as e:
            raise ValueError(
                "Incorrect `initial_args`: HOOMD class cannot be instantiated. "
                + "See traceback for details."
            ) from e
        
        # Ensure the hoomd class can be parameterized
        nlist = hoomd.md.nlist.Tree(2)
        try:
            _ = self.to_hoomd_pair()
        except (AttributeError, KeyError) as e:
            raise ValueError(
                "Incorrect `default_params` or `typed_params`: an instance of "
                + "the HOOMD class cannot be parameterized. See traceback for "
                + "details."
            ) from e
        
        # Ensure the parameterized hoomd class can be used in a simple example
        # simulation
        nlist = hoomd.md.nlist.Tree(2)
        test_types = self._interacting_types("all")
        if not test_types:
            test_types = ["A", "B"] # catch case with no typed params
        simulation = hoomd.util.make_example_simulation(
            particle_types=test_types
        )
        if not any("params" in v for v in self.typed_params.values()):
            max_r_cut = self.default_params["r_cut"]
        else:
            max_r_cut = max([
                v.get("r_cut", 0) for v in self.typed_params.values()
            ])
        
        simulation = hoomd.util.make_example_simulation(
            particle_types=test_types
        )
        s = 10 * max_r_cut
        simulation.state.set_box([s, s, s, 0, 0, 0])
        simulation = util.simulation.add_integrator(simulation)
        simulation = util.simulation.add_interaction(
            simulation=simulation,
            nlist=nlist,
            interaction=self
        )

        box_length = 10 * max(max_r_cut, 1.0)
        simulation.state.set_box([box_length, box_length, box_length, 0, 0, 0])
        simulation = util.simulation.add_integrator(simulation)
        simulation = util.simulation.add_interaction(
            simulation=simulation,
            nlist=nlist,
            interaction=self
        )

        try:
           simulation.run(0)
        except RuntimeError as e:
            raise ValueError(
                "Incorrect `default_params` or `typed_params`: the "
                + "parameterized HOOMD instance cannot be used in a running "
                + "simulation. See traceback for details."
            ) from e

    # ------------------------------- PROPERTIES -------------------------------

    @property
    def hoomd_class(self) -> hoomd.md.pair.Pair:
        """The constructor for the HOOMD-blue class.
        
        To change the constructor, create a new ``Interaction`` instance.
        """
        return self._hoomd_class

    @property
    def initial_args(self) -> dict:
        """The parameters for instantiating the HOOMD-blue class."""
        return self._initial_args

    @initial_args.setter
    def initial_args(self, value: dict[str, Any]):
        """Set the parameters for instantiating HOOMD-blue class."""
        original_value = deepcopy(self._initial_args)
        self._initial_args = util.data.sanitize(value)
        try:
            self.validate()
        except:
            self._initial_args = original_value
            raise

    @property
    def default_params(self) -> dict:
        """The default parameters for all single and pair types."""
        return self._default_params

    @default_params.setter
    def default_params(self, value: dict[str, Any]):
        """Set the default parameters for all single and pair types."""
        original_value = deepcopy(self._default_params)
        self._default_params = util.data.sanitize(value)
        try:
            self.validate()
        except:
            self._default_params = original_value
            raise

    @property
    def typed_params(self) -> dict:
        """Parameters for specific single and pair types."""
        return self._typed_params

    @typed_params.setter
    def typed_params(self, value: dict):
        """Set parameters for specific single and pair types."""
        original_value = deepcopy(self._typed_params)
        self._typed_params = util.data.sanitize(value)
        try:
            self.validate()
        except:
            self._typed_params = original_value
            raise

    # ---------------------------------- FROM ----------------------------------

    @classmethod
    def from_hoomd_simulation(
        cls,
        simulation: hoomd.Simulation,
    ) -> list[Interaction]:
        """Parse a HOOMD-blue `Simulation`_ to create interactions.

        .. _Simulation: https://hoomd-blue.readthedocs.io/en/latest/hoomd/simulation.html

        This is a convenience method that is equivalent to
        
        .. code-block::

            p4.Interaction.from_hoomd_integrator(simulation.operations.integrator)

        Parameters
        ----------
        simulation : hoomd.Simulation
            The simulation whose integrator contains the pair potentials.
        """
        if simulation.operations.integrator is None:
            raise ValueError("`simulation` must have an integrator")
        if not simulation.operations.integrator.forces:
            raise ValueError("integrator must have forces")
        return cls.from_hoomd_integrator(simulation.operations.integrator)

    @classmethod
    def from_hoomd_integrator(
        cls,
        integrator: hoomd.md.Integrator
    ) -> list[Interaction] | Interaction:
        """Parse a HOOMD-blue `MD Integrator`_ to create interactions.

        .. _MD Integrator: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/integrator.html#hoomd.md.Integrator
        
        This is a convenience method that is equivalent to

        .. code-block::

            [p4.Interaction.from_hoomd_pair(p) for p in integrator.forces]

        Parameters
        ----------
        integrator : hoomd.md.Integrator
            The integrator that contains the pair potentials.
        """
        if not integrator.forces:
            raise ValueError("`integrator` must have forces")
        return [cls.from_hoomd_pair(p) for p in integrator.forces]

    @classmethod
    def from_hoomd_pair(cls, pair: hoomd.md.pair.Pair) -> Interaction:
        """Parse a HOOMD-blue `MD pair potential`_ to create an interaction.

        .. _MD pair potential: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/pair/pair.html
        
        Parameters
        ----------
        pair : hoomd.md.pair.Pair
            The pair potential.
        """
        # Get typeparam dict as a native Python object
        tpd = {k: v.to_base() for k, v in pair._typeparam_dict.items()}

        # Constructor
        hoomd_class = type(pair)
        
        # Parse initial args
        initial_args = pair._param_dict.to_base()

        # Delete wrongly identified initial args [Review]
        if initial_args.get("mode") == "none":
            del initial_args["mode"]

        # Parse default params
        base_instance = hoomd_class(**initial_args)
        default_params = {}
        for param_name in tpd:
            pair_param_default = getattr(pair, param_name).default
            base_param_default = getattr(base_instance, param_name).default
            if pair_param_default != base_param_default:
                default_params[param_name] = pair_param_default
        
        # Delete unnecessary initial args
        del initial_args["nlist"]
        if initial_args.get("tail_correction") is False:
            del initial_args["tail_correction"]
        
        # Get a list of pairs of interacting types. Pairs of types are
        # 'interacting' if they have non-zero r_cut values.
        interacting_types = [
            type_pair for type_pair, value in tpd["r_cut"].items() if value > 0
        ]

        # Parse typed params. It is possible that the pair might specify
        # non-default param values for types that are also not interacting. In
        # such cases, put those param values into default_params only if the
        # param name is not already there.
        typed_params = {}

        for param_name, typeparam in tpd.items():
            for type_name, param_value in typeparam.items():
                # Typed params
                if type_name in interacting_types:
                    if type_name not in typed_params:
                        typed_params[type_name] = {}
                    if hasattr(param_value, "to_base"):
                        typed_params[type_name][param_name] = (
                            param_value.to_base()
                        )
                    else:
                        typed_params[type_name][param_name] = param_value

                # Default params
                else:
                    if param_name not in default_params:
                        if hasattr(param_value, "to_base"):
                            default_params[param_name] = param_value.to_base()
                        else:
                            default_params[param_name] = param_value

        kwargs=dict(
            hoomd_class=hoomd_class,
            initial_args=initial_args,
            default_params=default_params,
            typed_params=typed_params,
        )

        return cls(**kwargs)

    @classmethod
    def from_json(
        cls,
        filename: os.PathLike,
        json_path: str = "p4.interaction"
    ):
        """Create an interaction from JSON.

        A JSON path may be provided to control the location that the
        interaction data is retrieved from. See :py:meth:`to_json` for an
        explanation of JSON path formatting.

        .. note::
            ``typed_params`` is not a JSON-compliant dictionary, so in the JSON
            representation it is changed to an array of dictionaries, where each
            dictionary represents a [key, value] pair from the Python
            representation.
        
        Parameters
        ----------
        filename : os.PathLike
            The name or path of the JSON file.
        json_path : str, default='p4.interaction'
            The location within the JSON file to retrieve the interaction's
            representation from.
        
        Raises
        ------
        ValueError
            If the JSON file does not have the keys and values required for
            instantiating an Interaction.
        """
        with open(filename, "r") as f:
            data = json.load(f)

        if json_path == ".":
            data = cls._convert_json_dict(data)            
        
        else:
            current_container = data
            for name in json_path.split("."):
                current_container = current_container[name]
            data = cls._convert_json_dict(current_container)
        
        required_args = [
            "hoomd_class",
            "initial_args",
            "default_params",
            "typed_params"
        ]
        for required_arg in required_args:
            if required_arg not in data:
                raise ValueError(
                    f"Required arg {required_arg} not found in '{filename}' "
                    + f"at path '{json_path}'."
                )
        for key in copy(data):
            if key not in required_args:
                del data[key]

        return cls(**data)

    @classmethod
    def _convert_json_dict(cls, data: dict):
        """Convert a JSON-compliant dict into an instantiation-ready dict."""
        # Convert hoomd class path from string to type
        class_path = data["hoomd_class"]
        module_name, class_name = class_path.rsplit(".", 1)
        hoomd_class = getattr(importlib.import_module(module_name), class_name)
        data["hoomd_class"] = hoomd_class

        # Convert typed params back into a dict with tuples and strings as keys
        typed_params = {}
        for item in data["typed_params"]:
            if isinstance(item["types"], str):
                key = item["types"]
            elif isinstance(item["types"], list):
                key = tuple(item["types"])
            typed_params[key] = item["params"]
        
        data["typed_params"] = typed_params
        
        return data

    # ----------------------------------- TO -----------------------------------

    def to_hoomd_pair(
        self,
        nlist: hoomd.md.nlist.NeighborList | None = None,
    ) -> hoomd.md.pair.Pair:
        """Return an instance of the HOOMD-blue class.
        
        Parameters
        ----------
        nlist : hoomd.md.nlist.NeighborList, optional
            The neighbor list with which to instantiate the class. Pass an
            existing neighbor list to add it to the instance. If not provided,
            a bounding volume hierarchy-based neighbor list is created on the
            fly.
        """
        if nlist is None:
            nlist = hoomd.md.nlist.Tree(2)
        
        instance = self.hoomd_class(nlist, **self.initial_args)

        # Set default params
        for param_name, param_value in self.default_params.items():
            getattr(instance, param_name).default = param_value
        
        # Set typed params
        for type_name, type_params in self.typed_params.items():
            for param_name, param_value in type_params.items():
                getattr(instance, param_name)[type_name] = param_value
                
        return instance

    @classmethod
    def _parse_params(
        cls,
        hoomd_class: hoomd.md.pair.Pair,
        initial_args: dict,
        single_or_pair: Literal["single", "pair"],
        required_or_optional: Literal["required", "optional", "all"]
    ):
        """Return a param dictionary for a given hoomd class.

        Parameters
        ----------
        single_or_pair : 'single' or 'pair'
            Whether to return parameters that are assigned to single particle
            types or pairs of particle types.
        required_or_optional : 'required' or 'optional' or 'all'
            Whether to return parameters that are required, optional, or both.
        """
        def flatten(array):
            """Flatten arbitrarily nested arrays, ignoring strings and bytes."""
            for item in array:
                if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                    yield from flatten(item)
                else:
                    yield item

        def is_required(tp_default_item):
            """Whether input is a required arg or 1+ nested arrays thereof."""
            return (
                (tp_default_item is hoomd.data.typeconverter.RequiredArg)
                or (
                    isinstance(tp_default_item, Iterable)
                    and any([
                        subitem is hoomd.data.typeconverter.RequiredArg
                        for subitem in flatten(tp_default_item)
                    ])
                )
            )

        def parse_typeparam_default_item_dict(d, required_or_optional):
            """Return dicts of values for typeparams.
            
            Values for required typeparams are marked "required". Values for
            optional typeparams are the defaults.
            
            This goes two layers deep.
            """  
            subdict = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    subsubdict = {}
                    for sk, sv in v.items():
                        if isinstance(sv, dict):
                            raise Exception("typeparam dict recursion depth exceeded.")
                        
                        if is_required(sv):
                            if required_or_optional in ["required", "all"]:
                                subsubdict[sk] = "required"
                        else:
                            if required_or_optional in ["optional", "all"]:
                                subsubdict[sk] = sv
                    
                    if len(subsubdict.keys()) > 0:
                        subdict[k] = subsubdict
                
                else:
                    if is_required(v):
                        if required_or_optional in ["required", "all"]:
                            subdict[k] = "required"
                    else:
                        if required_or_optional in ["optional", "all"]:
                            subdict[k] = v
                
            return subdict
        
        params = {}
        tpd = hoomd_class(
            nlist=hoomd.md.nlist.Tree(2),
            **initial_args
        )._typeparam_dict

        for name, typeparam in tpd.items():
            something_to_add = False    # TODO: this flag indicates refactoring needed

            # If the typeparam is a dictionary mapping names to some subtypeparams,
            # then each of those subtypeparams must also be checked. This process
            # is not recursive - i.e., it does not perform this dictionary check on
            #  each of those subtypeparams.
            if type(typeparam.default) is dict:
                value = parse_typeparam_default_item_dict(typeparam.default, required_or_optional)
                
                if len(value.keys()) > 0:
                    something_to_add = True

            # If the typeparam is not a dictionary, everything's simple
            else:
                if is_required(typeparam.default):
                    if required_or_optional in ["required", "all"]:
                        value = "required"
                        something_to_add = True
                else:
                    if required_or_optional in ["optional", "all"]:
                        value = typeparam.default
                        something_to_add = True
            
            # Add the attribute value to the container
            if something_to_add:
                if typeparam._indexer.len_key == 1:
                    if single_or_pair in ["single", "all"]:
                            params[name] = value
                
                elif typeparam._indexer.len_key == 2:
                    if single_or_pair in ["pair", "all"]:
                            params[name] = value

                else:
                    raise Exception(
                        f"Encountered an unknown typearam type: '{name}' has "
                        f"{typeparam._indexer.len_key=}, but only 1 and 2 are allowed."
                    )
        
        return params
    
    def to_json(
        self,
        filename: os.PathLike,
        json_path: str = "p4.interaction",
        indent: str | int | None = None
    ):
        """Export the interaction to JSON.
        
        If ``filename`` points to an existing file, a JSON path may be specified
        to ensure the interaction data does not clash with existing data in the
        file.

        A JSON path that looks like ``'a.b.c'`` represents the following
        location:

        .. code-block::

            <root>
            └─ a
               └─ b
                  └─ c
                     └─ <data will go here>

        If the path specifies a location that already contains data, the
        contents of that location may be overwritten.

        .. note::
            ``typed_params`` is not a JSON-compliant dictionary, so in the JSON
            representation it is changed to an array of dictionaries, where each
            dictionary represents a [key, value] pair from the Python
            representation.
                     
        Parameters
        ----------
        filename : os.PathLike
            The name or path of the JSON file.
        json_path : str or None, default='p4.interactions'
            The location within the JSON file to put the interaction's
            representation in. Only used if ``filename`` already exists. If
            ``'.'`` is provided, then the representation is placed at the root
            level.
        indent : str or int, optional
            The string or number of spaces to use when indenting newlines in the
            JSON file. If not provided, there are no newlines.
        """
        path = Path(filename)
        data = self._to_json_dict()

        if not path.exists():
            path.touch()
            existing_data = {}
        else:
            with open(path, "r") as f:
                existing_data = json.load(f)
            
        if json_path == ".":
            for k, v in data.items():
                existing_data[k] = v
        
        else:
            names = json_path.split(".")
            current_container = existing_data
            for i, name in enumerate(names):
                if name not in current_container:
                    current_container[name] = {}
                if i < (len(names) - 1):
                    current_container = current_container[name]
                else:
                    # try to write alongside existing data if possible...
                    if isinstance(current_container[name], dict):
                        current_container[name].update(data)
                    elif isinstance(current_container[name], list):
                        current_container[name].append(data)
                    # ... and insert or overwrite if not
                    else:
                        current_container[name] = data

        with open(path, "w") as f:
            json.dump(existing_data, f, indent=indent)

    def _to_json_dict(self) -> dict:
        """Return a JSON-compliant dictionary representing this interaction."""
        data = dict(
            hoomd_class=self.hoomd_class,
            initial_args=self.initial_args,
            default_params=self.default_params,
            typed_params=self.typed_params,
        )

        # The hoomd class must be a string
        data["hoomd_class"] = (
            f"{data["hoomd_class"].__module__}.{data["hoomd_class"].__name__}"
        )

        # The typed params must not have tuples for keys
        typed_params = []
        for k, v in data["typed_params"].items():
            typed_params.append({
                "types": k,
                "params": v
            })
        data["typed_params"] = typed_params

        return data

    # -------------------------------- PLOTTING --------------------------------

    def plot(
        self,
        r: list[float],
        type_pairs: list[tuple] | None = None,
        pair_styles: dict[tuple, dict] | None = None,
        cmap: str = "Pastel",
        ylim: list[float] | None = None,
        include_default: bool = False,
        mode: Literal["lines", "lines+markers", "markers"] = "lines",
        marker_size: float = 6,
        line_width: float = 2,
        **kwargs
    ) -> tuple[plotly.graph_objects.Figure, list]:
        """Plot the interaction potential energy curve for pairs of types.
        
        Plotting is only supported for isotropic interactions.

        Styles may be specified for specific pairs of types. A style must
        specified as a dictionary which may have the following keys and values:

        * **mode** [``'lines'``, ``'lines+markers'``, ``'markers'``] The
          `drawing mode`_ for the plotly trace.

        * **color** [``str``] - The symbol's color. Plotly accepts color strings
          in `standard HTML/CSS formats`_ (for example, `rgb`_), as well as
          `many named colors`_.
        
        * **marker_size** [``float`` > 0] - The marker size.

        * **line_width** [``float`` > 0] - The line width.

        .. _drawing mode: https://plotly.com/python/reference/scatter/#scatter-mode
        .. _standard HTML/CSS formats: https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/color
        .. _rgb: https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Values/color_value/rgb
        .. _many named colors: https://plotly.com/python/css-colors/

        Parameters
        ----------
        r : array of floats
            The distances (x axis) at which to calculate the potential energy.
        type_pairs : array of tuples of strings, optional
            The pairs of types to plot the interaction for. If not provided, all
            pairs are used.
        pair_styles : dict, optional
            Style dictionaries to apply to the markers. If not provided, a
            default set of styles is used.
        cmap : str, default='Pastel'
            The name of the Plotly colormap to use. This must be a continuous
            or qualitative colorscale.
        ylim : list of floats, optional
            The lower and upper limits of the y axis. If not provided, limits
            will be calculated that attempt to give a clear view of the data.
        include_default : bool, default=False
            Whether to include the curve defined by ``default_params`` in
            the plot.
        marker_mode : 'lines+markers', 'lines', or 'markers', default='lines'
            Whether to show only lines, only markers, or both.
        marker_size : float, default=6
            The size of the marker in pixels.
        line_width : float, default=2
            The width of the line in pixels.
        **kwargs
            Other keyword arguments are passed to ``p4.plot_layout()``.
            TODO: add link.
        
        Returns
        -------
        figure, traces
            The plot figure and its associated traces.
        """
        # TODO: check if there's another way to structure package to prevent
        # imports here
        from .body import Body
        from .field import Field
        from .system import System
        
        # Defaults
        if not pair_styles:
            pair_styles = {}
            
        DEFAULT_STYLE = dict(
            color=None,
            marker_size=marker_size,
            line_width=line_width,
            mode=mode,
        )
        
        for pair, style in copy(pair_styles).items():
            for key, default_value in DEFAULT_STYLE.items():
                if key not in style:
                    style[key] = default_value
            pair_styles[pair] = style
        
        pair_styles = defaultdict(
            lambda: defaultdict(None, DEFAULT_STYLE),
            pair_styles
        )
        
        # Ensure the interaction is isotropic
        if self.hoomd_class.__module__ != "hoomd.md.pair.pair":
            raise TypeError(
                "Plotting is only supported for isotropic interactions, but "
                + f"hoomd_class is from the {self.hoomd_class.__module__} "
                + "module."
            )

        # Ensure there is a working colormap
        try:
            _ = plotly.colors.get_colorscale(cmap)
        except plotly.exceptions.PlotlyError:
            try:
                getattr(plotly.colors.qualitative, cmap.capitalize())
            except AttributeError:
                raise ValueError(
                    "`cmap` is not a valid name for a continuous or "
                    "qualitative plotly colorscale."
                )

        # If no type pairs are provided, use all of them
        if type_pairs is None:
            type_pairs = list(self.typed_params.keys())
        
        if include_default:
            type_pairs = ["default"] + type_pairs

        # Calculate kwargs for the measure function call
        # still needed for every call: system, nlist
        positions = np.array([[0 + value, 0, 0] for value in r])
        orientations = np.array([[[1, 0, 0, 0]] for _ in positions])

        max_r_cut = 0
        max_r_cut = max(max_r_cut, self.initial_args.get("default_r_cut", 0))
        max_r_cut = max(max_r_cut, self.default_params.get("r_cut", 0))
        for param_dict in self.typed_params.values():
            max_r_cut = max(max_r_cut, param_dict.get("r_cut", 0))

        box_length = 1.1 * 2 * max(max_r_cut, max(r))

        measure_kwargs = dict(
            quantities="U",
            positions=positions,
            orientations=orientations,
            simulation_box=[box_length, box_length, box_length, 0, 0, 0],
            included_interactions=[self],
            gsd_filename=None
        )

        # Build plot traces pair by pair
        traces = []

        for i, pair in enumerate(type_pairs):
            if pair == "default":
                probe=Body("skvblejy")
                analyte=Body("dhytkgvle")
            else:
                probe=Body(pair[0])
                analyte=Body(pair[1])
            
            system = System(probe, analyte, [self])
            
            table = util.simulation.measure(
                system=system,
                nlist=hoomd.md.nlist.Tree(2),
                **measure_kwargs
            )
            table = util.data.clean_header(table)
            table.seek(0)

            field = Field(
                np.rec.array(
                    np.genfromtxt(
                        table,
                        names=True,
                        dtype=None,
                        delimiter=",",
                        encoding="utf-8"
                    )
                )
            )

            style = pair_styles[pair]
            
            if style["color"] is None:
                try:
                    color = plotly.colors.get_colorscale(cmap)[i]
                except plotly.exceptions.PlotlyError:
                    color = getattr(
                        plotly.colors.qualitative,
                        cmap.capitalize()
                    )[i]
            else:
                color=style["color"]

            _, trace = field.plot(
                slice=dict(z=0, y=0),
                marker_color_1d=color,
                marker_mode_1d=style["mode"],
                marker_size_1d=style["marker_size"],
                line_width_1d=style["line_width"],
                show_title=False,
            )
            trace["name"] = str(pair)

            traces.append(trace)

        # Create and style the figure
        figure = plotly.graph_objects.Figure()
        figure.add_traces(traces)

        allowed_kwarg_names = (
            inspect.signature(plot_layout).parameters.keys()
        )
        layout_kwargs = {
            k: v for k, v in kwargs.items() if k in allowed_kwarg_names
        }
        layout = plot_layout(
            slice=dict(z=0, y=0),
            **layout_kwargs
        )
        figure.update_layout(**layout)

        figure.update_layout(xaxis=dict(title="r", range=[min(r), max(r)]))
        
        if ylim is None:
            overall_min = min(min(s["y"]) for s in figure.data)
            overall_max = max(max(s["y"]) for s in figure.data)
            min_too_large = overall_min < -1e2
            max_too_large = overall_max > 1e2

            if min_too_large and not max_too_large:
                ylim = [-0.5 * np.abs(overall_max), 1.1 * overall_max]
            elif max_too_large and not min_too_large:
                ylim = [1.1 * overall_min, 0.5 * np.abs(overall_min)]
            elif max_too_large and min_too_large:
                min_magnitude = min(min(np.abs(s["y"])) for s in figure.data)
                ylim = [-2 * min_magnitude, 2 * min_magnitude]
            else:
                ylim = [overall_min, overall_max]
        
        figure.update_layout(yaxis=dict(range=ylim))
        figure.update_layout(width=500, height=500)

        return figure, traces

    # ---------------------------------- OTHER ---------------------------------

    def _interacting_types(
        self,
        category=Literal["single", "pair", "all"]
    ) -> list[str] | list[tuple[str, str]]:
        """A list of particle types from ``typed_params``.
        
        Parameters
        ----------
        category : 'single', 'pair', or 'all'
            Which types to return. If set to 'single', single types (strings)
            will be returned. If set to 'pair', pair types (2-tuples of strings) 
            will be returned. If set to 'all', a union of all single types and
            the contents of all pair types will be returned.
        """
        if category == "single":
            return [key for key in self.typed_params if isinstance(key, str)]

        elif category == "pair":
            return [
                key
                for key in self.typed_params
                if isinstance(key, Iterable)
                    and not isinstance(key, (str, bytes))
                    and len(key) == 2
            ]
        
        elif category == "all":
            types = [t for t in self._interacting_types("single")]
            for p in self._interacting_types("pair"):
                for t in p:
                    if t not in types:
                        types.append(t)
            return types
        
    def __eq__(self, other) -> bool:
        """Interactions are equal if their properties are equal."""
        return (
            type(self.hoomd_class) is type(other.hoomd_class)
            and self.initial_args == other.initial_args
            and self.default_params == other.default_params
            and self.typed_params == other.typed_params
        )

    def __repr__(self) -> str:
        return (
            "Interaction ("
            + f"\n\thoomd_class={self.hoomd_class},"
            + f"\n\tinitial_args={self.initial_args},"
            + f"\n\tdefault_params={self.default_params},"
            + f"\n\ttyped_params={self.typed_params}"
            + "\n)"
        )
