# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

import itertools
from typing import Iterable
import hoomd

import p4.util


class Interaction:
    """The data for instantiating and parameterizing a hoomd.md.pair potential.

    `Interaction` is self-validating, i.e., every instance is guaranteed to
    successfully create and parameterize its provided HOOMD pair potential.

    # Example
    ```python
    i = Interaction(
        hoomd_class=hoomd.md.pair.LJ,
        initial_args=dict(),
        no_params=dict(
            r_cut=0,
            params=dict(epsilon=0, sigma=1)
        ),
        all_types=["A", "B"],
        yes_params={
            ("A", "B"): dict(
                r_cut=5.0},
                params=dict(epsilon=1, sigma=1)}
            )
        }
    )
    ```

    Parameters
    ----------
    hoomd_class : hoomd.md.pair.Pair
        The constructor for the HOOMD class. Must be in `hoomd.md.pair`.
    initial_args : dict[str, float | str]
        All parameters (that aren't `nlist`) that are needed for instantiating
        the class from its constructor.
    no_params : dict
        The names and values of parameters that will be set for the
        **non-interacting** individual and pairs of particle types. These values
        are the "default" for particle types covered by this interaction.
        To determine the required params for `hoomd_class`, consult the docs
        or use the static method `Interaction.get_param_schema()`.
    all_types : list[str]
        The particle types that are covered by this interaction. This list may
        include one or more effectively non-interacting types, whose pairwise
        interactions are parameterized with `no_params`.
    yes_params : dict
        The names and type-parameterized values of parameters that will be set
        for the **interacting** individual and pairs of particle types. To
        determine the required params for `hoomd_class`, consult the docs
        or use the static method `Interaction.get_param_schema()`.
    """
    def __init__(
        self,
        hoomd_class: hoomd.md.pair.Pair,
        initial_args: dict[str, float | str],
        no_params: dict[str, float],
        all_types: list[str],
        yes_params: dict[str, float],
    ):
        self.hoomd_class = hoomd_class
        self.initial_args = initial_args
        self.no_params = no_params
        self.all_types = [str(t) for t in all_types]
        self.yes_params = yes_params

        self.validate()

    def validate(self):
        """Ensure this Interaction behaves properly."""
        # Ensure all "yes" types are provided in all_types
        for type_name in self.yes_params:
            if isinstance(type_name, Iterable):
                for t in type_name:
                    reason = (
                        f"yes type pair '{type_name}' contains '{t}', which is "
                        + "not in all_types"
                    )
                    assert t in self.all_types, reason
            else:
                reason = f"yes type '{t}' is not in all_types"
                assert type_name in self.all_types, reason

        # Ensure the hoomd class can be instantiated
        nlist = hoomd.md.nlist.Cell(2)
        try:
            _ = self.to_hoomd_instance(nlist)
        except ValueError as e:
            msg = "Validation failed: the HOOMD class cannot be instantiated."
            raise ValueError(msg) from e
        
        # Ensure the hoomd class can be parameterized
        nlist = hoomd.md.nlist.Cell(2)
        try:
            _ = self.to_parameterized_hoomd_instance(nlist)
        except AttributeError as e:
            msg = "Validation failed: the HOOMD class cannot be parameterized."
            raise ValueError(msg) from e
        
        # Ensure the parameterized hoomd class can be used in a simulation
        nlist = hoomd.md.nlist.Cell(2)
        simulation = hoomd.util.make_example_simulation(
            particle_types=self.all_types
        )
        if not any("params" in v for v in self.yes_params.values()):
            max_r_cut = self.no_params["r_cut"]
        else:
            max_r_cut = max([
                v.get("r_cut", 0) for v in self.yes_params.values()
            ])
        box_length = 10 * max(max_r_cut, 1.0)
        simulation.state.set_box([box_length, box_length, box_length, 0, 0, 0])
        simulation = p4.util.add_integrator(simulation)
        simulation = p4.util.add_interaction(simulation, nlist, self)

        try:
           simulation.run(0)
        except Exception as e:
            msg = (
                "Validation failed: the parameterized HOOMD instance cannot be "
                + "used in a running simulation. See traceback for details."
            )
            raise ValueError(msg) from e

    @property
    def yes_single_types(self):
        """A list of all single particle types for which yes params are given."""
        return [
            key
            for key in self.yes_params
            if isinstance(key, str) and key in self.all_types
        ]

    @property
    def yes_pair_types(self):
        """A list of all pairs of particle types for which yes params are given."""
        return [
            key
            for key in self.yes_params
            if isinstance(key, Iterable)
                and len(key) == 2
                and all(i in self.all_types for i in key)
        ]

    def to_hoomd_instance(
        self,
        nlist: hoomd.md.nlist.NeighborList
    ) -> hoomd.md.pair.Pair:
        """Return an unparameterized instance of the HOOMD class.

        Parameters
        ----------
        nlist : hoomd.md.nlist.NeighborList
            The neighbor list to use.

        Returns
        -------
        instance

        Raises
        ------
        ValueError
            If the initial inputs are wrong.
        """
        try:
            return self.hoomd_class(nlist, **self.initial_args)
        except (TypeError, hoomd.error.TypeConversionError) as e:
            msg = "'initial_args' are wrong. See traceback for details."
            raise ValueError(msg) from e

    def to_parameterized_hoomd_instance(
        self,
        nlist: hoomd.md.nlist.NeighborList,
    ) -> hoomd.md.pair.Pair:
        """Return a parameterized instance of the HOOMD class.
        
        Parameters
        ----------
        nlist : hoomd.md.nlist.NeighborList
            The neighbor list to use.
        all_types : list[str]
            The names of all the particle types for which the instance should be
            parameterized.

        Returns
        -------
        parameterized_instance

        Raises
        ------
        AttributeError
            If the typed params are wrong.
        """
        def wrong_type_msg(param, no_or_yes, single_or_pair, hoomd_class):
            return (
                f"'{param}' was provided as a {no_or_yes} {single_or_pair}"
                f"-typed-param, but no such param was found in hoomd "
                f"class '{hoomd_class}'."
            )

        instance = self.to_hoomd_instance(nlist)
        
        # Calculate the pairwise combinations of all types and interacting types
        all_type_pairs = list(
            itertools.combinations_with_replacement(self.all_types, 2)
        )

        single_typed_params = self._parse_params("single", "all")
        pair_typed_params = self._parse_params("pair", "all")

        # Set no single-typed params
        for a_t in self.all_types:
            for name, typed_param in self.no_params.items():
                if name in single_typed_params:
                    try:    # TODO: probably don't need try-block if there's a parsing method
                        getattr(instance, name)[a_t] = typed_param
                    except AttributeError:
                        raise AttributeError(
                            wrong_type_msg(
                                name, "no", "single", self.hoomd_class
                            )
                        )

        # Set no pair-typed params
        for a_p in all_type_pairs:
            for name, typed_param in self.no_params.items():
                if name in pair_typed_params:
                    try:
                        getattr(instance, name)[a_p] = typed_param
                    except AttributeError:
                        raise AttributeError(
                            wrong_type_msg(
                                name, "no", "pair", self.hoomd_class
                            )
                        )

        # Modify yes single-typed params
        for y_t in self.yes_single_types:
            for param_name, param_value in self.yes_params[y_t].items():
                try:
                    getattr(instance, param_name)[y_t] = param_value
                except AttributeError:
                    raise AttributeError(
                        wrong_type_msg(
                            param_name, "yes", "single", self.hoomd_class
                        )
                    )

        # Modify yes pair-typed params
        for y_p in self.yes_pair_types:
            for param_name, param_value in self.yes_params[y_p].items():
                try:
                    getattr(instance, param_name)[y_p] = param_value
                except AttributeError:
                    raise AttributeError(
                        wrong_type_msg(
                            param_name, "yes", "pair", self.hoomd_class
                        )
                    )

        return instance

    def _parse_params(self, single_or_pair, required_or_optional):
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
            """Recursively flatten arbitrarily nested arrays, ignoring strings and bytes."""
            for item in array:
                if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                    yield from flatten(item)
                else:
                    yield item

        def is_required(tp_default_item):
            """Return True if input is a required arg or 1+ arbitrarily nested arrays thereof."""
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
        tpd = self.hoomd_class(
            nlist=hoomd.md.nlist.Cell(2),
            **self.initial_args
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
