# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from copy import deepcopy
import inspect
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
        self.all_types = [str(t) for t in all_types]  # Review: enforce uniqueness?
        self.yes_params = yes_params

        self.validate()

    def validate(self):
        """Ensure this Interaction behaves properly."""
        # Ensure all "yes" types are provided in all_types
        for type_name in self.yes_params:
            if isinstance(type_name, Iterable) and not isinstance(type_name, (str, bytes)):
                for t in type_name:
                    if t not in self.all_types:
                        raise ValueError(
                            f"yes type pair '{type_name}' contains '{t}', "
                            + "which is not in all_types"
                        )
            else:
                if type_name not in self.all_types:
                    raise ValueError(
                        f"yes type '{type_name}' is not in all_types"
                    )

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
        except (AttributeError, KeyError) as e:
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
        
        simulation = self._get_test_simulation(
            particle_types=self.all_types,
            max_r_cut=max_r_cut,
            nlist=hoomd.md.nlist.Cell(2),
            interaction=self
        )
        box_length = 10 * max(max_r_cut, 1.0)
        simulation.state.set_box([box_length, box_length, box_length, 0, 0, 0])
        simulation = p4.util.add_integrator(simulation)
        simulation = p4.util.add_interaction(simulation, nlist, self)

        try:
           simulation.run(0)
        except RuntimeError as e:
            msg = (
                "Validation failed: the parameterized HOOMD instance cannot be "
                + "used in a running simulation. See traceback for details."
            )
            raise ValueError(msg) from e
        
        # Simplify yes_params where possible, moving full repeats into no_params
        # [TODO: reduce code duplication here]
        for type_name, param_dict in deepcopy(self.yes_params).items():
            for param_name, param_value in param_dict.items():
                
                types_with_same_param = []
                other_params = {
                    k: v
                    for k, v in self.yes_params.items()
                    if k != type_name
                }
                for t, d in other_params.items():
                    if param_name in d:
                        if d[param_name] == param_value:
                            types_with_same_param.append(t)

                if isinstance(type_name, str):
                    if (
                        len(types_with_same_param) == len(self.all_types) - 1 and
                        all(t in self.all_types for t in types_with_same_param)
                    ):
                        for k, v in param_dict.items():
                            self.no_params[k] = v
                        for t in types_with_same_param:
                            del self.yes_params[t][param_name]
                
                elif isinstance(type_name, Iterable) and len(type_name) == 2:
                    all_type_pairs = list(itertools.combinations_with_replacement(
                        self.all_types, 2
                    ))
                    if (
                        len(types_with_same_param) == len(all_type_pairs) - 1 and
                        all(p in all_type_pairs for p in types_with_same_param)
                    ):
                        for k, v in param_dict.items():
                            self.no_params[k] = v
                        for t in types_with_same_param:
                            try:    # Review: make this less hacky
                                del self.yes_params[t][param_name]
                            except KeyError:
                                continue
        
        self.yes_params = {k: v for k, v in self.yes_params.items() if v != {}}
        
        # Convert all tuples to lists in values (NOT in keys)
        # [TODO: this implementation is horribly hacky. Improve later.]
        def tuples_to_lists(item):
            """Convert item to list if it is a tuple, same for its elements."""
            if isinstance(item, tuple):
                item = list(item)
            if isinstance(item, list):
                item = [list(i) if isinstance(i, tuple) else i for i in item]
            return item
        params = [self.no_params, self.yes_params]
        for p in params:
            for k, v in p.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        if isinstance(sv, dict):
                            for ssk, ssv in sv.items():
                                p[k][sk][ssk] = tuples_to_lists(ssv)
                        else:
                            p[k][sk] = tuples_to_lists(sv)
                else:
                    p[k] = tuples_to_lists(v)               

    @staticmethod
    def _get_test_simulation(particle_types, max_r_cut, nlist, interaction):
        """Return a small example simulation with an interaction that is ready to run.
        TODO
        """
        simulation = hoomd.util.make_example_simulation(
            particle_types=particle_types
        )
        s = 10 * max_r_cut
        simulation.state.set_box([s, s, s, 0, 0, 0])
        simulation = p4.util.add_integrator(simulation)
        simulation = p4.util.add_interaction(simulation, nlist, interaction)
        
        return simulation
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
            if isinstance(key, Iterable) and not isinstance(key, (str, bytes))
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

    @classmethod
    def from_hoomd_pair(
        cls,
        pair: hoomd.md.pair.Pair
    ) -> list[Interaction] | Interaction:
        """Parse a hoomd.md.pair.Pair object into an Interaction.
        
        Parameters
        ----------
        pair : hoomd.md.pair.Pair
            The hoomd pairwise force instance.
        """
        def get_particle_types(typeparam_dict, yes_or_all):
            """Return a list of particle type names in a typeparameter dictionary.
            
            'yes' types are defined as having non-zero r_cut values and are
            returned in pairs. 'all' types are returned as a flat list of names.
            """
            if yes_or_all == "yes":
                particle_types = []
                for type_pair, value in typeparam_dict["r_cut"].items():
                    if value != 0:
                        particle_types.append(type_pair)
                return particle_types

            particle_types = []
            for param_name, typeparam in typeparam_dict.items():
                for type_name in typeparam:
                    if isinstance(type_name, tuple):
                        for i in type_name:
                            if i not in particle_types:
                                particle_types.append(i)
                    elif isinstance(type_name, str):
                        if type_name not in particle_types:
                            particle_types.append(i)
                    else:
                        raise ValueError(
                            f"Malformed typeparam dict: the value for key "
                            + f"'{param_name}' must be a dict with tuples or "
                            + "strings for keys, but it has a "
                            + f"key '{type_name}'."
                        )
            return particle_types

        # Get typeparam dict as a native Python object
        tpd = {k: v.to_base() for k, v in pair._typeparam_dict.items()}

        # Constructor
        hoomd_class = type(pair)
        
        # Initial args
        initial_args = pair._param_dict.to_base()
        
        # Delete unnecessary initial args
        del initial_args["nlist"]
        if "mode" in initial_args and initial_args["mode"] == "none":
            del initial_args["mode"]
        
        # Params
        all_types = get_particle_types(tpd, "all")  # includes ONLY singles
        yes_types = get_particle_types(tpd, "yes")  # includes singles AND pairs

        no_params = {}
        yes_params = {}

        for param_name, typeparam in tpd.items():
            for type_name, param_value in typeparam.items():
                # Yes params
                if type_name in yes_types:
                    if type_name not in yes_params:
                        yes_params[type_name] = {}
                    if hasattr(param_value, "to_base"):
                        yes_params[type_name][param_name] = param_value.to_base()
                    else:
                        yes_params[type_name][param_name] = param_value

                # No params
                else:
                    if param_name not in no_params:
                        if hasattr(param_value, "to_base"):
                            no_params[param_name] = param_value.to_base()
                        else:
                            no_params[param_name] = param_value

        kwargs=dict(
            hoomd_class=hoomd_class,
            initial_args=initial_args,
            no_params=no_params,
            all_types=all_types,
            yes_params=yes_params,
        )

        return cls(**kwargs)

    @classmethod
    def from_hoomd_integrator(
        cls,
        integrator: hoomd.md.Integrator
    ) -> list[Interaction] | Interaction:
        """Parse a hoomd.md.Integrator object into 1 or more Interactions.
        
        This is a convenience method that is equivalent to

        ```python
        [p4.Interaction.from_hoomd_pair(p) for p in integrator.forces]
        ```

        Parameters
        ----------
        integrator : hoomd.md.Integrator
            The integrator that contains the pairwise forces.
        """
        if not integrator.forces:
            raise ValueError("`integrator` must have forces")
        return [cls.from_hoomd_pair(p) for p in integrator.forces]

    @classmethod
    def from_hoomd_simulation(
        cls,
        simulation: hoomd.Simulation,
    ) -> list[Interaction] | Interaction:
        """Parse a hoomd.Simulation object into 1 or more Interactions.

        This is a convenience method that is equivalent to
        
        ```python
        p4.Interaction.from_hoomd_integrator(simulation.operations.integrator)
        ```
        Parameters
        ----------
        integrator : hoomd.md.Integrator
            The simulation whose integrator contains the pairwise forces.
        """
        if simulation.operations.integrator is None:
            raise ValueError("`simulation` must have an integrator")
        if not simulation.operations.integrator.forces:
            raise ValueError("integrator must have forces")
        return cls.from_hoomd_integrator(simulation.operations.integrator)

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

    def __eq__(self, other):
        """Interactions are equal if they have equivalent properties."""
        # [TODO: much of this logic ought to be in a separate parsing module]
        same_type = type(self) is type(other)

        if not same_type:
            return False
        
        same_initial_args = self.initial_args == other.initial_args
        same_no_params = self.no_params == other.no_params
        same_all_types = self.all_types == other.all_types
        equivalent_all_types = set(self.all_types) == set(other.all_types)
        same_yes_params = self.yes_params == other.yes_params

        def without_keys(d, keys):
            """Return a copy of a dict without items specified by keys."""
            return {k: d[k] for k in d if k not in keys}

        # If initial_args are different, they should be considered equivalent if
        # the only differences are between
        #   - `default_r_cut` or `default_r_on` parameters that are
        #     superceded by equivalent values in no_params
        #   - the presence of optional parameters that are set to their default
        #     values
        equivalent_initial_args = same_initial_args
        if not same_initial_args:
            # Check for equivalence based on r_cut and/or r_on values
            default_args = ["default_r_cut", "default_r_on"]
            same_without_rs = (
                without_keys(self.initial_args, default_args)
                == without_keys(other.initial_args, default_args)
            )

            different_rs = [
                r
                for r in ["r_cut", "r_on"]
                if (
                    self.initial_args.get(f"default_{r}")
                    != other.initial_args.get(f"default_{r}")
                )
            ]
            rs_equivalent = [True] if not different_rs else []
            for r in different_rs:
                # For r_cut, there MUST be a value in either initial_args or
                # no_params, otherwise the hoomd pair cannot be parameterized.
                # For r_on, there does not need to be a value, and if this is
                # the case then hoomd defaults the parameter to zero.
                if r in self.no_params:
                    self_dominant_r = self.no_params[r]
                elif f"default_{r}" in self.initial_args:
                    self_dominant_r = self.initial_args[f"default_{r}"]
                else:
                    if r == "r_on":
                        self_dominant_r = 0
                    else:
                        msg = (
                            "During equivalence comparison, encountered a "
                            + "problem: can find neither 'default_r_cut' in "
                            + "`self.initial_args` nor 'r_cut' in "
                            + "`self.no_params`. This is undefined behavior."
                        )
                        raise ValueError(msg)
                
                if r in other.no_params:
                    other_dominant_r = other.no_params[r]
                elif f"default_{r}" in other.initial_args:
                    other_dominant_r = other.initial_args[f"default_{r}"]
                else:
                    if r == "r_on":
                        other_dominant_r = 0
                    else:
                        msg = (
                            "During equivalence comparison, encountered a "
                            + "problem: can find neither 'default_r_cut' in "
                            + "`other.initial_args` nor 'r_cut' in "
                            + "`other.no_params`. This is undefined behavior."
                        )
                        raise ValueError(msg)

                rs_equivalent.append(self_dominant_r == other_dominant_r)

            # Check for equivalence based on optional parameters
            optional_args_equivalent = []
            if not same_without_rs:
                signature = inspect.signature(self.hoomd_class.__init__)
                for name, param in signature.parameters.items():
                    # Skip args that aren't supposed to be in initial_args
                    # Skip default r args (already analyzed elsewhere)
                    if name not in ["self", "nlist", "default_r_cut", "default_r_on"]:
                        # Loop over all optional args
                        if param.default is not inspect._empty:
                            default_args.append(name)
                            # Check self
                            if name in self.initial_args:
                                if name not in other.initial_args:
                                    optional_args_equivalent.append(
                                        param.default == self.initial_args[name]
                                    )
                                else:
                                    optional_args_equivalent.append(
                                        self.initial_args[name]
                                        == other.initial_args[name]
                                    )

                            # Check other
                            if name in other.initial_args:
                                if name not in self.initial_args:
                                    optional_args_equivalent.append(
                                        param.default == other.initial_args[name]
                                    )

            equivalent_without_rs = (
                without_keys(self.initial_args, default_args)
                == without_keys(other.initial_args, default_args)
            )
            equivalent_initial_args = (
                (same_without_rs and all(rs_equivalent))
                or (
                    equivalent_without_rs
                    and all(optional_args_equivalent)
                    and all(rs_equivalent)
                )
            )
        
        # If no_params are different, they should be considered equivalent if
        # the only differences are between
        #   - the absence of `r_cut` or `r_on` parameters that are covered by
        #     equivalent default values in initial_args
        #   - the presence of optional parameters that are set to their default
        #     values
        #   - the presence of the same parameter value for every single or pair
        equivalent_no_params = same_no_params
        if not same_no_params:
            # Check for equivalence based on r_cut and/or r_on values
            same_without_rs = (
                without_keys(self.no_params, ["r_cut", "r_on"])
                == without_keys(other.no_params, ["r_cut", "r_on"])
            )

            different_rs = [
                r
                for r in ["r_cut", "r_on"]
                if self.no_params.get(r) != other.no_params.get(r)
            ]
            rs_equivalent = [True] if not different_rs else []
            for r in different_rs:
                # For r_cut, there MUST be a value in either initial_args or
                # no_params, otherwise the hoomd pair cannot be parameterized.
                # For r_on, there does not need to be a value, and if this is
                # the case then hoomd defaults the parameter to zero.
                if r not in self.no_params:
                    if f"default_{r}" in self.initial_args:
                        self_dominant_r = self.initial_args[f"default_{r}"]
                    else:
                        if r == "r_on":
                            self_dominant_r = 0
                        else:
                            msg = (
                                "During equivalence comparison, encountered a "
                                + "problem: can find neither 'default_r_cut' "
                                + "in `self.initial_args` nor 'r_cut' in "
                                + "`self.no_params`. This is undefined "
                                + "behavior."
                            )
                            raise ValueError(msg)
                else:
                    self_dominant_r = self.no_params[r]
                
                if r not in other.no_params:
                    if f"default_{r}" in other.initial_args:
                        other_dominant_r = other.initial_args[f"default_{r}"]
                    else:
                        if r == "r_on":
                            other_dominant_r = 0
                        else:
                            msg = (
                                "During equivalence comparison, encountered a "
                                + "problem: can find neither 'default_r_cut' "
                                + "in `other.initial_args` nor 'r_cut' in "
                                + "`other.no_params`. This is undefined "
                                + "behavior."
                            )
                            raise ValueError(msg)
                else:
                    other_dominant_r = other.no_params[r]

                rs_equivalent.append(self_dominant_r == other_dominant_r)
            
            # Check for equivalence based on optional parameters
            optional_params_equivalent = []
            if not same_without_rs:
                instance = self.to_hoomd_instance(hoomd.md.nlist.Cell(0))
                tpd = instance._typeparam_dict
                for name, typeparam in tpd.items():
                    # If the typeparam is a dictionary mapping names to some
                    # subtypeparams, then each of those subtypeparams must also
                    # be checked. This process is not recursive - i.e., it does
                    # not perform this dictionary check on each of those
                    # subtypeparams.
                    if isinstance(typeparam.default, dict):
                        for k, v in typeparam.default.items():
                            if isinstance(v, dict):
                                for sk, sv in v.items():
                                    if isinstance(sv, dict):
                                        for ssk, ssv in sv.items():
                                            # Check self
                                            if (
                                                name in self.no_params
                                                and isinstance(self.no_params[name], dict)
                                                and k in self.no_params[name]
                                                and isinstance(self.no_params[name][k], dict)
                                                and sk in self.no_params[name][k]
                                                and isinstance(self.no_params[name][k][sk], dict)
                                                and ssk in self.no_params[name][k][sk]
                                            ):
                                                optional_params_equivalent.append(
                                                    ssv == self.no_params[name][k][sk][ssk]
                                                )
                                            # Check other
                                            if (
                                                name in other.no_params
                                                and isinstance(other.no_params[name], dict)
                                                and k in other.no_params[name]
                                                and isinstance(other.no_params[name][k], dict)
                                                and sk in other.no_params[name][k]
                                                and isinstance(other.no_params[name][k][sk], dict)
                                                and ssk in other.no_params[name][k][sk]
                                            ):
                                                optional_params_equivalent.append(
                                                    sv == other.no_params[name][k][sk][ssk]
                                                )
                                    else:
                                        # Check self
                                        if (
                                            name in self.no_params
                                            and isinstance(self.no_params[name], dict)
                                            and k in self.no_params[name]
                                            and isinstance(self.no_params[name][k], dict)
                                            and sk in self.no_params[name][k]
                                        ):
                                            optional_params_equivalent.append(
                                                sv == self.no_params[name][k][sk]
                                            )
                                        # Check other
                                        if (
                                            name in other.no_params
                                            and isinstance(other.no_params[name], dict)
                                            and k in other.no_params[name]
                                            and isinstance(other.no_params[name][k], dict)
                                            and sk in other.no_params[name][k]
                                        ):
                                            optional_params_equivalent.append(
                                                sv == other.no_params[name][k][sk]
                                            )
                            else:
                                # Check self
                                if (
                                    name in self.no_params
                                    and isinstance(self.no_params[name], dict)
                                    and k in self.no_params[name]
                                ):
                                    optional_params_equivalent.append(
                                        v == self.no_params[name][k]
                                    )
                                # Check other
                                if (
                                    name in other.no_params
                                    and isinstance(other.no_params[name], dict)
                                    and k in other.no_params[name]
                                ):
                                    optional_params_equivalent.append(
                                        v == other.no_params[name][k]
                                    )
                                

                    # If the typeparam is not a dictionary, everything's simple
                    else:
                        # Loop over all optional params
                        if typeparam.default is not hoomd.data.typeconverter.RequiredArg:
                            # Check self
                            if name in self.no_params:
                                optional_params_equivalent.append(
                                    typeparam.default == self.no_params[name]
                                )
                            # Check other
                            if name in other.no_params:
                                optional_params_equivalent.append(
                                    typeparam.default == other.no_params[name]
                                )

            equivalent_no_params = (
                (same_without_rs and all(rs_equivalent))
                or (all(optional_params_equivalent) and all(rs_equivalent))
            )

        # If yes and/or no_params are different, they should be considered
        # equivalent if the only differences are between
        #   - the absence of `r_cut` or `r_on` parameters that are covered by
        #     equivalent default values in initial_args
        #   - the presence of optional parameters that are set to their default
        #     values
        #   - the presence of the same parameter value for every single or pair
        #   - TODO: add support for type pairs that are tuples in a different
        #     order (Review: consider changing tuples to sets)
        equivalent_yes_params = same_yes_params
        if not same_yes_params:
            # Check for equivalence based on r_cut and/or r_on values
            same_without_rs = (
                without_keys(self.yes_params, ["r_cut", "r_on"])
                == without_keys(other.yes_params, ["r_cut", "r_on"])
            )

            different_rs = [
                r
                for r in ["r_cut", "r_on"]
                if self.yes_params.get(r) != other.no_params.get(r)
            ]
            rs_equivalent = [True] if not different_rs else []
            for r in different_rs:
                # For r_cut, there MUST be a value in either initial_args or
                # no_params, otherwise the hoomd pair cannot be parameterized.
                # For r_on, there does not need to be a value, and if this is
                # the case then hoomd defaults the parameter to zero.
                if r not in self.yes_params:
                    if r in self.no_params:
                        self_dominant_r = self.no_params[r]
                    elif f"default_{r}" in self.initial_args:
                        self_dominant_r = self.initial_args[f"default_{r}"]
                    else:
                        if r == "r_on":
                            self_dominant_r = 0
                        else:
                            msg = (
                                "During equivalence comparison, encountered a "
                                + "problem: can find neither 'default_r_cut' "
                                + "in `self.initial_args` nor 'r_cut' in "
                                + "`self.no_params`. This is undefined "
                                + "behavior."
                            )
                            raise ValueError(msg)
                else:
                    self_dominant_r = self.yes_params[r]
                
                if r not in other.yes_params:
                    if r in other.no_params:
                        other_dominant_r = other.no_params[r]
                    elif f"default_{r}" in other.initial_args:
                        other_dominant_r = other.initial_args[f"default_{r}"]
                    else:
                        if r == "r_on":
                            other_dominant_r = 0
                        else:
                            msg = (
                                "During equivalence comparison, encountered a "
                                + "problem: can find neither 'default_r_cut' "
                                + "in `other.initial_args` nor 'r_cut' in "
                                + "`other.no_params`. This is undefined "
                                + "behavior."
                            )
                            raise ValueError(msg)
                else:
                    other_dominant_r = other.yes_params[r]

                rs_equivalent.append(self_dominant_r == other_dominant_r)
            
            # Check for equivalence based on optional parameters
            optional_params_equivalent = []
            if not same_without_rs:
                instance = self.to_hoomd_instance(hoomd.md.nlist.Cell(0))
                tpd = instance._typeparam_dict
                for name, typeparam in tpd.items():
                    # If the typeparam is a dictionary mapping names to some
                    # subtypeparams, then each of those subtypeparams must also
                    # be checked. This process is not recursive - i.e., it does
                    # not perform this dictionary check on each of those
                    # subtypeparams.
                    if isinstance(typeparam.default, dict):
                        for k, v in typeparam.default.items():
                            if isinstance(v, dict):
                                for sk, sv in v.items():
                                    if isinstance(sv, dict):
                                        for ssk, ssv in sv.items():
                                            # Check self
                                            if (
                                                name in self.yes_params
                                                and isinstance(self.yes_params[name], dict)
                                                and k in self.yes_params[name]
                                                and isinstance(self.yes_params[name][k], dict)
                                                and sk in self.yes_params[name][k]
                                                and isinstance(self.yes_params[name][k][sk], dict)
                                                and ssk in self.yes_params[name][k][sk]
                                            ):
                                                optional_params_equivalent.append(
                                                    ssv == self.yes_params[name][k][sk][ssk]
                                                )
                                            # Check other
                                            if (
                                                name in other.yes_params
                                                and isinstance(other.yes_params[name], dict)
                                                and k in other.yes_params[name]
                                                and isinstance(other.yes_params[name][k], dict)
                                                and sk in other.yes_params[name][k]
                                                and isinstance(other.yes_params[name][k][sk], dict)
                                                and ssk in other.yes_params[name][k][sk]
                                            ):
                                                optional_params_equivalent.append(
                                                    sv == other.yes_params[name][k][sk][ssk]
                                                )
                                    else:
                                        # Check self
                                        if (
                                            name in self.yes_params
                                            and isinstance(self.yes_params[name], dict)
                                            and k in self.yes_params[name]
                                            and isinstance(self.yes_params[name][k], dict)
                                            and sk in self.yes_params[name][k]
                                        ):
                                            optional_params_equivalent.append(
                                                sv == self.yes_params[name][k][sk]
                                            )
                                        # Check other
                                        if (
                                            name in other.yes_params
                                            and isinstance(other.yes_params[name], dict)
                                            and k in other.yes_params[name]
                                            and isinstance(other.yes_params[name][k], dict)
                                            and sk in other.yes_params[name][k]
                                        ):
                                            optional_params_equivalent.append(
                                                sv == other.yes_params[name][k][sk]
                                            )
                            else:
                                # Check self
                                if (
                                    name in self.yes_params
                                    and isinstance(self.yes_params[name], dict)
                                    and k in self.yes_params[name]
                                ):
                                    optional_params_equivalent.append(
                                        v == self.yes_params[name][k]
                                    )
                                # Check other
                                if (
                                    name in other.yes_params
                                    and isinstance(other.yes_params[name], dict)
                                    and k in other.yes_params[name]
                                ):
                                    optional_params_equivalent.append(
                                        v == other.yes_params[name][k]
                                    )
                                

                    # If the typeparam is not a dictionary, everything's simple
                    else:
                        # Loop over all optional params
                        if typeparam.default is not hoomd.data.typeconverter.RequiredArg:
                            # Check self
                            if name in self.yes_params:
                                optional_params_equivalent.append(
                                    typeparam.default == self.yes_params[name]
                                )
                            # Check other
                            if name in other.yes_params:
                                optional_params_equivalent.append(
                                    typeparam.default == other.yes_params[name]
                                )

            equivalent_yes_params = (
                (same_without_rs and all(rs_equivalent))
                or (all(optional_params_equivalent) and all(rs_equivalent))
            )

        return (
            same_type
            and (same_initial_args or equivalent_initial_args)
            and (same_no_params or equivalent_no_params)
            and (same_all_types or equivalent_all_types)
            and (same_yes_params or equivalent_yes_params)
        )

    def __repr__(self):
        return (
            "Interaction ("
            + f"\n\thoomd_class={self.hoomd_class},"
            + f"\n\tinitial_args={self.initial_args},"
            + f"\n\tno_params={self.no_params},"
            + f"\n\tall_types={self.all_types},"
            + f"\n\tyes_params={self.yes_params}"
            + "\n)"
        )