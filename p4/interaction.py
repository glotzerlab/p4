# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

import itertools
import hoomd

import p4.util


class Interaction:
    """The data for instantiating and parameterizing a hoomd.md.pair potential.

    `Interaction` is self-validating, i.e., every instance is guaranteed to
    successfully create and parameterize its provided HOOMD pair potential.

    Parameters
    ----------
    hoomd_class : hoomd.md.pair.Pair
        The constructor for the HOOMD class. Must be in `hoomd.md.pair`.
    initial_args : dict[str, float | str]
        All parameters (that aren't `nlist`) that are needed for instantiating
        the class from its constructor.
    no_single_typed_attributes : dict
        The names and non-interacting values of attributes that must be set for
        individual particle types. This is usually an empty dict for isotropic
        interactions.
    no_pair_typed_attributes : dict[str, float]
        The names and non-interacting values of attributes that must be set for
        pairs of particle types. This is usually an empty dict for isotropic
        interactions.
    yes_types : list[str]
        The particle types that can interact with each other under this
        potential. All interacting type pairs are assumed to use the same
        interaction attributes.
    yes_single_typed_attributes : dict[str, float] or dict[str, dict[str, float]]
        Either the names and values of the attributes for interacting particle
        types that must be set for individual types, or the names of the
        interacting particle types and dictionaries of their corresponding
        attributes. If the former, the same attributes are used for every
        interacting type. This is usually an empty dict for isotropic
        interactions.
    yes_pair_typed_attributes : dict[str, float] or dict[tuple, dict[str, float]]
        Either the names and values of the attributes for interacting particle
        types that must be set for pairs of types, or the pairs of the
        interacting types and dictionaries of their corresponding attributes. If
        the former, the same attributes are used for every pair of interacting
        types.
    """
    def __init__(
        self,
        hoomd_class: hoomd.md.pair.Pair,
        initial_args: dict[str, float | str],
        no_single_typed_attributes: dict,
        no_pair_typed_attributes: dict[str, float],
        yes_types: list[str],
        yes_single_typed_attributes: dict[str, float],
        yes_pair_typed_attributes: dict[str, float],
    ):
        self.hoomd_class = hoomd_class
        self.initial_args = initial_args
        self.no_single_typed_attributes = no_single_typed_attributes
        self.no_pair_typed_attributes = no_pair_typed_attributes
        self.yes_types = [str(t) for t in yes_types]
        self.yes_single_typed_attributes = yes_single_typed_attributes
        self.yes_pair_typed_attributes = yes_pair_typed_attributes

        self.validate()

    def validate(self):
        """Ensure the HOOMD class can be created, parameterized, and used."""
        # Creation
        nlist = hoomd.md.nlist.Cell(2)
        try:
            _ = self.to_hoomd_instance(nlist)
        except ValueError as e:
            msg = "Validation failed: the HOOMD class cannot be instantiated."
            raise ValueError(msg) from e
        
        # Parameterization
        nlist = hoomd.md.nlist.Cell(2)
        try:
            _ = self.to_parameterized_hoomd_instance(nlist, self.yes_types)
        except AttributeError as e:
            msg = "Validation failed: the HOOMD instance cannot be parameterized."
            raise ValueError(msg) from e
        
        # Usage
        if "params" in self.yes_pair_typed_attributes:
                yes_r_cut = self.yes_pair_typed_attributes["r_cut"]
        else:
            yes_r_cut = max([
                v["r_cut"] for v in self.yes_pair_typed_attributes.values()
            ])
        max_r_cut = max([
            self.no_pair_typed_attributes.get("r_cut", 0.0),
            yes_r_cut
        ])
        
        simulation = self._get_test_simulation(
            particle_types=self.yes_types,
            max_r_cut=max_r_cut,
            nlist=hoomd.md.nlist.Cell(2),
            interaction_or_pair=self
        )

        try:
           simulation.run(0)
        except RuntimeError as e:
            msg = (
                "Validation failed: the parameterized HOOMD instance cannot be "
                + "used in a running simulation. See traceback for details."
            )
            raise ValueError(msg) from e

    @staticmethod
    def _get_test_simulation(particle_types, max_r_cut, nlist, interaction_or_pair):
        """Return a small example simulation with an interaction that is ready to run.
        TODO
        """
        simulation = hoomd.util.make_example_simulation(
            particle_types=particle_types
        )
        s = 10 * max_r_cut
        simulation.state.set_box([s, s, s, 0, 0, 0])
        simulation = p4.util.add_integrator(simulation)
        if isinstance(interaction_or_pair, Interaction):
            breakpoint()
            simulation = p4.util.add_interaction(simulation, nlist, interaction_or_pair)
        elif isinstance(interaction_or_pair, hoomd.md.pair.Pair):
            simulation.operations.integrator.forces.append(interaction_or_pair)
        
        return simulation

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
        all_types: list[str]
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
            If the typed attributes are wrong.
        """
        def wrong_type_msg(att, no_or_yes, single_or_double, hoomd_class):
            return (
                f"'{att}' was provided as a {no_or_yes} {single_or_double}"
                f"-typed-attribute, but no such attribute was found in hoomd "
                f"class '{hoomd_class}'."
            )

        instance = self.to_hoomd_instance(nlist)
        
        # Calculate the pairwise combinations of all types and interacting types
        breakpoint()
        all_type_pairs = list(itertools.combinations(all_types, 2))
        all_type_pairs.extend([(t, t) for t in all_types])
        yes_type_pairs = []
        for p in all_type_pairs:
            if (
                p[0] in self.yes_types
                and p[1] in self.yes_types
                and p[0] != p[1]
            ):
                yes_type_pairs.append(p)

        # Set no single attributes
        for a_t in all_types:
            for k, v in self.no_single_typed_attributes.items():
                try:
                    getattr(instance, k)[a_t] = v
                except AttributeError:
                    raise AttributeError(
                        wrong_type_msg(
                            k, "no", "single", self.hoomd_class
                        )
                    )

        # Set no pair attributes
        for a_p in all_type_pairs:
            for k, v in self.no_pair_typed_attributes.items():
                try:
                    getattr(instance, k)[a_p] = v
                except AttributeError:
                    raise AttributeError(
                        wrong_type_msg(
                            k, "no", "pair", self.hoomd_class
                        )
                    )

        # Modify single attributes for interacting types
        if self.yes_single_typed_attributes != {}:
            for y_t in self.yes_types:
                # Specific attributes for the current interacting type
                if y_t in self.yes_single_typed_attributes:
                    atts_for_this_type = self.yes_single_typed_attributes[y_t]

                # Attributes for all interacting types
                else:
                    atts_for_this_type = self.yes_single_typed_attributes

                for k, v in atts_for_this_type.items():
                    try:
                        getattr(instance, k)[y_t] = v
                    except AttributeError:
                        raise AttributeError(
                            wrong_type_msg(
                                k, "yes", "single", self.hoomd_class
                            )
                        )

        # Modify pair attributes for interacting types
        for y_p in yes_type_pairs:
            # Specific attributes for the current interacting pair
            if type(list(self.yes_pair_typed_attributes.keys())[0]) is tuple:
                atts_for_this_pair = self.yes_pair_typed_attributes[y_p]

            # Attributes for all interacting pairs
            else:
                atts_for_this_pair = self.yes_pair_typed_attributes

            for k, v in atts_for_this_pair.items():
                try:
                    getattr(instance, k)[y_p] = v
                except AttributeError:
                    raise AttributeError(
                        wrong_type_msg(
                            k, "yes", "pair", self.hoomd_class
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
            
            'yes' types are defined as having non-zero r_cut values.
            """
            if yes_or_all == "yes":
                particle_types = []
                for k, v in typeparam_dict["r_cut"].items():
                    if v != 0:
                        particle_types.extend(i for i in k if i not in particle_types)
                return particle_types

            particle_types = []
            for name, typeparam in typeparam_dict.items():
                for t_k, t_v in typeparam.items():
                    if isinstance(t_k, tuple):
                        for i in t_k:
                            if i not in particle_types:
                                particle_types.append(i)
                    elif isinstance(t_k, str):
                        if t_k not in particle_types:
                            particle_types.append(i)
                    else:
                        raise ValueError(
                            f"Malformed typeparam dict: the value for key '{name}' must "
                            + "be a dict with tuples or strings for keys, but it has a "
                            + f"key '{t_k}'."
                        )
            return particle_types

        # Ensure pair is fully parameterized
        tpd = {k: v.to_base() for k, v in pair._typeparam_dict.items()}

        max_r_cut = max([v for v in tpd["r_cut"].values()])

        simulation = cls._get_test_simulation(
            particle_types=get_particle_types(tpd, "all"),
            max_r_cut=max_r_cut,
            nlist=hoomd.md.nlist.Cell(2),
            interaction_or_pair=pair
        )
        try:
            simulation.run(0)
        except RuntimeError as e:
            msg = "pair is not properly parameterized. See traceback for details."
            raise ValueError(msg) from e

        # Constructor
        hoomd_class = type(pair)
        
        # Initial args
        initial_args = pair._param_dict.to_base()   # TODO: default_r_cut, default_r_on??
        
        # Delete unnecessary initial args
        del initial_args["nlist"]
        if "mode" in initial_args and initial_args["mode"] == "none":
            del initial_args["mode"]
        
        # Attributes
        yes_types = get_particle_types(tpd, "yes")

        no_single_typed_attributes = {}
        no_pair_typed_attributes = {}
        yes_single_typed_attributes = {}
        yes_pair_typed_attributes = {}

        for name, typeparam in tpd.items():
            for t_k, t_v in typeparam.items():
                if isinstance(t_k, tuple):
                    if all(t in yes_types for t in t_k):
                        if t_k not in yes_pair_typed_attributes:
                            yes_pair_typed_attributes[t_k] = {}
                        if name not in yes_pair_typed_attributes[t_k]:
                            yes_pair_typed_attributes[t_k][name] = {}
                        yes_pair_typed_attributes[t_k][name] = t_v  # NOTE the order of keys # TODO: cannot currently have yes (B, B), yes (C, C), but no (B, C)
                    else:
                        if name not in no_pair_typed_attributes:
                            no_pair_typed_attributes[name] = {}
                        no_pair_typed_attributes[name] = t_v    # NOTE: this can overwrite itself
                elif isinstance(t_k, str):
                    if t_k in yes_types:
                        if t_k not in yes_single_typed_attributes:
                            yes_single_typed_attributes[t_k] = {}
                        if name not in yes_single_typed_attributes[t_k]:
                            yes_single_typed_attributes[t_k][name] = {}
                        yes_single_typed_attributes[t_k][name] = t_v  # NOTE the order of keys
                    else:
                        if name not in no_single_typed_attributes:
                            no_single_typed_attributes[name] = {}
                        no_single_typed_attributes[name] = t_v  # NOTE: this can overwrite itself

        kwargs=dict(
            hoomd_class=hoomd_class,
            initial_args=initial_args,
            no_single_typed_attributes=no_single_typed_attributes,
            no_pair_typed_attributes=no_pair_typed_attributes,
            yes_types=yes_types,
            yes_single_typed_attributes=yes_single_typed_attributes,
            yes_pair_typed_attributes=yes_pair_typed_attributes,
        )

        return cls(**kwargs)

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

    def __repr__(self):
        return (
            "Interaction ("
            + f"\n\thoomd_class={self.hoomd_class},"
            + f"\n\tinitial_args={self.initial_args},"
            + f"\n\tno_single_typed_attributes={self.no_single_typed_attributes},"
            + f"\n\tno_pair_typed_attributes={self.no_pair_typed_attributes},"
            + f"\n\tyes_types={self.yes_types},"
            + f"\n\tyes_single_typed_attributes={self.yes_single_typed_attributes},"
            + f"\n\tyes_pair_typed_attributes={self.yes_pair_typed_attributes},"
            + "\n)"
        )
