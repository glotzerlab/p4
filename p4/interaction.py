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
        nlist = hoomd.md.nlist.Cell(2)
        simulation = hoomd.util.make_example_simulation(
            particle_types=self.yes_types
        )
        if "params" in self.yes_pair_typed_attributes.keys():
                yes_r_cut = self.yes_pair_typed_attributes["r_cut"]
        else:
            yes_r_cut = max([
                v["r_cut"] for v in self.yes_pair_typed_attributes.values()
            ])
        box_length = 10 * max([
            self.no_pair_typed_attributes.get("r_cut", 0.0),
            yes_r_cut
        ])
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
                if y_t in self.yes_single_typed_attributes.keys():
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

    # @classmethod
    # def from_hoomd_md_pair(cls, pair: hoomd.md.pair.Pair) -> Interaction:
    #     """Create an Interaction from a HOOMD pair instance.

    #     Parameters
    #     ----------
    #     pair : hoomd.md.pair.Pair
    #         The HOOMD pair instance.
    #     """
    #     hoomd_class = pair.__class__
    #     initial_args = None
    #     no_single_typed_attributes = None
    #     no_pair_typed_attributes = None
    #     yes_types = None
    #     yes_single_typed_attributes = None
    #     yes_pair_typed_attributes = None

    #     return cls(
    #         hoomd_class=hoomd_class
    #         initial_args=initial_args
    #         no_single_typed_attributes=no_single_typed_attributes
    #         no_pair_typed_attributes=no_pair_typed_attributes
    #         yes_types=yes_types
    #         yes_single_typed_attributes=yes_single_typed_attributes
    #         yes_pair_typed_attributes=yes_pair_typed_attributes
    #     )