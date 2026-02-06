# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from __future__ import annotations
import hoomd


class Body:
    """The names and spatial data for a body's primary and secondary types.

    Every body must have a primary particle type, but secondary types are
    optional.
    
    When secondary types **are not** provided, the body represents a
    simple particle with a single type and no further information is needed.
    
    When secondary types **are** provided, the body represents a rigid body
    with a central particle (`primary_type`) and one or more constituent
    particles (`secondary_types`). In this case, the body needs a way to
    determine the position(s) of each type of constituent particle, and the user
    must provide a function that does so.

    .. code-block::
        :caption: A cubic body with primary particle 'A' at the center and
            secondary particles 'B' at the vertices.

        body = p4.Body(
            primary_type="A",
            secondary_types=["B"],
            positions_by_type=dict(
                B=[
                    [-1, -1, -1],
                    [-1, -1,  1],
                    [-1,  1, -1],
                    [-1,  1,  1],
                    [ 1, -1, -1],
                    [ 1, -1,  1],
                    [ 1,  1, -1],
                    [ 1,  1,  1]
                ]
            )
        )


    Parameters
    ----------
    primary_type : str
        The name of the primary type.
    secondary_types : list[str], optional
        The names of the secondary types.
    positions_by_type : dict[str, list[list[float]]], optional
        A mapping of secondary particle type names to position(s). Required if
        `secondary_types` is provided, otherwise ignored.
    orientations_by_type : dict[str, list[list[float]]], optional
        A mapping of secondary particle type names to orientation(s) in
        quaternion form. Can only be provided if `secondary_types` and
        `positions_by_type` are also provided.
    """
    def __init__(
        self,
        primary_type: str,
        secondary_types: list[str] = [],
        positions_by_type: dict[str, list[list[float]]] = {},
        orientations_by_type: dict[str, list[list[float]]] = {}
    ):
        if secondary_types != [] and not positions_by_type:
            raise ValueError(
                "'positions_by_type' is required if "
                + "'secondary_types' is provided"
            )

        self.primary_type = str(primary_type)
        self.secondary_types = [str(t) for t in secondary_types]
        self.positions_by_type = positions_by_type
        self.orientations_by_type = orientations_by_type

        if self.positions_by_type:
            self._validate_secondary_positions()
        if self.orientations_by_type:
            self._validate_secondary_orientations()
        if (
            self.positions_by_type
            and self.orientations_by_type
        ):
            self._validate_secondary_orientations_and_positions_match()
    
    def is_rigid(self, interactions: list["Interaction"]) -> bool:
        """Whether the body must represent a rigid body for some interactions."""
        common_single_types = any(
            t in self.secondary_types
            for interaction in interactions
            for t in interaction.interacting_types("single")
        )
        common_pair_types = any(
            p[0] in self.secondary_types or p[1] in self.secondary_types
            for interaction in interactions
            for p in interaction.interacting_types("pair")
        )
        nonzero_default_r_cut = any(
            (
                interaction.default_params["r_cut"] > 0
                or interaction.initial_args.get("default_r_cut", 0) > 0
            )
            for interaction in interactions
        )
        return common_single_types or common_pair_types or nonzero_default_r_cut

    def _validate_secondary_positions(self):
        """Ensure that the provided callable works for all secondary types."""
        for t in self.secondary_types:
            if t not in self.positions_by_type.keys():
                raise ValueError(
                    "`positions_by_type` does not specify positions "
                    + f"for secondary type '{t}'."
                )

    def _validate_secondary_orientations(self):
        """Ensure that the provided callable works for all secondary types."""
        for t in self.secondary_types:
            if t not in self.orientations_by_type.keys():
                raise ValueError(
                    "`orientations_by_type` does not specify "
                    + f"orientations for secondary type '{t}'."
                )

    def _validate_secondary_orientations_and_positions_match(self):
        """Ensure secondary types' numbers of positions and orientations match."""
        for t in self.secondary_types:
            n_positions = len(self.positions_by_type[t])
            n_orientations = len(self.orientations_by_type[t])
            
            if n_positions != n_orientations:
                raise ValueError(
                    "The number of positions and orientations for secondary  "
                    + f"type {t} do not match."
                )

    @classmethod
    def from_hoomd_rigid(
        cls,
        rigid: hoomd.md.constrain.Rigid,
        primary_type: str | None = None
    ) -> list[Body] | Body:
        """Parse a `hoomd.md.constrain.Rigid <https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/constrain/rigid.html>`_ to create one or more :class:`~p4.Body`.
        
        Parameters
        ----------
        rigid : hoomd.md.constrain.Rigid
            The constraint that defines rigid bodies.
        primary_type : str, optional
            The name of the primary type of a single body. If provided, just
            that body is returned. If not provided, all possible bodies are
            returned in a list. If there is no body defined for the provided
            primary type, a single-particle body is returned.
        """
        # If primary type is supplied, it must be in the rigid's primary types
        if primary_type and primary_type not in rigid.body.keys():
            raise ValueError(
                f"`primary_type` ({primary_type}) not in rigid's primary types "
                f"({list(rigid.body.keys())})"
            )
        
        # Hoomd does not detect nested body definitions until sim.run(), so a
        # check is needed here
        for p_t in rigid.body.keys():
            for k, v in rigid.body.items():
                if v is not None:
                    if p_t in v["constituent_types"] and rigid.body[p_t] is not None:
                        raise ValueError("Nested bodies are not supported.")

        if primary_type:
            primary_types = [primary_type]
        else:
            primary_types = rigid.body.keys()

        def unique(strings):
            """Find unique values in a list of strings."""
            searched = []
            for s in strings:
                if s not in searched:
                    searched.append(s)
            return searched
        
        def data_by_type(types, data):
            """Return a mapping of unique types to their corresponding data."""
            d = {}
            for t in unique(types):
                d[t] = [x for i, x in enumerate(data) if types[i] == t]
            return d
        
        # Construct bodies
        bodies = []
        for p_t in primary_types:
            if rigid.body[p_t] is not None:
                bodies.append(cls(
                    primary_type=p_t,
                    secondary_types=unique(rigid.body[p_t]["constituent_types"]),
                    positions_by_type=data_by_type(
                        rigid.body[p_t]["constituent_types"],
                        [list(p) for p in rigid.body[p_t]["positions"]]
                    ),
                    orientations_by_type=data_by_type(
                        rigid.body[p_t]["constituent_types"],
                        [list(p) for p in rigid.body[p_t]["orientations"]]
                    )
                ))

        if len(bodies) == 0:
            return cls(primary_type=primary_type)
        if len(bodies) == 1:
            return bodies[0]
        else:
            return bodies

    @classmethod
    def from_hoomd_simulation(
        cls,
        simulation: hoomd.Simulation,
        primary_type: str | None = None
    ) -> list[Body] | Body:
        """Parse a `hoomd.Simulation <https://hoomd-blue.readthedocs.io/en/latest/hoomd/simulation.html>`_ to create one or more :class:`~p4.Body`.

        This is a convenience method that is equivalent to
        
        .. code-block::
            
            p4.Body.from_hoomd_rigid(sim.operations.integrator.rigid, primary_type)
        
        Parameters
        ----------
        simulation : hoomd.Simulation
            The simulation to parse.
        primary_type : str, optional
            The name of the primary type of a single body. If provided, just
            that body is returned. If not provided, all possible bodies are
            returned in a list.
        """
        if simulation.operations.integrator is None:
            raise ValueError("`simulation` must have an integrator")
        if simulation.operations.integrator.rigid is None:
            raise ValueError("integrator must have a rigid constraint")
        types_in_state = simulation.state.get_snapshot().particles.types
        if primary_type is not None and primary_type not in types_in_state:
            raise ValueError(
                f"`simulation` does not contain primary_type {primary_type}"
            )
        return cls.from_hoomd_rigid(
            simulation.operations.integrator.rigid, primary_type
        )

    def __eq__(self, other):
        """Bodies are equal if their attributes are the same or equivalent."""
        primary_same = self.primary_type == other.primary_type
        if not primary_same:
            return False
        
        secondary_same = self.secondary_types == other.secondary_types
        if not secondary_same:
            return False
        
        positions_same = self.positions_by_type == other.positions_by_type
        if not positions_same:
            return False

        orientations_same = (
            self.orientations_by_type == other.orientations_by_type
        )

        orientations_missing_from_self = (
            set(other.orientations_by_type) - set(self.orientations_by_type)
        )
        orientations_missing_from_other = (
            set(self.orientations_by_type) - set(other.orientations_by_type)
        )
        common_types = set(self.orientations_by_type).intersection(
            set(other.orientations_by_type)
        )
        orientations_equivalent = (
            all(
                self.orientations_by_type[t] == other.orientations_by_type[t]
                for t in common_types
            )
            and
            all(
                all(list(i) == [1,0,0,0] for i in other.orientations_by_type[t])
                for t in orientations_missing_from_self
            )
            and
            all(
                all(list(i) == [1,0,0,0] for i in self.orientations_by_type[t])
                for t in orientations_missing_from_other
            )
        )

        return (
            primary_same and secondary_same and positions_same
            and (orientations_same or orientations_equivalent)
        )
    
    def __repr__(self):
        return (
            "Body ("
            + f"\n\tprimary_type='{self.primary_type}',"
            + f"\n\tsecondary_types={self.secondary_types},"
            + f"\n\tpositions_by_type={self.positions_by_type},"
            + f"\n\torientations_by_type={self.orientations_by_type},"
            + "\n)"
        )
