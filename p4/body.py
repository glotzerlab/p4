# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from p4 import Interaction
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
    
    Parameters
    ----------
    primary_type : str
        The name of the primary type.
    secondary_types : list[str], optional
        The names of the secondary types.
    secondary_positions_by_type : Callable, optional
        A mapping of secondary particle type names to position(s). Required if
        `secondary_types` is provided, otherwise ignored.
    secondary_orientations_by_type : Callable, optional
        A mapping of secondary particle type names to orientation(s) in
        quaternion form. Can only be provided if `secondary_types` and
        `secondary_positions_by_type` are also provided.
    """
    def __init__(
        self,
        primary_type: str,
        secondary_types: list[str] = [],
        secondary_positions_by_type: dict[str, list[list[float]]] = {},
        secondary_orientations_by_type: dict[str, list[list[float]]] = {}
    ):
        if secondary_types != [] and not secondary_positions_by_type:
            raise ValueError(
                "'get_secondary_positions_by_type' is required if "
                + "'secondary_types' is provided"
            )

        self.primary_type = str(primary_type)
        self.secondary_types = [str(t) for t in secondary_types]
        self.secondary_positions_by_type = secondary_positions_by_type
        self.secondary_orientations_by_type = secondary_orientations_by_type

        if self.secondary_positions_by_type:
            self.validate_secondary_positions()
        if self.secondary_orientations_by_type:
            self.validate_secondary_orientations()
        if (
            self.secondary_positions_by_type
            and self.secondary_orientations_by_type
        ):
            self.validate_secondary_orientations_and_positions_match()
    
    def must_be_rigid_body(self, interaction: Interaction) -> bool:
        """Whether the body must represent a rigid body for some interaction."""
        common_single_types = any(
            t in self.secondary_types
            for t in interaction.yes_single_types
        )
        common_pair_types = any(
            p[0] in self.secondary_types or p[1] in self.secondary_types
            for p in interaction.yes_pair_types
        )
        return common_single_types or common_pair_types

    def validate_secondary_positions(self):
        """Ensure that the provided callable works for all secondary types."""
        for t in self.secondary_types:
            if t not in self.secondary_positions_by_type.keys():
                raise ValueError(
                    "`secondary_positions_by_type` does not specify positions "
                    + f"for secondary type '{t}'."
                )

    def validate_secondary_orientations(self):
        """Ensure that the provided callable works for all secondary types."""
        for t in self.secondary_types:
            if t not in self.secondary_orientations_by_type.keys():
                raise ValueError(
                    "`secondary_orientations_by_type` does not specify "
                    + f"orientations for secondary type '{t}'."
                )

    def validate_secondary_orientations_and_positions_match(self):
        """Ensure secondary types' numbers of positions and orientations match."""
        for t in self.secondary_types:
            n_positions = len(self.secondary_positions_by_type[t])
            n_orientations = len(self.secondary_orientations_by_type[t])
            
            if n_positions != n_orientations:
                raise ValueError(
                    "The number of positions and orientations for secondary  "
                    + f"type {t} do not match."
                )

    @classmethod
    def from_hoomd_simulation(
        cls,
        simulation: hoomd.Simulation,
        primary_type: str | None = None
    ) -> list[Body] | Body:
        """Parse a hoomd.Simulation object into 1 or more Bodies.

        This is a convenience method that is equivalent to
        
        ```python
        p4.Body.from_hoomd_rigid(sim.operations.integrator.rigid)
        ```
        
        Parameters
        ----------
        rigid : hoomd.md.constrain.Rigid
            The constraint that defines rigid bodies.
        primary_type : str, optional
            The name of the primary type of a single body. If provided, just
            that body is returned. If not provided, all possible bodies are
            returned in a list.
        """
        if simulation.operations.integrator is None:
            raise ValueError("`simulation` must have an integrator")
        if simulation.operations.integrator.rigid is None:
            raise ValueError("integrator must have a rigid constraint")
        return cls.from_hoomd_rigid(
            simulation.operations.integrator.rigid, primary_type
        )

    @classmethod
    def from_hoomd_rigid(
        cls,
        rigid: hoomd.md.constrain.Rigid,
        primary_type: str | None = None
    ) -> list[Body] | Body:
        """Parse a hoomd.md.constrain.Rigid object into 1 or more Bodies.
        
        Parameters
        ----------
        rigid : hoomd.md.constrain.Rigid
            The constraint that defines rigid bodies.
        primary_type : str, optional
            The name of the primary type of a single body. If provided, just
            that body is returned. If not provided, all possible bodies are
            returned in a list.
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
            bodies.append(cls(
                primary_type=p_t,
                secondary_types=unique(rigid.body[p_t]["constituent_types"]),
                secondary_positions_by_type=data_by_type(
                    rigid.body[p_t]["constituent_types"],
                    [list(p) for p in rigid.body[p_t]["positions"]]
                ),
                secondary_orientations_by_type=data_by_type(
                    rigid.body[p_t]["constituent_types"],
                    [list(p) for p in rigid.body[p_t]["orientations"]]
                )
            ))

        if len(bodies) == 1:
            return bodies[0]
        else:
            return bodies

    def __eq__(self, other):
        """Bodies are equal if their attributes are the same or equivalent."""
        if self.primary_type != other.primary_type:
            return False
        else:
            primary_same = True
        
        if self.secondary_types != other.secondary_types:
            return False
        else:
            secondary_same = True
        
        if self.secondary_positions_by_type != other.secondary_positions_by_type:
            return False
        else:
            positions_same = True

        orientations_same = (
            self.secondary_orientations_by_type == other.secondary_orientations_by_type
        )

        orientations_equivalent = False
        if not self.secondary_orientations_by_type:
            if all([
                list(i) == [1, 0, 0, 0]
                for t in other.secondary_types
                for i in other.secondary_orientations_by_type[t]
            ]):
                orientations_equivalent = True
        elif not other.secondary_orientations_by_type:
            if all([
                list(i) == [1, 0, 0, 0]
                for t in self.secondary_types
                for i in self.secondary_orientations_by_type[t]
            ]):
                orientations_equivalent = True

        return (
            primary_same and secondary_same and positions_same
            and (orientations_same or orientations_equivalent)
        )
    
    def __repr__(self):
        return (
            "Body ("
            + f"\n\tprimary_type='{self.primary_type}',"
            + f"\n\tsecondary_types={self.secondary_types},"
            + f"\n\tsecondary_positions_by_type={self.secondary_positions_by_type},"
            + f"\n\tsecondary_orientations_by_type={self.secondary_orientations_by_type},"
            + "\n)"
        )
