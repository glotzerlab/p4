from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from p4 import Interaction


class BodyModel:
    """The names and spatial data for a body's primary and secondary types.

    Every body model must have a primary particle type, but secondary types are
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
        secondary_positions_by_type: dict[str, list[list[float]]] | None = None,
        secondary_orientations_by_type: dict[str, list[list[float]]] | None = None
    ):
        if secondary_types != [] and secondary_positions_by_type is None:
            raise ValueError(
                "'get_secondary_positions_by_type' is required if "
                + "'secondary_types' is provided"
            )

        self.primary_type = primary_type
        self.secondary_types = secondary_types
        self.secondary_positions_by_type = secondary_positions_by_type
        self.secondary_orientations_by_type = secondary_orientations_by_type

        if self.secondary_positions_by_type is not None:
            self.validate_secondary_positions()
        if self.secondary_orientations_by_type is not None:
            self.validate_secondary_orientations()
        if (
            (self.secondary_positions_by_type is not None)
            and (self.secondary_orientations_by_type is not None)
        ):
            self.validate_secondary_orientations_and_positions_match()
    
    def can_be_rigid_body(self) -> bool:
        """Whether the particle can be a rigid body."""
        return len(self.secondary_types) == 0
    
    def must_be_rigid_body(self, interaction: Interaction) -> bool:
        """Whether the model must represent a rigid body for some interaction."""
        return any([t in self.secondary_types for t in interaction.yes_types])

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
    def from_hoomd_simulation(cls, simulation, primary_type):
        pass

    @classmethod
    def from_hoomd_rigid(cls, rigid, primary_type):
        pass
