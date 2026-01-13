# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from p4 import Interaction

from marshmallow import (
    Schema, fields, post_load, validate, ValidationError, validates_schema
)

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
        _ = BodySchema().load(dict(
            primary_type = str(primary_type),
            secondary_types = [str(t) for t in secondary_types],
            secondary_positions_by_type = secondary_positions_by_type,
            secondary_orientations_by_type = secondary_orientations_by_type
        ))

        self.primary_type = str(primary_type)
        self.secondary_types = [str(t) for t in secondary_types]
        self.secondary_positions_by_type = secondary_positions_by_type
        self.secondary_orientations_by_type = secondary_orientations_by_type
    
    def must_be_rigid_body(self, interaction: Interaction) -> bool:
        """Whether the body must represent a rigid body for some interaction."""
        return any([t in self.secondary_types for t in interaction.yes_types])

    @classmethod
    def from_hoomd_rigid(cls, rigid, primary_type):
        pass

    def from_hoomd_snapshot(cls, snapshot, secondary_types_by_primary_type):
        pass

    @classmethod
    def from_hoomd_simulation(cls, simulation, primary_type):
        pass

class BodySchema(Schema):
    primary_type = fields.Str(required=True)
    secondary_types = fields.List(fields.Str())
    secondary_positions_by_type = fields.Dict(
        keys=fields.Str(),
        values=fields.List(
            fields.List(fields.Float(), validate=validate.Length(equal=3))
        ),
    )
    secondary_orientations_by_type = fields.Dict(
        keys=fields.Str(),
        values=fields.List(
            fields.List(fields.Float(), validate=validate.Length(equal=4))
        ),
    )

    @validates_schema
    def validate_no_type_clash(self, data, **kwargs):
        errors = {}

        if data["primary_type"] in data["secondary_types"]:
            errors["secondary_types"] = ["must not include `primary_type`"]
        
        if errors:
            raise ValidationError(errors)

    @validates_schema
    def validate_no_spatial_data_without_secondary_types(self, data, **kwargs):
        errors = {}

        if "secondary_types" not in data.keys():
            if "secondary_positions_by_type" in data.keys():
                errors["secondary_positions_by_type"] = ["cannot be provided if `secondary_types` is not provided"]
            if "secondary_orientations_by_type" in data.keys():
                errors["secondary_orientations_by_type"] = ["cannot be provided if `secondary_types` is not provided"]
        
        if errors:
            raise ValidationError(errors)

    @validates_schema
    def validate_secondary_types_require_positions(self, data, **kwargs):
        errors = {}
        
        if "secondary_types" in data.keys():
            if "secondary_positions_by_type" not in data.keys():
                errors["secondary_positions_by_type"] = ["required if `secondary_types` is provided"]
        
        if errors:
            raise ValidationError(errors)

    @validates_schema
    def validate_all_types_in_positions(self, data, **kwargs):
        errors = {}
        suberrors = []

        if (
            ("secondary_types" in data.keys())
            and ("secondary_positions_by_type" in data.keys())
        ):
            for t in data["secondary_types"]:
                if t not in data["secondary_positions_by_type"].keys():
                    suberrors.append(f"missing '{t}' from `secondary_types`")
        
            errors["secondary_positions_by_type"] = suberrors

        if suberrors:
            raise ValidationError(errors)

    @validates_schema    
    def validate_orientations_require_positions(self, data, **kwargs):
        errors = {}

        if (
            ("secondary_orientations_by_type" in data.keys())
            and ("secondary_positions_by_type" not in data.keys())
        ):
            errors["secondary_orientations_by_type"] = "can only be provided if `secondary_positions_by_type` is also provided"
        
        if errors:
            raise ValidationError(errors)

    @validates_schema
    def validate_orientations_and_positions_match(self, data, **kwargs):
        errors = {}
        suberrors = []
        if (
            ("secondary_orientations_by_type" in data.keys())
            and ("secondary_positions_by_type" in data.keys())
        ):
            for t in data["secondary_positions_by_type"].keys():
                if t in data["secondary_orientations_by_type"].keys():
                    n_p = len(data["secondary_positions_by_type"][t])
                    n_o = len(data["secondary_orientations_by_type"][t])
                    if n_p !=  n_o:
                        suberrors.append(f"for type {t}, the number of positions ({n_p}) does not match the number of orientations ({n_o})")
            
            errors["secondary_orientations_by_type"] = suberrors
        
        if suberrors:
            raise ValidationError(errors)

    # @staticmethod
    # def validate_constructor(fn):


    # def make_body(self, data, **kwargs):
    #     return Body(**data)