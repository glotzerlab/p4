import hoomd
import pytest
from p4 import Body, Interaction
from copy import deepcopy

# Verify that
#   1. instantiation works given valid args
#   2. instantiation fails expectedly given various kinds of invalid args
#   3. methods produce the expected output

VALID_KWARGS = [
    # Required kawrgs only
    dict(
        primary_type="A",
    ),

    # Required and optional kwargs
    dict(   # 1 seconary type, positions only
        primary_type="A",
        secondary_types=["B"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]]
        )
    ),
    dict(   # 1 seconary type, positions and orientations
        primary_type="A",
        secondary_types=["B"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0,]]
        )
    ),
    dict(   # 2 seconary types, positions only
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        )
    ),
    dict(   # 2 seconary types, positions and orientations
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0], [0,1,0,0]]
        )
    ),
]

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_valid_instantiation(kwargs):
    """Ensure instantiation works with valid arguments."""
    _ = Body(**kwargs)

INVALID_KWARGS = [
    # Secondary types given with no orientations, position keys are missing
    dict(   # 1 secondary type (no positions given)
        primary_type="A",
        secondary_types=["B"]
    ),
    dict(   # 1 secondary type (empty positions given)
        primary_type="A",
        secondary_types=["B"],
        secondary_positions_by_type=dict()
    ),
    dict(   # 1 secondary type (positions given with wrong key)
        primary_type="A",
        secondary_types=["B"],
        secondary_positions_by_type=dict(
            wrong=[[1,1,1]]
        )
    ),
    dict(   # 2 secondary types (first is wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            C=[[1,0,0], [0,1,0]]
        )
    ),
    dict(   # 2 secondary types (second is wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]]
        )
    ),
    dict(   # 2 secondary types (both are wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict()
    ),

    # Secondary types with orientations, position keys are missing
    dict(   # 1 secondary type
        primary_type="A",
        secondary_types=["B"],
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0,]]
        )
    ),
    dict(   # 1 secondary type (positions given with wrong key)
        primary_type="A",
        secondary_types=["B"],
        secondary_positions_by_type=dict(
            wrong=[[1,1,1]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0,]]
        )
    ),
    dict(   # 2 secondary types (first is wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            C=[[1,0,0], [0,1,0]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0], [0,1,0,0]]
        )
    ),
    dict(   # 2 secondary types (second is wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0], [0,1,0,0]]
        )
    ),
    dict(   # 2 secondary types (both are wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0], [0,1,0,0]]
        )
    ),

    # Secondary types given with mismatched positions and orientations
    dict(   # 1 secondary type
        primary_type="A",
        secondary_types=["B"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0,], [1,1,0,0,]]
        )
    ),
    dict(   # 2 secondary types (first does not match)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0], [1,1,0,0,]],
            C=[[1,0,0,0], [0,1,0,0]]
        )
    ),
    dict(   # 2 secondary types (second does not match)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0]]
        )
    ),
    dict(   # 2 secondary types (both do not match)
        primary_type="A",
        secondary_types=["B", "C"],
        secondary_positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        ),
        secondary_orientations_by_type=dict(
            B=[[1,1,0,0], [1,1,0,0,]],
            C=[[1,0,0,0]]
        )
    )
]

@pytest.mark.parametrize("kwargs", INVALID_KWARGS)
def test_invalid_instantiation(kwargs):
    """Ensure instantiation fails predictably with invalid arguments."""
    with pytest.raises(ValueError):
        _ = Body(**kwargs)

VALID_INTERACTION_KWARGS = dict(
    hoomd_class=hoomd.md.pair.LJ,
    initial_args=dict(),
    no_single_typed_attributes=dict(),
    no_pair_typed_attributes=dict(
        params=dict(epsilon=0, sigma=1),
        r_cut=0
    ),
    yes_types=["A", "B"],
    yes_single_typed_attributes=dict(),
    yes_pair_typed_attributes=dict(
        params=dict(epsilon=1, sigma=1),
        r_cut=5
    )
)

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
@pytest.mark.parametrize("variant", ["none", "some", "all"])
def test_must_be_rigid_body(kwargs, variant):
    """Ensure method returns True if the body has any secondary types in common with an interction."""
    b = Body(**kwargs)
    interaction_kwargs = deepcopy(VALID_INTERACTION_KWARGS)

    if variant == "none":
        interaction_kwargs["yes_types"] = ["Z"]
    elif variant in ["some", "all"]:
        if "secondary_types" in kwargs.keys():
            if variant == "some":
                interaction_kwargs["yes_types"] = [kwargs["secondary_types"][0]]
            elif variant == "all":
                interaction_kwargs["yes_types"] = kwargs["secondary_types"]

    i = Interaction(**interaction_kwargs)

    if variant == "none" or "secondary_types" not in kwargs.keys() or len(kwargs["secondary_types"]) == 0:
        assert b.must_be_rigid_body(i) == False
    elif variant in ["some", "all"]:
        assert b.must_be_rigid_body(i) == True
