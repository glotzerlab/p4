import hoomd
import pytest
from p4 import Body, Interaction
from copy import deepcopy

# Verify that
#   1. instantiation works given valid args
#   2. instantiation fails expectedly given various kinds of invalid args
#   3. methods produce the expected output
#   4. parsing produces the expected output

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

def make_rigid(
    primary_type,
    secondary_types=[],
    secondary_positions_by_type={},
    secondary_orientations_by_type={}
):
    """Create a rigid constraint that corresponds to the args for a Body."""
    rigid = hoomd.md.constrain.Rigid()

    types = []
    positions = []
    orientations = []
    for t in secondary_types:
        for i, p in enumerate(secondary_positions_by_type[t]):
            types.append(t)
            positions.append(p)
            if secondary_orientations_by_type and t in secondary_orientations_by_type.keys():
                orientations.append(secondary_orientations_by_type[t][i])
            else:
                orientations.append([1, 0, 0, 0])

    rigid.body[primary_type] = {
        "constituent_types": types,
        "positions": positions,
        "orientations": orientations
    }

    return rigid

def make_simulation(
    primary_type,
    secondary_types=[],
    secondary_positions_by_type={},
    secondary_orientations_by_type={}
):
    """Create a simulation with a rigid constraint that corresponds to the args for a Body."""
    sim = hoomd.util.make_example_simulation(particle_types=[primary_type] + secondary_types)
    rigid = make_rigid(
        primary_type=primary_type,
        secondary_types=secondary_types,
        secondary_positions_by_type=secondary_positions_by_type,
        secondary_orientations_by_type=secondary_orientations_by_type
    )
    rigid.create_bodies(sim.state)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.1)
    sim.operations.integrator.rigid = rigid
    sim.run(0)  # make sure it runs
    return sim

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_from_hoomd_rigid(kwargs):
    """Ensure parsing from hoomd.md.constrain.Rigid produces the expected output."""
    # Skip case when there are not secondary types
    if "secondary_types" in kwargs.keys():
        body = Body(**kwargs)
        rigid = make_rigid(**kwargs)
        assert body == Body.from_hoomd_rigid(rigid, body.primary_type)

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_from_hoomd_simulation(kwargs):
    """Ensure parsing from hoomd.Simulation produces the expected output."""
    # Skip case when there are not secondary types
    if "secondary_types" in kwargs.keys():
        body = Body(**kwargs)
        simulation = make_simulation(**kwargs)
        assert body == Body.from_hoomd_simulation(simulation, body.primary_type)
