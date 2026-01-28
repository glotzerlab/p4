import itertools
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
        positions_by_type=dict(
            B=[[1,1,1]]
        )
    ),
    dict(   # 1 seconary type, positions and orientations
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(
            B=[[1,1,1]]
        ),
        orientations_by_type=dict(
            B=[[1,1,0,0,]]
        )
    ),
    dict(   # 2 seconary types, positions only
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        )
    ),
    dict(   # 2 seconary types, positions and orientations
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        ),
        orientations_by_type=dict(
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
        positions_by_type=dict()
    ),
    dict(   # 1 secondary type (positions given with wrong key)
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(
            wrong=[[1,1,1]]
        )
    ),
    dict(   # 2 secondary types (first is wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            C=[[1,0,0], [0,1,0]]
        )
    ),
    dict(   # 2 secondary types (second is wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            B=[[1,1,1]]
        )
    ),
    dict(   # 2 secondary types (both are wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict()
    ),

    # Secondary types with orientations, position keys are missing
    dict(   # 1 secondary type
        primary_type="A",
        secondary_types=["B"],
        orientations_by_type=dict(
            B=[[1,1,0,0,]]
        )
    ),
    dict(   # 1 secondary type (positions given with wrong key)
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(
            wrong=[[1,1,1]]
        ),
        orientations_by_type=dict(
            B=[[1,1,0,0,]]
        )
    ),
    dict(   # 2 secondary types (first is wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            C=[[1,0,0], [0,1,0]]
        ),
        orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0], [0,1,0,0]]
        )
    ),
    dict(   # 2 secondary types (second is wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            B=[[1,1,1]]
        ),
        orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0], [0,1,0,0]]
        )
    ),
    dict(   # 2 secondary types (both are wrong)
        primary_type="A",
        secondary_types=["B", "C"],
        orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0], [0,1,0,0]]
        )
    ),

    # Secondary types given with mismatched positions and orientations
    dict(   # 1 secondary type
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(
            B=[[1,1,1]]
        ),
        orientations_by_type=dict(
            B=[[1,1,0,0,], [1,1,0,0,]]
        )
    ),
    dict(   # 2 secondary types (first does not match)
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        ),
        orientations_by_type=dict(
            B=[[1,1,0,0], [1,1,0,0,]],
            C=[[1,0,0,0], [0,1,0,0]]
        )
    ),
    dict(   # 2 secondary types (second does not match)
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        ),
        orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0]]
        )
    ),
    dict(   # 2 secondary types (both do not match)
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        ),
        orientations_by_type=dict(
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
    no_params=dict(
        params=dict(epsilon=0, sigma=1),
        r_cut=0
    ),
    all_types=["A", "B", "C"],
    yes_params={
        ("A", "B"): dict(
            params=dict(epsilon=1, sigma=1),
            r_cut=5
        )
    }
)

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
@pytest.mark.parametrize("variant", ["none", "some", "all"])
def test_is_rigid(kwargs, variant):
    """Ensure method returns True if the body has any secondary types in common with an interction."""
    b = Body(**kwargs)
    interaction_kwargs = deepcopy(VALID_INTERACTION_KWARGS)

    if variant == "none":
        interaction_kwargs["all_types"] = ["Y", "Z"]
        for type_name, param_dict in deepcopy(interaction_kwargs["yes_params"]).items():
            interaction_kwargs["yes_params"][("Y", "Z")] = param_dict # type: ignore
            del interaction_kwargs["yes_params"][type_name]

    elif variant in ["some", "all"]:
        if "secondary_types" in kwargs.keys():
            if variant == "some":
                t = kwargs["secondary_types"][0]
                interaction_kwargs["all_types"] = [t]
                for type_name, param_dict in deepcopy(interaction_kwargs["yes_params"]).items():
                    interaction_kwargs["yes_params"][(t, t)] = param_dict
                    del interaction_kwargs["yes_params"][type_name]

            elif variant == "all":
                interaction_kwargs["all_types"] = kwargs["secondary_types"]
                for type_name, param_dict in deepcopy(interaction_kwargs["yes_params"]).items():
                    for pair in itertools.combinations_with_replacement(kwargs["secondary_types"], 2):
                        interaction_kwargs["yes_params"][pair] = param_dict
                    del interaction_kwargs["yes_params"][type_name]

    i = Interaction(**interaction_kwargs)

    if variant == "none" or "secondary_types" not in kwargs.keys() or len(kwargs["secondary_types"]) == 0:
        assert not b.is_rigid([i])
    elif variant in ["some", "all"]:
        assert b.is_rigid([i])

def make_rigid(
    primary_type,
    secondary_types=[],
    positions_by_type={},
    orientations_by_type={}
):
    """Create a rigid constraint that corresponds to the args for a Body."""
    rigid = hoomd.md.constrain.Rigid()

    types = []
    positions = []
    orientations = []
    for t in secondary_types:
        for i, p in enumerate(positions_by_type[t]):
            types.append(t)
            positions.append(p)
            if orientations_by_type and t in orientations_by_type.keys():
                orientations.append(orientations_by_type[t][i])
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
    positions_by_type={},
    orientations_by_type={}
):
    """Create a simulation with a rigid constraint that corresponds to the args for a Body."""
    sim = hoomd.util.make_example_simulation(particle_types=[primary_type] + secondary_types)
    rigid = make_rigid(
        primary_type=primary_type,
        secondary_types=secondary_types,
        positions_by_type=positions_by_type,
        orientations_by_type=orientations_by_type
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
