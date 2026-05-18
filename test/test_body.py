import itertools
import tempfile
import gsd
import hoomd
import numpy as np
import pytest
import p4
from copy import copy, deepcopy
import coxeter
from pathlib import Path

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
        ),
        mass_by_type=dict(A=2, B=3),
        moi_by_type=dict(A=[1, 0, 0], B=[0, 1, 0])
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
        ),
    ),
    dict(   # 2 seconary types, positions and orientations (plus masses and mois)
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            B=[[1,1,1]],
            C=[[1,0,0], [0,1,0]]
        ),
        orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[1,0,0,0], [0,1,0,0]]
        ),
        mass_by_type=dict(A=2, B=3),
        moi_by_type=dict(A=[1, 0, 0], B=[0, 1, 0])
    ),
]

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_valid_instantiation(kwargs):
    """Ensure instantiation works with valid arguments."""
    _ = p4.Body(**kwargs)

INVALID_KWARGS = [
    # Secondary types given with no orientations, position keys are missing/wrong
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
    dict(   # 2 secondary types (first is missing)
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            C=[[1,0,0], [0,1,0]]
        )
    ),
    dict(   # 2 secondary types (second is missing)
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            B=[[1,1,1]]
        )
    ),
    dict(   # 2 secondary types (both are missing)
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict()
    ),

    # Secondary types with orientations, position keys are missing/wrong
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
    dict(   # 2 secondary types (first is missing)
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
    dict(   # 2 secondary types (second is missing)
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
    dict(   # 2 secondary types (both are missing)
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
    with pytest.raises((TypeError, ValueError)):
        _ = p4.Body(**kwargs)

VALID_INTERACTION_KWARGS = dict(
    hoomd_class=hoomd.md.pair.LJ,
    initial_args=dict(),
    default_params=dict(
        params=dict(epsilon=0, sigma=1),
        r_cut=0
    ),
    typed_params={
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
    b = p4.Body(**kwargs)
    interaction_kwargs = deepcopy(VALID_INTERACTION_KWARGS)

    if variant == "none":
        for type_name, param_dict in deepcopy(interaction_kwargs["typed_params"]).items():
            interaction_kwargs["typed_params"][("Y", "Z")] = param_dict # type: ignore
            del interaction_kwargs["typed_params"][type_name]

    elif variant in ["some", "all"]:
        if "secondary_types" in kwargs.keys():
            if variant == "some":
                t = kwargs["secondary_types"][0]
                for type_name, param_dict in deepcopy(interaction_kwargs["typed_params"]).items():
                    interaction_kwargs["typed_params"][(t, t)] = param_dict
                    del interaction_kwargs["typed_params"][type_name]

            elif variant == "all":
                for type_name, param_dict in deepcopy(interaction_kwargs["typed_params"]).items():
                    for pair in itertools.combinations_with_replacement(kwargs["secondary_types"], 2):
                        interaction_kwargs["typed_params"][pair] = param_dict
                    del interaction_kwargs["typed_params"][type_name]

    i = p4.Interaction(**interaction_kwargs)

    if variant == "none" or "secondary_types" not in kwargs.keys() or len(kwargs["secondary_types"]) == 0:
        assert not b._is_rigid([i])
    elif variant in ["some", "all"]:
        assert b._is_rigid([i])

def make_rigid(
    primary_type,
    secondary_types=[],
    positions_by_type={},
    orientations_by_type={},
    mass_by_type={},    # ignored
    moi_by_type={}      # ignored
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

def merge_rigids(rigid1, rigid2):
    """Combine two rigid constraints together."""
    merged = hoomd.md.constrain.Rigid()
    
    for primary_type, body_dict in rigid1.body.items():
        merged.body[primary_type] = body_dict
    
    for primary_type, body_dict in rigid2.body.items():
        merged.body[primary_type] = body_dict
    
    return merged

def make_rigid_and_expected_bodies_for_variant(
    single_or_multi_body,
    single_or_multi_particle,
):
    """Create a rigid constraint that matches the variants and return it with the expected bodies."""
    rigid = hoomd.md.constrain.Rigid()

    if single_or_multi_body == "single" and single_or_multi_particle == "single":
        rigid.body["A"] = None
        expected_bodies = [p4.Body("A")]

    elif single_or_multi_body == "multi" and single_or_multi_particle == "single":
        rigid.body["A"] = None
        rigid.body["B"] = dict(constituent_types=[], positions=[], orientations=[])
        expected_bodies = [p4.Body("A"), p4.Body("B")]

    elif single_or_multi_body == "single" and single_or_multi_particle == "multi":
        rigid.body["A"] = dict(
            constituent_types=["B", "C"],
            positions=[[1, 0, 0], [0, 1, 0]],
            orientations=[[0, 1, 0, 0], [0, 0, 1, 0]]
        )
        expected_bodies = [
            p4.Body(
                primary_type="A",
                secondary_types=["B", "C"],
                positions_by_type=dict(B=[[1, 0, 0]], C=[[0, 1, 0]]),
                orientations_by_type=dict(B=[[0, 1, 0, 0]], C=[[0, 0, 1, 0]])
            )
        ]

    elif single_or_multi_body == "multi" and single_or_multi_particle == "multi":
        rigid.body["A"] = dict(
            constituent_types=["B", "C"],
            positions=[[1, 0, 0], [0, 1, 0]],
            orientations=[(0, 1, 0, 0), (0, 0, 1, 0)]
        )
        rigid.body["D"] = dict(
            constituent_types=["E", "F"],
            positions=[[-1, 0, 0], [0, -1, 0]],
            orientations=[(0, 1, 0, 0), (0, 0.707, 0.707, 0)]
        )
        expected_bodies = [
            p4.Body(
                primary_type="A",
                secondary_types=["B", "C"],
                positions_by_type=dict(B=[[1, 0, 0]], C=[[0, 1, 0]]),
                orientations_by_type=dict(B=[[0, 1, 0, 0]], C=[[0, 0, 1, 0]])
            ),
            p4.Body(
                primary_type="D",
                secondary_types=["E", "F"],
                positions_by_type=dict(E=[[-1, 0, 0]], F=[[0, -1, 0]]),
                orientations_by_type=dict(E=[[0, 1, 0, 0]], F=[[0, 0.707, 0.707, 0]])
            )
        ]
    
    return rigid, expected_bodies

@pytest.mark.parametrize("single_or_multi_body", ["single", "multi"])
@pytest.mark.parametrize("single_or_multi_particle", ["single", "multi"])
@pytest.mark.parametrize("include_singles", [False, True])
def test_from_hoomd_rigid(
    single_or_multi_body,
    single_or_multi_particle,
    include_singles,
):
    """Ensure parsing from hoomd.md.constrain.Rigid produces the expected output.
    
    Parameters
    ----------
    single_or_multi_body : 'single' or 'multi'
        Whether there is one or more vodies in the rigid constraint.
    single_or_multi_particle : 'single' or 'multi'
        Whether the bodies in the rigid constraint are single or multi-particle.
        Note: in multi-body rigid with single particles, one body is set to
        None and the other is set to the empty dict.
    include_singles : bool
        Whether to include single-particle bodies.
    """
    rigid, expected_bodies = make_rigid_and_expected_bodies_for_variant(
        single_or_multi_body,
        single_or_multi_particle,
    )
    if single_or_multi_particle == "single" and not include_singles:
        assert p4.Body.from_hoomd_rigid(rigid, include_singles) == []
    else:
        assert p4.Body.from_hoomd_rigid(rigid, include_singles) == expected_bodies

def make_simulation(
    primary_type,
    secondary_types=[],
    positions_by_type={},
    orientations_by_type={},
    mass_by_type={},
    moi_by_type={}
):
    """Create a simulation with a rigid constraint that corresponds to the args for a Body."""
    simulation = hoomd.Simulation(device=hoomd.device.CPU())
    
    frame = gsd.hoomd.Frame()
    frame.particles.types = [primary_type] + secondary_types
    frame.particles.N = 1
    
    simulation.create_state_from_snapshot(
        hoomd.Snapshot.from_gsd_frame(
            gsd_snap=frame,
            communicator=hoomd.communicator.Communicator()
        )
    )

    rigid = make_rigid(
        primary_type=primary_type,
        secondary_types=secondary_types,
        positions_by_type=positions_by_type,
        orientations_by_type=orientations_by_type
    )

    rigid.create_bodies(simulation.state)

    # Add mass and moi if necessary
    snapshot = simulation.state.get_snapshot()
    if mass_by_type:
        for i, t in enumerate(snapshot.particles.types):
            if t in mass_by_type:
                t_indices = np.argwhere(snapshot.particles.typeid == i)
                snapshot.particles.mass[t_indices] = mass_by_type[t]

    if moi_by_type:
        for i, t in enumerate(snapshot.particles.types):
            if t in moi_by_type:
                t_indices = np.argwhere(snapshot.particles.typeid == i)
                snapshot.particles.moment_inertia[t_indices] = moi_by_type[t]     
    
    simulation.state.set_snapshot(snapshot)

    simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)
    simulation.operations.integrator.rigid = rigid
    simulation.run(0)  # make sure it runs

    return simulation

@pytest.mark.parametrize("has_integrator", [False, True])
@pytest.mark.parametrize("has_particles_in_state", [False, True])
@pytest.mark.parametrize("include_singles", [False, True])
def test_from_hoomd_simulation_without_rigid(
    has_integrator,
    has_particles_in_state,
    include_singles,
):
    """Ensure parsing from a simulation without a rigid constraint produces the expected output."""
    if has_particles_in_state:
        simulation = hoomd.util.make_example_simulation(particle_types=["A"])
        expected_bodies = [p4.Body("A")] if include_singles else []
    else:
        simulation = hoomd.Simulation(device=hoomd.device.CPU())
        simulation.create_state_from_snapshot(hoomd.Snapshot())
        expected_bodies = []
    
    if has_integrator:
        simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)
    
    assert p4.Body.from_hoomd_simulation(simulation, include_singles) == expected_bodies

def make_snapshot_for_variant(
    single_or_multi_body,
    single_or_multi_particle,
):
    """Create a snapshot that matches the variants."""
    frame = gsd.hoomd.Frame()
    frame.configuration.box = [100, 100, 100, 0, 0, 0]

    if single_or_multi_body == "single" and single_or_multi_particle == "single":
        frame.particles.types = ["A"]
        frame.particles.typeid = np.array([0, 0], dtype=np.uint32)
        frame.particles.position = np.array([[0, 1, 0], [0, 2, 0]], dtype=np.float32)
        frame.particles.orientation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        frame.particles.body = np.array([-1, -1], dtype=np.int32)

    elif single_or_multi_body == "multi" and single_or_multi_particle == "single":
        frame.particles.types = ["A", "B"]
        frame.particles.typeid = np.array([0, 1], dtype=np.uint32)
        frame.particles.position = np.array([[0, 1, 0], [0, 2, 0]], dtype=np.float32)
        frame.particles.orientation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        frame.particles.body = np.array([-1, -1], dtype=np.int32)

    elif single_or_multi_body == "single" and single_or_multi_particle == "multi":
        frame.particles.types = ["A", "B", "C"]
        frame.particles.typeid = np.array([0, 1, 2], dtype=np.uint32)
        frame.particles.position = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        frame.particles.orientation = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
        frame.particles.body = np.array([0, 0, 0], dtype=np.int32)

    elif single_or_multi_body == "multi" and single_or_multi_particle == "multi":
        frame.particles.types = ["A", "B", "C", "D", "E", "F"]
        frame.particles.typeid = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
        frame.particles.position = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [-1, 0, 1],
                [0, -1, 1],
            ],
            dtype=np.float32
        )
        frame.particles.orientation = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, -0.707, -0.707, 0],
            ],
            dtype=np.float32
        )
        frame.particles.body = np.array([0, 0, 0, 3, 3, 3], dtype=np.int32)

    frame.particles.N = frame.particles.position.shape[0]
    
    return hoomd.Snapshot.from_gsd_frame(frame, communicator=hoomd.communicator.Communicator())
    
@pytest.mark.parametrize("single_or_multi_body", ["single", "multi"])
@pytest.mark.parametrize("single_or_multi_particle", ["single", "multi"])
@pytest.mark.parametrize("include_singles", [False, True])
@pytest.mark.parametrize("has_particles_in_state", [False, True])
def test_from_hoomd_simulation_with_rigid(
    single_or_multi_body,
    single_or_multi_particle,
    include_singles,
    has_particles_in_state,
):
    """Ensure parsing from a simulation with a rigid constraint produces the expected output."""
    rigid, expected_bodies_from_rigid = make_rigid_and_expected_bodies_for_variant(
        single_or_multi_body,
        single_or_multi_particle,
    )

    expected_bodies = []
    expected_bodies.extend(expected_bodies_from_rigid)

    simulation = hoomd.Simulation(hoomd.device.CPU())
    simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)
    
    if has_particles_in_state:
        snapshot = make_snapshot_for_variant(
            single_or_multi_body,
            single_or_multi_particle,
        )

        simulation.create_state_from_snapshot(snapshot)

    else:
        simulation.create_state_from_snapshot(hoomd.Snapshot())


    simulation.operations.integrator.rigid = rigid

    # Test
    if single_or_multi_particle == "single" and not include_singles:
        assert p4.Body.from_hoomd_simulation(simulation, include_singles) == []
    else:
        actual_bodies = p4.Body.from_hoomd_simulation(simulation, include_singles)
        assert len(actual_bodies) == len(expected_bodies)
        assert all(b in actual_bodies for b in expected_bodies)

@pytest.mark.parametrize("single_or_multi_body", ["single", "multi"])
@pytest.mark.parametrize("single_or_multi_particle", ["single", "multi"])
@pytest.mark.parametrize("include_singles", [False, True])
@pytest.mark.parametrize("has_particles_in_state", [False, True])
def test_from_hoomd_snapshot(
    single_or_multi_body,
    single_or_multi_particle,
    include_singles,
    has_particles_in_state,
):
    """Ensure parsing from a snapshot produces the expected result."""
    snapshot = make_snapshot_for_variant(
        single_or_multi_body,
        single_or_multi_particle,
    )

    expected_bodies = p4.Body.from_hoomd_snapshot(snapshot, include_singles)

    if not has_particles_in_state:
        original_snapshot = snapshot
        snapshot = hoomd.Snapshot()
        snapshot.particles.types = original_snapshot.particles.types

        expected_bodies = []
    
    if not include_singles:
        expected_bodies = [b for b in expected_bodies if b.secondary_types]
    
    actual_bodies = p4.Body.from_hoomd_snapshot(snapshot, include_singles)
    assert len(actual_bodies) == len(expected_bodies)
    assert all(b in actual_bodies for b in expected_bodies)

def assert_rigids_are_equal(rigid1, rigid2):
    """Assert that two rigid constraints are equivalent."""
    r1_body = rigid1.body.to_base()
    r2_body = rigid2.body.to_base()
    assert r1_body == r2_body

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_to_hoomd_rigid(kwargs):
    """Ensure converting to hoomd.md.constrain.Rigid produces the expected output."""
    kwargs = copy(kwargs)

    body = p4.Body(**kwargs)

    # Test with all secondary types and no existing rigid constraint
    rigid = make_rigid(**kwargs)
    assert_rigids_are_equal(rigid, body.to_hoomd_rigid())

    # Test with existing rigid constraint
    existing_rigid = make_rigid("X", ["Y"], dict(Y=[[0,0,1], [0,0,-1]]))
    merged_rigid = merge_rigids(existing_rigid, rigid)
    assert_rigids_are_equal(merged_rigid, body.to_hoomd_rigid(rigid=existing_rigid))

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_to_hoomd_snapshot(kwargs):
    """Ensure export to snapshot produces the expected output."""
    arrangement = p4.Body(**kwargs)
    simulation = make_simulation(**kwargs)
    ref_snap = simulation.state.get_snapshot()
    test_snap = arrangement.to_hoomd_snapshot()

    ref_typeids = ref_snap.particles.typeid.tolist()
    ref_positions = np.round(ref_snap.particles.position, 3).tolist()       # rounded because run(0) warps the exact values
    ref_orientations = np.round(ref_snap.particles.orientation, 3).tolist()
    
    test_typeids = test_snap.particles.typeid.tolist()
    test_positions = np.round(test_snap.particles.position, 3).tolist()
    test_orientations = np.round(test_snap.particles.orientation, 3).tolist()

    ref_data = list(zip(ref_typeids, ref_positions, ref_orientations))
    test_data = list(zip(test_typeids, test_positions, test_orientations))
    
    assert ref_snap.particles.N == test_snap.particles.N
    assert ref_snap.particles.types == test_snap.particles.types
    assert all(row in test_data for row in ref_data)

REFERENCE_FOLDER = Path(__file__).parent / "data"

CUBE_VERTICES = [
    [-1/4, -1/4, -1/4],
    [-1/4, -1/4,  1/4],
    [-1/4,  1/4, -1/4],
    [-1/4,  1/4,  1/4],
    [ 1/4, -1/4, -1/4],
    [ 1/4, -1/4,  1/4],
    [ 1/4,  1/4, -1/4],
    [ 1/4,  1/4,  1/4]
]

@pytest.mark.parametrize("kwargs,ref_filename,type_shapes", [
    [    # 2 seconary types, positions and orientations, type shapes are specified
        dict(
            primary_type="A",
            secondary_types=["B", "C"],
            positions_by_type=dict(
                B=[[1,1,1]],
                C=[[1,0,0], [0,1,0]]
            ),
            orientations_by_type=dict(
                B=[[1,1,0,0]],
                C=[[1,0,0,0], [0,1,0,0]]
            ),
            mass_by_type=dict(A=2, B=3),
            moi_by_type=dict(A=[1, 0, 0], B=[0, 1, 0])
        ),
        "body.gsd",
        dict(
            A=coxeter.shapes.Ellipsoid(a=0.25, b=0.5, c=0.75),
            B=coxeter.shapes.ConvexPolyhedron(vertices=CUBE_VERTICES),
        )
    ]
])
def test_to_gsd(kwargs, ref_filename, type_shapes):
    """Ensure export to GSD file produces the expected output."""
    body = p4.Body(**kwargs)

    with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
        test_path = Path(tempdir) / f"test_body.gsd"
        body.to_gsd(test_path, type_shapes=type_shapes)

        with gsd.hoomd.open(test_path, "r") as test_file:
            test_frame = test_file[0]
        
        with gsd.hoomd.open(REFERENCE_FOLDER / ref_filename, "r") as ref_file:
            ref_frame = ref_file[0]

        assert test_frame.particles.N == ref_frame.particles.N
        assert test_frame.particles.types == ref_frame.particles.types
        assert np.array_equal(test_frame.particles.typeid, ref_frame.particles.typeid)
        assert np.array_equal(test_frame.particles.position, ref_frame.particles.position)
        assert np.array_equal(test_frame.particles.orientation, ref_frame.particles.orientation)
        assert test_frame.particles.type_shapes == ref_frame.particles.type_shapes
        assert np.array_equal(test_frame.particles.mass, ref_frame.particles.mass)
        assert np.array_equal(test_frame.particles.moment_inertia, ref_frame.particles.moment_inertia)
        assert np.array_equal(test_frame.particles.body, ref_frame.particles.body)

