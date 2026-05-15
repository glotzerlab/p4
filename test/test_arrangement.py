from collections import defaultdict
from pathlib import Path
import tempfile

import coxeter
import gsd
import hoomd
import numpy as np
import pytest

import p4

VALID_KWARGS = [
    # 1 single-particle body, without orientations
    dict(
        bodies=[p4.Body("A")],
        positions_by_type=dict(A=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ),

    # 2 multi-particle bodies, with orientations
    dict(
        bodies=[
            p4.Body(
                primary_type="A",
                secondary_types=["B", "C"],
                positions_by_type=dict(
                    B=[[-1,0,0]],
                    C=[[1,0,0]]
                )
            ),
            p4.Body(
                primary_type="D",
                secondary_types=["E", "F"],
                positions_by_type=dict(
                    E=[[0, -1, 0], [0, 1, 0]],
                    F=[[0, 0, -1], [0, 0, 1]]
                ),
                orientations_by_type=dict(
                    E=[[1,0,0,0], [0, 0.707, 0.707, 0]],
                    F=[[0.707, 0, -0.707, 0], [0.707, 0, 0.707, 0]]
                ),
            )
        ],
        positions_by_type=dict(
            A=[[10, 0, 0]],
            D=[[0, 10, 0], [0, 20, 0]]
        ),
        orientations_by_type=dict(
            A=[[1,0,0,0]],
            D=[[0, 0.707, 0.707, 0], [0.707, 0, 0.707, 0]]
        )
    )
]

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_valid_instantiation(kwargs):
    """Ensure instantiation works with valid arguments."""
    _ = p4.Arrangement(**kwargs)

INVALID_KWARGS = [
    # Wrong body types
    dict(
        bodies=[dict(primary_type="A", secondary_types=[], positions_by_type={}, orientations_by_type={})],
        positions_by_type=dict(A=[[0,0,0]]),
    ),

    # Bodies given with no orientations, position keys are missing/wrong
    dict(   # 1 body (empty positions given)
        bodies=[p4.Body("A")],
        positions_by_type=dict(),
    ),
    dict(   # 1 body (positions given with wrong key)
        bodies=[p4.Body("A")],
        positions_by_type=dict(B=[[0,0,0]]),
    ),
    dict(   # 2 bodies (first is missing)
        bodies=[p4.Body("A"), p4.Body("B")],
        positions_by_type=dict(B=[[0,0,0]]),
    ),
    dict(   # 2 bodies (second is missing)
        bodies=[p4.Body("A"), p4.Body("B")],
        positions_by_type=dict(A=[[0,0,0]]),
    ),
    dict(   # 2 bodies (both are missing but there's still a key)
        bodies=[p4.Body("A"), p4.Body("B")],
        positions_by_type=dict(wrong=[[0,0,0]]),
    ),

    # Bodies with orientations, position keys are missing/wrong
    dict(   # 1 body
        bodies=[p4.Body("A")],
        positions_by_type=dict(),
        orientations_by_type=dict(A=[[1,0,0,0]]),
    ),
    dict(   # 1 body (positions given with wrong key)
        bodies=[p4.Body("A")],
        positions_by_type=dict(wrong=[[0,0,0]]),
        orientations_by_type=dict(A=[[1,0,0,0]]),
    ),
    dict(   # 2 bodies (first is missing)
        bodies=[p4.Body("A"), p4.Body("B")],
        positions_by_type=dict(B=[[0,0,0]]),
        orientations_by_type=dict(A=[[1,0,0,0]], B=[[1,0,0,0]]),
    ),
    dict(   # 2 bodies (second is missing)
        bodies=[p4.Body("A"), p4.Body("B")],
        positions_by_type=dict(A=[[0,0,0]]),
        orientations_by_type=dict(A=[[1,0,0,0]], B=[[1,0,0,0]]),
    ),
    dict(   # 2 bodies (both are missing but there's still a key)
        bodies=[p4.Body("A"), p4.Body("B")],
        positions_by_type=dict(wrong=[[0,0,0]]),
        orientations_by_type=dict(A=[[1,0,0,0]], B=[[1,0,0,0]]),
    ),

    # Bodies given with mismatched positions and orientations
    dict(   # 1 body
        bodies=[p4.Body("A")],
        positions_by_type=dict(A=[[0,0,0]]),
        orientations_by_type=dict(A=[[1,0,0,0], [1,0,0,0]]),
    ),
    dict(   # 2 bodies (first does not match)
        bodies=[p4.Body("A"), p4.Body("B")],
        positions_by_type=dict(A=[[0,0,0]], B=[[1,1,1]]),
        orientations_by_type=dict(A=[[1,0,0,0], [1,0,0,0]], B=[[1,0,0,0]]),
    ),
    dict(   # 2 bodies (second does not match)
        bodies=[p4.Body("A"), p4.Body("B")],
        positions_by_type=dict(A=[[0,0,0]], B=[[1,1,1]]),
        orientations_by_type=dict(A=[[1,0,0,0]], B=[[1,0,0,0], [1,0,0,0]]),
    ),
    dict(   # 2 bodies (both do not match)
        bodies=[p4.Body("A"), p4.Body("B")],
        positions_by_type=dict(A=[[0,0,0]], B=[[1,1,1]]),
        orientations_by_type=dict(A=[[1,0,0,0], [1,0,0,0]], B=[[1,0,0,0], [1,0,0,0]]),
    )
]

@pytest.mark.parametrize("kwargs", INVALID_KWARGS)
def test_invalid_instantiation(kwargs):
    """Ensure instantiation fails predictably with invalid arguments."""
    with pytest.raises((TypeError, ValueError)):
        _ = p4.Arrangement(**kwargs)

def make_simulation(
    bodies,
    positions_by_type,
    orientations_by_type={}
):
    """Create a simulation with a state that matches the kwargs for an Arrangement."""
    # Create rigid
    rigid = hoomd.md.constrain.Rigid()

    for b in bodies:
        rigid = b.to_hoomd_rigid(rigid)
    
    # Create simulation state via GSD frame
    frame = gsd.hoomd.Frame()

    # types = list(positions_by_type.keys())
    types = []
    for b in bodies:
        types.append(b.primary_type)
        types.extend(b.secondary_types)

    typeids = []
    positions = []#np.empty((1,3), dtype=np.float32)
    orientations = []#np.empty((1,4), dtype=np.float32)

    for t, ps in positions_by_type.items():
        os = orientations_by_type.get(t, [[1,0,0,0] for _ in ps])
        
        tid = types.index(t)

        typeids.extend([tid for _ in ps])
        positions.extend(ps)
        orientations.extend(os)

    frame.particles.N = len(positions)
    frame.particles.types = types
    frame.particles.typeid = typeids
    frame.particles.position = np.array(positions)
    frame.particles.orientation = np.array(orientations)
    frame.configuration.box = [
        10 * frame.particles.position[:,0].max(),
        10 * frame.particles.position[:,1].max(),
        10 * frame.particles.position[:,2].max(),
        0.0,
        0.0,
        0.0
    ]
    
    snapshot = hoomd.Snapshot.from_gsd_frame(
        gsd_snap=frame,
        communicator=hoomd.communicator.Communicator()
    )

    simulation = hoomd.Simulation(device=hoomd.device.CPU())
    simulation.create_state_from_snapshot(snapshot)

    # Add secondary types to simulation state via rigid.create_bodies
    rigid.create_bodies(simulation.state)

    # Ensure it runs
    simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)
    simulation.operations.integrator.rigid = rigid
    simulation.run(0)

    return simulation

def assert_arrangements_are_equal(a, b):
    """Arrangements are equivalent if their bodies, positions, and orientations are equivalent."""
    assert (
        len(a.bodies) == len(b.bodies)
        and all([i in b.bodies for i in a.bodies])
    )
    assert (
        a.positions_by_type.keys() == b.positions_by_type.keys()
        and all(
            np.isclose(a.positions_by_type[t], b.positions_by_type[t]).all()
            for t in a.positions_by_type
        )
    )
    assert (
        all(
            np.isclose(
                a.orientations_by_type.get(t, [1, 0, 0, 0]),
                b.orientations_by_type.get(t, [1, 0, 0, 0])
            ).all()
            for t in set(a.orientations_by_type.keys()).union(b.orientations_by_type.keys())
        )
        or (
            a.orientations_by_type == {}
            and all(
                list(i) == [1, 0, 0, 0]
                for v in b.orientations_by_type.values()
                for i in v
            )
        )
        or (
            b.orientations_by_type == {}
            and all(
                list(i) == [1, 0, 0, 0]
                for v in a.orientations_by_type.values()
                for i in v
            )
        )
    )

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_from_hoomd_simulation(kwargs):
    """Ensure parsing from hoomd simulation produces the expected output."""
    arrangement = p4.Arrangement(**kwargs)
    simulation = make_simulation(**kwargs)
    include_singles = any(b.secondary_types == [] for b in kwargs["bodies"])
    assert_arrangements_are_equal(
        arrangement,
        p4.Arrangement.from_hoomd_simulation(simulation, include_singles)
    )

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_from_hoomd_snapshot(kwargs):
    """Ensure parsing from hoomd snapshot produces the expected output."""
    simulation = make_simulation(**kwargs)

    snapshot = simulation.state.get_snapshot()
    typeids = snapshot.particles.typeid
    positions = snapshot.particles.position
    orientations = snapshot.particles.orientation

    bodies = [p4.Body(t) for t in simulation.state.particle_types]
    positions_by_type = defaultdict(list)
    orientations_by_type = defaultdict(list)
    
    for tid, p, o in zip(typeids, positions, orientations):
        t = snapshot.particles.types[tid]
        positions_by_type[t].append(p)
        orientations_by_type[t].append(o)
    
    expected_arrangement = p4.Arrangement(
        bodies=bodies,
        positions_by_type=positions_by_type,
        orientations_by_type=orientations_by_type
    )
    
    assert_arrangements_are_equal(
        expected_arrangement,
        p4.Arrangement.from_hoomd_snapshot(snapshot)
    )

REFERENCE_FOLDER = Path(__file__).parent / "data"

@pytest.mark.parametrize("kwargs,filename,index", [
    [VALID_KWARGS[0], "single-particle-arrangement-sphere.gsd", 0],
    [VALID_KWARGS[1], "multi-particle-arrangement-ellipsoid-cpolyhedron-polyhedron.gsd", 0],
    [VALID_KWARGS[0], "multi-frame-single-particle-arrangement.gsd", 1],
])
def test_from_gsd(kwargs, filename, index):
    """Ensure parsing from GSD file produces the expected output."""
    simulation = make_simulation(**kwargs)

    snapshot = simulation.state.get_snapshot()
    typeids = snapshot.particles.typeid
    positions = snapshot.particles.position
    orientations = snapshot.particles.orientation

    bodies = [p4.Body(t) for t in simulation.state.particle_types]
    positions_by_type = defaultdict(list)
    orientations_by_type = defaultdict(list)
    
    for tid, p, o in zip(typeids, positions, orientations):
        t = snapshot.particles.types[tid]
        positions_by_type[t].append(p)
        orientations_by_type[t].append(o)
    
    expected_arrangement = p4.Arrangement(
        bodies=bodies,
        positions_by_type=positions_by_type,
        orientations_by_type=orientations_by_type
    )

    assert_arrangements_are_equal(
        expected_arrangement,
        p4.Arrangement.from_gsd(REFERENCE_FOLDER / filename, index)
    )

def assert_rigids_are_equal(rigid1, rigid2):
    """Assert that two rigid constraints are equivalent."""
    r1_body = rigid1.body.to_base()
    r2_body = rigid2.body.to_base()
    assert r1_body == r2_body


@pytest.mark.parametrize("kwargs", VALID_KWARGS)
@pytest.mark.parametrize("variant", ["with_rigid", "without_rigid"])
def test_to_hoomd_rigid(kwargs, variant):
    """Ensure export to rigid produces the expected result."""
    a = p4.Arrangement(**kwargs)
    
    if variant == "with_rigid":
        ref_rigid = hoomd.md.constrain.Rigid()
        ref_rigid.body["Z"] = dict(
            constituent_types=["Y"],
            positions=[[1,1,1]],
            orientations=[[0,1,0,0]]
        )

        for b in a.bodies:
            ref_rigid = b.to_hoomd_rigid(ref_rigid)
        
        test_rigid = hoomd.md.constrain.Rigid()
        test_rigid.body["Z"] = dict(
            constituent_types=["Y"],
            positions=[[1,1,1]],
            orientations=[[0,1,0,0]]
        )

        test_rigid = a.to_hoomd_rigid(test_rigid)

    elif variant == "without_rigid":
        ref_rigid = hoomd.md.constrain.Rigid()
        for b in a.bodies:
            ref_rigid = b.to_hoomd_rigid(ref_rigid)

        test_rigid = a.to_hoomd_rigid()
    
    assert_rigids_are_equal(ref_rigid, test_rigid)

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_to_hoomd_snapshot(kwargs):
    """Ensure export to snapshot produces the expected output."""
    arrangement = p4.Arrangement(**kwargs)
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
CUBE_FACES = [
    [0, 2, 6, 4],
    [0, 4, 5, 1],
    [4, 6, 7, 5],
    [0, 1, 3, 2],
    [2, 3, 7, 6],
    [1, 5, 7, 3]
]

CONCAVE_VERTICES = np.array(
    [
        [-1/4, -1/4, -1/4],
        [-1/4,  1/4, -1/4],
        [ 1/4,  1/4, -1/4],
        [ 1/4, -1/4, -1/4],
        [-1/4,  0,  0],
        [ 1/4,  0,  0],
        [-1/4, -1/4,  1/4],
        [-1/4,  1/4,  1/4],
        [ 1/4,  1/4,  1/4],
        [ 1/4, -1/4,  1/4],
    ],
    dtype=np.float32
)
CONCAVE_FACES = [
    [0, 1, 2, 3],
    [0, 3, 5, 4],
    [4, 5, 9, 6],
    [3, 2, 8, 9, 5],
    [2, 1, 7, 8],
    [1, 0, 4, 6, 7],
    [6, 9, 8, 7],
]

@pytest.mark.parametrize("kwargs,ref_filename,type_shapes", [
    [   # 1 single-particle body, without orientations, type_shapes are a sphere
        dict(
            bodies=[p4.Body("A")],
            positions_by_type=dict(A=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        ),
        "single-particle-arrangement-sphere.gsd",
        dict(A=coxeter.shapes.Sphere(0.5))
    ],
    [   # 2 multi-particle bodies, with orientations, multiple type_shapes, some not specified
        dict(
            bodies=[
                p4.Body(
                    primary_type="A",
                    secondary_types=["B", "C"],
                    positions_by_type=dict(
                        B=[[-1,0,0]],
                        C=[[1,0,0]]
                    )
                ),
                p4.Body(
                    primary_type="D",
                    secondary_types=["E", "F"],
                    positions_by_type=dict(
                        E=[[0, -1, 0], [0, 1, 0]],
                        F=[[0, 0, -1], [0, 0, 1]]
                    ),
                    orientations_by_type=dict(
                        E=[[1,0,0,0], [0, 0.707, 0.707, 0]],
                        F=[[0.707, 0, -0.707, 0], [0.707, 0, 0.707, 0]]
                    ),
                )
            ],
            positions_by_type=dict(
                A=[[10, 0, 0]],
                D=[[0, 10, 0], [0, 20, 0]]
            ),
            orientations_by_type=dict(
                A=[[1,0,0,0]],
                D=[[0, 0.707, 0.707, 0], [0.707, 0, 0.707, 0]]
            )
        ),
        "multi-particle-arrangement-ellipsoid-cpolyhedron-polyhedron.gsd",
        dict(
            A=coxeter.shapes.Ellipsoid(a=0.25, b=4, c=6),
            E=coxeter.shapes.ConvexPolyhedron(vertices=CUBE_VERTICES),
            F=coxeter.shapes.Polyhedron(vertices=CONCAVE_VERTICES, faces=CONCAVE_FACES)
        )
    ],
])
def test_to_gsd(kwargs, ref_filename, type_shapes):
    """Ensure export to GSD file produces the expected output."""
    arrangement = p4.Arrangement(**kwargs)

    with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
        test_path = Path(tempdir) / f"test_arrangement.gsd"
        arrangement.to_gsd(test_path, type_shapes=type_shapes)

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
