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

    # 2 multi-particle bodies, with orientations, masses, and mois, with np dtypes
    dict(
        bodies=[
            p4.Body(
                primary_type="A",
                secondary_types=["B", "C"],
                positions_by_type=dict(
                    B=[[-1, 0, 0]],
                    C=[[1, 0, 0]]
                ),
                mass_by_type=dict(A=1, B=2, C=3),
                moi_by_type=dict(A=[1, 1, 1], B=[1, 0, 0], C=[0, 1, 0])
            ),
            p4.Body(
                primary_type="D",
                secondary_types=["E", "F"],
                positions_by_type=dict(
                    E=[[0, -1, 0], [0, 1, 0]],
                    F=[[0, 0, -1], [0, 0, 1]]
                ),
                orientations_by_type=dict(
                    E=[[1, 0, 0, 0], [0, 0.707, 0.707, 0]],
                    F=[[0.707, 0, -0.707, 0], [0.707, 0, 0.707, 0]]
                ),
            )
        ],
        positions_by_type=dict(
            A=[np.array([10, 0, 0])],
            D=np.array([[0, 10, 0], [0, 20, 0]])
        ),
        orientations_by_type=dict(
            A=[[1, 0, 0, 0]],
            D=[[0, np.float32(0.707), 0.707, 0], [0.707, 0, 0.707, 0]]
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

def test_sanitized_instantiation():
    # Sanitize as follows: np -> list, 
    # TODO
    pass

def make_simulation(arrangement_kwargs, filename, index=0):
    """Create a simulation with a state from a named GSD file.
    
    A rigid constraint is created from arrangement kwargs and attached to the
    simulation.
    """
    # Create rigid
    rigid = p4.Arrangement(**arrangement_kwargs).to_hoomd_rigid()
    
    # Create simulation with a state from the GSD file
    with gsd.hoomd.open(filename, "r") as f:
        frame = f[index]

    snapshot = hoomd.Snapshot.from_gsd_frame(
        gsd_snap=frame,
        communicator=hoomd.communicator.Communicator()
    )

    simulation = hoomd.Simulation(device=hoomd.device.CPU())
    simulation.create_state_from_snapshot(snapshot)

    # Ensure it runs
    simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)
    simulation.operations.integrator.rigid = rigid
    simulation.run(0)

    return simulation

def assert_bodies_equivalent(body1, body2):
    """Ensure that two bodies are essentially equivalent, accounting for floating point issues etc."""
    # primary same
    assert body1.primary_type == body2.primary_type
    
    # secondary same
    assert body1.secondary_types == body2.secondary_types
    
    # positions same
    assert (
        len(body1.positions_by_type) == len(body2.positions_by_type)
        and all(
            (
                np.round(body1.positions_by_type[t], 3)
                == np.round(body2.positions_by_type[t], 3)
            ).all()
            for t in body1.positions_by_type
        )
    )
    
    # orientations same or equivalent
    orientations_same = (
        len(body1.orientations_by_type) == len(body2.orientations_by_type)
        and all(
            (
                np.round(body1.orientations_by_type[t], 3)
                == np.round(body2.orientations_by_type[t], 3)
            ).all()
            for t in body1.orientations_by_type
        )
    )

    orientations_missing_from_self = (
        set(body2.orientations_by_type) - set(body1.orientations_by_type)
    )
    orientations_missing_from_other = (
        set(body1.orientations_by_type) - set(body2.orientations_by_type)
    )
    common_types = set(body1.orientations_by_type).intersection(
        set(body2.orientations_by_type)
    )
    orientations_equivalent = (
        all(
            (
                np.round(body1.orientations_by_type[t], 3)
                == np.round(body2.orientations_by_type[t], 3)
            ).all()
            for t in common_types
        )
        and
        all(
            all(list(i) == [1,0,0,0] for i in body2.orientations_by_type[t])
            for t in orientations_missing_from_self
        )
        and
        all(
            all(list(i) == [1,0,0,0] for i in body1.orientations_by_type[t])
            for t in orientations_missing_from_other
        )
    )

    # masses same or equivalent
    masses_same = (
        body1.mass_by_type == body2.mass_by_type
    )
    masses_missing_from_self = (
        set(body2.mass_by_type) - set(body1.mass_by_type)
    )
    masses_missing_from_other = (
        set(body1.mass_by_type) - set(body2.mass_by_type)
    )
    common_types = set(body1.mass_by_type).intersection(
        set(body2.mass_by_type)
    )
    masses_equivalent = (
        all(
            body1.mass_by_type[t] == body2.mass_by_type[t]
            for t in common_types
        )
        and all(
            body2.mass_by_type[t] == 1.0 for t in masses_missing_from_self
        )
        and all(
            body1.mass_by_type[t] == 1.0 for t in masses_missing_from_other
        )
    )

    # moi same or equivalent
    moi_same = (
        body1.moi_by_type == body2.moi_by_type
    )
    moi_missing_from_self = (
        set(body2.moi_by_type) - set(body1.moi_by_type)
    )
    moi_missing_from_other = (
        set(body1.moi_by_type) - set(body2.moi_by_type)
    )
    common_types = set(body1.moi_by_type).intersection(
        set(body2.moi_by_type)
    )
    moi_equivalent = (
        all(
            (
                np.round(body1.moi_by_type[t], 3)
                == np.round(body2.moi_by_type[t], 3)
            ).all()
            for t in common_types
        )
        and
        all(
            body2.moi_by_type[t] == [1, 1, 1]
            for t in moi_missing_from_self
        )
        and all(
            body1.moi_by_type[t] == [1, 1, 1]
            for t in moi_missing_from_other
        )
    )

    assert orientations_same or orientations_equivalent    
    assert masses_same or masses_equivalent
    assert moi_same or moi_equivalent

def assert_arrangements_are_equal(arrangement1, arrangement2):
    """Arrangements are equivalent if their bodies, positions, and orientations are equivalent."""
    assert len(arrangement1.bodies) == len(arrangement2.bodies)
    for b1 in arrangement1.bodies:
        b2 = [b for b in arrangement2.bodies if b.primary_type == b1.primary_type][0]
        assert_bodies_equivalent(b1, b2)
    assert (
        arrangement1.positions_by_type.keys() == arrangement2.positions_by_type.keys()
        and all(
            np.isclose(arrangement1.positions_by_type[t], arrangement2.positions_by_type[t]).all()
            for t in arrangement1.positions_by_type
        )
    )
    assert (
        all(
            np.isclose(
                arrangement1.orientations_by_type.get(t, [1, 0, 0, 0]),
                arrangement2.orientations_by_type.get(t, [1, 0, 0, 0])
            ).all()
            for t in set(arrangement1.orientations_by_type.keys()).union(arrangement2.orientations_by_type.keys())
        )
        or (
            arrangement1.orientations_by_type == {}
            and all(
                list(i) == [1, 0, 0, 0]
                for v in arrangement2.orientations_by_type.values()
                for i in v
            )
        )
        or (
            arrangement2.orientations_by_type == {}
            and all(
                list(i) == [1, 0, 0, 0]
                for v in arrangement1.orientations_by_type.values()
                for i in v
            )
        )
    )

REFERENCE_FOLDER = Path(__file__).parent / "data"

@pytest.mark.parametrize("kwargs,ref_filename", [
    [VALID_KWARGS[0], "single-particle-arrangement-sphere.gsd"],
    [VALID_KWARGS[1], "multi-particle-arrangement-ellipsoid-cpolyhedron-polyhedron.gsd"]
])
def test_from_hoomd_simulation(kwargs, ref_filename):
    """Ensure parsing from hoomd simulation produces the expected output."""
    arrangement = p4.Arrangement(**kwargs)
    simulation = make_simulation(kwargs, REFERENCE_FOLDER / ref_filename)
    include_singles = any(b.secondary_types == [] for b in kwargs["bodies"])
    assert_arrangements_are_equal(
        arrangement,
        p4.Arrangement.from_hoomd_simulation(simulation, include_singles)
    )

@pytest.mark.parametrize("kwargs,ref_filename", [
    [VALID_KWARGS[0], "single-particle-arrangement-sphere.gsd"],
    [VALID_KWARGS[1], "multi-particle-arrangement-ellipsoid-cpolyhedron-polyhedron.gsd"]
])
def test_from_hoomd_snapshot(kwargs, ref_filename):
    """Ensure parsing from hoomd snapshot produces the expected output."""
    simulation = make_simulation(kwargs, REFERENCE_FOLDER / ref_filename)

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

ARRANGEMENT_KWARGS_FOR_GSD_1 = VALID_KWARGS[0]
ARRANGEMENT_KWARGS_FOR_GSD_2 = VALID_KWARGS[1]
ARRANGEMENT_KWARGS_FOR_GSD_3 = VALID_KWARGS[0]

GSD_FILENAME_1 = "single-particle-arrangement-sphere.gsd"
GSD_FILENAME_2 = "multi-particle-arrangement-ellipsoid-cpolyhedron-polyhedron.gsd"
GSD_FILENAME_3 = "multi-frame-single-particle-arrangement.gsd"

@pytest.mark.parametrize("kwargs,ref_filename,index", [
    [ARRANGEMENT_KWARGS_FOR_GSD_1, GSD_FILENAME_1, 0],
    [ARRANGEMENT_KWARGS_FOR_GSD_2, GSD_FILENAME_2, 0],
    [ARRANGEMENT_KWARGS_FOR_GSD_3, GSD_FILENAME_3, 1],
])
def test_from_gsd(kwargs, ref_filename, index):
    """Ensure parsing from GSD file produces the expected output."""
    simulation = make_simulation(kwargs, REFERENCE_FOLDER / ref_filename, index)

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
        p4.Arrangement.from_gsd(REFERENCE_FOLDER / ref_filename, index)
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

@pytest.mark.parametrize("kwargs,ref_filename", [
    [VALID_KWARGS[0], "single-particle-arrangement-sphere.gsd"],
    [VALID_KWARGS[1], "multi-particle-arrangement-ellipsoid-cpolyhedron-polyhedron.gsd"]
])
def test_to_hoomd_snapshot(kwargs, ref_filename):
    """Ensure export to snapshot produces the expected output."""
    arrangement = p4.Arrangement(**kwargs)
    simulation = make_simulation(kwargs, REFERENCE_FOLDER / ref_filename)
    ref_snap = simulation.state.get_snapshot()
    test_snap = arrangement.to_hoomd_snapshot()

    ref_typeids = ref_snap.particles.typeid.tolist()
    ref_positions = np.round(ref_snap.particles.position, 3).tolist()       # rounded because run(0) warps the exact values
    ref_orientations = np.round(ref_snap.particles.orientation, 3).tolist()
    ref_masses = ref_snap.particles.mass.tolist()
    ref_mois = ref_snap.particles.moment_inertia.tolist()
    ref_bodyids = ref_snap.particles.body.tolist()
    
    test_typeids = test_snap.particles.typeid.tolist()
    test_positions = np.round(test_snap.particles.position, 3).tolist()
    test_orientations = np.round(test_snap.particles.orientation, 3).tolist()
    test_masses = test_snap.particles.mass.tolist()
    test_mois = test_snap.particles.moment_inertia.tolist()
    test_bodyids = test_snap.particles.body.tolist()

    ref_data = list(zip(ref_typeids, ref_positions, ref_orientations, ref_masses, ref_mois, ref_bodyids))
    test_data = list(zip(test_typeids, test_positions, test_orientations, test_masses, test_mois, test_bodyids))
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

TYPE_SHAPES_FOR_GSD_1 = dict(A=coxeter.shapes.Sphere(0.5))
TYPE_SHAPES_FOR_GSD_2 = dict(
    A=coxeter.shapes.Ellipsoid(a=0.25, b=4, c=6),
    E=coxeter.shapes.ConvexPolyhedron(vertices=CUBE_VERTICES),
    F=coxeter.shapes.Polyhedron(vertices=CONCAVE_VERTICES, faces=CONCAVE_FACES)
)

@pytest.mark.parametrize("kwargs,ref_filename,type_shapes", [
    [ARRANGEMENT_KWARGS_FOR_GSD_1, GSD_FILENAME_1, TYPE_SHAPES_FOR_GSD_1],
    [ARRANGEMENT_KWARGS_FOR_GSD_2, GSD_FILENAME_2, TYPE_SHAPES_FOR_GSD_2],
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
        assert np.array_equal(test_frame.particles.mass, ref_frame.particles.mass)
        assert np.array_equal(test_frame.particles.moment_inertia, ref_frame.particles.moment_inertia)
        assert np.array_equal(test_frame.particles.body, ref_frame.particles.body)


if __name__ == "__main__":
    # Regenerate reference files
    arrangement = p4.Arrangement(**ARRANGEMENT_KWARGS_FOR_GSD_1)
    file_path = REFERENCE_FOLDER / GSD_FILENAME_1
    arrangement.to_gsd(file_path, type_shapes=TYPE_SHAPES_FOR_GSD_1)

    arrangement = p4.Arrangement(**ARRANGEMENT_KWARGS_FOR_GSD_2)
    file_path = REFERENCE_FOLDER / GSD_FILENAME_2
    arrangement.to_gsd(file_path, type_shapes=TYPE_SHAPES_FOR_GSD_2)

    with gsd.hoomd.open(REFERENCE_FOLDER / GSD_FILENAME_1, "r") as f1:
        frame = f1[0]
    with gsd.hoomd.open(REFERENCE_FOLDER / GSD_FILENAME_3, "w") as f2:
        f2.append(gsd.hoomd.Frame())
        f2.append(frame)
