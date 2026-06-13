# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

import gsd
import hoomd
import numpy as np
import pytest
import p4

def get_cube_vertices(side_length):
    s = side_length
    return [
        [-s/2, -s/2, -s/2],
        [-s/2, -s/2,  s/2],
        [-s/2,  s/2, -s/2],
        [-s/2,  s/2,  s/2],
        [ s/2, -s/2, -s/2],
        [ s/2, -s/2,  s/2],
        [ s/2,  s/2, -s/2],
        [ s/2,  s/2,  s/2]
    ]

def get_cube_faces():
    return [
        [0, 2, 6, 4],
        [0, 4, 5, 1],
        [4, 6, 7, 5],
        [0, 1, 3, 2],
        [2, 3, 7, 6],
        [1, 5, 7, 3],
    ]


def lj_interaction(interacting_types):
    """Return a simple LJ Interaction with the specified interacting types."""
    return p4.Interaction(
        hoomd_class=hoomd.md.pair.LJ,
        initial_args={},
        default_params=dict(
            r_cut=np.float32(0),
            params=dict(epsilon=0, sigma=1)
        ),
        typed_params={
            p: dict(r_cut=5, params=dict(epsilon=1, sigma=np.float64(1)))
            for p in interacting_types
        }
    )

def alj_interaction(interacting_pairs, interacting_singles):
    """Return a simple ALJ Interaction with the specified interacting types."""
    kwargs = dict(
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args={},
        default_params=dict(
            r_cut=0,
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0),
            shape=dict(vertices=[], faces=[])
        ),
        typed_params={
            p: dict(
                r_cut=5,
                params=dict(epsilon=1, sigma_i=0.1, sigma_j=0.1, alpha=0),
            )
            for p in interacting_pairs
        }
    )

    for t in interacting_singles:
        kwargs["typed_params"][t] = dict(
            shape=dict(vertices=get_cube_vertices(1), faces=get_cube_faces())
        )

    return p4.Interaction(**kwargs)

VALID_KWARGS = [
    # single-particle probe, single-particle analyte
    dict(   # one interaction
        probe=p4.Body(primary_type="A"),
        analyte=p4.Body(primary_type="A"),
        interactions=[lj_interaction([("A", "A")])]
    ),
    dict(   # two interactions (no extra types)
        probe=p4.Body(primary_type="A"),
        analyte=p4.Body(primary_type="B"),
        interactions=[
            lj_interaction([("A", "B")]),
            lj_interaction([("A", "A")])
        ]
    ),
    dict(   # two interactions (extra types)
        probe=p4.Body(primary_type="A"),
        analyte=p4.Body(primary_type="B"),
        interactions=[
            lj_interaction([("A", "B")]),
            lj_interaction([("A", "A"), ("C", "C")])
        ]
    ),

    # single-particle probe, multi-particle analyte
    dict(   # one interaction
        probe=p4.Body(primary_type="A"),
        analyte=p4.Body(
            primary_type="B",
            secondary_types=["C"],
            positions_by_type={"C": [[1,0,0]]},
            orientations_by_type={"C": [[0,1,0,0]]}
        ),
        interactions=[
            lj_interaction([("A", "C")])
        ]
    ),
    dict(   # two interactions (no extra types)
        probe=p4.Body(primary_type="A"),
        analyte=p4.Body(
            primary_type="B",
            secondary_types=["C"],
            positions_by_type={"C": [[1,0,0]]},
            orientations_by_type={"C": [[0,1,0,0]]}
        ),
        interactions=[
            lj_interaction([("A", "C")]),
            lj_interaction([("A", "B")])
        ]
    ),
    dict(   # two interactions (extra types)
        probe=p4.Body(primary_type="A"),
        analyte=p4.Body(
            primary_type="B",
            secondary_types=["C"],
            positions_by_type={"C": [[1,0,0]]},
            orientations_by_type={"C": [[0,1,0,0]]}
        ),
        interactions=[
            lj_interaction([("A", "C")]),
            lj_interaction([("A", "B")])
        ]
    ),

    # multi-particle probe, single-particle analyte
    dict(   # one interaction
        probe=p4.Body(
            primary_type="B",
            secondary_types=["C"],
            positions_by_type={"C": [[1,0,0]]},
            orientations_by_type={"C": [[0,1,0,0]]}
        ),
        analyte=p4.Body(primary_type="A"),
        interactions=[
            lj_interaction([("A", "C")])
        ]
    ),
    dict(   # two interactions (no extra types)
        probe=p4.Body(
            primary_type="B",
            secondary_types=["C"],
            positions_by_type={"C": [[1,0,0]]},
            orientations_by_type={"C": [[0,1,0,0]]}
        ),
        analyte=p4.Body(primary_type="A"),
        interactions=[
            lj_interaction([("A", "C")]),
            lj_interaction([("A", "B")])
        ]
    ),
    dict(   # two interactions (extra types)
        probe=p4.Body(
            primary_type="B",
            secondary_types=["C"],
            positions_by_type={"C": [[1,0,0]]},
            orientations_by_type={"C": [[0,1,0,0]]}
        ),
        analyte=p4.Body(primary_type="A"),
        interactions=[
            lj_interaction([("A", "C")]),
            lj_interaction([("A", "B")])
        ]
    ),

    # multi-particle probe, multi-particle analyte
    dict(   # one interaction
        probe=p4.Body(
            primary_type="B",
            secondary_types=["C"],
            positions_by_type={"C": [[1,0,0]]},
            orientations_by_type={"C": [[0,1,0,0]]}
        ),
        analyte=p4.Body(
            primary_type="B",
            secondary_types=["C"],
            positions_by_type={"C": [[1,0,0]]},
            orientations_by_type={"C": [[0,1,0,0]]}
        ),
        interactions=[
            lj_interaction([("C", "C")])
        ]
    ),
    dict(   # two interactions (no extra types)
        probe=p4.Body(
            primary_type="A",
            secondary_types=["B"],
            positions_by_type={"B": [[0,1,0]]},
        ),
        analyte=p4.Body(
            primary_type="C",
            secondary_types=["D"],
            positions_by_type={"D": [[1,0,0]]},
        ),
        interactions=[
            lj_interaction([("B", "D")]),
            lj_interaction([("A", "C")])
        ]
    ),
    dict(   # two interactions (extra types)
        probe=p4.Body(
            primary_type="A",
            secondary_types=["B"],
            positions_by_type={"B": [[0,1,0]]},
        ),
        analyte=p4.Body(
            primary_type="C",
            secondary_types=["D"],
            positions_by_type={"D": [[1,0,0]]},
        ),
        interactions=[
            lj_interaction([("B", "D")]),
            lj_interaction([("A", "C"), ("E", "F")])
        ]
    ),

    # Arrangement analyte
    dict(
        probe=p4.Body("A"),
        analyte=p4.Arrangement([p4.Body("B", ["C"], {"C": [[1,0,0]]})], dict(B=[[0, 0, 0]])),
        interactions=[lj_interaction([("A", "B")])]
    ),
]

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_instantiation_valid(kwargs):
    """Ensure instantiation works for valid kwargs."""
    _ = p4.System(**kwargs)

INVALID_KWARGS = [
    # Wrong types
    dict(   # wrong probe type
        probe="wrong",
        analyte=p4.Body("A"),
        interactions=[lj_interaction([("A", "A")])]
    ),
    dict(   # wrong analyte type
        probe=p4.Body("A"),
        analyte="wrong",
        interactions=[lj_interaction([("A", "A")])]
    ),
    dict(   # wrong interactions type
        probe=p4.Body("A"),
        analyte=p4.Body("A"),
        interactions="wrong"
    ),
    dict(   # wrong interactions item type
        probe=p4.Body("A"),
        analyte=p4.Body("A"),
        interactions=["wrong"]
    ),

    # Clashing bodies
    dict(   # primary-primary clash (analyte is a body)
        probe=p4.Body("A"),
        analyte=p4.Body(
            primary_type="A",
            secondary_types=["B"],
            positions_by_type={"B": [[1,0,0]]}
        ),
        interactions=[lj_interaction([("A", "B")])]
    ),
    dict(   # primary-primary clash (analyte is an arrangement)
        probe=p4.Body("A"),
        analyte=p4.Arrangement([p4.Body("A", ["B"], {"B": [[1,0,0]]})], dict(A=[[0, 0, 0]])),
        interactions=[lj_interaction([("A", "B")])]
    ),
    dict(   # probe primary-analyte secondary clash (analyte is a body)
        probe=p4.Body("A"),
        analyte=p4.Body(
            primary_type="B",
            secondary_types=["A"],
            positions_by_type={"A": [[1,0,0]]}
        ),
        interactions=[lj_interaction([("A", "B")])]
    ),
    dict(   # probe primary-analyte secondary clash (analyte is an arrangement)
        probe=p4.Body("A"),
        analyte=p4.Arrangement([p4.Body("B", ["A"], {"A": [[1,0,0]]})], dict(B=[[0, 0, 0]])),
        interactions=[lj_interaction([("A", "B")])]
    ),
    dict(   # analyte primary-probe secondary clash (analyte is a body)
        probe=p4.Body(
            primary_type="C",
            secondary_types=["A"],
            positions_by_type={"A": [[1,0,0]]}
        ),
        analyte=p4.Body(
            primary_type="A",
            secondary_types=["B"],
            positions_by_type={"B": [[1,0,0]]}
        ),
        interactions=[lj_interaction([("A", "B")])]
    ),
    dict(   # analyte primary-probe secondary clash (analyte is an arrangement)
        probe=p4.Body(
            primary_type="C",
            secondary_types=["B"],
            positions_by_type={"B": [[1,0,0]]}
        ),
        analyte=p4.Arrangement([p4.Body("B", ["A"], {"A": [[1,0,0]]})], dict(B=[[0, 0, 0]])),
        interactions=[lj_interaction([("A", "B")])]
    ),
]

@pytest.mark.parametrize("kwargs", INVALID_KWARGS)
def test_instantiation_invalid(kwargs):
    """Ensure instantiation fails expectedly for invalid kwargs."""
    with pytest.raises((ValueError, TypeError)):
        _ = p4.System(**kwargs)

def test_property_getters_and_setters_valid():
    """Ensure property getters and setters work as expected with valid inputs."""
    body1 = p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type={"B": [[0,1,0]]},
    )
    body2 = p4.Body(
        primary_type="C",
        secondary_types=["D"],
        positions_by_type={"D": [[1,0,0]]},
    )
    kwargs = dict(
        probe=body1,
        analyte=body2,
        interactions=[
            lj_interaction([("B", "D")]),
            lj_interaction([("A", "C")])
        ]
    )

    system = p4.System(**kwargs)

    # Getters
    assert system.probe == kwargs["probe"]
    assert system.analyte == kwargs["analyte"]
    assert system.interactions == kwargs["interactions"]

    # Setters
    system.probe = p4.Body("X")
    assert system.probe == p4.Body("X")
    system.probe = body2
    assert system.probe == body2
    system.analyte = p4.Arrangement([p4.Body("Y"), p4.Body("Z")], dict(Y=[[0,0,0]], Z=[[1,1,1]]))
    assert system.analyte == p4.Arrangement([p4.Body("Y"), p4.Body("Z")], dict(Y=[[0,0,0]], Z=[[1,1,1]]))
    system.interactions = []
    assert system.interactions == []

def test_property_setters_invalid():
    """Ensure property setters fail expectedly with invalid inputs."""
    body1 = p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type={"B": [[0,1,0]]},
    )
    body2 = p4.Body(
        primary_type="C",
        secondary_types=["D"],
        positions_by_type={"D": [[1,0,0]]},
    )
    kwargs = dict(
        probe=body1,
        analyte=body2,
        interactions=[
            lj_interaction([("B", "D")]),
            lj_interaction([("A", "C")])
        ]
    )

    system = p4.System(**kwargs)

    # Setting probe with wrong type
    with pytest.raises(TypeError):
        system.probe = p4.Arrangement([p4.Body("Y"), p4.Body("Z")], dict(Y=[[0,0,0]], Z=[[1,1,1]]))
        
    # Setting analyte with wrong type
    with pytest.raises(TypeError):
        system.analyte = None

    # Setting probe with type clash (analyte is a body)
    with pytest.raises(ValueError):
        system.probe = p4.Body("C")
    with pytest.raises(ValueError):
        system.probe = p4.Body("A", ["C"], dict(A=[[1,0,0]]))
    
    # Setting probe with type clash (analyte is an arrangement)
    system.analyte = p4.Arrangement([p4.Body("C", ["D"], dict(D=[[0,0,1]]))], dict(C=[[1,0,0]]))
    with pytest.raises(ValueError):
        system.probe = p4.Body("C")
    with pytest.raises(ValueError):
        system.probe = p4.Body("A", ["C"], dict(C=[[1,0,0]]))
    system.analyte = body2

    # Setting analyte with type clash
    with pytest.raises(ValueError):
        system.analyte = p4.Body("A")

    # Ensure none of the invalid operations above mutated the internal data
    assert system.probe == kwargs["probe"]
    assert system.analyte == kwargs["analyte"]
    assert system.interactions == kwargs["interactions"]

@pytest.mark.parametrize("kwargs,expected", [
    # one active
    [   # common single types (probe primary, analyte primary)
        dict(
            probe=p4.Body(
                primary_type="A",
                secondary_types=["B"],
                positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=p4.Body(
                primary_type="C",
                secondary_types=["D"],
                positions_by_type={"D": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction([], ["A", "C"])
            ]
        ),
        [alj_interaction([], ["A", "C"])]
    ],
    [   # common single types (probe primary, analyte secondary)
        dict(
            probe=p4.Body(
                primary_type="A",
                secondary_types=["B"],
                positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=p4.Body(
                primary_type="C",
                secondary_types=["D"],
                positions_by_type={"D": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction([], ["A", "D"])
            ]
        ),
        [alj_interaction([], ["A", "D"])]
    ],
    [   # common single types (probe secondary, analyte primary)
        dict(
            probe=p4.Body(
                primary_type="A",
                secondary_types=["B"],
                positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=p4.Body(
                primary_type="C",
                secondary_types=["D"],
                positions_by_type={"D": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction([], ["B", "C"])
            ]
        ),
        [alj_interaction([], ["B", "C"])]
    ],
    [   # common single types (probe secondary, analyte secondary)
        dict(
            probe=p4.Body(
                primary_type="A",
                secondary_types=["B"],
                positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=p4.Body(
                primary_type="C",
                secondary_types=["D"],
                positions_by_type={"D": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction([], ["B", "D"])
            ]
        ),
        [alj_interaction([], ["B", "D"])]
    ],
    [   # common pair types (one member of pair, same for both probe and analyte)
        dict(
            probe=p4.Body(
                primary_type="A",
                secondary_types=["B"],
                positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=p4.Body(
                primary_type="C",
                secondary_types=["B"],
                positions_by_type={"B": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction([("B", "E")], [])
            ]
        ),
        [alj_interaction([("B", "E")], [])]
    ],
    [   # common pair types (different members for probe and analyte)
        dict(
            probe=p4.Body(
                primary_type="A",
                secondary_types=["B"],
                positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=p4.Body(
                primary_type="C",
                secondary_types=["D"],
                positions_by_type={"D": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction([("A", "C")], [])
            ]
        ),
        [alj_interaction([("A", "C")], [])]
    ],
    # multiple active
    [
        dict(
            probe=p4.Body(
                primary_type="A",
                secondary_types=["B"],
                positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=p4.Body(
                primary_type="C",
                secondary_types=["D"],
                positions_by_type={"D": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction([("A", "C")], ["A", "C"]),
                lj_interaction([("B", "D")])
            ]
        ),
        [
            alj_interaction([("A", "C")], ["A", "C"]),
            lj_interaction([("B", "D")])
        ]
    ],
])
def test_active_interactions(kwargs, expected):
    """Ensure active_interactions method returns expected results."""
    system = p4.System(**kwargs)
    assert system.active_interactions == expected

@pytest.mark.parametrize("kwargs,expected", [
    # single-particle probe and analyte 
    [   # one type
        dict(
            probe=p4.Body(primary_type="A"),
            analyte=p4.Body(primary_type="A"),
            interactions=[lj_interaction([("A", "A")])]
        ),
        ["A"]
    ],
    [   # multiple types, one interaction
        dict(
            probe=p4.Body(primary_type="A"),
            analyte=p4.Body(primary_type="B"),
            interactions=[lj_interaction([("A", "B")])]
        ),
        ["A", "B"]
    ],
    [   # multiple types, multiple interactions
        dict(
            probe=p4.Body(primary_type="A"),
            analyte=p4.Body(primary_type="B"),
            interactions=[
                lj_interaction([("A", "B")]),
                lj_interaction([("A", "B")])
            ]
        ),
        ["A", "B"]
    ],
    # single-particle probe and multi-particle analyte 
    [
        dict(
            probe=p4.Body(primary_type="A"),
            analyte=p4.Body(
                primary_type="B",
                secondary_types=["C"],
                positions_by_type={"C": [[1,0,0]]}
            ),
            interactions=[
                lj_interaction([("A", "C")]),
                lj_interaction([("A", "B")])
            ]
        ),
        ["A", "B", "C"]
    ],
    # multi-particle probe and single-particle analyte 
    [
        dict(
            probe=p4.Body(
                primary_type="B",
                secondary_types=["C"],
                positions_by_type={"C": [[1,0,0]]}
            ),
            analyte=p4.Body(primary_type="A"),
            interactions=[
                lj_interaction([("A", "C")]),
                lj_interaction([("A", "B")])
            ]
        ),
        ["A", "B", "C"]
    ],
    # multi-particle probe and multi-particle analyte 
    [
        dict(
            probe=p4.Body(
                primary_type="A",
                secondary_types=["B"],
                positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=p4.Body(
                primary_type="C",
                secondary_types=["D"],
                positions_by_type={"D": [[0,1,0]]}
            ),
            interactions=[
                lj_interaction([("B", "D")]),
                lj_interaction([("A", "C")])
            ]
        ),
        ["A", "B", "C", "D"]
    ]
])
def test_all_types(kwargs, expected):
    """Ensure all_types method returns expected results."""
    system = p4.System(**kwargs)
    assert len(system.all_types) == len(expected)
    assert set(system.all_types) == set(expected)

def make_example_snapshot(body1: p4.Body, body2: p4.Body):
    """Return a small example snapshot with two bodies."""
    return p4.util.simulation.get_initial_frame(
        probe=body1, analyte=body2, simulation_box=[100, 100, 100, 0, 0, 0]
    )

def get_valid_simulations_and_kwargs():
    """Return an array of valid simulations with corresponding kwargs.
    
    The kwargs assume that when parsing the simulation, the primary type for
    the probe will always be 'A', and for the analyte the primary type will
    always be 'C'.
    """
    simulations_and_kwargs = []

    # single-particle probe, single-particle analyte
    snapshot = make_example_snapshot(p4.Body("A"), p4.Body("C"))

    simulation = hoomd.Simulation(device=hoomd.device.CPU())
    simulation.create_state_from_snapshot(snapshot)
    simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)

    lj = hoomd.md.pair.LJ(hoomd.md.nlist.Tree(2))
    lj.r_cut[("A", "A")] = 0
    lj.r_cut[("C", "C")] = 0
    lj.r_cut[("A", "C")] = 5
    lj.params[("A", "A")] = dict(epsilon=0, sigma=1)
    lj.params[("C", "C")] = dict(epsilon=0, sigma=1)
    lj.params[("A", "C")] = dict(epsilon=1, sigma=1)
    simulation.operations.integrator.forces.append(lj)

    kwargs = dict(
        probe=p4.Body("A"),
        analyte=p4.Body("C"),
        interactions=[lj_interaction([("A", "C")])]
    )

    simulations_and_kwargs.append([simulation, kwargs])

    # single-particle probe, multi-particle analyte
    snapshot = make_example_snapshot(
        p4.Body("A"),
        p4.Body("C", secondary_types=["D"], positions_by_type=dict(D=[[1,0,0]])),
    )

    simulation = hoomd.Simulation(device=hoomd.device.CPU())
    simulation.create_state_from_snapshot(snapshot)
    simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)

    lj = hoomd.md.pair.LJ(hoomd.md.nlist.Tree(2))
    lj.r_cut[("A", "A")] = 0
    lj.r_cut[("A", "C")] = 0
    lj.r_cut[("C", "C")] = 0
    lj.r_cut[("C", "D")] = 0
    lj.r_cut[("D", "D")] = 0
    lj.r_cut[("A", "D")] = 5
    lj.params[("A", "A")] = dict(epsilon=0, sigma=1)
    lj.params[("A", "C")] = dict(epsilon=0, sigma=1)
    lj.params[("C", "C")] = dict(epsilon=0, sigma=1)
    lj.params[("C", "D")] = dict(epsilon=0, sigma=1)
    lj.params[("D", "D")] = dict(epsilon=0, sigma=1)
    lj.params[("A", "D")] = dict(epsilon=1, sigma=1)
    simulation.operations.integrator.forces.append(lj)

    rigid = hoomd.md.constrain.Rigid()
    rigid.body["C"] = {
        "constituent_types": ["D"],
        "positions": [(1,0,0)],
        "orientations": [(1, 0, 0, 0)],
    }

    rigid.create_bodies(simulation.state)
    simulation.operations.integrator.rigid = rigid

    snapshot = simulation.state.get_snapshot()              # TODO: return here - from_hoomd_simulation is getting wrong moi values
    for i, tid in enumerate(snapshot.particles.typeid):
        if snapshot.particles.types[tid] == "D":
            snapshot.particles.mass[i] = 0
    simulation.state.set_snapshot(snapshot)

    kwargs = dict(
        probe=p4.Body("A"),
        analyte=p4.Body(
            primary_type="C",
            secondary_types=["D"],
            positions_by_type={"D": [[1,0,0]]}
        ),
        interactions=[lj_interaction([("A", "D")])]
    )

    simulations_and_kwargs.append([simulation, kwargs])

    # multi-particle probe, single-particle analyte
    snapshot = make_example_snapshot(
        p4.Body("A", secondary_types=["B"], positions_by_type=dict(B=[[1,0,0]])),
        p4.Body("C"),
    )

    simulation = hoomd.Simulation(device=hoomd.device.CPU())
    simulation.create_state_from_snapshot(snapshot)
    simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)

    lj = hoomd.md.pair.LJ(hoomd.md.nlist.Tree(2))
    lj.r_cut[("C", "C")] = 0
    lj.r_cut[("C", "A")] = 0
    lj.r_cut[("A", "A")] = 0
    lj.r_cut[("A", "B")] = 0
    lj.r_cut[("B", "B")] = 0
    lj.r_cut[("C", "B")] = 5
    lj.params[("C", "C")] = dict(epsilon=0, sigma=1)
    lj.params[("C", "A")] = dict(epsilon=0, sigma=1)
    lj.params[("A", "A")] = dict(epsilon=0, sigma=1)
    lj.params[("A", "B")] = dict(epsilon=0, sigma=1)
    lj.params[("B", "B")] = dict(epsilon=0, sigma=1)
    lj.params[("C", "B")] = dict(epsilon=1, sigma=1)
    simulation.operations.integrator.forces.append(lj)

    rigid = hoomd.md.constrain.Rigid()
    rigid.body["A"] = {
        "constituent_types": ["B"],
        "positions": [(1,0,0)],
        "orientations": [(1, 0, 0, 0)],
    }

    rigid.create_bodies(simulation.state)
    simulation.operations.integrator.rigid = rigid

    snapshot = simulation.state.get_snapshot()
    for i, tid in enumerate(snapshot.particles.typeid):
        if snapshot.particles.types[tid] == "B":
            snapshot.particles.mass[i] = 0
    simulation.state.set_snapshot(snapshot)

    kwargs = dict(
        probe=p4.Body(
            primary_type="A",
            secondary_types=["B"],
            positions_by_type={"B": [[1,0,0]]}
        ),
        analyte=p4.Body("C"),
        interactions=[lj_interaction([("B", "C")])]
    )

    simulations_and_kwargs.append([simulation, kwargs])

    # multi-particle probe, multi-particle analyte
    snapshot = make_example_snapshot(
        p4.Body("A", secondary_types=["B"], positions_by_type=dict(B=[[1,0,0]])),
        p4.Body("C", secondary_types=["D"], positions_by_type=dict(D=[[1,0,0]])),
    )

    simulation = hoomd.Simulation(device=hoomd.device.CPU())
    simulation.create_state_from_snapshot(snapshot)
    simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)

    lj = hoomd.md.pair.LJ(hoomd.md.nlist.Tree(2))
    lj.r_cut[("A", "A")] = 0
    lj.r_cut[("A", "B")] = 0
    lj.r_cut[("A", "C")] = 0
    lj.r_cut[("A", "D")] = 0
    lj.r_cut[("B", "B")] = 0
    lj.r_cut[("B", "C")] = 0
    lj.r_cut[("B", "D")] = 5
    lj.r_cut[("C", "C")] = 0
    lj.r_cut[("C", "D")] = 0
    lj.r_cut[("D", "D")] = 0
    lj.params[("A", "A")] = dict(epsilon=0, sigma=1)
    lj.params[("A", "B")] = dict(epsilon=0, sigma=1)
    lj.params[("A", "C")] = dict(epsilon=0, sigma=1)
    lj.params[("A", "D")] = dict(epsilon=0, sigma=1)
    lj.params[("B", "B")] = dict(epsilon=0, sigma=1)
    lj.params[("B", "C")] = dict(epsilon=0, sigma=1)
    lj.params[("B", "D")] = dict(epsilon=1, sigma=1)
    lj.params[("C", "C")] = dict(epsilon=0, sigma=1)
    lj.params[("C", "D")] = dict(epsilon=0, sigma=1)
    lj.params[("D", "D")] = dict(epsilon=0, sigma=1)
    simulation.operations.integrator.forces.append(lj)

    rigid = hoomd.md.constrain.Rigid()
    rigid.body["A"] = {
        "constituent_types": ["B"],
        "positions": [(1,0,0)],
        "orientations": [(1, 0, 0, 0)],
    }
    rigid.body["C"] = {
        "constituent_types": ["D"],
        "positions": [(0,1,0)],
        "orientations": [(1, 0, 0, 0)],
    }

    rigid.create_bodies(simulation.state)
    simulation.operations.integrator.rigid = rigid

    snapshot = simulation.state.get_snapshot()
    for i, tid in enumerate(snapshot.particles.typeid):
        if snapshot.particles.types[tid] in ("B", "D"):
            snapshot.particles.mass[i] = 0
    simulation.state.set_snapshot(snapshot)

    kwargs = dict(
        probe=p4.Body(
            primary_type="A",
            secondary_types=["B"],
            positions_by_type={"B": [[1,0,0]]}
        ),
        analyte=p4.Body(
            primary_type="C",
            secondary_types=["D"],
            positions_by_type={"D": [[0,1,0]]}
        ),
        interactions=[lj_interaction([("B", "D")])]
    )

    simulations_and_kwargs.append([simulation, kwargs])

    return simulations_and_kwargs

@pytest.mark.parametrize("simulation,kwargs", get_valid_simulations_and_kwargs())
def test_from_hoomd_simulation_valid(simulation, kwargs):
    """Ensure parsing from hoomd simulations works for valid simulations."""
    ref_system = p4.System(**kwargs)
    test_system = p4.System.from_hoomd_simulation(simulation, "A", "C")
    assert test_system == ref_system

@pytest.mark.parametrize("variant", ["no-forces", "no-integrator", "missing-probe-type", "missing-analyte-type"])
def test_from_hoomd_simulation_invalid(variant):
    """Ensure parsing from hoomd simulations fails expectedly for invalid simulations."""
    if variant == "no-forces":
        simulation = hoomd.util.make_example_simulation(particle_types=["A", "B"])
        simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)
        with pytest.raises(ValueError):
            _ = p4.System.from_hoomd_simulation(simulation, "A", "B")

    elif variant == "no-integrator":
        simulation = hoomd.util.make_example_simulation(particle_types=["A", "B"])
        with pytest.raises(ValueError):
            _ = p4.System.from_hoomd_simulation(simulation, "A", "B")

    elif variant == "missing-probe-type":
        simulation = hoomd.util.make_example_simulation(particle_types=["X", "B"])
        with pytest.raises(ValueError):
            _ = p4.System.from_hoomd_simulation(simulation, "A", "B")

    elif variant == "missing-analyte-type":
        simulation = hoomd.util.make_example_simulation(particle_types=["A", "X"])
        with pytest.raises(ValueError):
            _ = p4.System.from_hoomd_simulation(simulation, "A", "B")
