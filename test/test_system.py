import hoomd
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
]

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_instantiation_valid(kwargs):
    """Ensure instantiation works for valid kwargs."""
    _ = p4.System(**kwargs)

INVALID_KWARGS = [
    dict(   # wrong probe type
        probe="wrong",
        analyte=p4.Body(primary_type="A"),
        interactions=[lj_interaction([("A", "A")])]
    ),
    dict(   # wrong analyte type
        probe=p4.Body(primary_type="A"),
        analyte="wrong",
        interactions=[lj_interaction([("A", "A")])]
    ),
    dict(   # wrong interactions type
        probe=p4.Body(primary_type="A"),
        analyte=p4.Body(primary_type="A"),
        interactions="wrong"
    ),
    dict(   # wrong interactions item type
        probe=p4.Body(primary_type="A"),
        analyte=p4.Body(primary_type="A"),
        interactions=["wrong"]
    ),
    dict(   # same primary type but are otherwise different
        probe=p4.Body(primary_type="A"),
        analyte=p4.Body(
            primary_type="A",
            secondary_types=["B"],
            positions_by_type={"B": [[1,0,0]]}
        ),
        interactions=[lj_interaction([("A", "B")])]
    ),
    dict(   # probe primary type is in analyte secondary types
        probe=p4.Body(primary_type="A"),
        analyte=p4.Body(
            primary_type="B",
            secondary_types=["A"],
            positions_by_type={"A": [[1,0,0]]}
        ),
        interactions=[lj_interaction([("A", "B")])]
    ),
    dict(   # analyte primary type is in probe secondary types
        probe=p4.Body(
            primary_type="B",
            secondary_types=["A"],
            positions_by_type={"A": [[1,0,0]]}
        ),
        analyte=p4.Body(primary_type="A"),
        interactions=[lj_interaction([("A", "B")])]
    ),
]

@pytest.mark.parametrize("kwargs", INVALID_KWARGS)
def test_instantiation_invalid(kwargs):
    """Ensure instantiation fails expectedly for invalid kwargs."""
    with pytest.raises((ValueError, TypeError)):
        _ = p4.System(**kwargs)

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

def test_probe_potential_valid():
    """Ensure probe_potential method works and returns expected results for valid kwargs."""
    # TODO
    pass

def get_valid_simulations_and_kwargs():
    """Return an array of valid simulations with corresponding kwargs.
    
    The kwargs assume that when parsing the simulation, the primary type for
    the probe will always be 'A', and for the analyte the primary type will
    always be 'C'.
    """
    simulations_and_kwargs = []

    # single-particle probe, single-particle analyte
    simulation = hoomd.util.make_example_simulation(particle_types=["A", "C"])
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
    simulation = hoomd.util.make_example_simulation(particle_types=["A", "C", "D"])
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
    simulation.operations.integrator.rigid = rigid
    kwargs = dict(
        probe=p4.Body("A"),
        analyte=p4.Body(
            primary_type="C",
            secondary_types="D",
            positions_by_type={"D": [[1,0,0]]}
        ),
        interactions=[lj_interaction([("A", "D")])]
    )
    simulations_and_kwargs.append([simulation, kwargs])

    # multi-particle probe, single-particle analyte
    simulation = hoomd.util.make_example_simulation(particle_types=["A", "B", "C"])
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
    simulation.operations.integrator.rigid = rigid
    kwargs = dict(
        probe=p4.Body(
            primary_type="A",
            secondary_types="B",
            positions_by_type={"B": [[1,0,0]]}
        ),
        analyte=p4.Body("C"),
        interactions=[lj_interaction([("B", "C")])]
    )
    simulations_and_kwargs.append([simulation, kwargs])

    # multi-particle probe, multi-particle analyte
    simulation = hoomd.util.make_example_simulation(particle_types=["A", "B", "C", "D"])
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
    simulation.operations.integrator.rigid = rigid
    kwargs = dict(
        probe=p4.Body(
            primary_type="A",
            secondary_types=["B"],
            positions_by_type={"B": [[1,0,0]]}
        ),
        analyte=p4.Body(
            primary_type="C",
            secondary_types="D",
            positions_by_type={"D": [[0,1,0]]}
        ),
        interactions=[lj_interaction([("B", "D")])]
    )
    simulations_and_kwargs.append([simulation, kwargs])

    return simulations_and_kwargs

@pytest.mark.parametrize("simulation,kwargs", get_valid_simulations_and_kwargs())
def test_from_hoomd_simulation_valid(simulation, kwargs):
    """Ensure parsing from hoomd simulations works for valid simulations."""
    a = p4.System.from_hoomd_simulation(simulation, "A", "C")
    b = p4.System(**kwargs)
    assert a == b

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
