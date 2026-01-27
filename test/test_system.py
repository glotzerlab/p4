import hoomd
import pytest
from p4 import Interaction, System, Body

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


def lj_interaction(all_types, yes_pairs):
    """Return a simple LJ Interaction with the specified all and yes types."""
    return Interaction(
        hoomd_class=hoomd.md.pair.LJ,
        initial_args={},
        all_types=all_types,
        no_params=dict(
            r_cut=0,
            params=dict(epsilon=0, sigma=1)
        ),
        yes_params={
            p: dict(r_cut=5, params=dict(epsilon=1, sigma=1))
            for p in yes_pairs
        }
    )

def alj_interaction(all_types, yes_pairs, yes_singles):
    """Return a simple ALJ Interaction with the specified all and yes types."""
    kwargs = dict(
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args={},
        all_types=all_types,
        no_params=dict(
            r_cut=0,
            params=dict(epsilon=0, sigma_i=0.1, sigma_j=0.1, alpha=0),
            shape=dict(vertices=[], faces=[])
        ),
        yes_params={
            p: dict(
                r_cut=5,
                params=dict(epsilon=1, sigma_i=0.1, sigma_j=0.1, alpha=0),
            )
            for p in yes_pairs
        }
    )

    for t in yes_singles:
        kwargs["yes_params"][t] = dict(
            shape=dict(vertices=get_cube_vertices(1), faces=get_cube_faces())
        )

    return Interaction(**kwargs)

VALID_KWARGS = [
    # single-particle probe, single-particle analyte
    dict(   # one interaction
        probe=Body(primary_type="A"),
        analyte=Body(primary_type="A"),
        interactions=[lj_interaction(["A"], [("A", "A")])]
    ),
    dict(   # two interactions (no extra types)
        probe=Body(primary_type="A"),
        analyte=Body(primary_type="B"),
        interactions=[
            lj_interaction(["A", "B"], [("A", "B")]),
            lj_interaction(["A", "B"], [("A", "A")])
        ]
    ),
    dict(   # two interactions (extra types)
        probe=Body(primary_type="A"),
        analyte=Body(primary_type="B"),
        interactions=[
            lj_interaction(["A", "B", "C"], [("A", "B")]),
            lj_interaction(["A", "B", "C", "D"], [("A", "A"), ("C", "C")])
        ]
    ),

    # single-particle probe, multi-particle analyte
    dict(   # one interaction
        probe=Body(primary_type="A"),
        analyte=Body(
            primary_type="B",
            secondary_types=["C"],
            secondary_positions_by_type={"C": [[1,0,0]]},
            secondary_orientations_by_type={"C": [[0,1,0,0]]}
        ),
        interactions=[
            lj_interaction(["A", "B", "C"], [("A", "C")])
        ]
    ),
    dict(   # two interactions (no extra types)
        probe=Body(primary_type="A"),
        analyte=Body(
            primary_type="B",
            secondary_types=["C"],
            secondary_positions_by_type={"C": [[1,0,0]]},
            secondary_orientations_by_type={"C": [[0,1,0,0]]}
        ),
        interactions=[
            lj_interaction(["A", "B", "C"], [("A", "C")]),
            lj_interaction(["A", "B", "C"], [("A", "B")])
        ]
    ),
    dict(   # two interactions (extra types)
        probe=Body(primary_type="A"),
        analyte=Body(
            primary_type="B",
            secondary_types=["C"],
            secondary_positions_by_type={"C": [[1,0,0]]},
            secondary_orientations_by_type={"C": [[0,1,0,0]]}
        ),
        interactions=[
            lj_interaction(["A", "B", "C", "D"], [("A", "C")]),
            lj_interaction(["A", "B", "C", "D", "E"], [("A", "B")])
        ]
    ),

    # multi-particle probe, single-particle analyte
    dict(   # one interaction
        probe=Body(
            primary_type="B",
            secondary_types=["C"],
            secondary_positions_by_type={"C": [[1,0,0]]},
            secondary_orientations_by_type={"C": [[0,1,0,0]]}
        ),
        analyte=Body(primary_type="A"),
        interactions=[
            lj_interaction(["A", "B", "C"], [("A", "C")])
        ]
    ),
    dict(   # two interactions (no extra types)
        probe=Body(
            primary_type="B",
            secondary_types=["C"],
            secondary_positions_by_type={"C": [[1,0,0]]},
            secondary_orientations_by_type={"C": [[0,1,0,0]]}
        ),
        analyte=Body(primary_type="A"),
        interactions=[
            lj_interaction(["A", "B", "C"], [("A", "C")]),
            lj_interaction(["A", "B", "C"], [("A", "B")])
        ]
    ),
    dict(   # two interactions (extra types)
        probe=Body(
            primary_type="B",
            secondary_types=["C"],
            secondary_positions_by_type={"C": [[1,0,0]]},
            secondary_orientations_by_type={"C": [[0,1,0,0]]}
        ),
        analyte=Body(primary_type="A"),
        interactions=[
            lj_interaction(["A", "B", "C", "D"], [("A", "C")]),
            lj_interaction(["A", "B", "C", "D", "E"], [("A", "B")])
        ]
    ),

    # multi-particle probe, multi-particle analyte
    dict(   # one interaction
        probe=Body(
            primary_type="B",
            secondary_types=["C"],
            secondary_positions_by_type={"C": [[1,0,0]]},
            secondary_orientations_by_type={"C": [[0,1,0,0]]}
        ),
        analyte=Body(
            primary_type="B",
            secondary_types=["C"],
            secondary_positions_by_type={"C": [[1,0,0]]},
            secondary_orientations_by_type={"C": [[0,1,0,0]]}
        ),
        interactions=[
            lj_interaction(["B", "C"], [("C", "C")])
        ]
    ),
    dict(   # two interactions (no extra types)
        probe=Body(
            primary_type="A",
            secondary_types=["B"],
            secondary_positions_by_type={"B": [[0,1,0]]},
        ),
        analyte=Body(
            primary_type="C",
            secondary_types=["D"],
            secondary_positions_by_type={"D": [[1,0,0]]},
        ),
        interactions=[
            lj_interaction(["A", "B", "C", "D"], [("B", "D")]),
            lj_interaction(["A", "B", "C", "D"], [("A", "C")])
        ]
    ),
    dict(   # two interactions (extra types)
        probe=Body(
            primary_type="A",
            secondary_types=["B"],
            secondary_positions_by_type={"B": [[0,1,0]]},
        ),
        analyte=Body(
            primary_type="C",
            secondary_types=["D"],
            secondary_positions_by_type={"D": [[1,0,0]]},
        ),
        interactions=[
            lj_interaction(["A", "B", "C", "D", "E"], [("B", "D")]),
            lj_interaction(["A", "B", "C", "D", "E", "F"], [("A", "C"), ("E", "F")])
        ]
    ),
]

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_instantiation_valid(kwargs):
    """Ensure instantiation works for valid kwargs."""
    _ = System(**kwargs)

INVALID_KWARGS = [
    dict(   # wrong probe type
        probe="wrong",
        analyte=Body(primary_type="A"),
        interactions=[lj_interaction(["A"], [("A", "A")])]
    ),
    dict(   # wrong analyte type
        probe=Body(primary_type="A"),
        analyte="wrong",
        interactions=[lj_interaction(["A"], [("A", "A")])]
    ),
    dict(   # wrong interactions type
        probe=Body(primary_type="A"),
        analyte=Body(primary_type="A"),
        interactions="wrong"
    ),
    dict(   # wrong interactions item type
        probe=Body(primary_type="A"),
        analyte=Body(primary_type="A"),
        interactions=["wrong"]
    ),
    dict(   # same primary type but are otherwise different
        probe=Body(primary_type="A"),
        analyte=Body(
            primary_type="A",
            secondary_types=["B"],
            secondary_positions_by_type={"B": [[1,0,0]]}
        ),
        interactions=[lj_interaction(["A", "B"], [("A", "B")])]
    ),
    dict(   # probe primary type is in analyte secondary types
        probe=Body(primary_type="A"),
        analyte=Body(
            primary_type="B",
            secondary_types=["A"],
            secondary_positions_by_type={"A": [[1,0,0]]}
        ),
        interactions=[lj_interaction(["A", "B"], [("A", "B")])]
    ),
    dict(   # analyte primary type is in probe secondary types
        probe=Body(
            primary_type="B",
            secondary_types=["A"],
            secondary_positions_by_type={"A": [[1,0,0]]}
        ),
        analyte=Body(primary_type="A"),
        interactions=[lj_interaction(["A", "B"], [("A", "B")])]
    ),
    dict(   # probe primary type not covered
        probe=Body(
            primary_type="A",
            secondary_types=["B"],
            secondary_positions_by_type={"B": [[1,0,0]]}
        ),
        analyte=Body(
            primary_type="C",
            secondary_types=["D"],
            secondary_positions_by_type={"D": [[1,0,0]]}
        ),
        interactions=[lj_interaction(["B", "C", "D"], [("B", "D")])]
    ),
    dict(   # probe secondary type not covered
        probe=Body(
            primary_type="A",
            secondary_types=["B"],
            secondary_positions_by_type={"B": [[1,0,0]]}
        ),
        analyte=Body(
            primary_type="C",
            secondary_types=["D"],
            secondary_positions_by_type={"D": [[1,0,0]]}
        ),
        interactions=[lj_interaction(["A", "C", "D"], [("A", "D")])]
    ),
    dict(   # analyte primary type not covered
        probe=Body(
            primary_type="A",
            secondary_types=["B"],
            secondary_positions_by_type={"B": [[1,0,0]]}
        ),
        analyte=Body(
            primary_type="C",
            secondary_types=["D"],
            secondary_positions_by_type={"D": [[1,0,0]]}
        ),
        interactions=[lj_interaction(["A", "B", "D"], [("B", "D")])]
    ),
    dict(   # analyte secondary type not covered
        probe=Body(
            primary_type="A",
            secondary_types=["B"],
            secondary_positions_by_type={"B": [[1,0,0]]}
        ),
        analyte=Body(
            primary_type="C",
            secondary_types=["D"],
            secondary_positions_by_type={"D": [[1,0,0]]}
        ),
        interactions=[lj_interaction(["A", "B", "C"], [("B", "C")])]
    )
]

@pytest.mark.parametrize("kwargs", INVALID_KWARGS)
def test_instantiation_invalid(kwargs):
    """Ensure instantiation fails expectedly for invalid kwargs."""
    with pytest.raises((ValueError, TypeError)):
        _ = System(**kwargs)

@pytest.mark.parametrize("kwargs,expected", [
    # one active
    [   # common single types (probe primary, analyte primary)
        dict(
            probe=Body(
                primary_type="A",
                secondary_types=["B"],
                secondary_positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=Body(
                primary_type="C",
                secondary_types=["D"],
                secondary_positions_by_type={"D": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction(("A", "B", "C", "D"), [], ["A", "C"])
            ]
        ),
        [alj_interaction(("A", "B", "C", "D"), [], ["A", "C"])]
    ],
    [   # common single types (probe primary, analyte secondary)
        dict(
            probe=Body(
                primary_type="A",
                secondary_types=["B"],
                secondary_positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=Body(
                primary_type="C",
                secondary_types=["D"],
                secondary_positions_by_type={"D": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction(("A", "B", "C", "D"), [], ["A", "D"])
            ]
        ),
        [alj_interaction(("A", "B", "C", "D"), [], ["A", "D"])]
    ],
    [   # common single types (probe secondary, analyte primary)
        dict(
            probe=Body(
                primary_type="A",
                secondary_types=["B"],
                secondary_positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=Body(
                primary_type="C",
                secondary_types=["D"],
                secondary_positions_by_type={"D": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction(("A", "B", "C", "D"), [], ["B", "C"])
            ]
        ),
        [alj_interaction(("A", "B", "C", "D"), [], ["B", "C"])]
    ],
    [   # common single types (probe secondary, analyte secondary)
        dict(
            probe=Body(
                primary_type="A",
                secondary_types=["B"],
                secondary_positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=Body(
                primary_type="C",
                secondary_types=["D"],
                secondary_positions_by_type={"D": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction(("A", "B", "C", "D"), [], ["B", "D"])
            ]
        ),
        [alj_interaction(("A", "B", "C", "D"), [], ["B", "D"])]
    ],
    [   # common pair types (one member of pair, same for both probe and analyte)
        dict(
            probe=Body(
                primary_type="A",
                secondary_types=["B"],
                secondary_positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=Body(
                primary_type="C",
                secondary_types=["B"],
                secondary_positions_by_type={"B": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction(("A", "B", "C", "D", "E"), [("B", "E")], [])
            ]
        ),
        [alj_interaction(("A", "B", "C", "D", "E"), [("B", "E")], [])]
    ],
    [   # common pair types (different members for probe and analyte)
        dict(
            probe=Body(
                primary_type="A",
                secondary_types=["B"],
                secondary_positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=Body(
                primary_type="C",
                secondary_types=["D"],
                secondary_positions_by_type={"D": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction(("A", "B", "C", "D"), [("A", "C")], [])
            ]
        ),
        [alj_interaction(("A", "B", "C", "D"), [("A", "C")], [])]
    ],
    # multiple active
    [
        dict(
            probe=Body(
                primary_type="A",
                secondary_types=["B"],
                secondary_positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=Body(
                primary_type="C",
                secondary_types=["D"],
                secondary_positions_by_type={"D": [[0,1,0]]},
            ),
            interactions=[
                alj_interaction(("A", "B", "C", "D"), [("A", "C")], ["A", "C"]),
                lj_interaction(("A", "B", "C", "D"), [("B", "D")])
            ]
        ),
        [
            alj_interaction(("A", "B", "C", "D"), [("A", "C")], ["A", "C"]),
            lj_interaction(("A", "B", "C", "D"), [("B", "D")])
        ]
    ],
])
def test_active_interactions(kwargs, expected):
    """Ensure active_interactions method returns expected results."""
    system = System(**kwargs)
    assert system.active_interactions == expected

@pytest.mark.parametrize("kwargs,expected", [
    # single-particle probe and analyte 
    [   # one type
        dict(
            probe=Body(primary_type="A"),
            analyte=Body(primary_type="A"),
            interactions=[lj_interaction(["A"], [("A", "A")])]
        ),
        ["A"]
    ],
    [   # multiple types, one interaction
        dict(
            probe=Body(primary_type="A"),
            analyte=Body(primary_type="B"),
            interactions=[lj_interaction(["A", "B"], [("A", "B")])]
        ),
        ["A", "B"]
    ],
    [   # multiple types, multiple interactions
        dict(
            probe=Body(primary_type="A"),
            analyte=Body(primary_type="B"),
            interactions=[
                lj_interaction(["A", "B"], [("A", "B")]),
                lj_interaction(["A", "B"], [("A", "B")])
            ]
        ),
        ["A", "B"]
    ],
    # single-particle probe and multi-particle analyte 
    [
        dict(
            probe=Body(primary_type="A"),
            analyte=Body(
                primary_type="B",
                secondary_types=["C"],
                secondary_positions_by_type={"C": [[1,0,0]]}
            ),
            interactions=[
                lj_interaction(["A", "B", "C"], [("A", "C")]),
                lj_interaction(["A", "B", "C"], [("A", "B")])
            ]
        ),
        ["A", "B", "C"]
    ],
    # multi-particle probe and single-particle analyte 
    [
        dict(
            probe=Body(
                primary_type="B",
                secondary_types=["C"],
                secondary_positions_by_type={"C": [[1,0,0]]}
            ),
            analyte=Body(primary_type="A"),
            interactions=[
                lj_interaction(["A", "B", "C"], [("A", "C")]),
                lj_interaction(["A", "B", "C"], [("A", "B")])
            ]
        ),
        ["A", "B", "C"]
    ],
    # multi-particle probe and multi-particle analyte 
    [
        dict(
            probe=Body(
                primary_type="A",
                secondary_types=["B"],
                secondary_positions_by_type={"B": [[1,0,0]]}
            ),
            analyte=Body(
                primary_type="C",
                secondary_types=["D"],
                secondary_positions_by_type={"D": [[0,1,0]]}
            ),
            interactions=[
                lj_interaction(["A", "B", "C", "D"], [("B", "D")]),
                lj_interaction(["A", "B", "C", "D"], [("A", "C")])
            ]
        ),
        ["A", "B", "C", "D"]
    ]
])
def test_all_types(kwargs, expected):
    """Ensure all_types method returns expected results."""
    system = System(**kwargs)
    assert len(system.all_types) == len(expected)
    assert set(system.all_types) == set(expected)

def test_probe_potential_valid():
    """Ensure probe_potential method works and returns expected results for valid kwargs."""
    pass

def test_from_hoomd_simulation_valid():
    """Ensure parsing from hoomd simulations works for valid simulations."""
    pass

def test_from_hoomd_simulation_invalid():
    """Ensure parsing from hoomd simulations fails expectedly for invalid simulations."""
    pass
