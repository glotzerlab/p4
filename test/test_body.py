# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

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
            all(list(o) == [1,0,0,0] for o in body2.orientations_by_type[t])
            for t in orientations_missing_from_self
        )
        and
        all(
            all(list(o) == [1,0,0,0] for o in body1.orientations_by_type[t])
            for t in orientations_missing_from_other
        )
    )

    assert orientations_same or orientations_equivalent
    assert body1.mass == body2.mass
    assert body1.moi == body2.moi

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
        mass=2,
        moi=[1, 0, 0]
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
    dict(   # 2 seconary types, positions and orientations (plus masses and mois), with np dtypes
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            B=[np.array([1,1,1])],
            C=np.array([[1,0,0], [0,1,0]])
        ),
        orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[np.float32(1),0,0,0], [0,1,0,0]]
        ),
        mass=np.float64(2),
        moi=[1, 0, 0]
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
    ),

    # Positions are wrong length
    dict(
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(B=[[1,0]], C=[[0,1,0]])
    ),
    # Orientations are wrong length
    dict(
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(B=[[1,0]], C=[[0,1,0]]),
        orientations_by_type=dict(B=[[1,0]], C=[[0,1,0]]),
    ),
    # MoI is wrong length
    dict(
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(B=[[1,0]], C=[[0,1,0]]),
        moi=[1,0],
    )
]

@pytest.mark.parametrize("kwargs", INVALID_KWARGS)
def test_invalid_instantiation(kwargs):
    """Ensure instantiation fails predictably with invalid arguments."""
    with pytest.raises((ValueError, TypeError)):
        _ = p4.Body(**kwargs)

def test_property_getters_and_setters_valid():
    """Ensure property getters and setters work as expected with valid inputs."""
    kwargs = dict(
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
        mass=2,
        moi=[1, 0, 0]
    )

    body = p4.Body(**deepcopy(kwargs))

    # Getters
    assert body.primary_type == kwargs["primary_type"]
    assert body.secondary_types == kwargs["secondary_types"]
    assert body.positions_by_type == kwargs["positions_by_type"]
    assert body.orientations_by_type == kwargs["orientations_by_type"]
    assert body.mass == float(kwargs["mass"])
    assert body.moi == kwargs["moi"]

    # Setters (secondary types, positions, and orientations) (test coercion, too)
    body.add(
        name="Z",
        positions=[np.array([1,1,1])],
        orientations=np.array([[0,0,1,0]]),
    )
    assert body.secondary_types == kwargs["secondary_types"] + ["Z"]
    assert body.positions_by_type == {**kwargs["positions_by_type"], **dict(Z=[[1,1,1]])}
    assert body.orientations_by_type == {**kwargs["orientations_by_type"], **dict(Z=[[0,0,1,0]])}
    
    body.remove(name="Z")
    assert body.secondary_types == kwargs["secondary_types"]
    assert body.positions_by_type == kwargs["positions_by_type"]
    assert body.orientations_by_type == kwargs["orientations_by_type"]

    body.update(
        name="B",
        positions=[np.array([0,0,0])],
        orientations=np.array([[0,1,0,0]]),
    )
    assert body.secondary_types == kwargs["secondary_types"]
    assert body.positions_by_type == dict(B=[[0,0,0]], C=[[1,0,0], [0,1,0]])
    assert body.orientations_by_type == dict(B=[[0,1,0,0]], C=[[1,0,0,0], [0,1,0,0]])

    # Setters (everything else)
    body.primary_type = "Z"
    assert body.primary_type == "Z"
    body.orientations_by_type = {}
    assert body.orientations_by_type == {}
    body.mass = 5
    assert body.mass == 5
    body.moi = [5, 5, 5]
    assert body.moi == [5, 5, 5]

def test_property_setters_invalid():
    """Ensure property setters fail expectedly with invalid inputs."""
    kwargs = dict(
        primary_type="A",
        secondary_types=["B", "C"],
        positions_by_type=dict(
            B=[np.array([1,1,1])],
            C=np.array([[1,0,0], [0,1,0]])
        ),
        orientations_by_type=dict(
            B=[[1,1,0,0]],
            C=[[np.float32(1),0,0,0], [0,1,0,0]]
        ),
        mass=np.float64(2),
        moi=[1, 0, 0]
    )

    body = p4.Body(**deepcopy(kwargs))

    # Adding/removing/updating clashing secondary type
    with pytest.raises(ValueError):
        body.add(name="B", positions=[[0,0,0]])
    with pytest.raises(ValueError):
        body.remove(name="Z")
    with pytest.raises(ValueError):
        body.update(name="Z", positions=[[0,0,0]])

    # Adding with missing positions
    with pytest.raises(TypeError):
        body.add(name="Z")

    # Adding/updating with positions of wrong length
    with pytest.raises(ValueError):
        body.add(name="Z", positions=[[0,0]])
    with pytest.raises(ValueError):
        body.update(name="B", positions=[[0,0]])

    # Adding/updating with mismatched numbers of positions and orientations
    with pytest.raises(ValueError):
        body.add(name="Z", positions=[[0,0,0]], orientations=[[0,0,1,0], [1,0,0,0]])
    with pytest.raises(ValueError):
        body.update(name="B", positions=[[0,0,0]], orientations=[[0,0,1,0], [1,0,0,0]])

    # Adding/updating with orientations of wrong length
    with pytest.raises(ValueError):
        body.add(name="Z", positions=[[1,1,1]], orientations=[[0,0,1]])
    with pytest.raises(ValueError):
        body.update(name="B", positions=[[1,1,1]], orientations=[[0,0,1]])
    
    # Ensure none of the invalid operations above mutated the internal data
    assert body.secondary_types == kwargs["secondary_types"]
    assert body.positions_by_type == p4.util.data.sanitize(kwargs["positions_by_type"])
    assert body.orientations_by_type == p4.util.data.sanitize(kwargs["orientations_by_type"])
    assert body.mass == float(kwargs["mass"])
    assert body.moi == kwargs["moi"]

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

def merge_rigids(rigid1, rigid2):
    """Combine two rigid constraints together."""
    merged = hoomd.md.constrain.Rigid()
    
    for primary_type, body_dict in rigid1.body.items():
        merged.body[primary_type] = body_dict
    
    for primary_type, body_dict in rigid2.body.items():
        merged.body[primary_type] = body_dict
    
    return merged

def make_rigid_or_snapshot_and_expected_bodies_from_variants(
    rigid_or_snapshot,
    single_or_multi_body,
    single_or_multi_particle,
    rigid_and_snapshot_variant=None
):
    """Create a rigid constraint or snapshot that matches the variants, and return it with the expected bodies.

    Parameters
    ----------
    rigid_or_snapshot : 'rigid' or 'snapshot' or 'both'
        Whether to create a rigid constraint or a snapshot. If 'both', then both
        are returned. If this parameter is set to 'both' and
        ``single_or_multi_particle`` is set to 'both', then
        ``rigid_and_snapshot_variant`` controls what goes in each.
    single_or_multi_body : 'single' or 'multi'
        Whether the snapshot or rigid contains a single or multiple composite
        bodies.
    single_or_multi_particle : 'single' or 'multi' or 'both'
        Whether each defined body has a single or multiple constituent
        particles. If 'both', then one body is single-particle and the other
        is multi-particle. 'both' can only be chosen when
        ``single_or_multi_body`` is 'multi'.
    rigid_and_snapshot_variant : 'rigid-missing-body' or 'snapshot-missing-body' or 'clashing-bodies', optional
        Ignored unless ``rigid_or_snapshot='both'``,``single_or_multi_body='multi'``,
        and ``single_or_multi_particle='both'`. The variants are as follows.
        'rigid-missing-body': the single-particle body is present in snapshot
        but missing from rigid. 'snapshot-missing-body': the single-particle
        body is present in rigid but missing from snapshot. 'clashing-bodies':
        the single-particle body is present in both, but the multi-particle body
        is defined differently in rigid and snapshot. In this case, the expected
        bodies reflects the definition in snapshot.
    """
    if rigid_or_snapshot in ["rigid", "both"]:
        rigid = hoomd.md.constrain.Rigid()
    if rigid_or_snapshot in ["snapshot", "both"]:
        frame = gsd.hoomd.Frame()
        frame.configuration.box = [100, 100, 100, 0, 0, 0]

    if single_or_multi_body == "single" and single_or_multi_particle == "single":
        expected_bodies = [p4.Body("A", mass=2, moi=[2, 0, 0])]
        
        if rigid_or_snapshot in ["rigid", "both"]:
            rigid.body["A"] = None

        if rigid_or_snapshot in ["snapshot", "both"]:
            frame.particles.types = ["A"]
            frame.particles.typeid = np.array([0, 0], dtype=np.uint32)
            frame.particles.position = np.array([[0, 1, 0], [0, 2, 0]], dtype=np.float32)
            frame.particles.orientation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
            frame.particles.body = np.array([-1, -1], dtype=np.int32)
            frame.particles.mass = np.array([2, 2], dtype=np.float32)
            frame.particles.moment_inertia = np.array(
                [
                    [2, 0, 0],
                    [2, 0, 0],
                ],
                dtype=np.float32
            )

    elif single_or_multi_body == "multi" and single_or_multi_particle == "single":
        expected_bodies = [
            p4.Body("A", mass=2, moi=[2, 0, 0]),
            p4.Body("D", mass=5, moi=[5, 0, 0])
        ]

        if rigid_or_snapshot in ["rigid", "both"]:
            rigid.body["A"] = None
            rigid.body["D"] = dict(constituent_types=[], positions=[], orientations=[])

        if rigid_or_snapshot in ["snapshot", "both"]:
            frame.particles.types = ["A", "D"]
            frame.particles.typeid = np.array([0, 1], dtype=np.uint32)
            frame.particles.position = np.array([[0, 1, 0], [0, 2, 0]], dtype=np.float32)
            frame.particles.orientation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
            frame.particles.body = np.array([-1, -1], dtype=np.int32)
            frame.particles.mass = np.array([2, 5], dtype=np.float32)
            frame.particles.moment_inertia = np.array(
                [
                    [2, 0, 0],
                    [5, 0, 0],
                ],
                dtype=np.float32
            )

    elif single_or_multi_body == "single" and single_or_multi_particle == "multi":
        expected_bodies = [
            p4.Body(
                primary_type="A",
                secondary_types=["B", "C"],
                positions_by_type=dict(B=[[1, 0, 0]], C=[[0, 1, 0]]),
                orientations_by_type=dict(B=[[0, 1, 0, 0]], C=[[0, 0, 1, 0]]),
                mass=2,
                moi=[2, 0, 0]
            )
        ]

        if rigid_or_snapshot in ["rigid", "both"]:
            rigid.body["A"] = dict(
                constituent_types=["B", "C"],
                positions=[[1, 0, 0], [0, 1, 0]],
                orientations=[[0, 1, 0, 0], [0, 0, 1, 0]]
            )

        if rigid_or_snapshot in ["snapshot", "both"]:
            frame.particles.types = ["A", "B", "C"]
            frame.particles.typeid = np.array([0, 1, 2], dtype=np.uint32)
            frame.particles.position = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            frame.particles.orientation = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
            frame.particles.body = np.array([0, 0, 0], dtype=np.int32)
            frame.particles.mass = np.array([2, 0, 0], dtype=np.float32)
            frame.particles.moment_inertia = np.array(
                [
                    [2, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=np.float32
            )

    elif single_or_multi_body == "multi" and single_or_multi_particle == "multi":
        expected_bodies = [
            p4.Body(
                primary_type="A",
                secondary_types=["B", "C"],
                positions_by_type=dict(B=[[1, 0, 0]], C=[[0, 1, 0]]),
                orientations_by_type=dict(B=[[0, 1, 0, 0]], C=[[0, 0, 1, 0]]),
                mass=2,
                moi=[2, 0, 0]
            ),
            p4.Body(
                primary_type="D",
                secondary_types=["E", "F"],
                positions_by_type=dict(E=[[-1, 0, 0]], F=[[0, -1, 0]]),
                orientations_by_type=dict(E=[[0, 1, 0, 0]], F=[[0, 0.707, 0.707, 0]]),
                mass=5,
                moi=[5, 0, 0]
            )
        ]

        if rigid_or_snapshot in ["rigid", "both"]:
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

        if rigid_or_snapshot in ["snapshot", "both"]:
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
            frame.particles.mass = np.array([2, 0, 0, 5, 0, 0], dtype=np.float32)
            frame.particles.moment_inertia = np.array(
                [
                    [2, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [5, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=np.float32
            )

    elif single_or_multi_body == "multi" and single_or_multi_particle == "both":
        expected_bodies = [
            p4.Body("A", mass=2, moi=[2, 0, 0]),
            p4.Body(
                primary_type="D",
                secondary_types=["E", "F"],
                positions_by_type=dict(E=[[-1, 0, 0]], F=[[0, -1, 0]]),
                orientations_by_type=dict(E=[[0, 1, 0, 0]], F=[[0, 0.707, 0.707, 0]]),
                mass=5,
                moi=[5, 0, 0]
            )
        ]

        if rigid_or_snapshot == "both":
            if rigid_and_snapshot_variant == "rigid-missing-body":
                rigid.body["A"] = None
                
                frame.particles.types = ["A", "D", "E", "F"]
                frame.particles.typeid = np.array([0, 1, 2, 3], dtype=np.uint32)
                frame.particles.position = np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 1],
                        [-1, 0, 1],
                        [0, -1, 1],
                    ],
                    dtype=np.float32
                )
                frame.particles.orientation = np.array(
                    [
                        [1, 0, 0, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, -0.707, -0.707, 0],
                    ],
                    dtype=np.float32
                )
                frame.particles.body = np.array([0, 1, 1, 1], dtype=np.int32)   # test that 0 instead of -1 still works
                frame.particles.mass = np.array([2, 5, 0, 0], dtype=np.float32)
                frame.particles.moment_inertia = np.array(
                    [
                        [2, 0, 0],
                        [5, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    dtype=np.float32
                )
                

            elif rigid_and_snapshot_variant == "snapshot-missing-body":
                rigid.body["A"] = None
                rigid.body["D"] = dict(
                    constituent_types=["E", "F"],
                    positions=[[-1, 0, 0], [0, -1, 0]],
                    orientations=[(0, 1, 0, 0), (0, 0.707, 0.707, 0)]
                )
                
                frame.particles.types = ["A", "D", "E", "F"]
                frame.particles.typeid = np.array([0], dtype=np.uint32)
                frame.particles.position = np.array([[0, 0, 0]], dtype=np.float32)
                frame.particles.orientation = np.array([[1, 0, 0, 0]], dtype=np.float32)
                frame.particles.body = np.array([-1], dtype=np.int32)
                frame.particles.mass = np.array([2], dtype=np.float32)
                frame.particles.moment_inertia = np.array(
                    [
                        [2, 0, 0],
                    ],
                    dtype=np.float32
                )

            elif rigid_and_snapshot_variant == "clashing-bodies":
                rigid.body["A"] = None
                rigid.body["D"] = dict(
                    constituent_types=["E", "F"],
                    positions=[[-2, 0, 0], [0, -2, 0]],
                    orientations=[(0, 0.707, 0.707, 0), (0, 1, 0, 0)]
                )

                frame.particles.types = ["A", "D", "E", "F"]
                frame.particles.typeid = np.array([0, 1, 2, 3], dtype=np.uint32)
                frame.particles.position = np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 1],
                        [-1, 0, 1],
                        [0, -1, 1],
                    ],
                    dtype=np.float32
                )
                frame.particles.orientation = np.array(
                    [
                        [1, 0, 0, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, -0.707, -0.707, 0],
                    ],
                    dtype=np.float32
                )
                frame.particles.body = np.array([-1, 1, 1, 1], dtype=np.int32)
                frame.particles.mass = np.array([2, 5, 0, 0], dtype=np.float32)
                frame.particles.moment_inertia = np.array(
                    [
                        [2, 0, 0],
                        [5, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    dtype=np.float32
                )

        else:
            if rigid_or_snapshot == "rigid":
                rigid.body["A"] = None
                rigid.body["D"] = dict(
                    constituent_types=["E", "F"],
                    positions=[[-1, 0, 0], [0, -1, 0]],
                    orientations=[(0, 1, 0, 0), (0, 0.707, 0.707, 0)]
                )

            if rigid_or_snapshot == "snapshot":
                frame.particles.types = ["A", "D", "E", "F"]
                frame.particles.typeid = np.array([0, 1, 2, 3], dtype=np.uint32)
                frame.particles.position = np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 1],
                        [-1, 0, 1],
                        [0, -1, 1],
                    ],
                    dtype=np.float32
                )
                frame.particles.orientation = np.array(
                    [
                        [1, 0, 0, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, -0.707, -0.707, 0],
                    ],
                    dtype=np.float32
                )
                frame.particles.body = np.array([0, 1, 1, 1], dtype=np.int32)   # test that 0 instead of -1 still works
                frame.particles.mass = np.array([2, 5, 0, 0], dtype=np.float32)
                frame.particles.moment_inertia = np.array(
                    [
                        [2, 0, 0],
                        [5, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    dtype=np.float32
                )

    if rigid_or_snapshot == "rigid":
        return rigid, expected_bodies
    
    if rigid_or_snapshot == "snapshot":
        frame.particles.N = frame.particles.position.shape[0]
        snapshot = hoomd.Snapshot.from_gsd_frame(frame, communicator=hoomd.communicator.Communicator())
        return snapshot, expected_bodies
    
    if rigid_or_snapshot == "both":
        frame.particles.N = frame.particles.position.shape[0]
        snapshot = hoomd.Snapshot.from_gsd_frame(frame, communicator=hoomd.communicator.Communicator())
        return rigid, snapshot, expected_bodies

@pytest.mark.parametrize("single_or_multi_body", ["single", "multi"])
@pytest.mark.parametrize("single_or_multi_particle", ["single", "multi"])
@pytest.mark.parametrize("include_singles", [False, True])
def test_from_hoomd_rigid(
    single_or_multi_body,
    single_or_multi_particle,
    include_singles,
):
    """Ensure parsing from rigid produces the expected output.
    
    Note: in a multi-body rigid with single particles, one body is set to
    None and the other is set to the empty dict. This tests both possible
    sets of defaults.
    
    Parameters
    ----------
    single_or_multi_body : 'single' or 'multi'
        Whether the rigid defines a single or multiple separate bodies.
    single_or_multi_particle : 'single' or 'multi'
        Whether each defined body has a single or multiple constituent
        particles.
    include_singles : bool
        Whether to include single-particle bodies when doing the parsing.
    """
    rigid, expected_bodies = make_rigid_or_snapshot_and_expected_bodies_from_variants(
        rigid_or_snapshot="rigid",
        single_or_multi_body=single_or_multi_body,
        single_or_multi_particle=single_or_multi_particle,
    )
    
    if not include_singles:
        expected_bodies = [b for b in expected_bodies if b.secondary_types]

    # When parsing rigid, mass and moi will always be default
    for b in expected_bodies:
        b.mass = 1
        if b.secondary_types:
            b.moi = [1, 1, 1]
        else:
            b.moi = [0, 0, 0]

    actual_bodies = p4.Body.from_hoomd_rigid(rigid, include_singles)

    assert len(actual_bodies) == len(expected_bodies)
    for b_a in actual_bodies:
        b_e = [b for b in expected_bodies if b.primary_type == b_a.primary_type][0] # gross but it works
        assert_bodies_equivalent(b_a, b_e)

@pytest.mark.parametrize("single_or_multi_body", ["single", "multi"])
@pytest.mark.parametrize("single_or_multi_particle", ["single", "multi"])
@pytest.mark.parametrize("include_singles", [False, True])
def test_from_hoomd_snapshot(
    single_or_multi_body,
    single_or_multi_particle,
    include_singles,
):
    """Ensure parsing from a snapshot produces the expected result."""
    snapshot, expected_bodies = make_rigid_or_snapshot_and_expected_bodies_from_variants(
        rigid_or_snapshot="snapshot",
        single_or_multi_body=single_or_multi_body,
        single_or_multi_particle=single_or_multi_particle,
    )
    
    if not include_singles:
        expected_bodies = [b for b in expected_bodies if b.secondary_types]
    
    actual_bodies = p4.Body.from_hoomd_snapshot(snapshot, include_singles)

    assert len(actual_bodies) == len(expected_bodies)
    for b_a in actual_bodies:
        b_e = [b for b in expected_bodies if b.primary_type == b_a.primary_type][0] # gross but it works
        assert_bodies_equivalent(b_a, b_e)

def make_simulation_and_expected_bodies_from_variants(
    single_or_multi_body,
    single_or_multi_particle,
    include_mass_and_moi,
    integrator_variant,
    rigid_and_snapshot_variant=None
):
    """Create a simulation that matches the variants and return it with the expected bodies.
    
    Parameters
    ----------
    single_or_multi_body : 'single' or 'multi'
        Whether the simulation contains a single or multiple composite bodies.
    single_or_multi_particle : 'single' or 'multi' or 'both'
        Whether each defined body has a single or multiple constituent
        particles. If 'both', then one body is single-particle and the other
        is multi-particle. 'both' can only be chosen when
        ``single_or_multi_body`` is 'multi'.
    include_mass_and_moi : bool
        Whether to include mass and moment of inertia in the body definitions.
    integrator_variant : 'no-integrator', 'integrator-no-rigid', 'integrator-rigid'
        If 'no-integrator', then the simulation has no integrator (and therefore
        no rigid). If 'integrator-no-rigid', the simulation has an integrator
        but no rigid is attached to it. If 'integrator-rigid', the simulation
        has an integrator with an attached rigid.
    rigid_and_snapshot_variant : 'rigid-missing-body' or 'snapshot-missing-body' or 'clashing-bodies', optional
        Ignored unless ``single_or_multi_body='multi'`` and ``single_or_multi_particle='both'``.
        The variants are as follows. 'rigid-missing-body': the single-particle
        body is present in snapshot but missing from rigid. 'snapshot-missing-body':
        the single-particle body is present in rigid but missing from snapshot.
        'clashing-bodies': the single-particle body is present in both, but the
        multi-particle body is defined differently in rigid and snapshot. In
        this case, the expected bodies reflects the definition in snapshot.
    """
    simulation = hoomd.Simulation(device=hoomd.device.CPU())

    rigid, snapshot, expected_bodies = make_rigid_or_snapshot_and_expected_bodies_from_variants(
        rigid_or_snapshot="both",
        single_or_multi_body=single_or_multi_body,
        single_or_multi_particle=single_or_multi_particle,
        rigid_and_snapshot_variant=rigid_and_snapshot_variant
    )

    # Mass and MoI cannot be included if the snapshot doesn't contain the body
    if include_mass_and_moi:
        mass_by_type = dict(A=2, D=5)
        moi_by_type = dict(
            A=[2, 0, 0],
            D=[5, 0, 0],
        )
        for i, tid in enumerate(snapshot.particles.typeid):
            t = snapshot.particles.types[tid]
            snapshot.particles.mass[i] = mass_by_type.get(t, 0)
            snapshot.particles.moment_inertia[i] = moi_by_type.get(t, [0, 0, 0])

    if integrator_variant == "integrator-no-rigid":
        simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)
    elif integrator_variant == "integrator-rigid":
        simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)
        simulation.operations.integrator.rigid = rigid

    # If bodies are missing from the snapshot and there is no rigid, then
    # they cannot be included in the expected bodies.
    if (
        rigid_and_snapshot_variant == "snapshot-missing-body"
        and integrator_variant != "integrator-rigid"
    ):
        expected_bodies = [b for b in expected_bodies if not b.secondary_types]

    # If bodies are missing from the snapshot and there is a rigid, then the
    # mass and moi for the rigid body will be defaults
    if (
        rigid_and_snapshot_variant == "snapshot-missing-body"
        and integrator_variant == "integrator-rigid"
    ):
        for b in expected_bodies:
            if b.secondary_types:
                b.mass = 1
                b.moi = [1, 1, 1]

    simulation.create_state_from_snapshot(snapshot)
    
    return simulation, expected_bodies

@pytest.mark.parametrize("single_or_multi_body,single_or_multi_particle,rigid_and_snapshot_variant", [
    # ["single", "single", None],
    ["multi", "single", None],
    # ["single", "multi", None],
    # ["multi", "multi", None],
    # ["multi", "both", "rigid-missing-body"],
    # ["multi", "both", "snapshot-missing-body"],
    # ["multi", "both", "clashing-bodies"],
])
@pytest.mark.parametrize("include_singles", [True])
@pytest.mark.parametrize("include_mass_and_moi",[True])
@pytest.mark.parametrize("integrator_variant", ["no-integrator", "integrator-no-rigid", "integrator-rigid"])
def test_from_hoomd_simulation(
    single_or_multi_body,
    single_or_multi_particle,
    rigid_and_snapshot_variant,
    include_singles,
    include_mass_and_moi,
    integrator_variant,
):
    """Ensure parsing from simulation produces the expected result."""
    simulation, expected_bodies = make_simulation_and_expected_bodies_from_variants(
        single_or_multi_body=single_or_multi_body,
        single_or_multi_particle=single_or_multi_particle,
        include_mass_and_moi=include_mass_and_moi,
        rigid_and_snapshot_variant=rigid_and_snapshot_variant,
        integrator_variant=integrator_variant
    )

    if not include_singles:
        expected_bodies = [b for b in expected_bodies if b.secondary_types]

    actual_bodies = p4.Body.from_hoomd_simulation(simulation, include_singles)

    assert len(actual_bodies) == len(expected_bodies)
    for b_a in actual_bodies:
        b_e = [b for b in expected_bodies if b.primary_type == b_a.primary_type][0] # gross but it works
        assert_bodies_equivalent(b_a, b_e)

def assert_rigids_are_equal(rigid1, rigid2):
    """Assert that two rigid constraints are equivalent."""
    r1_body = rigid1.body.to_base()
    r2_body = rigid2.body.to_base()
    assert r1_body == r2_body

def make_rigid_from_kwargs(
    primary_type,
    secondary_types=[],
    positions_by_type={},
    orientations_by_type={},
    **kwargs    # ignored
):
    """Create a rigid constraint that corresponds to the kwargs for a Body."""
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

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_to_hoomd_rigid(kwargs):
    """Ensure converting to rigid produces the expected output."""
    kwargs = copy(kwargs)

    body = p4.Body(**kwargs)

    # Test with all secondary types and no existing rigid
    rigid = make_rigid_from_kwargs(**kwargs)
    assert_rigids_are_equal(rigid, body.to_hoomd_rigid())

    # Test with existing rigid
    existing_rigid = make_rigid_from_kwargs("X", ["Y"], dict(Y=[[0,0,1], [0,0,-1]]))
    merged_rigid = merge_rigids(existing_rigid, rigid)
    assert_rigids_are_equal(merged_rigid, body.to_hoomd_rigid(rigid=existing_rigid))

def make_snapshot_from_kwargs(
    primary_type,
    secondary_types=[],
    positions_by_type={},
    orientations_by_type={},
    mass=None,
    moi=None
):
    """Create a snapshot that corresponds to the kwargs for a Body."""
    if mass is None:
        mass = 1
    if not secondary_types:
        moi = [0, 0, 0]
    elif moi is None:
        moi = [1, 1, 1]
    
    frame = gsd.hoomd.Frame()
    frame.particles.types = [primary_type] + secondary_types

    typeids = []
    positions = np.empty((0, 3), dtype=np.float32)
    orientations = np.empty((0, 4), dtype=np.float32)
    masses = []
    mois = np.empty((0, 3), dtype=np.float32)
    bodyids = []

    # Add primary particle
    typeids.append(0)
    positions = np.vstack((positions, [[0, 0, 0]]))
    orientations = np.vstack((orientations, [[1, 0, 0, 0]]))
    masses.append(mass)
    mois = np.vstack((mois, moi))
    bodyids.append(0)

    # Add secondary particles
    for i, t in enumerate(secondary_types):
        typeids.extend([i+1 for _ in positions_by_type[t]])
        positions = np.vstack((positions, positions_by_type[t]))
        orientations = np.vstack((orientations, orientations_by_type.get(t, [[1, 0, 0, 0] for _ in positions_by_type[t]])))
        masses.extend([0 for _ in positions_by_type[t]])
        mois = np.vstack((mois, [[0, 0, 0] for _ in positions_by_type[t]]))
        bodyids.extend([0 for _ in positions_by_type[t]])

    frame.particles.N = positions.shape[0]
    frame.particles.typeid = typeids
    frame.particles.position = positions
    frame.particles.orientation = orientations
    frame.particles.mass = masses
    frame.particles.moment_inertia = mois
    frame.particles.body = bodyids

    return hoomd.Snapshot.from_gsd_frame(
        gsd_snap=frame,
        communicator=hoomd.communicator.Communicator()
    )

@pytest.mark.parametrize("kwargs", VALID_KWARGS)
def test_to_hoomd_snapshot(kwargs):
    """Ensure export to snapshot produces the expected output."""
    arrangement = p4.Body(**kwargs)
    ref_snap = make_snapshot_from_kwargs(**kwargs)
    test_snap = arrangement.to_hoomd_snapshot()

    ref_typeids = ref_snap.particles.typeid.tolist()
    ref_positions = ref_snap.particles.position.tolist()
    ref_orientations = ref_snap.particles.orientation.tolist()
    
    test_typeids = test_snap.particles.typeid.tolist()
    test_positions = test_snap.particles.position.tolist()
    test_orientations = test_snap.particles.orientation.tolist()

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

BODY_KWARGS_FOR_GSD = dict(
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
    mass=2,
    moi=[1, 0, 0]
)

TYPE_SHAPES_FOR_GSD = dict(
    A=coxeter.shapes.Ellipsoid(a=0.25, b=0.5, c=0.75),
    B=coxeter.shapes.ConvexPolyhedron(vertices=CUBE_VERTICES),
)

@pytest.mark.parametrize("kwargs,ref_filename,type_shapes", [
    [    # 2 seconary types, positions and orientations, type shapes are specified
        BODY_KWARGS_FOR_GSD,
        "body.gsd",
        TYPE_SHAPES_FOR_GSD
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


if __name__ == "__main__":
    # Regenerate reference file
    body = p4.Body(**BODY_KWARGS_FOR_GSD)
    file_path = Path(REFERENCE_FOLDER) / f"body.gsd"
    body.to_gsd(file_path, TYPE_SHAPES_FOR_GSD)
