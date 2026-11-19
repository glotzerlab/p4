# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from pathlib import Path

import pytest
import hoomd
import numpy as np
import tempfile

import p4

REFERENCE_FOLDER = Path(__file__).parent / "data"

POSITIONS = p4.positions_on_regular_grid(box=[20, 20, 20], resolution=[6, 6, 6])
ORIENTATIONS = p4.orientations_about_axis(n=2, axis=[0, 0, 1])


def test_few_positions_as_ndarray():
    """Ensure System.measure() does not error when there are more positions than available processes."""
    s = p4.System(
        probe=p4.Body("A"),
        analyte=p4.Body("A"),
        interactions=[
            p4.Interaction(
                hoomd_class=hoomd.md.pair.LJ,
                initial_args={},
                default_params=dict(r_cut=5, params=dict(epsilon=1, sigma=1)),
                typed_params={}
            )
        ]
    )

    s.measure(
        quantities=["U", "F", "T"],
        positions=p4.positions_on_regular_grid([2, 2, 2], [1, 1, 1]),
        orientations=ORIENTATIONS,
        n_processes=2,
        disable_pbar=True,
    )

def test_few_positions_as_list():
    """Ensure System.measure() does not error when positions are provided as a list."""
    s = p4.System(
        probe=p4.Body("A"),
        analyte=p4.Body("A"),
        interactions=[
            p4.Interaction(
                hoomd_class=hoomd.md.pair.LJ,
                initial_args={},
                default_params=dict(r_cut=5, params=dict(epsilon=1, sigma=1)),
                typed_params={}
            )
        ]
    )

    s.measure(
        quantities=["U", "F", "T"],
        positions=[p for p in p4.positions_on_regular_grid([2, 2, 2], [1, 1, 1])],
        orientations=ORIENTATIONS,
        n_processes=2,
        disable_pbar=True,
    )

def test_same_probe_and_analyte():
    """Ensure System.measure() does not error when the probe and analyte are identical."""
    s = p4.System(
        probe=p4.Body("A"),
        analyte=p4.Body("A"),
        interactions=[
            p4.Interaction(
                hoomd_class=hoomd.md.pair.LJ,
                initial_args={},
                default_params=dict(r_cut=5, params=dict(epsilon=1, sigma=1)),
                typed_params={}
            )
        ]
    )

    s.measure(
        quantities=["U", "F", "T"],
        positions=POSITIONS,
        orientations=ORIENTATIONS,
        n_processes=2,
        disable_pbar=True,
    )

def test_single_particle_probe_multi_particle_body_analyte():
    """Ensure System.measure() does not error when the probe is single-particle and the analyte is multi-particle body."""
    s = p4.System(
        probe=p4.Body("A"),
        analyte=p4.Body("B", ["C"], dict(C=[[1, 0, 0]])),
        interactions=[
            p4.Interaction(
                hoomd_class=hoomd.md.pair.LJ,
                initial_args={},
                default_params=dict(r_cut=0, params=dict(epsilon=1, sigma=1)),
                typed_params={
                    ("A", "B"): dict(r_cut=3, params=dict(epsilon=1, sigma=1)),
                    ("A", "C"): dict(r_cut=3, params=dict(epsilon=1, sigma=2))
                }
            )
        ]
    )

    s.measure(
        quantities=["U", "F", "T"],
        positions=POSITIONS,
        orientations=ORIENTATIONS,
        n_processes=2,
        disable_pbar=True,
    )

def test_multi_particle_probe_single_particle_body_analyte():
    """Ensure System.measure() does not error when the probe is multi-particle and the analyte is a single-particle body."""
    s = p4.System(
        probe=p4.Body("B", ["C"], dict(C=[[1, 0, 0]])),
        analyte=p4.Body("A"),
        interactions=[
            p4.Interaction(
                hoomd_class=hoomd.md.pair.LJ,
                initial_args={},
                default_params=dict(r_cut=0, params=dict(epsilon=1, sigma=1)),
                typed_params={
                    ("A", "B"): dict(r_cut=3, params=dict(epsilon=1, sigma=1)),
                    ("A", "C"): dict(r_cut=3, params=dict(epsilon=1, sigma=2))
                }
            )
        ]
    )

    s.measure(
        quantities=["U", "F", "T"],
        positions=POSITIONS,
        orientations=ORIENTATIONS,
        n_processes=2,
        disable_pbar=True,
    )

def test_single_particle_probe_single_particle_arrangement_analyte():
    """Ensure System.measure() does not error when the probe is multi-particle and the analyte is an arrangement of single-particle bodies."""
    s = p4.System(
        probe=p4.Body("A"),
        analyte=p4.Arrangement([p4.Body("B"), p4.Body("C")], dict(B=[[0, 0, 0]], C=[[1, 0, 0]])),
        interactions=[
            p4.Interaction(
                hoomd_class=hoomd.md.pair.LJ,
                initial_args={},
                default_params=dict(r_cut=0, params=dict(epsilon=1, sigma=1)),
                typed_params={
                    ("A", "B"): dict(r_cut=3, params=dict(epsilon=1, sigma=1)),
                    ("A", "C"): dict(r_cut=3, params=dict(epsilon=1, sigma=2))
                }
            )
        ]
    )

    s.measure(
        quantities=["U", "F", "T"],
        positions=POSITIONS,
        orientations=ORIENTATIONS,
        n_processes=2,
        disable_pbar=True,
    )

def test_single_particle_probe_multi_particle_arrangement_analyte():
    """Ensure System.measure() does not error when the probe is multi-particle and the analyte is an arrangement of multi-particle bodies."""
    s = p4.System(
        probe=p4.Body("A"),
        analyte=p4.Arrangement(
            bodies=[
                p4.Body(
                    primary_type="B",
                    secondary_types=["D"],
                    positions_by_type=dict(D=[[1, 0, 0]])
                ),
                p4.Body(
                    primary_type="C",
                    secondary_types=["E"],
                    positions_by_type=dict(E=[[-1, 0, 0]])
                )
            ],
            positions_by_type=dict(B=[[0, 0, 0]], C=[[1, 0, 0]]),
            orientations_by_type=dict(B=[[1, 0, 0, 0]], C=[[0, 0.707, 0.707, 0]])
        ),
        interactions=[
            p4.Interaction(
                hoomd_class=hoomd.md.pair.LJ,
                initial_args={},
                default_params=dict(r_cut=0, params=dict(epsilon=1, sigma=1)),
                typed_params={
                    ("A", "D"): dict(r_cut=3, params=dict(epsilon=1, sigma=1)),
                    ("A", "E"): dict(r_cut=3, params=dict(epsilon=1, sigma=2))
                }
            )
        ]
    )

    s.measure(
        quantities=["U", "F", "T"],
        positions=POSITIONS,
        orientations=ORIENTATIONS,
        n_processes=2,
        disable_pbar=True,
    )

@pytest.mark.parametrize("n_processes", [1, 2])
@pytest.mark.parametrize("return_field", [False, True])
@pytest.mark.parametrize("filename", [None, "test.csv"])
def test_parts_handled_correctly(n_processes, return_field, filename):
    """Ensure measure() does not error with settings that result in different part handling."""
    s = p4.System(
        probe=p4.Body("A"),
        analyte=p4.Body("A"),
        interactions=[
            p4.Interaction(
                hoomd_class=hoomd.md.pair.LJ,
                initial_args={},
                default_params=dict(r_cut=5, params=dict(epsilon=1, sigma=1)),
                typed_params={}
            )
        ]
    )

    positions = POSITIONS
    orientations = ORIENTATIONS

    with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
        test_path = None if filename is None else Path(tempdir) / filename
        field = s.measure(
            quantities=["U", "F", "T"],
            positions=positions,
            orientations=orientations,
            n_processes=n_processes,
            disable_pbar=True,
            return_field=return_field,
            filename=test_path,
        )

        if return_field:
            assert len(field.table["x"]) == positions.shape[0] * orientations.shape[0]

        if return_field:
            assert type(field) is p4.Field
        else:
            assert field is None

def test_parts_merged_correctly():
    """Ensure measure() merges part files correctly."""
    s = p4.System(
        probe=p4.Body("A"),
        analyte=p4.Body("A"),
        interactions=[
            p4.Interaction(
                hoomd_class=hoomd.md.pair.LJ,
                initial_args={},
                default_params=dict(r_cut=5, params=dict(epsilon=1, sigma=1)),
                typed_params={}
            )
        ]
    )

    positions = p4.positions_on_regular_grid([2, 2, 2], [20, 20, 20])
    orientations = p4.orientations_from_fibonacci_lattice(5)

    with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
        test_path = Path(tempdir) / "test.csv"
        field = s.measure(
            quantities=["U", "F", "T"],
            positions=positions,
            orientations=orientations,
            n_processes=3,
            disable_pbar=True,
            return_field=True,
            filename=test_path,
        )

        assert len(field.table["x"]) == positions.shape[0] * orientations.shape[0]
