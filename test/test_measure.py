# Copyright (c) 2025-2026, The Regents of the University of Michigan
# This file is from the p4 project, released under the BSD 3-Clause License.

from pathlib import Path

import pytest
import hoomd
import numpy as np
import tempfile

import p4

REFERENCE_FOLDER = Path(__file__).parent / "data"

POSITIONS = p4.positions_on_regular_grid(box=[20, 20, 20], resolution=[2, 2, 2])
ORIENTATIONS = p4.orientations_about_axis(n=2, axis=[0, 0, 1])

def test_measure_same_probe_and_analyte():
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

    with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
        test_path = Path(tempdir) / f"test.csv"
        s.measure(
            quantities=["U", "F", "T"],
            positions=POSITIONS,
            orientations=ORIENTATIONS,
            csv_filename=test_path,
        )

def test_measure_single_particle_probe_multi_particle_body_analyte():
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

    with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
        test_path = Path(tempdir) / f"test.csv"
        s.measure(
            quantities=["U", "F", "T"],
            positions=POSITIONS,
            orientations=ORIENTATIONS,
            csv_filename=test_path,
        )

def test_measure_multi_particle_probe_single_particle_body_analyte():
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

    with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
        test_path = Path(tempdir) / f"test.csv"
        s.measure(
            quantities=["U", "F", "T"],
            positions=POSITIONS,
            orientations=ORIENTATIONS,
            csv_filename=test_path,
        )

def test_measure_single_particle_probe_single_particle_arrangement_analyte():
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

    with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
        test_path = Path(tempdir) / f"test.csv"
        s.measure(
            quantities=["U", "F", "T"],
            positions=POSITIONS,
            orientations=ORIENTATIONS,
            csv_filename=test_path,
        )

def test_measure_single_particle_probe_multi_particle_arrangement_analyte():
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

    with tempfile.TemporaryDirectory(dir=REFERENCE_FOLDER) as tempdir:
        test_path = Path(tempdir) / f"test.csv"
        s.measure(
            quantities=["U", "F", "T"],
            positions=POSITIONS,
            orientations=ORIENTATIONS,
            csv_filename=test_path,
        )