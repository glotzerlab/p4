import os
import time
from typing import Literal

import coxeter
import p4 as pp
import hoomd
import pandas as pd
import numpy as np

import p4.util as util


def cube_verts():
    verts = [(-0.5, -0.5, -0.5),
             (-0.5, -0.5,  0.5),
             (-0.5,  0.5, -0.5),
             (-0.5,  0.5,  0.5),
             ( 0.5, -0.5, -0.5),
             ( 0.5, -0.5,  0.5),
             ( 0.5,  0.5, -0.5),
             ( 0.5,  0.5,  0.5)]
    return verts

def cube_faces():
    faces = [[0, 2, 6],
              [6, 4, 0],
              [5, 0, 4],
              [5,1,0],
              [5,4,6],
              [5,6,7],
              [3,2,0],
              [3,0,1],
              [3,6,2],
              [3,7,6],
              [3,1,5],
              [3,5,7]]
    return faces

def square_verts():
    verts = [
        [-0.5, -0.5, 0.0],
        [-0.5,  0.5, 0.0],
        [ 0.5,  0.5, 0.0],
        [ 0.5, -0.5, 0.0]
    ]
    return verts


def test_lj_sphere(n_processes=-1):
    probe_model = pp.Body("P")
    analyte_model = pp.Body("A")

    interactions = [
        pp.Interaction(
            hoomd_class=hoomd.md.pair.LJ,
            initial_args=dict(),
            no_params=dict(
                params=dict(
                    epsilon=0.0,
                    sigma=1.0
                ),
                r_cut=0.0
            ),
            all_types=["A", "P"],
            yes_params={
                ("A", "P"): dict(
                    params=dict(
                        epsilon=1.0,
                        sigma=0.5
                    ),
                    r_cut=2.0
                )
            }
        )
    ]

    system = pp.System(probe_model, analyte_model, interactions)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[30, 30, 1],
        orientation_resolutions=[1, 1, 1],
        symmetries=[1, 1, 1],
        nlist=nlist,
        csv_filename="test-lj-sphere.csv",
        cutoff_shape=None,
        outside_cutoff=2.0,
        inside_cutoff=0.2,
        n_processes=n_processes,
    )

def test_lj_sphere_3d():
    probe_model = pp.Body("P")
    analyte_model = pp.Body("A")

    interactions = [
        pp.Interaction(
            hoomd_class=hoomd.md.pair.LJ,
            initial_args=dict(),
            no_params=dict(
                params=dict(
                    epsilon=0.0,
                    sigma=1.0
                ),
                r_cut=0.0
            ),
            all_types=["A", "P"],
            yes_params={
                ("A", "P"): dict(
                    params=dict(
                        epsilon=1.0,
                        sigma=0.5
                    ),
                    r_cut=2.0
                )
            }
        )
    ]

    system = pp.System(probe_model, analyte_model, interactions)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[30, 30, 30],
        orientation_resolutions=[1, 1, 1],
        symmetries=[1, 1, 1],
        nlist=nlist,
        csv_filename="test-lj-sphere-3d.csv",
        cutoff_shape=None,
        outside_cutoff=2.0,
        inside_cutoff=0.2,
    )

def test_lj_sites():
    probe_model = pp.Body("P")
    analyte_model = pp.Body(
        primary_type="A",
        secondary_types=["B"],
        secondary_positions_by_type=dict(B=square_verts())
    )

    interactions = [
        pp.Interaction(
            hoomd_class=hoomd.md.pair.LJ,
            initial_args=dict(),
            no_params=dict(
                params=dict(
                    epsilon=0.0,
                    sigma=1.0
                ),
                r_cut=0.0
            ),
            all_types=["A", "B", "P"],  # TODO: I shouldn't need to include A, but it errors if I don't...
            yes_params={
                ("B", "P"): dict(
                    params=dict(
                        epsilon=1.0,
                        sigma=0.5
                    ),
                    r_cut=2.0
                )
            }
        )
    ]

    system = pp.System(probe_model, analyte_model, interactions)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[10, 10, 1],
        orientation_resolutions=[1, 1, 1],
        symmetries= [1, 1, 1],
        nlist=nlist,
        csv_filename="test-lj-sites.csv",
        cutoff_shape=None,
        outside_cutoff=2.0,
    )

def test_alj_cube():
    probe_model = pp.Body("P")
    analyte_model = pp.Body("A")

    interactions = [
        pp.Interaction(
            hoomd_class=hoomd.md.pair.aniso.ALJ,
            initial_args=dict(),
            no_params=dict(
                params=dict(
                    epsilon=0.0,
                    sigma_i=1.0,
                    sigma_j=1.0,
                    alpha=0,
                ),
                r_cut=0.0,
                shape=dict(vertices=[], faces=[])
            ),
            all_types=["A", "P"],
            yes_params={
                ("A", "P"): dict(
                    params=dict(
                        epsilon=1.0,
                        sigma_i=0.5,
                        sigma_j=0.5,
                        alpha=1
                    ),
                    r_cut=2.0
                ),
                "A": dict(shape=dict(vertices=cube_verts(), faces=cube_faces())),
                "P": dict(shape=dict(vertices=cube_verts(), faces=cube_faces()))
            }
        )
    ]

    system = pp.System(probe_model, analyte_model, interactions)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[30, 30, 1],
        orientation_resolutions=[1, 1, 10],
        symmetries= [1, 1, 4],
        nlist=nlist,
        csv_filename="test-alj-cube.csv",
        save_gsd=True,
        cutoff_shape=None,
        outside_cutoff=2.0,
        # probe_cutoff_inside_distance=0.0
    )

def test_alj_cube_with_eg_sites(n_processes=-1):
    probe_model = pp.Body(
        primary_type="P",
        secondary_types=["P2"],
        secondary_positions_by_type=dict(P2=square_verts())
    )
    analyte_model = pp.Body(
        primary_type="A",
        secondary_types=["B"],
        secondary_positions_by_type=dict(B=square_verts())
    )

    interactions = [
        pp.Interaction(
            hoomd_class=hoomd.md.pair.aniso.ALJ,
            initial_args=dict(),
            no_params=dict(
                params=dict(
                    epsilon=0.0,
                    sigma_i=1.0,
                    sigma_j=1.0,
                    alpha=0,
                ),
                r_cut=0.0,
                shape=dict(vertices=[], faces=[])
            ),
            all_types=["A", "P", "B", "P2"],  # TODO: I shouldn't need to include B and P2, but it errors if I don't...
            yes_params={
                ("A", "P"): dict(
                    params=dict(
                        epsilon=1.0,
                        sigma_i=0.2,
                        sigma_j=0.2,
                        alpha=1
                    ),
                    r_cut=2.0
                ),
                "A": dict(shape=dict(vertices=cube_verts(), faces=cube_faces())),
                "P": dict(shape=dict(vertices=cube_verts(), faces=cube_faces()))
            }
        ),
        pp.Interaction(
            hoomd_class=hoomd.md.pair.ExpandedGaussian,
            initial_args=dict(
                default_r_cut=1.0,
                default_r_on=0.0,
            ),
            no_params=dict(
                params=dict(
                    epsilon=0.0,
                    sigma=0.1,
                    delta=0.0
                ),
                r_cut=0.0
            ),
            all_types=["B", "P2", "A", "P"],  # TODO: I shouldn't need to include A and P, but it errors if I don't...
            yes_params={
                ("B", "P2"): dict(
                    params=dict(
                        epsilon=-1.0,
                        sigma=0.1,
                        delta=0.2
                    ),
                    r_cut=2.0
                )
            }
        )
    ]

    system = pp.System(probe_model, analyte_model, interactions)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[30, 30, 1],
        orientation_resolutions=[1, 1, 4],
        symmetries= [1, 1, 4],
        nlist=nlist,
        csv_filename="test-alj-cube-with-eg-sites.csv",
        save_gsd=True,
        cutoff_shape=None,
        outside_cutoff=2.0,
        inside_cutoff=0.2,
        n_processes=n_processes
    )

if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, lambda _a, _b: None)
    # TODO: note that GSD throws a weird error...
    # TODO: check on resolution because it seems to change in 3D...
    # for n_processes in range(1, os.process_cpu_count(),3):
    #     start_time = time.perf_counter()
    #     test_alj_cube_with_eg_sites(n_processes)
    #     print(f"{n_processes} processes completed in {round(time.perf_counter() - start_time, 2)} s.")
    # test_alj_cube_with_eg_sites(2)
    test_lj_sites()
    test_alj_cube()
    test_alj_cube_with_eg_sites()
    test_lj_sphere_3d()

    # f = pp.Field.from_csv("test-lj-sphere-3d.csv", "mean")
    # f.save_image("test-lj-sphere-3d.vti")
