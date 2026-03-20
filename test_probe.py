import os
import time
from typing import Literal

import coxeter
import p4
import hoomd
import pandas as pd
import numpy as np

import p4.util as util


def cube_vertices(side_length):
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

def cube_faces():
    return [
        [0, 2, 6, 4],
        [0, 4, 5, 1],
        [4, 6, 7, 5],
        [0, 1, 3, 2],
        [2, 3, 7, 6],
        [1, 5, 7, 3],
    ]


def square_verts():
    verts = [
        [-0.5, -0.5, 0.0],
        [-0.5,  0.5, 0.0],
        [ 0.5,  0.5, 0.0],
        [ 0.5, -0.5, 0.0]
    ]
    return verts


def test_lj_sphere():
    """Probe the potential around an LJ sphere."""
    probe = p4.Body("A")
    analyte = p4.Body("A")

    interactions = [
        p4.Interaction(
            hoomd_class=hoomd.md.pair.LJ,
            initial_args=dict(),
            default_params=dict(
                r_cut=0,
                params=dict(
                    epsilon=0,
                    sigma=1
                )
            ),
            typed_params={
                ("A", "A"): dict(
                    r_cut=5,
                    params=dict(
                        epsilon=1,
                        sigma=0.5
                    )
                )
            }
        )
    ]

    system = p4.System(probe, analyte, interactions)

    nlist = hoomd.md.nlist.Cell(10)

    system.measure(
        position_resolutions=[50, 50, 50],
        orientation_resolutions=[1, 1, 1],
        symmetries=[1, 1, 1],
        nlist=nlist,
        csv_filename="lj-sphere.csv",
        outside_cutoff=2.0,
        n_processes=-1
    )

def test_alj_cube():
    """Probe the potential around an ALJ cube."""
    probe = p4.Body("A")
    analyte = p4.Body("B")

    interactions = [
        p4.Interaction(
            hoomd_class=hoomd.md.pair.aniso.ALJ,
            initial_args=dict(),
            default_params=dict(
                r_cut=0,
                params=dict(
                    epsilon=0,
                    sigma_i=1,
                    sigma_j=1,
                    alpha=0
                )
            ),
            typed_params={
                ("A", "B"): dict(
                    r_cut=5,
                    params=dict(
                        epsilon=2,
                        sigma_i=2,
                        sigma_j=2,
                        alpha=3
                    )
                ),
                "A": dict(
                    shape=dict(
                        vertices=cube_vertices(2),
                        faces=cube_faces()
                    )
                ),
                "B": dict(
                    shape=dict(
                        vertices=cube_vertices(2),
                        faces=cube_faces()
                    )
                )
            }
        )
    ]

    system = p4.System(probe, analyte, interactions)

    nlist = hoomd.md.nlist.Cell(10)

    system.measure(
        position_resolutions=[50, 50, 1],
        orientation_resolutions=[1, 1, 10],
        symmetries=[1, 1, 4],
        nlist=nlist,
        csv_filename="alj-cube.csv",
        outside_cutoff=10,
        n_processes=-1
    )

def test_alj_cube_with_g_sites(n_processes=-1):
    """Probe potential energy of an ALJ cube dotted with EG interaction sites."""
    probe = p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(B=cube_vertices(2))
    )
    analyte = p4.Body(
        primary_type="C",
        secondary_types=["D"],
        positions_by_type=dict(D=cube_vertices(2))
    )

    interactions = [
        p4.Interaction(
            hoomd_class=hoomd.md.pair.aniso.ALJ,
            initial_args=dict(),
            default_params=dict(
                r_cut=0,
                params=dict(
                    epsilon=0,
                    sigma_i=1,
                    sigma_j=1,
                    alpha=0
                ),
                shape=dict(
                    vertices=[],
                    faces=[]
                )
            ),
            typed_params={
                ("A", "C"): dict(
                    r_cut=5,
                    params=dict(
                        epsilon=2,
                        sigma_i=2,
                        sigma_j=2,
                        alpha=0
                    )
                ),
                "A": dict(
                    shape=dict(
                        vertices=cube_vertices(2),
                        faces=cube_faces()
                    )
                ),
                "C": dict(
                    shape=dict(
                        vertices=cube_vertices(2),
                        faces=cube_faces()
                    )
                )
            }
        ),
        p4.Interaction(
            hoomd_class=hoomd.md.pair.Gaussian,
            initial_args=dict(),
            default_params=dict(
                r_cut=0,
                params=dict(
                    epsilon=0,
                    sigma=1,
                )
            ),
            typed_params={
                ("B", "D"): dict(
                    r_cut=5,
                    params=dict(
                        epsilon=-5,
                        sigma=0.25,
                    )
                )
            }
        )
    ]

    system = p4.System(probe, analyte, interactions)

    nlist = hoomd.md.nlist.Cell(10)

    system.measure(
        quantities=["U", "F", "T"],
        position_resolutions=[10, 10, 1],
        orientation_resolutions=[1, 1, 10],
        symmetries=[1, 1, 4],
        nlist=nlist,
        csv_filename="alj-cube-with-gauss-sites.csv",
        outside_cutoff=10,
        n_processes=-1,
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
    test_lj_sphere()
    # test_alj_cube()
    # test_alj_cube_with_g_sites()
    # test_lj_sphere_3d()

    # f = p4.Field.from_csv("test-lj-sphere-3d.csv", "mean")
    # f.save_image("test-lj-sphere-3d.vti")
