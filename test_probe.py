import os
import time
from typing import Literal

import coxeter
import p4 as pp
import hoomd
import pandas as pd
import numpy as np

import util


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
    probe_model = pp.ParticleModel("P")
    analyte_model = pp.ParticleModel("A")

    interaction_model = {
        "LJ": pp.Interaction(
            hoomd_class=hoomd.md.pair.LJ,
            initial_inputs=dict(),
            default_single_typed_attributes=dict(),
            default_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.0,
                    sigma=1.0
                ),
                r_cut=0.0
            ),
            yes_types=["A", "P"],
            yes_single_typed_attributes=dict(),
            yes_pair_typed_attributes=dict(
                params=dict(
                    epsilon=1.0,
                    sigma=0.5
                ),
                r_cut=2.0
            )
        )
    }

    system = pp.System(probe_model, analyte_model, interaction_model)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[30, 30, 1],
        orientation_resolutions=[1, 1, 1],
        orientation_symmetries=[1, 1, 1],
        interactions_to_include=["LJ"],
        nlist=nlist,
        csv_filename="test-lj-sphere.csv",
        probe_cutoff_shape=None,
        probe_cutoff_outside_distance=lambda _: 2.0,
        probe_cutoff_inside_distance=lambda _: 0.2,
        n_processes=n_processes,
    )

def test_lj_sphere_3d():
    probe_model = pp.ParticleModel("P")
    analyte_model = pp.ParticleModel("A")

    interaction_model = {
        "LJ": pp.Interaction(
            hoomd_class=hoomd.md.pair.LJ,
            initial_inputs=dict(),
            default_single_typed_attributes=dict(),
            default_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.0,
                    sigma=1.0
                ),
                r_cut=0.0
            ),
            yes_types=["A", "P"],
            yes_single_typed_attributes=dict(),
            yes_pair_typed_attributes=dict(
                params=dict(
                    epsilon=1.0,
                    sigma=0.5
                ),
                r_cut=2.0
            )
        )
    }

    system = pp.System(probe_model, analyte_model, interaction_model)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[30, 30, 30],
        orientation_resolutions=[1, 1, 1],
        orientation_symmetries=[1, 1, 1],
        interactions_to_include=["LJ"],
        nlist=nlist,
        csv_filename="test-lj-sphere-3d.csv",
        probe_cutoff_shape=None,
        probe_cutoff_outside_distance=lambda _: 2.0,
        probe_cutoff_inside_distance=lambda _: 0.2,
    )

def test_lj_sites():
    probe_model = pp.ParticleModel("P")
    analyte_model = pp.ParticleModel(
        primary_type="A",
        secondary_types=["B"],
        get_secondary_positions_by_type=lambda _: square_verts()
    )

    interaction_model = {
        "LJ": pp.Interaction(
            hoomd_class=hoomd.md.pair.LJ,
            initial_inputs=dict(),
            default_single_typed_attributes=dict(),
            default_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.0,
                    sigma=1.0
                ),
                r_cut=0.0
            ),
            yes_types=["B", "P"],
            yes_single_typed_attributes=dict(),
            yes_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.1,
                    sigma=0.2
                ),
                r_cut=2.0
            )
        )
    }

    system = pp.System(probe_model, analyte_model, interaction_model)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[30, 30, 1],
        orientation_resolutions=[1, 1, 1],
        orientation_symmetries= [1, 1, 1],
        interactions_to_include=["LJ"],
        nlist=nlist,
        csv_filename="test-lj-sites.csv",
        probe_cutoff_shape=None,
        probe_cutoff_outside_distance=lambda _: 2.0,
        # probe_cutoff_inside_distance=lambda _: 0.0
    )

def test_alj_cube():
    probe_model = pp.ParticleModel("P")
    analyte_model = pp.ParticleModel("A")

    interaction_model = {
        "ALJ": pp.Interaction(
            hoomd_class=hoomd.md.pair.aniso.ALJ,
            initial_inputs=dict(),
            default_single_typed_attributes=dict(
                shape=dict(
                    vertices=[],
                    faces=[]
                )
            ),
            default_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.0,
                    sigma_i=1.0,
                    sigma_j=1.0,
                    alpha=0,
                ),
                r_cut=0.0
            ),
            yes_types=["A", "P"],
            yes_single_typed_attributes=dict(
                shape=dict(
                    vertices=cube_verts(),
                    faces=cube_faces()
                )
            ),
            yes_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.1,
                    sigma_i=1.0,
                    sigma_j=1.0,
                    alpha=0,
                ),
                r_cut=2.0
            )
        )
    }

    system = pp.System(probe_model, analyte_model, interaction_model)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[30, 30, 1],
        orientation_resolutions=[1, 1, 10],
        orientation_symmetries= [1, 1, 4],
        interactions_to_include=["ALJ"],
        nlist=nlist,
        csv_filename="test-alj-cube.csv",
        gsd_filename="test-alj-cube.gsd",
        probe_cutoff_shape=None,
        probe_cutoff_outside_distance=lambda _: 2.0,
        # probe_cutoff_inside_distance=lambda _: 0.0
    )

def test_alj_cube_with_eg_sites(n_processes=-1):
    probe_model = pp.ParticleModel(
        primary_type="P",
        secondary_types=["P2"],
        get_secondary_positions_by_type=lambda _: square_verts()
    )
    analyte_model = pp.ParticleModel(
        primary_type="A",
        secondary_types=["B"],
        get_secondary_positions_by_type=lambda _: square_verts()
    )

    interaction_model = {
        "ALJ": pp.Interaction(
            hoomd_class=hoomd.md.pair.aniso.ALJ,
            initial_inputs=dict(),
            default_single_typed_attributes=dict(
                shape=dict(
                    vertices=[],
                    faces=[]
                )
            ),
            default_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.0,
                    sigma_i=1.0,
                    sigma_j=1.0,
                    alpha=0,
                ),
                r_cut=0.0
            ),
            yes_types=["A", "P"],
            yes_single_typed_attributes=dict(
                shape=dict(
                    vertices=cube_verts(),
                    faces=cube_faces()
                )
            ),
            yes_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.1,
                    sigma_i=1.0,
                    sigma_j=1.0,
                    alpha=0,
                ),
                r_cut=2.0
            )
        ),
        "EG": pp.Interaction(
            hoomd_class=hoomd.md.pair.ExpandedGaussian,
            initial_inputs=dict(
                default_r_cut=1.0,
                default_r_on=0.0,
            ),
            default_single_typed_attributes=dict(),
            default_pair_typed_attributes=dict(
                params=dict(
                    epsilon=0.0,
                    sigma=0.1,
                    delta=0.0
                ),
                r_cut=0.0
            ),
            yes_types=["B", "P2"],
            yes_single_typed_attributes=dict(),
            yes_pair_typed_attributes=dict(
                params=dict(
                    epsilon=-1.0,
                    sigma=0.1,
                    delta=0.2
                ),
                r_cut=2.0
            )
        )
    }

    system = pp.System(probe_model, analyte_model, interaction_model)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[10, 10, 1],
        orientation_resolutions=[1, 1, 4],
        orientation_symmetries= [1, 1, 4],
        interactions_to_include=["ALJ", "EG"],
        nlist=nlist,
        csv_filename="test-alj-cube-with-eg-sites.csv",
        save_gsd=True,
        probe_cutoff_shape=None,
        probe_cutoff_outside_distance=lambda _: 2.0,
        probe_cutoff_inside_distance=lambda _: 0.2,
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
    test_lj_sites()               # looks good
    # test_alj_cube()               # looks good
    # test_alj_cube_with_eg_sites()   # looks good
    # test_lj_sphere_3d()

    # f = pp.Field.from_csv("test-lj-sphere-3d.csv", "mean")
    # f.save_image("test-lj-sphere-3d.vti")
