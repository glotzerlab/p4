"""Create the CSV files and plotly HTML plots for the documentation.

This script MUST be run before

1. doctests are run via Sybill
2. documentation is built with Sphinx

To satisfy #1, it is included in conftest.py

To satisfy #2, it is included in the readthedocs build process. When building
the docs locally, users must run it themselves.
"""

# --

import coxeter
import rowan

import p4
import hoomd
import numpy as np
from pathlib import Path

BODY_PLOT_DIMENSIONS = dict(width=300, height=300)
FIELD_PLOT_DIMENSIONS = dict(width=500, height=500)

BODY_3D_LAYOUT = dict(
    scene_camera=dict(eye=dict(x=2, y=2, z=2)), # Adjust x, y, z to zoom out
    **BODY_PLOT_DIMENSIONS
)
FIELD_3D_LAYOUT = dict(
    scene_camera=dict(eye=dict(x=2, y=2, z=2)), # Adjust x, y, z to zoom out
    **FIELD_PLOT_DIMENSIONS
)

data_path = str(Path(__file__).parent / "source/data")

# Ensure data path points to an existing directory
Path(data_path).mkdir(parents=True, exist_ok=True)


# Protect all calls to System.measure in an if name main block to ensure that
# multiprocessing doesn't cause problems
if __name__ == "__main__":

    # ---------------------------- Quickstart: Section 1 -----------------------
    cube = coxeter.families.PlatonicFamily.get_shape("Cube")

    body = p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(B=cube.vertices)
    )

    # --
    fig, _ = body.plot(type_shapes=dict(A=cube))
    fig.update_layout(**BODY_3D_LAYOUT)
    fig.write_html(data_path+"/quickstart-section1-body.html", include_plotlyjs='cdn')

    # --
    interaction = p4.Interaction(
        hoomd_class=hoomd.md.pair.LJ,
        initial_args={},
        default_params=dict(r_cut=0, params=dict(epsilon=0, sigma=1)),
        typed_params={
            ("B", "B"): dict(
                r_cut=5,
                params=dict(epsilon=0.3, sigma=1.0)
            )
        }
    )

    # --
    fig, _ = interaction.plot(r=np.linspace(0, 3, 1000), ylim=[-1, 1], show_legend=True)
    fig.update_layout(**FIELD_PLOT_DIMENSIONS)
    fig.write_html(data_path+"/quickstart-section1-interaction.html", include_plotlyjs='cdn')

    # --
    system = p4.System(
        probe=body,
        analyte=body,
        interactions=[interaction]
    )

    # --
    positions = p4.positions_on_regular_grid(
        box=[5, 5, 5],
        resolution=[30, 30, 30]
    )

    orientations = p4.orientations_from_fibonacci_lattice(50)

    field = system.measure(
        quantities=["U"],
        positions=positions,
        orientations=orientations,
    )

    # --
    avg = field.aggregate_over_orientations("U", "mean")
    fig, _ = avg.plot(opacityscale_scalar_3d="max")
    fig.update_layout(**FIELD_3D_LAYOUT)
    fig.write_html(data_path+"/quickstart-section1-field.html", include_plotlyjs='cdn')

    # ---------------------------- Quickstart: Section 2 -----------------------
    cube = coxeter.families.PlatonicFamily.get_shape("Cube")
    cube.volume = cube.volume / 4

    # Make a simulation with two 'A' particles
    simulation = hoomd.util.make_example_simulation(
        particle_types=["A", "B"]
    )

    # Create a rigid body constraint that constrains 'B' particles to a dumbbell
    # configuration around 'A'.
    rigid = hoomd.md.constrain.Rigid()
    rigid.body["A"] = {
        "constituent_types": ["B" for _ in cube.vertices],
        "positions": cube.vertices,
        "orientations": [(1, 0, 0, 0) for _ in cube.vertices],
    }

    # Add constituent particles to the simulation state
    rigid.create_bodies(simulation.state)

    # Add an integrator and a simple B-B LJ force
    simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Tree(2, exclusions=("body",)), default_r_cut=0)
    lj.r_cut[("B", "B")] = 5
    lj.params[("A", "A")] = {"epsilon": 0.0, "sigma": 1.0}
    lj.params[("A", "B")] = {"epsilon": 0.0, "sigma": 1.0}
    lj.params[("B", "B")] = {"epsilon": 0.3, "sigma": 1.0}

    # Attach the force and constraint to the integrator
    simulation.operations.integrator.rigid = rigid
    simulation.operations.integrator.forces.append(lj)

    # --
    fig, tr = p4.plot_state(simulation)
    fig.update_layout(**BODY_3D_LAYOUT)
    fig.write_html(data_path+"/quickstart-section2-state.html", include_plotlyjs='cdn')

    # --
    interactions = p4.Interaction.from_hoomd_simulation(simulation)
    fig, _ = interactions[0].plot(r=np.linspace(0, 3, 1000), ylim=[-1, 1], show_legend=True)
    fig.update_layout(**FIELD_PLOT_DIMENSIONS)
    fig.write_html(data_path+"/quickstart-section2-interaction.html", include_plotlyjs='cdn')

    # --
    bodies = p4.Body.from_hoomd_simulation(simulation)
    fig, _ = bodies[0].plot(type_shapes=dict(A=cube))
    fig.update_layout(**BODY_3D_LAYOUT)
    fig.write_html(data_path+"/quickstart-section2-body.html", include_plotlyjs='cdn')

    # ------------------------- User Guide: Particle Models --------------------
    body = p4.Body("A")
    fig, _ = body.plot()
    fig.update_layout(**BODY_3D_LAYOUT)
    fig.write_html(data_path+"/body-1type.html", include_plotlyjs='cdn')

    # --
    octahedron_vertices = np.array([
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0, -1],
        [ 0,  0,  1],
        [ 0, -1,  0],
        [ 1,  0,  0]
    ])
    body = p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(B=octahedron_vertices)
    )
    fig, _ = body.plot()
    fig.update_layout(**BODY_3D_LAYOUT)
    fig.write_html(data_path+"/body-2types.html", include_plotlyjs='cdn')

    # --
    fig, _ = body.plot(
        type_shapes=dict(A=coxeter.shapes.ConvexPolyhedron(octahedron_vertices))
    )
    fig.update_layout(**BODY_3D_LAYOUT)
    fig.write_html(data_path+"/body-2types-1shape.html", include_plotlyjs='cdn')

    # --
    body2 = p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(B=octahedron_vertices),
        orientations_by_type=dict(B=rowan.random.rand(len(octahedron_vertices)))
    )
    fig, _ = body2.plot(
        type_shapes=dict(
            A=coxeter.shapes.ConvexPolyhedron(octahedron_vertices),
            B=coxeter.shapes.ConvexPolyhedron(0.35 * octahedron_vertices)
        ),
        show_legend=False
    )
    fig.update_layout(**BODY_3D_LAYOUT)
    fig.write_html(data_path+"/body-2types-2shapes.html", include_plotlyjs='cdn')

    # --
    fig, _ = body.plot(
        slice=dict(x=0),    # <--
        type_shapes=dict(A=coxeter.shapes.ConvexPolyhedron(octahedron_vertices))
    )
    fig.update_layout(**BODY_PLOT_DIMENSIONS)
    fig.write_html(data_path+"/body-2types-2shapes-slice2D.html", include_plotlyjs='cdn')


    # --
    fig, _ = body.plot(
        slice=dict(x=0, y=0),    # <--
        type_shapes=dict(A=coxeter.shapes.ConvexPolyhedron(octahedron_vertices))
    )
    fig.update_layout(**BODY_PLOT_DIMENSIONS)
    fig.write_html(data_path+"/body-2types-2shapes-slice1D.html", include_plotlyjs='cdn')


    # ------------------------ User Guide: Interaction Models ------------------
    interaction = p4.Interaction(
        hoomd_class=hoomd.md.pair.LJ,
        initial_args=dict(),
        default_params=dict(
            r_cut=0,
            params=dict(epsilon=0, sigma=1)
        ),
        typed_params={
            ("A", "C"): dict(
                r_cut=5.0,
                params=dict(epsilon=0.1, sigma=1)
            )
        }
    )

    # --
    fig, tr = interaction.plot(r=np.linspace(0, 8, 100), show_default=True)
    fig.update_layout(**FIELD_PLOT_DIMENSIONS)
    fig.write_html(data_path+"/basic-interaction.html", include_plotlyjs='cdn')


    # ------------------------- User Guide: Isotropic Fields -------------------
    interaction = p4.Interaction(
        hoomd_class=hoomd.md.pair.LJ,
        initial_args=dict(),
        default_params=dict(
            r_cut=0,
            params=dict(epsilon=0, sigma=1)
        ),
        typed_params={
            ("A", "C"): dict(
                r_cut=5.0,
                params=dict(epsilon=0.1, sigma=1)
            )
        }
    )

    system = p4.System(
        probe=p4.Body("A"),
        analyte=p4.Body("C"),
        interactions=[interaction]
    )

    # --
    positions = p4.positions_on_regular_grid(
        box=[5, 5, 5],
        resolution=[30, 30, 30]
    )

    field = system.measure(
        quantities=["U", "F"],
        positions=positions,
        orientations=[1, 0, 0, 0], # <-- note: single orientation
    )

    # --
    fig, tr = field.plot("U")
    fig.update_layout(**FIELD_3D_LAYOUT)
    fig.write_html(data_path+"/ac-lj-u-3d.html", include_plotlyjs='cdn')

    # --
    fig, tr = field.plot("U", slice=dict(z=0), contours=None) # note: Field auto-finds the closest value
    fig.update_layout(**FIELD_PLOT_DIMENSIONS)
    fig.write_html(data_path+"/ac-lj-u-2d.html", include_plotlyjs='cdn')

    # --
    fig, tr = field.plot("U", slice=dict(z=0, x=0), ylim_1d=[-0.12, 0.05])
    fig.update_layout(**FIELD_PLOT_DIMENSIONS)
    fig.write_html(data_path+"/ac-lj-u-1d.html", include_plotlyjs='cdn')

    # --
    fig, tr = field.plot("F", clim=[-0.1, 1])
    fig.update_layout(**FIELD_3D_LAYOUT)
    fig.write_html(data_path+"/ac-lj-f-scalar-3d.html", include_plotlyjs='cdn')

    # --
    fig, tr = field.plot("F", vectors=True)
    fig.update_layout(**FIELD_3D_LAYOUT)
    fig.write_html(data_path+"/ac-lj-f-vector-3d.html", include_plotlyjs='cdn')

    # --
    fig, tr = field.plot("F", vectors=True, slice=dict(z=0))
    fig.update_layout(**FIELD_PLOT_DIMENSIONS)
    fig.write_html(data_path+"/ac-lj-f-vector-2d.html", include_plotlyjs='cdn')

    # --
    fig, tr = field.plot("Fx", clim=[-0.1, 0.1])
    fig.update_layout(**FIELD_3D_LAYOUT)
    fig.write_html(data_path+"/ac-lj-fx-3d.html", include_plotlyjs='cdn')

    # --
    fig, tr = field.plot("Fx", slice=dict(z=0), clim=[-0.1, 0.1], contours=None)
    fig.update_layout(**FIELD_3D_LAYOUT)
    fig.write_html(data_path+"/ac-lj-fx-2d.html", include_plotlyjs='cdn')


    # ------------------------ User Guide: Anisotropic Fields ------------------
    cube_vertices = [
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1]
    ]
    cube_faces = [
        [0, 2, 6, 4],
        [0, 4, 5, 1],
        [4, 6, 7, 5],
        [0, 1, 3, 2],
        [2, 3, 7, 6],
        [1, 5, 7, 3],
    ]

    # --
    cubic_body = p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(B=cube_vertices)
    )

    fig, tr = cubic_body.plot(
        type_shapes=dict(
            A=coxeter.shapes.ConvexPolyhedron(cube_vertices)
        )
    )
    fig.update_layout(**BODY_3D_LAYOUT)
    fig.write_html(data_path+"/body-anisotropic.html", include_plotlyjs='cdn')

    # --
    point_body = p4.Body("D")

    # --
    attraction = p4.Interaction(
        hoomd_class=hoomd.md.pair.Gaussian,
        initial_args=dict(),
        default_params=dict(
            r_cut=0,
            params=dict(epsilon=0, sigma=1)
        ),
        typed_params={
            ("B", "D"): dict(
                r_cut=5.0,
                params=dict(epsilon=-1, sigma=0.2)
            )
        }
    )

    # --
    repulsion = p4.Interaction(
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(),
        default_params=dict(
            r_cut=0,
            params=dict(epsilon=0, sigma_i=0.2, sigma_j=0.2, alpha=0),
            shape=dict(vertices=[], faces=[])
        ),
        typed_params={
            ("A", "D"): dict(
                r_cut=5.0,
                params=dict(epsilon=0.1, sigma_i=0.2, sigma_j=0.2, alpha=0)
            ),
            "A": dict(shape=dict(vertices=cube_vertices, faces=cube_faces))
        }
    )

    # --
    system = p4.System(
        probe=point_body,
        analyte=cubic_body,
        interactions=[attraction, repulsion]
    )
    positions = p4.positions_on_regular_grid(
        box=[5, 5, 5],
        resolution=[20, 20, 20]
    )
    field = system.measure(
        quantities=["U", "F", "T"],
        positions=positions,
        orientations=[1, 0, 0, 0],
    )

    fig, tr = field.plot("U", clim=[-0.2, 1], fill_nan_with_inf=True)
    fig.update_layout(**FIELD_3D_LAYOUT)
    fig.write_html(data_path+"/abd-aljg-u-3d.html", include_plotlyjs='cdn')

    # --
    _, body_tr = cubic_body.plot(
        type_shapes=dict(
            A=coxeter.shapes.ConvexPolyhedron(cube_vertices)
        )
    )

    fig.add_traces(body_tr)
    fig.write_html(data_path+"/abd-aljg-u-3d-with-body.html", include_plotlyjs='cdn')

    # --
    cubic_body2 = p4.Body(
        primary_type="C",
        secondary_types=["D"],
        positions_by_type=dict(D=cube_vertices)
    )

    # --
    repulsion2 = p4.Interaction(
        hoomd_class=hoomd.md.pair.aniso.ALJ,
        initial_args=dict(),
        default_params=dict(
            r_cut=0,
            params=dict(epsilon=0, sigma_i=0.2, sigma_j=0.2, alpha=0),
            shape=dict(vertices=[], faces=[])
        ),
        typed_params={
            ("A", "C"): dict(
                r_cut=5.0,
                params=dict(epsilon=0.1, sigma_i=0.2, sigma_j=0.2, alpha=0)
            ),
            "A": dict(shape=dict(vertices=cube_vertices, faces=cube_faces)),
            "C": dict(shape=dict(vertices=cube_vertices, faces=cube_faces))
        }
    )

    # --
    system2 = p4.System(
        probe=cubic_body2,
        analyte=cubic_body,
        interactions=[attraction, repulsion2]
    )
    positions = p4.positions_on_regular_grid(
        box=[10, 10, 10],
        resolution=[20, 20, 20]
    )
    orientations = p4.orientations_from_fibonacci_lattice(n=100)
    field = system2.measure(
        quantities=["U", "F", "T"],
        positions=positions,
        orientations=orientations,
    )

    # --
    avg_u = field.aggregate_over_orientations(quantity="U", method="mean")

    fig, tr = avg_u.plot(fill_nan_with_inf=True, cmap="PuBU_r", clim=[-1e-2, 0])
    fig.update_layout(**FIELD_3D_LAYOUT)
    fig.add_traces(body_tr)
    fig.write_html(data_path+"/abcd-aljg-avg-u-3d-with-body.html", include_plotlyjs='cdn')

    # --
    avg_f = field.aggregate_over_orientations(quantity="F", method="mean")
    fig, tr = avg_f.plot(slice=dict(z=0), vectors=True)
    fig.update_layout(**FIELD_PLOT_DIMENSIONS)
    fig.write_html(data_path+"/abcd-aljg-avg-f-vector-2d.html", include_plotlyjs='cdn')

   # ------------------------- User Guide: Arrangements ------------------------
    arrangement = p4.Arrangement(
        bodies=[
            p4.Body(
                primary_type="A",
                secondary_types=["B"],
                positions_by_type=dict(B=[[0, 0, -1], [0, 0, 1]])
            )
        ],
        positions_by_type=dict(
            A=[
                [-1, -1, 0],
                [-1,  1, 0],
                [ 1, -1, 0],
                [ 1,  1, 0]
            ]
        )
    )

    fig, _ = arrangement.plot()
    fig.update_layout(**BODY_3D_LAYOUT)
    fig.write_html(data_path+"/arrangement-1body.html", include_plotlyjs='cdn')

    # --
    arrangement2 = p4.Arrangement(
        bodies=[
            p4.Body(
                primary_type="A",
                secondary_types=["B"],
                positions_by_type=dict(B=[[0, 0, -1], [0, 0, 1]])
            ),
            p4.Body(
                primary_type="C",
            )
        ],
        positions_by_type=dict(
            A=[
                [-1, -1, 0],
                [ 1,  1, 0]
            ],
            C=[
                [-1,  1, 0],
                [ 1, -1, 0],
            ],
        )
    )

    fig, _ = arrangement2.plot()
    fig.update_layout(**BODY_3D_LAYOUT)
    fig.write_html(data_path+"/arrangement-2bodies.html", include_plotlyjs='cdn')

    # --
    system = p4.System(
        probe=p4.Body("C"),
        analyte=arrangement,
        interactions=[
            # A-C repulsion
            p4.Interaction(
                hoomd_class=hoomd.md.pair.Gaussian,
                initial_args={},
                default_params=dict(
                    r_cut=0,
                    params=dict(epsilon=0, sigma=1)
                ),
                typed_params={
                    ("A", "C"): dict(
                        r_cut=4,
                        params=dict(epsilon=10, sigma=0.5)
                    )
                }
            ),

            # B-C attraction
            p4.Interaction(
                hoomd_class=hoomd.md.pair.Gaussian,
                initial_args={},
                default_params=dict(
                    r_cut=0,
                    params=dict(epsilon=0, sigma=1)
                ),
                typed_params={
                    ("B", "C"): dict(
                        r_cut=4,
                        params=dict(epsilon=-1, sigma=1)
                    )
                }
            ),
        ]
    )

    positions = p4.positions_on_regular_grid(
        box=[6, 6, 6],
        resolution=[30, 30, 30]
    )

    field = system.measure(
        quantities=["U"],
        positions=positions,
        orientations=[1, 0, 0, 0],
        n_processes=2
    )

    _, tr = arrangement.plot()
    fig, _ = field.plot(clim=[-1, 1], opacityscale_scalar_3d="extremes")
    fig.add_traces(tr)
    fig.update_layout(**FIELD_3D_LAYOUT)
    fig.write_html(data_path+"/arrangement-abc-u-scalar3d.html", include_plotlyjs='cdn')

    # --------------------------- User Guide: Sampling -------------------------
    
    cube = coxeter.families.PlatonicFamily.get_shape("Cube")

    body = p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(B=cube.vertices)
    )

    sample_positions = p4.positions_on_regular_grid(
        box=[2, 2, 2],
        resolution=[10, 10, 10]
    )

    _, body_trace = body.plot(type_shapes=dict(A=cube))
    fig, _ = p4.plot_positions(positions=sample_positions, color="black", size=2)
    fig.add_traces(body_trace)
    fig.write_html(data_path+"/sampling-positions.html", include_plotlyjs='cdn')

    # ---

    other_body = p4.Body(
        primary_type="C",
        secondary_types=["D"],
        positions_by_type=dict(D=cube.vertices)
    )

    gauss = p4.Interaction(
        hoomd_class=hoomd.md.pair.Gaussian,
        initial_args={},
        default_params=dict(
            r_cut=0,
            params=dict(epsilon=0, sigma=0.2),
        ),
        typed_params={
            ("D", "B"): dict(
                r_cut=5,
                params=dict(epsilon=-1, sigma=0.125)
            )
        }
    )

    system = p4.System(
        probe=other_body,
        analyte=body,
        interactions=[gauss]
    )

    fig, _ = p4.plot_field_vs_number_of_orientations(
        system=system,
        n_orientations=np.arange(100, 600, 25),
        positions=[[float(i)]*3 for i in np.linspace(1, 1.25, 6)],
        method="fibonacci_lattice",
    )
    fig.write_html(data_path+"/sampling-orientations.html", include_plotlyjs='cdn')
