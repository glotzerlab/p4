"""Create the CSV files and plotly HTML plots for the documentation.

This script MUST be run before

1. doctests are run via Sybill
2. documentation is built with Sphinx

To satisfy #1, it is included in conftest.py

To satisfy #2, it is included in the readthedocs build process. When building
the docs locally, users must run it themselves.
"""

# %%
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


# %% ------------------------------ Basic Usage --------------------------------
# Make a simulation with two 'A' particles
example_simulation = hoomd.util.make_example_simulation(
    particle_types=["A", "B"]
)

# Create a rigid body constraint that constrains 'B' particles to a dumbbell
# configuration around 'A'.
rigid = hoomd.md.constrain.Rigid()
rigid.body["A"] = {
    "constituent_types": ["B", "B"],
    "positions": [(0,0,1),(0,0,-1)],
    "orientations": [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
}

# Add constituent particles to the simulation state
rigid.create_bodies(example_simulation.state)

# Add an integrator and a simple B-B LJ force
example_simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)
lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Tree(2), default_r_cut=0)
lj.r_cut[("B", "B")] = 5
lj.params[("A", "A")] = {"epsilon": 0.0, "sigma": 1.0}
lj.params[("A", "B")] = {"epsilon": 0.0, "sigma": 1.0}
lj.params[("B", "B")] = {"epsilon": 1.0, "sigma": 1.0}

# Attach the force and constraint to the integrator
example_simulation.operations.integrator.rigid = rigid
example_simulation.operations.integrator.forces.append(lj)

# %%
body = p4.Body.from_hoomd_simulation(example_simulation)

# %%
interactions = p4.Interaction.from_hoomd_simulation(example_simulation)

# %%
system = p4.System.from_hoomd_simulation(
    example_simulation,
    probe_primary_type="A",     # change to fit your system
    analyte_primary_type="A"    # change to fit your system
)

# %%
positions = p4.positions_on_regular_grid(
    box=[10, 10, 10],
    resolution=[10, 10, 10]
)

system.measure(
    quantities=["U"],
    positions=positions,                    # change to fit your system
    orientations=[1, 0, 0, 0],              # change to fit your system
    csv_filename=data_path+"/field.csv",    # change to fit your system
)

# %%
field = p4.Field.from_csv(data_path+"/field.csv")

# %%
average_field = field.aggregate_over_orientations("U", "mean")

# %%
fig, tr = field.plot()
fig.update_layout(**FIELD_3D_LAYOUT)
fig.write_html(data_path+"/basic-field.html", include_plotlyjs='cdn')


# %% ---------------------- User Guide: Particle Models ------------------------
body = p4.Body("A")
fig, _ = body.plot()
fig.update_layout(**BODY_3D_LAYOUT)
fig.write_html(data_path+"/body-1type.html", include_plotlyjs='cdn')

# %%
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

# %%
fig, _ = body.plot(
    type_shapes=dict(A=coxeter.shapes.ConvexPolyhedron(octahedron_vertices))
)
fig.update_layout(**BODY_3D_LAYOUT)
fig.write_html(data_path+"/body-2types-1shape.html", include_plotlyjs='cdn')

# %%
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

# %%
fig, _ = body.plot(
    slice=dict(x=0),    # <--
    type_shapes=dict(A=coxeter.shapes.ConvexPolyhedron(octahedron_vertices))
)
fig.update_layout(**BODY_PLOT_DIMENSIONS)
fig.write_html(data_path+"/body-2types-2shapes-slice2D.html", include_plotlyjs='cdn')


# %%
fig, _ = body.plot(
    slice=dict(x=0, y=0),    # <--
    type_shapes=dict(A=coxeter.shapes.ConvexPolyhedron(octahedron_vertices))
)
fig.update_layout(**BODY_PLOT_DIMENSIONS)
fig.write_html(data_path+"/body-2types-2shapes-slice1D.html", include_plotlyjs='cdn')


# %% --------------------- User Guide: Interaction Models ----------------------
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

# %%
fig, tr = interaction.plot(r=np.linspace(0, 8, 100), include_default=True)
fig.update_layout(**FIELD_PLOT_DIMENSIONS)
fig.write_html(data_path+"/basic-interaction.html", include_plotlyjs='cdn')


# %% ---------------------- User Guide: Isotropic Fields -----------------------
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

# %%
positions = p4.positions_on_regular_grid(
    box=[5, 5, 5],
    resolution=[30, 30, 30]
)

system.measure(
    quantities=["U", "F"],
    positions=positions,
    orientations=[1, 0, 0, 0], # <-- note: single orientation
    csv_filename=data_path+"/ac-lj-uf.csv",
)

# %%
field = p4.Field.from_csv(data_path+"/ac-lj-uf.csv")

# %%
fig, tr = field.plot("U")
fig.update_layout(**FIELD_3D_LAYOUT)
fig.write_html(data_path+"/ac-lj-u-3d.html", include_plotlyjs='cdn')

# %%
fig, tr = field.plot("U", slice=dict(z=0), contours=None) # note: Field auto-finds the closest value
fig.update_layout(**FIELD_PLOT_DIMENSIONS)
fig.write_html(data_path+"/ac-lj-u-2d.html", include_plotlyjs='cdn')

# %%
fig, tr = field.plot("U", slice=dict(z=0, x=0), ylim_1d=[-0.12, 0.05])
fig.update_layout(**FIELD_PLOT_DIMENSIONS)
fig.write_html(data_path+"/ac-lj-u-1d.html", include_plotlyjs='cdn')

# %%
fig, tr = field.plot("F", clim=[-0.1, 1])
fig.update_layout(**FIELD_3D_LAYOUT)
fig.write_html(data_path+"/ac-lj-f-scalar-3d.html", include_plotlyjs='cdn')

# %%
fig, tr = field.plot("F", vectors=True)
fig.update_layout(**FIELD_3D_LAYOUT)
fig.write_html(data_path+"/ac-lj-f-vector-3d.html", include_plotlyjs='cdn')

# %%
fig, tr = field.plot("F", vectors=True, slice=dict(z=0))
fig.update_layout(**FIELD_PLOT_DIMENSIONS)
fig.write_html(data_path+"/ac-lj-f-vector-2d.html", include_plotlyjs='cdn')

# %%
fig, tr = field.plot("Fx")
fig.update_layout(**FIELD_3D_LAYOUT)
fig.write_html(data_path+"/ac-lj-fx-3d.html", include_plotlyjs='cdn')

# %%
fig, tr = field.plot("Fx", slice=dict(z=0))
fig.update_layout(**FIELD_3D_LAYOUT)
fig.write_html(data_path+"/ac-lj-fx-2d.html", include_plotlyjs='cdn')


# %% --------------------- User Guide: Anisotropic Fields ----------------------
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

# %%
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

# %%
point_body = p4.Body("D")

# %%
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

# %%
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

# %%
system = p4.System(
    probe=point_body,
    analyte=cubic_body,
    interactions=[attraction, repulsion]
)
positions = p4.positions_on_regular_grid(
    box=[5, 5, 5],
    resolution=[20, 20, 20]
)
system.measure(
    quantities=["U", "F", "T"],
    positions=positions,
    orientations=[1, 0, 0, 0],
    csv_filename=data_path+"/abd-aljg-uft.csv",
)

field = p4.Field.from_csv(data_path+"/abd-aljg-uft.csv")

fig, tr = field.plot("U", clim=[-0.2, 1], fill_nan_with_inf=True)
fig.update_layout(**FIELD_3D_LAYOUT)
fig.write_html(data_path+"/abd-aljg-u-3d.html", include_plotlyjs='cdn')

# %%
_, body_tr = cubic_body.plot(
    type_shapes=dict(
        A=coxeter.shapes.ConvexPolyhedron(cube_vertices)
    )
)

fig.add_traces(body_tr)
fig.write_html(data_path+"/abd-aljg-u-3d-with-body.html", include_plotlyjs='cdn')

# %%
cubic_body2 = p4.Body(
    primary_type="C",
    secondary_types=["D"],
    positions_by_type=dict(D=cube_vertices)
)

# %%
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

# %%
system2 = p4.System(
    probe=cubic_body2,
    analyte=cubic_body,
    interactions=[attraction, repulsion2]
)
positions = p4.positions_on_regular_grid(
    box=[10, 10, 10],
    resolution=[15, 15, 15]
)
orientations = p4.orientations_from_fibonacci_lattice(n=10, group="O")
system2.measure(
    quantities=["U", "F", "T"],
    positions=positions,
    orientations=orientations,
    csv_filename=data_path+"/abcd-aljg-uft.csv",
)

# %%
field = p4.Field.from_csv(data_path+"/abcd-aljg-uft.csv")

avg_u = field.aggregate_over_orientations(quantity="U", method="mean")

fig, tr = avg_u.plot(fill_nan_with_inf=True)
fig.update_layout(**FIELD_3D_LAYOUT)
fig.add_traces(body_tr)
fig.write_html(data_path+"/abcd-aljg-avg-u-3d-with-body.html", include_plotlyjs='cdn')

# %%
avg_f = field.aggregate_over_orientations(quantity="F", method="mean")
fig, tr = avg_f.plot(slice=dict(z=0), vectors=True)
fig.update_layout(**FIELD_PLOT_DIMENSIONS)
fig.write_html(data_path+"/abcd-aljg-avg-f-vector-2d.html", include_plotlyjs='cdn')
