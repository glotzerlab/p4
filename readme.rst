p4 - Pairwise Potential Particle Probe
======================================

Probe the effective potential landscape around a rigid body of particles using
`HOOMD-blue <https://hoomd-blue.readthedocs.io/en/latest/>`_.

**p4** provides declarative wrappers around `hoomd.md.constrain.Rigid <https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/constrain/rigid.html>`_
and subclasses of `hoomd.md.pair.Pair <https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/pair/pair.html>`_,
making it easy to define *analyte* and *probe* `bodies <doc/source/api/body.rst>`_
governed by one or more `interactions <doc/source/api/interaction.rst>`_, and
then to probe the corresponding `system's <doc/source/api/system.rst>`_ potential
energy landscape. The corresponding potential energy distribution can then be
analyzed, sliced, and plotted using **p4**'s `Field <doc/source/api/field.rst>`_ API.

++++++++++++
Installation
++++++++++++

Clone the repository, navigate to the package's root directory, and run `pip install .`

+++++++++++++
Example usage
+++++++++++++

Probe the potential around a LJ sphere

.. code-block::

    import p4

    probe = p4.Body("A")
    analyte = p4.Body("A")

    interactions = [
        p4.Interaction(
            hoomd_class=hoomd.md.pair.LJ,
            initial_args=dict(),
            no_params=dict(
                r_cut=0,
                params=dict(
                    epsilon=0,
                    sigma=1
                )
            ),
            all_types=["A"],
            yes_params={
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

    system.probe_potential(
        position_resolutions=[30, 30, 1],
        orientation_resolutions=[1, 1, 1],
        symmetries=[1, 1, 1],
        nlist=nlist,
        csv_filename="lj-sphere.csv",
        outside_cutoff=2.0,
    )

    field = p4.Field.from_csv("lj-sphere.csv", "mean")
    field.plot()

.. image:: ../assets/lj-sphere-plot.png


Probe the potential around an attractive and repulsive ALJ cube

.. code-block::

    import p4

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


    probe = p4.Body("A")
    analyte = p4.Body("B")

    interactions = [
        p4.Interaction(
            hoomd_class=hoomd.md.pair.aniso.ALJ,
            initial_args=dict(),
            no_params=dict(
                r_cut=0,
                params=dict(
                    epsilon=0,
                    sigma_i=1,
                    sigma_j=1,
                    alpha=0
                )
            ),
            all_types=["A", "B"],
            yes_params={
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
                        vertices=get_cube_vertices(2),
                        faces=get_cube_faces()
                    )
                )
            }
        )
    ]

    system = p4.System(probe, analyte, interactions)

    nlist = hoomd.md.nlist.Cell(10)

    system.probe_potential(
        position_resolutions=[50, 50, 1],
        orientation_resolutions=[1, 1, 10],
        symmetries=[1, 1, 4],
        nlist=nlist,
        csv_filename="alj-cube.csv",
        outside_cutoff=10,
        n_processes=-1
    )

    field = p4.Field.from_csv("alj-cube.csv", "mean")
    field.plot(fill_nan_with_inf=True)

.. image:: ../assets/alj-cube-plot.png

Probe the potential around a repulsive ALJ cube with attractive gaussian sites

.. code-block::

    import p4

    probe = p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(B=get_cube_vertices(2))
    )
    analyte = p4.Body(
        primary_type="C",
        secondary_types=["D"],
        positions_by_type=dict(D=get_cube_vertices(2))
    )

    interactions = [
        p4.Interaction(
            hoomd_class=hoomd.md.pair.aniso.ALJ,
            initial_args=dict(),
            no_params=dict(
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
            all_types=["A", "B", "C", "D"],
            yes_params={
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
                        vertices=get_cube_vertices(2),
                        faces=get_cube_faces()
                    )
                ),
                "C": dict(
                    shape=dict(
                        vertices=get_cube_vertices(2),
                        faces=get_cube_faces()
                    )
                )
            }
        ),
        p4.Interaction(
            hoomd_class=hoomd.md.pair.Gaussian,
            initial_args=dict(),
            no_params=dict(
                r_cut=0,
                params=dict(
                    epsilon=0,
                    sigma=1,
                )
            ),
            all_types=["A", "B", "C", "D"],
            yes_params={
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

    system.probe_potential(
        position_resolutions=[100, 100, 1],
        orientation_resolutions=[1, 1, 10],
        symmetries=[1, 1, 4],
        nlist=nlist,
        csv_filename="alj-cube-with-gauss-sites.csv",
        outside_cutoff=10,
        n_processes=-1,
    )

    field = p4.Field.from_csv("alj-cube-with-gauss-sites.csv", "mean")
    field.plot(fill_nan_with_inf=True)

.. image:: ../assets/alj-cube-gauss-sites-plot.png

+++++
To-Do
+++++

`Interaction`
- [x] Allow the interaction model to specify different parameters for different interacting type pairs
  ~~- [ ] Add support for HPMC potentials (contributes to goal of writing a simulation parser)~~
    ~~- to accomplish this, I think  `InteractionModel` would need to be its own class that checks to make sure every constituent `Interaction` has the same  simulation type (MD or HPMC). Ideally, it wouldn't matter (see HOOMD-rs), but for HOOMD-blue it certainly does.~~
- [x] write tests
- [x] add parsing from `hoomd.md.pair.Pair` subclasses

`Body`
- [x] write tests
- [x] add parsing from `hoomd.md.constraint.Rigid`, ~~`hoomd.Snapshot`~~, and `hoomd.Simulation`
- [ ] add a basic plotting method

`System`
- [ ] add a custom table writer to prevent memory issues if the number of points gets too large
- [ ] add one or more methods for improving the sampling of position and orientation-space
  1. Calculate an array of positions and orientations that do not produce effective overlaps between provided shapes
     - have high orientation resolution (e.g. 10)
     - before the run_probe step, at each position and orientation:
      - check if the probe and analyte would overlap
      - if yes, remove the orientation
      - tally the number of rejected orientations - if the number of accepted ones falls below some threshold (e.g. 2), then remove that position
  2. Dynamical meshing during simulation (c.f. ChIMES-informed ML anisotropic potential)
     - Note that this would affect and/or depend on the implementation of the custom table writer
- [ ] (optional) remove need to provide separate probe body model to define a system, generating a suitable particle on-the-fly during the probe simulation with 1+ types that depend on the provided `yes_types`
  - to accomplish this, need to change how index of probe particle is calculated. Does that index ever change?
  - PROBLEM: what if you aren't interested in body A - body A interactions, but rather in body A - body B? How do you investigate those with no notion of a probe particle? The notion of a System would have to change - it would be a collection of named particle models and a single interaction model (Or maybe a combination of a Particle Model and an Interaction Model - a ParticleModel being a named collection of BodyModels...)
- [ ] (optional) add support for Frame analytes
- [x] write tests
- [x] add parsing from `hoomd.Simulation`

`Field`
- [ ] refine dimensionality handling (always 3D? if not, need checks when calling various methods)
- [ ] method for saving to npz
- [ ] methods or overloads for adding/overlaying multiple fields to create a new one
- [ ] write tests

`util`
  - [ ] functions for saving/creating interaction models to/from json


Other
  - [ ] double-check all indexing/array orientation logic
  - [ ] Allow user to probe systems with an analyte frame (with multiple particles), rather than just a single analyte particle. This could involve re-structuring the API to something like
      - Probe class
          - contains probe particle model and probing simulation logic
      - System class
          - contains analyte particle models (can be more than one now) as well as a reference frame of positions and orientations to reset to after every "time step"


## Roadmap

### Open to group members
functionality
  - [x] parse hoomd objects to create Interaction, ParticleModel, System
  - [ ] plot 1D and 2D slices of Fields

tests
  - [ ] non-validation coverage 100%

documentation
  - [ ] API + 1 example

### Open to public
TBD

### Publication
TBD

### 'Completion'
TBD
