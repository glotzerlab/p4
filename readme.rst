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

To install **p4**, clone the repository, navigate to the package's root directory, and run

.. code-block::

    pip install .

To build the documentation, install the requirements, navigate to the package's ``doc`` directory, and run

.. code-block::

    make html

+++++++++++++
Example usage
+++++++++++++

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Probe the potential around a LJ sphere
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    import p4

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

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Probe the potential around an attractive and repulsive ALJ cube
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Probe the potential around a repulsive ALJ cube with attractive gaussian sites
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
