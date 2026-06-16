.. _measuring-fields-arrangement:

=======================================
Measuring Fields Around Multiple Bodies
=======================================

The Arrangement Class
+++++++++++++++++++++

Just as a :py:class:`~p4.body.Body` is a mapping from *particle types* to
positions and orientations, an :py:class:`~p4.arrangement.Arrangement` is a
mapping from *bodies* to positions and orientations. In HOOMD-blue, this notion
of an arrangement is expressed as the `Snapshot`_, which is equivalent to a
`GSD Frame`_.

.. _Snapshot: https://hoomd-blue.readthedocs.io/en/latest/hoomd/snapshot.html
.. _GSD Frame: https://gsd.readthedocs.io/en/latest/python-module-gsd.hoomd.html#gsd.hoomd.Frame

.. note::
    As with :ref:`body <body-orientation-note>`, an arrangement assigns default
    orientations to bodies when their orientations are not provided.

To create an arrangement, provide body definitions and positions to the
``Arrangement`` constructor. For example, you can create an arrangement with
multiple instances of a single body as follows.

.. code-block:: python

    import p4

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
    fig.show()

.. raw:: html
    :file: ../data/arrangement-1body.html

A multi-body arrangement would then be created in the following way.

.. code-block:: python

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
    fig.show()

.. raw:: html
    :file: ../data/arrangement-2bodies.html

.. note::
    As with bodies, ``type_shapes`` and ``slice`` parameters can also be passed
    to ``Arrangement.plot()``, but performance degrades quickly as
    the size of the arrangement increases. For more information, see
    :ref:`using-vtk-for-large-datasets`.


Measuring Fields Around an Arrangement
++++++++++++++++++++++++++++++++++++++

Just like ``Body``, the ``Arrangement`` class can be used as an analyte in a
``System``. Let's construct a system from our single-body arrangement above. In
this system, the probe will be a single-particle body of type "C". There will
be a repulsive potential between A and C, and an attractive potential between
B and C.

.. code-block:: python

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
    fig.show()

.. raw:: html
    :file: ../data/arrangement-abc-u-scalar3d.html


.. _using-vtk-for-large-datasets:

Future Work: Using VTK for Large Datasets
+++++++++++++++++++++++++++++++++++++++++

For fields larger than 100 x 100 x 100 and arrangements with more than a few
hundred particles, Plotly figures start to lag and slicing becomes prohibitively
slow. A future version of p4 will have ``to_vtk()`` methods so users can write
their models and field data as VTK files and use VTK software such as
`Paraview`_ or `Tomviz`_ to visualize them efficiently.

.. _Paraview: https://www.paraview.org/
.. _Tomviz: https://tomviz.org/
