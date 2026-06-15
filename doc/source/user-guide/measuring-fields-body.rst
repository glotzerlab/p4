.. _measuring-fields-body:

==============================
Measuring Fields Around a Body
==============================

Together, particle models and interaction models define a *system* with a unique
set of energy and force fields. Appropriately, in p4 this combination is
expressed in the :py:class:`System` class. A p4 ``System`` stores two kinds of
bodies: a *probe* and an *analyte*. The measured fields reflect the force and
torque experienced by the *probe* from the *analyte*.

Isotropic Fields
----------------

Let's define a ``System`` using the bodies and interaction that we created in
the previous sections. We will start small, with a single-particle probe and
a single-particle analyte.

.. code-block:: python

    import p4
    import hoomd

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

``System`` is self-validating (as are ``Body`` and ``Interaction``), so it is
ready for measuring as soon as it is created. Let's measure the system's energy
and force fields. (Since the particle models and interaction model are all
isotropic, torque will be zero everywhere.)

.. code-block:: python

    data_path = "doc/source/data" # change to your own directory path

    positions = p4.positions_on_regular_grid(
        box=[5, 5, 5],
        resolution=[30, 30, 30]
    )

    field = system.measure(
        quantities=["U", "F"],
        positions=positions,
        orientations=[1, 0, 0, 0], # <-- note: single orientation
    )

.. note::
    Recall that the following options may be passed in ``quantities``:

    * ``"U"`` - potential energy
    * ``"F"`` - net force
    * ``"T"`` - net torque

    For more information, see :py:meth:`~p4.system.System.measure`.

.. warning::
    By default, ``system.measure()`` uses Python's `multiprocessing module`_
    to distribute the measurement process across the maximum number of processes
    allowed on your computer. For more information about this default behavior,
    see the :ref:`warning <measure-multiprocessing-warning>` in the method's
    documentation.

.. _multiprocessing module: https://docs.python.org/3/library/multiprocessing.html

:py:meth:`~p4.system.System.measure` returns the measurement data as a
:py:class:`Field`. Since everything is isotropic, we only conducted measurements
over a single probe orientation, and the field can be plotted immediately.

.. code-block:: python

    fig, tr = field.plot("U")
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-u-3d.html

As with ``Body``, we can plot slices of the field.

.. code-block:: python

    fig, tr = field.plot("U", slice=dict(z=0), contours=None) # note: Field auto-finds the closest value
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-u-2d.html

.. code-block:: python

    fig, tr = field.plot("U", slice=dict(z=0, x=0), ylim_1d=[-0.12, 0.05])
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-u-1d.html

The force field can also be plotted, both as a scalar plot and a vector plot.

.. code-block:: python

    fig, tr = field.plot("F", clim=[-0.1, 1])
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-f-scalar-3d.html

.. code-block:: python

    fig, tr = field.plot("F", vectors=True)
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-f-vector-3d.html

Vector plots can also be sliced into 2D (though not 1D).

.. code-block:: python

    fig, tr = field.plot("F", vectors=True, slice=dict(z=0))
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-f-vector-2d.html

Individual components of the force can also be plotted as scalars.

.. code-block:: python

    fig, tr = field.plot("Fx", clim=[-0.1, 0.1])
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-fx-3d.html

.. code-block:: python

    fig, tr = field.plot("Fx", slice=dict(z=0), clim=[-0.1, 0.1], contours=None)
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-fx-2d.html


Anisotropic Fields
------------------

Next, let's investigate the fields for a repulsive cube with attractive sites
on its vertices.

.. code-block:: python

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

The ``Body`` for this particle model will have the cube be particle type "A",
and the sites be particle type "B".

.. code-block:: python

    import coxeter
    
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
    fig.show()

.. raw:: html
    :file: ../data/body-anisotropic.html

We'll first investigate the fields experienced by a point particle probe, to
which we'll give the particle type "D".

.. code-block:: python

    point_body = p4.Body("D")

We can model the attractive interaction using an isotropic `Gaussian`_
potential with a negative ``epsilon``. The gaussian potential should only be
active for B-D pairs.

.. _Gaussian: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/pair/gaussian.html

.. code-block:: python

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

We can model the repulsive interaction using an `anisotropic Lennard-Jones`_
(ALJ) potential with ``alpha=0``. The ALJ potential should only be active for
A-D pairs.

.. _anisotropic Lennard-Jones: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/pair/aniso/alj.html

.. code-block:: python

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

Let's create the system and measure its energy, and force fields.

.. code-block:: python

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
    fig.show()

.. note::

    The ALJ potential is numerically unstable when the probe overlaps with the cube.
    This means that there are ``NaN`` values in the field. To plot it effectively,
    we overwrite those ``NaN`` values with very large numbers using the
    ``fill_nan_with_inf`` flag. We then must manually set the limits of the colorbar
    with ``clim``.

.. raw:: html
    :file: ../data/abd-aljg-u-3d.html

If you like, you can add your body plot traces to the energy plot.

.. code-block:: python

    _, body_tr = cubic_body.plot(
        type_shapes=dict(
            A=coxeter.shapes.ConvexPolyhedron(cube_vertices)
        )
    )

    fig.add_traces(body_tr)
    fig.show()

.. raw:: html
    :file: ../data/abd-aljg-u-3d-with-body.html

Next, let's investigate the fields experienced that the cubic body would
experience as a probe. Because we cannot have the same particle types in both
the probe and the analyte, we must create a new cubic body with new particle
types. The new repulsive cube be particle type "C", and the sites be particle
type "D".

.. code-block:: python

    cubic_body2 = p4.Body(
        primary_type="C",
        secondary_types=["D"],
        positions_by_type=dict(D=cube_vertices)
    )

We can re-use the attractive potential from earlier, but we must make a new
repulsive potential that is only active between the cube particle types "A" and
"C".

.. code-block:: python

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

Now we can make a new system, and this time we can measure its energy, force,
*and* torque fields. Since the probe is anisotropic, it is no longer correct
to measure at a single probe orientation, so we must specify multiple
orientations for each position.The probe has octahedral symmetry (point group
"O\ :sub:`h`"), and we can use that symmetry to reduce the number of
orientations required for accurate sampling of orientation-space.

.. code-block:: python

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
    field = system2.measure(
        quantities=["U", "F", "T"],
        positions=positions,
        orientations=orientations,
    )

.. note::
    We must double the size of the box containing the positions, because the
    effective shape of the repulsive cube (as experienced by another repulsive
    cube) is double the size of the original cube. This will be evident in the
    plots below.

.. note::
    Measuring over multiple orientations at each position quickly becomes
    computationally expensive, so HPC resources are often necessary when
    measuring anisotropic fields. When running measurement jobs on external
    systems, consider passing the ``filename`` parameter to continuously save
    measurement data to disk. This ensures that data is not lost in the case of
    exceeded walltime, service outages, etc. For more information, see
    :py:meth:`~p4.system.System.measure`.

As before, ``system2.measure()`` produces a ``Field``, but this time we
must decide how to handle the multiple orientations. The simplest thing to do
is to average over them.

.. code-block:: python
    
    avg_u = field.aggregate_over_orientations(quantity="U", method="mean")

    fig, tr = avg_u.plot(fill_nan_with_inf=True)
    fig.add_traces(body_tr)
    fig.show()

.. raw:: html
    :file: ../data/abcd-aljg-avg-u-3d-with-body.html

.. note::
    Each quantity (U, F, T) must be aggregated individually because vector
    quantities are aggregated differently from scalar quantities. See
    :py:meth:`~p4.field.Field.aggregate_over_orientations`.

.. TODO: we need to add a flag to the aggregation method that controls how it deals with nan and inf

Recall what was mentioned above: the ALJ potential is numerically unstable
and unphysical when it is evaluated for particle shape overlaps. You can see
this if you plot the forces and torques. The measurements outside of the
effective shape are accurate, but those inside are not. For example, plot
the force vectors at the slice ``z=0``, which passes through the approximate
center of the cubic body analyte.

.. code-block:: python
    
    avg_f = field.aggregate_over_orientations(quantity="F", method="mean")
    fig, tr = avg_f.plot(slice=dict(z=0), vectors=True)
    fig.show()

.. raw:: html
    :file: ../data/abcd-aljg-avg-f-vector-2d.html

The vectors inside the effective shape result from the sum of the attractive
gaussian potential and the unphysical ALJ potential.


Conclusion
----------

You have reached the end of the User Guide. In the future, another section
will be added to demonstrate more advanced uses for p4, including parameter
sweeps with `Signac`_ (``Body``, ``Interaction``, and ``System`` can be stored
in Signac statepoints) and evaluating larger and more complex systems
(support will be added for analyzing fields created by an entire simulation
state, and for exporting bodies, states, and fields to `VTK`_ for more complex
plotting and slicing).

.. _Signac: https://signac.readthedocs.io/en/latest/index.html
.. _VTK: https://docs.vtk.org/en/latest/index.html
