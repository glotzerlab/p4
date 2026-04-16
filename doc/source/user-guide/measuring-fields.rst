.. _measuring-fields:

================
Measuring Fields
================

Together, particle models and interaction models define a *system* with a unique
set of energy and force fields. Appropriately, in p4 this combination is
expressed in the :py:class:`System` class. A p4 ``System`` stores two kinds of
bodies: a *probe* and an *analyte*. The measured fields reflect the force and
torque experienced by the *probe* from the *analyte*.

Isotropic Field
---------------

Let's define a ``System`` using the bodies and interaction that we created in
the previous sections. We will begin simply, with a single-particle probe and
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

    system.measure(
        quantities=["U", "F"],
        position_resolutions=[30, 30, 30],
        orientation_resolutions=[1, 1, 1], # <-- note: single orientation
        symmetries=[1, 1, 1],
        nlist=hoomd.md.nlist.Cell(2),
        csv_filename="doc/source/data/ac-lj.csv",   # change to fit your system
        outside_cutoff=5,
    )

:py:meth:`~p4.System.measure` writes all of the measurements into a CSV file.
To analyze and view the fields, import that CSV file using the :py:class:`Field`
class.

.. code-block:: python

    # change path to fit your system
    field = p4.Field.from_csv("doc/source/data/ac-lj.csv")

Since everything is isotropic, we only conducted measurements over a single probe
orientation, so the field can be plotted immediately.

.. code-block:: python

    fig, tr = field.plot("U")
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-u-3d.html

| 

As with ``Body``, we can plot slices of the field.

.. code-block:: python

    fig, tr = field.plot("U", slice=dict(z=0)) # note: Field auto-finds the closest value
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-u-2d.html

.. code-block:: python

    fig, tr = field.plot("U", slice=dict(z=0, x=0))
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-u-1d.html

| 

The force field can also be plotted, both as a scalar plot and a vector plot.

.. code-block:: python

    fig, tr = field.plot("F")
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-f-scalar-3d.html

.. code-block:: python

    fig, tr = field.plot("F", vectors=True)
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-f-vector-3d.html

| 

Vector plots can also be sliced into 2D (though not 1D).

.. code-block:: python

    fig, tr = field.plot("F", vectors=True, slice=dict(z=0))
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-f-vector-2d.html

| 

Individual components of the force can also be plotted as scalars.

.. code-block:: python

    fig, tr = field.plot("Fx")
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-fx-3d.html

| 

.. code-block:: python

    fig, tr = field.plot("Fx", slice=dict(z=0))
    fig.show()

.. raw:: html
    :file: ../data/ac-lj-fx-2d.html

| 