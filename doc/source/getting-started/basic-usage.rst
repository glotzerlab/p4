.. _basic usage:

===========
Basic Usage
===========

Starting from a HOOMD simulation
--------------------------------

If you already have an existing HOOMD-blue MD simulation, you can parse it to
extract its rigid constraint as one or more :py:class:`Bodies <p4.Body>` using 

.. code:: python

    bodies = p4.Body.from_hoomd_simulation(<your_simulation>)

and its pairwise forces as one or more :py:class:`Interactions <p4.Interaction>`
using

.. code:: python

    interactions = p4.Interaction.from_hoomd_simulation(<your_simulation>)

These objects can then be printed to show their primary and secondary types
(in the case of ``bodies``) and their interacting types (in the case of
``interactions``).

If you already know the types of your particles of interest, you can parse a
simulation directly into a :py:class:`~p4.System` using

.. code:: python

    system = p4.System.from_hoomd_simulation(<your_simulation>)

Because all of these classes are self-validating, the ``system`` is ready
to be probed as soon as it is created. You can measure its potential energy field
using

.. code:: python

    system.measure(
        quantities=["U"],
        position_resolutions=<your_spatial_resolution>,
        orientation_resolutions=<your_orientation_resolution>,
        symmetries=<your_body_symmetry>,
        nlist=<your_nlist>,
        csv_filename="field.csv",
        outside_cutoff=<your_cutoff>,
    )

This method saves a csv file that records the potential energy at each position
and orientation under the name "field.csv" in the current directory. You can
then visualize this field using the :py:class:`~p4.Field` class.

.. code:: python

    field = p4.Field.from_csv("field.csv")

If you measured the potential energy at multiple orientations, you must
aggregate the field's quantities over those orientations before plotting.

.. code:: python

    average_field = p4.Field.aggregate_over_orientations("U", "mean")

The ``"mean"`` option causes the method to average the potential energy at each
position over all orientations. Other options are available---see
:py:meth:`~p4.Field.aggregate_over_orientations`.

The potential energy can be plotted as a 3D volume, or as a slice in 2D or
1D---see :py:meth:`~p4.Field.plot`.

.. code:: python
    
    field.plot()
