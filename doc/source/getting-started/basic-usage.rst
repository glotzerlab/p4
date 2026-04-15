.. _basic usage:

===========
Basic Usage
===========

Starting from a HOOMD simulation
--------------------------------

If you already have an existing HOOMD-blue MD simulation, you can parse it
into most of the major classes. If you do not have an existing simulation but
wish to follow along to this quickstart guide, you can quickly create an
example simulation using the code below.

.. code:: python

    import p4
    import hoomd

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

    # Add an integrator and a simple B-B LJ force
    example_simulation.operations.integrator = hoomd.md.Integrator(dt=0.1)
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(2), default_r_cut=0)
    lj.r_cut[("B", "B")] = 5
    lj.params[("A", "A")] = {"epsilon": 0.0, "sigma": 1.0}
    lj.params[("A", "B")] = {"epsilon": 0.0, "sigma": 1.0}
    lj.params[("B", "B")] = {"epsilon": 1.0, "sigma": 1.0}

    # Attach the force and constraint to the integrator
    example_simulation.operations.integrator.rigid = rigid
    example_simulation.operations.integrator.forces.append(lj)

You can directly parse your simulation extract its rigid constraint as one or
more :py:class:`~p4.Body`.

.. code:: python

    body = p4.Body.from_hoomd_simulation(example_simulation)

Likewise, you can parse your simulation's pairwise forces as one or more
:py:class:`~p4.Interaction`.

.. code:: python

    interactions = p4.Interaction.from_hoomd_simulation(example_simulation)

These objects can then be printed to show their primary and secondary types
(in the case of ``bodies``) and their interacting types (in the case of
``interactions``).

If you already know the types of your particles of interest, you can parse a
simulation directly into a :py:class:`~p4.System` using

.. code:: python

    system = p4.System.from_hoomd_simulation(
        example_simulation,
        probe_primary_type="A",     # change to fit your system
        analyte_primary_type="A"    # change to fit your system
    )

Because all of these classes are self-validating, the ``system`` is ready
to be probed as soon as it is created. You can measure its potential energy field
using

.. code:: python

    system.measure(
        quantities=["U"],
        position_resolutions=[10, 10, 10],  # change to fit your system
        orientation_resolutions=[1, 1, 1],  # change to fit your system
        symmetries=[1, 1, 1],               # change to fit your system
        nlist=hoomd.md.nlist.Cell(2),
        csv_filename="field.csv",
        outside_cutoff=10,                  # change to fit your system
    )

This method saves a csv file that records the potential energy at each position
and orientation under the name "field.csv" in the current directory. You can
then visualize this field using :py:class:`~p4.Field`.

.. code:: python

    # change path to fit your system
    field = p4.Field.from_csv("doc/source/data/field.csv")  

If you measured the potential energy at multiple orientations, you must
aggregate the field's quantities over those orientations before plotting.

.. code:: python

    average_field = field.aggregate_over_orientations("U", "mean")

The ``"mean"`` option causes the method to average the potential energy at each
position over all orientations. Other options are available---see
:py:meth:`~p4.Field.aggregate_over_orientations`.

The potential energy can be plotted as a 3D volume, or as a slice in 2D or
1D---see :py:meth:`~p4.Field.plot`.

.. code:: python
    
    field.plot()

.. raw:: html
    :file: ../data/field.html
