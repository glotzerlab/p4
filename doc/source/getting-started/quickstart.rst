.. _quickstart:

==========
Quickstart
==========

Using p4 to develop a patchy particle model
-------------------------------------------

Let's develop a model of a patchy particle shaped like a cube, using interaction
sites at the cube's vertices that can interact with each other through a
Lennard-Jones potential.

First, construct your particle model using the :py:class:`~p4.body.Body` class.
A ``Body`` is a collection of particle types, with one "primary" type located at
the body's center, and any number of other "secondary" types placed at specified
positions and orientations around the primary type. For more information, see
:ref:`specifying-particle-models`.

.. code-block:: python

    import p4
    import coxeter
    import hoomd

    cube = coxeter.families.PlatonicFamily.get_shape("Cube")

    body = p4.Body(
        primary_type="A",
        secondary_types=["B"],
        positions_by_type=dict(B=cube.vertices)
    )

Plot the body to examine its geometry.

.. code-block:: python

    fig, _ = body.plot(type_shapes=dict(A=cube))
    fig.show()

.. raw:: html
    :file: ../data/quickstart-section1-body.html

Next, construct your interaction model using the
:py:class:`~p4.interaction.Interaction` class. An ``Interaction`` is essentially
a python dictionary containing all the information needed to create and
parameterize a HOOMD-blue `MD pairwise force`_ for any number of interacting
particle types. For more information, see :ref:`specifying-interaction-models`.

.. _MD pairwise force: https://hoomd-blue.readthedocs.io/en/v7.0.1/hoomd/md/module-pair.html

.. code-block:: python

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

Plot the interaction to examine the shape of its potential curve.

.. code-block:: python

    fig, _ = interaction.plot(r=np.linspace(0, 3, 1000), ylim=[-1, 1], show_legend=True)
    fig.show()

.. raw:: html
    :file: ../data/quickstart-section1-interaction.html

The steep repulsive section of the potential curve ends at about 1, which means
that your patchy cube will have an effective side length of 3 (the original cube
has a side length of 1). To verify this, combine your body and interaction
together to construct a :py:class:`~p4.system.System`. A ``System`` is a
combination of a probe ``Body``, an analyte ``Body`` or
:py:class:`~p4.arrangement.Arrangement`, and a list of ``Interaction`` objects
that define how the particles in the bodies can interact. For more information,
see :ref:`measuring-fields-body`.

.. code-block:: python

    system = p4.System(
        probe=body,
        analyte=body,
        interactions=[interaction]
    )

A system has unique energy and force fields, which describe the
effective action experienced by the probe body as it is moved to various
positions and orientations around the static analyte. Measure the potential
energy field for the system. (This will take several minutes.)

.. code-block:: python

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

.. note::
    The ``quantities`` argument determines what you measure. You can pass one or
    more of the following options:

    * ``"U"`` - potential energy
    * ``"F"`` - net force
    * ``"T"`` - net torque

    For more information about these quantities, see
    :py:meth:`~p4.system.System.measure`.

This method returns a :py:class:`~p4.field.Field`, which has a plotting method
for visualizing the measured quantities. The potential energy can be plotted as
a 3D volume, or as a slice in 2D or 1D---for more information, see
:py:meth:`~p4.field.Field.plot`.

.. note::
    
    Since we measured the potential energy at multiple orientations, we must
    first aggregate the field's quantities over those orientations before
    plotting. For more information, see
    :py:meth:`~p4.field.Field.aggregate_over_orientations`.

.. code-block:: python

    avg = field.aggregate_over_orientations("U", "mean")
    fig, _ = avg.plot(opacityscale_scalar_3d="max")
    fig.show()

.. raw:: html
    :file: ../data/quickstart-section1-field.html

Using p4 to visualize simulation components
-------------------------------------------

If you have an existing HOOMD-blue MD simulation, you can use p4 to visualize

1. simulation state,
2. pairwise potential curve shape, and
3. rigid body geometry.

Let's say you have a simulation of cubic rigid bodies with interaction sites at
the vertices. Particles at the vertices can interact through a Lennard-Jones
potential.

.. code-block:: python

    import p4
    import hoomd
    import coxeter
    import numpy as np

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
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Tree(2), default_r_cut=0)
    lj.r_cut[("B", "B")] = 5
    lj.params[("A", "A")] = {"epsilon": 0.0, "sigma": 1.0}
    lj.params[("A", "B")] = {"epsilon": 0.0, "sigma": 1.0}
    lj.params[("B", "B")] = {"epsilon": 0.3, "sigma": 1.0}

    # Attach the force and constraint to the integrator
    simulation.operations.integrator.rigid = rigid
    simulation.operations.integrator.forces.append(lj)

Visualize the simulation state.

.. code-block:: python
    
    fig, tr = p4.plot_state(simulation)
    fig.show()

.. raw:: html
    :file: ../data/quickstart-section2-state.html

You can parse your simulation to extract its pairwise forces as one
or more ``Interaction`` objects. This allows you to visualize the potential
energy curves of your pairwise forces.

.. _MD pairwise force: https://hoomd-blue.readthedocs.io/en/v7.0.1/hoomd/md/module-pair.html

.. code-block:: python

    interactions = p4.Interaction.from_hoomd_simulation(simulation)
    fig, _ = interactions[0].plot(r=np.linspace(0, 3, 1000), ylim=[-1, 1], show_legend=True)
    fig.show()

.. raw:: html
    :file: ../data/quickstart-section2-interaction.html

You can parse your simulation to extract its rigid constraint as one or
more ``Body`` objects. This allows you to visualize your rigid body geometries.

.. code-block:: python

    bodies = p4.Body.from_hoomd_simulation(simulation)
    fig, _ = bodies[0].plot(type_shapes=dict(A=cube))
    fig.show()

.. raw:: html
    :file: ../data/quickstart-section2-body.html
