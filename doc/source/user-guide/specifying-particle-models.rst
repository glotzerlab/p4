.. _specifying-particle-models:

==========================
Specifying Particle Models
==========================

A particle model is fully described by a set of particle types and the
relative positions where the particles of each type are placed. Every model must
have at least one particle type, and for all types beyond this one the positions
and orientations relative to the first type must be specified.

In HOOMD-blue, this notion of a particle model is expressed in the MD module's
`Rigid Constraint`_, which calls the primary type the "central" particle and the
secondary types the "constituent" particles.

.. _Rigid Constraint: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/constrain/rigid.html

In p4, the particle model is expressed in the :py:class:`Body` class. Within
this class, the first/required particle type is called the "primary type", and
all subsequent types are called "secondary types".

.. note::
    Since secondary type orientations are only used when secondary types
    interact via anisotropic forces, they are given default values and do not
    need to be provided when creating a ``Body`` for which anisotropic forces
    will not be used.
    
To create a body, simply provide the names of the types and their positions
to the ``Body`` constructor. For example, you can create a single-particle body
as follows.

.. code-block:: python
    
    import p4

    body = p4.Body("A")
    fig, _ = body.plot()
    fig.show()

.. raw:: html
    :file: ../data/single-particle-body.html

| 

A multi-particle body would then be created in the following way.

.. code-block:: python

    import numpy as np

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
    fig.show()

.. raw:: html
    :file: ../data/multi-particle-body.html

| 

When plotting bodies, it is often useful to represent the primary particle type
with a polyhedron. To do so, pass a `Coxeter Polyhedron`_ for a type using the
``type_shapes`` parameter.

.. _Coxeter Polyhedron: https://coxeter.readthedocs.io/en/latest/package-shapes.html#coxeter.shapes.Polyhedron

.. code-block:: python

    import coxeter

    fig, _ = body.plot(
        type_shapes=dict(A=coxeter.shapes.ConvexPolyhedron(octahedron_vertices))
    )
    fig.show()

.. raw:: html
    :file: ../data/multi-particle-body-with-type-shape.html

| 

The ``type_shapes`` parameter can also be used to indicate the orientation of
secondary types.

.. code-block:: python

    import rowan

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
    fig.show()

.. raw:: html
    :file: ../data/multi-particle-body-with-type-shape-and-orientations.html

| 

Orthogonal slices of bodies can also be plotted. This is especially useful when
overlaying a body's extents on a plot of its field (see 
:doc:`measuring-fields`).

.. code-block:: python

    fig, _ = body.plot(
        slice=dict(x=0),    # <--
        type_shapes=dict(A=coxeter.shapes.ConvexPolyhedron(octahedron_vertices))
    )
    fig.show()

.. raw:: html
    :file: ../data/multi-particle-body-with-type-shape-sliced.html

| 
