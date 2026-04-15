.. _specifying-interaction-models:

=============================
Specifying Interaction Models
=============================

An interaction model mathematically defines the forces and torques that
particles exhert on each other.

In HOOMD-blue, this notion of an interaction model is expressed in the MD
module's `Pair Force`_, which stores a set of parameters to apply to the
underlying mathematical function for each possible pair of particle types.

.. _Pair Force: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/module-pair.html

In p4, the interaction model is expressed in the :py:class:`Interaction` class.
Unlike HOOMD-blue's ``Pair``, this class stores all of the function parameters
in two attributes: a "default" attribute whose parameters are applied first to
every possible pair of types, and a "typed" attribute whose parameters are
then applied to specifically named pairs of types.

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

In the above example, the interaction is a `Lennard-Jones force`_ that is active
only for A-C pairs.

.. _Lennard-Jones force: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/pair/lj.html

.. TODO: add plotting for interaction