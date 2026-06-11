.. _specifying-interaction-models:

=============================
Specifying Interaction Models
=============================

An interaction model mathematically defines the forces and torques that
particles exert on each other.

In HOOMD-blue, this notion of an interaction model is mostly expressed in the MD
module's `Pair Force`_, which stores a set of parameters to apply to the
underlying mathematical function for each possible pair of particle types.

.. _Pair Force: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/module-pair.html

In p4, the interaction model is expressed in the :py:class:`~p4.Interaction`
class. Unlike HOOMD-blue's ``Pair``, this class stores all of the function
parameters in two attributes: a "default" attribute whose parameters are applied
first to every possible pair of types, and a "typed" attribute whose parameters
are then applied to specifically named pairs of types.

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
only for A-C pairs. Just like a ``Body``, and ``Interaction`` can be plotted.

.. code-block:: python

    import numpy as np

    fig, tr = interaction.plot(r=np.linspace(0, 8, 100), include_default=True)
    fig.show()

.. raw:: html
    :file: ../data/basic-interaction.html

.. _Lennard-Jones force: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/pair/lj.html

An ``Interaction`` can be created for any instantiable subclass of HOOMD-blue's
``Pair``, including classes in the `anisotropic`_ and `friction`_ submodules.
Once created, an ``Interaction`` can be added to a system, or converted into
its parameterized HOOMD-blue class using
:py:meth:`~p4.Interaction.to_hood_pair`. Because ``Interaction`` is
self-validating, creating a HOOMD-blue ``Pair`` in this way prevents mistakes
that are common when using HOOMD-blue, such as forgetting to specify a parameter
or forgetting to specify parameters for a pair of particle types.

.. _anisotropic: https://hoomd-blue.readthedocs.io/en/v7.0.1/hoomd/md/pair/module-aniso.html
.. _friction: https://hoomd-blue.readthedocs.io/en/v7.0.1/hoomd/md/pair/module-friction.html

.. TODO: add plotting for interaction