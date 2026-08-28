====================================
Patchy Particle Prototyper in Python
====================================

p4 is a package for creating and analyzing models of particles and their
interactions, based on `HOOMD-blue`_ and `Plotly`_. Users can measure energy,
force, and torque fields created by rigid bodies of particles interacting via
pairwise potentials, and then interactively plot those bodies and fields
in 3D, 2D, or 1D.

.. _HOOMD-blue: https://hoomd-blue.readthedocs.io/en/latest/
.. _Plotly: https://plotly.com/

p4 has three objectives:

1. to make it easy to see the effective shape of a rigid body's interaction
   fields, rather than inferring it from simulation results;
2. to make it easy to share rigid body models and interaction models without
   requiring specific software; and
3. to make it easier to use `HOOMD-blue`_ for patchy particle systems research.

p4 accomplishes these objectives through an API that is designed to be

* *declarative*, enabling easy serialization/deserialization to/from JSON;
* *comprehensive*, accommodating arbitrary combinations of HOOMD-blue pair
  potentials; and
* *interactive*, providing a rich plotting interface for rigid body models and
  fields.


How to Use this Documentation
+++++++++++++++++++++++++++++

Select the section that best applies to you.

.. dropdown:: I am learning how to use HOOMD-blue.

    p4 will help you learn HOOMD-blue. p4's API is much simpler and more
    beginner-friendly, and most of p4's classes can be directly converted to
    HOOMD-blue equivalents.
   
    .. attention::
      
        In p4, interactions are currently confined to HOOMD-blue's Molecular
        Dynamics (MD) module, so if you are only interested in Monte Carlo
        methods then p4 will not be as helpful to you.

    Start by :ref:`installing p4 <installation>`, and then read the following
    sections of the User Guide. They will help you understand the purpose and
    function of p4's core classes.

    #. :ref:`specifying-particle-models` covers the :py:class:`~p4.body.Body`
       class, which corresponds to HOOMD-blue's `rigid constraint`_.
    #. :ref:`specifying-interaction-models` covers the
       :py:class:`~p4.interaction.Interaction` class, which corresponds to
       HOOMD-blue's `MD pair potential`_.
    #. :ref:`measuring-fields-body` covers the :py:class:`~p4.system.System` and
       :py:class:`~p4.field.Field` classes, which have no equivalents in
       HOOMD-blue.
    #. :ref:`measuring-fields-arrangement` covers the
       :py:class:`~p4.arrangement.Arrangement` class, which corresponds to
       HOOMD-blue's `Snapshot`_.

    .. _rigid constraint: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/constrain/rigid.html
    .. _MD pair potential: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/module-pair.html
    .. _Snapshot: https://hoomd-blue.readthedocs.io/en/latest/hoomd/snapshot.html

    To learn more about the relationship between p4 and HOOMD-blue, read the
    section :ref:`creating-hoomd-blue-objects-from-p4`.


.. dropdown:: I want to use p4 in a current HOOMD-blue project.

    Read :ref:`quickstart`. It will guide you through p4's core workflow
    beginning from an existing HOOMD-blue simulation in only a few minutes.

    Next, read :ref:`accelerating-workflows`, which will show you ways to
    integrate p4 into existing HOOMD-blue workflows. If you use `Signac`_, that
    page also covers p4-Signac integration.

    .. _Signac: https://signac.readthedocs.io/en/latest/
   
    Afterwards, if you want to learn more about p4's design and the functions of
    its various classes, read through the User Guide, starting with
    :ref:`specifying-particle-models`.


.. dropdown:: Um... what?

    Welcome! If you'd like to learn more about p4's goals and motivation, read
    :ref:`project-goals`.

    If you are not a programmer, still take a look at the
    :ref:`User Guide <specifying-particle-models>`---all of the plots
    are interactive!

    If you are a programmer and you are curious about p4's codebase, read
    :ref:`for-developers`.


.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :hidden:

    getting-started/installation
    getting-started/quickstart
    getting-started/project-goals
    getting-started/citing-p4

.. toctree::
    :maxdepth: 2
    :caption: User Guide
    :hidden:

    user-guide/specifying-particle-models
    user-guide/specifying-interaction-models
    user-guide/measuring-fields-body
    user-guide/measuring-fields-arrangement
    user-guide/advanced-topics

.. toctree::
    :maxdepth: 1
    :caption: API
    :hidden:

    api/module-p4
    api/types

.. toctree::
    :maxdepth: 1
    :caption: Reference
    :hidden:

    schemas
    for-developers
    changelog
    credits
    license
    genindex
