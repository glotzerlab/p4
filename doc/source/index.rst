======================================
p4 - Pairwise Potential Particle Probe
======================================

Probe the effective potential landscape around a rigid body of particles using
HOOMD-blue.

**p4** provides a declarative interface for creating rigid bodies and pairwise
interactions using `HOOMD-blue <https://hoomd-blue.readthedocs.io/en/latest/>`_,
and an object-oriented framework for measuring, aggregating, and plotting
potential energy landscapes using `NumPy <https://numpy.org/>`_ and `Plotly <https://plotly.com/>`_.

More specifically, **p4** provides schema-like wrappers around
`hoomd.md.constrain.Rigid <https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/constrain/rigid.html>`_
and subclasses of `hoomd.md.pair.Pair <https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/pair/pair.html>`_,
making it easy to define *analyte* and *probe* :ref:`bodies <body>`
governed by one or more :ref:`interactions <interaction>`, and
then to probe the corresponding :ref:`system's <system>` potential
energy landscape. The corresponding potential energy distribution can then be
analyzed, sliced, and plotted using **p4**'s :ref:`Field <field>` API.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/overview
   getting-started/installation
   getting-started/basic-usage
   getting-started/project-goals

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/how-p4-works
   user-guide/specifying-particle-models
   user-guide/specifying-interaction-models
   user-guide/measuring-fields
   user-guide/advanced-topics

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/plotting-for-exploration-and-publication
   tutorials/evaluate-fields-of-particle-interaction-models
   tutorials/obtaining-an-effective-shape
   tutorials/collecting-data-for-ml-models

.. toctree::
   :maxdepth: 1
   :caption: API

   api/body
   api/interaction
   api/system
   api/field

.. toctree::
   :maxdepth: 1
   :caption: Reference

   genindex
   development
   changelog
   credits
   license
