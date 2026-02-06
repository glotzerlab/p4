======================================
p4 - Pairwise Potential Particle Probe
======================================

Probe the effective potential landscape around a rigid body of particles using
`HOOMD-blue <https://hoomd-blue.readthedocs.io/en/latest/>`_.

**p4** provides a declarative interface for creating rigid bodies and
pairwise interactions using HOOMD-blue, and an object-oriented framework
for measuring, aggregating, and plotting potential energy landscapes using
NumPy and Plotly.

TODO

**p4** provides a declarative wrappers around `hoomd.md.constrain.Rigid <https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/constrain/rigid.html>`_
and subclasses of `hoomd.md.pair.Pair <https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/pair/pair.html>`_,
making it easy to define *analyte* and *probe* `bodies <doc/source/api/body.rst>`_
governed by one or more `interactions <doc/source/api/interaction.rst>`_, and
then to probe the corresponding `system's <doc/source/api/system.rst>`_ potential
energy landscape. The corresponding potential energy distribution can then be
analyzed, sliced, and plotted using **p4**'s `Field <doc/source/api/field.rst>`_ API.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   user-guide
   examples

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
