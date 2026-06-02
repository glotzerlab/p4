=========================================
p4 - Patchy Particle Prototyper in Python
=========================================

Rapidly prototype new particle and interaction models and evaluate existing
models using `HOOMD-blue`_.

p4 provides a simple declarative interface for specifying particle models and
pairwise interaction models, and for measuring the energy and force fields that
those models create. Particle models and their fields can then be easily
plotted with methods built on the interactive plotting library `Plotly`_.

.. _HOOMD-blue: https://hoomd-blue.readthedocs.io/en/latest/
.. _Plotly: https://plotly.com/


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting-started/overview
   getting-started/installation
   getting-started/basic-usage
   getting-started/project-goals

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user-guide/specifying-particle-models
   user-guide/specifying-interaction-models
   user-guide/measuring-fields

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   api/module-p4
   api/util
   api/types

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   schema
   for-developers
   changelog
   credits
   license
   genindex
