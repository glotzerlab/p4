.. _overview:

========
Overview
========

p4 is a package for creating and evaluating particle interaction models,
based on `HOOMD-blue`_ and `Plotly`_. Its API is designed to be *declarative*,
enabling for easy serialization/deserialization to/from JSON, *comprehensive*,
accommodating arbitrary combinations of HOOMD-blue `pair potentials`_, and
*interactive*, providing a rich plotting interface for scalar and vector fields
in 1D, 2D, and 3D.

.. _HOOMD-blue: https://hoomd-blue.readthedocs.io/en/latest/
.. _Plotly: https://plotly.com/
.. _pair potentials: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/pair/pair.html

The name "p4" stands for "Patchy Particle Prototyper in Python." The core goal
of this package is to make it easy to evaluate the effective shape of Molecular
Dynamics particle interaction models. By providing a simple API for measuring
and visualizing energy, force, and torque fields, p4 makes it easy to quickly
evaluate existing models, develop new ones, and share those models with others.