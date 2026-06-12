p4
==

This is the top of the namespace for the p4 module. This module contains the
:ref:`core classes <core-classes>`, as well as convenience functions for
:ref:`sampling <top-level-sampling>` and :ref:`plotting <top-level-plotting>`.


.. _core-classes:

Core Classes
++++++++++++

.. toctree::
    :maxdepth: 1

    body
    arrangement
    interaction
    system
    field


.. _top-level-sampling:

Sampling Functions
++++++++++++++++++

Functions for sampling position and orientation space.

.. py:currentmodule:: p4

.. autofunction:: p4.positions_on_regular_grid

.. autofunction:: p4.orientations_about_axis

.. autofunction:: p4.orientations_from_fibonacci_lattice


.. _top-level-plotting:

Plotting Functions
++++++++++++++++++

Constants and functions for visualizing samples and HOOMD-blue objects using
`Plotly`_.

.. _Plotly: https://plotly.com/

.. py:currentmodule:: p4

.. autofunction:: p4.plot_positions

.. autofunction:: p4.plot_state

.. autofunction:: p4.plot_layout

.. autofunction:: p4.snapshot_schematic_slice_trace
